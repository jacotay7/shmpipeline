"""Showcase: a live AO control loop as a shmpipeline dataflow graph.

Renders an animated clip of the observatory AO pipeline in `pipeline.yaml`
actually running, so the picture shows what shmpipeline *is*: one YAML file
turned into a DAG of compute kernels, each in its own pinned OS process, joined
by zero-copy `pyshmem` shared-memory streams.

The clip carries three live panels along the loop --

  * the 256x256 Shack-Hartmann WFS image written into the input stream,
  * the 32x32 centroid field the `cpu.shack_hartmann_centroid` kernel measures,
  * the 1024 deformable-mirror commands that come out the far end,

-- over a dataflow diagram of the five kernel stages, each labelled with the
per-stage execution time the running workers actually reported.

Every number on the clip is measured on the machine that renders it:
per-stage times come from the manager's worker metrics, and the loop rate and
frame spacing come from `PipelineManager.benchmark()`. Note that the framework
reports terminal *inter-arrival spacing*, not true end-to-end latency, so the
clip says "frame spacing" rather than "latency".

Run:  ``python examples/observatory_ao_system/showcase.py``
      ``python examples/observatory_ao_system/showcase.py --frames 40``
      ``python examples/observatory_ao_system/showcase.py --out clip.webp``

Needs matplotlib + pillow on top of the base install.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

from shmpipeline import PipelineManager
from shmpipeline.synthetic import SyntheticInputConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_example import (  # noqa: E402  (local example helper module)
    ACTUATOR_COUNT,
    SPOT_SIGMA_PX,
    SUBAPERTURE_GRID,
    TILE_SIZE,
    make_command_limits,
    make_flux_map,
    make_reference_centroids,
    render_shack_hartmann_image,
    synthesize_residual_centroids,
    wait_for_next_write,
)

CONFIG_PATH = Path(__file__).with_name("pipeline.yaml")
SEED = 20260403

# The five kernel stages of pipeline.yaml, in dataflow order, with the shape of
# the stream each one publishes.
STAGES = (
    ("centroid", "cpu.shack_hartmann_centroid", "32×32×2"),
    ("flatten", "cpu.flatten", "2048"),
    ("reconstruct", "cpu.affine_transform", "1024"),
    ("integrate", "cpu.leaky_integrator", "1024"),
    ("clip", "cpu.custom_operation", "1024"),
)
STAGE_KEYS = (
    "centroid_stage",
    "flatten_stage",
    "reconstructor_stage",
    "controller_stage",
    "saturation_stage",
)

N_FRAMES = 120
PLAYBACK_FPS = 25
BENCH_SECONDS = 3.0


def _cpu_model() -> str:
    try:
        for line in (
            Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
        ):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip().replace("AMD ", "")
    except OSError:
        pass
    import platform

    return platform.processor() or "unknown CPU"


def modal_control_matrix(rng: np.random.Generator) -> np.ndarray:
    """A smooth modal reconstructor mapping 2048 slopes to 1024 actuators.

    `run_example.py` uses an i.i.d. random control matrix, which is fine for
    verifying arithmetic but turns the deformable-mirror command into white
    noise on screen. Factoring the matrix through a handful of smooth spatial
    modes on the actuator grid keeps the same shapes and the same kernel work
    while producing the low-order mirror shapes a real reconstructor makes.
    """
    rows, cols = SUBAPERTURE_GRID
    y, x = np.mgrid[0:rows, 0:cols] / max(rows - 1, 1)
    modes = [np.ones_like(y)]
    for order in range(1, 5):
        modes += [
            np.cos(np.pi * order * x),
            np.cos(np.pi * order * y),
            np.cos(np.pi * order * (x + y)),
            np.sin(np.pi * order * (x - y)),
        ]
    basis = np.stack([mode.ravel() for mode in modes], axis=1).astype(
        np.float32
    )
    mixing = rng.normal(
        0.0, 1.0, size=(basis.shape[1], int(np.prod(SUBAPERTURE_GRID)) * 2)
    )
    matrix = basis @ mixing.astype(np.float32)
    # Scale so a typical slope vector lands inside the actuator stroke limits.
    return (matrix * (0.02 / np.std(matrix))).astype(np.float32)


def collect_frames(
    manager: PipelineManager, n_frames: int
) -> dict[str, np.ndarray]:
    """Drive the live pipeline frame by frame and record what each stage produced.

    This loop is deliberately unhurried -- it renders a Shack-Hartmann image in
    Python for every frame -- because it exists to capture pictures, not to
    measure speed. The throughput numbers come from :func:`measure`.
    """
    rng = np.random.default_rng(SEED)
    subap_y = np.linspace(-1.0, 1.0, SUBAPERTURE_GRID[0], dtype=np.float32)
    subap_x = np.linspace(-1.0, 1.0, SUBAPERTURE_GRID[1], dtype=np.float32)
    grid_y, grid_x = np.meshgrid(subap_y, subap_x, indexing="ij")

    reference = make_reference_centroids(grid_y, grid_x, rng)
    control_matrix = modal_control_matrix(rng)
    modal_bias = rng.normal(0.0, 0.02, size=(ACTUATOR_COUNT,)).astype(
        np.float32
    )
    effective_bias = (
        modal_bias - control_matrix @ reference.reshape(-1)
    ).astype(np.float32)
    low_limit, high_limit = make_command_limits(rng, ACTUATOR_COUNT)
    base_flux = rng.uniform(70.0, 140.0, size=SUBAPERTURE_GRID).astype(
        np.float32
    )

    manager.get_stream("obs_control_matrix").write(control_matrix)
    manager.get_stream("obs_modal_bias").write(effective_bias)
    manager.get_stream("obs_command_low_limit").write(low_limit)
    manager.get_stream("obs_command_high_limit").write(high_limit)

    image_stream = manager.get_stream("obs_wfs_image")
    centroid_stream = manager.get_stream("obs_measured_centroids")
    command_stream = manager.get_stream("obs_dm_command")

    images = np.empty((n_frames, 256, 256), dtype=np.float32)
    centroids = np.empty((n_frames, *SUBAPERTURE_GRID, 2), dtype=np.float32)
    commands = np.empty((n_frames, ACTUATOR_COUNT), dtype=np.float32)

    for index in range(n_frames):
        measured = np.clip(
            reference
            + synthesize_residual_centroids(index, grid_y, grid_x, rng),
            -1.45,
            1.45,
        ).astype(np.float32)
        image = render_shack_hartmann_image(
            measured,
            make_flux_map(base_flux, index, grid_y, grid_x, rng),
            tile_size=TILE_SIZE,
            spot_sigma_px=SPOT_SIGMA_PX,
        )
        baseline = command_stream.count
        image_stream.write(image)
        command = wait_for_next_write(command_stream, baseline, timeout=5.0)
        images[index] = image
        centroids[index] = centroid_stream.read()
        commands[index] = command

    return {"images": images, "centroids": centroids, "commands": commands}


def measure(manager: PipelineManager, seconds: float) -> dict:
    """Drive the pipeline flat out and return its own throughput report."""
    return manager.benchmark(
        duration_s=seconds,
        source=SyntheticInputConfig(
            stream_name="obs_wfs_image", pattern="random"
        ),
    )


def render_animation(frames, report, out_path, playback_fps):
    """Draw the dataflow graph with the live panels above it, as animated WebP."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from PIL import Image

    ink, sub, bg = "#e6edf3", "#9aa7b4", "#0b0f14"
    accent, highlight, wire = "#ffd166", "#7fd1c1", "#3a4a5c"

    images = frames["images"]
    centroids = frames["centroids"]
    commands = frames["commands"]
    n_frames = len(images)

    workers = report.get("workers", {})
    stage_ms = [
        float(workers.get(key, {}).get("avg_exec_ms", 0.0))
        for key in STAGE_KEYS
    ]
    throughput = float(report.get("throughput_hz", 0.0))
    spacing = report.get("inter_arrival_ms", {})

    fig = plt.figure(figsize=(8.6, 5.9), dpi=100)
    fig.patch.set_facecolor(bg)

    # --- header --------------------------------------------------------------
    fig.text(
        0.5,
        0.975,
        "shmpipeline — one YAML file, five pinned processes, a live AO loop",
        color=ink,
        fontsize=15.5,
        fontweight="bold",
        ha="center",
        va="top",
    )
    fig.text(
        0.5,
        0.935,
        "each kernel runs in its own OS process; the arrows are zero-copy pyshmem "
        "shared-memory streams",
        color=sub,
        fontsize=9.5,
        ha="center",
        va="top",
    )

    # --- live panels ---------------------------------------------------------
    panel_titles = (
        "WFS image  (256²)",
        "measured centroids  (32×32×2)",
        "DM command  (1024)",
    )
    axes = [
        fig.add_axes([0.045 + 0.325 * i, 0.455, 0.255, 0.36]) for i in range(3)
    ]
    for ax, title in zip(axes, panel_titles):
        ax.set_facecolor(bg)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#233040")
        ax.set_title(title, color=ink, fontsize=10, fontweight="bold", pad=7)

    wfs_art = axes[0].imshow(
        images[0],
        cmap="inferno",
        vmin=0.0,
        vmax=float(np.percentile(images, 99.8)),
        origin="lower",
        interpolation="nearest",
    )

    # Quiver every other subaperture: 32x32 arrows is an unreadable thicket at
    # this panel size, 16x16 still reads as a slope field.
    step = 2
    qy, qx = np.mgrid[
        0 : SUBAPERTURE_GRID[0] : step, 0 : SUBAPERTURE_GRID[1] : step
    ]
    quiver_art = axes[1].quiver(
        qx,
        qy,
        centroids[0, ::step, ::step, 1],
        centroids[0, ::step, ::step, 0],
        color=highlight,
        scale=18.0,
        width=0.007,
    )
    axes[1].set_xlim(-1, SUBAPERTURE_GRID[1])
    axes[1].set_ylim(-1, SUBAPERTURE_GRID[0])
    axes[1].set_aspect("equal")

    side = int(np.sqrt(ACTUATOR_COUNT))
    command_scale = float(np.percentile(np.abs(commands), 99.5)) or 1.0
    dm_art = axes[2].imshow(
        commands[0].reshape(side, side),
        cmap="RdBu_r",
        vmin=-command_scale,
        vmax=command_scale,
        origin="lower",
        interpolation="nearest",
    )

    # --- dataflow graph ------------------------------------------------------
    flow = fig.add_axes([0.03, 0.10, 0.94, 0.29])
    flow.set_facecolor(bg)
    flow.set_xlim(0, 1)
    flow.set_ylim(0, 1)
    flow.axis("off")

    n_stages = len(STAGES)
    box_w, box_h, box_y = 0.148, 0.40, 0.36
    gap = (1.0 - n_stages * box_w) / (n_stages + 1)
    centers = [gap + box_w / 2 + i * (box_w + gap) for i in range(n_stages)]

    for center, (label, kind, out_shape), exec_ms in zip(
        centers, STAGES, stage_ms
    ):
        flow.add_patch(
            FancyBboxPatch(
                (center - box_w / 2, box_y),
                box_w,
                box_h,
                boxstyle="round,pad=0.012",
                linewidth=1.4,
                edgecolor="#2c3846",
                facecolor="#141b24",
            )
        )
        flow.text(
            center,
            box_y + box_h * 0.70,
            label,
            color=ink,
            fontsize=10.5,
            fontweight="bold",
            ha="center",
            va="center",
        )
        flow.text(
            center,
            box_y + box_h * 0.44,
            kind,
            color=sub,
            fontsize=6.6,
            ha="center",
            va="center",
        )
        flow.text(
            center,
            box_y + box_h * 0.17,
            f"{exec_ms * 1e3:,.0f} µs",
            color=accent,
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
        )
        flow.text(
            center,
            box_y - 0.10,
            f"→ {out_shape}",
            color=wire,
            fontsize=7.5,
            ha="center",
            va="center",
        )

    # arrows: into the first stage, between stages, and out of the last
    edges = [(gap * 0.18, centers[0] - box_w / 2)]
    edges += [
        (centers[i] + box_w / 2, centers[i + 1] - box_w / 2)
        for i in range(n_stages - 1)
    ]
    edges.append((centers[-1] + box_w / 2, 1.0 - gap * 0.18))
    for x0, x1 in edges:
        flow.annotate(
            "",
            xy=(x1, box_y + box_h / 2),
            xytext=(x0, box_y + box_h / 2),
            arrowprops={"arrowstyle": "-|>", "color": wire, "lw": 1.3},
        )
    flow.text(
        gap * 0.18,
        box_y + box_h / 2 + 0.17,
        "WFS image\n256²",
        color=wire,
        fontsize=7.5,
        ha="left",
        va="bottom",
        linespacing=1.4,
    )
    flow.text(
        1.0 - gap * 0.18,
        box_y + box_h / 2 + 0.17,
        "DM command\n1024",
        color=wire,
        fontsize=7.5,
        ha="right",
        va="bottom",
        linespacing=1.4,
    )

    # A marker tracing one frame's path along the chain. It hops along the
    # arrow segments only -- the stages run concurrently on different frames,
    # and a dot sliding across the boxes would both obscure them and imply a
    # single stage is "active".
    (pulse,) = flow.plot([], [], "o", color=accent, markersize=7)
    pulse_x = np.concatenate(
        [np.linspace(x0, x1, 4, endpoint=False) for x0, x1 in edges]
    )

    # --- footer --------------------------------------------------------------
    p50 = float(spacing.get("p50", 0.0))
    fig.text(
        0.5,
        0.068,
        f"sustained {throughput:,.0f} Hz end to end   ·   frame spacing p50 {p50:.2f} ms   ·   "
        f"p99 {float(spacing.get('p99', 0.0)):.2f} ms",
        color=accent,
        fontsize=11,
        fontweight="bold",
        ha="center",
        va="center",
    )
    fig.text(
        0.5,
        0.028,
        f"measured by PipelineManager.benchmark() on {_cpu_model()} · "
        "spacing is terminal inter-arrival, not end-to-end latency",
        color=sub,
        fontsize=8,
        ha="center",
        va="center",
    )
    counter = fig.text(
        0.978,
        0.418,
        "",
        color=sub,
        fontsize=8.5,
        ha="right",
        va="center",
        family="monospace",
    )

    def rgba_frame() -> Image.Image:
        fig.canvas.draw()
        return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert(
            "RGB"
        )

    pil_frames = []
    for index in range(n_frames):
        wfs_art.set_data(images[index])
        quiver_art.set_UVC(
            centroids[index, ::step, ::step, 1],
            centroids[index, ::step, ::step, 0],
        )
        dm_art.set_data(commands[index].reshape(side, side))
        pulse.set_data([pulse_x[index % len(pulse_x)]], [box_y + box_h / 2])
        counter.set_text(f"frame {index + 1:3d}/{n_frames}")
        pil_frames.append(rgba_frame())
    plt.close(fig)

    pil_frames[0].save(
        out_path,
        format="WEBP",
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / playback_fps),
        loop=0,
        quality=90,
        method=6,
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--frames",
        type=int,
        default=N_FRAMES,
        help=f"animation frames (default {N_FRAMES})",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=BENCH_SECONDS,
        help=f"throughput measurement window (default {BENCH_SECONDS})",
    )
    parser.add_argument("--out", default="pipeline_showcase.webp")
    args = parser.parse_args()
    if args.frames < 2:
        raise SystemExit("--frames must be >= 2")

    logging.basicConfig(level=logging.WARNING)
    spawn_method = "fork" if sys.platform.startswith("linux") else "spawn"
    manager = PipelineManager(CONFIG_PATH, spawn_method=spawn_method)
    manager.build()
    manager.start()
    try:
        print(f"collecting {args.frames} frames through the live pipeline ...")
        frames = collect_frames(manager, args.frames)

        print(f"measuring throughput for {args.seconds:g} s ...")
        report = measure(manager, args.seconds)
        print(
            f"  {report['throughput_hz']:,.0f} Hz  p50 {report['inter_arrival_ms']['p50']:.2f} ms"
        )
        for (label, _, _), key in zip(STAGES, STAGE_KEYS):
            exec_ms = report["workers"].get(key, {}).get("avg_exec_ms", 0.0)
            print(f"  {label:12s} {exec_ms * 1e3:7,.0f} µs")
    finally:
        manager.shutdown(force=True)

    time.sleep(0.2)
    out = render_animation(frames, report, args.out, PLAYBACK_FPS)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
