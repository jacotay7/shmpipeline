"""Run the synthetic three-loop tomographic AO stress workload."""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from shmpipeline import PipelineConfig, PipelineManager
from shmpipeline.logging_utils import configure_colored_logging
from shmpipeline.scheduling import NoAffinityPlacementPolicy

WFS_COUNT = 8
WFS_SHAPE = (256, 256)
RECONSTRUCTOR_SHAPE = (4096, 65536)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument(
        "--profile", choices=("baseline", "sustained"), default="sustained"
    )
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--warmup", type=float, default=1.0)
    parser.add_argument(
        "--main-rate",
        type=float,
        default=None,
        help="Camera-set rate in Hz; omit for unthrottled publication.",
    )
    parser.add_argument("--loop-a-rate", type=float, default=700.0)
    parser.add_argument("--loop-b-rate", type=float, default=233.0)
    parser.add_argument("--camera-jitter-us", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2401)
    parser.add_argument("--blas-threads", type=int, default=16)
    parser.add_argument(
        "--gpu-unbatched",
        action="store_true",
        help="GPU sustained profile only: drive the fused controller from "
        "eight separate camera streams through the all_new barrier instead "
        "of one pre-stacked (8, 256, 256) cube.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    return parser


def _fill_stream(stream: Any, value: float) -> None:
    """Publish one constant without allocating a same-sized temporary."""
    with stream.write_view() as view:
        fill = getattr(view, "fill_", None)
        if callable(fill):
            fill(value)
        else:
            view.fill(value)


def _make_payload(stream: Any, value: float):
    if stream.gpu_enabled:
        import torch

        return torch.full(
            stream.shape,
            value,
            dtype=torch.float32,
            device=stream.gpu_device,
        )
    return np.full(stream.shape, value, dtype=stream.dtype)


def _load_calibrations(manager: PipelineManager) -> None:
    logger = logging.getLogger("shmpipeline.example.tomographic_ao")
    logger.info("initializing static calibration streams")
    names = manager.config.shared_memory_by_name
    if "tomo_wfs_dark" in names:
        _fill_stream(manager.get_stream("tomo_wfs_dark"), 0.0)
        _fill_stream(manager.get_stream("tomo_wfs_inverse_flat"), 1.0)
        _fill_stream(manager.get_stream("tomo_wfs_slope_offset"), 0.0)
    else:
        for index in range(WFS_COUNT):
            prefix = f"tomo_wfs{index}"
            _fill_stream(manager.get_stream(f"{prefix}_dark"), 0.0)
            _fill_stream(manager.get_stream(f"{prefix}_inverse_flat"), 1.0)
            _fill_stream(manager.get_stream(f"{prefix}_slope_offset"), 0.0)

    # Filling the shared stream directly avoids a second 1 GiB host/device
    # allocation. Zeros are sufficient because this benchmark measures data
    # movement and matrix multiplication, not reconstruction fidelity.
    _fill_stream(manager.get_stream("tomo_reconstructor"), 0.0)
    _fill_stream(manager.get_stream("tomo_reconstructor_bias"), 0.0)
    _fill_stream(manager.get_stream("tomo_command_offset"), 0.0)
    _fill_stream(manager.get_stream("tomo_command_low"), -2.5)
    _fill_stream(manager.get_stream("tomo_command_high"), 2.5)

    angle_a = np.deg2rad(12.0)
    angle_b = np.deg2rad(-7.0)
    for loop, angle in (("a", angle_a), ("b", angle_b)):
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ],
            dtype=np.float32,
        )
        manager.get_stream(f"tt_{loop}_rotation").write(rotation)
        _fill_stream(manager.get_stream(f"tt_{loop}_bias"), 0.0)


def _paced_writer(
    streams: list[Any],
    payloads: list[Any],
    stop: threading.Event,
    *,
    rate_hz: float | None,
    jitter_s: float = 0.0,
    rng: np.random.Generator | None = None,
    stats: dict[str, int] | None = None,
    stats_key: str = "frames",
) -> None:
    interval = None if rate_hz is None else 1.0 / rate_hz
    deadline = time.perf_counter()
    while not stop.is_set():
        for stream, payload in zip(streams, payloads):
            if jitter_s > 0.0 and rng is not None:
                delay = float(rng.uniform(0.0, jitter_s))
                if stop.wait(delay):
                    return
            stream.write(payload)
        if stats is not None:
            stats[stats_key] = stats.get(stats_key, 0) + 1
        if interval is not None:
            deadline += interval
            remaining = deadline - time.perf_counter()
            if remaining > 0.0:
                stop.wait(remaining)
            else:
                deadline = time.perf_counter()


def _start_sources(manager: PipelineManager, args):
    stop = threading.Event()
    stats: dict[str, int] = {}
    rng = np.random.default_rng(args.seed)
    names = manager.config.shared_memory_by_name
    if "tomo_wfs_cube_raw" in names:
        camera_streams = [manager.get_stream("tomo_wfs_cube_raw")]
    else:
        camera_streams = [
            manager.get_stream(f"tomo_wfs{index}_raw")
            for index in range(WFS_COUNT)
        ]
    camera_payloads = [
        _make_payload(stream, 1.0 + 0.01 * index)
        for index, stream in enumerate(camera_streams)
    ]
    threads = [
        threading.Thread(
            target=_paced_writer,
            args=(camera_streams, camera_payloads, stop),
            kwargs={
                "rate_hz": args.main_rate,
                "jitter_s": args.camera_jitter_us * 1e-6,
                "rng": rng,
                "stats": stats,
                "stats_key": "main",
            },
            name="tomographic-camera-set",
            daemon=True,
        )
    ]
    for loop, rate, value in (
        ("a", args.loop_a_rate, 1.0),
        ("b", args.loop_b_rate, 2.0),
    ):
        stream = manager.get_stream(f"tt_{loop}_image")
        threads.append(
            threading.Thread(
                target=_paced_writer,
                args=([stream], [_make_payload(stream, value)], stop),
                kwargs={
                    "rate_hz": rate,
                    "stats": stats,
                    "stats_key": f"loop_{loop}",
                },
                name=f"tip-tilt-{loop}-source",
                daemon=True,
            )
        )
    for thread in threads:
        thread.start()
    return stop, threads, stats


def _counts(manager: PipelineManager) -> dict[str, int]:
    return {
        "main": int(manager.get_stream("tomo_dm_command").count),
        "loop_a": int(manager.get_stream("tt_a_command").count),
        "loop_b": int(manager.get_stream("tt_b_command").count),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.duration <= 0.0 or args.warmup < 0.0:
        raise ValueError("duration must be positive and warmup non-negative")
    config_name = f"pipeline_{args.backend}"
    if args.backend == "gpu" and args.profile == "sustained":
        config_name += (
            "_sustained" if args.gpu_unbatched else "_sustained_batched"
        )
    config_path = Path(__file__).with_name(f"{config_name}.yaml")
    config = PipelineConfig.from_yaml(config_path)
    if args.backend == "cpu" and args.profile == "sustained":
        kernels = []
        for kernel in config.kernels:
            if kernel.name == "tomo_reconstruction":
                parameters = dict(kernel.parameters)
                parameters["blas_threads"] = args.blas_threads
                kernel = replace(kernel, parameters=parameters)
            kernels.append(kernel)
        config = replace(config, kernels=tuple(kernels))
    matrix_bytes = int(np.prod(RECONSTRUCTOR_SHAPE)) * 4
    logger = logging.getLogger("shmpipeline.example.tomographic_ao")
    logger.info(
        "building %s workload: WFS=%d image=%s slopes=65536 "
        "reconstructor=%s (%.2f GiB)",
        args.backend,
        WFS_COUNT,
        WFS_SHAPE,
        RECONSTRUCTOR_SHAPE,
        matrix_bytes / 1024**3,
    )
    placement = (
        NoAffinityPlacementPolicy()
        if args.backend == "cpu" and args.profile == "sustained"
        else None
    )
    manager = PipelineManager(config, placement_policy=placement)
    source_stop = None
    threads: list[threading.Thread] = []
    source_stats: dict[str, int] = {}
    try:
        manager.build()
        _load_calibrations(manager)
        manager.start()
        source_stop, threads, source_stats = _start_sources(manager, args)
        time.sleep(args.warmup)
        before = _counts(manager)
        source_before = dict(source_stats)
        started = time.monotonic()
        deadline = started + args.duration
        while time.monotonic() < deadline:
            manager.poll_events()
            manager.raise_if_failed()
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        elapsed = time.monotonic() - started
        after = _counts(manager)
        manager.poll_events()
        manager.raise_if_failed()
        frames = {name: after[name] - before[name] for name in before}
        source_frames = {
            name: source_stats.get(name, 0) - source_before.get(name, 0)
            for name in frames
        }
        return {
            "backend": args.backend,
            "profile": args.profile,
            "config": str(config_path),
            "duration_s": elapsed,
            "warmup_s": args.warmup,
            "reconstructor_shape": list(RECONSTRUCTOR_SHAPE),
            "reconstructor_bytes": matrix_bytes,
            "frames": frames,
            "source_frames": source_frames,
            "delivery_ratio": {
                name: (
                    frames[name] / source_frames[name]
                    if source_frames[name]
                    else 0.0
                )
                for name in frames
            },
            "throughput_hz": {
                name: count / elapsed for name, count in frames.items()
            },
            "worker_metrics": manager.status().get("metrics", {}),
        }
    finally:
        if source_stop is not None:
            source_stop.set()
        for thread in threads:
            thread.join(timeout=2.0)
        manager.shutdown(force=True)


def main() -> int:
    args = _parser().parse_args()
    configure_colored_logging(level=logging.INFO)
    report = run(args)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
