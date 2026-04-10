"""Benchmark observatory reconstructor timing under sustained WFS load."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from shmpipeline import PipelineManager
from shmpipeline.logging_utils import configure_colored_logging

DEFAULT_RATE_HZ = 500.0
DEFAULT_WARMUP_S = 1.0
DEFAULT_DURATION_S = 4.0
DEFAULT_FRAME_BANK_SIZE = 64
DEFAULT_SEED = 20260403


def _load_observatory_example_module():
    module_path = Path(__file__).with_name("run_example.py")
    spec = importlib.util.spec_from_file_location(
        "observatory_ao_example",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load observatory example from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _wait_for_next_write(stream, previous_count: int, *, timeout: float) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if stream.count > previous_count:
            return stream.read()
        time.sleep(1e-4)
    raise TimeoutError(f"timed out waiting for a new write on {stream.name!r}")


def _wait_until(deadline: float) -> None:
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0.0:
            return
        if remaining > 5e-4:
            time.sleep(remaining - 2e-4)


def _runtime_snapshot(manager: PipelineManager) -> dict[str, Any]:
    snapshot = manager.runtime_snapshot()
    if snapshot["state"] == "failed":
        manager.raise_if_failed()
    return snapshot


def _prepare_benchmark_inputs(
    frame_bank_size: int,
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    example = _load_observatory_example_module()
    rng = np.random.default_rng(seed)
    subap_y = np.linspace(
        -1.0,
        1.0,
        example.SUBAPERTURE_GRID[0],
        dtype=np.float32,
    )
    subap_x = np.linspace(
        -1.0,
        1.0,
        example.SUBAPERTURE_GRID[1],
        dtype=np.float32,
    )
    grid_y, grid_x = np.meshgrid(subap_y, subap_x, indexing="ij")

    reference_centroids = example.make_reference_centroids(grid_y, grid_x, rng)
    control_matrix = rng.normal(
        0.0,
        0.015,
        size=(
            example.ACTUATOR_COUNT,
            np.prod(example.SUBAPERTURE_GRID) * 2,
        ),
    ).astype(np.float32)
    modal_bias = rng.normal(
        0.0,
        0.02,
        size=(example.ACTUATOR_COUNT,),
    ).astype(np.float32)
    effective_bias = (
        modal_bias - control_matrix @ reference_centroids.reshape(-1)
    ).astype(np.float32)
    low_limit, high_limit = example.make_command_limits(
        rng,
        example.ACTUATOR_COUNT,
    )
    base_flux_map = rng.uniform(
        70.0,
        140.0,
        size=example.SUBAPERTURE_GRID,
    ).astype(np.float32)

    frames = np.empty(
        (frame_bank_size, *example.IMAGE_SHAPE),
        dtype=np.float32,
    )
    for index in range(frame_bank_size):
        residual_centroids = example.synthesize_residual_centroids(
            index,
            grid_y,
            grid_x,
            rng,
        )
        measured_centroids = np.clip(
            reference_centroids + residual_centroids,
            -1.45,
            1.45,
        ).astype(np.float32)
        flux_map = example.make_flux_map(
            base_flux_map,
            index,
            grid_y,
            grid_x,
            rng,
        )
        frames[index] = example.render_shack_hartmann_image(
            measured_centroids,
            flux_map,
            tile_size=example.TILE_SIZE,
            spot_sigma_px=example.SPOT_SIGMA_PX,
        )

    return {
        "control_matrix": control_matrix,
        "effective_bias": effective_bias,
        "low_limit": low_limit,
        "high_limit": high_limit,
        "frames": frames,
    }


def _drive_wfs_stream(
    manager: PipelineManager,
    image_stream: Any,
    frames: np.ndarray,
    *,
    frame_count: int,
    rate_hz: float,
) -> dict[str, Any]:
    poll_interval_frames = max(1, min(frame_count, int(round(rate_hz / 20.0))))
    interval_s = 1.0 / rate_hz
    late_frames = 0
    max_lateness_s = 0.0
    next_deadline = time.perf_counter()
    started_at = next_deadline
    last_snapshot = _runtime_snapshot(manager)

    for frame_index in range(frame_count):
        _wait_until(next_deadline)
        now = time.perf_counter()
        lateness_s = max(0.0, now - next_deadline)
        if lateness_s > 1e-4:
            late_frames += 1
            max_lateness_s = max(max_lateness_s, lateness_s)
        image_stream.write(frames[frame_index % len(frames)])
        next_deadline += interval_s
        if (
            (frame_index + 1) % poll_interval_frames == 0
            or frame_index + 1 == frame_count
        ):
            last_snapshot = _runtime_snapshot(manager)

    elapsed_s = time.perf_counter() - started_at
    return {
        "frames_written": frame_count,
        "effective_rate_hz": frame_count / max(elapsed_s, 1e-12),
        "late_frames_100us": late_frames,
        "max_lateness_us": 1_000_000.0 * max_lateness_s,
        "last_snapshot": last_snapshot,
    }


def _stage_metrics(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    metrics = snapshot.get("metrics", {})
    return {name: dict(values) for name, values in metrics.items()}


def run_benchmark(
    *,
    rate_hz: float = DEFAULT_RATE_HZ,
    warmup_s: float = DEFAULT_WARMUP_S,
    duration_s: float = DEFAULT_DURATION_S,
    frame_bank_size: int = DEFAULT_FRAME_BANK_SIZE,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    if rate_hz <= 0.0:
        raise ValueError("rate_hz must be positive")
    if warmup_s < 0.0:
        raise ValueError("warmup_s must be non-negative")
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    if frame_bank_size <= 0:
        raise ValueError("frame_bank_size must be positive")

    config_path = Path(__file__).with_name("pipeline.yaml")
    spawn_method = "fork" if sys.platform.startswith("linux") else "spawn"
    manager = PipelineManager(config_path, spawn_method=spawn_method)
    benchmark_inputs = _prepare_benchmark_inputs(frame_bank_size, seed=seed)
    warmup_frames = int(round(warmup_s * rate_hz))
    measurement_frames = max(1, int(round(duration_s * rate_hz)))

    manager.build()
    manager.start()

    try:
        manager.get_stream("obs_control_matrix").write(
            benchmark_inputs["control_matrix"]
        )
        manager.get_stream("obs_modal_bias").write(
            benchmark_inputs["effective_bias"]
        )
        manager.get_stream("obs_command_low_limit").write(
            benchmark_inputs["low_limit"]
        )
        manager.get_stream("obs_command_high_limit").write(
            benchmark_inputs["high_limit"]
        )

        image_stream = manager.get_stream("obs_wfs_image")
        command_stream = manager.get_stream("obs_dm_command")
        baseline = command_stream.count
        image_stream.write(benchmark_inputs["frames"][0])
        _wait_for_next_write(command_stream, baseline, timeout=5.0)

        warmup_summary: dict[str, Any] | None = None
        if warmup_frames > 0:
            warmup_summary = _drive_wfs_stream(
                manager,
                image_stream,
                benchmark_inputs["frames"],
                frame_count=warmup_frames,
                rate_hz=rate_hz,
            )

        measurement_command_start = command_stream.count
        measurement_summary = _drive_wfs_stream(
            manager,
            image_stream,
            benchmark_inputs["frames"],
            frame_count=measurement_frames,
            rate_hz=rate_hz,
        )

        expected_command_count = measurement_command_start + measurement_frames
        final_snapshot = measurement_summary["last_snapshot"]
        drain_deadline = time.monotonic() + 2.0
        while (
            command_stream.count < expected_command_count
            and time.monotonic() < drain_deadline
        ):
            time.sleep(0.01)
            final_snapshot = _runtime_snapshot(manager)

        stages = _stage_metrics(final_snapshot)
        if not stages:
            raise RuntimeError("no worker metrics were collected")
        slowest_stage_name = max(
            stages,
            key=lambda name: float(stages[name].get("avg_exec_us") or 0.0),
        )
        return {
            "config": str(config_path),
            "spawn_method": spawn_method,
            "requested_rate_hz": rate_hz,
            "warmup_s": warmup_s,
            "measurement_s": duration_s,
            "frame_bank_size": frame_bank_size,
            "seed": seed,
            "measurement_frames_written": measurement_frames,
            "measurement_command_frames_emitted": (
                command_stream.count - measurement_command_start
            ),
            "warmup": warmup_summary,
            "measurement": {
                key: value
                for key, value in measurement_summary.items()
                if key != "last_snapshot"
            },
            "slowest_stage": {
                "name": slowest_stage_name,
                **stages[slowest_stage_name],
            },
            "stages": stages,
        }
    finally:
        manager.shutdown(force=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the observatory AO reconstructor using the same "
            "256x256 WFS workload as the example pipeline."
        )
    )
    parser.add_argument("--rate-hz", type=float, default=DEFAULT_RATE_HZ)
    parser.add_argument("--warmup", type=float, default=DEFAULT_WARMUP_S)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument(
        "--frame-bank-size",
        type=int,
        default=DEFAULT_FRAME_BANK_SIZE,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="WARNING",
    )
    return parser


def _print_text_report(result: dict[str, Any]) -> None:
    print(
        "Observatory affine benchmark "
        f"rate={result['requested_rate_hz']:.1f}Hz "
        f"warmup={result['warmup_s']:.1f}s "
        f"duration={result['measurement_s']:.1f}s"
    )
    measurement = result["measurement"]
    print(
        "Input writer "
        f"effective_rate={measurement['effective_rate_hz']:.1f}Hz "
        f"late_frames_100us={measurement['late_frames_100us']} "
        f"max_lateness={measurement['max_lateness_us']:.1f}us"
    )
    print(
        "Command frames emitted "
        f"{result['measurement_command_frames_emitted']}/"
        f"{result['measurement_frames_written']}"
    )
    slowest = result["slowest_stage"]
    print(
        "Slowest stage "
        f"{slowest['name']} avg_exec_us={slowest.get('avg_exec_us', 0.0):.1f} "
        f"jitter_us_rms={slowest.get('jitter_us_rms', 0.0):.1f} "
        f"throughput_hz={slowest.get('throughput_hz', 0.0):.1f}"
    )
    print("Stage metrics (runtime avg_exec_us includes compute plus output publish):")
    for name, metrics in sorted(
        result["stages"].items(),
        key=lambda item: float(item[1].get("avg_exec_us") or 0.0),
        reverse=True,
    ):
        print(
            f"  {name}: avg_exec_us={metrics.get('avg_exec_us', 0.0):.1f} "
            f"jitter_us_rms={metrics.get('jitter_us_rms', 0.0):.1f} "
            f"throughput_hz={metrics.get('throughput_hz', 0.0):.1f} "
            f"window={metrics.get('metrics_window', 0)}"
        )


def main() -> int:
    args = _build_parser().parse_args()
    configure_colored_logging(level=getattr(logging, args.log_level))
    result = run_benchmark(
        rate_hz=args.rate_hz,
        warmup_s=args.warmup,
        duration_s=args.duration,
        frame_bank_size=args.frame_bank_size,
        seed=args.seed,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        _print_text_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())