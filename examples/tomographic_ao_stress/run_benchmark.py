"""Run the synthetic three-loop tomographic AO stress workload.

Loads the single CPU or GPU pipeline (`pipeline_<backend>.yaml`), replaces the
self-contained YAML camera source with an instrumented one that stamps frame_id
tokens and records source timestamps, loads deterministic calibration arrays,
and measures sustained rate, delivery ratio, end-to-end latency, and the
matching_frame_id barrier's skew behaviour. The YAML null sinks remain the fake
hardware endpoints.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

import shmpipeline
from shmpipeline import PipelineConfig, PipelineManager
from shmpipeline.logging_utils import configure_colored_logging

WFS_COUNT = 8
WFS_SHAPE = (256, 256)
RECONSTRUCTOR_SHAPE = (4096, 65536)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Override pipeline_<backend>.yaml (for example the batched GPU profile).",
    )
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--warmup", type=float, default=1.0)
    parser.add_argument(
        "--main-rate",
        type=float,
        default=None,
        help="Camera-set rate in Hz; omit for unthrottled (saturation) mode.",
    )
    parser.add_argument("--loop-a-rate", type=float, default=700.0)
    parser.add_argument("--loop-b-rate", type=float, default=233.0)
    parser.add_argument("--camera-jitter-us", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2401)
    parser.add_argument(
        "--trace-latency",
        action="store_true",
        help="Stamp a frame_id token on every camera set, propagate it to the "
        "DM command, and measure true input-to-sink latency percentiles.",
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
            stream.shape, value, dtype=torch.float32, device=stream.gpu_device
        )
    return np.full(stream.shape, value, dtype=stream.dtype)


def _load_calibrations(manager: PipelineManager) -> None:
    """Load deterministic static calibration arrays (outside the timed run).

    Zeros/ones are sufficient: this benchmark measures data movement and the
    matrix multiply, not reconstruction fidelity.
    """
    logger = logging.getLogger("shmpipeline.example.tomographic_ao")
    logger.info("initializing static calibration streams")
    names = manager.config.shared_memory_by_name
    if "tomo_wfs_cube_raw" in names:
        _fill_stream(manager.get_stream("tomo_wfs_dark"), 0.0)
        _fill_stream(manager.get_stream("tomo_wfs_inverse_flat"), 1.0)
        _fill_stream(manager.get_stream("tomo_wfs_slope_offset"), 0.0)
    else:
        for index in range(WFS_COUNT):
            prefix = f"tomo_wfs{index}"
            _fill_stream(manager.get_stream(f"{prefix}_dark"), 0.0)
            _fill_stream(manager.get_stream(f"{prefix}_inverse_flat"), 1.0)
            _fill_stream(manager.get_stream(f"{prefix}_slope_offset"), 0.0)
    _fill_stream(manager.get_stream("tomo_reconstructor"), 0.0)
    _fill_stream(manager.get_stream("tomo_reconstructor_bias"), 0.0)
    _fill_stream(manager.get_stream("tomo_command_offset"), 0.0)
    _fill_stream(manager.get_stream("tomo_command_low"), -2.5)
    _fill_stream(manager.get_stream("tomo_command_high"), 2.5)
    for loop, angle in (("a", np.deg2rad(12.0)), ("b", np.deg2rad(-7.0))):
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ],
            dtype=np.float32,
        )
        manager.get_stream(f"tt_{loop}_rotation").write(rotation)
        _fill_stream(manager.get_stream(f"tt_{loop}_bias"), 0.0)


class _LatencyTrace:
    """Map a propagated frame_id token to its source publication timestamp.

    The camera thread records a monotonic timestamp when it publishes a
    generation; the tracer thread resolves it when the DM command carrying the
    same token reaches the terminal stage. Unresolved tokens (dropped frames)
    are pruned so the table stays bounded.
    """

    def __init__(self, capacity: int = 200_000) -> None:
        self._lock = threading.Lock()
        self._starts: dict[int, float] = {}
        self._samples: list[float] = []
        self._capacity = capacity

    def record_start(self, token: int, timestamp: float) -> None:
        with self._lock:
            self._starts[token] = timestamp
            if len(self._starts) > self._capacity:
                stale = list(self._starts)[: self._capacity // 2]
                for key in stale:
                    del self._starts[key]

    def resolve(self, token: int, timestamp: float) -> None:
        with self._lock:
            start = self._starts.pop(token, None)
        if start is not None:
            self._samples.append(timestamp - start)

    def samples(self) -> list[float]:
        with self._lock:
            return list(self._samples)

    def clear_samples(self) -> None:
        with self._lock:
            self._samples.clear()


def _latency_tracer(
    stream: Any, trace: _LatencyTrace, stop: threading.Event
) -> None:
    """Resolve DM-command tokens to end-to-end latency samples."""
    last = int(stream.count)
    while not stop.is_set():
        try:
            last = int(stream.wait_for_count(after=last, timeout=0.1))
        except TimeoutError:
            continue
        moment = time.perf_counter()
        token = int(stream.frame_id)
        if token:
            trace.resolve(token, moment)


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
    trace: _LatencyTrace | None = None,
) -> None:
    interval = None if rate_hz is None else 1.0 / rate_hz
    deadline = time.perf_counter()
    token = 0
    while not stop.is_set():
        token += 1
        frame_id = token if trace is not None else None
        if trace is not None:
            trace.record_start(token, time.perf_counter())
        for stream, payload in zip(streams, payloads):
            if jitter_s > 0.0 and rng is not None:
                delay = float(rng.uniform(0.0, jitter_s))
                if stop.wait(delay):
                    return
            stream.write(payload, frame_id=frame_id)
        if stats is not None:
            stats[stats_key] = stats.get(stats_key, 0) + 1
        if interval is not None:
            deadline += interval
            remaining = deadline - time.perf_counter()
            if remaining > 0.0:
                stop.wait(remaining)
            else:
                deadline = time.perf_counter()


def _start_sources(manager: PipelineManager, args, trace=None):
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
                "trace": trace,
            },
            name="tomographic-camera-set",
            daemon=True,
        )
    ]
    if trace is not None:
        threads.append(
            threading.Thread(
                target=_latency_tracer,
                args=(manager.get_stream("tomo_dm_command"), trace, stop),
                name="tomographic-latency-tracer",
                daemon=True,
            )
        )
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


def _latency_summary_us(samples: list[float]) -> dict[str, Any]:
    """Return input-to-sink latency percentiles in microseconds."""
    if not samples:
        return {"resolved_frames": 0}
    ordered = sorted(samples)

    def percentile(fraction: float) -> float:
        if len(ordered) == 1:
            return ordered[0] * 1e6
        position = fraction * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return (
            ordered[lower] * (1.0 - weight) + ordered[upper] * weight
        ) * 1e6

    return {
        "resolved_frames": len(ordered),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p99": percentile(0.99),
        "p99_9": percentile(0.999),
        "max": ordered[-1] * 1e6,
    }


def _git_revision() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parent,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def _environment_metadata(backend: str) -> dict[str, Any]:
    """Collect enough environment detail to reproduce and compare runs."""
    meta: dict[str, Any] = {
        "git_revision": _git_revision(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "shmpipeline_version": getattr(shmpipeline, "__version__", "unknown"),
    }
    try:
        import pyshmem

        meta["pyshmem_version"] = getattr(pyshmem, "__version__", "unknown")
    except Exception:
        meta["pyshmem_version"] = None
    meta["numpy_version"] = np.__version__
    if backend == "gpu":
        try:
            import torch

            meta["torch_version"] = torch.__version__
            if torch.cuda.is_available():
                meta["cuda_device"] = torch.cuda.get_device_name(0)
                meta["cuda_version"] = torch.version.cuda
        except Exception:
            meta["torch_version"] = None
    return meta


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.duration <= 0.0 or args.warmup < 0.0:
        raise ValueError("duration must be positive and warmup non-negative")
    config_path = (
        args.config
        if args.config is not None
        else Path(__file__).with_name(f"pipeline_{args.backend}.yaml")
    )
    config = PipelineConfig.from_yaml(config_path)
    # The harness owns the cameras (rate control + frame-id timestamps), so
    # drop the YAML's self-contained camera/tip-tilt sources.
    config = replace(config, sources=())
    if args.trace_latency:
        # Propagate the camera-set token along every stage so it reaches the
        # DM command. Kernels that already gate on the token keep their policy.
        config = replace(
            config,
            kernels=tuple(
                kernel
                if kernel.tracks_frame_id
                else replace(kernel, propagate_frame_id=True)
                for kernel in config.kernels
            ),
        )
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
    manager = PipelineManager(config)
    trace = _LatencyTrace() if args.trace_latency else None
    source_stop = None
    threads: list[threading.Thread] = []
    source_stats: dict[str, int] = {}
    try:
        manager.build()
        _load_calibrations(manager)
        manager.start()
        source_stop, threads, source_stats = _start_sources(
            manager, args, trace
        )
        time.sleep(args.warmup)
        if trace is not None:
            trace.clear_samples()
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
        metrics = manager.status().get("metrics", {})
        controller = metrics.get("tomo_controller", {})
        return {
            "backend": args.backend,
            "config": str(config_path),
            "environment": _environment_metadata(args.backend),
            "duration_s": elapsed,
            "warmup_s": args.warmup,
            "requested_main_rate_hz": args.main_rate,
            "camera_jitter_us": args.camera_jitter_us,
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
            "frame_sync": {
                "skew_events": controller.get("frame_sync_skew_events"),
                "skipped_generations": controller.get(
                    "frame_sync_skipped_generations"
                ),
                "timeouts": controller.get("frame_sync_timeouts"),
            },
            "latency_us": (
                _latency_summary_us(trace.samples())
                if trace is not None
                else None
            ),
            "worker_metrics": metrics,
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
