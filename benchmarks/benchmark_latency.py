"""Measure end-to-end frame latency through a running pipeline and write JSON.

The driver publishes one frame into the input stream, spins until the output
stream's publication count advances, and records the elapsed time before
publishing the next frame (ping-pong, one frame in flight).  Unlike
``benchmark_pipeline.py``, which reports output inter-arrival spacing, this
measures the latency a closed AO loop sees from publication to result.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import socket
import time
from pathlib import Path
from typing import Any

import numpy as np

from shmpipeline.config import PipelineConfig
from shmpipeline.manager import PipelineManager
from shmpipeline.scheduling import RoundRobinPlacementPolicy


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("input_stream")
    parser.add_argument("output_stream")
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument(
        "--driver-cpu",
        type=int,
        default=None,
        help="pin the driver process to this CPU (Linux only)",
    )
    parser.add_argument(
        "--worker-offset",
        type=int,
        default=0,
        help="first CPU of the round-robin worker placement",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    return parser


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _wait_for_count(stream: Any, target: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while stream.count < target:
        if time.monotonic() > deadline:
            raise TimeoutError("the pipeline did not publish an output")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.driver_cpu is not None:
        os.sched_setaffinity(0, {args.driver_cpu})
    manager = PipelineManager(
        PipelineConfig.from_yaml(args.config),
        placement_policy=RoundRobinPlacementPolicy(offset=args.worker_offset),
        worker_start_timeout=120.0,
    )
    try:
        manager.build()
        manager.start()
        source = manager.get_stream(args.input_stream)
        sink = manager.get_stream(args.output_stream)
        rng = np.random.default_rng(0)
        frames = [
            rng.random(source.shape).astype(source.dtype) for _ in range(8)
        ]
        for index in range(args.warmup):
            target = sink.count + 1
            source.write(frames[index % len(frames)])
            _wait_for_count(sink, target, 30.0)
        samples = np.empty(args.iterations, dtype=np.int64)
        base = sink.count
        started = time.perf_counter_ns()
        for index in range(args.iterations):
            frame_started = time.perf_counter_ns()
            source.write(frames[index % len(frames)])
            _wait_for_count(sink, base + index + 1, 10.0)
            samples[index] = time.perf_counter_ns() - frame_started
        elapsed_s = (time.perf_counter_ns() - started) / 1e9
        manager.raise_if_failed()
        metrics = manager.status().get("metrics", {})
    finally:
        manager.shutdown(force=True)
    latency_us = samples / 1e3
    return {
        "benchmark": "pipeline_latency",
        "config": str(args.config),
        "iterations": args.iterations,
        "frames_per_s": args.iterations / elapsed_s,
        "latency_us": {
            "p50": float(np.percentile(latency_us, 50)),
            "p99": float(np.percentile(latency_us, 99)),
            "p99.9": float(np.percentile(latency_us, 99.9)),
            "max": float(latency_us.max()),
        },
        "workers_avg_exec_us": {
            name: worker.get("avg_exec_us") for name, worker in metrics.items()
        },
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyshmem": _version("pyshmem"),
            "shmpipeline": _version("shmpipeline"),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.iterations < 1 or args.warmup < 0:
        raise SystemExit("iterations must be positive and warmup >= 0")
    report = run(args)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
