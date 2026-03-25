"""Benchmark the AO pipeline on a 120x120 Shack-Hartmann workload."""

from __future__ import annotations

import logging
from pathlib import Path
import time

import numpy as np

from shmpipeline import PipelineManager
from shmpipeline.logging_utils import configure_colored_logging


def wait_for_next_write(stream, previous_count: int, *, timeout: float = 2.0):
    """Wait until a stream count advances and return its latest payload."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if stream.count > previous_count:
            return stream.read()
        time.sleep(1e-5)
    raise TimeoutError(f"timed out waiting for a new write on {stream.name!r}")


def main() -> None:
    """Run a throughput benchmark for a 120x120 AO workload."""
    configure_colored_logging()
    logger = logging.getLogger("shmpipeline.example.ao.benchmark")
    config_path = Path(__file__).with_name("benchmark_120x120.yaml")
    manager = PipelineManager(config_path)
    rng = np.random.default_rng(17)

    warmup_frames = 200
    benchmark_frames = 2000

    centroid_offset = rng.normal(0.0, 0.01, size=(15, 15, 2)).astype(np.float32)
    reconstructor = rng.normal(0.0, 0.03, size=(128, 450)).astype(np.float32)
    affine_offset = rng.normal(0.0, 0.01, size=(128,)).astype(np.float32)

    manager.build()
    manager.start()
    try:
        manager.get_stream("ao_bench_centroid_offset").write(centroid_offset)
        manager.get_stream("ao_bench_reconstructor_matrix").write(reconstructor)
        manager.get_stream("ao_bench_affine_offset").write(affine_offset)

        input_stream = manager.get_stream("ao_bench_sensor_image")
        output_stream = manager.get_stream("ao_bench_dm_command")

        logger.info(
            "starting benchmark: image=120x120 tile=8 subapertures=15x15 actuators=128 warmup=%d frames=%d",
            warmup_frames,
            benchmark_frames,
        )

        for _ in range(warmup_frames):
            baseline = output_stream.count
            frame = rng.uniform(0.01, 1.0, size=(120, 120)).astype(np.float32)
            input_stream.write(frame)
            wait_for_next_write(output_stream, baseline, timeout=2.0)

        start = time.perf_counter()
        for _ in range(benchmark_frames):
            baseline = output_stream.count
            frame = rng.uniform(0.01, 1.0, size=(120, 120)).astype(np.float32)
            input_stream.write(frame)
            wait_for_next_write(output_stream, baseline, timeout=2.0)
        elapsed = time.perf_counter() - start

        rate_hz = benchmark_frames / elapsed
        logger.info(
            "benchmark complete: frames=%d elapsed=%.3fs rate=%.1f Hz",
            benchmark_frames,
            elapsed,
            rate_hz,
        )

        target_hz = 2000.0
        if rate_hz >= target_hz:
            logger.info("target met: %.1f Hz >= %.1f Hz", rate_hz, target_hz)
        else:
            logger.warning("target not met yet: %.1f Hz < %.1f Hz", rate_hz, target_hz)

        print(f"AO benchmark rate: {rate_hz:.1f} Hz")
    finally:
        manager.shutdown(force=True)


if __name__ == "__main__":
    main()