"""Run the GPU affine transformation example pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
import time

import numpy as np
import torch

from shmpipeline import PipelineManager
from shmpipeline.logging_utils import configure_colored_logging


def wait_for_next_write(stream, previous_count: int, *, timeout: float = 2.0):
    """Wait until a stream's write count advances beyond a known baseline."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if stream.count > previous_count:
            return stream.read()
        time.sleep(1e-4)
    raise TimeoutError(f"timed out waiting for a new write on {stream.name!r}")


def to_device(array: np.ndarray) -> torch.Tensor:
    """Move one NumPy array onto the configured CUDA device."""
    return torch.as_tensor(array, device="cuda:0")


def to_host(tensor: torch.Tensor) -> np.ndarray:
    """Detach one CUDA tensor into a NumPy array for verification."""
    return tensor.detach().cpu().numpy().copy()


def main() -> None:
    """Build the example pipeline and verify thousands of affine transforms."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this GPU example")

    configure_colored_logging()
    logger = logging.getLogger("shmpipeline.example.gpu_affine")
    config_path = Path(__file__).with_name("pipeline.yaml")
    manager = PipelineManager(config_path)
    rng = np.random.default_rng(7)
    sample_count = 5000

    transform_matrix = np.array(
        [
            [1.0, 2.0, 0.0],
            [0.0, -1.0, 3.0],
        ],
        dtype=np.float32,
    )
    offset_vector = np.array([0.5, -2.0], dtype=np.float32)

    logger.info("building GPU affine example pipeline from %s", config_path)
    manager.build()
    manager.start()

    try:
        matrix_stream = manager.get_stream("affine_transform_matrix")
        offset_stream = manager.get_stream("affine_offset_vector")
        input_stream = manager.get_stream("affine_input_vector")
        output_stream = manager.get_stream("affine_output_vector")

        logger.info("loading transform matrix and offset vector into GPU shared memory")
        matrix_stream.write(to_device(transform_matrix))
        offset_stream.write(to_device(offset_vector))

        start = time.perf_counter()
        for index in range(sample_count):
            vector = rng.standard_normal(3, dtype=np.float32)
            expected = transform_matrix @ vector + offset_vector
            baseline = output_stream.count
            input_stream.write(to_device(vector))
            result = to_host(wait_for_next_write(output_stream, baseline, timeout=2.0))
            np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
            if index == 0:
                logger.info(
                    "first sample verified: input=%s output=%s",
                    vector.tolist(),
                    result.tolist(),
                )
            elif (index + 1) % 1000 == 0:
                logger.info("verified %d/%d vectors", index + 1, sample_count)

        elapsed = time.perf_counter() - start
        logger.info(
            "verification complete: samples=%d elapsed=%.3fs throughput=%.1f vectors/s",
            sample_count,
            elapsed,
            sample_count / elapsed,
        )
        print(f"Verified {sample_count} GPU affine transforms in {elapsed:.3f}s")
    finally:
        manager.shutdown(force=True)


if __name__ == "__main__":
    main()