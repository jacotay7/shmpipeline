"""Run the affine transformation example pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
import time

import numpy as np

from shmpipeline import PipelineManager


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging() -> None:
    """Configure console logging for the example script."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def main() -> None:
    """Build the example pipeline and verify thousands of affine transforms."""
    configure_logging()
    logger = logging.getLogger("shmpipeline.example.affine")
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

    logger.info("building affine example pipeline from %s", config_path)
    manager.build()
    manager.start()

    try:
        matrix_stream = manager.get_stream("affine_transform_matrix")
        offset_stream = manager.get_stream("affine_offset_vector")
        input_stream = manager.get_stream("affine_input_vector")
        output_stream = manager.get_stream("affine_output_vector")

        logger.info("loading transform matrix and offset vector into shared memory")
        matrix_stream.write(transform_matrix)
        offset_stream.write(offset_vector)

        start = time.perf_counter()
        for index in range(sample_count):
            vector = rng.standard_normal(3, dtype=np.float32)
            expected = transform_matrix @ vector + offset_vector
            input_stream.write(vector)
            result = output_stream.read_new(timeout=2.0)
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
        print(f"Verified {sample_count} affine transforms in {elapsed:.3f}s")
    finally:
        manager.shutdown(force=True)


if __name__ == "__main__":
    main()