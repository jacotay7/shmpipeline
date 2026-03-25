from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from shmpipeline import PipelineManager
from shmpipeline.logging_utils import configure_colored_logging


def wait_for_next_write(stream, previous_count: int, *, timeout: float = 2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if stream.count > previous_count:
            return stream.read()
        time.sleep(1e-4)
    raise TimeoutError(f"timed out waiting for a new write on {stream.name!r}")


def main() -> None:
    configure_colored_logging(level=logging.INFO)
    config_path = Path(__file__).with_name("pipeline.yaml")
    manager = PipelineManager(config_path)
    manager.build()
    manager.start()

    raw_image = (np.arange(16, dtype=np.float32).reshape(4, 4) + 20.0).astype(np.float32)
    dark_frame = np.full((4, 4), 2.0, dtype=np.float32)
    flat_field = np.linspace(1.0, 2.5, 16, dtype=np.float32).reshape(4, 4)
    expected = (raw_image - dark_frame) / flat_field

    manager.get_stream("dark_frame").write(dark_frame)
    manager.get_stream("flat_field").write(flat_field)
    output_stream = manager.get_stream("clean_image")
    baseline = output_stream.count
    manager.get_stream("raw_image").write(raw_image)
    result = wait_for_next_write(output_stream, baseline)

    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
    logging.getLogger("shmpipeline.example.custom_operations").info(
        "custom operation verified: expression=%s output_shape=%s",
        "(input - dark) / flat",
        result.shape,
    )

    manager.shutdown()


if __name__ == "__main__":
    main()