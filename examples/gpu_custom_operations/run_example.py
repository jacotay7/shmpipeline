"""Run the GPU custom-operations example pipeline."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch

from shmpipeline import PipelineManager
from shmpipeline.logging_utils import configure_colored_logging


def wait_for_next_write(stream, previous_count: int, *, timeout: float = 2.0):
    """Wait until a stream count advances and return its latest payload."""
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
    """Build the GPU custom-operation example and verify one fused expression."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this GPU example")

    configure_colored_logging(level=logging.INFO)
    config_path = Path(__file__).with_name("pipeline.yaml")
    manager = PipelineManager(config_path)
    manager.build()
    manager.start()

    raw_image = (np.arange(16, dtype=np.float32).reshape(4, 4) + 20.0).astype(
        np.float32
    )
    dark_frame = np.full((4, 4), 2.0, dtype=np.float32)
    flat_field = np.linspace(1.0, 2.5, 16, dtype=np.float32).reshape(4, 4)
    expected = (raw_image - dark_frame) / flat_field

    try:
        manager.get_stream("dark_frame").write(to_device(dark_frame))
        manager.get_stream("flat_field").write(to_device(flat_field))
        output_stream = manager.get_stream("clean_image")
        baseline = output_stream.count
        manager.get_stream("raw_image").write(to_device(raw_image))
        result = to_host(wait_for_next_write(output_stream, baseline))

        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
        logging.getLogger("shmpipeline.example.gpu_custom_operations").info(
            "GPU custom operation verified: expression=%s output_shape=%s",
            "(input - dark) / flat",
            list(result.shape),
        )
    finally:
        manager.shutdown(force=True)


if __name__ == "__main__":
    main()
