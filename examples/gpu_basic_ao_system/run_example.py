"""Run the basic GPU AO system example pipeline."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch

from shmpipeline import PipelineManager
from shmpipeline.logging_utils import configure_colored_logging


def compute_centroids(image: np.ndarray, tile_size: int) -> np.ndarray:
    """Compute local tile centroids as [delta_row, delta_col] pairs."""
    tiles_y = image.shape[0] // tile_size
    tiles_x = image.shape[1] // tile_size
    center = 0.5 * (tile_size - 1)
    centroids = np.zeros((tiles_y, tiles_x, 2), dtype=np.float32)
    for tile_y in range(tiles_y):
        row_start = tile_y * tile_size
        for tile_x in range(tiles_x):
            col_start = tile_x * tile_size
            patch = image[
                row_start : row_start + tile_size,
                col_start : col_start + tile_size,
            ]
            total = float(np.sum(patch))
            if total <= 0.0:
                continue
            y_coords, x_coords = np.indices(patch.shape, dtype=np.float32)
            centroids[tile_y, tile_x, 0] = (
                np.sum(y_coords * patch) / total - center
            )
            centroids[tile_y, tile_x, 1] = (
                np.sum(x_coords * patch) / total - center
            )
    return centroids


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
    """Verify a basic AO pipeline across many random Shack-Hartmann frames."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this GPU example")

    configure_colored_logging()
    logger = logging.getLogger("shmpipeline.example.gpu_ao")
    config_path = Path(__file__).with_name("pipeline.yaml")
    manager = PipelineManager(config_path)

    rng = np.random.default_rng(11)
    frame_count = 3000
    tile_size = 2
    centroid_gain = 1.75
    control_leak = 0.92
    control_gain = 0.35
    centroid_offset = rng.normal(0.0, 0.03, size=(4, 4, 2)).astype(np.float32)
    reconstructor_matrix = rng.normal(0.0, 0.2, size=(6, 32)).astype(
        np.float32
    )
    affine_offset = rng.normal(0.0, 0.1, size=(6,)).astype(np.float32)
    controller_state = np.zeros(6, dtype=np.float32)

    logger.info("building GPU AO example pipeline from %s", config_path)
    manager.build()
    manager.start()

    try:
        manager.get_stream("ao_centroid_offset").write(
            to_device(centroid_offset)
        )
        manager.get_stream("ao_reconstructor_matrix").write(
            to_device(reconstructor_matrix)
        )
        manager.get_stream("ao_affine_offset").write(to_device(affine_offset))
        logger.info("loaded static GPU AO calibration streams")

        start = time.perf_counter()
        for index in range(frame_count):
            image = rng.uniform(0.0, 1.0, size=(8, 8)).astype(np.float32)
            image += 0.05

            expected_centroids = compute_centroids(image, tile_size)
            expected_corrected = (
                centroid_gain * expected_centroids - centroid_offset
            )
            expected_flattened = expected_corrected.reshape(-1)
            expected_open_loop = (
                reconstructor_matrix @ expected_flattened + affine_offset
            )
            controller_state = (
                control_leak * controller_state
                + control_gain * expected_open_loop
            )
            expected_dm_command = controller_state.copy()

            baseline = manager.get_stream("ao_dm_command").count
            manager.get_stream("ao_sensor_image").write(to_device(image))
            dm_command = to_host(
                wait_for_next_write(
                    manager.get_stream("ao_dm_command"),
                    baseline,
                    timeout=2.0,
                )
            )

            observed_centroids = to_host(
                manager.get_stream("ao_centroids").read()
            )
            observed_corrected = to_host(
                manager.get_stream("ao_corrected_centroids").read()
            )
            observed_flattened = to_host(
                manager.get_stream("ao_flattened_slopes").read()
            )
            observed_open_loop = to_host(
                manager.get_stream("ao_open_loop_command").read()
            )

            np.testing.assert_allclose(
                observed_centroids, expected_centroids, rtol=1e-5, atol=1e-5
            )
            np.testing.assert_allclose(
                observed_corrected, expected_corrected, rtol=1e-5, atol=1e-5
            )
            np.testing.assert_allclose(
                observed_flattened, expected_flattened, rtol=1e-5, atol=1e-5
            )
            np.testing.assert_allclose(
                observed_open_loop, expected_open_loop, rtol=1e-5, atol=1e-5
            )
            np.testing.assert_allclose(
                dm_command, expected_dm_command, rtol=1e-5, atol=1e-5
            )

            if index == 0:
                logger.info(
                    "first frame verified: centroids=%s dm_command=%s",
                    observed_centroids[0, 0].tolist(),
                    dm_command.tolist(),
                )
            elif (index + 1) % 500 == 0:
                logger.info(
                    "verified %d/%d GPU AO frames", index + 1, frame_count
                )

        elapsed = time.perf_counter() - start
        logger.info(
            "GPU AO verification complete: frames=%d elapsed=%.3fs throughput=%.1f frames/s",
            frame_count,
            elapsed,
            frame_count / elapsed,
        )
        print(f"Verified {frame_count} GPU AO frames in {elapsed:.3f}s")
    finally:
        manager.shutdown(force=True)


if __name__ == "__main__":
    main()
