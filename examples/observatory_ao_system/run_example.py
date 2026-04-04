"""Run a representative 8-10m class observatory AO control chain."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

from shmpipeline import PipelineManager
from shmpipeline.logging_utils import configure_colored_logging

IMAGE_SHAPE = (256, 256)
SUBAPERTURE_GRID = (32, 32)
TILE_SIZE = 8
ACTUATOR_COUNT = 1024
FRAME_COUNT = 120
CONTROL_LEAK = 0.985
CONTROL_GAIN = 0.28
SPOT_SIGMA_PX = 1.05


def compute_centroids(image: np.ndarray, tile_size: int) -> np.ndarray:
    """Compute local centroids as `[delta_row, delta_col]` pairs."""
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


def wait_for_next_write(stream, previous_count: int, *, timeout: float = 5.0):
    """Wait until a stream count advances and return its latest payload."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if stream.count > previous_count:
            return stream.read()
        time.sleep(1e-4)
    raise TimeoutError(f"timed out waiting for a new write on {stream.name!r}")


def render_shack_hartmann_image(
    spot_offsets: np.ndarray,
    flux_map: np.ndarray,
    *,
    tile_size: int,
    spot_sigma_px: float,
) -> np.ndarray:
    """Render one tiled Shack-Hartmann image with Gaussian spots."""
    image = np.zeros(
        (spot_offsets.shape[0] * tile_size, spot_offsets.shape[1] * tile_size),
        dtype=np.float32,
    )
    y_coords, x_coords = np.indices((tile_size, tile_size), dtype=np.float32)
    center = 0.5 * (tile_size - 1)
    sigma2 = float(spot_sigma_px) ** 2
    for tile_y in range(spot_offsets.shape[0]):
        row_start = tile_y * tile_size
        for tile_x in range(spot_offsets.shape[1]):
            col_start = tile_x * tile_size
            delta_row = float(spot_offsets[tile_y, tile_x, 0])
            delta_col = float(spot_offsets[tile_y, tile_x, 1])
            row_center = center + delta_row
            col_center = center + delta_col
            patch = np.exp(
                -((y_coords - row_center) ** 2 + (x_coords - col_center) ** 2)
                / (2.0 * sigma2)
            )
            image[
                row_start : row_start + tile_size,
                col_start : col_start + tile_size,
            ] = flux_map[tile_y, tile_x] * patch
    return image


def make_reference_centroids(
    grid_y: np.ndarray,
    grid_x: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a static reference slope field with low-order structure."""
    reference_y = 0.10 * grid_y + 0.04 * np.sin(np.pi * grid_x)
    reference_x = -0.10 * grid_x + 0.04 * np.cos(np.pi * grid_y)
    reference = np.stack([reference_y, reference_x], axis=-1)
    reference += rng.normal(0.0, 0.015, size=reference.shape)
    return np.clip(reference, -0.35, 0.35).astype(np.float32)


def synthesize_residual_centroids(
    frame_index: int,
    grid_y: np.ndarray,
    grid_x: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a deterministic high-order residual slope field."""
    phase = 0.17 * frame_index
    residual_y = 0.55 * np.sin(
        2.0 * np.pi * (0.45 * grid_x + 0.18 * grid_y) + phase
    )
    residual_y += 0.25 * np.cos(
        2.0 * np.pi * (0.12 * grid_x - 0.34 * grid_y) - 0.60 * phase
    )
    residual_x = 0.48 * np.cos(
        2.0 * np.pi * (0.38 * grid_x - 0.22 * grid_y) - 0.80 * phase
    )
    residual_x -= 0.23 * np.sin(
        2.0 * np.pi * (0.17 * grid_x + 0.52 * grid_y) + 0.40 * phase
    )
    residual_y += 0.09 * grid_y * (1.0 + 0.5 * np.sin(0.3 * phase))
    residual_x -= 0.09 * grid_x * (1.0 + 0.5 * np.cos(0.3 * phase))
    residual = np.stack([residual_y, residual_x], axis=-1)
    residual += rng.normal(0.0, 0.02, size=residual.shape)
    return np.clip(residual, -1.2, 1.2).astype(np.float32)


def make_flux_map(
    base_flux_map: np.ndarray,
    frame_index: int,
    grid_y: np.ndarray,
    grid_x: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply a mild scintillation-like modulation to the spot flux."""
    modulation = 1.0 + 0.10 * np.sin(
        2.0 * np.pi * (0.22 * grid_x - 0.14 * grid_y) + 0.11 * frame_index
    )
    modulation += rng.normal(0.0, 0.015, size=base_flux_map.shape)
    return np.clip(base_flux_map * modulation, 45.0, None).astype(np.float32)


def make_command_limits(
    rng: np.random.Generator,
    actuator_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create per-actuator low and high stroke limits."""
    base_limit = 2.4 + 0.25 * rng.random(actuator_count)
    low_limit = -(base_limit + 0.10 * rng.random(actuator_count))
    high_limit = base_limit + 0.10 * rng.random(actuator_count)
    return low_limit.astype(np.float32), high_limit.astype(np.float32)


def main() -> None:
    """Verify a high-order observatory-style AO loop."""
    configure_colored_logging(level=logging.INFO)
    logger = logging.getLogger("shmpipeline.example.observatory_ao")
    config_path = Path(__file__).with_name("pipeline.yaml")
    spawn_method = "fork" if sys.platform.startswith("linux") else "spawn"
    manager = PipelineManager(config_path, spawn_method=spawn_method)
    rng = np.random.default_rng(20260403)

    subap_y = np.linspace(-1.0, 1.0, SUBAPERTURE_GRID[0], dtype=np.float32)
    subap_x = np.linspace(-1.0, 1.0, SUBAPERTURE_GRID[1], dtype=np.float32)
    grid_y, grid_x = np.meshgrid(subap_y, subap_x, indexing="ij")

    reference_centroids = make_reference_centroids(grid_y, grid_x, rng)
    control_matrix = rng.normal(
        0.0,
        0.015,
        size=(ACTUATOR_COUNT, np.prod(SUBAPERTURE_GRID) * 2),
    ).astype(np.float32)
    modal_bias = rng.normal(0.0, 0.02, size=(ACTUATOR_COUNT,)).astype(
        np.float32
    )
    effective_bias = (
        modal_bias - control_matrix @ reference_centroids.reshape(-1)
    ).astype(np.float32)
    low_limit, high_limit = make_command_limits(rng, ACTUATOR_COUNT)
    base_flux_map = rng.uniform(70.0, 140.0, size=SUBAPERTURE_GRID).astype(
        np.float32
    )
    integrated_state = np.zeros(ACTUATOR_COUNT, dtype=np.float32)
    total_saturated = 0

    logger.info(
        "building observatory AO pipeline from %s with spawn_method=%s",
        config_path,
        spawn_method,
    )
    manager.build()
    manager.start()

    try:
        manager.get_stream("obs_control_matrix").write(control_matrix)
        manager.get_stream("obs_modal_bias").write(effective_bias)
        manager.get_stream("obs_command_low_limit").write(low_limit)
        manager.get_stream("obs_command_high_limit").write(high_limit)

        image_stream = manager.get_stream("obs_wfs_image")
        centroid_stream = manager.get_stream("obs_measured_centroids")
        flattened_stream = manager.get_stream("obs_flattened_slopes")
        open_loop_stream = manager.get_stream("obs_open_loop_command")
        integrated_stream = manager.get_stream("obs_integrated_command")
        command_stream = manager.get_stream("obs_dm_command")

        logger.info(
            "loaded static AO calibrations: image=%sx%s subapertures=%sx%s actuators=%d",
            IMAGE_SHAPE[0],
            IMAGE_SHAPE[1],
            SUBAPERTURE_GRID[0],
            SUBAPERTURE_GRID[1],
            ACTUATOR_COUNT,
        )

        start = time.perf_counter()
        for index in range(FRAME_COUNT):
            residual_centroids = synthesize_residual_centroids(
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
            flux_map = make_flux_map(
                base_flux_map,
                index,
                grid_y,
                grid_x,
                rng,
            )
            wfs_image = render_shack_hartmann_image(
                measured_centroids,
                flux_map,
                tile_size=TILE_SIZE,
                spot_sigma_px=SPOT_SIGMA_PX,
            )

            expected_centroids = compute_centroids(wfs_image, TILE_SIZE)
            expected_flattened = expected_centroids.reshape(-1)
            expected_open_loop = (
                control_matrix @ expected_flattened + effective_bias
            )
            integrated_state = (
                CONTROL_LEAK * integrated_state
                + CONTROL_GAIN * expected_open_loop
            )
            expected_integrated = integrated_state.copy()
            expected_command = np.clip(
                expected_integrated,
                low_limit,
                high_limit,
            )
            expected_residual = expected_centroids - reference_centroids

            baseline = command_stream.count
            image_stream.write(wfs_image)
            observed_command = wait_for_next_write(
                command_stream,
                baseline,
                timeout=5.0,
            )
            observed_centroids = centroid_stream.read()
            observed_flattened = flattened_stream.read()
            observed_open_loop = open_loop_stream.read()
            observed_integrated = integrated_stream.read()

            np.testing.assert_allclose(
                observed_centroids,
                expected_centroids,
                rtol=1e-5,
                atol=1e-5,
            )
            np.testing.assert_allclose(
                observed_flattened,
                expected_flattened,
                rtol=1e-5,
                atol=1e-5,
            )
            np.testing.assert_allclose(
                observed_open_loop,
                expected_open_loop,
                rtol=2e-5,
                atol=2e-5,
            )
            np.testing.assert_allclose(
                observed_integrated,
                expected_integrated,
                rtol=2e-5,
                atol=2e-5,
            )
            np.testing.assert_allclose(
                observed_command,
                expected_command,
                rtol=2e-5,
                atol=2e-5,
            )

            saturated_count = int(
                np.count_nonzero(
                    np.abs(expected_command - expected_integrated) > 1e-6
                )
            )
            total_saturated += saturated_count
            residual_rms = float(np.sqrt(np.mean(expected_residual**2)))
            command_rms = float(np.sqrt(np.mean(expected_command**2)))

            if index == 0:
                logger.info(
                    "first frame verified: residual_rms_px=%.3f command_rms=%.3f saturated=%d",
                    residual_rms,
                    command_rms,
                    saturated_count,
                )
            elif (index + 1) % 20 == 0:
                logger.info(
                    "verified %d/%d frames residual_rms_px=%.3f command_rms=%.3f saturated=%d",
                    index + 1,
                    FRAME_COUNT,
                    residual_rms,
                    command_rms,
                    saturated_count,
                )

        elapsed = time.perf_counter() - start
        logger.info(
            "observatory AO verification complete: frames=%d elapsed=%.3fs throughput=%.1f frames/s saturation_fraction=%.4f",
            FRAME_COUNT,
            elapsed,
            FRAME_COUNT / elapsed,
            total_saturated / (FRAME_COUNT * ACTUATOR_COUNT),
        )
        print(
            f"Verified {FRAME_COUNT} observatory AO frames in {elapsed:.3f}s"
        )
    finally:
        manager.shutdown(force=True)


if __name__ == "__main__":
    main()
