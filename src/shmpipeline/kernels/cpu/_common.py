"""Shared helpers for CPU kernel implementations."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from shmpipeline.config import KernelConfig, SharedMemoryConfig
from shmpipeline.errors import ConfigValidationError

try:
    from numba import njit
except Exception:  # pragma: no cover - exercised when numba is unavailable
    njit = None


def require_numeric_parameter(
    config: KernelConfig,
    *,
    name: str,
) -> float:
    """Return a numeric kernel parameter as float."""
    value = config.parameters.get(name)
    if not isinstance(value, (int, float)):
        raise ConfigValidationError(
            f"kernel {config.name!r} requires numeric parameter {name!r}"
        )
    return float(value)


def validate_unary_same_shape_and_dtype(
    config: KernelConfig,
    shared_memory: Mapping[str, SharedMemoryConfig],
) -> None:
    """Require unary kernels to preserve shape and dtype."""
    input_spec = shared_memory[config.input]
    output_spec = shared_memory[config.output]
    if input_spec.shape != output_spec.shape:
        raise ConfigValidationError(
            f"kernel {config.name!r} requires matching input/output shapes"
        )
    if input_spec.dtype != output_spec.dtype:
        raise ConfigValidationError(
            f"kernel {config.name!r} requires matching input/output dtypes"
        )


def validate_same_dtype(
    config: KernelConfig,
    shared_memory: Mapping[str, SharedMemoryConfig],
    *,
    names: tuple[str, ...],
    description: str,
) -> None:
    """Require a group of streams to share the same dtype."""
    dtypes = {shared_memory[name].dtype for name in names}
    if len(dtypes) != 1:
        raise ConfigValidationError(
            f"kernel {config.name!r} requires matching dtypes for {description}"
        )


def validate_binary_same_shape_and_dtype(
    config: KernelConfig,
    shared_memory: Mapping[str, SharedMemoryConfig],
) -> None:
    """Require input, one auxiliary stream, and output to share shape/dtype."""
    input_spec = shared_memory[config.input]
    auxiliary_spec = shared_memory[config.auxiliary_names[0]]
    output_spec = shared_memory[config.output]
    if (
        input_spec.shape != auxiliary_spec.shape
        or input_spec.shape != output_spec.shape
    ):
        raise ConfigValidationError(
            f"kernel {config.name!r} requires matching shapes for input, auxiliary, and output"
        )
    validate_same_dtype(
        config,
        shared_memory,
        names=(config.input, config.auxiliary_names[0], config.output),
        description="binary-operation streams",
    )


if njit is not None:

    @njit(cache=True)
    def scale_array(source, destination, factor):
        """Scale a flat array into the destination buffer."""
        for index in range(source.size):
            destination.flat[index] = source.flat[index] * factor

    @njit(cache=True)
    def add_constant_array(source, destination, constant):
        """Add a scalar constant into the destination buffer."""
        for index in range(source.size):
            destination.flat[index] = source.flat[index] + constant

    @njit(cache=True)
    def affine_transform_array(matrix, vector, offset, destination):
        """Apply y = A x + b for one dense matrix and vector."""
        for row in range(matrix.shape[0]):
            total = offset[row]
            for column in range(matrix.shape[1]):
                total += matrix[row, column] * vector[column]
            destination[row] = total

else:

    def scale_array(source, destination, factor):
        """Scale a flat array into the destination buffer."""
        np.multiply(source, factor, out=destination, casting="unsafe")

    def add_constant_array(source, destination, constant):
        """Add a scalar constant into the destination buffer."""
        np.add(source, constant, out=destination, casting="unsafe")

    def affine_transform_array(matrix, vector, offset, destination):
        """Apply y = A x + b for one dense matrix and vector."""
        destination[...] = matrix @ vector + offset


if njit is not None:

    @njit(cache=True)
    def scale_offset_array(source, offset, destination, gain):
        """Apply destination = gain * source - offset."""
        for index in range(source.size):
            destination.flat[index] = (
                gain * source.flat[index] - offset.flat[index]
            )

    @njit(cache=True)
    def flatten_array(source, destination):
        """Flatten an array into a contiguous output vector."""
        for index in range(source.size):
            destination[index] = source.flat[index]

    @njit(cache=True)
    def leaky_integrator_step(input_vector, state, destination, leak, gain):
        """Advance one leaky-integrator state update."""
        for index in range(input_vector.size):
            value = leak * state[index] + gain * input_vector[index]
            state[index] = value
            destination[index] = value

    @njit(cache=True)
    def centroid_tiles(image, destination, tile_size):
        """Compute local centroids for contiguous Shack-Hartmann tiles."""
        tiles_y = image.shape[0] // tile_size
        tiles_x = image.shape[1] // tile_size
        center = 0.5 * (tile_size - 1)
        for tile_y in range(tiles_y):
            row_start = tile_y * tile_size
            for tile_x in range(tiles_x):
                col_start = tile_x * tile_size
                total = 0.0
                y_weighted = 0.0
                x_weighted = 0.0
                for local_y in range(tile_size):
                    for local_x in range(tile_size):
                        value = image[row_start + local_y, col_start + local_x]
                        total += value
                        y_weighted += value * local_y
                        x_weighted += value * local_x
                if total <= 0.0:
                    destination[tile_y, tile_x, 0] = 0.0
                    destination[tile_y, tile_x, 1] = 0.0
                else:
                    destination[tile_y, tile_x, 0] = (
                        y_weighted / total
                    ) - center
                    destination[tile_y, tile_x, 1] = (
                        x_weighted / total
                    ) - center

else:

    def scale_offset_array(source, offset, destination, gain):
        """Apply destination = gain * source - offset."""
        destination[...] = gain * source - offset

    def flatten_array(source, destination):
        """Flatten an array into a contiguous output vector."""
        destination[...] = np.ravel(source)

    def leaky_integrator_step(input_vector, state, destination, leak, gain):
        """Advance one leaky-integrator state update."""
        destination[...] = leak * state + gain * input_vector
        state[...] = destination

    def centroid_tiles(image, destination, tile_size):
        """Compute local centroids for contiguous Shack-Hartmann tiles."""
        tiles_y = image.shape[0] // tile_size
        tiles_x = image.shape[1] // tile_size
        center = 0.5 * (tile_size - 1)
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
                    destination[tile_y, tile_x, 0] = 0.0
                    destination[tile_y, tile_x, 1] = 0.0
                    continue
                y_coords, x_coords = np.indices(patch.shape, dtype=np.float32)
                destination[tile_y, tile_x, 0] = (
                    np.sum(y_coords * patch) / total - center
                )
                destination[tile_y, tile_x, 1] = (
                    np.sum(x_coords * patch) / total - center
                )
