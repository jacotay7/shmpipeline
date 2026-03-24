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
    input_spec = shared_memory[config.inputs[0]]
    output_spec = shared_memory[config.outputs[0]]
    if input_spec.shape != output_spec.shape:
        raise ConfigValidationError(
            f"kernel {config.name!r} requires matching input/output shapes"
        )
    if input_spec.dtype != output_spec.dtype:
        raise ConfigValidationError(
            f"kernel {config.name!r} requires matching input/output dtypes"
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