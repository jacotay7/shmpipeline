"""CPU affine transformation kernel."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from shmpipeline.config import KernelConfig, SharedMemoryConfig
from shmpipeline.errors import ConfigValidationError
from shmpipeline.kernels.cpu._common import affine_transform_array
from shmpipeline.kernels.cpu.base import CpuKernel


class AffineTransformCpuKernel(CpuKernel):
    """Apply an affine transform `y = A x + b` to a vector input."""

    kind = "cpu.affine_transform"
    input_arity = 3

    @classmethod
    def validate_config(
        cls,
        config: KernelConfig,
        shared_memory: Mapping[str, SharedMemoryConfig],
    ) -> None:
        """Validate vector, matrix, offset, and output compatibility."""
        super().validate_config(config, shared_memory)
        vector_spec = shared_memory[config.inputs[0]]
        matrix_spec = shared_memory[config.inputs[1]]
        offset_spec = shared_memory[config.inputs[2]]
        output_spec = shared_memory[config.outputs[0]]

        if len(vector_spec.shape) != 1:
            raise ConfigValidationError(
                f"kernel {config.name!r} requires a 1D input vector"
            )
        if len(matrix_spec.shape) != 2:
            raise ConfigValidationError(
                f"kernel {config.name!r} requires a 2D transform matrix"
            )
        if len(offset_spec.shape) != 1:
            raise ConfigValidationError(
                f"kernel {config.name!r} requires a 1D offset vector"
            )
        if len(output_spec.shape) != 1:
            raise ConfigValidationError(
                f"kernel {config.name!r} requires a 1D output vector"
            )
        if matrix_spec.shape[1] != vector_spec.shape[0]:
            raise ConfigValidationError(
                f"kernel {config.name!r} requires matrix columns to match "
                "input vector length"
            )
        if matrix_spec.shape[0] != offset_spec.shape[0]:
            raise ConfigValidationError(
                f"kernel {config.name!r} requires matrix rows to match "
                "offset vector length"
            )
        if output_spec.shape[0] != matrix_spec.shape[0]:
            raise ConfigValidationError(
                f"kernel {config.name!r} requires output vector length to "
                "match matrix rows"
            )
        dtypes = {
            vector_spec.dtype,
            matrix_spec.dtype,
            offset_spec.dtype,
            output_spec.dtype,
        }
        if len(dtypes) != 1:
            raise ConfigValidationError(
                f"kernel {config.name!r} requires matching dtypes across all "
                "affine inputs and outputs"
            )

    def compute(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        """Compute the affine transform into a fresh output vector."""
        vector_name, matrix_name, offset_name = self.context.config.inputs
        output_name = self.context.config.outputs[0]
        vector = np.asarray(inputs[vector_name])
        matrix = np.asarray(inputs[matrix_name])
        offset = np.asarray(inputs[offset_name])
        destination = np.empty(offset.shape, dtype=vector.dtype)
        affine_transform_array(matrix, vector, offset, destination)
        return {output_name: destination}