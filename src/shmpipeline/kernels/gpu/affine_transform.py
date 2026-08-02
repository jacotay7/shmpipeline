"""GPU affine transformation kernel."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from shmpipeline.config import KernelConfig, SharedMemoryConfig
from shmpipeline.errors import ConfigValidationError
from shmpipeline.kernels.gpu.base import GpuKernel, as_gpu_tensor


class AffineTransformGpuKernel(GpuKernel):
    """Apply an affine transform y = A x + b to a vector input."""

    kind = "gpu.affine_transform"
    auxiliary_arity = 2

    def __init__(self, context) -> None:
        """Cache aliases and the optional fixed matrix storage transform."""
        super().__init__(context)
        self._matrix_alias = context.config.auxiliary_aliases[0]
        self._offset_alias = context.config.auxiliary_aliases[1]
        self._matrix_layout = str(
            context.config.parameters.get("matrix_layout", "source")
        )
        self._matrix_source: torch.Tensor | None = None
        self._matrix_storage: torch.Tensor | None = None
        self._matrix_view: torch.Tensor | None = None

    @classmethod
    def validate_config(
        cls,
        config: KernelConfig,
        shared_memory: Mapping[str, SharedMemoryConfig],
    ) -> None:
        super().validate_config(config, shared_memory)
        vector_spec = shared_memory[config.input]
        matrix_spec = shared_memory[config.auxiliary_names[0]]
        offset_spec = shared_memory[config.auxiliary_names[1]]
        output_spec = shared_memory[config.output]
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
                f"kernel {config.name!r} requires matrix columns to match input vector length"
            )
        if matrix_spec.shape[0] != offset_spec.shape[0]:
            raise ConfigValidationError(
                f"kernel {config.name!r} requires matrix rows to match offset vector length"
            )
        if output_spec.shape[0] != matrix_spec.shape[0]:
            raise ConfigValidationError(
                f"kernel {config.name!r} requires output vector length to match matrix rows"
            )
        dtypes = {
            vector_spec.dtype,
            matrix_spec.dtype,
            offset_spec.dtype,
            output_spec.dtype,
        }
        if len(dtypes) != 1:
            raise ConfigValidationError(
                f"kernel {config.name!r} requires matching dtypes across all affine inputs and outputs"
            )
        matrix_layout = config.parameters.get("matrix_layout", "source")
        if matrix_layout not in {"source", "column_major"}:
            raise ConfigValidationError(
                f"kernel {config.name!r} requires optional parameter "
                "'matrix_layout' to be 'source' or 'column_major'"
            )

    def _matrix_for_compute(self, matrix: torch.Tensor) -> torch.Tensor:
        if self._matrix_layout == "source":
            return matrix
        if matrix is not self._matrix_source:
            # PyTorch has no independent column-major tensor flag.  Keep a
            # contiguous transposed owner and expose its transpose as the
            # original logical (commands, slopes) matrix.
            self._matrix_storage = matrix.transpose(0, 1).contiguous()
            self._matrix_view = self._matrix_storage.transpose(0, 1)
            self._matrix_source = matrix
        assert self._matrix_view is not None
        return self._matrix_view

    def compute_into(
        self,
        trigger_input: Any,
        output: Any,
        auxiliary_inputs: Mapping[str, Any],
    ) -> None:
        vector = as_gpu_tensor(trigger_input, device=self.device)
        matrix = as_gpu_tensor(
            auxiliary_inputs[self._matrix_alias],
            device=self.device,
        )
        offset = as_gpu_tensor(
            auxiliary_inputs[self._offset_alias],
            device=self.device,
        )
        torch.matmul(self._matrix_for_compute(matrix), vector, out=output)
        torch.add(output, offset, out=output)
