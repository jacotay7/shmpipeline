"""CPU copy kernel."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from shmpipeline.config import KernelConfig, SharedMemoryConfig
from shmpipeline.kernels.cpu._common import validate_unary_same_shape_and_dtype
from shmpipeline.kernels.cpu.base import CpuKernel


class CopyCpuKernel(CpuKernel):
    """Copy one CPU shared-memory payload into another."""

    kind = "cpu.copy"

    @classmethod
    def validate_config(
        cls,
        config: KernelConfig,
        shared_memory: Mapping[str, SharedMemoryConfig],
    ) -> None:
        """Require matching input and output stream metadata."""
        super().validate_config(config, shared_memory)
        validate_unary_same_shape_and_dtype(config, shared_memory)

    def compute(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return a copy of the input payload."""
        source = inputs[self.context.config.inputs[0]]
        return {self.context.config.outputs[0]: np.array(source, copy=True)}