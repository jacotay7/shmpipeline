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

    def compute_into(
        self,
        trigger_input: Any,
        output: Any,
        auxiliary_inputs: Mapping[str, Any],
    ) -> None:
        """Copy the input payload into the reusable output buffer."""
        del auxiliary_inputs
        np.copyto(output, np.asarray(trigger_input))