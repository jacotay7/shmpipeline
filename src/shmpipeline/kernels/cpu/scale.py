"""CPU scale kernel."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from shmpipeline.config import KernelConfig, SharedMemoryConfig
from shmpipeline.kernel import KernelContext
from shmpipeline.kernels.cpu._common import (
    require_numeric_parameter,
    scale_array,
    validate_unary_same_shape_and_dtype,
)
from shmpipeline.kernels.cpu.base import CpuKernel


class ScaleCpuKernel(CpuKernel):
    """Multiply the input payload by a constant factor."""

    kind = "cpu.scale"

    @classmethod
    def validate_config(
        cls,
        config: KernelConfig,
        shared_memory: Mapping[str, SharedMemoryConfig],
    ) -> None:
        """Require a numeric factor and matching stream metadata."""
        super().validate_config(config, shared_memory)
        validate_unary_same_shape_and_dtype(config, shared_memory)
        require_numeric_parameter(config, name="factor")

    def __init__(self, context: KernelContext) -> None:
        """Store the scale factor after validation."""
        super().__init__(context)
        self.factor = require_numeric_parameter(context.config, name="factor")

    def compute_into(
        self,
        trigger_input: Any,
        output: Any,
        auxiliary_inputs: Mapping[str, Any],
    ) -> None:
        """Scale the incoming array into the reusable output buffer."""
        del auxiliary_inputs
        scale_array(np.asarray(trigger_input), output, self.factor)
