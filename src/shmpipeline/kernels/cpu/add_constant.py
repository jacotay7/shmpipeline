"""CPU add-constant kernel."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from shmpipeline.config import KernelConfig, SharedMemoryConfig
from shmpipeline.kernel import KernelContext
from shmpipeline.kernels.cpu._common import (
    add_constant_array,
    require_numeric_parameter,
    validate_unary_same_shape_and_dtype,
)
from shmpipeline.kernels.cpu.base import CpuKernel


class AddConstantCpuKernel(CpuKernel):
    """Add a scalar constant to the input payload."""

    kind = "cpu.add_constant"

    @classmethod
    def validate_config(
        cls,
        config: KernelConfig,
        shared_memory: Mapping[str, SharedMemoryConfig],
    ) -> None:
        """Require a numeric constant and matching stream metadata."""
        super().validate_config(config, shared_memory)
        validate_unary_same_shape_and_dtype(config, shared_memory)
        require_numeric_parameter(config, name="constant")

    def __init__(self, context: KernelContext) -> None:
        """Store the additive constant after validation."""
        super().__init__(context)
        self.constant = require_numeric_parameter(
            context.config,
            name="constant",
        )

    def compute_into(
        self,
        trigger_input: Any,
        output: Any,
        auxiliary_inputs: Mapping[str, Any],
    ) -> None:
        """Add the configured constant into the reusable output buffer."""
        del auxiliary_inputs
        add_constant_array(np.asarray(trigger_input), output, self.constant)
