"""CPU elementwise multiplication kernel."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from shmpipeline.config import KernelConfig, SharedMemoryConfig
from shmpipeline.kernels.cpu._common import validate_binary_same_shape_and_dtype
from shmpipeline.kernels.cpu.base import CpuKernel


class ElementwiseMultiplyCpuKernel(CpuKernel):
    """Compute output = input * auxiliary elementwise."""

    kind = "cpu.elementwise_multiply"
    auxiliary_arity = 1

    @classmethod
    def validate_config(
        cls,
        config: KernelConfig,
        shared_memory: Mapping[str, SharedMemoryConfig],
    ) -> None:
        super().validate_config(config, shared_memory)
        validate_binary_same_shape_and_dtype(config, shared_memory)

    def compute_into(
        self,
        trigger_input: Any,
        output: Any,
        auxiliary_inputs: Mapping[str, Any],
    ) -> None:
        rhs = np.asarray(auxiliary_inputs[self.context.config.auxiliary_aliases[0]])
        np.multiply(np.asarray(trigger_input), rhs, out=output)