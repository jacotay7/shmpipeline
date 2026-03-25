"""GPU scale kernel."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from shmpipeline.config import KernelConfig, SharedMemoryConfig
from shmpipeline.kernel import KernelContext
from shmpipeline.kernels.gpu._common import (
    require_numeric_parameter,
    validate_unary_same_shape_and_dtype,
)
from shmpipeline.kernels.gpu.base import GpuKernel, as_gpu_tensor


class ScaleGpuKernel(GpuKernel):
    """Multiply the input payload by a constant factor."""

    kind = "gpu.scale"

    @classmethod
    def validate_config(
        cls,
        config: KernelConfig,
        shared_memory: Mapping[str, SharedMemoryConfig],
    ) -> None:
        super().validate_config(config, shared_memory)
        validate_unary_same_shape_and_dtype(config, shared_memory)
        require_numeric_parameter(config, name="factor")

    def __init__(self, context: KernelContext) -> None:
        super().__init__(context)
        self.factor = require_numeric_parameter(context.config, name="factor")

    def compute_into(
        self,
        trigger_input: Any,
        output: Any,
        auxiliary_inputs: Mapping[str, Any],
    ) -> None:
        del auxiliary_inputs
        torch.mul(as_gpu_tensor(trigger_input, device=self.device), self.factor, out=output)
        torch.cuda.synchronize(output.device)