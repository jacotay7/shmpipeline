"""GPU elementwise addition kernel."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from shmpipeline.config import KernelConfig, SharedMemoryConfig
from shmpipeline.kernels.gpu._common import validate_binary_same_shape_and_dtype
from shmpipeline.kernels.gpu.base import GpuKernel, as_gpu_tensor


class ElementwiseAddGpuKernel(GpuKernel):
    """Compute output = input + auxiliary elementwise."""

    kind = "gpu.elementwise_add"
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
        alias = self.context.config.auxiliary_aliases[0]
        rhs = as_gpu_tensor(auxiliary_inputs[alias], device=self.device)
        torch.add(as_gpu_tensor(trigger_input, device=self.device), rhs, out=output)
        torch.cuda.synchronize(output.device)