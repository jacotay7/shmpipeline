"""GPU copy kernel."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from shmpipeline.config import KernelConfig, SharedMemoryConfig
from shmpipeline.kernels.gpu._common import validate_unary_same_shape_and_dtype
from shmpipeline.kernels.gpu.base import GpuKernel, as_gpu_tensor


class CopyGpuKernel(GpuKernel):
    """Copy one GPU shared-memory payload into another."""

    kind = "gpu.copy"

    @classmethod
    def validate_config(
        cls,
        config: KernelConfig,
        shared_memory: Mapping[str, SharedMemoryConfig],
    ) -> None:
        super().validate_config(config, shared_memory)
        validate_unary_same_shape_and_dtype(config, shared_memory)

    def compute_into(
        self,
        trigger_input: Any,
        output: Any,
        auxiliary_inputs: Mapping[str, Any],
    ) -> None:
        del auxiliary_inputs
        output.copy_(as_gpu_tensor(trigger_input, device=self.device))
        if isinstance(output, torch.Tensor):
            torch.cuda.synchronize(output.device)
