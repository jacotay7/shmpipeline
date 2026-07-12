"""GPU synchronized concatenation kernel."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from shmpipeline.config import KernelConfig, SharedMemoryConfig
from shmpipeline.kernels.cpu.concatenate import validate_concatenate_config
from shmpipeline.kernels.gpu.base import GpuKernel, as_gpu_tensor


class ConcatenateGpuKernel(GpuKernel):
    """Concatenate multiple newly published CUDA trigger inputs."""

    kind = "gpu.concatenate"
    input_arity = None

    @classmethod
    def validate_config(
        cls,
        config: KernelConfig,
        shared_memory: Mapping[str, SharedMemoryConfig],
    ) -> None:
        super().validate_config(config, shared_memory)
        validate_concatenate_config(config, shared_memory)

    def __init__(self, context) -> None:
        super().__init__(context)
        self.axis = int(context.config.parameters.get("axis", 0))

    def compute_into(
        self,
        trigger_input: Any,
        output: Any,
        auxiliary_inputs: Mapping[str, Any],
    ) -> None:
        del auxiliary_inputs
        values = tuple(
            as_gpu_tensor(value, device=self.device) for value in trigger_input
        )
        torch.cat(values, dim=self.axis, out=output)
        torch.cuda.synchronize(output.device)
