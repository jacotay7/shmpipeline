"""GPU scale-and-offset kernel."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from shmpipeline.config import KernelConfig, SharedMemoryConfig
from shmpipeline.errors import ConfigValidationError
from shmpipeline.kernels.gpu._common import require_numeric_parameter
from shmpipeline.kernels.gpu._common import validate_same_dtype
from shmpipeline.kernels.gpu.base import GpuKernel, as_gpu_tensor


class ScaleOffsetGpuKernel(GpuKernel):
    """Apply output = gain * input - offset elementwise."""

    kind = "gpu.scale_offset"
    auxiliary_arity = 1

    @classmethod
    def validate_config(
        cls,
        config: KernelConfig,
        shared_memory: Mapping[str, SharedMemoryConfig],
    ) -> None:
        super().validate_config(config, shared_memory)
        require_numeric_parameter(config, name="gain")
        input_spec = shared_memory[config.input]
        offset_spec = shared_memory[config.auxiliary_names[0]]
        output_spec = shared_memory[config.output]
        if input_spec.shape != offset_spec.shape or input_spec.shape != output_spec.shape:
            raise ConfigValidationError(
                f"kernel {config.name!r} requires matching shapes for input, offset, and output"
            )
        validate_same_dtype(
            config,
            shared_memory,
            names=(config.input, config.auxiliary_names[0], config.output),
            description="scale-offset streams",
        )

    def __init__(self, context) -> None:
        super().__init__(context)
        self.gain = require_numeric_parameter(context.config, name="gain")

    def compute_into(
        self,
        trigger_input: Any,
        output: Any,
        auxiliary_inputs: Mapping[str, Any],
    ) -> None:
        alias = self.context.config.auxiliary_aliases[0]
        offset = as_gpu_tensor(auxiliary_inputs[alias], device=self.device)
        torch.mul(as_gpu_tensor(trigger_input, device=self.device), self.gain, out=output)
        torch.sub(output, offset, out=output)
        torch.cuda.synchronize(output.device)