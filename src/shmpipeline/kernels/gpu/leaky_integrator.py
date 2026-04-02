"""GPU leaky-integrator control kernel."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from shmpipeline.config import KernelConfig, SharedMemoryConfig
from shmpipeline.errors import ConfigValidationError
from shmpipeline.kernels.gpu._common import (
    require_numeric_parameter,
    validate_unary_same_shape_and_dtype,
)
from shmpipeline.kernels.gpu.base import GpuKernel, as_gpu_tensor


class LeakyIntegratorGpuKernel(GpuKernel):
    """Apply the control law u_k = leak * u_{k-1} + gain * e_k."""

    kind = "gpu.leaky_integrator"

    @classmethod
    def validate_config(
        cls,
        config: KernelConfig,
        shared_memory: Mapping[str, SharedMemoryConfig],
    ) -> None:
        super().validate_config(config, shared_memory)
        validate_unary_same_shape_and_dtype(config, shared_memory)
        input_spec = shared_memory[config.input]
        output_spec = shared_memory[config.output]
        if len(input_spec.shape) != 1 or len(output_spec.shape) != 1:
            raise ConfigValidationError(
                f"kernel {config.name!r} requires 1D input and output vectors"
            )
        require_numeric_parameter(config, name="leak")
        require_numeric_parameter(config, name="gain")

    def __init__(self, context) -> None:
        super().__init__(context)
        self.leak = require_numeric_parameter(context.config, name="leak")
        self.gain = require_numeric_parameter(context.config, name="gain")
        self.state = torch.zeros_like(self.output_buffer)

    def compute_into(
        self,
        trigger_input: Any,
        output: Any,
        auxiliary_inputs: Mapping[str, Any],
    ) -> None:
        del auxiliary_inputs
        torch.mul(self.state, self.leak, out=output)
        torch.add(
            output,
            as_gpu_tensor(trigger_input, device=self.device),
            alpha=self.gain,
            out=output,
        )
        self.state.copy_(output)
        torch.cuda.synchronize(output.device)
