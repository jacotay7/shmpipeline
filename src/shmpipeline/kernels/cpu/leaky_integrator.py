"""CPU leaky-integrator control kernel."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from shmpipeline.config import KernelConfig, SharedMemoryConfig
from shmpipeline.errors import ConfigValidationError
from shmpipeline.kernels.cpu._common import leaky_integrator_step
from shmpipeline.kernels.cpu._common import require_numeric_parameter
from shmpipeline.kernels.cpu._common import validate_unary_same_shape_and_dtype
from shmpipeline.kernels.cpu.base import CpuKernel


class LeakyIntegratorCpuKernel(CpuKernel):
    """Apply the control law `u_k = leak * u_{k-1} + gain * e_k`."""

    kind = "cpu.leaky_integrator"

    @classmethod
    def validate_config(
        cls,
        config: KernelConfig,
        shared_memory: Mapping[str, SharedMemoryConfig],
    ) -> None:
        """Require vector-compatible streams and numeric leak/gain."""
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
        """Store the leak, gain, and persistent controller state."""
        super().__init__(context)
        self.leak = require_numeric_parameter(context.config, name="leak")
        self.gain = require_numeric_parameter(context.config, name="gain")
        self.state = np.zeros(
            self.context.output_spec.shape,
            dtype=self.context.output_spec.dtype,
        )

    def compute_into(
        self,
        trigger_input: Any,
        output: Any,
        auxiliary_inputs: Mapping[str, Any],
    ) -> None:
        """Advance the controller one step into the reusable output vector."""
        del auxiliary_inputs
        leaky_integrator_step(
            np.asarray(trigger_input),
            self.state,
            output,
            self.leak,
            self.gain,
        )