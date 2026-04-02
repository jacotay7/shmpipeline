"""CPU flatten kernel."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from shmpipeline.config import KernelConfig, SharedMemoryConfig
from shmpipeline.errors import ConfigValidationError
from shmpipeline.kernels.cpu._common import flatten_array, validate_same_dtype
from shmpipeline.kernels.cpu.base import CpuKernel


class FlattenCpuKernel(CpuKernel):
    """Flatten any CPU array into a contiguous 1D vector."""

    kind = "cpu.flatten"

    @classmethod
    def validate_config(
        cls,
        config: KernelConfig,
        shared_memory: Mapping[str, SharedMemoryConfig],
    ) -> None:
        """Require the output vector to match the flattened input size."""
        super().validate_config(config, shared_memory)
        input_spec = shared_memory[config.input]
        output_spec = shared_memory[config.output]
        if len(output_spec.shape) != 1:
            raise ConfigValidationError(
                f"kernel {config.name!r} requires a 1D output vector"
            )
        expected_size = int(np.prod(input_spec.shape, dtype=np.int64))
        if output_spec.shape[0] != expected_size:
            raise ConfigValidationError(
                f"kernel {config.name!r} requires output length {expected_size}"
            )
        validate_same_dtype(
            config,
            shared_memory,
            names=(config.input, config.output),
            description="flatten input/output",
        )

    def compute_into(
        self,
        trigger_input: Any,
        output: Any,
        auxiliary_inputs: Mapping[str, Any],
    ) -> None:
        """Flatten the input array into the reusable output vector."""
        del auxiliary_inputs
        flatten_array(np.asarray(trigger_input), output)
