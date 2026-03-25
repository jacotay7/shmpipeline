"""GPU flatten kernel."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from shmpipeline.config import KernelConfig, SharedMemoryConfig
from shmpipeline.errors import ConfigValidationError
from shmpipeline.kernels.gpu._common import validate_same_dtype
from shmpipeline.kernels.gpu.base import GpuKernel, as_gpu_tensor


class FlattenGpuKernel(GpuKernel):
    """Flatten any GPU array into a contiguous 1D vector."""

    kind = "gpu.flatten"

    @classmethod
    def validate_config(
        cls,
        config: KernelConfig,
        shared_memory: Mapping[str, SharedMemoryConfig],
    ) -> None:
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
        del auxiliary_inputs
        output.copy_(torch.reshape(as_gpu_tensor(trigger_input, device=self.device), (-1,)))
        torch.cuda.synchronize(output.device)