"""GPU error kernel used for supervision tests."""

from __future__ import annotations

from typing import Any, Mapping

from shmpipeline.config import KernelConfig, SharedMemoryConfig
from shmpipeline.errors import ConfigValidationError
from shmpipeline.kernels.gpu.base import GpuKernel


class RaiseErrorGpuKernel(GpuKernel):
    """Raise a configured error to exercise worker supervision paths."""

    kind = "gpu.raise_error"

    @classmethod
    def validate_config(
        cls,
        config: KernelConfig,
        shared_memory: Mapping[str, SharedMemoryConfig],
    ) -> None:
        super().validate_config(config, shared_memory)
        message = config.parameters.get("message")
        if not isinstance(message, str) or not message.strip():
            raise ConfigValidationError(
                f"kernel {config.name!r} requires a non-empty 'message' parameter"
            )

    def compute_into(
        self,
        trigger_input: Any,
        output: Any,
        auxiliary_inputs: Mapping[str, Any],
    ) -> None:
        del trigger_input, output, auxiliary_inputs
        raise RuntimeError(self.context.config.parameters["message"])