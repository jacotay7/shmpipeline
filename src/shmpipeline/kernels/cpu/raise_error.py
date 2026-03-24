"""CPU error kernel used for supervision tests."""

from __future__ import annotations

from typing import Any, Mapping

from shmpipeline.config import KernelConfig, SharedMemoryConfig
from shmpipeline.errors import ConfigValidationError
from shmpipeline.kernels.cpu.base import CpuKernel


class RaiseErrorCpuKernel(CpuKernel):
    """Raise a configured error to exercise worker supervision paths."""

    kind = "cpu.raise_error"

    @classmethod
    def validate_config(
        cls,
        config: KernelConfig,
        shared_memory: Mapping[str, SharedMemoryConfig],
    ) -> None:
        """Require a non-empty error message."""
        super().validate_config(config, shared_memory)
        message = config.parameters.get("message")
        if not isinstance(message, str) or not message.strip():
            raise ConfigValidationError(
                f"kernel {config.name!r} requires a non-empty 'message' "
                "parameter"
            )

    def compute_into(
        self,
        trigger_input: Any,
        output: Any,
        auxiliary_inputs: Mapping[str, Any],
    ) -> None:
        """Always raise the configured failure."""
        del trigger_input, output, auxiliary_inputs
        raise RuntimeError(self.context.config.parameters["message"])