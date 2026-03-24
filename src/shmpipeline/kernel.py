"""Kernel abstractions used by worker processes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

from shmpipeline.config import KernelConfig, SharedMemoryConfig
from shmpipeline.errors import ConfigValidationError


@dataclass(frozen=True)
class KernelContext:
    """Static information available to a kernel instance."""

    config: KernelConfig
    shared_memory: Mapping[str, SharedMemoryConfig]

    @property
    def input_specs(self) -> tuple[SharedMemoryConfig, ...]:
        """Return input stream specifications in config order."""
        return tuple(self.shared_memory[name] for name in self.config.inputs)

    @property
    def output_specs(self) -> tuple[SharedMemoryConfig, ...]:
        """Return output stream specifications in config order."""
        return tuple(self.shared_memory[name] for name in self.config.outputs)


class Kernel(ABC):
    """Base class for compute kernels executed by the runtime."""

    kind = "kernel.base"
    storage = "cpu"
    input_arity: int | None = 1
    output_arity: int | None = 1

    def __init__(self, context: KernelContext) -> None:
        """Store validated kernel context and normalized parameters."""
        self.context = context

    @classmethod
    def validate_config(
        cls,
        config: KernelConfig,
        shared_memory: Mapping[str, SharedMemoryConfig],
    ) -> None:
        """Validate arity and storage constraints before build."""
        if cls.input_arity is not None and len(config.inputs) != cls.input_arity:
            raise ConfigValidationError(
                f"kernel kind {cls.kind!r} expects {cls.input_arity} inputs"
            )
        if (
            cls.output_arity is not None
            and len(config.outputs) != cls.output_arity
        ):
            raise ConfigValidationError(
                f"kernel kind {cls.kind!r} expects {cls.output_arity} outputs"
            )
        for name in (*config.inputs, *config.outputs):
            if shared_memory[name].storage != cls.storage:
                raise ConfigValidationError(
                    f"kernel {config.name!r} of kind {cls.kind!r} requires "
                    f"{cls.storage} shared memory for {name!r}"
                )

    @abstractmethod
    def compute(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        """Compute output payloads from input payloads."""