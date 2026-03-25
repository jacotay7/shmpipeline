"""Kernel abstractions used by worker processes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from shmpipeline.config import KernelConfig, SharedMemoryConfig
from shmpipeline.errors import ConfigValidationError


@dataclass(frozen=True)
class KernelContext:
    """Static information available to a kernel instance."""

    config: KernelConfig
    shared_memory: Mapping[str, SharedMemoryConfig]

    @property
    def input_specs(self) -> tuple[SharedMemoryConfig, ...]:
        """Return trigger plus auxiliary stream specifications in config order."""
        return tuple(self.shared_memory[name] for name in self.config.all_inputs)

    @property
    def trigger_input_spec(self) -> SharedMemoryConfig:
        """Return the primary input stream specification."""
        return self.shared_memory[self.config.input]

    @property
    def auxiliary_specs(self) -> tuple[SharedMemoryConfig, ...]:
        """Return auxiliary stream specifications in config order."""
        return tuple(self.shared_memory[name] for name in self.config.auxiliary_names)

    @property
    def output_spec(self) -> SharedMemoryConfig:
        """Return the primary output stream specification."""
        return self.shared_memory[self.config.output]


class Kernel(ABC):
    """Base class for compute kernels executed by the runtime."""

    kind = "kernel.base"
    storage = "cpu"
    auxiliary_arity: int | None = 0

    def __init__(self, context: KernelContext) -> None:
        """Store validated kernel context and normalized parameters."""
        self.context = context
        self.output_buffer = np.empty(
            self.context.output_spec.shape,
            dtype=self.context.output_spec.dtype,
        )

    @classmethod
    def validate_config(
        cls,
        config: KernelConfig,
        shared_memory: Mapping[str, SharedMemoryConfig],
    ) -> None:
        """Validate arity and storage constraints before build."""
        if cls.auxiliary_arity is not None and len(config.auxiliary) != cls.auxiliary_arity:
            raise ConfigValidationError(
                f"kernel kind {cls.kind!r} expects {cls.auxiliary_arity} auxiliary streams"
            )
        for name in (*config.all_inputs, config.output):
            if shared_memory[name].storage != cls.storage:
                raise ConfigValidationError(
                    f"kernel {config.name!r} of kind {cls.kind!r} requires "
                    f"{cls.storage} shared memory for {name!r}"
                )

    @abstractmethod
    def compute_into(
        self,
        trigger_input: Any,
        output: Any,
        auxiliary_inputs: Mapping[str, Any],
    ) -> None:
        """Compute into the provided reusable output buffer."""