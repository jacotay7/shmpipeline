"""Kernel abstractions used by worker processes."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any, Mapping

from shmpipeline.config import KernelConfig, SharedMemoryConfig
from shmpipeline.errors import ConfigValidationError


@dataclass(frozen=True)
class KernelContext:
    """Static information available to a kernel instance.

    The runtime constructs one context per worker so kernels can inspect their
    validated configuration and the shared-memory specifications for the
    streams they read and write.
    """

    config: KernelConfig
    shared_memory: Mapping[str, SharedMemoryConfig]

    @property
    def input_specs(self) -> tuple[SharedMemoryConfig, ...]:
        """Return trigger plus auxiliary stream specifications in config order."""
        return tuple(
            self.shared_memory[name] for name in self.config.all_inputs
        )

    @property
    def trigger_input_spec(self) -> SharedMemoryConfig:
        """Return the primary input stream specification."""
        return self.shared_memory[self.config.input]

    @property
    def auxiliary_specs(self) -> tuple[SharedMemoryConfig, ...]:
        """Return auxiliary stream specifications in config order."""
        return tuple(
            self.shared_memory[name] for name in self.config.auxiliary_names
        )

    @property
    def output_spec(self) -> SharedMemoryConfig:
        """Return the primary output stream specification."""
        return self.shared_memory[self.config.output]

    @property
    def output_specs(self) -> tuple[SharedMemoryConfig, ...]:
        """Return every output stream specification in declaration order."""
        return tuple(
            self.shared_memory[name] for name in self.config.all_outputs
        )


class Kernel(ABC):
    """Base class for compute kernels executed by the runtime.

    Custom kernels normally override :meth:`validate_config` when they need
    stage-specific parameter checks and implement :meth:`compute_into` to write
    results into the provided output buffer.
    """

    kind = "kernel.base"
    storage = "cpu"
    auxiliary_arity: int | None = 0
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
        if (
            cls.auxiliary_arity is not None
            and len(config.auxiliary) != cls.auxiliary_arity
        ):
            raise ConfigValidationError(
                f"kernel kind {cls.kind!r} expects {cls.auxiliary_arity} auxiliary streams"
            )
        if (
            cls.output_arity is not None
            and len(config.all_outputs) != cls.output_arity
        ):
            raise ConfigValidationError(
                f"kernel kind {cls.kind!r} expects {cls.output_arity} output "
                f"stream(s), got {len(config.all_outputs)}"
            )
        for name in (*config.all_inputs, *config.all_outputs):
            if shared_memory[name].storage != cls.storage:
                raise ConfigValidationError(
                    f"kernel {config.name!r} of kind {cls.kind!r} requires "
                    f"{cls.storage} shared memory for {name!r}"
                )

    def compute_into(
        self,
        trigger_input: Any,
        output: Any,
        auxiliary_inputs: Mapping[str, Any],
    ) -> None:
        """Compute into the single reusable output buffer.

        Single-output kernels override this method.  Multi-output kernels
        (``output_arity != 1``) override :meth:`compute_into_multiple` instead.
        """
        raise NotImplementedError(
            f"kernel kind {type(self).kind!r} does not implement compute_into"
        )

    def compute_into_multiple(
        self,
        trigger_input: Any,
        outputs: list[Any],
        auxiliary_inputs: Mapping[str, Any],
    ) -> None:
        """Compute into one or more reusable output buffers.

        ``outputs`` aligns positionally with
        :attr:`~shmpipeline.config.KernelConfig.all_outputs`.  The default
        implementation forwards the first (primary) output to
        :meth:`compute_into`, so existing single-output kernels need not change.
        """
        self.compute_into(trigger_input, outputs[0], auxiliary_inputs)
