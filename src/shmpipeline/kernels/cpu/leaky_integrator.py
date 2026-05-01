"""CPU leaky-integrator control kernel."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from shmpipeline.config import KernelConfig, SharedMemoryConfig
from shmpipeline.errors import ConfigValidationError
from shmpipeline.kernels.cpu._common import (
    leaky_integrator_step,
    require_numeric_parameter,
    validate_unary_same_shape_and_dtype,
)
from shmpipeline.kernels.cpu.base import CpuKernel


class LeakyIntegratorCpuKernel(CpuKernel):
    """Apply the control law `u_k = leak * u_{k-1} + gain * e_k`."""

    kind = "cpu.leaky_integrator"
    auxiliary_arity = None

    _DYNAMIC_PARAMETER_ALIASES = frozenset(
        {"gain", "leak", "override_enabled"}
    )

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
        aliases = set(config.auxiliary_aliases)
        unknown_aliases = aliases - cls._DYNAMIC_PARAMETER_ALIASES
        if unknown_aliases:
            unknown_list = ", ".join(
                sorted(repr(alias) for alias in unknown_aliases)
            )
            raise ConfigValidationError(
                f"kernel {config.name!r} supports only auxiliary aliases 'gain', 'leak', and 'override_enabled', got {unknown_list}"
            )
        for binding in config.auxiliary:
            spec = shared_memory[binding.name]
            if int(np.prod(spec.shape)) != 1:
                raise ConfigValidationError(
                    f"kernel {config.name!r} requires auxiliary {binding.name!r} to contain exactly one numeric value"
                )
            if not np.issubdtype(spec.dtype, np.number):
                raise ConfigValidationError(
                    f"kernel {config.name!r} requires auxiliary {binding.name!r} to use a numeric dtype"
                )
        if "leak" not in aliases:
            require_numeric_parameter(config, name="leak")
        if "gain" not in aliases:
            require_numeric_parameter(config, name="gain")

    def __init__(self, context) -> None:
        """Store the leak, gain, and persistent controller state."""
        super().__init__(context)
        self.leak = float(context.config.parameters.get("leak", 0.0))
        self.gain = float(context.config.parameters.get("gain", 0.0))
        self._override_enabled_alias = (
            "override_enabled"
            if "override_enabled" in context.config.auxiliary_by_alias
            else None
        )
        self._leak_alias = (
            "leak" if "leak" in context.config.auxiliary_by_alias else None
        )
        self._gain_alias = (
            "gain" if "gain" in context.config.auxiliary_by_alias else None
        )
        self.state = np.zeros(
            self.context.output_spec.shape,
            dtype=self.context.output_spec.dtype,
        )

    def _resolve_scalar(
        self,
        auxiliary_inputs: Mapping[str, Any],
        alias: str | None,
        fallback: float,
    ) -> float:
        if alias is None:
            return fallback
        value = np.asarray(auxiliary_inputs[alias], dtype=np.float64)
        return float(value.reshape(-1)[0])

    def compute_into(
        self,
        trigger_input: Any,
        output: Any,
        auxiliary_inputs: Mapping[str, Any],
    ) -> None:
        """Advance the controller one step into the reusable output vector."""
        use_auxiliary = bool(self._override_enabled_alias is None)
        if self._override_enabled_alias is not None:
            use_auxiliary = bool(
                self._resolve_scalar(
                    auxiliary_inputs,
                    self._override_enabled_alias,
                    0.0,
                )
            )
        leak = self.leak
        gain = self.gain
        if use_auxiliary:
            leak = self._resolve_scalar(
                auxiliary_inputs, self._leak_alias, leak
            )
            gain = self._resolve_scalar(
                auxiliary_inputs, self._gain_alias, gain
            )
        leaky_integrator_step(
            np.asarray(trigger_input),
            self.state,
            output,
            leak,
            gain,
        )
