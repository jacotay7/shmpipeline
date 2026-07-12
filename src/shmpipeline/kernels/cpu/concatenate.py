"""CPU synchronized concatenation kernel."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from shmpipeline.config import KernelConfig, SharedMemoryConfig
from shmpipeline.errors import ConfigValidationError
from shmpipeline.kernels.cpu.base import CpuKernel


def validate_concatenate_config(
    config: KernelConfig,
    shared_memory: Mapping[str, SharedMemoryConfig],
) -> None:
    """Validate backend-independent concatenate geometry."""
    if len(config.trigger_inputs) < 2:
        raise ConfigValidationError(
            f"kernel {config.name!r} requires at least two trigger inputs"
        )
    if config.trigger_policy != "all_new":
        raise ConfigValidationError(
            f"kernel {config.name!r} requires trigger_policy 'all_new'"
        )
    axis = config.parameters.get("axis", 0)
    if not isinstance(axis, int):
        raise ConfigValidationError(
            f"kernel {config.name!r} requires integer parameter 'axis'"
        )
    specs = [shared_memory[name] for name in config.trigger_inputs]
    output = shared_memory[config.output]
    ndim = len(specs[0].shape)
    normalized_axis = axis + ndim if axis < 0 else axis
    if normalized_axis < 0 or normalized_axis >= ndim:
        raise ConfigValidationError(
            f"kernel {config.name!r} axis {axis} is out of bounds"
        )
    if any(len(spec.shape) != ndim for spec in specs):
        raise ConfigValidationError(
            f"kernel {config.name!r} requires equal input ranks"
        )
    if any(spec.dtype != specs[0].dtype for spec in specs):
        raise ConfigValidationError(
            f"kernel {config.name!r} requires matching input dtypes"
        )
    expected = list(specs[0].shape)
    expected[normalized_axis] = sum(
        spec.shape[normalized_axis] for spec in specs
    )
    for spec in specs[1:]:
        for index in range(ndim):
            same_dimension = spec.shape[index] == expected[index]
            if index != normalized_axis and not same_dimension:
                raise ConfigValidationError(
                    f"kernel {config.name!r} requires matching "
                    "non-concatenated dimensions"
                )
    if output.shape != tuple(expected) or output.dtype != specs[0].dtype:
        raise ConfigValidationError(
            f"kernel {config.name!r} requires output shape "
            f"{tuple(expected)} and dtype {specs[0].dtype}"
        )


class ConcatenateCpuKernel(CpuKernel):
    """Concatenate multiple newly published trigger inputs."""

    kind = "cpu.concatenate"
    input_arity = None

    @classmethod
    def validate_config(
        cls,
        config: KernelConfig,
        shared_memory: Mapping[str, SharedMemoryConfig],
    ) -> None:
        super().validate_config(config, shared_memory)
        validate_concatenate_config(config, shared_memory)

    def __init__(self, context) -> None:
        super().__init__(context)
        self.axis = int(context.config.parameters.get("axis", 0))

    def compute_into(
        self,
        trigger_input: Any,
        output: Any,
        auxiliary_inputs: Mapping[str, Any],
    ) -> None:
        del auxiliary_inputs
        np.concatenate(
            tuple(np.asarray(value) for value in trigger_input),
            axis=self.axis,
            out=output,
        )
