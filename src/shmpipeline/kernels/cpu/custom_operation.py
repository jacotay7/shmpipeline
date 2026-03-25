"""CPU custom operation kernel backed by a restricted Python expression."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from shmpipeline.config import KernelConfig, SharedMemoryConfig
from shmpipeline.errors import ConfigValidationError
from shmpipeline.kernel import KernelContext
from shmpipeline.kernels.cpu._expression import compile_custom_operation
from shmpipeline.kernels.cpu.base import CpuKernel


class CustomOperationCpuKernel(CpuKernel):
    """Fuse a restricted arithmetic expression into one CPU kernel."""

    kind = "cpu.custom_operation"
    auxiliary_arity = None

    @classmethod
    def validate_config(
        cls,
        config: KernelConfig,
        shared_memory: Mapping[str, SharedMemoryConfig],
    ) -> None:
        super().validate_config(config, shared_memory)
        if config.operation is None:
            raise ConfigValidationError(
                f"kernel {config.name!r} of kind {cls.kind!r} requires an 'operation' field"
            )
        if "input" in config.auxiliary_by_alias:
            raise ConfigValidationError(
                f"kernel {config.name!r} cannot bind auxiliary alias 'input'"
            )
        input_spec = shared_memory[config.input]
        output_spec = shared_memory[config.output]
        compile_custom_operation(
            expression=config.operation,
            input_shape=input_spec.shape,
            input_dtype=input_spec.dtype,
            auxiliary_specs={
                binding.alias: (
                    shared_memory[binding.name].shape,
                    shared_memory[binding.name].dtype,
                )
                for binding in config.auxiliary
            },
            output_shape=output_spec.shape,
            output_dtype=output_spec.dtype,
            kernel_name=config.name,
        )

    def __init__(self, context: KernelContext) -> None:
        super().__init__(context)
        input_spec = context.trigger_input_spec
        output_spec = context.output_spec
        self.plan = compile_custom_operation(
            expression=context.config.operation or "",
            input_shape=input_spec.shape,
            input_dtype=input_spec.dtype,
            auxiliary_specs={
                binding.alias: (
                    context.shared_memory[binding.name].shape,
                    context.shared_memory[binding.name].dtype,
                )
                for binding in context.config.auxiliary
            },
            output_shape=output_spec.shape,
            output_dtype=output_spec.dtype,
            kernel_name=context.config.name,
        )
        self.temporaries = self.plan.allocate_temporaries()

    def compute_into(
        self,
        trigger_input: Any,
        output: Any,
        auxiliary_inputs: Mapping[str, Any],
    ) -> None:
        self.plan.evaluate(
            trigger_input=trigger_input,
            auxiliary_inputs=auxiliary_inputs,
            output=np.asarray(output),
            temporaries=self.temporaries,
        )