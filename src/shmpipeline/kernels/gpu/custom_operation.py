"""GPU custom operation kernel backed by a restricted Python expression."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from shmpipeline.config import KernelConfig, SharedMemoryConfig
from shmpipeline.errors import ConfigValidationError
from shmpipeline.kernel import KernelContext
from shmpipeline.kernels.cpu._expression import compile_custom_operation
from shmpipeline.kernels.gpu.base import (
    GpuKernel,
    as_gpu_tensor,
    torch_dtype_from_numpy,
)


class CustomOperationGpuKernel(GpuKernel):
    """Fuse a restricted arithmetic expression into one GPU kernel."""

    kind = "gpu.custom_operation"
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
        self.temporaries = tuple(
            torch.empty(
                shape, dtype=torch_dtype_from_numpy(dtype), device=self.device
            )
            for shape, dtype in zip(
                self.plan.temp_shapes, self.plan.temp_dtypes
            )
        )

    def compute_into(
        self,
        trigger_input: Any,
        output: Any,
        auxiliary_inputs: Mapping[str, Any],
    ) -> None:
        values = {"input": as_gpu_tensor(trigger_input, device=self.device)}
        values.update(
            {
                name: as_gpu_tensor(value, device=self.device)
                for name, value in auxiliary_inputs.items()
            }
        )
        for instruction in self.plan.instructions:
            destination = self._resolve_destination(
                instruction.destination, output
            )
            operands = [
                self._resolve_operand(operand, values, output)
                for operand in instruction.operands
            ]
            if instruction.operation == "copy":
                destination.copy_(operands[0])
            elif instruction.operation == "neg":
                torch.neg(operands[0], out=destination)
            elif instruction.operation == "pos":
                destination.copy_(operands[0])
            elif instruction.operation == "add":
                torch.add(operands[0], operands[1], out=destination)
            elif instruction.operation == "sub":
                torch.sub(operands[0], operands[1], out=destination)
            elif instruction.operation == "mul":
                torch.mul(operands[0], operands[1], out=destination)
            elif instruction.operation == "div":
                torch.div(operands[0], operands[1], out=destination)
            elif instruction.operation == "matmul":
                torch.matmul(operands[0], operands[1], out=destination)
            elif instruction.operation == "abs":
                torch.abs(operands[0], out=destination)
            elif instruction.operation == "minimum":
                torch.minimum(operands[0], operands[1], out=destination)
            elif instruction.operation == "maximum":
                torch.maximum(operands[0], operands[1], out=destination)
            elif instruction.operation == "clip":
                torch.clamp(
                    operands[0],
                    min=operands[1],
                    max=operands[2],
                    out=destination,
                )
            else:
                raise RuntimeError(
                    f"unknown operation {instruction.operation!r}"
                )
        torch.cuda.synchronize(output.device)

    def _resolve_destination(
        self, operand, output: torch.Tensor
    ) -> torch.Tensor:
        if operand.kind == "output":
            return output
        if operand.kind == "temp":
            return self.temporaries[operand.value]
        raise RuntimeError(
            f"invalid destination operand kind: {operand.kind!r}"
        )

    def _resolve_operand(self, operand, values, output: torch.Tensor):
        if operand.kind == "value":
            return values[operand.value]
        if operand.kind == "temp":
            return self.temporaries[operand.value]
        if operand.kind == "output":
            return output
        if operand.kind == "const":
            return operand.value
        raise RuntimeError(f"invalid operand kind: {operand.kind!r}")
