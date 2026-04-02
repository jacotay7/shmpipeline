"""Restricted expression compiler for fused CPU custom operations."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from shmpipeline.errors import ConfigValidationError


@dataclass(frozen=True)
class OperandRef:
    """Reference one runtime operand."""

    kind: str
    value: Any


@dataclass(frozen=True)
class Instruction:
    """One compiled expression step."""

    operation: str
    destination: OperandRef
    operands: tuple[OperandRef, ...]


@dataclass(frozen=True)
class CompiledExpression:
    """Executable plan for one custom operation expression."""

    expression: str
    instructions: tuple[Instruction, ...]
    temp_shapes: tuple[tuple[int, ...], ...]
    temp_dtypes: tuple[np.dtype, ...]
    result_shape: tuple[int, ...]
    result_dtype: np.dtype
    used_names: tuple[str, ...]

    def allocate_temporaries(self) -> tuple[np.ndarray, ...]:
        """Allocate reusable scratch buffers for intermediate results."""
        return tuple(
            np.empty(shape, dtype=dtype)
            for shape, dtype in zip(self.temp_shapes, self.temp_dtypes)
        )

    def evaluate(
        self,
        *,
        trigger_input: Any,
        auxiliary_inputs: Mapping[str, Any],
        output: np.ndarray,
        temporaries: tuple[np.ndarray, ...],
    ) -> None:
        """Execute the compiled plan into the provided output buffer."""
        values = {"input": np.asarray(trigger_input)}
        values.update(
            {
                name: np.asarray(value)
                for name, value in auxiliary_inputs.items()
            }
        )
        for instruction in self.instructions:
            destination = self._resolve_destination(
                instruction.destination,
                output=output,
                temporaries=temporaries,
            )
            operands = tuple(
                self._resolve_operand(
                    operand,
                    values=values,
                    output=output,
                    temporaries=temporaries,
                )
                for operand in instruction.operands
            )
            if instruction.operation == "copy":
                np.copyto(destination, operands[0], casting="unsafe")
                continue
            if instruction.operation == "neg":
                np.negative(operands[0], out=destination)
                continue
            if instruction.operation == "pos":
                np.copyto(destination, operands[0], casting="unsafe")
                continue
            if instruction.operation == "add":
                np.add(
                    operands[0], operands[1], out=destination, casting="unsafe"
                )
            elif instruction.operation == "sub":
                np.subtract(
                    operands[0], operands[1], out=destination, casting="unsafe"
                )
            elif instruction.operation == "mul":
                np.multiply(
                    operands[0], operands[1], out=destination, casting="unsafe"
                )
            elif instruction.operation == "div":
                np.divide(
                    operands[0], operands[1], out=destination, casting="unsafe"
                )
            elif instruction.operation == "matmul":
                np.matmul(operands[0], operands[1], out=destination)
            elif instruction.operation == "abs":
                np.abs(operands[0], out=destination)
            elif instruction.operation == "minimum":
                np.minimum(operands[0], operands[1], out=destination)
            elif instruction.operation == "maximum":
                np.maximum(operands[0], operands[1], out=destination)
            elif instruction.operation == "clip":
                np.clip(operands[0], operands[1], operands[2], out=destination)
            else:  # pragma: no cover - guarded by compilation
                raise RuntimeError(
                    f"unknown operation {instruction.operation!r}"
                )

    @staticmethod
    def _resolve_destination(
        operand: OperandRef,
        *,
        output: np.ndarray,
        temporaries: tuple[np.ndarray, ...],
    ) -> np.ndarray:
        if operand.kind == "output":
            return output
        if operand.kind == "temp":
            return temporaries[operand.value]
        raise RuntimeError(
            f"invalid destination operand kind: {operand.kind!r}"
        )

    @staticmethod
    def _resolve_operand(
        operand: OperandRef | None,
        *,
        values: Mapping[str, np.ndarray],
        output: np.ndarray,
        temporaries: tuple[np.ndarray, ...],
    ) -> Any:
        if operand is None:
            return None
        if operand.kind == "value":
            return values[operand.value]
        if operand.kind == "temp":
            return temporaries[operand.value]
        if operand.kind == "output":
            return output
        if operand.kind == "const":
            return operand.value
        raise RuntimeError(f"invalid operand kind: {operand.kind!r}")


@dataclass(frozen=True)
class _NodeResult:
    operand: OperandRef
    sample: Any
    used_names: frozenset[str]


class ExpressionCompiler:
    """Compile a restricted Python expression into ndarray operations."""

    _BINARY_OPERATIONS = {
        ast.Add: ("add", np.add),
        ast.Sub: ("sub", np.subtract),
        ast.Mult: ("mul", np.multiply),
        ast.Div: ("div", np.divide),
        ast.MatMult: ("matmul", np.matmul),
    }
    _UNARY_OPERATIONS = {
        ast.UAdd: ("pos", lambda value: value),
        ast.USub: ("neg", np.negative),
    }
    _CALL_OPERATIONS = {
        "abs": ("abs", np.abs, 1),
        "max": ("maximum", np.maximum, 2),
        "min": ("minimum", np.minimum, 2),
        "minimum": ("minimum", np.minimum, 2),
        "maximum": ("maximum", np.maximum, 2),
        "clip": ("clip", np.clip, 3),
    }

    def __init__(
        self,
        *,
        expression: str,
        input_sample: np.ndarray,
        auxiliary_samples: Mapping[str, np.ndarray],
        output_shape: tuple[int, ...],
        output_dtype: np.dtype,
        kernel_name: str,
    ) -> None:
        self.expression = expression
        self.samples = {"input": input_sample, **auxiliary_samples}
        self.output_shape = output_shape
        self.output_dtype = np.dtype(output_dtype)
        self.kernel_name = kernel_name
        self._instructions: list[Instruction] = []
        self._temp_shapes: list[tuple[int, ...]] = []
        self._temp_dtypes: list[np.dtype] = []

    def compile(self) -> CompiledExpression:
        """Return a validated compiled plan for the configured expression."""
        try:
            tree = ast.parse(self.expression, mode="eval")
        except SyntaxError as exc:
            raise ConfigValidationError(
                f"kernel {self.kernel_name!r} has invalid operation syntax: {exc.msg}"
            ) from exc
        result = self._compile_node(
            tree.body, destination=OperandRef("output", None)
        )
        if result.operand.kind != "output":
            self._instructions.append(
                Instruction(
                    operation="copy",
                    destination=OperandRef("output", None),
                    operands=(result.operand,),
                )
            )
            result = _NodeResult(
                operand=OperandRef("output", None),
                sample=result.sample,
                used_names=result.used_names,
            )
        if np.shape(result.sample) != self.output_shape:
            raise ConfigValidationError(
                f"kernel {self.kernel_name!r} operation result shape {np.shape(result.sample)!r} "
                f"does not match output shape {self.output_shape!r}"
            )
        result_dtype = np.asarray(result.sample).dtype
        if result_dtype != self.output_dtype:
            raise ConfigValidationError(
                f"kernel {self.kernel_name!r} operation result dtype {result_dtype!r} "
                f"does not match output dtype {self.output_dtype!r}"
            )
        return CompiledExpression(
            expression=self.expression,
            instructions=tuple(self._instructions),
            temp_shapes=tuple(self._temp_shapes),
            temp_dtypes=tuple(self._temp_dtypes),
            result_shape=self.output_shape,
            result_dtype=self.output_dtype,
            used_names=tuple(sorted(result.used_names)),
        )

    def _compile_node(
        self,
        node: ast.AST,
        *,
        destination: OperandRef | None = None,
    ) -> _NodeResult:
        if isinstance(node, ast.Name):
            return self._compile_name(node)
        if isinstance(node, ast.Constant):
            return self._compile_constant(node)
        if isinstance(node, ast.Call):
            return self._compile_call(node, destination=destination)
        if isinstance(node, ast.UnaryOp):
            return self._compile_unary(node, destination=destination)
        if isinstance(node, ast.BinOp):
            return self._compile_binary(node, destination=destination)
        raise ConfigValidationError(
            f"kernel {self.kernel_name!r} operation uses unsupported syntax: "
            f"{node.__class__.__name__}"
        )

    def _compile_name(self, node: ast.Name) -> _NodeResult:
        if node.id not in self.samples:
            allowed = ", ".join(sorted(self.samples))
            raise ConfigValidationError(
                f"kernel {self.kernel_name!r} operation references unknown name {node.id!r}; "
                f"allowed names are: {allowed}"
            )
        return _NodeResult(
            operand=OperandRef("value", node.id),
            sample=self.samples[node.id],
            used_names=frozenset({node.id}),
        )

    def _compile_constant(self, node: ast.Constant) -> _NodeResult:
        value = node.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigValidationError(
                f"kernel {self.kernel_name!r} operation only supports numeric constants"
            )
        scalar = np.asarray(value)
        return _NodeResult(
            operand=OperandRef("const", scalar.item()),
            sample=scalar.item(),
            used_names=frozenset(),
        )

    def _compile_unary(
        self,
        node: ast.UnaryOp,
        *,
        destination: OperandRef | None,
    ) -> _NodeResult:
        operation = self._UNARY_OPERATIONS.get(type(node.op))
        if operation is None:
            raise ConfigValidationError(
                f"kernel {self.kernel_name!r} operation uses unsupported unary operator"
            )
        operand = self._compile_node(node.operand)
        op_name, op_func = operation
        sample = op_func(operand.sample)
        if np.ndim(sample) == 0:
            return _NodeResult(
                operand=OperandRef("const", np.asarray(sample).item()),
                sample=np.asarray(sample).item(),
                used_names=operand.used_names,
            )
        destination = destination or self._allocate_temp(sample)
        self._instructions.append(
            Instruction(
                operation=op_name,
                destination=destination,
                operands=(operand.operand,),
            )
        )
        return _NodeResult(destination, sample, operand.used_names)

    def _compile_call(
        self,
        node: ast.Call,
        *,
        destination: OperandRef | None,
    ) -> _NodeResult:
        if node.keywords:
            raise ConfigValidationError(
                f"kernel {self.kernel_name!r} operation does not support keyword arguments"
            )
        if not isinstance(node.func, ast.Name):
            raise ConfigValidationError(
                f"kernel {self.kernel_name!r} operation only supports direct intrinsic calls"
            )
        operation = self._CALL_OPERATIONS.get(node.func.id)
        if operation is None:
            allowed = ", ".join(sorted(self._CALL_OPERATIONS))
            raise ConfigValidationError(
                f"kernel {self.kernel_name!r} operation uses unsupported function {node.func.id!r}; "
                f"allowed functions are: {allowed}"
            )
        op_name, op_func, arity = operation
        if len(node.args) != arity:
            raise ConfigValidationError(
                f"kernel {self.kernel_name!r} operation function {node.func.id!r} expects {arity} arguments"
            )
        compiled_args = tuple(
            self._compile_node(argument) for argument in node.args
        )
        try:
            sample = op_func(*(argument.sample for argument in compiled_args))
        except Exception as exc:
            raise ConfigValidationError(
                f"kernel {self.kernel_name!r} operation is invalid for the configured stream shapes: {exc}"
            ) from exc
        used_names = frozenset().union(
            *(argument.used_names for argument in compiled_args)
        )
        if np.ndim(sample) == 0:
            return _NodeResult(
                operand=OperandRef("const", np.asarray(sample).item()),
                sample=np.asarray(sample).item(),
                used_names=used_names,
            )
        destination = destination or self._allocate_temp(sample)
        self._instructions.append(
            Instruction(
                operation=op_name,
                destination=destination,
                operands=tuple(argument.operand for argument in compiled_args),
            )
        )
        return _NodeResult(
            operand=destination,
            sample=sample,
            used_names=used_names,
        )

    def _compile_binary(
        self,
        node: ast.BinOp,
        *,
        destination: OperandRef | None,
    ) -> _NodeResult:
        operation = self._BINARY_OPERATIONS.get(type(node.op))
        if operation is None:
            raise ConfigValidationError(
                f"kernel {self.kernel_name!r} operation uses unsupported binary operator"
            )
        left = self._compile_node(node.left)
        right = self._compile_node(node.right)
        op_name, op_func = operation
        try:
            sample = op_func(left.sample, right.sample)
        except Exception as exc:
            raise ConfigValidationError(
                f"kernel {self.kernel_name!r} operation is invalid for the configured stream shapes: {exc}"
            ) from exc
        if np.ndim(sample) == 0:
            return _NodeResult(
                operand=OperandRef("const", np.asarray(sample).item()),
                sample=np.asarray(sample).item(),
                used_names=left.used_names | right.used_names,
            )
        destination = destination or self._allocate_temp(sample)
        self._instructions.append(
            Instruction(
                operation=op_name,
                destination=destination,
                operands=(left.operand, right.operand),
            )
        )
        return _NodeResult(
            operand=destination,
            sample=sample,
            used_names=left.used_names | right.used_names,
        )

    def _allocate_temp(self, sample: Any) -> OperandRef:
        self._temp_shapes.append(np.shape(sample))
        self._temp_dtypes.append(np.asarray(sample).dtype)
        return OperandRef("temp", len(self._temp_shapes) - 1)


def compile_custom_operation(
    *,
    expression: str,
    input_shape: tuple[int, ...],
    input_dtype: np.dtype,
    auxiliary_specs: Mapping[str, tuple[tuple[int, ...], np.dtype]],
    output_shape: tuple[int, ...],
    output_dtype: np.dtype,
    kernel_name: str,
) -> CompiledExpression:
    """Compile and validate one custom operation against stream metadata."""
    compiler = ExpressionCompiler(
        expression=expression,
        input_sample=np.ones(input_shape, dtype=input_dtype),
        auxiliary_samples={
            alias: np.ones(shape, dtype=dtype)
            for alias, (shape, dtype) in auxiliary_specs.items()
        },
        output_shape=output_shape,
        output_dtype=output_dtype,
        kernel_name=kernel_name,
    )
    return compiler.compile()
