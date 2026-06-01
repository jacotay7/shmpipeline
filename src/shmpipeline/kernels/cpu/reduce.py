"""CPU reduce kernel — aggregate an input array to a scalar output."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from shmpipeline.config import KernelConfig, SharedMemoryConfig
from shmpipeline.errors import ConfigValidationError
from shmpipeline.kernel import KernelContext
from shmpipeline.kernels.cpu.base import CpuKernel

_SUPPORTED_OPERATIONS = ("sum", "mean", "max", "min")


class ReduceCpuKernel(CpuKernel):
    """Reduce an input array to a scalar using sum, mean, max, or min.

    The output stream must be a scalar (single-element) array.  The input
    stream can have any shape.  Both must share the same dtype.

    Parameters
    ----------
    operation : str
        Reduction operation — one of ``"sum"``, ``"mean"``, ``"max"``,
        ``"min"``.  Default is ``"mean"``.

    Example YAML config::

        kernels:
          - name: mean_signal
            kind: cpu.reduce
            input: raw_stream
            output: scalar_stream
            parameters:
              operation: mean
    """

    kind = "cpu.reduce"
    auxiliary_arity = 0

    @classmethod
    def validate_config(
        cls,
        config: KernelConfig,
        shared_memory: Mapping[str, SharedMemoryConfig],
    ) -> None:
        super().validate_config(config, shared_memory)
        operation = config.parameters.get("operation", "mean")
        if operation not in _SUPPORTED_OPERATIONS:
            raise ConfigValidationError(
                f"kernel {config.name!r}: unsupported reduce operation "
                f"{operation!r}; expected one of: "
                + ", ".join(_SUPPORTED_OPERATIONS)
            )
        input_spec = shared_memory[config.input]
        output_spec = shared_memory[config.output]
        if input_spec.dtype != output_spec.dtype:
            raise ConfigValidationError(
                f"kernel {config.name!r}: reduce requires matching input/output "
                f"dtypes (got {input_spec.dtype} and {output_spec.dtype})"
            )
        output_size = int(np.prod(output_spec.shape, dtype=np.int64))
        if output_size != 1:
            raise ConfigValidationError(
                f"kernel {config.name!r}: reduce output must be a scalar "
                f"(single-element array), got shape {output_spec.shape}"
            )

    def __init__(self, context: KernelContext) -> None:
        super().__init__(context)
        self.operation = context.config.parameters.get("operation", "mean")

    def compute_into(
        self,
        trigger_input: Any,
        output: Any,
        auxiliary_inputs: Mapping[str, Any],
    ) -> None:
        del auxiliary_inputs
        arr = np.asarray(trigger_input)
        if self.operation == "sum":
            result = arr.sum()
        elif self.operation == "mean":
            result = arr.mean()
        elif self.operation == "max":
            result = arr.max()
        else:
            result = arr.min()
        output.reshape(-1)[0] = result
