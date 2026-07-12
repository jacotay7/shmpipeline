"""Fused CPU tip/tilt control kernel."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from shmpipeline.config import KernelConfig, SharedMemoryConfig
from shmpipeline.errors import ConfigValidationError
from shmpipeline.kernels.cpu._common import spot_centroid
from shmpipeline.kernels.cpu.base import CpuKernel


class TipTiltControllerCpuKernel(CpuKernel):
    """Fuse spot centroid, leaky integration, and affine rotation."""

    kind = "cpu.tip_tilt_controller"
    auxiliary_arity = 2

    @classmethod
    def validate_config(
        cls,
        config: KernelConfig,
        shared_memory: Mapping[str, SharedMemoryConfig],
    ) -> None:
        super().validate_config(config, shared_memory)
        image = shared_memory[config.input]
        matrix = shared_memory[config.auxiliary_names[0]]
        bias = shared_memory[config.auxiliary_names[1]]
        output = shared_memory[config.output]
        if len(image.shape) != 2 or matrix.shape != (2, 2):
            raise ConfigValidationError(
                f"kernel {config.name!r} requires a 2D image and 2x2 matrix"
            )
        if bias.shape != (2,) or output.shape != (2,):
            raise ConfigValidationError(
                f"kernel {config.name!r} requires 2-vector bias and output"
            )
        if len({image.dtype, matrix.dtype, bias.dtype, output.dtype}) != 1:
            raise ConfigValidationError(
                f"kernel {config.name!r} requires matching dtypes"
            )

    def __init__(self, context) -> None:
        super().__init__(context)
        parameters = context.config.parameters
        self.background = float(parameters.get("background", 0.0))
        self.threshold = float(parameters.get("threshold", 0.0))
        self.weight_power = float(parameters.get("weight_power", 1.0))
        self.leak = float(parameters.get("leak", 1.0))
        self.control_gain = float(parameters.get("control_gain", 1.0))
        dtype = context.output_spec.dtype
        self._centroid = np.empty(2, dtype=dtype)
        self._state = np.zeros(2, dtype=dtype)

    def compute_into(
        self,
        trigger_input: Any,
        output: Any,
        auxiliary_inputs: Mapping[str, Any],
    ) -> None:
        spot_centroid(
            np.asarray(trigger_input),
            self._centroid,
            self.threshold,
            self.background,
            self.weight_power,
        )
        self._state *= self.leak
        self._state += self.control_gain * self._centroid
        aliases = self.context.config.auxiliary_aliases
        np.matmul(
            np.asarray(auxiliary_inputs[aliases[0]]),
            self._state,
            out=output,
        )
        output += np.asarray(auxiliary_inputs[aliases[1]])
