"""CPU single-spot centroid kernel."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from shmpipeline.config import KernelConfig, SharedMemoryConfig
from shmpipeline.errors import ConfigValidationError
from shmpipeline.kernels.cpu._common import spot_centroid
from shmpipeline.kernels.cpu.base import CpuKernel


class SpotCentroidCpuKernel(CpuKernel):
    """Compute one centroid for a single guide-star image ROI.

    The output shape is `(2,)` storing `[delta_row, delta_col]` relative to
    the image center.
    """

    kind = "cpu.spot_centroid"

    @classmethod
    def validate_config(
        cls,
        config: KernelConfig,
        shared_memory: Mapping[str, SharedMemoryConfig],
    ) -> None:
        super().validate_config(config, shared_memory)
        image_spec = shared_memory[config.input]
        output_spec = shared_memory[config.output]
        if len(image_spec.shape) != 2:
            raise ConfigValidationError(
                f"kernel {config.name!r} requires a 2D input image"
            )
        if output_spec.shape != (2,):
            raise ConfigValidationError(
                f"kernel {config.name!r} requires output shape (2,)"
            )
        if image_spec.dtype != output_spec.dtype:
            raise ConfigValidationError(
                f"kernel {config.name!r} requires matching image/output dtypes"
            )

        threshold = config.parameters.get("threshold", 0.0)
        background = config.parameters.get("background", 0.0)
        weight_power = config.parameters.get("weight_power", 1.0)
        for name, value in (
            ("threshold", threshold),
            ("background", background),
            ("weight_power", weight_power),
        ):
            if not isinstance(value, (int, float)):
                raise ConfigValidationError(
                    f"kernel {config.name!r} requires numeric parameter {name!r}"
                )
        if float(weight_power) <= 0.0:
            raise ConfigValidationError(
                f"kernel {config.name!r} requires 'weight_power' to be positive"
            )

    def __init__(self, context) -> None:
        super().__init__(context)
        parameters = context.config.parameters
        self.threshold = float(parameters.get("threshold", 0.0))
        self.background = float(parameters.get("background", 0.0))
        self.weight_power = float(parameters.get("weight_power", 1.0))

    def compute_into(
        self,
        trigger_input: Any,
        output: Any,
        auxiliary_inputs: Mapping[str, Any],
    ) -> None:
        del auxiliary_inputs
        spot_centroid(
            np.asarray(trigger_input),
            output,
            self.threshold,
            self.background,
            self.weight_power,
        )