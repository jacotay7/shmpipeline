"""GPU single-spot centroid kernel."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from shmpipeline.config import KernelConfig, SharedMemoryConfig
from shmpipeline.errors import ConfigValidationError
from shmpipeline.kernels.gpu.base import GpuKernel, as_gpu_tensor


class SpotCentroidGpuKernel(GpuKernel):
    """Compute one background-subtracted centroid on a CUDA image."""

    kind = "gpu.spot_centroid"

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
        for name, default in (
            ("threshold", 0.0),
            ("background", 0.0),
            ("weight_power", 1.0),
        ):
            value = config.parameters.get(name, default)
            if not isinstance(value, (int, float)):
                raise ConfigValidationError(
                    f"kernel {config.name!r} requires numeric parameter {name!r}"
                )
        if float(config.parameters.get("weight_power", 1.0)) <= 0.0:
            raise ConfigValidationError(
                f"kernel {config.name!r} requires 'weight_power' to be positive"
            )

    def __init__(self, context) -> None:
        super().__init__(context)
        parameters = context.config.parameters
        self.threshold = float(parameters.get("threshold", 0.0))
        self.background = float(parameters.get("background", 0.0))
        self.weight_power = float(parameters.get("weight_power", 1.0))
        rows, columns = context.trigger_input_spec.shape
        dtype = self.output_buffer.dtype
        self._row_coords = torch.arange(
            rows, dtype=dtype, device=self.device
        ).view(rows, 1)
        self._column_coords = torch.arange(
            columns, dtype=dtype, device=self.device
        ).view(1, columns)
        self._row_center = 0.5 * (rows - 1)
        self._column_center = 0.5 * (columns - 1)
        self._weights = torch.empty(
            (rows, columns), dtype=dtype, device=self.device
        )

    def compute_into(
        self,
        trigger_input: Any,
        output: Any,
        auxiliary_inputs: Mapping[str, Any],
    ) -> None:
        del auxiliary_inputs
        image = as_gpu_tensor(trigger_input, device=self.device)
        torch.sub(image, self.background, out=self._weights)
        self._weights.clamp_(min=0.0)
        self._weights.masked_fill_(self._weights <= self.threshold, 0.0)
        if self.weight_power != 1.0:
            self._weights.pow_(self.weight_power)
        total = self._weights.sum()
        output.zero_()
        if total.item() > 0.0:
            output[0] = (
                self._weights * self._row_coords
            ).sum() / total - self._row_center
            output[1] = (
                self._weights * self._column_coords
            ).sum() / total - self._column_center
        torch.cuda.synchronize(output.device)
