"""GPU Shack-Hartmann centroid kernel."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from shmpipeline.config import KernelConfig, SharedMemoryConfig
from shmpipeline.errors import ConfigValidationError
from shmpipeline.kernels.gpu.base import GpuKernel, as_gpu_tensor


class ShackHartmannCentroidGpuKernel(GpuKernel):
    """Compute local centroids for a tiled Shack-Hartmann sensor image."""

    kind = "gpu.shack_hartmann_centroid"

    @classmethod
    def validate_config(
        cls,
        config: KernelConfig,
        shared_memory: Mapping[str, SharedMemoryConfig],
    ) -> None:
        super().validate_config(config, shared_memory)
        tile_size = config.parameters.get("tile_size")
        if not isinstance(tile_size, int) or tile_size <= 0:
            raise ConfigValidationError(
                f"kernel {config.name!r} requires a positive integer 'tile_size'"
            )
        image_spec = shared_memory[config.input]
        output_spec = shared_memory[config.output]
        if len(image_spec.shape) != 2:
            raise ConfigValidationError(
                f"kernel {config.name!r} requires a 2D input image"
            )
        if len(output_spec.shape) != 3 or output_spec.shape[2] != 2:
            raise ConfigValidationError(
                f"kernel {config.name!r} requires output shape (rows, cols, 2)"
            )
        if (
            image_spec.shape[0] % tile_size != 0
            or image_spec.shape[1] % tile_size != 0
        ):
            raise ConfigValidationError(
                f"kernel {config.name!r} requires image dimensions divisible by tile_size"
            )
        expected_shape = (
            image_spec.shape[0] // tile_size,
            image_spec.shape[1] // tile_size,
            2,
        )
        if output_spec.shape != expected_shape:
            raise ConfigValidationError(
                f"kernel {config.name!r} requires output shape {expected_shape}"
            )
        if image_spec.dtype != output_spec.dtype:
            raise ConfigValidationError(
                f"kernel {config.name!r} requires matching image/output dtypes"
            )

    def __init__(self, context) -> None:
        super().__init__(context)
        self.tile_size = int(context.config.parameters["tile_size"])
        coords = torch.arange(
            self.tile_size,
            device=self.device,
            dtype=self.output_buffer.dtype,
        )
        self._y_coords = coords.view(self.tile_size, 1)
        self._x_coords = coords.view(1, self.tile_size)
        self._center = 0.5 * (self.tile_size - 1)

    def compute_into(
        self,
        trigger_input: Any,
        output: Any,
        auxiliary_inputs: Mapping[str, Any],
    ) -> None:
        del auxiliary_inputs
        image = as_gpu_tensor(trigger_input, device=self.device)
        tiles_y = image.shape[0] // self.tile_size
        tiles_x = image.shape[1] // self.tile_size
        patches = image.reshape(
            tiles_y,
            self.tile_size,
            tiles_x,
            self.tile_size,
        ).permute(0, 2, 1, 3)
        total = patches.sum(dim=(-1, -2))
        y_weighted = (patches * self._y_coords).sum(dim=(-1, -2))
        x_weighted = (patches * self._x_coords).sum(dim=(-1, -2))
        output.zero_()
        mask = total > 0
        output[..., 0][mask] = y_weighted[mask] / total[mask] - self._center
        output[..., 1][mask] = x_weighted[mask] / total[mask] - self._center
        torch.cuda.synchronize(output.device)
