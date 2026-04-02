"""CPU Shack-Hartmann centroid kernel."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from shmpipeline.config import KernelConfig, SharedMemoryConfig
from shmpipeline.errors import ConfigValidationError
from shmpipeline.kernels.cpu._common import centroid_tiles
from shmpipeline.kernels.cpu.base import CpuKernel


class ShackHartmannCentroidCpuKernel(CpuKernel):
    """Compute local centroids for a tiled Shack-Hartmann sensor image.

    The output shape is `(rows, cols, 2)` where the last axis stores
    `[delta_row, delta_col]` relative to each tile center.
    """

    kind = "cpu.shack_hartmann_centroid"

    @classmethod
    def validate_config(
        cls,
        config: KernelConfig,
        shared_memory: Mapping[str, SharedMemoryConfig],
    ) -> None:
        """Require compatible image and centroid-output geometry."""
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
        """Store the subaperture tile size."""
        super().__init__(context)
        self.tile_size = int(context.config.parameters["tile_size"])

    def compute_into(
        self,
        trigger_input: Any,
        output: Any,
        auxiliary_inputs: Mapping[str, Any],
    ) -> None:
        """Compute centroids tile-by-tile into the reusable output array."""
        del auxiliary_inputs
        centroid_tiles(np.asarray(trigger_input), output, self.tile_size)
