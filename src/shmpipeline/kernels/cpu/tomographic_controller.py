"""Fused CPU kernel for sustained-rate tomographic AO control.

Mirrors :class:`shmpipeline.kernels.gpu.tomographic_controller
.TomographicControllerGpuKernel` on the host: it fuses eight WFS front ends
(dark/flat calibration, tiled Shack-Hartmann centroids, slope calibration), the
tomographic reconstruction, the leaky-integrator control law, and the final
command calibration and per-actuator clip into one worker. Accepting either one
batched ``(8, rows, cols)`` input or eight separate ``(rows, cols)`` WFS inputs
keeps the CPU and GPU example topologies identical.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from shmpipeline.config import KernelConfig, SharedMemoryConfig
from shmpipeline.errors import ConfigValidationError
from shmpipeline.kernels.cpu._common import centroid_tiles
from shmpipeline.kernels.cpu.base import CpuKernel


def validate_tomographic_controller_config(
    config: KernelConfig,
    shared_memory: Mapping[str, SharedMemoryConfig],
) -> None:
    """Validate the fused tomographic controller wiring (backend agnostic)."""
    if config.trigger_policy != "all_new":
        raise ConfigValidationError(
            f"kernel {config.name!r} requires trigger_policy 'all_new'"
        )
    batched = len(config.trigger_inputs) == 1
    if len(config.trigger_inputs) not in {1, 8}:
        raise ConfigValidationError(
            f"kernel {config.name!r} requires one batched or eight WFS inputs"
        )
    required = {
        "reconstructor",
        "reconstructor_bias",
        "command_offset",
        "command_low",
        "command_high",
    }
    if batched:
        required.update({"wfs_dark", "wfs_inverse_flat", "wfs_slope_offset"})
    else:
        for index in range(8):
            required.add(f"wfs{index}_dark")
            required.add(f"wfs{index}_inverse_flat")
            required.add(f"wfs{index}_slope_offset")
    aliases = set(config.auxiliary_aliases)
    if aliases != required:
        missing = sorted(required - aliases)
        extra = sorted(aliases - required)
        raise ConfigValidationError(
            f"kernel {config.name!r} auxiliary aliases mismatch: "
            f"missing={missing}, extra={extra}"
        )
    tile_size = config.parameters.get("tile_size", 4)
    if not isinstance(tile_size, int) or tile_size <= 0:
        raise ConfigValidationError(
            f"kernel {config.name!r} requires positive integer tile_size"
        )
    input_specs = [shared_memory[name] for name in config.trigger_inputs]
    if batched:
        if len(input_specs[0].shape) != 3 or input_specs[0].shape[0] != 8:
            raise ConfigValidationError(
                f"kernel {config.name!r} requires batched input shape "
                "(8, rows, columns)"
            )
        image_shape = input_specs[0].shape[1:]
    else:
        image_shape = input_specs[0].shape
    if len(image_shape) != 2 or (
        not batched and any(spec.shape != image_shape for spec in input_specs)
    ):
        raise ConfigValidationError(
            f"kernel {config.name!r} requires equal 2D WFS inputs"
        )
    if image_shape[0] % tile_size or image_shape[1] % tile_size:
        raise ConfigValidationError(
            f"kernel {config.name!r} image shape must divide by tile_size"
        )
    centroid_shape = (
        image_shape[0] // tile_size,
        image_shape[1] // tile_size,
        2,
    )
    slope_count = centroid_shape[0] * centroid_shape[1] * 2 * 8
    matrix = shared_memory[config.auxiliary_by_alias["reconstructor"]]
    if len(matrix.shape) != 2 or matrix.shape[1] != slope_count:
        raise ConfigValidationError(
            f"kernel {config.name!r} requires reconstructor columns to equal "
            f"{slope_count}"
        )
    output_spec = shared_memory[config.output]
    if output_spec.shape != (matrix.shape[0],):
        raise ConfigValidationError(
            f"kernel {config.name!r} output must match reconstructor rows"
        )
    for index in range(1 if batched else 8):
        for suffix, sensor_shape in (
            ("dark", image_shape),
            ("inverse_flat", image_shape),
            ("slope_offset", centroid_shape),
        ):
            alias = f"wfs_{suffix}" if batched else f"wfs{index}_{suffix}"
            expected_shape = (8, *sensor_shape) if batched else sensor_shape
            if shared_memory[config.auxiliary_by_alias[alias]].shape != (
                expected_shape
            ):
                raise ConfigValidationError(
                    f"kernel {config.name!r} requires {alias} shape "
                    f"{expected_shape}"
                )
    for alias in (
        "reconstructor_bias",
        "command_offset",
        "command_low",
        "command_high",
    ):
        if shared_memory[config.auxiliary_by_alias[alias]].shape != (
            matrix.shape[0],
        ):
            raise ConfigValidationError(
                f"kernel {config.name!r} requires {alias} shape "
                f"{(matrix.shape[0],)}"
            )


class TomographicControllerCpuKernel(CpuKernel):
    """Fuse eight WFS front ends, reconstruction, and DM control on the host."""

    kind = "cpu.tomographic_controller"
    input_arity = None
    auxiliary_arity = None

    @classmethod
    def validate_config(
        cls,
        config: KernelConfig,
        shared_memory: Mapping[str, SharedMemoryConfig],
    ) -> None:
        super().validate_config(config, shared_memory)
        validate_tomographic_controller_config(config, shared_memory)

    def __init__(self, context) -> None:
        super().__init__(context)
        parameters = context.config.parameters
        self.tile_size = int(parameters.get("tile_size", 4))
        self.slope_gain = float(parameters.get("slope_gain", 1.0))
        self.leak = float(parameters.get("leak", 1.0))
        self.control_gain = float(parameters.get("control_gain", 1.0))
        self.command_gain = float(parameters.get("command_gain", 1.0))
        self._batched = len(context.config.trigger_inputs) == 1
        input_shape = context.trigger_input_specs[0].shape
        rows, columns = input_shape[1:] if self._batched else input_shape
        dtype = context.output_spec.dtype
        tiles_y = rows // self.tile_size
        tiles_x = columns // self.tile_size
        self._centroids = np.empty((tiles_y, tiles_x, 2), dtype=dtype)
        self._slopes_per_wfs = tiles_y * tiles_x * 2
        self._slopes = np.empty(self._slopes_per_wfs * 8, dtype=dtype)
        actuator_count = context.output_spec.shape[0]
        self._state = np.zeros(actuator_count, dtype=dtype)
        self._calibrated = np.empty((rows, columns), dtype=dtype)

    def _aux(self, inputs: Mapping[str, Any], alias: str) -> np.ndarray:
        return np.asarray(inputs[alias])

    def _wfs_aux(
        self, inputs: Mapping[str, Any], suffix: str, index: int
    ) -> np.ndarray:
        if self._batched:
            return self._aux(inputs, f"wfs_{suffix}")[index]
        return self._aux(inputs, f"wfs{index}_{suffix}")

    def _centroid_into_slopes(
        self,
        image: np.ndarray,
        auxiliary_inputs: Mapping[str, Any],
        index: int,
    ) -> None:
        np.subtract(
            image,
            self._wfs_aux(auxiliary_inputs, "dark", index),
            out=self._calibrated,
        )
        np.multiply(
            self._calibrated,
            self._wfs_aux(auxiliary_inputs, "inverse_flat", index),
            out=self._calibrated,
        )
        centroid_tiles(self._calibrated, self._centroids, self.tile_size)
        start = index * self._slopes_per_wfs
        destination = self._slopes[start : start + self._slopes_per_wfs]
        np.multiply(
            self._centroids.reshape(-1), self.slope_gain, out=destination
        )
        np.subtract(
            destination,
            self._wfs_aux(auxiliary_inputs, "slope_offset", index).reshape(-1),
            out=destination,
        )

    def compute_into(
        self,
        trigger_input: Any,
        output: Any,
        auxiliary_inputs: Mapping[str, Any],
    ) -> None:
        images = trigger_input if self._batched else tuple(trigger_input)
        for index, image in enumerate(images):
            self._centroid_into_slopes(
                np.asarray(image), auxiliary_inputs, index
            )
        residual = self._aux(auxiliary_inputs, "reconstructor") @ self._slopes
        residual = residual + self._aux(auxiliary_inputs, "reconstructor_bias")
        self._state *= self.leak
        self._state += self.control_gain * residual
        np.multiply(self._state, self.command_gain, out=output)
        np.subtract(
            output, self._aux(auxiliary_inputs, "command_offset"), out=output
        )
        np.maximum(
            output, self._aux(auxiliary_inputs, "command_low"), out=output
        )
        np.minimum(
            output, self._aux(auxiliary_inputs, "command_high"), out=output
        )
