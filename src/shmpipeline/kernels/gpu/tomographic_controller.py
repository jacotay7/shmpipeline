"""Fused GPU kernels for sustained-rate tomographic AO control."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from shmpipeline.config import KernelConfig, SharedMemoryConfig
from shmpipeline.errors import ConfigValidationError
from shmpipeline.kernels.gpu.base import GpuKernel, as_gpu_tensor


class TomographicControllerGpuKernel(GpuKernel):
    """Fuse eight WFS front ends, reconstruction, and DM control."""

    kind = "gpu.tomographic_controller"
    input_arity = None
    auxiliary_arity = None

    @classmethod
    def validate_config(
        cls,
        config: KernelConfig,
        shared_memory: Mapping[str, SharedMemoryConfig],
    ) -> None:
        super().validate_config(config, shared_memory)
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
            required.update(
                {"wfs_dark", "wfs_inverse_flat", "wfs_slope_offset"}
            )
        else:
            required.update(f"wfs{index}_dark" for index in range(8))
            required.update(f"wfs{index}_inverse_flat" for index in range(8))
            required.update(f"wfs{index}_slope_offset" for index in range(8))
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
            not batched
            and any(spec.shape != image_shape for spec in input_specs)
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
        slopes_per_wfs = centroid_shape[0] * centroid_shape[1] * 2
        slope_count = slopes_per_wfs * 8
        output_spec = shared_memory[config.output]
        matrix = shared_memory[config.auxiliary_by_alias["reconstructor"]]
        if len(matrix.shape) != 2 or matrix.shape[1] != slope_count:
            raise ConfigValidationError(
                f"kernel {config.name!r} requires reconstructor columns "
                f"to equal {slope_count}"
            )
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
                expected_shape = (
                    (8, *sensor_shape) if batched else sensor_shape
                )
                spec = shared_memory[config.auxiliary_by_alias[alias]]
                if spec.shape != expected_shape:
                    raise ConfigValidationError(
                        f"kernel {config.name!r} requires {alias} "
                        f"shape {expected_shape}"
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
        dtypes = {
            shared_memory[name].dtype
            for name in (*config.all_inputs, *config.all_outputs)
        }
        if len(dtypes) != 1:
            raise ConfigValidationError(
                f"kernel {config.name!r} requires matching dtypes"
            )

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
        tiles_y = rows // self.tile_size
        tiles_x = columns // self.tile_size
        dtype = self.output_buffer.dtype
        self._calibrated = torch.empty(
            (rows, columns), dtype=dtype, device=self.device
        )
        self._centroids = torch.empty(
            (tiles_y, tiles_x, 2), dtype=dtype, device=self.device
        )
        self._slopes_per_wfs = tiles_y * tiles_x * 2
        self._slopes = torch.empty(
            self._slopes_per_wfs * 8, dtype=dtype, device=self.device
        )
        actuator_count = context.output_spec.shape[0]
        self._residual = torch.empty(
            actuator_count, dtype=dtype, device=self.device
        )
        self._state = torch.zeros(
            actuator_count, dtype=dtype, device=self.device
        )
        coords = torch.arange(self.tile_size, dtype=dtype, device=self.device)
        self._y_coords = coords.view(self.tile_size, 1)
        self._x_coords = coords.view(1, self.tile_size)
        self._center = 0.5 * (self.tile_size - 1)

    def _aux(self, inputs: Mapping[str, Any], alias: str) -> torch.Tensor:
        return as_gpu_tensor(inputs[alias], device=self.device)

    def _wfs_aux(
        self, inputs: Mapping[str, Any], suffix: str, index: int
    ) -> torch.Tensor:
        if self._batched:
            return self._aux(inputs, f"wfs_{suffix}")[index]
        return self._aux(inputs, f"wfs{index}_{suffix}")

    def _centroid_into_slopes(
        self,
        image: torch.Tensor,
        auxiliary_inputs: Mapping[str, Any],
        index: int,
    ) -> None:
        torch.sub(
            image,
            self._wfs_aux(auxiliary_inputs, "dark", index),
            out=self._calibrated,
        )
        torch.mul(
            self._calibrated,
            self._wfs_aux(auxiliary_inputs, "inverse_flat", index),
            out=self._calibrated,
        )
        rows, columns = self._calibrated.shape
        patches = self._calibrated.reshape(
            rows // self.tile_size,
            self.tile_size,
            columns // self.tile_size,
            self.tile_size,
        ).permute(0, 2, 1, 3)
        total = patches.sum(dim=(-1, -2))
        y_weighted = (patches * self._y_coords).sum(dim=(-1, -2))
        x_weighted = (patches * self._x_coords).sum(dim=(-1, -2))
        valid = total > 0
        safe_total = total.clamp_min(torch.finfo(total.dtype).tiny)
        self._centroids[..., 0].copy_(y_weighted / safe_total - self._center)
        self._centroids[..., 1].copy_(x_weighted / safe_total - self._center)
        self._centroids.masked_fill_(~valid.unsqueeze(-1), 0.0)
        start = index * self._slopes_per_wfs
        destination = self._slopes[start : start + self._slopes_per_wfs]
        torch.mul(
            self._centroids.reshape(-1), self.slope_gain, out=destination
        )
        torch.sub(
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
                as_gpu_tensor(image, device=self.device),
                auxiliary_inputs,
                index,
            )
        torch.matmul(
            self._aux(auxiliary_inputs, "reconstructor"),
            self._slopes,
            out=self._residual,
        )
        torch.add(
            self._residual,
            self._aux(auxiliary_inputs, "reconstructor_bias"),
            out=self._residual,
        )
        self._state.mul_(self.leak)
        self._state.add_(self._residual, alpha=self.control_gain)
        torch.mul(self._state, self.command_gain, out=output)
        torch.sub(
            output,
            self._aux(auxiliary_inputs, "command_offset"),
            out=output,
        )
        torch.maximum(
            output, self._aux(auxiliary_inputs, "command_low"), out=output
        )
        torch.minimum(
            output, self._aux(auxiliary_inputs, "command_high"), out=output
        )
        torch.cuda.synchronize(output.device)


class TipTiltControllerGpuKernel(GpuKernel):
    """Fuse spot centroid, leaky integration, and affine rotation."""

    kind = "gpu.tip_tilt_controller"
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

    def __init__(self, context) -> None:
        super().__init__(context)
        parameters = context.config.parameters
        self.background = float(parameters.get("background", 0.0))
        self.threshold = float(parameters.get("threshold", 0.0))
        self.leak = float(parameters.get("leak", 1.0))
        self.control_gain = float(parameters.get("control_gain", 1.0))
        rows, columns = context.trigger_input_spec.shape
        dtype = self.output_buffer.dtype
        self._weights = torch.empty(
            (rows, columns), dtype=dtype, device=self.device
        )
        self._centroid = torch.empty(2, dtype=dtype, device=self.device)
        self._state = torch.zeros(2, dtype=dtype, device=self.device)
        self._rows = torch.arange(rows, dtype=dtype, device=self.device).view(
            rows, 1
        )
        self._columns = torch.arange(
            columns, dtype=dtype, device=self.device
        ).view(1, columns)
        self._center_y = 0.5 * (rows - 1)
        self._center_x = 0.5 * (columns - 1)

    def compute_into(
        self,
        trigger_input: Any,
        output: Any,
        auxiliary_inputs: Mapping[str, Any],
    ) -> None:
        image = as_gpu_tensor(trigger_input, device=self.device)
        torch.sub(image, self.background, out=self._weights)
        self._weights.masked_fill_(self._weights <= self.threshold, 0.0)
        total = self._weights.sum()
        safe_total = total.clamp_min(torch.finfo(total.dtype).tiny)
        self._centroid[0] = (
            self._weights * self._rows
        ).sum() / safe_total - self._center_y
        self._centroid[1] = (
            self._weights * self._columns
        ).sum() / safe_total - self._center_x
        self._centroid.masked_fill_(total <= 0, 0.0)
        self._state.mul_(self.leak)
        self._state.add_(self._centroid, alpha=self.control_gain)
        aliases = self.context.config.auxiliary_aliases
        matrix = as_gpu_tensor(
            auxiliary_inputs[aliases[0]], device=self.device
        )
        bias = as_gpu_tensor(auxiliary_inputs[aliases[1]], device=self.device)
        torch.matmul(matrix, self._state, out=output)
        torch.add(output, bias, out=output)
        torch.cuda.synchronize(output.device)
