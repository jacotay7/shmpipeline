"""Example source and sink plugins exposed through entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from shmpipeline.errors import ConfigValidationError
from shmpipeline.sink import Sink
from shmpipeline.source import Source


class SimulatedCameraSource(Source):
    """Emit a moving Gaussian spot image as a simple camera stand-in."""

    kind = "example.simulated_camera"
    storage = "cpu"

    @classmethod
    def validate_config(cls, config, shared_memory) -> None:
        super().validate_config(config, shared_memory)
        shape = shared_memory[config.stream].shape
        if len(shape) != 2:
            raise ConfigValidationError(
                f"source kind {cls.kind!r} expects a 2D frame stream"
            )
        sigma = float(config.parameters.get("sigma", 5.0))
        period = float(config.parameters.get("period", 24.0))
        if sigma <= 0.0:
            raise ConfigValidationError("simulated camera sigma must be positive")
        if period <= 0.0:
            raise ConfigValidationError("simulated camera period must be positive")

    def open(self) -> None:
        shape = self.context.stream_spec.shape
        self._frame_index = 0
        self._yy, self._xx = np.indices(shape, dtype=np.float32)
        seed = int(self.context.config.parameters.get("seed", 0))
        self._rng = np.random.default_rng(seed)

    def read(self) -> np.ndarray | None:
        if self.wait(self.context.config.poll_interval):
            return None
        spec = self.context.stream_spec
        params = self.context.config.parameters
        amplitude = float(params.get("amplitude", 30.0))
        background = float(params.get("background", 5.0))
        noise_std = float(params.get("noise_std", 0.0))
        sigma = float(params.get("sigma", 5.0))
        period = float(params.get("period", 24.0))

        phase = 2.0 * np.pi * (self._frame_index / period)
        center_y = (spec.shape[0] - 1) * (0.5 + 0.3 * np.sin(phase))
        center_x = (spec.shape[1] - 1) * (0.5 + 0.3 * np.cos(1.7 * phase))
        distance_sq = (self._yy - center_y) ** 2 + (self._xx - center_x) ** 2
        frame = background + amplitude * np.exp(
            -distance_sq / (2.0 * sigma * sigma)
        )
        if noise_std > 0.0:
            frame = frame + noise_std * self._rng.standard_normal(spec.shape)
        self._frame_index += 1
        return np.asarray(frame, dtype=spec.dtype)


class NpyFrameSink(Sink):
    """Write processed frames to numbered .npy files."""

    kind = "example.npy_frame_sink"
    storage = "cpu"

    @classmethod
    def validate_config(cls, config, shared_memory) -> None:
        super().validate_config(config, shared_memory)
        output_dir = config.parameters.get("output_dir")
        if not isinstance(output_dir, str) or not output_dir.strip():
            raise ConfigValidationError(
                "npy frame sink requires a non-empty output_dir parameter"
            )
        max_saved_frames = int(config.parameters.get("max_saved_frames", 8))
        if max_saved_frames <= 0:
            raise ConfigValidationError(
                "npy frame sink max_saved_frames must be positive"
            )

    def open(self) -> None:
        params = self.context.config.parameters
        self._output_dir = Path(str(params["output_dir"])).expanduser()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._prefix = str(params.get("prefix", "processed_frame"))
        self._max_saved_frames = int(params.get("max_saved_frames", 8))
        self._saved_frames = 0

    def consume(self, value: Any) -> None:
        if self._saved_frames >= self._max_saved_frames:
            return
        path = self._output_dir / f"{self._prefix}_{self._saved_frames:04d}.npy"
        np.save(path, np.asarray(value).copy())
        self._saved_frames += 1


__all__ = ["NpyFrameSink", "SimulatedCameraSource"]
