"""Synthetic shared-memory inputs for testing and interactive demos."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from shmpipeline.errors import ConfigValidationError
from shmpipeline.logging_utils import get_logger

try:
    import torch
except Exception:  # pragma: no cover - exercised when torch is unavailable
    torch = None


_SYNTHETIC_PATTERNS = (
    "constant",
    "random",
    "ramp",
    "sine",
    "impulse",
    "checkerboard",
)


def available_synthetic_patterns() -> tuple[str, ...]:
    """Return supported synthetic input pattern names."""
    return _SYNTHETIC_PATTERNS


@dataclass(frozen=True)
class SyntheticInputConfig:
    """Configuration for one synthetic input writer.

    Synthetic writers can drive source streams without an external producer,
    which makes them useful for demos, GUI exploration, and automated tests.
    """

    stream_name: str
    pattern: str = "random"
    rate_hz: float | None = None
    seed: int = 0
    amplitude: float = 1.0
    offset: float = 0.0
    period: float = 64.0
    constant: float = 0.0
    impulse_interval: int = 64

    def __post_init__(self) -> None:
        pattern = self.pattern.strip().lower()
        object.__setattr__(self, "pattern", pattern)
        if pattern not in _SYNTHETIC_PATTERNS:
            supported = ", ".join(_SYNTHETIC_PATTERNS)
            raise ConfigValidationError(
                f"unsupported synthetic pattern {pattern!r}; expected one of: "
                f"{supported}"
            )
        if self.rate_hz is not None and self.rate_hz <= 0.0:
            raise ConfigValidationError(
                "synthetic input rate_hz must be positive"
            )
        if self.period <= 0.0:
            raise ConfigValidationError(
                "synthetic input period must be positive"
            )
        if self.impulse_interval <= 0:
            raise ConfigValidationError(
                "synthetic input impulse_interval must be positive"
            )


class SyntheticPatternGenerator:
    """Generate deterministic test frames for CPU and GPU streams."""

    def __init__(
        self,
        spec: SyntheticInputConfig,
        *,
        shape: tuple[int, ...],
        dtype: Any,
        storage: str,
        gpu_device: str | None = None,
    ) -> None:
        self.spec = spec
        self.shape = tuple(int(axis) for axis in shape)
        self.dtype = np.dtype(dtype)
        self.storage = storage
        self.gpu_device = gpu_device
        self._frame_index = 0
        self._size = int(np.prod(self.shape, dtype=np.int64))
        self._base_cpu = np.arange(self._size, dtype=np.float32).reshape(
            self.shape
        )
        self._rng = np.random.default_rng(self.spec.seed)

        if self.storage == "gpu":
            if torch is None:
                raise RuntimeError(
                    "synthetic GPU inputs require torch to be installed"
                )
            if gpu_device is None:
                raise RuntimeError(
                    "synthetic GPU inputs require a concrete gpu_device"
                )
            self._torch_dtype = _torch_dtype_for_numpy(self.dtype)
            self._buffer = torch.empty(
                self.shape,
                dtype=self._torch_dtype,
                device=gpu_device,
            )
            self._base_gpu = torch.arange(
                self._size,
                dtype=torch.float32,
                device=gpu_device,
            ).reshape(self.shape)
            self._torch_generator = torch.Generator(device=gpu_device)
            self._torch_generator.manual_seed(self.spec.seed)
        else:
            self._buffer = np.empty(self.shape, dtype=self.dtype)

    def next_frame(self):
        """Return the next generated frame, reusing an internal buffer."""
        frame_index = self._frame_index
        self._frame_index += 1
        if self.storage == "gpu":
            return self._next_gpu(frame_index)
        return self._next_cpu(frame_index)

    def _next_cpu(self, frame_index: int) -> np.ndarray:
        if self.spec.pattern == "constant":
            self._buffer.fill(self.spec.offset + self.spec.constant)
            return self._buffer

        if self.spec.pattern == "random":
            values = (
                self.spec.offset
                + self.spec.amplitude * self._rng.standard_normal(self.shape)
            )
            np.copyto(self._buffer, np.asarray(values, dtype=self.dtype))
            return self._buffer

        if self.spec.pattern == "ramp":
            scale = max(self.spec.period - 1.0, 1.0)
            values = (
                self.spec.offset
                + self.spec.amplitude
                * np.remainder(
                    self._base_cpu + float(frame_index),
                    self.spec.period,
                )
                / scale
            )
            np.copyto(self._buffer, np.asarray(values, dtype=self.dtype))
            return self._buffer

        if self.spec.pattern == "sine":
            values = self.spec.offset + self.spec.amplitude * np.sin(
                2.0
                * np.pi
                * (
                    (self._base_cpu / self.spec.period)
                    + (frame_index / self.spec.period)
                )
            )
            np.copyto(self._buffer, np.asarray(values, dtype=self.dtype))
            return self._buffer

        if self.spec.pattern == "impulse":
            self._buffer.fill(self.spec.offset)
            if frame_index % self.spec.impulse_interval == 0:
                flat_index = (
                    frame_index // self.spec.impulse_interval
                ) % self._size
                self._buffer.reshape(-1)[flat_index] = np.asarray(
                    self.spec.offset + self.spec.amplitude,
                    dtype=self.dtype,
                )
            return self._buffer

        values = self.spec.offset + self.spec.amplitude * np.remainder(
            self._base_cpu,
            2.0,
        )
        np.copyto(self._buffer, np.asarray(values, dtype=self.dtype))
        return self._buffer

    def _next_gpu(self, frame_index: int):
        assert torch is not None
        if self.spec.pattern == "constant":
            self._buffer.fill_(self.spec.offset + self.spec.constant)
            return self._buffer

        if self.spec.pattern == "random":
            if self._buffer.dtype.is_floating_point:
                self._buffer.normal_(
                    mean=self.spec.offset,
                    std=max(abs(self.spec.amplitude), 1e-6),
                    generator=self._torch_generator,
                )
            else:
                values = torch.randn(
                    self.shape,
                    generator=self._torch_generator,
                    device=self.gpu_device,
                )
                values.mul_(self.spec.amplitude).add_(self.spec.offset)
                self._buffer.copy_(values.to(dtype=self._buffer.dtype))
            return self._buffer

        if self.spec.pattern == "ramp":
            scale = max(self.spec.period - 1.0, 1.0)
            values = (
                self.spec.offset
                + self.spec.amplitude
                * torch.remainder(
                    self._base_gpu + float(frame_index),
                    self.spec.period,
                )
                / scale
            )
            self._buffer.copy_(values.to(dtype=self._buffer.dtype))
            return self._buffer

        if self.spec.pattern == "sine":
            values = self.spec.offset + self.spec.amplitude * torch.sin(
                2.0
                * math.pi
                * (
                    (self._base_gpu / self.spec.period)
                    + (frame_index / self.spec.period)
                )
            )
            self._buffer.copy_(values.to(dtype=self._buffer.dtype))
            return self._buffer

        if self.spec.pattern == "impulse":
            self._buffer.fill_(self.spec.offset)
            if frame_index % self.spec.impulse_interval == 0:
                flat_index = (
                    frame_index // self.spec.impulse_interval
                ) % self._size
                self._buffer.reshape(-1)[flat_index] = (
                    self.spec.offset + self.spec.amplitude
                )
            return self._buffer

        values = self.spec.offset + self.spec.amplitude * torch.remainder(
            self._base_gpu,
            2.0,
        )
        self._buffer.copy_(values.to(dtype=self._buffer.dtype))
        return self._buffer


class SyntheticSourceController:
    """Background writer that feeds a stream with synthetic test frames.

    The controller owns the worker thread and the timing loop for one active
    synthetic input source.
    """

    def __init__(self, stream: Any, spec: SyntheticInputConfig) -> None:
        self.stream = stream
        self.spec = spec
        self._logger = get_logger(f"synthetic.{spec.stream_name}")
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"shmpipeline-synthetic-{spec.stream_name}",
            daemon=True,
        )
        self._started_at_wall: float | None = None
        self._started_at_mono: float | None = None
        self._last_write_time: float | None = None
        self._last_write_duration_s: float | None = None
        self._frames_written = 0
        self._last_error: str | None = None
        self._generator = SyntheticPatternGenerator(
            spec,
            shape=stream.shape,
            dtype=stream.dtype,
            storage="gpu" if stream.gpu_enabled else "cpu",
            gpu_device=stream.gpu_device,
        )

    def start(self) -> None:
        """Start the background writer thread."""
        if self._thread.is_alive():
            return
        self._started_at_wall = time.time()
        self._started_at_mono = time.perf_counter()
        self._thread.start()

    def stop(self, *, timeout: float = 2.0) -> None:
        """Request that the background writer stop and wait for it."""
        self._stop_event.set()
        self._thread.join(timeout=timeout)

    def snapshot(self) -> dict[str, Any]:
        """Return a stable status snapshot for GUI and CLI consumers."""
        with self._lock:
            elapsed_s = 0.0
            if self._started_at_mono is not None:
                elapsed_s = max(
                    0.0, time.perf_counter() - self._started_at_mono
                )
            effective_rate_hz = 0.0
            if elapsed_s > 0.0:
                effective_rate_hz = self._frames_written / elapsed_s
            return {
                "stream_name": self.spec.stream_name,
                "pattern": self.spec.pattern,
                "rate_hz": self.spec.rate_hz,
                "requested_rate_hz": self.spec.rate_hz,
                "seed": self.spec.seed,
                "amplitude": self.spec.amplitude,
                "offset": self.spec.offset,
                "period": self.spec.period,
                "constant": self.spec.constant,
                "impulse_interval": self.spec.impulse_interval,
                "frames_written": self._frames_written,
                "alive": self._thread.is_alive(),
                "effective_rate_hz": effective_rate_hz,
                "started_at": self._started_at_wall,
                "last_write_time": self._last_write_time,
                "last_write_duration_ms": (
                    None
                    if self._last_write_duration_s is None
                    else 1000.0 * self._last_write_duration_s
                ),
                "last_error": self._last_error,
            }

    def _run(self) -> None:
        next_deadline = time.perf_counter()
        interval = None
        if self.spec.rate_hz is not None:
            interval = 1.0 / self.spec.rate_hz
        try:
            while not self._stop_event.is_set():
                if interval is not None:
                    remaining = next_deadline - time.perf_counter()
                    if remaining > 0.0 and self._stop_event.wait(remaining):
                        return

                frame = self._generator.next_frame()
                started = time.perf_counter()
                self.stream.write(frame)
                finished = time.perf_counter()

                with self._lock:
                    self._frames_written += 1
                    self._last_write_time = time.time()
                    self._last_write_duration_s = finished - started

                if interval is not None:
                    next_deadline = max(next_deadline + interval, finished)
        except BaseException as exc:
            self._logger.exception(
                "synthetic source failed: stream=%s pattern=%s",
                self.spec.stream_name,
                self.spec.pattern,
            )
            with self._lock:
                self._last_error = str(exc)


def _torch_dtype_for_numpy(dtype: np.dtype):
    dtype = np.dtype(dtype)
    if torch is None:
        raise RuntimeError("torch is required for GPU synthetic inputs")
    mapping = {
        np.dtype(np.float32): torch.float32,
        np.dtype(np.float64): torch.float64,
        np.dtype(np.int32): torch.int32,
        np.dtype(np.int64): torch.int64,
        np.dtype(np.uint8): torch.uint8,
        np.dtype(np.bool_): torch.bool,
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise RuntimeError(
            f"unsupported GPU synthetic dtype: {dtype}"
        ) from exc
