"""A coordinated multi-output synthetic camera source.

``synthetic.frame_set`` models one hardware trigger driving several cameras.
A single controller thread generates one deterministic frame per output stream
for generation N, applies configurable per-camera arrival jitter, optionally
injects drops to exercise barrier timeout/recovery, and publishes each stream
once.  It reuses one buffer per stream (no per-frame allocation) and, on GPU,
generates directly on the configured device.

Independent per-stream :class:`SyntheticSourceController` threads cannot model a
common trigger or inject controlled inter-camera skew; this source can, which is
what the synchronized fan-in (``all_new`` / ``matching_frame_id``) needs to be
exercised honestly.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Mapping

import numpy as np

from shmpipeline.config import SharedMemoryConfig, SourceConfig
from shmpipeline.errors import ConfigValidationError
from shmpipeline.source import Source
from shmpipeline.synthetic import (
    SyntheticPatternGenerator,
    synthetic_config_from_parameters,
)


class SyntheticFrameSetSource(Source):
    """Publish one synchronized frame set across several streams per cycle."""

    kind = "synthetic.frame_set"

    @classmethod
    def validate_config(
        cls,
        config: SourceConfig,
        shared_memory: Mapping[str, SharedMemoryConfig],
    ) -> None:
        """Validate stream membership, matching storage, and timing params."""
        streams = config.output_streams
        if len(streams) < 2:
            raise ConfigValidationError(
                f"source {config.name!r} requires at least two output streams; "
                "use 'streams: [...]'"
            )
        missing = [name for name in streams if name not in shared_memory]
        if missing:
            raise ConfigValidationError(
                f"source {config.name!r} references unknown streams "
                f"{missing!r}"
            )
        storages = {shared_memory[name].storage for name in streams}
        if len(storages) != 1:
            raise ConfigValidationError(
                f"source {config.name!r} requires all output streams to share "
                "one storage backend"
            )
        rate = config.parameters.get("rate_hz")
        if rate is not None and (
            not isinstance(rate, (int, float)) or rate <= 0.0
        ):
            raise ConfigValidationError(
                f"source {config.name!r} rate_hz must be a positive number "
                "or null"
            )
        for key in ("jitter_us", "drop_probability"):
            value = config.parameters.get(key, 0.0)
            if not isinstance(value, (int, float)) or value < 0.0:
                raise ConfigValidationError(
                    f"source {config.name!r} {key} must be a non-negative "
                    "number"
                )
        if config.parameters.get("drop_probability", 0.0) > 1.0:
            raise ConfigValidationError(
                f"source {config.name!r} drop_probability must be <= 1.0"
            )
        synthetic_config_from_parameters(streams[0], config.parameters)

    def __init__(self, context) -> None:
        super().__init__(context)
        parameters = context.config.parameters
        self._stream_names = tuple(context.config.output_streams)
        rate = parameters.get("rate_hz")
        self._interval = None if rate is None else 1.0 / float(rate)
        self._jitter_s = float(parameters.get("jitter_us", 0.0)) * 1e-6
        self._drop_probability = float(parameters.get("drop_probability", 0.0))
        # Assign the same monotonically increasing frame_id token to every
        # camera in one generation so a downstream matching_frame_id barrier
        # can combine images from the same hardware trigger.
        self._assign_frame_id = bool(parameters.get("assign_frame_id", True))
        self._rng = np.random.default_rng(int(parameters.get("seed", 0)))
        self._next_deadline: float | None = None
        self._generators: dict[str, SyntheticPatternGenerator] = {}
        for index, name in enumerate(self._stream_names):
            spec = context.shared_memory[name]
            # Offset each camera's seed so streams are not bitwise identical.
            per_camera = dict(parameters)
            per_camera["seed"] = int(parameters.get("seed", 0)) + index
            self._generators[name] = SyntheticPatternGenerator(
                synthetic_config_from_parameters(name, per_camera),
                shape=spec.shape,
                dtype=spec.dtype,
                storage=spec.storage,
                gpu_device=spec.gpu_device,
            )
        # Metrics are read by the manager's status thread while produce() runs
        # in the source thread. Keep the whole generation update behind one
        # lock so status never exposes a half-published frame set.
        self._metrics_lock = threading.Lock()
        self._generation = 0
        self._writes = {name: 0 for name in self._stream_names}
        self._dropped = {name: 0 for name in self._stream_names}
        self._last_skew_s = 0.0
        self._max_skew_s = 0.0

    def _sleep(self, seconds: float) -> bool:
        """Sleep up to ``seconds``, returning True if a stop was requested."""
        if seconds <= 0.0:
            return self.stop_requested()
        if self._stop_event is not None:
            return self._stop_event.wait(seconds)
        time.sleep(seconds)
        return self.stop_requested()

    def produce(self, writers: Mapping[str, Any]) -> int | None:
        """Publish one jittered, synchronized generation across all streams."""
        if self._interval is not None:
            now = time.perf_counter()
            if self._next_deadline is None:
                self._next_deadline = now
            remaining = self._next_deadline - now
            if remaining > 0.0 and self._sleep(remaining):
                return None
            self._next_deadline = max(
                self._next_deadline + self._interval, time.perf_counter()
            )
        # Token for this generation (1-based); shared by every camera write.
        token = self._generation + 1 if self._assign_frame_id else None
        with self._metrics_lock:
            first_write: float | None = None
            last_write: float | None = None
            for name in self._stream_names:
                if self._jitter_s > 0.0:
                    delay = float(self._rng.uniform(0.0, self._jitter_s))
                    if self._sleep(delay):
                        return None
                if (
                    self._drop_probability > 0.0
                    and float(self._rng.random()) < self._drop_probability
                ):
                    self._dropped[name] += 1
                    continue
                frame = self._generators[name].next_frame()
                moment = time.perf_counter()
                writers[name].write(frame, frame_id=token)
                self._writes[name] += 1
                first_write = moment if first_write is None else first_write
                last_write = moment
            self._generation += 1
            if first_write is not None and last_write is not None:
                self._last_skew_s = last_write - first_write
                self._max_skew_s = max(self._max_skew_s, self._last_skew_s)
        return 1

    def plugin_metrics(self) -> dict[str, Any]:
        """Report per-camera writes, drops, and generation skew."""
        with self._metrics_lock:
            return {
                "generations": self._generation,
                "per_stream_writes": dict(self._writes),
                "per_stream_drops": dict(self._dropped),
                "requested_rate_hz": (
                    None if self._interval is None else 1.0 / self._interval
                ),
                "jitter_us": self._jitter_s * 1e6,
                "drop_probability": self._drop_probability,
                "last_skew_us": self._last_skew_s * 1e6,
                "max_skew_us": self._max_skew_s * 1e6,
            }
