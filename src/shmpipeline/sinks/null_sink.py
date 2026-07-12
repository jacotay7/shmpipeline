"""A general-purpose terminal sink that discards payloads and measures them.

``null.sink`` stands in for a fake hardware endpoint (a deformable mirror, a
tip/tilt stage, a camera-frame drain).  It consumes every publication the
runtime hands it, optionally emulates a fixed device latency, and records
consume-time percentiles so a benchmark can measure the pipeline boundary
without measuring terminal console polling.  It never prints per frame and
never copies the payload to the host, so a GPU stream is drained on device.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Mapping

from shmpipeline.config import SharedMemoryConfig, SinkConfig
from shmpipeline.errors import ConfigValidationError
from shmpipeline.sink import Sink

_DEFAULT_SAMPLE_WINDOW = 8192


def _percentile(ordered: list[float], fraction: float) -> float:
    """Return the ``fraction`` percentile of a pre-sorted, non-empty list."""
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


class NullSink(Sink):
    """Consume and discard every payload, tracking consume-time percentiles."""

    kind = "null.sink"

    @classmethod
    def validate_config(
        cls,
        config: SinkConfig,
        shared_memory: Mapping[str, SharedMemoryConfig],
    ) -> None:
        """Accept any storage; validate the optional device-delay parameter."""
        if config.stream not in shared_memory:
            raise ConfigValidationError(
                f"sink {config.name!r} references unknown stream "
                f"{config.stream!r}"
            )
        delay = config.parameters.get("device_delay_s", 0.0)
        if not isinstance(delay, (int, float)) or delay < 0.0:
            raise ConfigValidationError(
                f"sink {config.name!r} device_delay_s must be a "
                "non-negative number"
            )
        window = config.parameters.get("sample_window", _DEFAULT_SAMPLE_WINDOW)
        if not isinstance(window, int) or window <= 0:
            raise ConfigValidationError(
                f"sink {config.name!r} sample_window must be a positive integer"
            )

    def __init__(self, context) -> None:
        super().__init__(context)
        parameters = context.config.parameters
        self._device_delay_s = float(parameters.get("device_delay_s", 0.0))
        window = int(parameters.get("sample_window", _DEFAULT_SAMPLE_WINDOW))
        self._lock = threading.Lock()
        self._samples: deque[float] = deque(maxlen=window)
        self._consumed = 0

    def consume(self, value: Any) -> None:
        """Discard ``value`` after an optional fake device delay."""
        del value
        started = time.perf_counter()
        if self._device_delay_s > 0.0:
            time.sleep(self._device_delay_s)
        elapsed = time.perf_counter() - started
        with self._lock:
            self._consumed += 1
            self._samples.append(elapsed)

    def plugin_metrics(self) -> dict[str, Any]:
        """Return consumed count and consume-time percentiles in microseconds."""
        with self._lock:
            consumed = self._consumed
            ordered = sorted(self._samples)
        return {
            "consumed": consumed,
            "device_delay_us": self._device_delay_s * 1e6,
            "consume_us": {
                "p50": _percentile(ordered, 0.50) * 1e6,
                "p90": _percentile(ordered, 0.90) * 1e6,
                "p99": _percentile(ordered, 0.99) * 1e6,
                "p99_9": _percentile(ordered, 0.999) * 1e6,
                "max": (ordered[-1] * 1e6) if ordered else 0.0,
            },
        }
