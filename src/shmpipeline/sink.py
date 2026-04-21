"""Sink plugin abstractions executed by the manager thread runtime."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import Event
from typing import Any, Mapping

from shmpipeline.config import SharedMemoryConfig, SinkConfig
from shmpipeline.errors import ConfigValidationError


@dataclass(frozen=True)
class SinkContext:
    """Static information available to one sink instance."""

    config: SinkConfig
    shared_memory: Mapping[str, SharedMemoryConfig]

    @property
    def stream_spec(self) -> SharedMemoryConfig:
        """Return the shared-memory specification consumed by the sink."""
        return self.shared_memory[self.config.stream]


class Sink(ABC):
    """Base class for sink plugins managed by the runtime.

    Sinks are manager-owned thread plugins. Implementations normally override
    :meth:`validate_config` for parameter checks, optionally perform setup and
    teardown in :meth:`open` and :meth:`close`, and implement :meth:`consume`
    to handle payloads read from the bound stream.
    """

    kind = "sink.base"
    storage = "cpu"

    def __init__(self, context: SinkContext) -> None:
        self.context = context
        self._stop_event: Event | None = None
        self._pause_event: Event | None = None

    @classmethod
    def validate_config(
        cls,
        config: SinkConfig,
        shared_memory: Mapping[str, SharedMemoryConfig],
    ) -> None:
        """Validate storage compatibility before the sink is built."""
        if shared_memory[config.stream].storage != cls.storage:
            raise ConfigValidationError(
                f"sink {config.name!r} of kind {cls.kind!r} requires "
                f"{cls.storage} shared memory for {config.stream!r}"
            )

    def _attach_runtime_events(
        self,
        *,
        stop_event: Event,
        pause_event: Event,
    ) -> None:
        self._stop_event = stop_event
        self._pause_event = pause_event

    def stop_requested(self) -> bool:
        """Return whether the manager has requested that the sink stop."""
        return bool(self._stop_event is not None and self._stop_event.is_set())

    def paused(self) -> bool:
        """Return whether the manager is currently paused."""
        return bool(
            self._pause_event is not None and self._pause_event.is_set()
        )

    def wait(self, duration: float) -> bool:
        """Wait cooperatively for up to ``duration`` seconds.

        Returns ``True`` when a stop was requested before the wait completed.
        This is useful for sinks that batch external I/O on a simple cadence.
        """
        deadline = time.monotonic() + max(0.0, float(duration))
        while not self.stop_requested():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return False
            time.sleep(min(remaining, 0.05))
        return True

    def open(self) -> None:
        """Prepare the sink before the runtime thread starts."""

    @abstractmethod
    def consume(self, value: Any) -> None:
        """Handle one payload read from the configured stream."""

    def close(self) -> None:
        """Release any external resources owned by the sink."""
