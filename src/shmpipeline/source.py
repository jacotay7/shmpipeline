"""Source plugin abstractions executed by the manager thread runtime."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from threading import Event
from typing import Any, Mapping

from shmpipeline.config import SharedMemoryConfig, SourceConfig
from shmpipeline.errors import ConfigValidationError


@dataclass(frozen=True)
class SourceContext:
    """Static information available to one source instance."""

    config: SourceConfig
    shared_memory: Mapping[str, SharedMemoryConfig]
    auxiliary_streams: Mapping[str, Any] = field(default_factory=dict)

    @property
    def stream_spec(self) -> SharedMemoryConfig:
        """Return the shared-memory specification written by the source."""
        return self.shared_memory[self.config.stream]

    @property
    def auxiliary_specs(self) -> dict[str, SharedMemoryConfig]:
        """Return auxiliary shared-memory specs keyed by alias."""
        return {
            binding.alias: self.shared_memory[binding.name]
            for binding in self.config.auxiliary
        }


class Source(ABC):
    """Base class for source plugins managed by the runtime.

    Sources are manager-owned thread plugins. Implementations normally
    override :meth:`validate_config` for parameter checks, optionally perform
    setup and teardown in :meth:`open` and :meth:`close`, and implement
    :meth:`read` to return the next payload for the bound stream.
    """

    kind = "source.base"
    storage = "cpu"

    def __init__(self, context: SourceContext) -> None:
        self.context = context
        self._stop_event: Event | None = None
        self._pause_event: Event | None = None

    @classmethod
    def validate_config(
        cls,
        config: SourceConfig,
        shared_memory: Mapping[str, SharedMemoryConfig],
    ) -> None:
        """Validate storage compatibility before the source is built."""
        if shared_memory[config.stream].storage != cls.storage:
            raise ConfigValidationError(
                f"source {config.name!r} of kind {cls.kind!r} requires "
                f"{cls.storage} shared memory for {config.stream!r}"
            )
        for binding in config.auxiliary:
            if shared_memory[binding.name].storage != cls.storage:
                raise ConfigValidationError(
                    f"source {config.name!r} of kind {cls.kind!r} requires "
                    f"{cls.storage} shared memory for auxiliary {binding.name!r}"
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
        """Return whether the manager has requested that the source stop."""
        return bool(self._stop_event is not None and self._stop_event.is_set())

    def paused(self) -> bool:
        """Return whether the manager is currently paused."""
        return bool(
            self._pause_event is not None and self._pause_event.is_set()
        )

    def wait(self, duration: float) -> bool:
        """Wait cooperatively for up to ``duration`` seconds.

        Returns ``True`` when a stop was requested before the wait completed.
        This is intended for sources that want a simple cadence without having
        to manage their own stop-event plumbing.
        """
        deadline = time.monotonic() + max(0.0, float(duration))
        while not self.stop_requested():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return False
            time.sleep(min(remaining, 0.05))
        return True

    def open(self) -> None:
        """Prepare the source before the runtime thread starts."""

    def read_auxiliary(self, alias: str, *, timeout: float = 0.01) -> Any | None:
        """Return one stable auxiliary payload when that stream has data."""
        auxiliary_streams = getattr(self.context, "auxiliary_streams", {})
        stream = auxiliary_streams.get(alias)
        if stream is None or stream.count <= 0:
            return None
        try:
            with stream.locked(timeout=timeout):
                if stream.count <= 0:
                    return None
                return stream.read(safe=True)
        except TimeoutError:
            return None

    @abstractmethod
    def read(self) -> Any | None:
        """Return the next payload for the configured stream.

        Returning ``None`` indicates that no new payload is currently ready and
        the runtime should sleep for the configured poll interval before trying
        again.
        """

    def close(self) -> None:
        """Release any external resources owned by the source."""
