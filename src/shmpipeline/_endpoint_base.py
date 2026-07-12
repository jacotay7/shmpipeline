"""Shared base for Source and Sink plugin abstractions."""

from __future__ import annotations

import time
from threading import Event
from typing import Any


class _EndpointBase:
    """Common runtime behaviour shared by Source and Sink plugins.

    Both source and sink plugins are driven by a manager-owned background
    thread.  They share identical plumbing for stop/pause signalling,
    cooperative waits, and auxiliary-stream reads.  This mixin holds that
    shared implementation so each subclass only declares what is unique.
    """

    def __init__(self) -> None:
        self._stop_event: Event | None = None
        self._pause_event: Event | None = None

    def _attach_runtime_events(
        self,
        *,
        stop_event: Event,
        pause_event: Event,
    ) -> None:
        self._stop_event = stop_event
        self._pause_event = pause_event

    def plugin_metrics(self) -> dict[str, Any]:
        """Return plugin-specific metrics merged into the status snapshot.

        The default implementation reports nothing.  Endpoints override this to
        surface counters (consume-time percentiles, injected drops, generation
        skew, ...) that the generic controller cannot know about.  It is called
        from the controller thread's ``snapshot`` while holding no plugin lock,
        so implementations must be cheap and thread-safe.
        """
        return {}

    def stop_requested(self) -> bool:
        """Return whether the manager has requested that this endpoint stop."""
        return bool(self._stop_event is not None and self._stop_event.is_set())

    def paused(self) -> bool:
        """Return whether the manager is currently paused."""
        return bool(
            self._pause_event is not None and self._pause_event.is_set()
        )

    def wait(self, duration: float) -> bool:
        """Wait cooperatively for up to ``duration`` seconds.

        Returns ``True`` when a stop was requested before the wait completed.
        """
        deadline = time.monotonic() + max(0.0, float(duration))
        while not self.stop_requested():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return False
            time.sleep(min(remaining, 0.05))
        return True

    def read_auxiliary(
        self, alias: str, *, timeout: float = 0.01
    ) -> Any | None:
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
