"""Pipeline state model."""

from __future__ import annotations

from enum import Enum


class PipelineState(str, Enum):
    """States managed by :class:`shmpipeline.manager.PipelineManager`."""

    INITIALIZED = "initialized"
    BUILT = "built"
    RUNNING = "running"
    PAUSED = "paused"
    FAILED = "failed"
    STOPPED = "stopped"
