"""Built-in source plugins bundled with shmpipeline."""

from __future__ import annotations

from shmpipeline.sources.array_source import SyntheticArraySource
from shmpipeline.sources.frame_set_source import SyntheticFrameSetSource

__all__ = ["SyntheticArraySource", "SyntheticFrameSetSource"]
