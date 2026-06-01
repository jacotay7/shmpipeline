"""Helpers for releasing pyshmem-backed streams without noisy teardown."""

from __future__ import annotations

from typing import Any

import pyshmem


def close_stream(stream: Any, *, unlink: bool) -> None:
    """Close one stream and optionally unlink its underlying segments.

    Uses pyshmem's public API to avoid coupling to internal segment naming.
    Unlinking is performed even if close() raises, so streams are always
    cleaned up on error paths.
    """
    if not unlink:
        stream.close()
        return
    name = stream.name
    try:
        stream.close()
    finally:
        pyshmem.unlink_quiet(name)


def unlink_stream_name(stream_name: str) -> None:
    """Unlink one stream by name, including stale partial creations."""
    pyshmem.unlink_quiet(stream_name)
