"""Helpers for releasing pyshmem-backed streams without noisy teardown."""

from __future__ import annotations

import os
import sys
from multiprocessing import shared_memory
from typing import Any

import pyshmem  # type: ignore[import-not-found]

try:
    import pyshmem._shared as pyshmem_shared  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - exercised if internals move
    pyshmem_shared = None


def close_stream(stream: Any, *, unlink: bool) -> None:
    """Close one stream and optionally unlink its underlying segments.

    On Linux, pyshmem's public unlink helper currently goes through a
    register/unregister path that can trigger noisy `resource_tracker`
    tracebacks during successful shutdown. When the current process already has
    the segment names, use direct POSIX unlink calls instead.
    """
    if not unlink:
        stream.close()
        return

    if _can_directly_unlink_posix_segments():
        segment_names = _stream_segment_names(stream)
        lock_path = _stream_lock_path(stream)
        _drop_local_gpu_cache(stream.name)
        try:
            stream.close()
        finally:
            for segment_name in segment_names:
                _safe_posix_shm_unlink(segment_name)
            _safe_remove(lock_path)
        return

    stream_name = stream.name
    try:
        stream.close()
    finally:
        pyshmem.unlink(stream_name)


def unlink_stream_name(stream_name: str) -> None:
    """Unlink one stream by name, including stale partial creations."""
    if _can_directly_unlink_posix_segments() and pyshmem_shared is not None:
        _drop_local_gpu_cache(stream_name)
        segment_names = _segment_names_for_stream_name(stream_name)
        lock_path = _lock_path_for_stream_name(stream_name)
        for segment_name in segment_names:
            _safe_posix_shm_unlink(segment_name)
        _safe_remove(lock_path)
        return

    pyshmem.unlink(stream_name)


def _can_directly_unlink_posix_segments() -> bool:
    return bool(
        os.name != "nt"
        and sys.platform != "darwin"
        and hasattr(shared_memory, "_posixshmem")
        and hasattr(shared_memory._posixshmem, "shm_unlink")
    )


def _stream_segment_names(stream: Any) -> tuple[str, ...]:
    names: list[str] = []
    for attr_name in ("_data_shm", "_metadata_shm", "_gpu_handle_shm"):
        segment = getattr(stream, attr_name, None)
        segment_name = getattr(segment, "_name", None)
        if segment_name:
            names.append(_normalize_segment_name(segment_name))

    names.extend(_segment_names_for_stream_name(stream.name))
    return tuple(dict.fromkeys(names))


def _segment_names_for_stream_name(stream_name: str) -> tuple[str, ...]:
    names: list[str] = []
    if pyshmem_shared is not None:
        for helper_name in ("_data_name", "_metadata_name"):
            helper = getattr(pyshmem_shared, helper_name, None)
            if helper is None:
                continue
            names.append(_normalize_segment_name(helper(stream_name)))
        helper = getattr(pyshmem_shared, "_gpu_handle_name", None)
        if helper is not None:
            names.append(_normalize_segment_name(helper(stream_name)))

    return tuple(dict.fromkeys(names))


def _stream_lock_path(stream: Any) -> str | None:
    lock_state = getattr(stream, "_lock_state", None)
    path = getattr(lock_state, "path", None)
    if path:
        return path
    return _lock_path_for_stream_name(stream.name)


def _lock_path_for_stream_name(stream_name: str) -> str | None:
    if pyshmem_shared is None:
        return None
    helper = getattr(pyshmem_shared, "_lock_path", None)
    if helper is None:
        return None
    return str(helper(stream_name))


def _normalize_segment_name(name: str) -> str:
    return name if name.startswith("/") else f"/{name}"


def _safe_posix_shm_unlink(name: str) -> None:
    try:
        shared_memory._posixshmem.shm_unlink(name)
    except FileNotFoundError:
        pass


def _safe_remove(path: str | None) -> None:
    if not path:
        return
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _drop_local_gpu_cache(stream_name: str) -> None:
    local_tensors = getattr(pyshmem_shared, "_LOCAL_GPU_TENSORS", None)
    if isinstance(local_tensors, dict):
        local_tensors.pop(stream_name, None)
