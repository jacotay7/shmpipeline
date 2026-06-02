"""Tests for shared-memory teardown via the public pyshmem.unlink_quiet API."""

from __future__ import annotations

import numpy as np
import pyshmem
import pytest

from shmpipeline.shm_cleanup import close_stream, unlink_stream_name

pytestmark = pytest.mark.unit


def test_pyshmem_unlink_quiet_is_public():
    assert callable(pyshmem.unlink_quiet)


def test_pyshmem_unlink_quiet_noop_when_stream_missing():
    """unlink_quiet must not raise when the stream does not exist."""
    pyshmem.unlink_quiet("shmpipeline_test_nonexistent_stream_xyz")


def test_pyshmem_unlink_quiet_removes_stream(shm_prefix):
    name = f"{shm_prefix}_quiet_test"
    stream = pyshmem.create(name, shape=(4,), dtype=np.float32)
    stream.close()
    pyshmem.unlink_quiet(name)
    # Second call must also succeed (idempotent).
    pyshmem.unlink_quiet(name)


def test_close_stream_delegates_to_pyshmem(monkeypatch):
    """close_stream calls pyshmem.unlink_quiet with the stream's name."""
    import shmpipeline.shm_cleanup as shm_cleanup

    calls: list[str] = []

    class _FakeStream:
        name = "test_stream_abc"

        def close(self):
            pass

    monkeypatch.setattr(shm_cleanup.pyshmem, "unlink_quiet", calls.append)
    close_stream(_FakeStream(), unlink=True)
    assert calls == ["test_stream_abc"]


def test_close_stream_no_unlink_skips_pyshmem(monkeypatch):
    import shmpipeline.shm_cleanup as shm_cleanup

    calls: list[str] = []

    class _FakeStream:
        name = "test_stream_abc"

        def close(self):
            pass

    monkeypatch.setattr(shm_cleanup.pyshmem, "unlink_quiet", calls.append)
    close_stream(_FakeStream(), unlink=False)
    assert calls == []


def test_unlink_stream_name_delegates_to_pyshmem(monkeypatch):
    import shmpipeline.shm_cleanup as shm_cleanup

    calls: list[str] = []
    monkeypatch.setattr(shm_cleanup.pyshmem, "unlink_quiet", calls.append)
    unlink_stream_name("some_stream")
    assert calls == ["some_stream"]


def test_close_stream_unlinks_even_after_close_raises(monkeypatch):
    """unlink_quiet is still called even when close() raises."""
    import shmpipeline.shm_cleanup as shm_cleanup

    calls: list[str] = []

    class _BrokenStream:
        name = "broken_stream"

        def close(self):
            raise RuntimeError("close failed")

    monkeypatch.setattr(shm_cleanup.pyshmem, "unlink_quiet", calls.append)
    with pytest.raises(RuntimeError, match="close failed"):
        close_stream(_BrokenStream(), unlink=True)
    assert calls == ["broken_stream"]
