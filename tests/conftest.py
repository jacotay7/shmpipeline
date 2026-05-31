from __future__ import annotations

import gc
import uuid

import pyshmem
import pytest


@pytest.fixture(autouse=True)
def _gc_between_tests():
    """Run garbage collection before each test to reduce memory pressure.

    Tests that spawn many worker processes accumulate CUDA IPC handles and
    other OS resources.  Explicit GC between tests reduces the chance of
    hitting kernel limits (e.g. max_map_count) when running the full suite.
    """
    gc.collect()
    yield
    gc.collect()


@pytest.fixture
def shm_prefix():
    prefix = f"shmpipeline_{uuid.uuid4().hex}"
    yield prefix
    for suffix in (
        "input",
        "output",
        "matrix",
        "offset",
        "image",
        "centroids",
        "centroid_offset",
        "corrected",
        "flattened",
        "reconstructor",
        "affine_offset",
        "open_loop",
        "command",
    ):
        try:
            pyshmem.unlink(f"{prefix}_{suffix}")
        except FileNotFoundError:
            pass
