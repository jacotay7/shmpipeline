from __future__ import annotations

import uuid

import pyshmem
import pytest


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