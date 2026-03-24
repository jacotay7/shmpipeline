from __future__ import annotations

import uuid

import pyshmem
import pytest


@pytest.fixture
def shm_prefix():
    prefix = f"shmpipeline_{uuid.uuid4().hex}"
    yield prefix
    for suffix in ("input", "output", "matrix", "offset"):
        try:
            pyshmem.unlink(f"{prefix}_{suffix}")
        except FileNotFoundError:
            pass