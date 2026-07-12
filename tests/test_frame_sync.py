"""Tests for frame_id propagation and the matching_frame_id barrier."""

from __future__ import annotations

import time

import numpy as np
import pytest

from shmpipeline import PipelineConfig, PipelineManager
from shmpipeline.errors import ConfigValidationError
from shmpipeline.runtime import _matching_frame_decision

pytestmark = [pytest.mark.unit, pytest.mark.integration]


def test_matching_frame_decision_all_equal():
    target, skew_gap, laggards = _matching_frame_decision(
        {"a": 5, "b": 5, "c": 5}
    )
    assert target == 5
    assert skew_gap == 0
    assert laggards == ()


def test_matching_frame_decision_identifies_laggards():
    target, skew_gap, laggards = _matching_frame_decision(
        {"a": 8, "b": 6, "c": 8, "d": 5}
    )
    assert target == 8
    assert skew_gap == 3
    assert set(laggards) == {"b", "d"}


def test_matching_frame_id_requires_all_new_policy():
    with pytest.raises(ConfigValidationError, match="requires trigger_policy"):
        PipelineConfig.from_dict(
            {
                "shared_memory": [
                    {"name": "c0", "shape": [4], "dtype": "float32"},
                    {"name": "c1", "shape": [4], "dtype": "float32"},
                    {"name": "j", "shape": [8], "dtype": "float32"},
                ],
                "kernels": [
                    {
                        "name": "join",
                        "kind": "cpu.concatenate",
                        "inputs": ["c0", "c1"],
                        "trigger_policy": "any_new",
                        "output": "j",
                        "parameters": {"axis": 0},
                        "synchronization": {"mode": "matching_frame_id"},
                    }
                ],
            }
        )


def _matching_barrier_config(shm_prefix):
    return PipelineConfig.from_dict(
        {
            "shared_memory": [
                {"name": f"{shm_prefix}_c0", "shape": [4], "dtype": "float32"},
                {"name": f"{shm_prefix}_c1", "shape": [4], "dtype": "float32"},
                {"name": f"{shm_prefix}_j", "shape": [8], "dtype": "float32"},
            ],
            "kernels": [
                {
                    "name": "join",
                    "kind": "cpu.concatenate",
                    "inputs": [f"{shm_prefix}_c0", f"{shm_prefix}_c1"],
                    "trigger_policy": "all_new",
                    "output": f"{shm_prefix}_j",
                    "parameters": {"axis": 0},
                    "synchronization": {
                        "mode": "matching_frame_id",
                        "max_skew_generations": 64,
                    },
                    "read_timeout": 0.2,
                }
            ],
        }
    )


def _wait_for_count(stream, target, *, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if stream.count >= target:
            return True
        time.sleep(5e-3)
    return False


def test_matching_barrier_never_combines_unequal_tokens(shm_prefix):
    config = _matching_barrier_config(shm_prefix)
    manager = PipelineManager(config)
    try:
        manager.build()
        manager.start()
        c0 = manager.get_stream(f"{shm_prefix}_c0")
        c1 = manager.get_stream(f"{shm_prefix}_c1")
        out = manager.get_stream(f"{shm_prefix}_j")
        payload = np.ones(4, dtype=np.float32)

        # Matching tokens (1, 1) -> the barrier fires and propagates token 1.
        c0.write(payload, frame_id=1)
        c1.write(payload, frame_id=1)
        assert _wait_for_count(out, 1)
        manager.raise_if_failed()
        assert out.frame_id == 1

        # Advance only c0 to token 2: tokens (2, 1) never match, so the
        # barrier must not produce a second output.
        c0.write(payload, frame_id=2)
        time.sleep(0.3)
        manager.raise_if_failed()
        assert out.count == 1

        # Align c1 to token 2: the barrier fires once more with token 2.
        c1.write(payload, frame_id=2)
        assert _wait_for_count(out, 2)
        manager.raise_if_failed()
        assert out.frame_id == 2
    finally:
        manager.shutdown(force=True)


def test_matching_barrier_drops_older_and_reports_skew(shm_prefix):
    config = _matching_barrier_config(shm_prefix)
    manager = PipelineManager(config)
    try:
        manager.build()
        manager.start()
        c0 = manager.get_stream(f"{shm_prefix}_c0")
        c1 = manager.get_stream(f"{shm_prefix}_c1")
        out = manager.get_stream(f"{shm_prefix}_j")
        payload = np.ones(4, dtype=np.float32)

        # c0 races ahead by three generations while c1 lags at token 1.
        c1.write(payload, frame_id=1)
        for token in (2, 3, 4):
            c0.write(payload, frame_id=token)
        time.sleep(0.2)
        manager.raise_if_failed()
        # Nothing combined yet: c1 never produced tokens 2-4.
        assert out.count == 0

        # c1 jumps straight to token 4 (older generations were dropped).
        c1.write(payload, frame_id=4)
        assert _wait_for_count(out, 1)
        manager.raise_if_failed()
        assert out.frame_id == 4

        metrics = manager.status()["metrics"].get("join", {})
        assert metrics.get("frame_sync_skew_events", 0) >= 1
        assert metrics.get("frame_sync_timeouts", 0) == 0
    finally:
        manager.shutdown(force=True)
