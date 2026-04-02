from __future__ import annotations

import time

import numpy as np
import pytest

from shmpipeline import PipelineConfig, PipelineManager
from shmpipeline.synthetic import (
    SyntheticInputConfig,
    SyntheticPatternGenerator,
)

try:
    import torch
except Exception:  # pragma: no cover - exercised when torch is unavailable
    torch = None


pytestmark = [pytest.mark.unit, pytest.mark.integration]

CUDA_AVAILABLE = torch is not None and torch.cuda.is_available()


def _wait_for_next_write(stream, previous_count: int, *, timeout: float = 2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if stream.count > previous_count:
            return stream.read()
        time.sleep(1e-4)
    raise TimeoutError(f"timed out waiting for a new write on {stream.name!r}")


def _to_host_array(value):
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()


def _make_scale_config(shm_prefix: str, *, storage: str) -> PipelineConfig:
    shared_memory = [
        {
            "name": f"{shm_prefix}_input",
            "shape": [4],
            "dtype": "float32",
            "storage": storage,
            **({"gpu_device": "cuda:0"} if storage == "gpu" else {}),
        },
        {
            "name": f"{shm_prefix}_output",
            "shape": [4],
            "dtype": "float32",
            "storage": storage,
            **({"gpu_device": "cuda:0"} if storage == "gpu" else {}),
        },
    ]
    return PipelineConfig.from_dict(
        {
            "shared_memory": shared_memory,
            "kernels": [
                {
                    "name": "stage",
                    "kind": f"{storage}.scale",
                    "input": f"{shm_prefix}_input",
                    "output": f"{shm_prefix}_output",
                    "parameters": {"factor": 2.0},
                    "read_timeout": 0.1,
                }
            ],
        }
    )


def test_random_pattern_generator_is_deterministic_for_cpu():
    spec = SyntheticInputConfig(stream_name="input", pattern="random", seed=11)
    lhs = SyntheticPatternGenerator(
        spec,
        shape=(4,),
        dtype=np.float32,
        storage="cpu",
    )
    rhs = SyntheticPatternGenerator(
        spec,
        shape=(4,),
        dtype=np.float32,
        storage="cpu",
    )

    for _ in range(3):
        np.testing.assert_allclose(
            lhs.next_frame().copy(), rhs.next_frame().copy()
        )


def test_manager_synthetic_input_drives_cpu_pipeline(shm_prefix):
    manager = PipelineManager(_make_scale_config(shm_prefix, storage="cpu"))
    try:
        manager.build()
        manager.start()
        output_stream = manager.get_stream(f"{shm_prefix}_output")
        baseline = output_stream.count
        manager.start_synthetic_input(
            SyntheticInputConfig(
                stream_name=f"{shm_prefix}_input",
                pattern="ramp",
                rate_hz=200.0,
                amplitude=1.0,
                seed=3,
            )
        )

        observed = _wait_for_next_write(output_stream, baseline, timeout=2.0)
        snapshot = manager.runtime_snapshot()
        deadline = time.monotonic() + 2.0
        while (
            time.monotonic() < deadline
            and snapshot["workers"]["stage"].get("frames_processed", 0) < 1
        ):
            time.sleep(0.05)
            snapshot = manager.runtime_snapshot()

        assert observed.shape == (4,)
        assert snapshot["graph"]["source_streams"] == (f"{shm_prefix}_input",)
        assert snapshot["workers"]["stage"].get("frames_processed", 0) >= 1
        assert (
            snapshot["synthetic_sources"][f"{shm_prefix}_input"]["alive"]
            is True
        )

        manager.stop_synthetic_input(f"{shm_prefix}_input")
        assert manager.synthetic_input_status() == {}
    finally:
        manager.shutdown(force=True)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_manager_synthetic_input_drives_gpu_pipeline(shm_prefix):
    manager = PipelineManager(_make_scale_config(shm_prefix, storage="gpu"))
    try:
        manager.build()
        manager.start()
        output_stream = manager.get_stream(f"{shm_prefix}_output")
        baseline = output_stream.count
        manager.start_synthetic_input(
            SyntheticInputConfig(
                stream_name=f"{shm_prefix}_input",
                pattern="sine",
                rate_hz=200.0,
                amplitude=0.5,
                offset=1.0,
                seed=5,
            )
        )

        observed = _to_host_array(
            _wait_for_next_write(output_stream, baseline, timeout=2.0)
        )
        snapshot = manager.runtime_snapshot()

        assert observed.shape == (4,)
        assert (
            snapshot["synthetic_sources"][f"{shm_prefix}_input"]["alive"]
            is True
        )
        assert output_stream.count > baseline
    finally:
        manager.shutdown(force=True)
