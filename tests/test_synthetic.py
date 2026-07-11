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


# ---------------------------------------------------------------------------
# Pattern vs stream-dtype validation warnings
# ---------------------------------------------------------------------------


def test_synthetic_random_pattern_warns_for_integer_dtype():
    import warnings

    spec = SyntheticInputConfig(stream_name="s", pattern="random")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SyntheticPatternGenerator(
            spec, shape=(4,), dtype=np.int32, storage="cpu"
        )
    assert any(issubclass(w.category, UserWarning) for w in caught)
    assert any("random" in str(w.message) for w in caught)


def test_synthetic_sine_pattern_warns_for_integer_dtype():
    import warnings

    spec = SyntheticInputConfig(stream_name="s", pattern="sine")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SyntheticPatternGenerator(
            spec, shape=(4,), dtype=np.uint8, storage="cpu"
        )
    assert any(issubclass(w.category, UserWarning) for w in caught)


def test_synthetic_ramp_pattern_warns_for_integer_dtype():
    import warnings

    spec = SyntheticInputConfig(stream_name="s", pattern="ramp")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SyntheticPatternGenerator(
            spec, shape=(4,), dtype=np.int32, storage="cpu"
        )
    assert any(issubclass(w.category, UserWarning) for w in caught)


def test_synthetic_constant_pattern_no_warning_for_integer():
    import warnings

    spec = SyntheticInputConfig(stream_name="s", pattern="constant")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SyntheticPatternGenerator(
            spec, shape=(4,), dtype=np.int32, storage="cpu"
        )
    assert not [w for w in caught if issubclass(w.category, UserWarning)]


def test_synthetic_random_pattern_no_warning_for_float_dtype():
    import warnings

    spec = SyntheticInputConfig(stream_name="s", pattern="random")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SyntheticPatternGenerator(
            spec, shape=(4,), dtype=np.float32, storage="cpu"
        )
    assert not [w for w in caught if issubclass(w.category, UserWarning)]


# ---------------------------------------------------------------------------
# CPU pattern generation for every supported pattern
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pattern",
    ["constant", "random", "ramp", "sine", "impulse", "checkerboard"],
)
def test_synthetic_pattern_generator_next_frame_cpu(pattern):
    spec = SyntheticInputConfig(
        stream_name="s",
        pattern=pattern,
        constant=2.0,
        amplitude=1.0,
        offset=0.5,
        period=8.0,
        impulse_interval=2,
        seed=1,
    )
    generator = SyntheticPatternGenerator(
        spec, shape=(4,), dtype=np.float32, storage="cpu"
    )
    frame_a = generator.next_frame()
    frame_b = generator.next_frame()
    assert np.asarray(frame_a).shape == (4,)
    assert np.asarray(frame_b).shape == (4,)
    assert np.all(np.isfinite(np.asarray(frame_a)))


def test_synthetic_constant_pattern_is_deterministic():
    spec = SyntheticInputConfig(
        stream_name="s", pattern="constant", constant=7.0
    )
    generator = SyntheticPatternGenerator(
        spec, shape=(3,), dtype=np.float32, storage="cpu"
    )
    np.testing.assert_allclose(generator.next_frame(), [7.0, 7.0, 7.0])
    np.testing.assert_allclose(generator.next_frame(), [7.0, 7.0, 7.0])


def test_available_synthetic_patterns_lists_known_patterns():
    from shmpipeline.synthetic import available_synthetic_patterns

    patterns = available_synthetic_patterns()
    assert {
        "constant",
        "random",
        "ramp",
        "sine",
        "impulse",
        "checkerboard",
    } <= set(patterns)


def test_synthetic_checkerboard_pattern_alternates_2d_tiles():
    spec = SyntheticInputConfig(
        stream_name="s",
        pattern="checkerboard",
        amplitude=1.0,
        offset=0.0,
        period=2.0,
    )
    generator = SyntheticPatternGenerator(
        spec, shape=(4, 4), dtype=np.float32, storage="cpu"
    )
    frame = np.array(generator.next_frame())
    expected = np.array(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(frame, expected)
    assert set(np.unique(frame)) == {0.0, 1.0}


def test_synthetic_checkerboard_pattern_scrolls_with_frame_index():
    spec = SyntheticInputConfig(
        stream_name="s",
        pattern="checkerboard",
        amplitude=1.0,
        offset=0.0,
        period=2.0,
    )
    generator = SyntheticPatternGenerator(
        spec, shape=(4,), dtype=np.float32, storage="cpu"
    )
    frame_a = np.array(generator.next_frame())
    frame_b = np.array(generator.next_frame())
    np.testing.assert_allclose(frame_a, [0.0, 0.0, 1.0, 1.0])
    np.testing.assert_allclose(frame_b, [0.0, 1.0, 1.0, 0.0])


def test_synthetic_checkerboard_pattern_scalar_shape_blinks():
    spec = SyntheticInputConfig(stream_name="s", pattern="checkerboard")
    generator = SyntheticPatternGenerator(
        spec, shape=(), dtype=np.float32, storage="cpu"
    )
    assert float(generator.next_frame()) == pytest.approx(0.0)
    assert float(generator.next_frame()) == pytest.approx(1.0)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_synthetic_checkerboard_pattern_gpu_matches_cpu():
    spec = SyntheticInputConfig(
        stream_name="s",
        pattern="checkerboard",
        amplitude=1.0,
        offset=0.0,
        period=2.0,
    )
    cpu_generator = SyntheticPatternGenerator(
        spec, shape=(4, 4), dtype=np.float32, storage="cpu"
    )
    gpu_generator = SyntheticPatternGenerator(
        spec,
        shape=(4, 4),
        dtype=np.float32,
        storage="gpu",
        gpu_device="cuda:0",
    )
    cpu_frame = np.array(cpu_generator.next_frame())
    gpu_frame = gpu_generator.next_frame().detach().cpu().numpy()
    np.testing.assert_allclose(gpu_frame, cpu_frame)
