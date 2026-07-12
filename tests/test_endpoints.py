"""Tests for the bundled built-in source and sink plugins."""

from __future__ import annotations

import time

import numpy as np
import pytest

from shmpipeline import PipelineConfig, PipelineManager, get_default_registry
from shmpipeline.errors import ConfigValidationError
from shmpipeline.sinks.null_sink import NullSink, _percentile

pytestmark = [pytest.mark.unit, pytest.mark.integration]


def _endpoint_config(shm_prefix, *, rate_hz=200.0, device_delay_s=0.0):
    return PipelineConfig.from_dict(
        {
            "shared_memory": [
                {
                    "name": f"{shm_prefix}_src",
                    "shape": [8],
                    "dtype": "float32",
                    "storage": "cpu",
                },
                {
                    "name": f"{shm_prefix}_out",
                    "shape": [8],
                    "dtype": "float32",
                    "storage": "cpu",
                },
            ],
            "sources": [
                {
                    "name": "gen",
                    "kind": "synthetic.array",
                    "stream": f"{shm_prefix}_src",
                    "parameters": {
                        "pattern": "constant",
                        "constant": 3.0,
                        "rate_hz": rate_hz,
                    },
                }
            ],
            "kernels": [
                {
                    "name": "scale_stage",
                    "kind": "cpu.scale",
                    "input": f"{shm_prefix}_src",
                    "output": f"{shm_prefix}_out",
                    "parameters": {"factor": 2.0},
                    "read_timeout": 0.1,
                }
            ],
            "sinks": [
                {
                    "name": "drain",
                    "kind": "null.sink",
                    "stream": f"{shm_prefix}_out",
                    "parameters": {"device_delay_s": device_delay_s},
                    "read_timeout": 0.1,
                }
            ],
        }
    )


def test_builtin_endpoints_are_registered():
    registry = get_default_registry()
    assert "synthetic.array" in registry.source_kinds()
    assert "null.sink" in registry.sink_kinds()


def test_synthetic_array_and_null_sink_run_end_to_end(shm_prefix):
    config = _endpoint_config(shm_prefix)
    manager = PipelineManager(config)
    try:
        manager.build()
        manager.start()
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            manager.poll_events()
            manager.raise_if_failed()
            if manager.status()["sinks"]["drain"]["frames_consumed"] > 20:
                break
            time.sleep(1e-2)
        status = manager.status()
        # The scale kernel doubled the synthetic constant of 3.0.
        np.testing.assert_allclose(
            manager.get_stream(f"{shm_prefix}_out").read(),
            np.full(8, 6.0, dtype=np.float32),
        )
        source = status["sources"]["gen"]
        sink = status["sinks"]["drain"]
        assert source["frames_written"] > 20
        assert sink["frames_consumed"] > 20
        assert sink["missed_writes"] >= 0
        metrics = sink["plugin_metrics"]
        assert metrics["consumed"] == sink["frames_consumed"]
        assert set(metrics["consume_us"]) == {
            "p50",
            "p90",
            "p99",
            "p99_9",
            "max",
        }
        assert metrics["consume_us"]["max"] >= metrics["consume_us"]["p50"]
    finally:
        manager.shutdown(force=True)


def test_synthetic_array_rejects_non_positive_rate(shm_prefix):
    config = _endpoint_config(shm_prefix, rate_hz=200.0)
    shared = config.shared_memory_by_name
    bad = config.sources[0].__class__(
        name="gen",
        kind="synthetic.array",
        stream=f"{shm_prefix}_src",
        parameters={"rate_hz": 0.0},
    )
    registry = get_default_registry()
    with pytest.raises(ConfigValidationError, match="rate_hz"):
        registry.validate_source(bad, shared)


def test_null_sink_rejects_negative_device_delay(shm_prefix):
    config = _endpoint_config(shm_prefix)
    shared = config.shared_memory_by_name
    bad = config.sinks[0].__class__(
        name="drain",
        kind="null.sink",
        stream=f"{shm_prefix}_out",
        parameters={"device_delay_s": -1.0},
    )
    registry = get_default_registry()
    with pytest.raises(ConfigValidationError, match="device_delay_s"):
        registry.validate_sink(bad, shared)


def test_null_sink_percentiles_are_monotonic():
    ordered = [float(value) for value in range(100)]
    assert _percentile(ordered, 0.0) == 0.0
    assert _percentile(ordered, 1.0) == 99.0
    assert _percentile([], 0.5) == 0.0
    assert _percentile([7.0], 0.9) == 7.0
    assert (
        _percentile(ordered, 0.5)
        <= _percentile(ordered, 0.9)
        <= _percentile(ordered, 0.99)
    )


def test_null_sink_plugin_metrics_track_consume_delay():
    from shmpipeline.config import SinkConfig
    from shmpipeline.sink import SinkContext

    spec = {
        "name": "s",
        "shape": (2,),
        "dtype": np.dtype("float32"),
        "storage": "cpu",
    }
    from shmpipeline.config import SharedMemoryConfig

    shared = {"s": SharedMemoryConfig(**spec)}
    sink_config = SinkConfig(
        name="drain",
        kind="null.sink",
        stream="s",
        parameters={"device_delay_s": 0.001},
    )
    sink = NullSink(SinkContext(config=sink_config, shared_memory=shared))
    for _ in range(5):
        sink.consume(np.zeros(2, dtype=np.float32))
    metrics = sink.plugin_metrics()
    assert metrics["consumed"] == 5
    # Each consume slept ~1 ms, so p50 should be at least several hundred us.
    assert metrics["consume_us"]["p50"] >= 500.0
