"""Unit tests for worker-runtime helpers (exercised in-process).

The full worker loop runs in spawned subprocesses, so these tests target the
pure helper functions directly with lightweight fake streams to keep runtime
behaviour covered without process spawning.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pytest

from shmpipeline import runtime
from shmpipeline.config import KernelConfig

pytestmark = pytest.mark.unit


class _FakeStream:
    """Minimal stand-in for a pyshmem stream usable inside a worker."""

    def __init__(self, name, array, *, count=0, gpu_enabled=False):
        self.name = name
        self._array = np.asarray(array)
        self.count = count
        self.gpu_enabled = gpu_enabled
        self.writes_started = 0
        self.writes_finished = 0

    def read(self, safe=False):
        return self._array

    @contextmanager
    def locked(self, timeout=None):
        yield self

    def _mark_write_started(self):
        self.writes_started += 1

    def _finish_write(self):
        self.writes_finished += 1


def test_wait_for_trigger_returns_when_count_advances():
    stream = _FakeStream("trig", [1.0], count=5)
    assert runtime._wait_for_trigger(stream, 4, timeout=1.0) == 5


def test_wait_for_trigger_times_out_returns_none():
    stream = _FakeStream("trig", [1.0], count=3)
    assert (
        runtime._wait_for_trigger(stream, 3, timeout=0.05, poll_interval=1e-3)
        is None
    )


def test_compute_rolling_throughput_hz():
    from collections import deque

    assert runtime._compute_rolling_throughput_hz(deque([1.0])) == 0.0
    times = deque([0.0, 1.0, 2.0])
    assert runtime._compute_rolling_throughput_hz(times) == pytest.approx(1.0)


def test_compute_rolling_exec_metrics():
    from collections import deque

    assert runtime._compute_rolling_exec_metrics(deque()) == (0.0, 0.0, 0.0)
    avg_ms, avg_us, jitter_us = runtime._compute_rolling_exec_metrics(
        deque([0.001, 0.003])
    )
    assert avg_ms == pytest.approx(2.0)
    assert avg_us == pytest.approx(2000.0)
    assert jitter_us == pytest.approx(1000.0)


def test_read_worker_input_uses_safe_read_for_gpu():
    calls = {}

    class _Probe(_FakeStream):
        def read(self, safe=False):
            calls["safe"] = safe
            return self._array

    gpu_stream = _Probe("g", [1.0], gpu_enabled=True)
    runtime._read_worker_input(gpu_stream)
    assert calls["safe"] is True
    cpu_stream = _Probe("c", [1.0], gpu_enabled=False)
    runtime._read_worker_input(cpu_stream)
    assert calls["safe"] is False


def test_read_locked_inputs_and_outputs_reads_all_streams():
    config = KernelConfig.from_dict(
        {
            "name": "k",
            "kind": "cpu.copy",
            "input": "in",
            "outputs": ["out_a", "out_b"],
        }
    )
    trigger = _FakeStream("in", [1.0, 2.0], count=7)
    out_a = _FakeStream("out_a", np.zeros(2, dtype=np.float32))
    out_b = _FakeStream("out_b", np.zeros(2, dtype=np.float32))
    output_streams = {"out_a": out_a, "out_b": out_b}

    count, trigger_input, aux, views = runtime._read_locked_inputs_and_outputs(
        config, trigger, {}, output_streams
    )
    assert count == 7
    np.testing.assert_array_equal(trigger_input, [1.0, 2.0])
    assert aux == {}
    assert set(views) == {"out_a", "out_b"}


def test_compute_and_publish_outputs_cpu_zero_copy_multi_output():
    from shmpipeline.kernel import Kernel

    class _Splitter(Kernel):
        kind = "test.rt_splitter"
        storage = "cpu"
        output_arity = 2

        def __init__(self):  # bypass context for this unit test
            pass

        def compute_into_multiple(self, trigger_input, outputs, aux):
            outputs[0][...] = trigger_input * 2.0
            outputs[1][...] = trigger_input + 1.0

    ordered = ("out_a", "out_b")
    out_a = _FakeStream("out_a", np.zeros(2, dtype=np.float32))
    out_b = _FakeStream("out_b", np.zeros(2, dtype=np.float32))
    views = {
        "out_a": np.zeros(2, dtype=np.float32),
        "out_b": np.zeros(2, dtype=np.float32),
    }
    streams = {"out_a": out_a, "out_b": out_b}
    runtime._compute_and_publish_outputs(
        _Splitter(),
        np.array([1.0, 2.0], dtype=np.float32),
        {},
        ordered,
        streams,
        views,
    )
    np.testing.assert_allclose(views["out_a"], [2.0, 4.0])
    np.testing.assert_allclose(views["out_b"], [2.0, 3.0])
    assert out_a.writes_started == 1 and out_a.writes_finished == 1
    assert out_b.writes_started == 1 and out_b.writes_finished == 1


def test_write_worker_output_ndarray_path():
    stream = _FakeStream("o", np.zeros(3, dtype=np.float32))
    view = np.zeros(3, dtype=np.float32)
    runtime._write_worker_output(stream, view, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(view, [1.0, 2.0, 3.0])
    assert stream.writes_started == 1 and stream.writes_finished == 1


def test_send_worker_event_supports_queue_and_pipe():
    sent = []

    class _Queue:
        def put(self, event):
            sent.append(("queue", event))

    class _Pipe:
        def send(self, event):
            sent.append(("pipe", event))

    runtime._send_worker_event(_Queue(), {"a": 1})
    runtime._send_worker_event(_Pipe(), {"b": 2})
    assert sent == [("queue", {"a": 1}), ("pipe", {"b": 2})]


def test_open_stream_routes_cpu_and_gpu(monkeypatch):
    from shmpipeline.config import SharedMemoryConfig

    calls = []

    def _fake_open(name, **kwargs):
        calls.append((name, kwargs))
        return f"stream:{name}"

    monkeypatch.setattr(runtime.pyshmem, "open", _fake_open)
    cpu_spec = SharedMemoryConfig.from_dict(
        {"name": "c", "shape": [4], "dtype": "float32"}
    )
    assert runtime._open_stream(cpu_spec) == "stream:c"
    assert calls[-1] == ("c", {})
    gpu_spec = SharedMemoryConfig.from_dict(
        {
            "name": "g",
            "shape": [4],
            "dtype": "float32",
            "storage": "gpu",
            "gpu_device": "cuda:0",
        }
    )
    assert runtime._open_stream(gpu_spec) == "stream:g"
    assert calls[-1] == ("g", {"gpu_device": "cuda:0"})
