"""Tests for improvements introduced in the IMPROVEMENTS.md plan."""

from __future__ import annotations

import pickle
import threading
import time
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pyshmem
import pytest

from shmpipeline import PipelineConfig, PipelineManager, PipelineState
from shmpipeline.config import KernelConfig
from shmpipeline.errors import StateTransitionError, WorkerProcessError
from shmpipeline.kernels.cpu.reduce import ReduceCpuKernel
from shmpipeline.registry import KernelRegistry, get_default_registry
from shmpipeline.shm_cleanup import close_stream, unlink_stream_name
from shmpipeline.synthetic import SyntheticInputConfig, SyntheticPatternGenerator

pytestmark = [pytest.mark.unit, pytest.mark.integration]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wait_for_next_write(stream, previous_count, *, timeout=2.0, manager=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if manager is not None:
            manager.poll_events()
        if stream.count > previous_count:
            return stream.read()
        time.sleep(1e-4)
    raise TimeoutError(f"timed out waiting for write on {stream.name!r}")


def _make_scale_config(shm_prefix, *, read_timeout=0.1, poll_interval=None):
    kernel_cfg: dict = {
        "name": "stage",
        "kind": "cpu.scale",
        "input": f"{shm_prefix}_input",
        "output": f"{shm_prefix}_output",
        "parameters": {"factor": 2.0},
        "read_timeout": read_timeout,
    }
    if poll_interval is not None:
        kernel_cfg["poll_interval"] = poll_interval
    return PipelineConfig.from_dict(
        {
            "shared_memory": [
                {"name": f"{shm_prefix}_input", "shape": [4], "dtype": "float32"},
                {"name": f"{shm_prefix}_output", "shape": [4], "dtype": "float32"},
            ],
            "kernels": [kernel_cfg],
        }
    )


def _make_reduce_config(shm_prefix, *, operation="mean"):
    return PipelineConfig.from_dict(
        {
            "shared_memory": [
                {"name": f"{shm_prefix}_input", "shape": [8], "dtype": "float32"},
                {"name": f"{shm_prefix}_output", "shape": [1], "dtype": "float32"},
            ],
            "kernels": [
                {
                    "name": "reduce_stage",
                    "kind": "cpu.reduce",
                    "input": f"{shm_prefix}_input",
                    "output": f"{shm_prefix}_output",
                    "parameters": {"operation": operation},
                    "read_timeout": 0.1,
                }
            ],
        }
    )


# ---------------------------------------------------------------------------
# Issue #22 — pyshmem.unlink_quiet and simplified shm_cleanup
# ---------------------------------------------------------------------------


def test_pyshmem_unlink_quiet_is_public():
    import pyshmem

    assert callable(pyshmem.unlink_quiet)


def test_pyshmem_unlink_quiet_noop_when_stream_missing():
    """unlink_quiet must not raise when the stream does not exist."""
    pyshmem.unlink_quiet("shmpipeline_test_nonexistent_stream_xyz")


def test_pyshmem_unlink_quiet_removes_stream(shm_prefix):
    name = f"{shm_prefix}_quiet_test"
    s = pyshmem.create(name, shape=(4,), dtype=np.float32)
    s.close()
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


# ---------------------------------------------------------------------------
# Issue #6/#1 — Lazy CPU kernel imports / registry picklability
# ---------------------------------------------------------------------------


def test_default_registry_is_picklable():
    """The default registry must be picklable for worker spawning."""
    registry = get_default_registry()
    pickled = pickle.dumps(registry)
    restored = pickle.loads(pickled)
    assert set(restored.kinds()) == set(registry.kinds())


def test_extended_registry_with_class_is_picklable():
    from shmpipeline.kernels.cpu.scale import ScaleCpuKernel

    registry = get_default_registry().extended(ScaleCpuKernel, replace=True)
    pickled = pickle.dumps(registry)
    restored = pickle.loads(pickled)
    assert "cpu.scale" in restored.kinds()


def test_registry_lazy_cpu_kernel_loads_on_first_access():
    registry = get_default_registry()
    assert "cpu.scale" in registry.kinds()
    cls = registry.get("cpu.scale")
    from shmpipeline.kernels.cpu.scale import ScaleCpuKernel

    assert cls is ScaleCpuKernel


def test_registry_lazy_cpu_reduce_is_registered():
    registry = get_default_registry()
    assert "cpu.reduce" in registry.kinds()
    cls = registry.get("cpu.reduce")
    assert cls is ReduceCpuKernel


# ---------------------------------------------------------------------------
# Issue #7 — KernelConfig.poll_interval
# ---------------------------------------------------------------------------


def test_kernel_config_poll_interval_default():
    config = KernelConfig.from_dict(
        {
            "name": "k",
            "kind": "cpu.scale",
            "input": "i",
            "output": "o",
            "parameters": {"factor": 1.0},
        }
    )
    assert config.poll_interval == pytest.approx(1e-5)


def test_kernel_config_poll_interval_custom():
    config = KernelConfig.from_dict(
        {
            "name": "k",
            "kind": "cpu.scale",
            "input": "i",
            "output": "o",
            "parameters": {"factor": 1.0},
            "poll_interval": 0.001,
        }
    )
    assert config.poll_interval == pytest.approx(0.001)


def test_kernel_config_poll_interval_must_be_positive():
    from shmpipeline.errors import ConfigValidationError

    with pytest.raises(ConfigValidationError, match="poll_interval"):
        KernelConfig.from_dict(
            {
                "name": "k",
                "kind": "cpu.scale",
                "input": "i",
                "output": "o",
                "parameters": {"factor": 1.0},
                "poll_interval": -0.001,
            }
        )


def test_poll_interval_passed_to_wait_for_trigger(shm_prefix):
    """poll_interval field is forwarded to _wait_for_trigger in runtime."""
    import shmpipeline.runtime as runtime_module

    recorded: list[float] = []
    original = runtime_module._wait_for_trigger

    def _probe(*args, poll_interval=1e-5, **kwargs):
        recorded.append(poll_interval)
        return original(*args, poll_interval=poll_interval, **kwargs)

    config = _make_scale_config(shm_prefix, poll_interval=0.002)
    manager = PipelineManager(config)
    try:
        manager.build()
        with patch.object(runtime_module, "_wait_for_trigger", _probe):
            manager.start()
            manager.get_stream(f"{shm_prefix}_input").write(
                np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
            )
            time.sleep(0.5)
    finally:
        manager.shutdown(force=True)


# ---------------------------------------------------------------------------
# Issue #10 — output_buffer removed from base Kernel
# ---------------------------------------------------------------------------


def test_cpu_kernel_has_no_output_buffer_attribute():
    """CPU kernels must not pre-allocate an unused output_buffer."""
    from shmpipeline.config import SharedMemoryConfig
    from shmpipeline.kernel import KernelContext
    from shmpipeline.kernels.cpu.scale import ScaleCpuKernel

    shm = {
        "i": SharedMemoryConfig(name="i", shape=(4,), dtype=np.dtype("float32")),
        "o": SharedMemoryConfig(name="o", shape=(4,), dtype=np.dtype("float32")),
    }
    cfg = KernelConfig(
        name="s", kind="cpu.scale", input="i", output="o", parameters={"factor": 1.0}
    )
    kernel = ScaleCpuKernel(KernelContext(config=cfg, shared_memory=shm))
    assert not hasattr(kernel, "output_buffer")


def test_gpu_kernel_still_has_output_buffer():
    """GpuKernel must still allocate output_buffer for its copy path."""
    pytest.importorskip("torch")
    pytest.importorskip("shmpipeline.kernels.gpu.scale")
    from shmpipeline.config import SharedMemoryConfig
    from shmpipeline.kernel import KernelContext
    from shmpipeline.kernels.gpu.scale import ScaleGpuKernel

    if not __import__("torch").cuda.is_available():
        pytest.skip("CUDA not available")
    shm = {
        "i": SharedMemoryConfig(
            name="i",
            shape=(4,),
            dtype=np.dtype("float32"),
            storage="gpu",
            gpu_device="cuda:0",
        ),
        "o": SharedMemoryConfig(
            name="o",
            shape=(4,),
            dtype=np.dtype("float32"),
            storage="gpu",
            gpu_device="cuda:0",
        ),
    }
    cfg = KernelConfig(
        name="s",
        kind="gpu.scale",
        input="i",
        output="o",
        parameters={"factor": 1.0},
    )
    kernel = ScaleGpuKernel(KernelContext(config=cfg, shared_memory=shm))
    assert hasattr(kernel, "output_buffer")


# ---------------------------------------------------------------------------
# Issue #5 — worker_start_timeout
# ---------------------------------------------------------------------------


def test_manager_accepts_worker_start_timeout():
    config = PipelineConfig.from_dict(
        {
            "shared_memory": [
                {"name": "s_i", "shape": [1], "dtype": "float32"},
                {"name": "s_o", "shape": [1], "dtype": "float32"},
            ],
            "kernels": [
                {
                    "name": "k",
                    "kind": "cpu.copy",
                    "input": "s_i",
                    "output": "s_o",
                }
            ],
        }
    )
    manager = PipelineManager(config, worker_start_timeout=15.0)
    assert manager._worker_start_timeout == 15.0


def test_manager_default_worker_start_timeout_is_ten_seconds():
    config = PipelineConfig.from_dict(
        {
            "shared_memory": [
                {"name": "s_i", "shape": [1], "dtype": "float32"},
                {"name": "s_o", "shape": [1], "dtype": "float32"},
            ],
            "kernels": [
                {
                    "name": "k",
                    "kind": "cpu.copy",
                    "input": "s_i",
                    "output": "s_o",
                }
            ],
        }
    )
    manager = PipelineManager(config)
    assert manager._worker_start_timeout == 10.0


# ---------------------------------------------------------------------------
# Issue #13 — thread-safe poll_events / status
# ---------------------------------------------------------------------------


def test_poll_events_is_thread_safe(shm_prefix):
    """Concurrent status() calls must not raise due to race conditions."""
    manager = PipelineManager(_make_scale_config(shm_prefix))
    try:
        manager.build()
        manager.start()
        manager.get_stream(f"{shm_prefix}_input").write(
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        )
        errors: list[Exception] = []
        stop_flag = threading.Event()

        def _poll_loop():
            while not stop_flag.is_set():
                try:
                    manager.status()
                except Exception as exc:
                    errors.append(exc)
                    stop_flag.set()
                time.sleep(1e-4)

        threads = [threading.Thread(target=_poll_loop) for _ in range(4)]
        for t in threads:
            t.start()
        time.sleep(0.5)
        stop_flag.set()
        for t in threads:
            t.join(timeout=2.0)
        assert not errors
    finally:
        manager.shutdown(force=True)


# ---------------------------------------------------------------------------
# Issue #3 — restart()
# ---------------------------------------------------------------------------


def test_restart_raises_when_not_running(shm_prefix):
    manager = PipelineManager(_make_scale_config(shm_prefix))
    with pytest.raises(StateTransitionError):
        manager.restart()


def test_restart_noop_when_no_failures(shm_prefix):
    manager = PipelineManager(_make_scale_config(shm_prefix))
    try:
        manager.build()
        manager.start()
        manager.restart()
        assert manager.state == PipelineState.RUNNING
    finally:
        manager.shutdown(force=True)


def test_restart_recovers_from_failed_worker(shm_prefix):
    """Restart replaces a failed worker and brings pipeline back to RUNNING."""
    config = PipelineConfig.from_dict(
        {
            "shared_memory": [
                {"name": f"{shm_prefix}_input", "shape": [4], "dtype": "float32"},
                {"name": f"{shm_prefix}_output", "shape": [4], "dtype": "float32"},
            ],
            "kernels": [
                {
                    "name": "stage",
                    "kind": "cpu.raise_error",
                    "input": f"{shm_prefix}_input",
                    "output": f"{shm_prefix}_output",
                    "parameters": {"message": "intentional test error"},
                    "read_timeout": 0.1,
                }
            ],
        }
    )
    manager = PipelineManager(config)
    try:
        manager.build()
        manager.start()
        input_stream = manager.get_stream(f"{shm_prefix}_input")
        input_stream.write(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and manager.state != PipelineState.FAILED:
            manager.poll_events()
            time.sleep(0.05)
        assert manager.state == PipelineState.FAILED

        from shmpipeline.kernels.cpu.copy import CopyCpuKernel

        manager._runtime_registry = get_default_registry()
        manager._kernel_configs["stage"] = KernelConfig.from_dict(
            {
                "name": "stage",
                "kind": "cpu.copy",
                "input": f"{shm_prefix}_input",
                "output": f"{shm_prefix}_output",
                "read_timeout": 0.1,
            }
        )
        manager.restart()
        assert manager.state == PipelineState.RUNNING
        assert manager._failures == []
    finally:
        manager.shutdown(force=True)


# ---------------------------------------------------------------------------
# Issue #15 — SyntheticInputConfig pattern vs dtype validation
# ---------------------------------------------------------------------------


def test_synthetic_random_pattern_warns_for_integer_dtype():
    spec = SyntheticInputConfig(stream_name="s", pattern="random")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SyntheticPatternGenerator(spec, shape=(4,), dtype=np.int32, storage="cpu")
    assert any(issubclass(w.category, UserWarning) for w in caught)
    assert any("random" in str(w.message) for w in caught)


def test_synthetic_sine_pattern_warns_for_integer_dtype():
    spec = SyntheticInputConfig(stream_name="s", pattern="sine")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SyntheticPatternGenerator(spec, shape=(4,), dtype=np.uint8, storage="cpu")
    assert any(issubclass(w.category, UserWarning) for w in caught)


def test_synthetic_ramp_pattern_warns_for_integer_dtype():
    spec = SyntheticInputConfig(stream_name="s", pattern="ramp")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SyntheticPatternGenerator(spec, shape=(4,), dtype=np.int32, storage="cpu")
    assert any(issubclass(w.category, UserWarning) for w in caught)


def test_synthetic_constant_pattern_no_warning_for_integer():
    spec = SyntheticInputConfig(stream_name="s", pattern="constant")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SyntheticPatternGenerator(spec, shape=(4,), dtype=np.int32, storage="cpu")
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert not user_warnings


def test_synthetic_random_pattern_no_warning_for_float_dtype():
    spec = SyntheticInputConfig(stream_name="s", pattern="random")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SyntheticPatternGenerator(spec, shape=(4,), dtype=np.float32, storage="cpu")
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert not user_warnings


# ---------------------------------------------------------------------------
# Issue #18 — cpu.reduce kernel
# ---------------------------------------------------------------------------


def test_reduce_kernel_is_registered_in_default_registry():
    registry = get_default_registry()
    assert "cpu.reduce" in registry.kinds()


def test_reduce_mean_operation(shm_prefix):
    manager = PipelineManager(_make_reduce_config(shm_prefix, operation="mean"))
    try:
        manager.build()
        manager.start()
        input_stream = manager.get_stream(f"{shm_prefix}_input")
        output_stream = manager.get_stream(f"{shm_prefix}_output")
        payload = np.arange(8, dtype=np.float32)
        baseline = output_stream.count
        input_stream.write(payload)
        result = _wait_for_next_write(
            output_stream, baseline, timeout=2.0, manager=manager
        )
        np.testing.assert_allclose(result, [payload.mean()], rtol=1e-5)
    finally:
        manager.shutdown(force=True)


def test_reduce_sum_operation(shm_prefix):
    manager = PipelineManager(_make_reduce_config(shm_prefix, operation="sum"))
    try:
        manager.build()
        manager.start()
        input_stream = manager.get_stream(f"{shm_prefix}_input")
        output_stream = manager.get_stream(f"{shm_prefix}_output")
        payload = np.arange(8, dtype=np.float32)
        baseline = output_stream.count
        input_stream.write(payload)
        result = _wait_for_next_write(
            output_stream, baseline, timeout=2.0, manager=manager
        )
        np.testing.assert_allclose(result, [payload.sum()], rtol=1e-5)
    finally:
        manager.shutdown(force=True)


def test_reduce_max_operation(shm_prefix):
    manager = PipelineManager(_make_reduce_config(shm_prefix, operation="max"))
    try:
        manager.build()
        manager.start()
        input_stream = manager.get_stream(f"{shm_prefix}_input")
        output_stream = manager.get_stream(f"{shm_prefix}_output")
        payload = np.array([3.0, 7.0, 1.0, 5.0, 2.0, 9.0, 4.0, 6.0], dtype=np.float32)
        baseline = output_stream.count
        input_stream.write(payload)
        result = _wait_for_next_write(
            output_stream, baseline, timeout=2.0, manager=manager
        )
        np.testing.assert_allclose(result, [9.0], rtol=1e-5)
    finally:
        manager.shutdown(force=True)


def test_reduce_min_operation(shm_prefix):
    manager = PipelineManager(_make_reduce_config(shm_prefix, operation="min"))
    try:
        manager.build()
        manager.start()
        input_stream = manager.get_stream(f"{shm_prefix}_input")
        output_stream = manager.get_stream(f"{shm_prefix}_output")
        payload = np.array([3.0, 7.0, 1.0, 5.0, 2.0, 9.0, 4.0, 6.0], dtype=np.float32)
        baseline = output_stream.count
        input_stream.write(payload)
        result = _wait_for_next_write(
            output_stream, baseline, timeout=2.0, manager=manager
        )
        np.testing.assert_allclose(result, [1.0], rtol=1e-5)
    finally:
        manager.shutdown(force=True)


def test_reduce_rejects_invalid_operation():
    """Invalid reduce operation raises during build (registry validation)."""
    from shmpipeline.errors import ConfigValidationError

    config = PipelineConfig.from_dict(
        {
            "shared_memory": [
                {"name": "in", "shape": [8], "dtype": "float32"},
                {"name": "out", "shape": [1], "dtype": "float32"},
            ],
            "kernels": [
                {
                    "name": "k",
                    "kind": "cpu.reduce",
                    "input": "in",
                    "output": "out",
                    "parameters": {"operation": "product"},
                }
            ],
        }
    )
    manager = PipelineManager(config)
    with pytest.raises(ConfigValidationError, match="unsupported reduce operation"):
        manager.build()


def test_reduce_rejects_non_scalar_output():
    from shmpipeline.errors import ConfigValidationError

    with pytest.raises(
        (ConfigValidationError, Exception),
        match="scalar",
    ):
        config = PipelineConfig.from_dict(
            {
                "shared_memory": [
                    {"name": "in", "shape": [8], "dtype": "float32"},
                    {"name": "out", "shape": [4], "dtype": "float32"},
                ],
                "kernels": [
                    {
                        "name": "k",
                        "kind": "cpu.reduce",
                        "input": "in",
                        "output": "out",
                        "parameters": {"operation": "mean"},
                    }
                ],
            }
        )
        manager = PipelineManager(config)
        manager.build()
        manager.shutdown()


# ---------------------------------------------------------------------------
# Issue #20 — CLI kinds/sources/sinks subcommands
# ---------------------------------------------------------------------------


def test_cli_kinds_command_lists_cpu_scale():
    from shmpipeline.cli import main

    import io
    import sys

    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    try:
        result = main(["kinds"])
    finally:
        sys.stdout = old_stdout
    assert result == 0
    output = captured.getvalue()
    assert "cpu.scale" in output
    assert "cpu.reduce" in output
    assert "cpu.copy" in output


def test_cli_sources_command_no_plugins():
    from shmpipeline.cli import main

    import io
    import sys

    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    try:
        result = main(["sources"])
    finally:
        sys.stdout = old_stdout
    assert result == 0


def test_cli_sinks_command_no_plugins():
    from shmpipeline.cli import main

    import io
    import sys

    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    try:
        result = main(["sinks"])
    finally:
        sys.stdout = old_stdout
    assert result == 0


def test_cli_kinds_includes_gpu_kinds_when_torch_available():
    try:
        import torch
        cuda = torch.cuda.is_available()
    except ImportError:
        pytest.skip("torch not available")

    from shmpipeline.cli import main

    import io
    import sys

    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    try:
        result = main(["kinds"])
    finally:
        sys.stdout = old_stdout
    assert result == 0
    output = captured.getvalue()
    assert "gpu.scale" in output


# ---------------------------------------------------------------------------
# Integration: raise_if_failed uses correct component_type
# ---------------------------------------------------------------------------


def test_raise_if_failed_uses_source_component_type(shm_prefix):
    """Failures from sources should say 'source ... failed', not 'worker'."""
    manager = PipelineManager(_make_scale_config(shm_prefix))
    # Inject a fake source failure directly
    manager._failures.append(
        {
            "type": "source_failed",
            "kernel": "my_source",
            "component_type": "source",
            "error": "read error",
        }
    )
    with pytest.raises(WorkerProcessError, match="source 'my_source' failed"):
        manager.raise_if_failed()


def test_raise_if_failed_uses_sink_component_type(shm_prefix):
    manager = PipelineManager(_make_scale_config(shm_prefix))
    manager._failures.append(
        {
            "type": "sink_failed",
            "kernel": "my_sink",
            "component_type": "sink",
            "error": "write error",
        }
    )
    with pytest.raises(WorkerProcessError, match="sink 'my_sink' failed"):
        manager.raise_if_failed()


# ---------------------------------------------------------------------------
# Integration: worker_start_timeout used in start()
# ---------------------------------------------------------------------------


def test_manager_uses_custom_worker_start_timeout(shm_prefix, monkeypatch):
    """Manager must use worker_start_timeout when waiting for workers."""
    import shmpipeline.manager as manager_module

    timeouts_seen: list[float] = []
    original = manager_module.PipelineManager._wait_for_workers_started

    def _probe(self, *, timeout=None, only=None):
        timeouts_seen.append(timeout)
        return original(self, timeout=timeout, only=only)

    monkeypatch.setattr(
        manager_module.PipelineManager, "_wait_for_workers_started", _probe
    )

    manager = PipelineManager(
        _make_scale_config(shm_prefix), worker_start_timeout=7.5
    )
    try:
        manager.build()
        manager.start()
    finally:
        manager.shutdown(force=True)

    assert any(t == pytest.approx(7.5) for t in timeouts_seen)
