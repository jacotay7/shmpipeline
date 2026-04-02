"""Worker runtime for process-based kernel execution."""

from __future__ import annotations

import math
import os
import time
import traceback
from collections import deque
from contextlib import ExitStack
from queue import Empty
from typing import Any

import numpy as np
import pyshmem

try:
    import torch
except Exception:  # pragma: no cover - exercised when torch is unavailable
    torch = None

from shmpipeline.config import KernelConfig, SharedMemoryConfig
from shmpipeline.logging_utils import get_logger
from shmpipeline.registry import get_default_registry

ROLLING_METRICS_WINDOW = 1000


def _compute_rolling_throughput_hz(
    completion_times: deque[float],
) -> float:
    """Return rolling throughput based on recent completion timestamps."""
    if len(completion_times) < 2:
        return 0.0
    duration = completion_times[-1] - completion_times[0]
    if duration <= 0.0:
        return 0.0
    return (len(completion_times) - 1) / duration


def _compute_rolling_exec_metrics(
    exec_samples_s: deque[float],
) -> tuple[float, float, float]:
    """Return rolling average ms/us and RMS jitter us for compute time."""
    if not exec_samples_s:
        return 0.0, 0.0, 0.0
    sample_count = len(exec_samples_s)
    mean_s = sum(exec_samples_s) / sample_count
    squared_error_sum = 0.0
    for sample_s in exec_samples_s:
        squared_error = sample_s - mean_s
        squared_error_sum += squared_error * squared_error
    jitter_s = math.sqrt(squared_error_sum / sample_count)
    return 1000.0 * mean_s, 1_000_000.0 * mean_s, 1_000_000.0 * jitter_s


def _open_stream(spec: SharedMemoryConfig):
    if spec.storage == "gpu":
        return pyshmem.open(spec.name, gpu_device=spec.gpu_device)
    return pyshmem.open(spec.name)


def _pin_current_process(cpu_slot: int | None) -> None:
    """Best-effort CPU placement for worker processes."""
    if cpu_slot is None or not hasattr(os, "sched_setaffinity"):
        return
    try:
        os.sched_setaffinity(0, {cpu_slot})
    except OSError:
        return


def _wait_for_trigger(
    stream, previous_count: float, *, timeout: float
) -> float | None:
    """Wait for a trigger stream count to advance beyond the previous count."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = stream.count
        if current > previous_count:
            return current
        time.sleep(1e-5)
    return None


def _read_locked_inputs_and_output(
    kernel_config: KernelConfig,
    trigger_stream: Any,
    auxiliary_streams: dict[str, Any],
    output_stream: Any,
):
    """Read a consistent set of inputs and the output view under locks.

    Lock acquisition can legitimately fail under load when an external writer
    is updating the trigger stream. That is treated as transient backpressure,
    not a worker failure.

    GPU streams must use the consistent read path even while the lock is held.
    The direct `safe=False` handle path can observe stale CUDA IPC contents
    across processes because it bypasses pyshmem's synchronization/copy logic.
    """
    locked_streams = {
        kernel_config.input: trigger_stream,
        **auxiliary_streams,
        kernel_config.output: output_stream,
    }
    with ExitStack() as stack:
        for name in sorted(locked_streams):
            stack.enter_context(
                locked_streams[name].locked(timeout=kernel_config.read_timeout)
            )

        current_count = trigger_stream.count
        trigger_input = _read_worker_input(trigger_stream)
        auxiliary_inputs = {
            name: _read_worker_input(stream)
            for name, stream in auxiliary_streams.items()
        }
        output_view = output_stream.read(safe=False)
        return current_count, trigger_input, auxiliary_inputs, output_view


def _read_worker_input(stream: Any):
    """Return one worker input snapshot with storage-aware consistency."""
    if getattr(stream, "gpu_enabled", False):
        return stream.read(safe=True)
    return stream.read(safe=False)


def _write_worker_output(
    output_stream: Any, output_view: Any, value: Any
) -> None:
    """Publish one output buffer while keeping GPU mirrors and metadata consistent."""
    output_stream._mark_write_started()
    if isinstance(output_view, np.ndarray):
        np.copyto(output_view, value)
        output_stream._finish_write()
        return

    output_view.copy_(value)
    if getattr(output_stream, "cpu_mirror", False):
        np.copyto(output_stream._array, value.detach().cpu().numpy())
    if torch is not None:
        torch.cuda.synchronize(device=output_stream.gpu_device)
    output_stream._finish_write()


def run_kernel_process(
    kernel_config: KernelConfig,
    shared_memory: tuple[SharedMemoryConfig, ...],
    pause_event: Any,
    stop_event: Any,
    event_queue: Any,
    cpu_slot: int | None = None,
) -> None:
    """Entry point executed in each worker subprocess."""
    logger = get_logger(f"worker.{kernel_config.name}")
    _pin_current_process(cpu_slot)
    shared_by_name = {item.name: item for item in shared_memory}
    registry = get_default_registry()
    kernel = registry.create(kernel_config, shared_by_name)
    trigger_stream = _open_stream(shared_by_name[kernel_config.input])
    auxiliary_streams = {
        binding.alias: _open_stream(shared_by_name[binding.name])
        for binding in kernel_config.auxiliary
    }
    output_stream = _open_stream(shared_by_name[kernel_config.output])
    event_queue.put(
        {
            "type": "worker_started",
            "kernel": kernel_config.name,
            "pid": os.getpid(),
            "cpu_slot": cpu_slot,
        }
    )
    logger.info(
        "worker runtime started: kernel=%s pid=%s cpu_slot=%s",
        kernel_config.name,
        os.getpid(),
        cpu_slot,
    )

    try:
        last_seen_count = trigger_stream.count
        frames_processed = 0
        exec_samples_s: deque[float] = deque(maxlen=ROLLING_METRICS_WINDOW)
        completion_times: deque[float] = deque(maxlen=ROLLING_METRICS_WINDOW)
        next_metrics_emit = time.monotonic() + 0.25
        started_at = time.monotonic()
        while not stop_event.is_set():
            if pause_event.is_set():
                time.sleep(kernel_config.pause_sleep)
                continue

            triggered_count = _wait_for_trigger(
                trigger_stream,
                last_seen_count,
                timeout=kernel_config.read_timeout,
            )
            if triggered_count is None:
                continue

            try:
                (
                    current_count,
                    trigger_input,
                    auxiliary_inputs,
                    output_view,
                ) = _read_locked_inputs_and_output(
                    kernel_config,
                    trigger_stream,
                    auxiliary_streams,
                    output_stream,
                )
            except TimeoutError:
                continue

            if current_count <= last_seen_count:
                continue

            compute_started = time.perf_counter()
            kernel.compute_into(
                trigger_input,
                kernel.output_buffer,
                auxiliary_inputs,
            )
            _write_worker_output(
                output_stream,
                output_view,
                kernel.output_buffer,
            )
            last_seen_count = current_count
            last_exec_s = time.perf_counter() - compute_started
            frames_processed += 1
            now = time.monotonic()
            exec_samples_s.append(last_exec_s)
            completion_times.append(now)
            if frames_processed == 1 or now >= next_metrics_emit:
                runtime_s = max(now - started_at, 1e-12)
                avg_exec_ms, avg_exec_us, jitter_us_rms = (
                    _compute_rolling_exec_metrics(exec_samples_s)
                )
                event_queue.put(
                    {
                        "type": "worker_metrics",
                        "kernel": kernel_config.name,
                        "pid": os.getpid(),
                        "cpu_slot": cpu_slot,
                        "frames_processed": frames_processed,
                        "last_exec_ms": 1000.0 * last_exec_s,
                        "last_exec_us": 1_000_000.0 * last_exec_s,
                        "avg_exec_ms": avg_exec_ms,
                        "avg_exec_us": avg_exec_us,
                        "jitter_us_rms": jitter_us_rms,
                        "throughput_hz": _compute_rolling_throughput_hz(
                            completion_times
                        ),
                        "runtime_s": runtime_s,
                        "last_output_count": output_stream.count,
                        "metrics_window": len(exec_samples_s),
                    }
                )
                next_metrics_emit = now + 0.25
    except BaseException as exc:
        logger.exception(
            "worker runtime failed: kernel=%s", kernel_config.name
        )
        event_queue.put(
            {
                "type": "worker_failed",
                "kernel": kernel_config.name,
                "pid": os.getpid(),
                "cpu_slot": cpu_slot,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        raise
    finally:
        trigger_stream.close()
        for stream in auxiliary_streams.values():
            stream.close()
        output_stream.close()
        logger.info(
            "worker runtime stopping: kernel=%s pid=%s",
            kernel_config.name,
            os.getpid(),
        )
        event_queue.put(
            {
                "type": "worker_stopped",
                "kernel": kernel_config.name,
                "pid": os.getpid(),
                "cpu_slot": cpu_slot,
            }
        )


def drain_events(event_queue: Any) -> list[dict[str, Any]]:
    """Drain all currently pending worker events from the queue."""
    events: list[dict[str, Any]] = []
    while True:
        try:
            events.append(event_queue.get_nowait())
        except Empty:
            return events
