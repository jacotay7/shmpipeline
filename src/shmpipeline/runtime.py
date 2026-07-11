"""Worker runtime for process-based kernel execution."""

from __future__ import annotations

import math
import os
import time
import traceback
from collections import deque
from contextlib import ExitStack, contextmanager
from queue import Empty
from typing import Any, Mapping

import pyshmem

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
    stream,
    previous_count: float,
    *,
    timeout: float,
    poll_interval: float = 1e-5,
) -> float | None:
    """Wait for a trigger count using pyshmem's level-triggered primitive."""
    try:
        return stream.wait_for_count(
            after=previous_count,
            timeout=timeout,
            poll_interval=poll_interval,
        )
    except TimeoutError:
        return None


@contextmanager
def _locked_inputs_and_outputs(
    kernel_config: KernelConfig,
    trigger_stream: Any,
    auxiliary_streams: dict[str, Any],
    output_streams: dict[str, Any],
):
    """Yield inputs and output views while all required locks are held.

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
        **output_streams,
    }
    with pyshmem.locked_many(
        tuple(locked_streams.values()),
        timeout=kernel_config.read_timeout,
    ):
        current_count = trigger_stream.count
        trigger_input = _read_worker_input(trigger_stream)
        auxiliary_inputs = {
            name: _read_worker_input(stream)
            for name, stream in auxiliary_streams.items()
        }
        yield current_count, trigger_input, auxiliary_inputs


def _read_worker_input(stream: Any):
    """Return one worker input snapshot with storage-aware consistency."""
    if getattr(stream, "gpu_enabled", False):
        return stream.read(safe=True)
    return stream.read(safe=False)


def _compute_and_publish_outputs(
    kernel: Any,
    trigger_input: Any,
    auxiliary_inputs: Mapping[str, Any],
    ordered_output_names: tuple[str, ...],
    output_streams: dict[str, Any],
) -> None:
    """Compute directly into exception-safe pyshmem output transactions."""
    with ExitStack() as stack:
        output_views = [
            stack.enter_context(output_streams[name].write_view_locked())
            for name in ordered_output_names
        ]
        kernel.compute_into_multiple(
            trigger_input, output_views, auxiliary_inputs
        )


def _send_worker_event(event_sink: Any, event: dict[str, Any]) -> None:
    """Send one worker event through either a queue or a pipe endpoint."""
    put = getattr(event_sink, "put", None)
    if callable(put):
        put(event)
        return
    send = getattr(event_sink, "send", None)
    if callable(send):
        send(event)
        return
    raise TypeError("unsupported worker event sink")


def run_kernel_process(
    kernel_config: KernelConfig,
    shared_memory: tuple[SharedMemoryConfig, ...],
    pause_event: Any,
    stop_event: Any,
    event_sink: Any,
    cpu_slot: int | None = None,
    registry: Any | None = None,
) -> None:
    """Entry point executed in each worker subprocess."""
    logger = get_logger(f"worker.{kernel_config.name}")
    _pin_current_process(cpu_slot)
    shared_by_name = {item.name: item for item in shared_memory}
    if registry is None:
        registry = get_default_registry()
    kernel = registry.create(kernel_config, shared_by_name)
    trigger_stream = _open_stream(shared_by_name[kernel_config.input])
    auxiliary_streams = {
        binding.alias: _open_stream(shared_by_name[binding.name])
        for binding in kernel_config.auxiliary
    }
    output_streams = {
        name: _open_stream(shared_by_name[name])
        for name in kernel_config.all_outputs
    }
    _send_worker_event(
        event_sink,
        {
            "type": "worker_started",
            "kernel": kernel_config.name,
            "pid": os.getpid(),
            "cpu_slot": cpu_slot,
        },
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
                poll_interval=kernel_config.poll_interval,
            )
            if triggered_count is None:
                continue

            try:
                with _locked_inputs_and_outputs(
                    kernel_config,
                    trigger_stream,
                    auxiliary_streams,
                    output_streams,
                ) as (
                    current_count,
                    trigger_input,
                    auxiliary_inputs,
                ):
                    if current_count <= last_seen_count:
                        continue

                    compute_started = time.perf_counter()
                    _compute_and_publish_outputs(
                        kernel,
                        trigger_input,
                        auxiliary_inputs,
                        kernel_config.all_outputs,
                        output_streams,
                    )
                    last_seen_count = current_count
            except TimeoutError:
                continue

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
                _send_worker_event(
                    event_sink,
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
                        "last_output_count": output_streams[
                            kernel_config.output
                        ].count,
                        "metrics_window": len(exec_samples_s),
                    },
                )
                next_metrics_emit = now + 0.25
    except BaseException as exc:
        logger.exception(
            "worker runtime failed: kernel=%s", kernel_config.name
        )
        _send_worker_event(
            event_sink,
            {
                "type": "worker_failed",
                "kernel": kernel_config.name,
                "pid": os.getpid(),
                "cpu_slot": cpu_slot,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        trigger_stream.close()
        for stream in auxiliary_streams.values():
            stream.close()
        for stream in output_streams.values():
            stream.close()
        logger.info(
            "worker runtime stopping: kernel=%s pid=%s",
            kernel_config.name,
            os.getpid(),
        )
        _send_worker_event(
            event_sink,
            {
                "type": "worker_stopped",
                "kernel": kernel_config.name,
                "pid": os.getpid(),
                "cpu_slot": cpu_slot,
            },
        )
        close = getattr(event_sink, "close", None)
        if callable(close):
            close()


def drain_events(event_queue: Any) -> list[dict[str, Any]]:
    """Drain all currently pending worker events from queues or pipes."""
    events: list[dict[str, Any]] = []
    if event_queue is None:
        return events
    if isinstance(event_queue, (list, tuple, set)):
        for event_source in event_queue:
            events.extend(drain_events(event_source))
        return events

    poll = getattr(event_queue, "poll", None)
    recv = getattr(event_queue, "recv", None)
    if callable(poll) and callable(recv):
        while True:
            try:
                if not event_queue.poll():
                    return events
                events.append(event_queue.recv())
            except (EOFError, OSError):
                return events

    while True:
        try:
            events.append(event_queue.get_nowait())
        except Empty:
            return events
