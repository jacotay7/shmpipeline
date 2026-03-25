"""Worker runtime for process-based kernel execution."""

from __future__ import annotations

from contextlib import ExitStack
import os
import time
import traceback
from queue import Empty
from typing import Any

import numpy as np

import pyshmem

from shmpipeline.config import KernelConfig, SharedMemoryConfig
from shmpipeline.errors import KernelExecutionError
from shmpipeline.logging_utils import get_logger
from shmpipeline.registry import get_default_registry


def _open_stream(spec: SharedMemoryConfig):
    if spec.storage == "gpu":
        return pyshmem.open(spec.name, gpu_device=spec.gpu_device)
    return pyshmem.open(spec.name)


def _pin_current_process(cpu_slot: int | None) -> None:
    """Best-effort CPU placement for worker processes on supported platforms."""
    if cpu_slot is None or not hasattr(os, "sched_setaffinity"):
        return
    try:
        os.sched_setaffinity(0, {cpu_slot})
    except OSError:
        return


def _wait_for_trigger(stream, previous_count: float, *, timeout: float) -> float | None:
    """Wait for a trigger stream count to advance beyond the previous count."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = stream.count
        if current > previous_count:
            return current
        time.sleep(1e-5)
    return None


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

            locked_streams = {kernel_config.input: trigger_stream, **auxiliary_streams, kernel_config.output: output_stream}
            with ExitStack() as stack:
                for name in sorted(locked_streams):
                    stack.enter_context(locked_streams[name].locked(timeout=kernel_config.read_timeout))

                current_count = trigger_stream.count
                if current_count <= last_seen_count:
                    continue

                trigger_input = trigger_stream.read(safe=False)
                auxiliary_inputs = {
                    name: stream.read(safe=False)
                    for name, stream in auxiliary_streams.items()
                }
                output_view = output_stream.read(safe=False)

                kernel.compute_into(
                    trigger_input,
                    kernel.output_buffer,
                    auxiliary_inputs,
                )
                output_stream._mark_write_started()
                if isinstance(output_view, np.ndarray):
                    np.copyto(output_view, kernel.output_buffer)
                else:
                    output_view.copy_(kernel.output_buffer)
                output_stream._finish_write()
                last_seen_count = current_count
    except BaseException as exc:
        logger.exception("worker runtime failed: kernel=%s", kernel_config.name)
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
        logger.info("worker runtime stopping: kernel=%s pid=%s", kernel_config.name, os.getpid())
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