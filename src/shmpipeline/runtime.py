"""Worker runtime for process-based kernel execution."""

from __future__ import annotations

import os
import time
import traceback
from queue import Empty
from typing import Any

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
    input_streams = {
        name: _open_stream(shared_by_name[name]) for name in kernel_config.inputs
    }
    output_streams = {
        name: _open_stream(shared_by_name[name]) for name in kernel_config.outputs
    }
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
        primary_input = kernel_config.inputs[0]
        while not stop_event.is_set():
            if pause_event.is_set():
                time.sleep(kernel_config.pause_sleep)
                continue

            try:
                inputs = {
                    primary_input: input_streams[primary_input].read_new(
                        timeout=kernel_config.read_timeout
                    )
                }
            except TimeoutError:
                continue
            for name in kernel_config.inputs[1:]:
                inputs[name] = input_streams[name].read()

            outputs = kernel.compute(inputs)
            expected_outputs = set(kernel_config.outputs)
            if set(outputs) != expected_outputs:
                raise KernelExecutionError(
                    f"kernel {kernel_config.name!r} produced outputs "
                    f"{sorted(outputs)} but expected {sorted(expected_outputs)}"
                )
            for name, value in outputs.items():
                output_streams[name].write(value)
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
        for stream in input_streams.values():
            stream.close()
        for stream in output_streams.values():
            stream.close()
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