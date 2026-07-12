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


def _wait_for_triggers(
    streams: Mapping[str, Any],
    previous_counts: Mapping[str, float],
    *,
    policy: str,
    timeout: float,
    poll_interval: float = 1e-5,
) -> dict[str, float] | None:
    """Wait until any or all dynamic input generations have advanced."""
    deadline = time.monotonic() + timeout
    while True:
        counts = {name: stream.count for name, stream in streams.items()}
        advanced = {
            name: counts[name] > previous_counts[name] for name in streams
        }
        if (policy == "all_new" and all(advanced.values())) or (
            policy == "any_new" and any(advanced.values())
        ):
            return counts
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return None
        pending = [
            name
            for name in streams
            if not advanced[name] or policy == "any_new"
        ]
        name = pending[0]
        wait_timeout = min(remaining, max(poll_interval, 0.01))
        try:
            streams[name].wait_for_count(
                after=previous_counts[name],
                timeout=wait_timeout,
                poll_interval=poll_interval,
            )
        except TimeoutError:
            continue


def _matching_frame_decision(
    tokens: Mapping[str, int],
) -> tuple[int, int, tuple[str, ...]]:
    """Resolve a ``matching_frame_id`` barrier against the current tokens.

    Returns ``(target, skew_gap, laggards)`` where ``target`` is the newest
    token across the triggers, ``skew_gap`` is the spread between newest and
    oldest, and ``laggards`` names the triggers still behind ``target``. An
    empty ``laggards`` means every trigger already carries ``target`` and the
    kernel may combine them. Determinism (no arrival-order dependence) is why
    this is a pure function of the token snapshot taken under the lock.
    """
    values = list(tokens.values())
    target = max(values)
    skew_gap = target - min(values)
    laggards = tuple(name for name, token in tokens.items() if token < target)
    return target, skew_gap, laggards


@contextmanager
def _locked_inputs_and_outputs(
    kernel_config: KernelConfig,
    trigger_stream: Any,
    auxiliary_streams: dict[str, Any],
    output_streams: dict[str, Any],
    *,
    ordered_locked_streams: tuple[tuple[str, Any], ...] | None = None,
    auxiliary_inputs: dict[str, Any] | None = None,
    auxiliary_cache: dict[str, tuple[Any, Any]] | None = None,
    track_frame_id: bool = False,
):
    """Yield inputs and output views while all required locks are held.

    Lock acquisition can legitimately fail under load when an external writer
    is updating the trigger stream. That is treated as transient backpressure,
    not a worker failure.

    GPU streams must use the consistent read path even while the lock is held.
    The direct `safe=False` handle path can observe stale CUDA IPC contents
    across processes because it bypasses pyshmem's synchronization/copy logic.
    """
    if isinstance(trigger_stream, Mapping):
        trigger_streams = dict(trigger_stream)
    else:
        trigger_streams = {kernel_config.input: trigger_stream}
    if ordered_locked_streams is None:
        locked_streams = {
            **trigger_streams,
            **auxiliary_streams,
            **output_streams,
        }
        ordered_locked_streams = tuple(
            sorted(
                locked_streams.items(),
                key=lambda item: item[1].name,
            )
        )
    with pyshmem.locked_many(
        tuple(stream for _, stream in ordered_locked_streams),
        timeout=kernel_config.read_timeout,
        poll_interval=kernel_config.poll_interval,
    ):
        current_counts = {
            name: stream.count for name, stream in trigger_streams.items()
        }
        trigger_values = tuple(
            _read_worker_input(
                trigger_streams[name],
                borrow_gpu=bool(
                    kernel_config.parameters.get("borrow_gpu_inputs", False)
                ),
            )
            for name in kernel_config.trigger_inputs
        )
        trigger_input = (
            trigger_values[0] if len(trigger_values) == 1 else trigger_values
        )
        if auxiliary_inputs is None:
            auxiliary_inputs = {}
        else:
            auxiliary_inputs.clear()
        for name, stream in auxiliary_streams.items():
            auxiliary_inputs[name] = _read_worker_input(
                stream,
                cache=auxiliary_cache,
                cache_key=name,
            )
        # Read the propagated tokens inside the same lock scope that snapshots
        # the payloads, so a token can never be torn from its value.
        trigger_frame_ids = (
            {name: stream.frame_id for name, stream in trigger_streams.items()}
            if track_frame_id
            else {}
        )
        current_count = (
            next(iter(current_counts.values()))
            if len(current_counts) == 1
            else current_counts
        )
        yield (
            current_count,
            trigger_input,
            auxiliary_inputs,
            trigger_frame_ids,
        )


def _read_worker_input(
    stream: Any,
    *,
    borrow_gpu: bool = False,
    cache: dict[str, tuple[Any, Any]] | None = None,
    cache_key: str | None = None,
):
    """Return one worker input snapshot with storage-aware consistency."""
    if getattr(stream, "gpu_enabled", False):
        if borrow_gpu:
            return stream.read(safe=False)
        count = getattr(stream, "count", None)
        if cache is not None and cache_key is not None:
            cached = cache.get(cache_key)
            if cached is not None and cached[0] == count:
                return cached[1]
        value = stream.read(safe=True)
        if cache is not None and cache_key is not None:
            cache[cache_key] = (
                getattr(stream, "last_read_count", count),
                value,
            )
        return value
    return stream.read(safe=False)


def _compute_and_publish_outputs(
    kernel: Any,
    trigger_input: Any,
    auxiliary_inputs: Mapping[str, Any],
    ordered_output_names: tuple[str, ...],
    output_streams: dict[str, Any],
    frame_id: int | None = None,
) -> None:
    """Compute directly into exception-safe pyshmem output transactions.

    ``frame_id`` propagates a publication token onto every output in the same
    locked transaction; ``None`` leaves each output's token unchanged.
    """
    with ExitStack() as stack:
        output_views = [
            stack.enter_context(
                output_streams[name].write_view_locked(frame_id=frame_id)
            )
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
    trigger_streams = {
        name: _open_stream(shared_by_name[name])
        for name in kernel_config.trigger_inputs
    }
    auxiliary_streams = {
        binding.alias: _open_stream(shared_by_name[binding.name])
        for binding in kernel_config.auxiliary
    }
    output_streams = {
        name: _open_stream(shared_by_name[name])
        for name in kernel_config.all_outputs
    }
    ordered_locked_streams = tuple(
        sorted(
            {
                **trigger_streams,
                **auxiliary_streams,
                **output_streams,
            }.items(),
            key=lambda item: item[1].name,
        )
    )
    auxiliary_inputs: dict[str, Any] = {}
    auxiliary_cache: dict[str, tuple[Any, Any]] = {}
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
        last_seen_counts = {
            name: stream.count for name, stream in trigger_streams.items()
        }
        frames_processed = 0
        exec_samples_s: deque[float] = deque(maxlen=ROLLING_METRICS_WINDOW)
        completion_times: deque[float] = deque(maxlen=ROLLING_METRICS_WINDOW)
        next_metrics_emit = time.monotonic() + 0.25
        started_at = time.monotonic()
        track_frame_id = kernel_config.tracks_frame_id
        matching = kernel_config.requires_matching_frame_id
        sync = kernel_config.synchronization
        skew_started_at: float | None = None
        matching_skew_events = 0
        matching_skipped_generations = 0
        matching_timeouts = 0
        while not stop_event.is_set():
            if pause_event.is_set():
                time.sleep(kernel_config.pause_sleep)
                continue

            triggered_counts = _wait_for_triggers(
                trigger_streams,
                last_seen_counts,
                policy=kernel_config.trigger_policy,
                timeout=kernel_config.read_timeout,
                poll_interval=kernel_config.poll_interval,
            )
            if triggered_counts is None:
                continue

            try:
                with _locked_inputs_and_outputs(
                    kernel_config,
                    trigger_streams,
                    auxiliary_streams,
                    output_streams,
                    ordered_locked_streams=ordered_locked_streams,
                    auxiliary_inputs=auxiliary_inputs,
                    auxiliary_cache=auxiliary_cache,
                    track_frame_id=track_frame_id,
                ) as (
                    current_counts,
                    trigger_input,
                    auxiliary_inputs,
                    trigger_frame_ids,
                ):
                    if not isinstance(current_counts, Mapping):
                        current_counts = {kernel_config.input: current_counts}
                    advanced = {
                        name: current_counts[name] > last_seen_counts[name]
                        for name in trigger_streams
                    }
                    if kernel_config.trigger_policy == "all_new":
                        should_compute = all(advanced.values())
                    else:
                        should_compute = any(advanced.values())
                    if not should_compute:
                        continue

                    frame_id = None
                    if matching:
                        target, skew_gap, laggards = _matching_frame_decision(
                            trigger_frame_ids
                        )
                        if not laggards:
                            frame_id = target or None
                            skew_started_at = None
                        else:
                            matching_skew_events += 1
                            now_m = time.monotonic()
                            if skew_started_at is None:
                                skew_started_at = now_m
                            waited = now_m - skew_started_at
                            give_up = skew_gap > sync.max_skew_generations or (
                                sync.max_wait_s is not None
                                and waited >= sync.max_wait_s
                            )
                            if give_up:
                                # Abandon the mismatched generation entirely so
                                # a failed branch cannot stall the loop.
                                matching_timeouts += 1
                                last_seen_counts = dict(current_counts)
                                skew_started_at = None
                            else:
                                # drop_older: consume the lagging branches and
                                # wait for their next (newer) publication.
                                matching_skipped_generations += 1
                                for name in laggards:
                                    last_seen_counts[name] = current_counts[
                                        name
                                    ]
                            continue
                    elif track_frame_id:
                        token = (
                            max(trigger_frame_ids.values())
                            if trigger_frame_ids
                            else 0
                        )
                        frame_id = token or None

                    compute_started = time.perf_counter()
                    _compute_and_publish_outputs(
                        kernel,
                        trigger_input,
                        auxiliary_inputs,
                        kernel_config.all_outputs,
                        output_streams,
                        frame_id=frame_id,
                    )
                    last_seen_counts = dict(current_counts)
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
                        "frame_sync_skew_events": matching_skew_events,
                        "frame_sync_skipped_generations": (
                            matching_skipped_generations
                        ),
                        "frame_sync_timeouts": matching_timeouts,
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
        for stream in trigger_streams.values():
            stream.close()
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
