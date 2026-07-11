"""Pipeline manager responsible for building and supervising workers."""

from __future__ import annotations

import concurrent.futures
import multiprocessing as mp
import os
import threading
import time
import traceback
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyshmem

from shmpipeline.config import (
    KernelConfig,
    PipelineConfig,
    SharedMemoryConfig,
)
from shmpipeline.errors import (
    ConfigValidationError,
    StateTransitionError,
    WorkerProcessError,
)
from shmpipeline.graph import PipelineGraph
from shmpipeline.logging_utils import get_logger
from shmpipeline.registry import KernelRegistry, get_default_registry
from shmpipeline.runtime import drain_events, run_kernel_process
from shmpipeline.scheduling import (
    WorkerPlacementPolicy,
    normalize_placement_policy,
)
from shmpipeline.shm_cleanup import close_stream, unlink_stream_name
from shmpipeline.state import PipelineState

if TYPE_CHECKING:
    from shmpipeline.synthetic import SyntheticInputConfig


@dataclass
class _WorkerHandle:
    """Runtime handle for one worker process."""

    name: str
    process: Any
    stop_event: Any
    event_reader: Any | None
    cpu_slot: int | None


def _read_sink_payload(
    stream: Any,
    previous_count: int,
    *,
    timeout: float,
) -> tuple[int, Any]:
    """Wait for and read one stable payload through pyshmem."""
    payload = stream.read_after(previous_count, timeout=timeout)
    return stream.last_read_count, payload


def _call_with_optional_timeout(
    func: Any,
    *,
    timeout: float | None,
    executor: concurrent.futures.ThreadPoolExecutor | None,
    label: str,
) -> Any:
    """Call ``func`` directly, or under a soft timeout via ``executor``.

    When ``timeout`` is ``None`` the plugin call runs inline on the controller
    thread (the zero-overhead default).  When a timeout is configured the call
    runs on a dedicated single-thread executor and a :class:`TimeoutError` is
    raised if it does not return in time, isolating the controller from a
    plugin that blocks indefinitely.  A timed-out call keeps running on the
    executor thread until it returns; its result is discarded.
    """
    if timeout is None or executor is None:
        return func()
    future = executor.submit(func)
    try:
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError as exc:
        future.cancel()
        raise TimeoutError(
            f"{label} exceeded its {timeout:g}s timeout"
        ) from exc


class _SourceController:
    """Manager-owned thread controller for one configured source plugin."""

    def __init__(
        self, *, stream: Any, source: Any, spec: Any, pause_event: Any
    ):
        self.stream = stream
        self.source = source
        self.spec = spec
        self._pause_event = pause_event
        self._stop_event = threading.Event()
        self._logger = get_logger(f"source.{spec.name}")
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run,
            name=f"shmpipeline-source-{spec.name}",
            daemon=True,
        )
        self._started_at_wall: float | None = None
        self._started_at_mono: float | None = None
        self._last_write_time: float | None = None
        self._last_write_duration_s: float | None = None
        self._frames_written = 0
        self._last_error: str | None = None
        self._traceback: str | None = None
        self._failure_reported = False
        self._read_timeout = getattr(spec, "read_timeout", None)
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        if self._read_timeout is not None:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"shmpipeline-source-read-{spec.name}",
            )
        self.source._attach_runtime_events(
            stop_event=self._stop_event,
            pause_event=self._pause_event,
        )

    def start(self) -> None:
        """Start the source thread."""
        if self._thread.is_alive():
            return
        self.source.open()
        self._started_at_wall = time.time()
        self._started_at_mono = time.perf_counter()
        self._thread.start()

    def stop(self, *, timeout: float = 2.0) -> None:
        """Request shutdown and wait for the source thread to exit."""
        self._stop_event.set()
        self._thread.join(timeout=timeout)

    def snapshot(self) -> dict[str, Any]:
        """Return a stable status snapshot for one source plugin."""
        with self._lock:
            elapsed_s = 0.0
            if self._started_at_mono is not None:
                elapsed_s = max(
                    0.0, time.perf_counter() - self._started_at_mono
                )
            effective_rate_hz = 0.0
            if elapsed_s > 0.0:
                effective_rate_hz = self._frames_written / elapsed_s
            return {
                "name": self.spec.name,
                "kind": self.spec.kind,
                "stream": self.spec.stream,
                "poll_interval": self.spec.poll_interval,
                "alive": self._thread.is_alive(),
                "frames_written": self._frames_written,
                "effective_rate_hz": effective_rate_hz,
                "started_at": self._started_at_wall,
                "last_write_time": self._last_write_time,
                "last_write_duration_ms": (
                    None
                    if self._last_write_duration_s is None
                    else 1000.0 * self._last_write_duration_s
                ),
                "last_error": self._last_error,
            }

    def consume_failure(self) -> dict[str, Any] | None:
        """Return one source failure payload once, if the source failed."""
        with self._lock:
            if self._last_error is None or self._failure_reported:
                return None
            self._failure_reported = True
            return {
                "type": "source_failed",
                "kernel": self.spec.name,
                "component_type": "source",
                "kind": self.spec.kind,
                "stream": self.spec.stream,
                "error": self._last_error,
                "traceback": self._traceback,
            }

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                if self._pause_event.is_set():
                    if self._stop_event.wait(self.spec.poll_interval):
                        return
                    continue
                payload = _call_with_optional_timeout(
                    self.source.read,
                    timeout=self._read_timeout,
                    executor=self._executor,
                    label=f"source {self.spec.name!r} read()",
                )
                if payload is None:
                    if self._stop_event.wait(self.spec.poll_interval):
                        return
                    continue
                started = time.perf_counter()
                self.stream.write(payload)
                finished = time.perf_counter()
                with self._lock:
                    self._frames_written += 1
                    self._last_write_time = time.time()
                    self._last_write_duration_s = finished - started
        except BaseException as exc:
            self._logger.exception(
                "source runtime failed: source=%s", self.spec.name
            )
            with self._lock:
                self._last_error = str(exc)
                self._traceback = traceback.format_exc()
        finally:
            try:
                self.source.close()
            except Exception:
                self._logger.exception(
                    "source close failed: source=%s", self.spec.name
                )
            if self._executor is not None:
                self._executor.shutdown(wait=False, cancel_futures=True)


class _SinkController:
    """Manager-owned thread controller for one configured sink plugin."""

    def __init__(self, *, stream: Any, sink: Any, spec: Any, pause_event: Any):
        self.stream = stream
        self.sink = sink
        self.spec = spec
        self._pause_event = pause_event
        self._stop_event = threading.Event()
        self._logger = get_logger(f"sink.{spec.name}")
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run,
            name=f"shmpipeline-sink-{spec.name}",
            daemon=True,
        )
        self._started_at_wall: float | None = None
        self._started_at_mono: float | None = None
        self._last_read_time: float | None = None
        self._last_read_duration_s: float | None = None
        self._frames_consumed = 0
        self._last_error: str | None = None
        self._traceback: str | None = None
        self._failure_reported = False
        self._consume_timeout = getattr(spec, "consume_timeout", None)
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        if self._consume_timeout is not None:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"shmpipeline-sink-consume-{spec.name}",
            )
        self.sink._attach_runtime_events(
            stop_event=self._stop_event,
            pause_event=self._pause_event,
        )

    def start(self) -> None:
        """Start the sink thread."""
        if self._thread.is_alive():
            return
        self.sink.open()
        self._started_at_wall = time.time()
        self._started_at_mono = time.perf_counter()
        self._thread.start()

    def stop(self, *, timeout: float = 2.0) -> None:
        """Request shutdown and wait for the sink thread to exit."""
        self._stop_event.set()
        self._thread.join(timeout=timeout)

    def snapshot(self) -> dict[str, Any]:
        """Return a stable status snapshot for one sink plugin."""
        with self._lock:
            elapsed_s = 0.0
            if self._started_at_mono is not None:
                elapsed_s = max(
                    0.0, time.perf_counter() - self._started_at_mono
                )
            effective_rate_hz = 0.0
            if elapsed_s > 0.0:
                effective_rate_hz = self._frames_consumed / elapsed_s
            return {
                "name": self.spec.name,
                "kind": self.spec.kind,
                "stream": self.spec.stream,
                "read_timeout": self.spec.read_timeout,
                "pause_sleep": self.spec.pause_sleep,
                "alive": self._thread.is_alive(),
                "frames_consumed": self._frames_consumed,
                "effective_rate_hz": effective_rate_hz,
                "started_at": self._started_at_wall,
                "last_read_time": self._last_read_time,
                "last_read_duration_ms": (
                    None
                    if self._last_read_duration_s is None
                    else 1000.0 * self._last_read_duration_s
                ),
                "last_error": self._last_error,
            }

    def consume_failure(self) -> dict[str, Any] | None:
        """Return one sink failure payload once, if the sink failed."""
        with self._lock:
            if self._last_error is None or self._failure_reported:
                return None
            self._failure_reported = True
            return {
                "type": "sink_failed",
                "kernel": self.spec.name,
                "component_type": "sink",
                "kind": self.spec.kind,
                "stream": self.spec.stream,
                "error": self._last_error,
                "traceback": self._traceback,
            }

    def _run(self) -> None:
        try:
            last_seen_count = self.stream.count
            while not self._stop_event.is_set():
                if self._pause_event.is_set():
                    if self._stop_event.wait(self.spec.pause_sleep):
                        return
                    continue
                try:
                    current_count, payload = _read_sink_payload(
                        self.stream,
                        last_seen_count,
                        timeout=self.spec.read_timeout,
                    )
                except TimeoutError:
                    continue
                if current_count <= last_seen_count:
                    continue
                started = time.perf_counter()
                _call_with_optional_timeout(
                    lambda: self.sink.consume(payload),
                    timeout=self._consume_timeout,
                    executor=self._executor,
                    label=f"sink {self.spec.name!r} consume()",
                )
                finished = time.perf_counter()
                last_seen_count = current_count
                with self._lock:
                    self._frames_consumed += 1
                    self._last_read_time = time.time()
                    self._last_read_duration_s = finished - started
        except BaseException as exc:
            self._logger.exception(
                "sink runtime failed: sink=%s", self.spec.name
            )
            with self._lock:
                self._last_error = str(exc)
                self._traceback = traceback.format_exc()
        finally:
            try:
                self.sink.close()
            except Exception:
                self._logger.exception(
                    "sink close failed: sink=%s", self.spec.name
                )
            if self._executor is not None:
                self._executor.shutdown(wait=False, cancel_futures=True)


class PipelineManager:
    """Create shared memory, spawn workers, and supervise pipeline state.

    The manager owns the pipeline lifecycle from validated config through
    runtime monitoring and teardown. It is the primary user-facing runtime API
    for Python callers.
    """

    def __init__(
        self,
        config: PipelineConfig | str | Path,
        *,
        spawn_method: str = "spawn",
        placement_policy: WorkerPlacementPolicy | None = None,
        registry: KernelRegistry | None = None,
        worker_start_timeout: float = 10.0,
    ) -> None:
        """Initialize a manager from a config object or YAML path.

        Parameters
        ----------
        config:
            Pipeline config object, YAML path, or str path.
        spawn_method:
            Multiprocessing start method. Default is ``"spawn"``.
        placement_policy:
            CPU affinity policy for worker processes.
        registry:
            Custom kernel registry. Defaults to the built-in registry.
        worker_start_timeout:
            Seconds to wait for each worker process to report a started event
            before raising :class:`~shmpipeline.errors.WorkerProcessError`.
            Increase this when running on loaded systems where Numba JIT
            compilation takes longer than usual. Default is 10 seconds.
        """
        if isinstance(config, (str, Path)):
            config = PipelineConfig.from_yaml(config)
        self.config = config
        self.registry = registry or get_default_registry()
        self._runtime_registry = registry
        self.context = mp.get_context(spawn_method)
        self.placement_policy = normalize_placement_policy(placement_policy)
        self._logger = get_logger("manager")
        self.state = PipelineState.INITIALIZED
        self._streams: dict[str, Any] = {}
        self._owned_streams: set[str] = set()
        self._pause_event: Any | None = None
        self._workers: dict[str, _WorkerHandle] = {}
        self._failures: list[dict[str, Any]] = []
        self._events: deque[dict[str, Any]] = deque(maxlen=256)
        self._worker_metrics: dict[str, dict[str, Any]] = {}
        self._worker_runtime: dict[str, dict[str, Any]] = {}
        self._synthetic_sources: dict[str, Any] = {}
        self._sources: dict[str, _SourceController] = {}
        self._sinks: dict[str, _SinkController] = {}
        self._kernel_configs = {
            kernel.name: kernel for kernel in self.config.kernels
        }
        self._source_configs = {
            source.name: source for source in self.config.sources
        }
        self._sink_configs = {sink.name: sink for sink in self.config.sinks}
        self._worker_start_timeout = float(worker_start_timeout)
        self._poll_lock = threading.Lock()
        self._logger.info(
            "manager initialized: spawn_method=%s placement_policy=%s "
            "kernels=%d sources=%d sinks=%d streams=%d",
            spawn_method,
            self.placement_policy.describe(),
            len(self.config.kernels),
            len(self.config.sources),
            len(self.config.sinks),
            len(self.config.shared_memory),
        )

    @property
    def graph(self) -> PipelineGraph:
        """Return a derived graph view of the current configuration."""
        return PipelineGraph.from_config(self.config)

    def _transition_state(
        self, new_state: PipelineState, *, reason: str
    ) -> None:
        """Update manager state and emit a state-transition log."""
        old_state = self.state
        self.state = new_state
        self._logger.info(
            "state transition: %s -> %s (%s)",
            old_state.value,
            new_state.value,
            reason,
        )

    def build(self) -> None:
        """Validate configuration and create shared-memory resources.

        This step prepares the pipeline graph and opens or creates the named
        streams without starting any worker processes.
        """
        if self.state not in {
            PipelineState.INITIALIZED,
            PipelineState.STOPPED,
        }:
            raise StateTransitionError(
                f"cannot build pipeline while state is {self.state.value!r}"
            )
        self._logger.info("build started")
        self._close_event_readers()
        self._events.clear()
        self._failures.clear()
        self._worker_metrics.clear()
        self._worker_runtime.clear()
        shared_by_name = self.config.shared_memory_by_name
        self._logger.info("validating pipeline config")
        graph_errors = self.graph.validation_errors()
        if graph_errors:
            raise ConfigValidationError(graph_errors[0])
        for kernel_config in self.config.kernels:
            self.registry.validate(kernel_config, shared_by_name)
            self._logger.info(
                "validated kernel: name=%s kind=%s input=%s output=%s "
                "auxiliary=%s",
                kernel_config.name,
                kernel_config.kind,
                kernel_config.input,
                kernel_config.output,
                kernel_config.auxiliary_by_alias,
            )
        for source_config in self.config.sources:
            self.registry.validate_source(source_config, shared_by_name)
            self._logger.info(
                "validated source: name=%s kind=%s stream=%s auxiliary=%s",
                source_config.name,
                source_config.kind,
                source_config.stream,
                source_config.auxiliary_by_alias,
            )
        for sink_config in self.config.sinks:
            self.registry.validate_sink(sink_config, shared_by_name)
            self._logger.info(
                "validated sink: name=%s kind=%s stream=%s auxiliary=%s",
                sink_config.name,
                sink_config.kind,
                sink_config.stream,
                sink_config.auxiliary_by_alias,
            )
        for spec in self.config.shared_memory:
            self._streams[spec.name] = self._build_stream(spec)
        self._transition_state(PipelineState.BUILT, reason="build complete")

    def _build_stream(self, spec) -> Any:
        """Create, attach, or replace one stream according to ``spec.mode``."""
        existing_stream = self._open_existing_stream(spec)
        if existing_stream is not None:
            if spec.mode == "create":
                existing_stream.close()
                raise FileExistsError(spec.name)
            if spec.mode == "replace":
                close_stream(existing_stream, unlink=True)
            elif self._stream_matches_spec(existing_stream, spec):
                self._logger.info(
                    "reusing shared memory: name=%s storage=%s shape=%s "
                    "dtype=%s",
                    spec.name,
                    spec.storage,
                    spec.shape,
                    spec.dtype,
                )
                return existing_stream
            elif spec.mode == "attach":
                existing_stream.close()
                raise ConfigValidationError(
                    f"existing shared memory {spec.name!r} does not match "
                    "the requested attach configuration"
                )
            else:
                close_stream(existing_stream, unlink=True)
        elif spec.mode == "attach":
            raise FileNotFoundError(
                f"shared memory stream {spec.name!r} does not exist"
            )

        create_kwargs = {
            "shape": spec.shape,
            "dtype": spec.dtype,
            "notify": (
                self._stream_should_notify(spec.name)
                if spec.notify is None
                else spec.notify
            ),
        }
        if spec.storage == "gpu":
            create_kwargs["gpu_device"] = spec.gpu_device
            create_kwargs["cpu_mirror"] = spec.cpu_mirror

        try:
            stream = pyshmem.create(spec.name, **create_kwargs)
        except FileExistsError:
            self._logger.info(
                "shared memory already exists during create; retrying attach: name=%s",
                spec.name,
            )
            existing_stream = self._open_existing_stream(spec)
            if existing_stream is not None:
                if spec.mode == "create":
                    existing_stream.close()
                    raise FileExistsError(spec.name)
                if spec.mode == "attach" and not self._stream_matches_spec(
                    existing_stream, spec
                ):
                    existing_stream.close()
                    raise ConfigValidationError(
                        f"existing shared memory {spec.name!r} does not "
                        "match the requested attach configuration"
                    )
                if spec.mode != "replace" and self._stream_matches_spec(
                    existing_stream, spec
                ):
                    return existing_stream
                close_stream(existing_stream, unlink=True)
            else:
                if spec.mode == "create":
                    raise FileExistsError(spec.name)
                if spec.mode == "attach":
                    raise FileNotFoundError(
                        f"shared memory stream {spec.name!r} does not exist"
                    )
                self._logger.warning(
                    "shared memory exists but is not attachable; recreating stale stream: name=%s",
                    spec.name,
                )
                unlink_stream_name(spec.name)
            stream = pyshmem.create(spec.name, **create_kwargs)

        self._logger.info(
            "created shared memory: name=%s storage=%s shape=%s dtype=%s",
            spec.name,
            spec.storage,
            spec.shape,
            spec.dtype,
        )
        self._owned_streams.add(spec.name)
        return stream

    def _stream_should_notify(self, name: str) -> bool:
        """Return the default wait-notification policy for one stream."""
        if any(kernel.input == name for kernel in self.config.kernels):
            return True
        return any(sink.stream == name for sink in self.config.sinks)

    def _open_existing_stream(self, spec) -> Any | None:
        """Open an existing stream if present.

        Metadata is inspected first so a dead GPU producer is not needlessly
        mapped.  A live GPU stream is then opened with pyshmem's recorded
        device resolution.
        """
        try:
            info = pyshmem.stat(spec.name)
            if info["gpu_enabled"] and not info["creator_alive"]:
                if not info["cpu_mirror"]:
                    return None
                return pyshmem.open(spec.name, gpu_device=False)
            return pyshmem.open(spec.name)
        except FileNotFoundError:
            return None
        except (OSError, RuntimeError, ValueError):
            return None

    def _stream_matches_spec(self, stream: Any, spec) -> bool:
        """Return whether an existing stream matches the requested config."""
        if tuple(stream.shape) != tuple(spec.shape):
            return False
        if stream.dtype != spec.dtype:
            return False
        if stream.gpu_enabled != (spec.storage == "gpu"):
            return False
        if spec.storage == "gpu":
            if spec.gpu_device is not None:
                if stream.gpu_device != spec.gpu_device:
                    return False
            if (
                spec.cpu_mirror is not None
                and stream.cpu_mirror != spec.cpu_mirror
            ):
                return False
        return True

    def start(self) -> None:
        """Start worker processes for every configured kernel.

        The manager must already be in the built state before workers can be
        spawned.
        """
        self.poll_events()
        if self.state != PipelineState.BUILT:
            raise StateTransitionError(
                f"cannot start pipeline while state is {self.state.value!r}"
            )
        self._logger.info("start requested")
        self._pause_event = self.context.Event()
        self._failures.clear()
        self._worker_metrics.clear()
        self._worker_runtime.clear()
        self._close_event_readers()
        cpu_count = max(1, os.cpu_count() or 1)
        try:
            self._start_sinks()
            for index, kernel_config in enumerate(self.config.kernels):
                self._spawn_worker(
                    kernel_config, index=index, cpu_count=cpu_count
                )
            self._wait_for_workers_started(timeout=self._worker_start_timeout)
            self._start_sources()
        except BaseException:
            self._stop_runtime_components(timeout=1.0, force=True)
            raise
        self._transition_state(
            PipelineState.RUNNING, reason="all workers started"
        )

    def _spawn_worker(
        self,
        kernel_config: Any,
        *,
        index: int,
        cpu_count: int | None = None,
    ) -> int | None:
        """Spawn one worker process for ``kernel_config`` and register it.

        Shared between :meth:`start`, :meth:`restart`, and :meth:`add_kernel`.
        Uses the current ``self.config.shared_memory``, so callers adding new
        streams must commit them to ``self.config`` before spawning.
        """
        if cpu_count is None:
            cpu_count = max(1, os.cpu_count() or 1)
        stop_event = self.context.Event()
        event_reader, event_writer = self.context.Pipe(duplex=False)
        cpu_slot = self.placement_policy.cpu_slot_for(
            kernel=kernel_config,
            index=index,
            cpu_count=cpu_count,
        )
        process = self.context.Process(
            target=run_kernel_process,
            args=(
                kernel_config,
                self.config.shared_memory,
                self._pause_event,
                stop_event,
                event_writer,
                cpu_slot,
                self._runtime_registry,
            ),
            name=f"shmpipeline-{kernel_config.name}",
        )
        try:
            process.start()
        except BaseException:
            event_reader.close()
            event_writer.close()
            raise
        event_writer.close()
        self._logger.info(
            "spawned worker: kernel=%s pid=%s cpu_slot=%s",
            kernel_config.name,
            process.pid,
            cpu_slot,
        )
        self._workers[kernel_config.name] = _WorkerHandle(
            name=kernel_config.name,
            process=process,
            stop_event=stop_event,
            event_reader=event_reader,
            cpu_slot=cpu_slot,
        )
        return cpu_slot

    def _start_sources(self) -> None:
        """Start all configured source plugins."""
        shared_by_name = self.config.shared_memory_by_name
        for source_config in self.config.sources:
            source = self.registry.create_source(
                source_config,
                shared_by_name,
                self._streams,
            )
            controller = _SourceController(
                stream=self._streams[source_config.stream],
                source=source,
                spec=source_config,
                pause_event=self._pause_event,
            )
            controller.start()
            self._sources[source_config.name] = controller

    def _start_sinks(self) -> None:
        """Start all configured sink plugins."""
        shared_by_name = self.config.shared_memory_by_name
        for sink_config in self.config.sinks:
            sink = self.registry.create_sink(
                sink_config,
                shared_by_name,
                self._streams,
            )
            controller = _SinkController(
                stream=self._streams[sink_config.stream],
                sink=sink,
                spec=sink_config,
                pause_event=self._pause_event,
            )
            controller.start()
            self._sinks[sink_config.name] = controller

    def _wait_for_workers_started(
        self,
        *,
        timeout: float | None = None,
        only: set[str] | None = None,
    ) -> None:
        """Block until all (or a named subset of) workers report started."""
        effective_timeout = (
            timeout if timeout is not None else self._worker_start_timeout
        )
        expected = only if only is not None else set(self._workers)
        end_time = time.monotonic() + effective_timeout
        while time.monotonic() < end_time:
            self.poll_events()
            started = {
                event["kernel"]
                for event in self._events
                if event.get("type") == "worker_started"
            }
            if started >= expected:
                return
            for name in expected:
                worker = self._workers.get(name)
                if worker is not None and worker.process.exitcode not in (
                    None,
                    0,
                ):
                    self.raise_if_failed()
            time.sleep(0.01)
        started = {
            event["kernel"]
            for event in self._events
            if event.get("type") == "worker_started"
        }
        missing = sorted(expected - started)
        raise WorkerProcessError(
            "timed out waiting for workers to start: " + ", ".join(missing)
        )

    def pause(self) -> None:
        """Pause all workers without tearing down the built pipeline."""
        self.poll_events()
        if self.state != PipelineState.RUNNING:
            raise StateTransitionError(
                f"cannot pause pipeline while state is {self.state.value!r}"
            )
        assert self._pause_event is not None
        self._pause_event.set()
        self._transition_state(PipelineState.PAUSED, reason="pause requested")

    def resume(self) -> None:
        """Resume work after a pause."""
        self.poll_events()
        if self.state != PipelineState.PAUSED:
            raise StateTransitionError(
                f"cannot resume pipeline while state is {self.state.value!r}"
            )
        assert self._pause_event is not None
        self._pause_event.clear()
        self._transition_state(
            PipelineState.RUNNING, reason="resume requested"
        )

    def stop(self, *, timeout: float = 5.0, force: bool = False) -> None:
        """Stop worker processes but keep shared-memory resources built.

        After a stop, callers may inspect stream contents, restart workers, or
        proceed to a final shutdown.
        """
        self.poll_events()
        if self.state not in {
            PipelineState.RUNNING,
            PipelineState.PAUSED,
            PipelineState.FAILED,
        }:
            if self.state == PipelineState.BUILT:
                return
            raise StateTransitionError(
                f"cannot stop pipeline while state is {self.state.value!r}"
            )
        self._logger.info(
            "stop requested: timeout=%s force=%s", timeout, force
        )
        self._stop_runtime_components(timeout=timeout, force=force)
        self._transition_state(PipelineState.BUILT, reason="workers stopped")

    def _stop_sources(self, *, timeout: float) -> None:
        """Stop all configured source plugins."""
        for name, controller in list(self._sources.items()):
            controller.stop(timeout=timeout)
            self._logger.info("source stopped: name=%s", name)
        self._sources.clear()

    def _stop_sinks(self, *, timeout: float) -> None:
        """Stop all configured sink plugins."""
        for name, controller in list(self._sinks.items()):
            controller.stop(timeout=timeout)
            self._logger.info("sink stopped: name=%s", name)
        self._sinks.clear()

    def _stop_runtime_components(
        self,
        *,
        timeout: float,
        force: bool,
    ) -> None:
        """Stop sources, workers, and sinks without a state transition."""
        self._stop_sources(timeout=timeout)
        for worker in self._workers.values():
            worker.stop_event.set()
        for worker in self._workers.values():
            worker.process.join(timeout)
            if worker.process.is_alive():
                if force and hasattr(worker.process, "kill"):
                    worker.process.kill()
                else:
                    worker.process.terminate()
                worker.process.join(timeout)
            self._logger.info(
                "worker stopped: kernel=%s pid=%s exitcode=%s",
                worker.name,
                worker.process.pid,
                worker.process.exitcode,
            )
        self._stop_sinks(timeout=timeout)
        self.poll_events()
        self._close_event_readers()
        self._workers.clear()
        self._pause_event = None

    def shutdown(
        self,
        *,
        unlink: bool = True,
        unlink_external: bool = False,
        force: bool = False,
    ) -> None:
        """Stop workers, close local handles, and optionally unlink streams.

        This is the terminal cleanup step for a manager instance.  By default
        only streams created by this manager are unlinked; attached external
        streams are merely closed.  Set ``unlink_external=True`` for the
        explicit administrative cleanup behavior.
        """
        self._logger.info(
            "shutdown requested: unlink=%s unlink_external=%s force=%s",
            unlink,
            unlink_external,
            force,
        )
        self.stop_all_synthetic_inputs()
        if self.state in {
            PipelineState.RUNNING,
            PipelineState.PAUSED,
            PipelineState.FAILED,
        }:
            self.stop(force=force)
        for name, stream in list(self._streams.items()):
            try:
                close_stream(
                    stream,
                    unlink=unlink
                    and (unlink_external or name in self._owned_streams),
                )
            finally:
                self._logger.info(
                    "released shared memory: name=%s unlink=%s",
                    name,
                    unlink,
                )
        self._streams.clear()
        self._owned_streams.clear()
        self._transition_state(
            PipelineState.STOPPED, reason="shutdown complete"
        )

    def poll_events(self) -> list[dict[str, Any]]:
        """Drain worker events and update manager failure state.

        Thread-safe: concurrent calls from multiple threads (e.g. a GUI
        polling thread and a status endpoint) are serialised internally.
        """
        with self._poll_lock:
            return self._poll_events_locked()

    def _poll_events_locked(self) -> list[dict[str, Any]]:
        """Inner poll_events implementation; caller must hold _poll_lock."""
        events = drain_events(
            [
                worker.event_reader
                for worker in self._workers.values()
                if worker.event_reader is not None
            ]
        )
        if events:
            for event in events:
                self._events.append(event)
                self._log_event(event)
                kernel_name = event.get("kernel")
                if event.get("type") == "worker_metrics":
                    observed_at = time.time()
                    self._worker_metrics[event["kernel"]] = {
                        "pid": event.get("pid"),
                        "cpu_slot": event.get("cpu_slot"),
                        "frames_processed": event.get("frames_processed", 0),
                        "last_exec_ms": event.get("last_exec_ms"),
                        "last_exec_us": event.get("last_exec_us"),
                        "avg_exec_ms": event.get("avg_exec_ms"),
                        "avg_exec_us": event.get("avg_exec_us"),
                        "jitter_us_rms": event.get("jitter_us_rms"),
                        "throughput_hz": event.get("throughput_hz"),
                        "runtime_s": event.get("runtime_s"),
                        "last_output_count": event.get("last_output_count"),
                        "metrics_window": event.get("metrics_window"),
                    }
                    runtime = self._worker_runtime.setdefault(
                        event["kernel"],
                        {},
                    )
                    runtime.setdefault(
                        "started_at",
                        observed_at - float(event.get("runtime_s") or 0.0),
                    )
                    runtime["last_metric_at"] = observed_at
                    previous_frames = int(runtime.get("frames_processed") or 0)
                    previous_output_count = runtime.get("last_output_count")
                    current_frames = int(event.get("frames_processed") or 0)
                    current_output_count = event.get("last_output_count")
                    if (
                        current_frames > previous_frames
                        or previous_output_count is None
                        or current_output_count is None
                        or current_output_count > previous_output_count
                    ):
                        runtime["last_progress_at"] = observed_at
                    runtime["frames_processed"] = current_frames
                    runtime["last_output_count"] = current_output_count
                elif event.get("type") == "worker_started" and kernel_name:
                    runtime = self._worker_runtime.setdefault(kernel_name, {})
                    runtime["started_at"] = time.time()
                elif event.get("type") == "worker_failed":
                    self._record_worker_failure(event)
        for worker in self._workers.values():
            if worker.process.exitcode not in (None, 0):
                already_recorded = any(
                    failure.get("kernel") == worker.name
                    for failure in self._failures
                )
                if not already_recorded:
                    self._record_worker_failure(
                        {
                            "type": "worker_failed",
                            "kernel": worker.name,
                            "pid": worker.process.pid,
                            "error": (
                                "worker exited with code "
                                f"{worker.process.exitcode}"
                            ),
                        }
                    )
        for controller in self._sources.values():
            failure = controller.consume_failure()
            if failure is not None:
                self._events.append(failure)
                self._log_event(failure)
                self._record_worker_failure(failure)
        for controller in self._sinks.values():
            failure = controller.consume_failure()
            if failure is not None:
                self._events.append(failure)
                self._log_event(failure)
                self._record_worker_failure(failure)
        if self._failures and self.state in {
            PipelineState.RUNNING,
            PipelineState.PAUSED,
        }:
            self._transition_state(
                PipelineState.FAILED,
                reason="worker failure recorded",
            )
        return events

    def _close_event_readers(self) -> None:
        """Close any open manager-side worker event readers."""
        for worker in self._workers.values():
            event_reader = worker.event_reader
            if event_reader is None:
                continue
            try:
                event_reader.close()
            except OSError:
                pass
            worker.event_reader = None

    def _log_event(self, event: dict[str, Any]) -> None:
        """Write manager-readable logs for worker lifecycle events."""
        event_type = event.get("type")
        if event_type == "worker_started":
            self._logger.info(
                "worker started: kernel=%s pid=%s cpu_slot=%s",
                event.get("kernel"),
                event.get("pid"),
                event.get("cpu_slot"),
            )
            return
        if event_type == "worker_stopped":
            self._logger.info(
                "worker stopped event: kernel=%s pid=%s cpu_slot=%s",
                event.get("kernel"),
                event.get("pid"),
                event.get("cpu_slot"),
            )
            return
        if event_type == "worker_failed":
            self._logger.error(
                "worker failed: kernel=%s pid=%s error=%s",
                event.get("kernel"),
                event.get("pid"),
                event.get("error"),
            )
            return
        if event_type in {"source_failed", "sink_failed"}:
            self._logger.error(
                "%s failed: name=%s stream=%s error=%s",
                event_type.split("_", 1)[0],
                event.get("kernel"),
                event.get("stream"),
                event.get("error"),
            )
            return
        if event_type == "worker_metrics":
            self._logger.debug(
                "worker metrics: kernel=%s frames=%s avg_exec_us=%s "
                "jitter_us_rms=%s throughput_hz=%s",
                event.get("kernel"),
                event.get("frames_processed"),
                event.get("avg_exec_us"),
                event.get("jitter_us_rms"),
                event.get("throughput_hz"),
            )
            return
        self._logger.debug("worker event: %s", event)

    def _record_worker_failure(self, failure: dict[str, Any]) -> None:
        """Record one worker failure, preferring detailed failures."""
        kernel_name = failure.get("kernel")
        if kernel_name is None:
            self._failures.append(failure)
            return

        existing_index: int | None = None
        for index, existing_failure in enumerate(self._failures):
            if existing_failure.get("kernel") == kernel_name:
                existing_index = index
                break

        if existing_index is None:
            self._failures.append(failure)
            return

        existing_failure = self._failures[existing_index]
        if self._failure_is_more_specific(failure, existing_failure):
            self._failures[existing_index] = failure

    @staticmethod
    def _failure_is_more_specific(
        candidate: dict[str, Any], existing: dict[str, Any]
    ) -> bool:
        """Return whether a candidate failure is more useful than existing."""
        candidate_has_traceback = bool(candidate.get("traceback"))
        existing_has_traceback = bool(existing.get("traceback"))
        if candidate_has_traceback != existing_has_traceback:
            return candidate_has_traceback

        candidate_error = str(candidate.get("error", ""))
        existing_error = str(existing.get("error", ""))
        candidate_is_generic = candidate_error.startswith(
            "worker exited with code"
        )
        existing_is_generic = existing_error.startswith(
            "worker exited with code"
        )
        if candidate_is_generic != existing_is_generic:
            return not candidate_is_generic
        return False

    def restart(self, *, timeout: float | None = None) -> None:
        """Restart failed or dead worker processes without stopping the pipeline.

        Only the workers that have failed or exited are restarted; healthy
        workers continue running uninterrupted. Shared-memory streams and
        metric history are preserved across the restart.

        After a successful restart of all failed workers the pipeline
        transitions from ``FAILED`` back to ``RUNNING`` (or remains ``PAUSED``
        if it was paused before failure).

        Parameters
        ----------
        timeout:
            Seconds to wait for restarted workers to report started.
            Defaults to the manager's ``worker_start_timeout``.
        """
        self.poll_events()
        if self.state not in {
            PipelineState.RUNNING,
            PipelineState.PAUSED,
            PipelineState.FAILED,
        }:
            raise StateTransitionError(
                f"cannot restart workers while state is {self.state.value!r}"
            )

        failed_names = {f["kernel"] for f in self._failures if f.get("kernel")}
        dead_names = {
            name
            for name, w in self._workers.items()
            if not w.process.is_alive()
        }
        to_restart = failed_names | dead_names

        if not to_restart:
            self._logger.info("restart called but no failed workers found")
            return

        effective_timeout = (
            timeout if timeout is not None else self._worker_start_timeout
        )
        cpu_count = max(1, os.cpu_count() or 1)
        kernel_index = {k.name: i for i, k in enumerate(self.config.kernels)}

        for kernel_name in to_restart:
            worker = self._workers.get(kernel_name)
            if worker is not None:
                worker.stop_event.set()
                worker.process.join(timeout=1.0)
                if worker.process.is_alive():
                    worker.process.kill()
                    worker.process.join(1.0)
                if worker.event_reader is not None:
                    try:
                        worker.event_reader.close()
                    except OSError:
                        pass

            self._failures = [
                f for f in self._failures if f.get("kernel") != kernel_name
            ]

            kernel_config = self._kernel_configs[kernel_name]
            index = kernel_index.get(kernel_name, 0)
            self._spawn_worker(kernel_config, index=index, cpu_count=cpu_count)

        self._wait_for_workers_started(
            timeout=effective_timeout, only=to_restart
        )

        if self.state == PipelineState.FAILED and not self._failures:
            self._transition_state(
                PipelineState.RUNNING, reason="failed workers restarted"
            )

    def add_kernel(
        self,
        kernel: "KernelConfig | dict[str, Any]",
        *,
        shared_memory: "tuple[Any, ...] | list[Any]" = (),
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Add one kernel to a running pipeline without stopping it.

        Spawns a new worker for ``kernel`` and creates any new shared-memory
        streams it needs, leaving existing workers and streams untouched.  This
        enables hot-reloading additional stages into a live pipeline.

        Parameters
        ----------
        kernel:
            A :class:`~shmpipeline.config.KernelConfig` (or mapping) for the new
            stage.
        shared_memory:
            New stream definitions the kernel introduces.  Streams that already
            exist are referenced as-is; only genuinely new names are created.
        timeout:
            Seconds to wait for the new worker to report started.  Defaults to
            the manager's ``worker_start_timeout``.

        Returns the manager status snapshot after the kernel is running.
        """
        self.poll_events()
        if self.state not in {PipelineState.RUNNING, PipelineState.PAUSED}:
            raise StateTransitionError(
                "add_kernel requires a RUNNING or PAUSED pipeline; "
                f"current state is {self.state.value!r}"
            )
        if isinstance(kernel, Mapping):
            kernel = KernelConfig.from_dict(kernel)

        new_specs: list[SharedMemoryConfig] = []
        for spec in shared_memory:
            if isinstance(spec, Mapping):
                spec = SharedMemoryConfig.from_dict(spec)
            new_specs.append(spec)

        existing_nodes = {
            *self._kernel_configs,
            *self._source_configs,
            *self._sink_configs,
        }
        if kernel.name in existing_nodes:
            raise ConfigValidationError(
                f"pipeline node {kernel.name!r} already exists"
            )
        for spec in new_specs:
            if spec.name in self._streams:
                raise ConfigValidationError(
                    f"shared memory {spec.name!r} already exists"
                )

        # Build and validate the candidate configuration as a whole. The
        # frozen PipelineConfig re-runs full reference/uniqueness validation,
        # and the graph check rejects duplicate producers (fan-in conflicts).
        candidate = replace(
            self.config,
            shared_memory=(*self.config.shared_memory, *new_specs),
            kernels=(*self.config.kernels, kernel),
        )
        graph_errors = PipelineGraph.from_config(candidate).validation_errors()
        if graph_errors:
            raise ConfigValidationError(graph_errors[0])
        self.registry.validate(kernel, candidate.shared_memory_by_name)

        # Create only the genuinely new streams, then commit the config so the
        # spawned worker (and graph/status) observe the extended pipeline.
        previous_config = self.config
        for spec in new_specs:
            self._streams[spec.name] = self._build_stream(spec)
        self.config = candidate
        self._kernel_configs[kernel.name] = kernel

        self._logger.info(
            "adding kernel to running pipeline: name=%s kind=%s",
            kernel.name,
            kernel.kind,
        )
        index = len(self.config.kernels) - 1
        try:
            self._spawn_worker(kernel, index=index)
            self._wait_for_workers_started(
                timeout=timeout or self._worker_start_timeout,
                only={kernel.name},
            )
        except BaseException:
            self._rollback_added_kernel(
                kernel.name, new_specs, previous_config
            )
            raise
        return self.status()

    def _rollback_added_kernel(
        self,
        kernel_name: str,
        new_specs: list[SharedMemoryConfig],
        previous_config: PipelineConfig,
    ) -> None:
        """Undo a partially applied :meth:`add_kernel` after a spawn failure."""
        worker = self._workers.pop(kernel_name, None)
        if worker is not None:
            worker.stop_event.set()
            worker.process.join(timeout=1.0)
            if worker.process.is_alive() and hasattr(worker.process, "kill"):
                worker.process.kill()
                worker.process.join(1.0)
            if worker.event_reader is not None:
                try:
                    worker.event_reader.close()
                except OSError:
                    pass
        self._failures = [
            f for f in self._failures if f.get("kernel") != kernel_name
        ]
        self._kernel_configs.pop(kernel_name, None)
        for spec in new_specs:
            stream = self._streams.pop(spec.name, None)
            if stream is not None:
                close_stream(stream, unlink=True)
        self.config = previous_config

    def status(self, *, poll: bool = True) -> dict[str, Any]:
        """Return a snapshot of manager state, workers, and failures.

        The result is intentionally JSON-friendly so CLI, GUI, and external
        tooling can consume the same structure.
        """
        if poll:
            self.poll_events()
        workers_status = {
            name: self._status_for_worker(name, worker)
            for name, worker in self._workers.items()
        }
        sources_status = self.source_status()
        sinks_status = self.sink_status()
        return {
            "state": self.state.value,
            "workers": workers_status,
            "sources": sources_status,
            "sinks": sinks_status,
            "failures": list(self._failures),
            "metrics": {
                name: dict(metrics)
                for name, metrics in self._worker_metrics.items()
            },
            "synthetic_sources": self.synthetic_input_status(),
            "placement_policy": self.placement_policy.describe(),
            "summary": self._status_summary(
                workers_status,
                sources_status,
                sinks_status,
            ),
        }

    def source_status(self) -> dict[str, dict[str, Any]]:
        """Return status snapshots for configured source plugins."""
        return {
            name: controller.snapshot()
            for name, controller in self._sources.items()
        }

    def sink_status(self) -> dict[str, dict[str, Any]]:
        """Return status snapshots for configured sink plugins."""
        return {
            name: controller.snapshot()
            for name, controller in self._sinks.items()
        }

    def _status_for_worker(
        self, name: str, worker: _WorkerHandle
    ) -> dict[str, Any]:
        metrics = dict(self._worker_metrics.get(name, {}))
        health, idle_s, last_metric_age_s = self._worker_health(name, worker)
        status = {
            "pid": worker.process.pid,
            "alive": worker.process.is_alive(),
            "exitcode": worker.process.exitcode,
            "cpu_slot": worker.cpu_slot,
            "health": health,
            "idle_s": idle_s,
            "last_metric_age_s": last_metric_age_s,
            **metrics,
        }
        return status

    def _worker_health(
        self, name: str, worker: _WorkerHandle
    ) -> tuple[str, float | None, float | None]:
        now = time.time()
        runtime = self._worker_runtime.get(name, {})
        started_at = runtime.get("started_at")
        last_progress_at = runtime.get("last_progress_at")
        last_metric_at = runtime.get("last_metric_at")
        frames_processed = int(
            self._worker_metrics.get(name, {}).get("frames_processed") or 0
        )

        if any(failure.get("kernel") == name for failure in self._failures):
            idle_s = None
            if last_progress_at is not None:
                idle_s = max(0.0, now - float(last_progress_at))
            last_metric_age_s = None
            if last_metric_at is not None:
                last_metric_age_s = max(0.0, now - float(last_metric_at))
            return "failed", idle_s, last_metric_age_s

        if not worker.process.is_alive():
            return "stopped", None, None

        if self.state == PipelineState.PAUSED:
            idle_s = None
            if last_progress_at is not None:
                idle_s = max(0.0, now - float(last_progress_at))
            last_metric_age_s = None
            if last_metric_at is not None:
                last_metric_age_s = max(0.0, now - float(last_metric_at))
            return "paused", idle_s, last_metric_age_s

        if started_at is None:
            return "starting", None, None

        reference_at = last_progress_at or started_at
        idle_s = max(0.0, now - float(reference_at))
        last_metric_age_s = None
        if last_metric_at is not None:
            last_metric_age_s = max(0.0, now - float(last_metric_at))

        if frames_processed <= 0:
            return "waiting-input", idle_s, last_metric_age_s

        kernel_config = self._kernel_configs.get(name)
        idle_threshold_s = 1.0
        if kernel_config is not None:
            idle_threshold_s = max(1.0, 4.0 * kernel_config.read_timeout)
        if self.state == PipelineState.RUNNING and idle_s > idle_threshold_s:
            return "idle", idle_s, last_metric_age_s
        return "active", idle_s, last_metric_age_s

    def _status_summary(
        self,
        workers_status: dict[str, dict[str, Any]],
        sources_status: dict[str, dict[str, Any]],
        sinks_status: dict[str, dict[str, Any]],
    ) -> dict[str, int]:
        summary = {
            "workers_total": len(workers_status),
            "active_workers": 0,
            "idle_workers": 0,
            "waiting_workers": 0,
            "paused_workers": 0,
            "failed_workers": 0,
            "stopped_workers": 0,
            "sources_total": len(sources_status),
            "active_sources": 0,
            "failed_sources": 0,
            "sinks_total": len(sinks_status),
            "active_sinks": 0,
            "failed_sinks": 0,
        }
        for worker in workers_status.values():
            health = worker.get("health")
            if health == "active":
                summary["active_workers"] += 1
            elif health == "idle":
                summary["idle_workers"] += 1
            elif health in {"waiting-input", "starting"}:
                summary["waiting_workers"] += 1
            elif health == "paused":
                summary["paused_workers"] += 1
            elif health == "failed":
                summary["failed_workers"] += 1
            elif health == "stopped":
                summary["stopped_workers"] += 1
        for source in sources_status.values():
            if source.get("last_error"):
                summary["failed_sources"] += 1
            elif source.get("alive"):
                summary["active_sources"] += 1
        for sink in sinks_status.values():
            if sink.get("last_error"):
                summary["failed_sinks"] += 1
            elif sink.get("alive"):
                summary["active_sinks"] += 1
        return summary

    def runtime_snapshot(self, *, poll: bool = True) -> dict[str, Any]:
        """Return a richer status snapshot for CLI and GUI consumers.

        This extends :meth:`status` with a timestamp and the derived graph
        description.
        """
        status = self.status(poll=poll)
        return {
            "timestamp": time.time(),
            **status,
            "graph": self.graph.to_dict(),
        }

    def raise_if_failed(self) -> None:
        """Raise the first worker failure, if any has been recorded."""
        self.poll_events()
        if not self._failures:
            return
        failure = self._failures[0]
        component_type = failure.get("component_type", "worker")
        message = (
            f"{component_type} {failure['kernel']!r} failed: "
            f"{failure['error']}"
        )
        raise WorkerProcessError(message)

    def get_stream(self, name: str):
        """Return the manager-owned shared-memory handle for one stream."""
        return self._streams[name]

    def start_synthetic_input(
        self,
        spec: "SyntheticInputConfig | dict[str, Any]",
    ) -> dict[str, Any]:
        """Start a synthetic input writer for one built stream.

        Synthetic writers are useful for demos, viewer testing, smoke tests,
        and deterministic regression scenarios.
        """
        from shmpipeline.synthetic import (
            SyntheticInputConfig,
            SyntheticSourceController,
        )

        if self.state not in {
            PipelineState.BUILT,
            PipelineState.RUNNING,
            PipelineState.PAUSED,
            PipelineState.FAILED,
        }:
            raise StateTransitionError(
                "synthetic inputs require the pipeline to be built first"
            )
        if isinstance(spec, dict):
            spec = SyntheticInputConfig(**spec)
        if spec.stream_name not in self._streams:
            raise KeyError(
                f"unknown shared memory stream: {spec.stream_name!r}"
            )
        configured_source = next(
            (
                source.name
                for source in self.config.sources
                if source.stream == spec.stream_name
            ),
            None,
        )
        if configured_source is not None:
            raise ConfigValidationError(
                f"stream {spec.stream_name!r} is already driven by source "
                f"{configured_source!r}"
            )
        self.stop_synthetic_input(spec.stream_name)
        controller = SyntheticSourceController(
            self._streams[spec.stream_name], spec
        )
        controller.start()
        self._synthetic_sources[spec.stream_name] = controller
        self._logger.info(
            "synthetic input started: stream=%s pattern=%s rate_hz=%s",
            spec.stream_name,
            spec.pattern,
            spec.rate_hz,
        )
        return controller.snapshot()

    def stop_synthetic_input(
        self, stream_name: str, *, timeout: float = 2.0
    ) -> None:
        """Stop one active synthetic input writer if it exists."""
        controller = self._synthetic_sources.pop(stream_name, None)
        if controller is None:
            return
        controller.stop(timeout=timeout)
        self._logger.info("synthetic input stopped: stream=%s", stream_name)

    def stop_all_synthetic_inputs(self, *, timeout: float = 2.0) -> None:
        """Stop every active synthetic input writer."""
        for stream_name in list(self._synthetic_sources):
            self.stop_synthetic_input(stream_name, timeout=timeout)

    def synthetic_input_status(self) -> dict[str, dict[str, Any]]:
        """Return status snapshots for active synthetic input writers."""
        return {
            stream_name: controller.snapshot()
            for stream_name, controller in self._synthetic_sources.items()
        }

    def _terminal_streams(self) -> list[str]:
        """Return output streams that no kernel or sink consumes."""
        consumed: set[str] = set()
        for kernel in self.config.kernels:
            consumed.update(kernel.all_inputs)
        for sink in self.config.sinks:
            consumed.add(sink.stream)
            consumed.update(sink.auxiliary_names)
        return [
            kernel.output
            for kernel in self.config.kernels
            if kernel.output not in consumed
        ]

    def benchmark(
        self,
        *,
        duration_s: float = 5.0,
        source: "SyntheticInputConfig | dict[str, Any] | None" = None,
        output_stream: str | None = None,
        warmup_s: float = 0.5,
        poll_interval: float = 1e-4,
    ) -> dict[str, Any]:
        """Drive the running pipeline and measure throughput and latency.

        The pipeline must already be ``RUNNING``.  When ``source`` is given, a
        synthetic input writer is started on its stream for the duration of the
        benchmark (and stopped afterwards); otherwise the pipeline's existing
        sources drive it.  Frame arrivals are sampled at ``output_stream`` (the
        graph's single terminal output by default) to compute throughput and
        inter-frame latency percentiles.

        Returns a JSON-friendly report with ``throughput_hz``, ``frames``,
        latency percentiles in milliseconds, and per-worker rolling metrics.
        Raises if a worker failed during the run.
        """
        if self.state != PipelineState.RUNNING:
            raise StateTransitionError(
                "benchmark requires the pipeline to be RUNNING"
            )
        if duration_s <= 0.0:
            raise ValueError("duration_s must be positive")

        if output_stream is None:
            terminals = self._terminal_streams()
            if len(terminals) != 1:
                raise ValueError(
                    "output_stream must be specified explicitly: the pipeline "
                    f"has {len(terminals)} terminal output streams "
                    f"({sorted(terminals)})"
                )
            output_stream = terminals[0]
        stream = self.get_stream(output_stream)

        started_synthetic: str | None = None
        if source is not None:
            from shmpipeline.synthetic import SyntheticInputConfig

            if isinstance(source, dict):
                source = SyntheticInputConfig(**source)
            self.start_synthetic_input(source)
            started_synthetic = source.stream_name

        try:
            warmup_deadline = time.monotonic() + max(0.0, warmup_s)
            while time.monotonic() < warmup_deadline:
                self.poll_events()
                time.sleep(poll_interval)

            intervals: list[float] = []
            last_count = stream.count
            start = time.monotonic()
            prev_time = start
            deadline = start + duration_s
            while time.monotonic() < deadline:
                self.poll_events()
                remaining = max(0.0, deadline - time.monotonic())
                try:
                    current = stream.wait_for_count(
                        after=last_count,
                        timeout=remaining,
                        poll_interval=poll_interval,
                    )
                except TimeoutError:
                    break
                now = time.monotonic()
                advanced = int(current - last_count)
                spacing = (now - prev_time) / advanced
                intervals.extend([spacing] * advanced)
                last_count = current
                prev_time = now
            elapsed = max(time.monotonic() - start, 1e-12)
        finally:
            if started_synthetic is not None:
                self.stop_synthetic_input(started_synthetic)

        self.raise_if_failed()

        frames = len(intervals)
        latencies_ms = np.asarray(intervals, dtype=np.float64) * 1000.0
        percentiles: dict[str, float] = {}
        if frames:
            for label, q in (("p50", 50), ("p90", 90), ("p99", 99)):
                percentiles[label] = float(np.percentile(latencies_ms, q))
        metrics = self.status().get("metrics", {})
        return {
            "output_stream": output_stream,
            "duration_s": elapsed,
            "frames": frames,
            "throughput_hz": frames / elapsed,
            "latency_ms": {
                "min": float(latencies_ms.min()) if frames else None,
                "mean": float(latencies_ms.mean()) if frames else None,
                "max": float(latencies_ms.max()) if frames else None,
                **percentiles,
            },
            "workers": {
                name: {
                    "avg_exec_ms": worker.get("avg_exec_ms"),
                    "jitter_us_rms": worker.get("jitter_us_rms"),
                    "throughput_hz": worker.get("throughput_hz"),
                }
                for name, worker in metrics.items()
            },
        }

    @property
    def failures(self) -> tuple[dict[str, Any], ...]:
        """Return recorded worker failures."""
        self.poll_events()
        return tuple(self._failures)

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        """Return all events received from worker processes so far."""
        self.poll_events()
        return tuple(self._events)

    def __enter__(self) -> "PipelineManager":
        """Support context-managed pipeline lifecycles."""
        return self

    def __exit__(self, *_: object) -> None:
        """Ensure worker processes and streams are torn down on exit."""
        self.shutdown()
