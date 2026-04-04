"""Pipeline manager responsible for building and supervising workers."""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyshmem

from shmpipeline.config import PipelineConfig
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


class PipelineManager:
    """Create shared memory, spawn workers, and supervise pipeline state."""

    def __init__(
        self,
        config: PipelineConfig | str | Path,
        *,
        spawn_method: str = "spawn",
        placement_policy: WorkerPlacementPolicy | None = None,
        registry: KernelRegistry | None = None,
    ) -> None:
        """Initialize a manager from a config object or YAML path."""
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
        self._pause_event: Any | None = None
        self._workers: dict[str, _WorkerHandle] = {}
        self._failures: list[dict[str, Any]] = []
        self._events: deque[dict[str, Any]] = deque(maxlen=256)
        self._worker_metrics: dict[str, dict[str, Any]] = {}
        self._worker_runtime: dict[str, dict[str, Any]] = {}
        self._synthetic_sources: dict[str, Any] = {}
        self._kernel_configs = {
            kernel.name: kernel for kernel in self.config.kernels
        }
        self._logger.info(
            "manager initialized: spawn_method=%s placement_policy=%s "
            "kernels=%d streams=%d",
            spawn_method,
            self.placement_policy.describe(),
            len(self.config.kernels),
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
        """Validate configuration and create shared-memory resources."""
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
        for spec in self.config.shared_memory:
            self._streams[spec.name] = self._build_stream(spec)
        self._transition_state(PipelineState.BUILT, reason="build complete")

    def _build_stream(self, spec) -> Any:
        """Create, reuse, or replace one shared-memory stream."""
        existing_stream = self._open_existing_stream(spec)
        if existing_stream is not None:
            reused_stream = self._reuse_or_replace_existing_stream(
                spec,
                existing_stream,
            )
            if reused_stream is not None:
                return reused_stream

        create_kwargs = {
            "shape": spec.shape,
            "dtype": spec.dtype,
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
                reused_stream = self._reuse_or_replace_existing_stream(
                    spec,
                    existing_stream,
                )
                if reused_stream is not None:
                    return reused_stream
            else:
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
        return stream

    def _reuse_or_replace_existing_stream(self, spec, existing_stream) -> Any | None:
        """Return a reusable existing stream or replace it when incompatible."""
        if self._stream_matches_spec(existing_stream, spec):
            self._logger.info(
                "reusing shared memory: name=%s storage=%s shape=%s dtype=%s",
                spec.name,
                spec.storage,
                spec.shape,
                spec.dtype,
            )
            return existing_stream

        self._logger.info(
            "replacing incompatible shared memory: name=%s expected_storage=%s "
            "expected_shape=%s expected_dtype=%s existing_storage=%s "
            "existing_shape=%s existing_dtype=%s",
            spec.name,
            spec.storage,
            spec.shape,
            spec.dtype,
            "gpu" if existing_stream.gpu_enabled else "cpu",
            existing_stream.shape,
            existing_stream.dtype,
        )
        close_stream(existing_stream, unlink=True)
        return None

    def _open_existing_stream(self, spec) -> Any | None:
        """Open an existing stream if present.

        GPU streams are probed without a CUDA attachment so build can inspect
        and replace stale streams without reopening their IPC handles.
        """
        try:
            return pyshmem.open(spec.name)
        except FileNotFoundError:
            return None
        except ValueError:
            return None

    def _stream_matches_spec(self, stream: Any, spec) -> bool:
        """Return whether an existing stream matches the requested config."""
        if spec.storage == "gpu":
            return False
        if tuple(stream.shape) != tuple(spec.shape):
            return False
        if stream.dtype != spec.dtype:
            return False
        if stream.gpu_enabled != (spec.storage == "gpu"):
            return False
        return True

    def start(self) -> None:
        """Start worker processes for every configured kernel."""
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
        for index, kernel_config in enumerate(self.config.kernels):
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
        self._wait_for_workers_started()
        self._transition_state(
            PipelineState.RUNNING, reason="all workers started"
        )

    def _wait_for_workers_started(self, *, timeout: float = 5.0) -> None:
        """Block until all worker processes report a started event."""
        expected = set(self._workers)
        end_time = time.monotonic() + timeout
        while time.monotonic() < end_time:
            self.poll_events()
            started = {
                event["kernel"]
                for event in self._events
                if event.get("type") == "worker_started"
            }
            if started >= expected:
                return
            for worker in self._workers.values():
                if worker.process.exitcode not in (None, 0):
                    self.raise_if_failed()
            time.sleep(0.01)
        missing = sorted(
            expected
            - {
                event["kernel"]
                for event in self._events
                if event.get("type") == "worker_started"
            }
        )
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
        """Stop worker processes but keep shared-memory resources built."""
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
        self.poll_events()
        self._close_event_readers()
        self._workers.clear()
        self._pause_event = None
        self._transition_state(PipelineState.BUILT, reason="workers stopped")

    def shutdown(self, *, unlink: bool = True, force: bool = False) -> None:
        """Stop workers, close local handles, and optionally unlink streams."""
        self._logger.info(
            "shutdown requested: unlink=%s force=%s", unlink, force
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
                close_stream(stream, unlink=unlink)
            finally:
                self._logger.info(
                    "released shared memory: name=%s unlink=%s",
                    name,
                    unlink,
                )
        self._streams.clear()
        self._transition_state(
            PipelineState.STOPPED, reason="shutdown complete"
        )

    def poll_events(self) -> list[dict[str, Any]]:
        """Drain worker events and update manager failure state."""
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

    def status(self) -> dict[str, Any]:
        """Return a snapshot of manager state, workers, and failures."""
        self.poll_events()
        workers_status = {
            name: self._status_for_worker(name, worker)
            for name, worker in self._workers.items()
        }
        return {
            "state": self.state.value,
            "workers": workers_status,
            "failures": list(self._failures),
            "metrics": {
                name: dict(metrics)
                for name, metrics in self._worker_metrics.items()
            },
            "synthetic_sources": self.synthetic_input_status(),
            "placement_policy": self.placement_policy.describe(),
            "summary": self._status_summary(workers_status),
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
        self, workers_status: dict[str, dict[str, Any]]
    ) -> dict[str, int]:
        summary = {
            "workers_total": len(workers_status),
            "active_workers": 0,
            "idle_workers": 0,
            "waiting_workers": 0,
            "paused_workers": 0,
            "failed_workers": 0,
            "stopped_workers": 0,
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
        return summary

    def runtime_snapshot(self) -> dict[str, Any]:
        """Return a richer status snapshot for CLI and GUI consumers."""
        status = self.status()
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
        message = f"worker {failure['kernel']!r} failed: {failure['error']}"
        raise WorkerProcessError(message)

    def get_stream(self, name: str):
        """Return the manager-owned shared-memory handle for one stream."""
        return self._streams[name]

    def start_synthetic_input(
        self,
        spec: "SyntheticInputConfig | dict[str, Any]",
    ) -> dict[str, Any]:
        """Start a synthetic input writer for one built stream."""
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
