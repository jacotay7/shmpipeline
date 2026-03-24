"""Pipeline manager responsible for building and supervising workers."""

from __future__ import annotations

from dataclasses import dataclass
import os
import multiprocessing as mp
from pathlib import Path
import time
from typing import Any

import pyshmem

from shmpipeline.config import PipelineConfig
from shmpipeline.errors import StateTransitionError, WorkerProcessError
from shmpipeline.logging_utils import get_logger
from shmpipeline.registry import get_default_registry
from shmpipeline.runtime import drain_events, run_kernel_process
from shmpipeline.state import PipelineState


@dataclass
class _WorkerHandle:
    """Runtime handle for one worker process."""

    name: str
    process: Any
    stop_event: Any


class PipelineManager:
    """Create shared memory, spawn workers, and supervise pipeline state."""

    def __init__(
        self,
        config: PipelineConfig | str | Path,
        *,
        spawn_method: str = "spawn",
    ) -> None:
        """Initialize a manager from a config object or YAML path."""
        if isinstance(config, (str, Path)):
            config = PipelineConfig.from_yaml(config)
        self.config = config
        self.registry = get_default_registry()
        self.context = mp.get_context(spawn_method)
        self._logger = get_logger("manager")
        self.state = PipelineState.INITIALIZED
        self._streams: dict[str, Any] = {}
        self._pause_event: Any | None = None
        self._event_queue: Any = self.context.Queue()
        self._workers: dict[str, _WorkerHandle] = {}
        self._failures: list[dict[str, Any]] = []
        self._events: list[dict[str, Any]] = []
        self._logger.info(
            "manager initialized: spawn_method=%s kernels=%d streams=%d",
            spawn_method,
            len(self.config.kernels),
            len(self.config.shared_memory),
        )

    def _transition_state(self, new_state: PipelineState, *, reason: str) -> None:
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
        if self.state not in {PipelineState.INITIALIZED, PipelineState.STOPPED}:
            raise StateTransitionError(
                f"cannot build pipeline while state is {self.state.value!r}"
            )
        self._logger.info("build started")
        self._event_queue = self.context.Queue()
        self._events.clear()
        self._failures.clear()
        shared_by_name = self.config.shared_memory_by_name
        self._logger.info("validating pipeline config")
        for kernel_config in self.config.kernels:
            self.registry.validate(kernel_config, shared_by_name)
            self._logger.info(
                "validated kernel: name=%s kind=%s inputs=%s outputs=%s",
                kernel_config.name,
                kernel_config.kind,
                list(kernel_config.inputs),
                list(kernel_config.outputs),
            )
        for spec in self.config.shared_memory:
            create_kwargs = {
                "shape": spec.shape,
                "dtype": spec.dtype,
            }
            if spec.storage == "gpu":
                create_kwargs["gpu_device"] = spec.gpu_device
                create_kwargs["cpu_mirror"] = spec.cpu_mirror
            self._streams[spec.name] = pyshmem.create(spec.name, **create_kwargs)
            self._logger.info(
                "created shared memory: name=%s storage=%s shape=%s dtype=%s",
                spec.name,
                spec.storage,
                spec.shape,
                spec.dtype,
            )
        self._transition_state(PipelineState.BUILT, reason="build complete")

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
        cpu_count = max(1, os.cpu_count() or 1)
        for kernel_config in self.config.kernels:
            stop_event = self.context.Event()
            cpu_slot = len(self._workers) % cpu_count
            process = self.context.Process(
                target=run_kernel_process,
                args=(
                    kernel_config,
                    self.config.shared_memory,
                    self._pause_event,
                    stop_event,
                    self._event_queue,
                    cpu_slot,
                ),
                name=f"shmpipeline-{kernel_config.name}",
            )
            process.start()
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
            )
        self._wait_for_workers_started()
        self._transition_state(PipelineState.RUNNING, reason="all workers started")

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
        self._transition_state(PipelineState.RUNNING, reason="resume requested")

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
        self._logger.info("stop requested: timeout=%s force=%s", timeout, force)
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
        self._workers.clear()
        self._pause_event = None
        self._transition_state(PipelineState.BUILT, reason="workers stopped")

    def shutdown(self, *, unlink: bool = True, force: bool = False) -> None:
        """Stop workers, close local handles, and optionally unlink streams."""
        self._logger.info("shutdown requested: unlink=%s force=%s", unlink, force)
        if self.state in {
            PipelineState.RUNNING,
            PipelineState.PAUSED,
            PipelineState.FAILED,
        }:
            self.stop(force=force)
        for name, stream in list(self._streams.items()):
            try:
                stream.close()
            finally:
                if unlink:
                    try:
                        pyshmem.unlink(name)
                    except FileNotFoundError:
                        pass
            self._logger.info("released shared memory: name=%s unlink=%s", name, unlink)
        self._streams.clear()
        self._transition_state(PipelineState.STOPPED, reason="shutdown complete")

    def poll_events(self) -> list[dict[str, Any]]:
        """Drain worker events and update manager failure state."""
        events = drain_events(self._event_queue)
        if events:
            self._events.extend(events)
            for event in events:
                self._log_event(event)
        for worker in self._workers.values():
            if worker.process.exitcode not in (None, 0):
                already_recorded = any(
                    failure.get("kernel") == worker.name for failure in self._failures
                )
                if not already_recorded:
                    self._failures.append(
                        {
                            "type": "worker_failed",
                            "kernel": worker.name,
                            "pid": worker.process.pid,
                            "error": (
                                f"worker exited with code {worker.process.exitcode}"
                            ),
                        }
                    )
        for event in events:
            if event.get("type") == "worker_failed":
                already_recorded = any(
                    failure.get("kernel") == event.get("kernel")
                    and failure.get("error") == event.get("error")
                    for failure in self._failures
                )
                if not already_recorded:
                    self._failures.append(event)
        if self._failures and self.state in {
            PipelineState.RUNNING,
            PipelineState.PAUSED,
        }:
            self._transition_state(
                PipelineState.FAILED,
                reason="worker failure recorded",
            )
        return events

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
        self._logger.debug("worker event: %s", event)

    def status(self) -> dict[str, Any]:
        """Return a snapshot of manager state, workers, and failures."""
        self.poll_events()
        return {
            "state": self.state.value,
            "workers": {
                name: {
                    "pid": worker.process.pid,
                    "alive": worker.process.is_alive(),
                    "exitcode": worker.process.exitcode,
                }
                for name, worker in self._workers.items()
            },
            "failures": list(self._failures),
        }

    def raise_if_failed(self) -> None:
        """Raise the first worker failure, if any has been recorded."""
        self.poll_events()
        if not self._failures:
            return
        failure = self._failures[0]
        message = (
            f"worker {failure['kernel']!r} failed: {failure['error']}"
        )
        raise WorkerProcessError(message)

    def get_stream(self, name: str):
        """Return the manager-owned shared-memory handle for one stream."""
        return self._streams[name]

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