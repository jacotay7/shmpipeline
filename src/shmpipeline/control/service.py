"""Thread-safe control service that owns a single pipeline manager."""

from __future__ import annotations

import threading
from collections import deque
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any, Mapping

from shmpipeline.config import PipelineConfig
from shmpipeline.document import (
    clone_document,
    config_to_document,
    load_document,
    normalize_document,
)
from shmpipeline.errors import ConfigValidationError, StateTransitionError
from shmpipeline.graph import PipelineGraph, validate_pipeline_config
from shmpipeline.manager import PipelineManager
from shmpipeline.registry import KernelRegistry
from shmpipeline.scheduling import WorkerPlacementPolicy
from shmpipeline.state import PipelineState
from shmpipeline.synthetic import available_synthetic_patterns


class ManagerService:
    """Own one manager instance and expose serialized control operations.

    The service is the safe in-process boundary for HTTP handlers, GUI
    adapters, or any other remote-control transport. Exactly one manager owns
    the worker lifecycle, and every control operation is serialized through a
    re-entrant lock.
    """

    def __init__(
        self,
        config: PipelineConfig | Mapping[str, Any] | str | Path,
        *,
        spawn_method: str = "spawn",
        placement_policy: WorkerPlacementPolicy | None = None,
        registry: KernelRegistry | None = None,
        poll_interval: float = 0.1,
    ) -> None:
        if poll_interval <= 0.0:
            raise ValueError("poll_interval must be positive")

        self._spawn_method = spawn_method
        self._placement_policy = placement_policy
        self._registry = registry
        self._config_path = None
        self._document = self._load_initial_document(config)
        self._document_revision = 1
        self._manager = self._create_manager_locked()
        self._poll_interval = float(poll_interval)
        self._lock = threading.RLock()
        self._event_lock = threading.Lock()
        self._closed = threading.Event()
        self._poll_thread: threading.Thread | None = None
        self._subscribers: dict[int, Queue[dict[str, Any]]] = {}
        self._event_history: deque[dict[str, Any]] = deque(maxlen=512)
        self._next_subscriber_id = 1
        self._next_event_id = 1

    def start_event_pump(self) -> None:
        """Start the background worker-event polling loop."""
        if self._poll_thread is not None and self._poll_thread.is_alive():
            return
        self._closed.clear()
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            name="shmpipeline-control-poll",
            daemon=True,
        )
        self._poll_thread.start()

    def close(self) -> None:
        """Stop the background poller and release manager resources."""
        self._closed.set()
        poll_thread = self._poll_thread
        if poll_thread is not None and poll_thread.is_alive():
            poll_thread.join(timeout=2.0)
        with self._lock:
            self._manager.shutdown(force=True)

    def health(self) -> dict[str, Any]:
        """Return a lightweight liveness payload."""
        with self._lock:
            return {
                "ok": True,
                "state": self._manager.state.value,
                "document_revision": self._document_revision,
            }

    def info(self) -> dict[str, Any]:
        """Return static service metadata and supported control features."""
        with self._lock:
            return {
                "config_path": self._config_path,
                "state": self._manager.state.value,
                "commands": [
                    "build",
                    "start",
                    "pause",
                    "resume",
                    "stop",
                    "shutdown",
                ],
                "synthetic_patterns": list(available_synthetic_patterns()),
                "document_revision": self._document_revision,
            }

    def document(self) -> dict[str, Any]:
        """Return the current editable document and revision metadata."""
        with self._lock:
            return {
                "config_path": self._config_path,
                "revision": self._document_revision,
                "document": clone_document(self._document),
            }

    def validate_document(
        self,
        document: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return validation results for one candidate document."""
        with self._lock:
            candidate = (
                normalize_document(document)
                if document is not None
                else clone_document(self._document)
            )
            return self._validate_document_locked(candidate)

    def update_document(
        self,
        document: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Replace the current editable document and recreate the manager."""
        with self._lock:
            candidate = normalize_document(document)
            validation = self._validate_document_locked(candidate)
            if validation["errors"]:
                raise ConfigValidationError(validation["errors"][0])
            payload = self._apply_document_locked(
                candidate,
                config_path=self._config_path,
                validation=validation,
            )
        self._publish_event(
            "document_updated",
            {
                "reason": "document_updated",
                **payload,
            },
        )
        return payload

    def load_document_path(self, path: str | Path) -> dict[str, Any]:
        """Load one config document from disk and replace the current server config."""
        resolved_path = str(Path(path).expanduser().resolve())
        candidate = load_document(resolved_path)
        with self._lock:
            validation = self._validate_document_locked(candidate)
            if validation["errors"]:
                raise ConfigValidationError(validation["errors"][0])
            payload = self._apply_document_locked(
                candidate,
                config_path=resolved_path,
                validation=validation,
            )
        self._publish_event(
            "document_updated",
            {
                "reason": "document_loaded",
                **payload,
            },
        )
        return payload

    def status(self) -> dict[str, Any]:
        """Return the current manager status payload."""
        with self._lock:
            status = self._manager.status(poll=not self._is_polling_active())
            status["document_revision"] = self._document_revision
            status["config_path"] = self._config_path
            return status

    def snapshot(self) -> dict[str, Any]:
        """Return the current runtime snapshot payload."""
        with self._lock:
            return self._snapshot_locked()

    def graph(self) -> dict[str, Any]:
        """Return the derived pipeline graph payload."""
        with self._lock:
            return PipelineGraph.from_config(
                PipelineConfig.from_dict(self._document)
            ).to_dict()

    def build(self) -> dict[str, Any]:
        """Build the manager-owned pipeline resources."""
        with self._lock:
            if self._manager.state == PipelineState.INITIALIZED:
                self._manager.build()
            elif self._manager.state == PipelineState.STOPPED:
                self._manager = self._create_manager_locked()
                self._manager.build()
            elif self._manager.state == PipelineState.BUILT:
                pass
            else:
                raise StateTransitionError(
                    "build is not allowed while the pipeline is active"
                )
            snapshot = self._snapshot_locked()
        self._publish_event(
            "snapshot",
            {
                "reason": "build",
                "snapshot": snapshot,
            },
        )
        return snapshot

    def start(self) -> dict[str, Any]:
        """Start all configured workers."""
        with self._lock:
            if self._manager.state == PipelineState.INITIALIZED:
                self._manager.build()
                self._manager.start()
            elif self._manager.state == PipelineState.STOPPED:
                self._manager = self._create_manager_locked()
                self._manager.build()
                self._manager.start()
            elif self._manager.state == PipelineState.BUILT:
                self._manager.start()
            else:
                raise StateTransitionError(
                    f"cannot start pipeline while state is {self._manager.state.value!r}"
                )
            snapshot = self._snapshot_locked()
        self._publish_event(
            "snapshot",
            {
                "reason": "start",
                "snapshot": snapshot,
            },
        )
        return snapshot

    def pause(self) -> dict[str, Any]:
        """Pause all workers."""
        return self._run_command("pause", self._manager.pause)

    def resume(self) -> dict[str, Any]:
        """Resume all paused workers."""
        return self._run_command("resume", self._manager.resume)

    def stop(
        self, *, timeout: float = 5.0, force: bool = False
    ) -> dict[str, Any]:
        """Stop all workers while keeping built streams attached."""
        return self._run_command(
            "stop",
            self._manager.stop,
            timeout=timeout,
            force=force,
        )

    def shutdown(
        self,
        *,
        unlink: bool = True,
        force: bool = False,
    ) -> dict[str, Any]:
        """Shutdown the manager and optionally unlink shared-memory streams."""
        return self._run_command(
            "shutdown",
            self._manager.shutdown,
            unlink=unlink,
            force=force,
        )

    def start_synthetic_input(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Start one synthetic input writer and return its controller snapshot."""
        with self._lock:
            controller_snapshot = self._manager.start_synthetic_input(spec)
            snapshot = self._snapshot_locked()
        self._publish_event(
            "snapshot",
            {
                "reason": "synthetic_start",
                "snapshot": snapshot,
            },
        )
        return {
            "synthetic_source": controller_snapshot,
            "snapshot": snapshot,
        }

    def stop_synthetic_input(
        self,
        stream_name: str,
        *,
        timeout: float = 2.0,
    ) -> dict[str, Any]:
        """Stop one synthetic input writer and return the updated snapshot."""
        with self._lock:
            self._manager.stop_synthetic_input(stream_name, timeout=timeout)
            snapshot = self._snapshot_locked()
        self._publish_event(
            "snapshot",
            {
                "reason": "synthetic_stop",
                "snapshot": snapshot,
            },
        )
        return {
            "stream_name": stream_name,
            "snapshot": snapshot,
        }

    def subscribe(
        self,
        *,
        last_event_id: int | None = None,
    ) -> tuple[int, Queue[dict[str, Any]], list[dict[str, Any]]]:
        """Register one event subscriber and return any replay backlog."""
        queue: Queue[dict[str, Any]] = Queue(maxsize=128)
        with self._event_lock:
            subscriber_id = self._next_subscriber_id
            self._next_subscriber_id += 1
            self._subscribers[subscriber_id] = queue
            backlog = [
                event
                for event in self._event_history
                if last_event_id is not None and event["id"] > last_event_id
            ]
        return subscriber_id, queue, backlog

    def unsubscribe(self, subscriber_id: int) -> None:
        """Remove one live subscriber queue if it exists."""
        with self._event_lock:
            self._subscribers.pop(subscriber_id, None)

    def _run_command(
        self,
        command_name: str,
        method,
        **kwargs: Any,
    ) -> dict[str, Any]:
        with self._lock:
            method(**kwargs)
            snapshot = self._snapshot_locked()
        self._publish_event(
            "snapshot",
            {
                "reason": command_name,
                "snapshot": snapshot,
            },
        )
        return snapshot

    def _snapshot_locked(self) -> dict[str, Any]:
        snapshot = self._manager.runtime_snapshot(
            poll=not self._is_polling_active()
        )
        snapshot["document_revision"] = self._document_revision
        snapshot["config_path"] = self._config_path
        return snapshot

    def _load_initial_document(
        self,
        config: PipelineConfig | Mapping[str, Any] | str | Path,
    ) -> dict[str, Any]:
        if isinstance(config, PipelineConfig):
            return config_to_document(config)
        if isinstance(config, (str, Path)):
            self._config_path = str(config)
            return load_document(config)
        return normalize_document(config)

    def _create_manager_locked(self) -> PipelineManager:
        return PipelineManager(
            PipelineConfig.from_dict(self._document),
            spawn_method=self._spawn_method,
            placement_policy=self._placement_policy,
            registry=self._registry,
        )

    def _apply_document_locked(
        self,
        document: Mapping[str, Any],
        *,
        config_path: str | None,
        validation: dict[str, Any],
    ) -> dict[str, Any]:
        state = self._manager.state
        if state in {
            PipelineState.RUNNING,
            PipelineState.PAUSED,
            PipelineState.FAILED,
        }:
            raise StateTransitionError(
                "cannot replace the server document while the pipeline is active; stop or shutdown first"
            )
        if state == PipelineState.BUILT:
            self._manager.shutdown(force=True)
        self._document = clone_document(document)
        self._config_path = config_path
        self._document_revision += 1
        self._manager = self._create_manager_locked()
        return {
            "config_path": self._config_path,
            "revision": self._document_revision,
            "document": clone_document(self._document),
            **validation,
        }

    def _validate_document_locked(
        self,
        document: Mapping[str, Any],
    ) -> dict[str, Any]:
        try:
            config = PipelineConfig.from_dict(document)
        except ConfigValidationError as exc:
            return {
                "valid": False,
                "errors": [str(exc)],
            }

        errors = validate_pipeline_config(config)
        payload: dict[str, Any] = {
            "valid": not errors,
            "errors": errors,
        }
        if not errors:
            payload["graph"] = PipelineGraph.from_config(config).to_dict()
        return payload

    def _is_polling_active(self) -> bool:
        poll_thread = self._poll_thread
        return poll_thread is not None and poll_thread.is_alive()

    def _poll_loop(self) -> None:
        while not self._closed.is_set():
            try:
                with self._lock:
                    events = self._manager.poll_events()
                for event in events:
                    self._publish_event(event.get("type", "event"), event)
            except Exception as exc:  # pragma: no cover - defensive path
                self._publish_event(
                    "service_error",
                    {
                        "error": str(exc),
                    },
                )
            self._closed.wait(self._poll_interval)

    def _publish_event(self, event_name: str, data: dict[str, Any]) -> None:
        with self._event_lock:
            event_record = {
                "id": self._next_event_id,
                "event": event_name,
                "data": data,
            }
            self._next_event_id += 1
            self._event_history.append(event_record)

            stale_subscribers: list[int] = []
            for subscriber_id, queue in self._subscribers.items():
                try:
                    queue.put_nowait(event_record)
                except Full:
                    try:
                        queue.get_nowait()
                    except Empty:
                        pass
                    try:
                        queue.put_nowait(event_record)
                    except Full:
                        stale_subscribers.append(subscriber_id)
            for subscriber_id in stale_subscribers:
                self._subscribers.pop(subscriber_id, None)
