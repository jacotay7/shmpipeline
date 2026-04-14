"""FastAPI application and server runner for remote manager control."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from ipaddress import ip_address
from queue import Empty
from typing import Any

import uvicorn
from fastapi import Body, Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from shmpipeline.control.service import ManagerService
from shmpipeline.errors import (
    ConfigValidationError,
    StateTransitionError,
    WorkerProcessError,
)


class StopRequest(BaseModel):
    """Request body for worker-stop operations."""

    timeout: float = 5.0
    force: bool = False


class ShutdownRequest(BaseModel):
    """Request body for manager-shutdown operations."""

    unlink: bool = True
    force: bool = False


class SyntheticStartRequest(BaseModel):
    """Request body for one synthetic-input writer."""

    stream_name: str
    pattern: str = "random"
    rate_hz: float | None = None
    seed: int = 0
    amplitude: float = 1.0
    offset: float = 0.0
    period: float = 64.0
    constant: float = 0.0
    impulse_interval: int = 64


class SyntheticStopRequest(BaseModel):
    """Request body for stopping one synthetic-input writer."""

    stream_name: str
    timeout: float = 2.0


class LoadDocumentPathRequest(BaseModel):
    """Request body for switching the server to a config file on disk."""

    path: str


def create_control_app(
    service: ManagerService,
    *,
    token: str | None = None,
) -> FastAPI:
    """Return a FastAPI app bound to one manager service."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service.start_event_pump()
        try:
            yield
        finally:
            service.close()

    app = FastAPI(
        title="shmpipeline control plane",
        version="1.0.0",
        lifespan=lifespan,
    )
    auth_dependencies = [Depends(_require_token(token))]

    @app.get("/health", dependencies=auth_dependencies)
    def health() -> dict[str, Any]:
        return service.health()

    @app.get("/info", dependencies=auth_dependencies)
    def info() -> dict[str, Any]:
        return service.info()

    @app.get("/document", dependencies=auth_dependencies)
    def document() -> dict[str, Any]:
        return service.document()

    @app.put("/document", dependencies=auth_dependencies)
    def update_document(payload: dict[str, Any]) -> dict[str, Any]:
        return _call_service(service.update_document, payload)

    @app.post("/document/validate", dependencies=auth_dependencies)
    def validate_document(
        payload: dict[str, Any] | None = Body(default=None),
    ) -> dict[str, Any]:
        return _call_service(service.validate_document, payload)

    @app.post("/document/load", dependencies=auth_dependencies)
    def load_document(payload: LoadDocumentPathRequest) -> dict[str, Any]:
        return _call_service(service.load_document_path, payload.path)

    @app.get("/status", dependencies=auth_dependencies)
    def status() -> dict[str, Any]:
        return service.status()

    @app.get("/snapshot", dependencies=auth_dependencies)
    def snapshot() -> dict[str, Any]:
        return service.snapshot()

    @app.get("/graph", dependencies=auth_dependencies)
    def graph() -> dict[str, Any]:
        return service.graph()

    @app.post("/commands/build", dependencies=auth_dependencies)
    def build() -> dict[str, Any]:
        return _call_service(service.build)

    @app.post("/commands/start", dependencies=auth_dependencies)
    def start() -> dict[str, Any]:
        return _call_service(service.start)

    @app.post("/commands/pause", dependencies=auth_dependencies)
    def pause() -> dict[str, Any]:
        return _call_service(service.pause)

    @app.post("/commands/resume", dependencies=auth_dependencies)
    def resume() -> dict[str, Any]:
        return _call_service(service.resume)

    @app.post("/commands/stop", dependencies=auth_dependencies)
    def stop(payload: StopRequest) -> dict[str, Any]:
        return _call_service(
            service.stop,
            timeout=payload.timeout,
            force=payload.force,
        )

    @app.post("/commands/shutdown", dependencies=auth_dependencies)
    def shutdown(payload: ShutdownRequest) -> dict[str, Any]:
        return _call_service(
            service.shutdown,
            unlink=payload.unlink,
            force=payload.force,
        )

    @app.post("/synthetic/start", dependencies=auth_dependencies)
    def start_synthetic(payload: SyntheticStartRequest) -> dict[str, Any]:
        return _call_service(
            service.start_synthetic_input,
            payload.model_dump(exclude_none=True),
        )

    @app.post("/synthetic/stop", dependencies=auth_dependencies)
    def stop_synthetic(payload: SyntheticStopRequest) -> dict[str, Any]:
        return _call_service(
            service.stop_synthetic_input,
            payload.stream_name,
            timeout=payload.timeout,
        )

    @app.get("/events", dependencies=auth_dependencies)
    def events(
        last_event_id: str | None = Header(
            default=None,
            alias="Last-Event-ID",
        ),
    ) -> StreamingResponse:
        parsed_last_event_id: int | None = None
        if last_event_id is not None:
            try:
                parsed_last_event_id = int(last_event_id)
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail="Last-Event-ID must be an integer",
                ) from exc

        subscriber_id, queue, backlog = service.subscribe(
            last_event_id=parsed_last_event_id
        )

        def generate():
            try:
                for event in backlog:
                    yield _encode_sse(event)
                yield _encode_sse(
                    {
                        "event": "snapshot",
                        "data": {
                            "reason": "initial",
                            "snapshot": service.snapshot(),
                        },
                    }
                )
                while True:
                    try:
                        event = queue.get(timeout=1.0)
                    except Empty:
                        yield ": keepalive\n\n"
                        continue
                    yield _encode_sse(event)
            finally:
                service.unsubscribe(subscriber_id)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app


def run_control_server(
    config: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    token: str | None = None,
    poll_interval: float = 0.1,
    log_level: str = "info",
) -> None:
    """Run the default HTTP control plane for one pipeline config."""
    from shmpipeline.control.discovery import LocalControlServerRegistration

    if not _is_local_host(host) and token is None:
        raise ValueError(
            "refusing to bind to a non-local interface without a bearer token"
        )
    service = ManagerService(config, poll_interval=poll_interval)
    app = create_control_app(service, token=token)
    registration = LocalControlServerRegistration(
        host=host,
        port=port,
        token_required=token is not None,
    )

    original_lifespan_context = app.router.lifespan_context

    @asynccontextmanager
    async def registered_lifespan(app: FastAPI):
        registration.register()
        try:
            async with original_lifespan_context(app):
                yield
        finally:
            registration.close()

    app.router.lifespan_context = registered_lifespan
    uvicorn.run(app, host=host, port=port, log_level=log_level)


def _call_service(method, *args: Any, **kwargs: Any) -> dict[str, Any]:
    try:
        return method(*args, **kwargs)
    except StateTransitionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ConfigValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except WorkerProcessError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _encode_sse(event: dict[str, Any]) -> str:
    lines: list[str] = []
    event_id = event.get("id")
    if event_id is not None:
        lines.append(f"id: {event_id}")
    event_name = event.get("event")
    if event_name is not None:
        lines.append(f"event: {event_name}")
    data = event.get("data")
    if data is not None:
        payload = json.dumps(data, sort_keys=True)
        for line in payload.splitlines():
            lines.append(f"data: {line}")
    return "\n".join(lines) + "\n\n"


def _is_local_host(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def _require_token(expected_token: str | None):
    def authorize(
        authorization: str | None = Header(default=None),
    ) -> None:
        if expected_token is None:
            return
        expected_value = f"Bearer {expected_token}"
        if authorization != expected_value:
            raise HTTPException(
                status_code=401,
                detail="missing or invalid bearer token",
            )

    return authorize
