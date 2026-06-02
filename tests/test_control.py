from __future__ import annotations

import signal
import textwrap
from queue import Empty

import httpx
import pytest
from fastapi.testclient import TestClient

import shmpipeline.control.discovery as discovery_module
from shmpipeline.control import RemoteManagerClient, discover_local_servers
from shmpipeline.control.api import create_control_app
from shmpipeline.control.discovery import (
    LocalControlServerRecord,
    LocalControlServerRegistration,
    terminate_local_server,
)
from shmpipeline.control.service import ManagerService
from shmpipeline.registry import get_default_registry
from shmpipeline.sink import Sink
from shmpipeline.source import Source
from shmpipeline.synthetic import SyntheticInputConfig

pytestmark = [pytest.mark.unit, pytest.mark.integration]


def _write_valid_config(path) -> None:
    path.write_text(
        textwrap.dedent(
            """
            shared_memory:
              - name: input_frame
                shape: [4]
                dtype: float32
                storage: cpu
              - name: output_frame
                shape: [4]
                dtype: float32
                storage: cpu
            kernels:
              - name: scale_stage
                kind: cpu.scale
                input: input_frame
                output: output_frame
                parameters:
                  factor: 2.0
                read_timeout: 0.1
            """
        ),
        encoding="utf-8",
    )


def test_control_api_rejects_missing_token(tmp_path):
    config_path = tmp_path / "pipeline.yaml"
    _write_valid_config(config_path)
    service = ManagerService(config_path, poll_interval=0.01)

    with TestClient(
        create_control_app(service, token="secret-token")
    ) as client:
        response = client.get("/status")

    assert response.status_code == 401
    assert response.json()["detail"] == "missing or invalid bearer token"


def test_remote_manager_client_controls_pipeline(tmp_path):
    config_path = tmp_path / "pipeline.yaml"
    _write_valid_config(config_path)
    service = ManagerService(config_path, poll_interval=0.01)

    with TestClient(
        create_control_app(service, token="secret-token")
    ) as test_client:
        client = RemoteManagerClient(
            "http://testserver",
            token="secret-token",
            client=test_client,
        )

        assert client.health()["state"] == "initialized"
        assert client.build()["state"] == "built"
        assert client.start()["state"] == "running"
        assert client.pause()["state"] == "paused"
        assert client.resume()["state"] == "running"

        synthetic = client.start_synthetic_input(
            SyntheticInputConfig(
                stream_name="input_frame",
                pattern="constant",
                constant=1.0,
                rate_hz=20.0,
            )
        )
        assert "input_frame" in synthetic["snapshot"]["synthetic_sources"]

        stopped_synthetic = client.stop_synthetic_input("input_frame")
        assert stopped_synthetic["snapshot"]["synthetic_sources"] == {}

        stopped = client.stop(force=True)
        assert stopped["state"] == "built"

        shutdown = client.shutdown(force=True)
        assert shutdown["state"] == "stopped"


def test_remote_manager_client_can_sync_documents(tmp_path):
    config_path = tmp_path / "pipeline.yaml"
    _write_valid_config(config_path)
    service = ManagerService(config_path, poll_interval=0.01)

    with TestClient(create_control_app(service)) as test_client:
        client = RemoteManagerClient("http://testserver", client=test_client)

        document_payload = client.document()
        assert document_payload["revision"] == 1
        assert (
            document_payload["document"]["kernels"][0]["name"] == "scale_stage"
        )

        validation = client.validate_document(
            {"shared_memory": [], "kernels": []}
        )
        assert validation["valid"] is False
        assert validation["errors"]

        updated_document = document_payload["document"]
        updated_document["kernels"][0]["parameters"]["factor"] = 3.0

        updated = client.update_document(updated_document)
        assert updated["revision"] == 2
        assert updated["document"]["kernels"][0]["parameters"]["factor"] == 3.0

        started = client.start()
        assert started["state"] == "running"

        response = test_client.put(
            "/document",
            json=updated_document,
        )
        assert response.status_code == 409
        assert "stop or shutdown first" in response.json()["detail"]


def test_remote_manager_client_can_load_document_path(tmp_path):
    config_path = tmp_path / "pipeline.yaml"
    replacement_path = tmp_path / "replacement.yaml"
    _write_valid_config(config_path)
    replacement_path.write_text(
        textwrap.dedent(
            """
            shared_memory:
              - name: source_frame
                shape: [4]
                dtype: float32
                storage: cpu
              - name: sink_frame
                shape: [4]
                dtype: float32
                storage: cpu
            kernels:
              - name: copy_stage
                kind: cpu.copy
                input: source_frame
                output: sink_frame
            """
        ),
        encoding="utf-8",
    )
    service = ManagerService(config_path, poll_interval=0.01)

    with TestClient(create_control_app(service)) as test_client:
        client = RemoteManagerClient("http://testserver", client=test_client)

        payload = client.load_document_path(str(replacement_path))

        assert payload["config_path"] == str(replacement_path.resolve())
        assert payload["document"]["kernels"][0]["name"] == "copy_stage"


def test_manager_service_uses_custom_registry_for_info_and_validation():
    class _ProbeSource(Source):
        kind = "test.control_source"
        storage = "cpu"

        def read(self):
            return None

    class _ProbeSink(Sink):
        kind = "test.control_sink"
        storage = "cpu"

        def consume(self, value):
            del value

    document = {
        "shared_memory": [
            {
                "name": "input_frame",
                "shape": [4],
                "dtype": "float32",
                "storage": "cpu",
            },
            {
                "name": "output_frame",
                "shape": [4],
                "dtype": "float32",
                "storage": "cpu",
            },
        ],
        "sources": [
            {
                "name": "camera",
                "kind": "test.control_source",
                "stream": "input_frame",
            }
        ],
        "kernels": [
            {
                "name": "copy_stage",
                "kind": "cpu.copy",
                "input": "input_frame",
                "output": "output_frame",
            }
        ],
        "sinks": [
            {
                "name": "display",
                "kind": "test.control_sink",
                "stream": "output_frame",
            }
        ],
    }
    registry = get_default_registry().extended(
        sources=(_ProbeSource,),
        sinks=(_ProbeSink,),
    )
    service = ManagerService(document, registry=registry, poll_interval=0.01)

    try:
        info = service.info()
        validation = service.validate_document(document)
    finally:
        service.close()

    assert "test.control_source" in info["source_kinds"]
    assert "test.control_sink" in info["sink_kinds"]
    assert validation["valid"] is True


def test_manager_service_publishes_snapshot_events(tmp_path):
    config_path = tmp_path / "pipeline.yaml"
    _write_valid_config(config_path)
    service = ManagerService(config_path, poll_interval=0.01)

    try:
        subscriber_id, queue, backlog = service.subscribe()
        assert backlog == []

        snapshot = service.build()
        assert snapshot["state"] == "built"

        event = queue.get(timeout=1.0)
        assert event["event"] == "snapshot"
        assert event["data"]["reason"] == "build"
        assert event["data"]["snapshot"]["state"] == "built"

        service.unsubscribe(subscriber_id)
        with pytest.raises(Empty):
            queue.get_nowait()
    finally:
        service.close()


def test_remote_manager_client_uses_extended_command_timeouts():
    captured: list[dict[str, object]] = []

    class _RecordingClient:
        def request(self, method, path, **kwargs):
            captured.append(
                {
                    "method": method,
                    "path": path,
                    "timeout": kwargs.get("timeout"),
                }
            )
            return httpx.Response(
                200,
                json={"state": "ok"},
                request=httpx.Request(method, f"http://testserver{path}"),
            )

        def close(self):
            return None

    client = RemoteManagerClient(
        "http://testserver", client=_RecordingClient()
    )

    client.build()
    client.start()
    client.stop(timeout=7.0)

    assert captured == [
        {"method": "POST", "path": "/commands/build", "timeout": 60.0},
        {"method": "POST", "path": "/commands/start", "timeout": 60.0},
        {"method": "POST", "path": "/commands/stop", "timeout": 12.0},
    ]


def test_local_control_server_discovery_round_trip(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "shmpipeline.control.discovery.discovery_directory",
        lambda: tmp_path,
    )
    registration = LocalControlServerRegistration(
        host="127.0.0.1",
        port=8765,
        token_required=False,
    )
    registration.register()
    try:
        records = discover_local_servers()
    finally:
        registration.close()

    assert len(records) == 1
    assert records[0].base_url == "http://127.0.0.1:8765"


def test_terminate_local_server_uses_sigterm(monkeypatch):
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "shmpipeline.control.discovery._kill_pid",
        lambda pid, sig: calls.append((pid, sig)),
    )
    record = LocalControlServerRecord(
        pid=4321,
        host="127.0.0.1",
        port=8765,
        token_required=False,
    )

    terminate_local_server(record)

    assert calls == [(record.pid, signal.SIGTERM)]


def test_pid_exists_uses_windows_query_helper(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(discovery_module, "_IS_WINDOWS", True)
    monkeypatch.setattr(
        discovery_module,
        "_pid_exists_windows",
        lambda pid: calls.append(pid) or True,
    )
    monkeypatch.setattr(
        discovery_module.os,
        "kill",
        lambda *_args: pytest.fail("os.kill should not be used on Windows"),
    )

    assert discovery_module._pid_exists(4321) is True
    assert calls == [4321]


# ---------------------------------------------------------------------------
# Role-based authorization scopes (read / control / admin)
# ---------------------------------------------------------------------------


def _scoped_client(tmp_path):
    config_path = tmp_path / "pipeline.yaml"
    _write_valid_config(config_path)
    service = ManagerService(config_path, poll_interval=0.01)
    app = create_control_app(
        service, tokens={"read": "R", "control": "C", "admin": "A"}
    )
    return service, app


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def test_scopes_read_token_allows_status_but_denies_build(tmp_path):
    service, app = _scoped_client(tmp_path)
    with TestClient(app) as client:
        assert client.get("/status", headers=_auth("R")).status_code == 200
        denied = client.post("/commands/build", headers=_auth("R"))
        assert denied.status_code == 403
    service.shutdown(force=True)


def test_scopes_control_token_denies_admin_operations(tmp_path):
    service, app = _scoped_client(tmp_path)
    with TestClient(app) as client:
        assert client.get("/status", headers=_auth("C")).status_code == 200
        assert (
            client.post("/commands/build", headers=_auth("C")).status_code
            == 403
        )
    service.shutdown(force=True)


def test_scopes_admin_token_has_full_access(tmp_path):
    service, app = _scoped_client(tmp_path)
    with TestClient(app) as client:
        assert client.get("/status", headers=_auth("A")).status_code == 200
        assert (
            client.post("/commands/build", headers=_auth("A")).status_code
            == 200
        )
    service.shutdown(force=True)


def test_scopes_missing_or_unknown_token_is_unauthorized(tmp_path):
    service, app = _scoped_client(tmp_path)
    with TestClient(app) as client:
        assert client.get("/status").status_code == 401
        assert client.get("/status", headers=_auth("nope")).status_code == 401
    service.shutdown(force=True)


def test_unknown_scope_name_is_rejected():
    from shmpipeline.control.api import _ScopeAuthorizer

    with pytest.raises(ValueError, match="unknown control-plane scope"):
        _ScopeAuthorizer(tokens={"superuser": "x"})


# ---------------------------------------------------------------------------
# SSE client auto-reconnect with exponential backoff
# ---------------------------------------------------------------------------


def test_stream_events_reconnects_and_resumes_from_last_id():
    class _FakeClient(RemoteManagerClient):
        def __init__(self):
            self._token = None
            self.requested_ids: list[int | None] = []
            self.attempt = 0

        def iter_events(self, *, last_event_id=None):
            self.attempt += 1
            self.requested_ids.append(last_event_id)
            if self.attempt == 1:
                yield {"id": 1, "event": "message", "data": {"x": 1}}
                raise RemoteManagerError("connection dropped")
            yield {"id": 2, "event": "message", "data": {"x": 2}}
            yield {"id": 3, "event": "message", "data": {"x": 3}}

    from shmpipeline.control.client import RemoteManagerError

    client = _FakeClient()
    sleeps: list[float] = []
    seen: list[int] = []
    for event in client.stream_events(
        sleeper=sleeps.append,
        should_continue=lambda: client.attempt < 2,
    ):
        seen.append(event["id"])

    assert seen == [1, 2, 3]
    # Reconnect resumes from the last id observed before the drop.
    assert client.requested_ids == [None, 1]
    assert sleeps and sleeps[0] == pytest.approx(0.5)


def test_stream_events_no_reconnect_raises_on_drop():
    from shmpipeline.control.client import RemoteManagerError

    class _FailingClient(RemoteManagerClient):
        def __init__(self):
            self._token = None

        def iter_events(self, *, last_event_id=None):
            raise RemoteManagerError("dropped")
            yield  # pragma: no cover - makes this a generator

    client = _FailingClient()
    with pytest.raises(RemoteManagerError):
        list(client.stream_events(reconnect=False))


# ---------------------------------------------------------------------------
# SSE /events endpoint round-trip and helpers
# ---------------------------------------------------------------------------


def test_iter_events_parses_sse_fields():
    """iter_events decodes id/event/data SSE frames into event dicts.

    Driven with a fake transport instead of the live infinite /events stream
    so the test terminates deterministically.
    """

    class _FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def iter_lines(self):
            yield "id: 1"
            yield "event: snapshot"
            yield 'data: {"state": "running"}'
            yield ""
            yield ": keepalive"
            yield "event: worker_metrics"
            yield 'data: {"kernel": "k"}'
            yield ""

    class _FakeStreamCtx:
        def __enter__(self):
            return _FakeResponse()

        def __exit__(self, *exc):
            return False

    class _FakeHttpx:
        def stream(self, method, path, headers=None):
            return _FakeStreamCtx()

    client = RemoteManagerClient("http://testserver", client=_FakeHttpx())
    events = list(client.iter_events())
    assert events[0] == {
        "id": 1,
        "event": "snapshot",
        "data": {"state": "running"},
    }
    assert events[1]["event"] == "worker_metrics"
    assert events[1]["data"] == {"kernel": "k"}


def test_encode_sse_formats_id_event_and_data():
    from shmpipeline.control.api import _encode_sse

    encoded = _encode_sse(
        {"id": 5, "event": "snapshot", "data": {"state": "running"}}
    )
    assert "id: 5" in encoded
    assert "event: snapshot" in encoded
    assert "data: " in encoded
    assert encoded.endswith("\n\n")


def test_is_local_host_recognizes_loopback():
    from shmpipeline.control.api import _is_local_host

    assert _is_local_host("localhost")
    assert _is_local_host("127.0.0.1")
    assert not _is_local_host("203.0.113.5")
