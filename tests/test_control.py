from __future__ import annotations

import textwrap
from queue import Empty

import httpx
import pytest
from fastapi.testclient import TestClient

from shmpipeline.control import RemoteManagerClient, discover_local_servers
from shmpipeline.control.api import create_control_app
from shmpipeline.control.discovery import (
    LocalControlServerRecord,
    LocalControlServerRegistration,
    terminate_local_server,
)
from shmpipeline.control.service import ManagerService
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
        "shmpipeline.control.discovery.os.kill",
        lambda pid, sig: calls.append((pid, sig)),
    )
    record = LocalControlServerRecord(
        pid=4321,
        host="127.0.0.1",
        port=8765,
        token_required=False,
    )

    terminate_local_server(record)

    assert calls == [(record.pid, 15)]
