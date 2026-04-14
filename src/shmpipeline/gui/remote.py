"""Remote GUI helpers for talking to the pipeline control plane."""

from __future__ import annotations

from dataclasses import dataclass
from ipaddress import ip_address
from typing import Any
from urllib.parse import urlparse

from shmpipeline.control import RemoteManagerClient
from shmpipeline.document import clone_document
from shmpipeline.state import PipelineState


def normalize_server_url(value: str) -> str:
    """Return a normalized HTTP base URL for one control server."""
    stripped = value.strip()
    if not stripped:
        raise ValueError("server URL must not be blank")
    if "://" not in stripped:
        stripped = f"http://{stripped}"
    normalized = stripped.rstrip("/")
    parsed = urlparse(normalized)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"invalid server URL: {value!r}")
    return normalized


def is_local_server_url(base_url: str) -> bool:
    """Return whether one control-server URL points at the local host."""
    parsed = urlparse(base_url)
    host = parsed.hostname
    if host in {"localhost", None}:
        return host == "localhost"
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class ServerConnection:
    """Connection details for one remote control server."""

    base_url: str
    token: str | None = None

    @classmethod
    def from_values(
        cls,
        base_url: str,
        token: str | None = None,
    ) -> "ServerConnection":
        cleaned_token = (token or "").strip() or None
        return cls(normalize_server_url(base_url), cleaned_token)

    @property
    def is_local(self) -> bool:
        return is_local_server_url(self.base_url)

    @property
    def display_name(self) -> str:
        return urlparse(self.base_url).netloc


class RemotePipelineSession:
    """Manager-like remote session used by the GUI windows."""

    def __init__(
        self,
        connection: ServerConnection,
        *,
        timeout: float = 5.0,
        client: Any | None = None,
    ) -> None:
        self.connection = connection
        self._client = RemoteManagerClient(
            connection.base_url,
            token=connection.token,
            timeout=timeout,
            client=client,
        )
        self._last_status: dict[str, Any] | None = None

    @property
    def is_local(self) -> bool:
        return self.connection.is_local

    @property
    def state(self) -> PipelineState:
        status = self.status()
        return PipelineState(status["state"])

    def close(self) -> None:
        self._client.close()

    def info(self) -> dict[str, Any]:
        return self._client.info()

    def document(self) -> dict[str, Any]:
        payload = self._client.document()
        payload["document"] = clone_document(payload["document"])
        return payload

    def update_document(self, document: dict[str, Any]) -> dict[str, Any]:
        return self._client.update_document(clone_document(document))

    def validate_document(
        self,
        document: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = None if document is None else clone_document(document)
        return self._client.validate_document(payload)

    def status(self) -> dict[str, Any]:
        self._last_status = self._client.status()
        return self._last_status

    def snapshot(self) -> dict[str, Any]:
        snapshot = self._client.snapshot()
        self._last_status = snapshot
        return snapshot

    def graph(self) -> dict[str, Any]:
        return self._client.graph()

    def build(self) -> dict[str, Any]:
        snapshot = self._client.build()
        self._last_status = snapshot
        return snapshot

    def start(self) -> dict[str, Any]:
        snapshot = self._client.start()
        self._last_status = snapshot
        return snapshot

    def pause(self) -> dict[str, Any]:
        snapshot = self._client.pause()
        self._last_status = snapshot
        return snapshot

    def resume(self) -> dict[str, Any]:
        snapshot = self._client.resume()
        self._last_status = snapshot
        return snapshot

    def stop(self, *, timeout: float = 5.0, force: bool = False) -> dict[str, Any]:
        snapshot = self._client.stop(timeout=timeout, force=force)
        self._last_status = snapshot
        return snapshot

    def shutdown(
        self,
        *,
        unlink: bool = True,
        force: bool = False,
    ) -> dict[str, Any]:
        snapshot = self._client.shutdown(unlink=unlink, force=force)
        self._last_status = snapshot
        return snapshot

    def synthetic_input_status(self) -> dict[str, Any]:
        return self.status().get("synthetic_sources", {})

    def start_synthetic_input(self, spec: Any) -> dict[str, Any]:
        payload = self._client.start_synthetic_input(spec)
        self._last_status = payload.get("snapshot")
        return payload

    def stop_synthetic_input(
        self,
        stream_name: str,
        *,
        timeout: float = 2.0,
    ) -> dict[str, Any]:
        payload = self._client.stop_synthetic_input(
            stream_name,
            timeout=timeout,
        )
        self._last_status = payload.get("snapshot")
        return payload
