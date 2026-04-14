"""Python client for the shmpipeline HTTP control plane."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any, Iterator

import httpx

_LIFECYCLE_REQUEST_TIMEOUT = 60.0
_STATE_CHANGE_REQUEST_TIMEOUT = 15.0
_STOP_TIMEOUT_BUFFER = 5.0


class RemoteManagerError(RuntimeError):
    """Raised when the remote control plane rejects or fails a request."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        payload: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


class RemoteManagerClient:
    """Convenience client for one manager-control service."""

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        timeout: float = 5.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._default_timeout = timeout
        self._token = token
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=self._base_url,
            timeout=timeout,
        )

    def close(self) -> None:
        """Close the underlying HTTP client when owned by this instance."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "RemoteManagerClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def health(self) -> dict[str, Any]:
        return self._request_json("GET", "/health")

    def info(self) -> dict[str, Any]:
        return self._request_json("GET", "/info")

    def document(self) -> dict[str, Any]:
        return self._request_json("GET", "/document")

    def update_document(self, document: dict[str, Any]) -> dict[str, Any]:
        return self._request_json("PUT", "/document", payload=document)

    def validate_document(
        self,
        document: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._request_json(
            "POST",
            "/document/validate",
            payload=document,
        )

    def load_document_path(self, path: str) -> dict[str, Any]:
        return self._request_json(
            "POST",
            "/document/load",
            payload={"path": path},
            timeout=_STATE_CHANGE_REQUEST_TIMEOUT,
        )

    def status(self) -> dict[str, Any]:
        return self._request_json("GET", "/status")

    def snapshot(self) -> dict[str, Any]:
        return self._request_json("GET", "/snapshot")

    def graph(self) -> dict[str, Any]:
        return self._request_json("GET", "/graph")

    def build(self) -> dict[str, Any]:
        return self._request_json(
            "POST",
            "/commands/build",
            timeout=_LIFECYCLE_REQUEST_TIMEOUT,
        )

    def start(self) -> dict[str, Any]:
        return self._request_json(
            "POST",
            "/commands/start",
            timeout=_LIFECYCLE_REQUEST_TIMEOUT,
        )

    def pause(self) -> dict[str, Any]:
        return self._request_json(
            "POST",
            "/commands/pause",
            timeout=_STATE_CHANGE_REQUEST_TIMEOUT,
        )

    def resume(self) -> dict[str, Any]:
        return self._request_json(
            "POST",
            "/commands/resume",
            timeout=_STATE_CHANGE_REQUEST_TIMEOUT,
        )

    def stop(
        self,
        *,
        timeout: float = 5.0,
        force: bool = False,
    ) -> dict[str, Any]:
        return self._request_json(
            "POST",
            "/commands/stop",
            payload={
                "timeout": timeout,
                "force": force,
            },
            timeout=max(self._default_timeout, timeout + _STOP_TIMEOUT_BUFFER),
        )

    def shutdown(
        self,
        *,
        unlink: bool = True,
        force: bool = False,
    ) -> dict[str, Any]:
        return self._request_json(
            "POST",
            "/commands/shutdown",
            payload={
                "unlink": unlink,
                "force": force,
            },
            timeout=_STATE_CHANGE_REQUEST_TIMEOUT,
        )

    def start_synthetic_input(self, spec: Any) -> dict[str, Any]:
        return self._request_json(
            "POST",
            "/synthetic/start",
            payload=self._normalize_payload(spec),
            timeout=_STATE_CHANGE_REQUEST_TIMEOUT,
        )

    def stop_synthetic_input(
        self,
        stream_name: str,
        *,
        timeout: float = 2.0,
    ) -> dict[str, Any]:
        return self._request_json(
            "POST",
            "/synthetic/stop",
            payload={
                "stream_name": stream_name,
                "timeout": timeout,
            },
            timeout=max(self._default_timeout, timeout + _STOP_TIMEOUT_BUFFER),
        )

    def iter_events(
        self,
        *,
        last_event_id: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Iterate over SSE events from the control plane."""
        headers = self._headers()
        if last_event_id is not None:
            headers["Last-Event-ID"] = str(last_event_id)
        with self._client.stream(
            "GET", "/events", headers=headers
        ) as response:
            self._raise_for_status(response)
            event_id: str | None = None
            event_name = "message"
            data_lines: list[str] = []
            for line in response.iter_lines():
                if line == "":
                    if (
                        event_id is None
                        and not data_lines
                        and event_name == "message"
                    ):
                        continue
                    payload = None
                    if data_lines:
                        payload = json.loads("\n".join(data_lines))
                    yield {
                        "id": int(event_id) if event_id is not None else None,
                        "event": event_name,
                        "data": payload,
                    }
                    event_id = None
                    event_name = "message"
                    data_lines = []
                    continue
                if line.startswith(":"):
                    continue
                field, _, value = line.partition(":")
                if value.startswith(" "):
                    value = value[1:]
                if field == "id":
                    event_id = value
                elif field == "event":
                    event_name = value
                elif field == "data":
                    data_lines.append(value)

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        try:
            request_kwargs: dict[str, Any] = {
                "headers": self._headers(),
            }
            if payload is not None:
                request_kwargs["json"] = payload
            if timeout is not None and self._supports_request_timeout():
                request_kwargs["timeout"] = timeout
            response = self._client.request(method, path, **request_kwargs)
        except httpx.ConnectError as exc:  # pragma: no cover - network failure
            raise RemoteManagerError(
                f"could not reach control server at {self._base_url}"
            ) from exc
        except httpx.ReadTimeout as exc:  # pragma: no cover - network failure
            raise RemoteManagerError(
                f"request to {self._base_url}{path} timed out; the server may still be processing the command"
            ) from exc
        except httpx.HTTPError as exc:  # pragma: no cover - network failure
            raise RemoteManagerError(str(exc)) from exc
        self._raise_for_status(response)
        if not response.content:
            return {}
        return response.json()

    def _normalize_payload(self, payload: Any) -> Any:
        if is_dataclass(payload) and not isinstance(payload, type):
            return asdict(payload)
        return payload

    def _supports_request_timeout(self) -> bool:
        return not type(self._client).__module__.startswith(
            "starlette.testclient"
        )

    def _raise_for_status(self, response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            payload = None
            detail = exc.response.text
            try:
                payload = exc.response.json()
            except ValueError:
                payload = None
            if isinstance(payload, dict) and "detail" in payload:
                detail = str(payload["detail"])
            raise RemoteManagerError(
                detail,
                status_code=exc.response.status_code,
                payload=payload,
            ) from exc

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
        }
        if self._token is not None:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers
