"""Local discovery helpers for shmpipeline control servers."""

from __future__ import annotations

import json
import os
import signal
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def discovery_directory() -> Path:
    """Return the filesystem location used for local server discovery."""
    path = Path(tempfile.gettempdir()) / "shmpipeline-control-servers"
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(frozen=True)
class LocalControlServerRecord:
    """One locally discoverable control-server process."""

    pid: int
    host: str
    port: int
    token_required: bool = False
    started_at: float = 0.0

    @property
    def connect_host(self) -> str:
        if self.host in {"0.0.0.0", "::", ""}:
            return "127.0.0.1"
        return self.host

    @property
    def base_url(self) -> str:
        return f"http://{self.connect_host}:{self.port}"

    @property
    def discovery_path(self) -> Path:
        return discovery_directory() / f"{self.pid}-{self.port}.json"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LocalControlServerRecord":
        return cls(
            pid=int(payload["pid"]),
            host=str(payload["host"]),
            port=int(payload["port"]),
            token_required=bool(payload.get("token_required", False)),
            started_at=float(payload.get("started_at", 0.0)),
        )


class LocalControlServerRegistration:
    """Lifecycle helper that registers one local server for discovery."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        token_required: bool,
    ) -> None:
        self._record = LocalControlServerRecord(
            pid=os.getpid(),
            host=host,
            port=port,
            token_required=token_required,
            started_at=time.time(),
        )
        self._path = self._record.discovery_path

    @property
    def record(self) -> LocalControlServerRecord:
        return self._record

    def register(self) -> None:
        self._path.write_text(
            json.dumps(self._record.to_dict(), sort_keys=True),
            encoding="utf-8",
        )

    def close(self) -> None:
        self._path.unlink(missing_ok=True)


def discover_local_servers() -> list[LocalControlServerRecord]:
    """Return all currently running locally registered control servers."""
    records: list[LocalControlServerRecord] = []
    for path in sorted(discovery_directory().glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            record = LocalControlServerRecord.from_dict(payload)
        except Exception:
            path.unlink(missing_ok=True)
            continue
        if not _pid_exists(record.pid):
            path.unlink(missing_ok=True)
            continue
        records.append(record)
    return sorted(records, key=lambda record: (record.port, record.pid))


def terminate_local_server(record: LocalControlServerRecord) -> None:
    """Terminate one locally registered control-server process."""
    os.kill(record.pid, signal.SIGTERM)


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True
