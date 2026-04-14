"""Remote control helpers for pipeline-manager services."""

from shmpipeline.control.client import RemoteManagerClient, RemoteManagerError
from shmpipeline.control.service import ManagerService

__all__ = [
    "ManagerService",
    "RemoteManagerClient",
    "RemoteManagerError",
]
