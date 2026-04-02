"""GUI document and validation helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from shmpipeline.config import PipelineConfig
from shmpipeline.errors import ConfigValidationError
from shmpipeline.graph import validate_pipeline_config
from shmpipeline.manager import PipelineManager
from shmpipeline.registry import get_default_registry

try:
    import torch
except Exception:  # pragma: no cover - exercised when torch is unavailable
    torch = None


Document = dict[str, Any]


def default_document() -> Document:
    """Return an empty editable document."""
    return {
        "shared_memory": [],
        "kernels": [],
    }


def clone_document(document: Mapping[str, Any]) -> Document:
    """Return a deep editable copy of one document."""
    return deepcopy(dict(document))


def normalize_document(document: Mapping[str, Any]) -> Document:
    """Ensure the GUI document always has the expected top-level keys."""
    normalized = clone_document(document)
    normalized.setdefault("shared_memory", [])
    normalized.setdefault("kernels", [])
    return normalized


def load_document(path: str | Path) -> Document:
    """Load a YAML configuration file into the editable document shape."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if raw is None:
        return default_document()
    if not isinstance(raw, Mapping):
        raise ConfigValidationError("pipeline config must be a mapping")
    return normalize_document(raw)


def document_to_yaml(document: Mapping[str, Any]) -> str:
    """Serialize one document to human-readable YAML."""
    return yaml.safe_dump(
        normalize_document(document),
        sort_keys=False,
        allow_unicode=False,
    )


def save_document(path: str | Path, document: Mapping[str, Any]) -> None:
    """Write one editable document to disk."""
    Path(path).write_text(document_to_yaml(document), encoding="utf-8")


def parse_inline_yaml(text: str, *, fallback: Any) -> Any:
    """Parse a small YAML fragment used by dialog editors."""
    stripped = text.strip()
    if not stripped:
        return deepcopy(fallback)
    value = yaml.safe_load(stripped)
    if value is None:
        return deepcopy(fallback)
    return value


def validate_document(document: Mapping[str, Any]) -> list[str]:
    """Return config and kernel-validation errors for one document."""
    try:
        config = PipelineConfig.from_dict(normalize_document(document))
    except ConfigValidationError as exc:
        return [str(exc)]

    return validate_pipeline_config(config)


def create_manager(document: Mapping[str, Any]) -> PipelineManager:
    """Instantiate a pipeline manager from the current GUI document."""
    config = PipelineConfig.from_dict(normalize_document(document))
    return PipelineManager(config)


def available_kernel_kinds() -> tuple[str, ...]:
    """Return all registered kernel kinds for UI selection widgets."""
    return get_default_registry().kinds()


def to_numpy(value: Any) -> np.ndarray:
    """Convert one shared-memory payload to a NumPy array for display."""
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)
