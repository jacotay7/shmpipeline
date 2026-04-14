"""GUI document and validation helpers."""

from __future__ import annotations

import multiprocessing as mp
import sys
from typing import Any, Mapping

import numpy as np

from shmpipeline.config import PipelineConfig
from shmpipeline.document import (
    clone_document,
    default_document,
    document_to_yaml,
    load_document,
    normalize_document,
    parse_inline_yaml,
    save_document,
)
from shmpipeline.errors import ConfigValidationError
from shmpipeline.graph import validate_pipeline_config
from shmpipeline.manager import PipelineManager
from shmpipeline.registry import get_default_registry

try:
    import torch
except Exception:  # pragma: no cover - exercised when torch is unavailable
    torch = None


Document = dict[str, Any]

__all__ = [
    "Document",
    "available_kernel_kinds",
    "clone_document",
    "create_manager",
    "default_document",
    "document_to_yaml",
    "load_document",
    "parse_inline_yaml",
    "recommended_spawn_method",
    "save_document",
    "to_numpy",
    "validate_document",
]


def validate_document(document: Mapping[str, Any]) -> list[str]:
    """Return config and kernel-validation errors for one document."""
    try:
        config = PipelineConfig.from_dict(normalize_document(document))
    except ConfigValidationError as exc:
        return [str(exc)]

    return validate_pipeline_config(config)


def recommended_spawn_method(config: PipelineConfig) -> str:
    """Return the preferred worker start method for GUI-launched pipelines."""
    if not sys.platform.startswith("linux"):
        return "spawn"
    if any(spec.storage == "gpu" for spec in config.shared_memory):
        return "spawn"
    available_methods = set(mp.get_all_start_methods())
    if "forkserver" in available_methods:
        return "forkserver"
    if "fork" in available_methods:
        return "fork"
    return "spawn"


def create_manager(document: Mapping[str, Any]) -> PipelineManager:
    """Instantiate a pipeline manager from the current GUI document."""
    config = PipelineConfig.from_dict(normalize_document(document))
    return PipelineManager(
        config,
        spawn_method=recommended_spawn_method(config),
    )


def available_kernel_kinds() -> tuple[str, ...]:
    """Return all registered kernel kinds for UI selection widgets."""
    return get_default_registry().kinds()


def to_numpy(value: Any) -> np.ndarray:
    """Convert one shared-memory payload to a NumPy array for display."""
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)
