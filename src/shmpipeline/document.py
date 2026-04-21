"""Editable document helpers shared by GUI and control services."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

from shmpipeline.config import PipelineConfig
from shmpipeline.errors import ConfigValidationError

Document = dict[str, Any]


def default_document() -> Document:
    """Return an empty editable pipeline document."""
    return {
        "shared_memory": [],
        "sources": [],
        "kernels": [],
        "sinks": [],
    }


def clone_document(document: Mapping[str, Any]) -> Document:
    """Return a deep editable copy of one document."""
    return deepcopy(dict(document))


def normalize_document(document: Mapping[str, Any]) -> Document:
    """Ensure a document always has the expected top-level keys."""
    normalized = clone_document(document)
    normalized.setdefault("shared_memory", [])
    normalized.setdefault("sources", [])
    normalized.setdefault("kernels", [])
    normalized.setdefault("sinks", [])
    return normalized


def load_document(path: str | Path) -> Document:
    """Load a YAML config file into the editable document shape."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if raw is None:
        return default_document()
    if not isinstance(raw, Mapping):
        raise ConfigValidationError("pipeline config must be a mapping")
    return normalize_document(raw)


def document_to_yaml(document: Mapping[str, Any]) -> str:
    """Serialize one document to readable YAML."""
    return yaml.safe_dump(
        normalize_document(document),
        sort_keys=False,
        allow_unicode=False,
    )


def save_document(path: str | Path, document: Mapping[str, Any]) -> None:
    """Write one editable document to disk."""
    Path(path).write_text(document_to_yaml(document), encoding="utf-8")


def parse_inline_yaml(text: str, *, fallback: Any) -> Any:
    """Parse a small YAML fragment used by editors and dialogs."""
    stripped = text.strip()
    if not stripped:
        return deepcopy(fallback)
    value = yaml.safe_load(stripped)
    if value is None:
        return deepcopy(fallback)
    return value


def config_to_document(config: PipelineConfig) -> Document:
    """Convert an immutable PipelineConfig back into editable document form."""
    shared_memory = []
    for spec in config.shared_memory:
        item: dict[str, Any] = {
            "name": spec.name,
            "shape": list(spec.shape),
            "dtype": str(spec.dtype),
            "storage": spec.storage,
        }
        if spec.gpu_device is not None:
            item["gpu_device"] = spec.gpu_device
        if spec.cpu_mirror is not None:
            item["cpu_mirror"] = spec.cpu_mirror
        shared_memory.append(item)

    kernels = []
    for kernel in config.kernels:
        item = {
            "name": kernel.name,
            "kind": kernel.kind,
            "input": kernel.input,
            "output": kernel.output,
            "parameters": deepcopy(kernel.parameters),
            "read_timeout": kernel.read_timeout,
            "pause_sleep": kernel.pause_sleep,
        }
        if kernel.operation is not None:
            item["operation"] = kernel.operation
        if kernel.auxiliary:
            if all(
                binding.alias == binding.name for binding in kernel.auxiliary
            ):
                item["auxiliary"] = [
                    binding.name for binding in kernel.auxiliary
                ]
            else:
                item["auxiliary"] = {
                    binding.alias: binding.name for binding in kernel.auxiliary
                }
        kernels.append(item)

    sources = []
    for source in config.sources:
        sources.append(
            {
                "name": source.name,
                "kind": source.kind,
                "stream": source.stream,
                "parameters": deepcopy(source.parameters),
                "poll_interval": source.poll_interval,
            }
        )

    sinks = []
    for sink in config.sinks:
        sinks.append(
            {
                "name": sink.name,
                "kind": sink.kind,
                "stream": sink.stream,
                "parameters": deepcopy(sink.parameters),
                "read_timeout": sink.read_timeout,
                "pause_sleep": sink.pause_sleep,
            }
        )

    return {
        "shared_memory": shared_memory,
        "sources": sources,
        "kernels": kernels,
        "sinks": sinks,
    }
