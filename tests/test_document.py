"""Tests for editable-document helpers used by the GUI and control plane."""

from __future__ import annotations

import pytest

from shmpipeline import PipelineConfig
from shmpipeline.document import (
    config_to_document,
    default_document,
    document_to_yaml,
    load_document,
    normalize_document,
    parse_inline_yaml,
    save_document,
)
from shmpipeline.gui.model import runtime_source_entries

pytestmark = pytest.mark.unit


def _config(**kernel_overrides):
    kernel = {
        "name": "scale",
        "kind": "cpu.scale",
        "input": "in",
        "output": "out",
        "parameters": {"factor": 2.0},
    }
    kernel.update(kernel_overrides)
    return PipelineConfig.from_dict(
        {
            "shared_memory": [
                {"name": "in", "shape": [4], "dtype": "float32"},
                {"name": "out", "shape": [4], "dtype": "float32"},
                {"name": "extra", "shape": [4], "dtype": "float32"},
            ],
            "kernels": [kernel],
        }
    )


def test_default_document_is_an_empty_skeleton():
    document = default_document()
    assert document["shared_memory"] == []
    assert document["kernels"] == []
    assert document["sources"] == []
    assert document["sinks"] == []


def test_config_to_document_round_trips_single_output():
    config = _config()
    document = config_to_document(config)
    assert document["kernels"][0]["output"] == "out"
    assert "outputs" not in document["kernels"][0]
    rebuilt = PipelineConfig.from_dict(document)
    assert rebuilt.kernels[0].kind == "cpu.scale"
    assert rebuilt.kernels[0].parameters["factor"] == 2.0


def test_config_to_document_emits_outputs_for_multi_output():
    config = PipelineConfig.from_dict(
        {
            "shared_memory": [
                {"name": "in", "shape": [4], "dtype": "float32"},
                {"name": "a", "shape": [4], "dtype": "float32"},
                {"name": "b", "shape": [4], "dtype": "float32"},
            ],
            "kernels": [
                {
                    "name": "split",
                    "kind": "cpu.copy",
                    "input": "in",
                    "outputs": ["a", "b"],
                }
            ],
        }
    )
    document = config_to_document(config)
    kernel = document["kernels"][0]
    assert kernel["outputs"] == ["a", "b"]
    assert "output" not in kernel


def test_document_to_yaml_and_load_round_trip(tmp_path):
    document = config_to_document(_config())
    text = document_to_yaml(document)
    assert "cpu.scale" in text
    path = tmp_path / "doc.yaml"
    save_document(path, document)
    loaded = load_document(path)
    assert loaded["kernels"][0]["name"] == "scale"


def test_normalize_document_returns_plain_mapping():
    document = config_to_document(_config())
    normalized = normalize_document(document)
    assert isinstance(normalized, dict)
    assert normalized["kernels"][0]["kind"] == "cpu.scale"


def test_parse_inline_yaml_uses_fallback_on_blank():
    assert parse_inline_yaml("", fallback={"a": 1}) == {"a": 1}
    assert parse_inline_yaml("{factor: 3.0}", fallback=None) == {"factor": 3.0}


def test_document_helpers_cover_sources_sinks_and_shared_memory_options():
    config = PipelineConfig.from_dict(
        {
            "shared_memory": [
                {
                    "name": "in",
                    "shape": [2],
                    "dtype": "float32",
                    "storage": "gpu",
                    "gpu_device": "cuda:0",
                    "cpu_mirror": True,
                    "notify": True,
                    "mode": "replace",
                },
                {"name": "out", "shape": [2], "dtype": "float32"},
                {"name": "aux", "shape": [2], "dtype": "float32"},
            ],
            "sources": [
                {
                    "name": "camera",
                    "kind": "test.source",
                    "stream": "in",
                    "auxiliary": {"cal": "aux"},
                }
            ],
            "kernels": [
                {
                    "name": "copy",
                    "kind": "cpu.copy",
                    "input": "in",
                    "output": "out",
                    "auxiliary": {"cal": "aux"},
                }
            ],
            "sinks": [
                {
                    "name": "display",
                    "kind": "test.sink",
                    "stream": "out",
                    "auxiliary": {"cal": "aux"},
                }
            ],
        }
    )
    document = config_to_document(config)
    assert document["shared_memory"][0]["notify"] is True
    assert document["sources"][0]["auxiliary"] == {"cal": "aux"}
    assert document["sinks"][0]["auxiliary"] == {"cal": "aux"}
    assert PipelineConfig.from_dict(document).sinks[0].name == "display"


def test_load_document_rejects_non_mapping(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text("- not a mapping\n", encoding="utf-8")
    with pytest.raises(Exception, match="mapping"):
        load_document(path)


def test_runtime_source_entries_projects_plugin_and_synthetic_status():
    entries = runtime_source_entries(
        {"sources": [{"name": "camera", "kind": "camera", "stream": "in"}]},
        {
            "sources": {
                "camera": {
                    "alive": True,
                    "frames_written": 4,
                    "effective_rate_hz": 12.5,
                }
            },
            "synthetic_sources": {"in": {"pattern": "random", "alive": False}},
        },
    )
    assert entries[0]["frames"] == 4
    assert entries[1]["name"] == "synthetic:in"
