from __future__ import annotations

from shmpipeline import PipelineConfig
from shmpipeline.gui.model import (
    available_sink_kinds,
    available_source_kinds,
    default_document,
    document_to_yaml,
    load_document,
    parse_inline_yaml,
    recommended_spawn_method,
    validate_document,
)
from shmpipeline.gui.remote import is_local_server_url, normalize_server_url


def test_default_document_is_empty_and_serializable(tmp_path):
    document = default_document()

    yaml_text = document_to_yaml(document)
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml_text, encoding="utf-8")

    loaded = load_document(path)

    assert loaded == document


def test_parse_inline_yaml_uses_fallback_for_blank_values():
    assert parse_inline_yaml("", fallback={}) == {}
    assert parse_inline_yaml("   ", fallback=[]) == []


def test_validate_document_reports_missing_lists_as_error():
    errors = validate_document(default_document())

    assert errors == [
        "pipeline config must define a non-empty shared_memory list"
    ]


def test_validate_document_accepts_valid_config():
    errors = validate_document(
        {
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
            "kernels": [
                {
                    "name": "copy_stage",
                    "kind": "cpu.copy",
                    "input": "input_frame",
                    "output": "output_frame",
                }
            ],
        }
    )

    assert errors == []


def test_available_source_and_sink_kinds_use_registry(monkeypatch):
    class _FakeRegistry:
        def source_kinds(self):
            return ("demo.source",)

        def sink_kinds(self):
            return ("demo.sink",)

    monkeypatch.setattr(
        "shmpipeline.gui.model.get_default_registry",
        lambda: _FakeRegistry(),
    )

    assert available_source_kinds() == ("demo.source",)
    assert available_sink_kinds() == ("demo.sink",)


def test_normalize_server_url_adds_scheme_and_strips_slash():
    assert normalize_server_url("127.0.0.1:8765/") == "http://127.0.0.1:8765"


def test_is_local_server_url_detects_loopback_hosts():
    assert is_local_server_url("http://127.0.0.1:8765") is True
    assert is_local_server_url("http://192.168.1.10:8765") is False


def test_recommended_spawn_method_prefers_forkserver_for_linux_cpu(
    monkeypatch,
):
    config = PipelineConfig.from_dict(
        {
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
            "kernels": [
                {
                    "name": "copy_stage",
                    "kind": "cpu.copy",
                    "input": "input_frame",
                    "output": "output_frame",
                }
            ],
        }
    )
    monkeypatch.setattr("shmpipeline.gui.model.sys.platform", "linux")
    monkeypatch.setattr(
        "shmpipeline.gui.model.mp.get_all_start_methods",
        lambda: ["fork", "spawn", "forkserver"],
    )

    assert recommended_spawn_method(config) == "forkserver"


def test_recommended_spawn_method_keeps_spawn_for_gpu(monkeypatch):
    config = PipelineConfig.from_dict(
        {
            "shared_memory": [
                {
                    "name": "input_frame",
                    "shape": [4],
                    "dtype": "float32",
                    "storage": "gpu",
                    "gpu_device": "cuda:0",
                },
                {
                    "name": "output_frame",
                    "shape": [4],
                    "dtype": "float32",
                    "storage": "gpu",
                    "gpu_device": "cuda:0",
                },
            ],
            "kernels": [
                {
                    "name": "copy_stage",
                    "kind": "gpu.copy",
                    "input": "input_frame",
                    "output": "output_frame",
                }
            ],
        }
    )
    monkeypatch.setattr("shmpipeline.gui.model.sys.platform", "linux")
    monkeypatch.setattr(
        "shmpipeline.gui.model.mp.get_all_start_methods",
        lambda: ["fork", "spawn", "forkserver"],
    )

    assert recommended_spawn_method(config) == "spawn"
