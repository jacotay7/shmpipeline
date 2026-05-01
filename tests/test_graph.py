from __future__ import annotations

import pytest

from shmpipeline import PipelineConfig, PipelineGraph, get_default_registry
from shmpipeline.graph import validate_pipeline_config
from shmpipeline.source import Source

pytestmark = pytest.mark.unit


def test_pipeline_graph_reports_sources_sinks_orphans_and_dependencies():
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
                    "name": "mid_frame",
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
                {
                    "name": "unused_frame",
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
                    "output": "mid_frame",
                },
                {
                    "name": "scale_stage",
                    "kind": "cpu.scale",
                    "input": "mid_frame",
                    "output": "output_frame",
                    "parameters": {"factor": 2.0},
                },
            ],
        }
    )

    graph = PipelineGraph.from_config(config)

    assert graph.source_streams() == ("input_frame",)
    assert graph.sink_streams() == ("output_frame",)
    assert graph.orphaned_streams() == ("unused_frame",)
    assert graph.upstream_kernels("scale_stage") == ("copy_stage",)
    assert graph.downstream_kernels("copy_stage") == ("scale_stage",)
    assert len(graph.edges) == 4
    assert "Pipeline Graph" in graph.describe()
    assert graph.to_dict()["source_streams"] == ("input_frame",)


def test_validate_pipeline_config_rejects_multiple_producers():
    config = PipelineConfig.from_dict(
        {
            "shared_memory": [
                {
                    "name": "input_a",
                    "shape": [4],
                    "dtype": "float32",
                    "storage": "cpu",
                },
                {
                    "name": "input_b",
                    "shape": [4],
                    "dtype": "float32",
                    "storage": "cpu",
                },
                {
                    "name": "shared_output",
                    "shape": [4],
                    "dtype": "float32",
                    "storage": "cpu",
                },
            ],
            "kernels": [
                {
                    "name": "copy_a",
                    "kind": "cpu.copy",
                    "input": "input_a",
                    "output": "shared_output",
                },
                {
                    "name": "copy_b",
                    "kind": "cpu.copy",
                    "input": "input_b",
                    "output": "shared_output",
                },
            ],
        }
    )

    errors = validate_pipeline_config(config)

    assert len(errors) == 1
    assert "multiple producer kernels" in errors[0]


def test_pipeline_graph_reports_explicit_sources_and_sinks():
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
            "sources": [
                {
                    "name": "camera",
                    "kind": "example.camera",
                    "stream": "input_frame",
                }
            ],
            "kernels": [
                {
                    "name": "copy_stage",
                    "kind": "cpu.copy",
                    "input": "input_frame",
                    "output": "output_frame",
                }
            ],
            "sinks": [
                {
                    "name": "display",
                    "kind": "example.display",
                    "stream": "output_frame",
                }
            ],
        }
    )

    graph = PipelineGraph.from_config(config)

    assert graph.source_streams() == ("input_frame",)
    assert graph.sink_streams() == ("output_frame",)
    assert any(edge.role == "source" for edge in graph.edges)
    assert any(edge.role == "sink" for edge in graph.edges)
    assert graph.to_dict()["sources"][0]["name"] == "camera"
    assert graph.to_dict()["sinks"][0]["name"] == "display"
    assert "Sources:" in graph.describe()
    assert "Sinks:" in graph.describe()


def test_pipeline_graph_reports_source_and_sink_auxiliary_edges():
    config = PipelineConfig.from_dict(
        {
            "shared_memory": [
                {
                    "name": "source_aux",
                    "shape": [1],
                    "dtype": "float32",
                    "storage": "cpu",
                },
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
                {
                    "name": "sink_aux",
                    "shape": [1],
                    "dtype": "float32",
                    "storage": "cpu",
                },
            ],
            "sources": [
                {
                    "name": "camera",
                    "kind": "example.camera",
                    "stream": "input_frame",
                    "auxiliary": {"enabled": "source_aux"},
                }
            ],
            "kernels": [
                {
                    "name": "copy_stage",
                    "kind": "cpu.copy",
                    "input": "input_frame",
                    "output": "output_frame",
                }
            ],
            "sinks": [
                {
                    "name": "display",
                    "kind": "example.display",
                    "stream": "output_frame",
                    "auxiliary": {"mask": "sink_aux"},
                }
            ],
        }
    )

    graph = PipelineGraph.from_config(config)
    edges = {(edge.source, edge.target, edge.role) for edge in graph.edges}
    assert ("source_aux", "camera", "auxiliary:enabled") in edges
    assert ("sink_aux", "display", "auxiliary:mask") in edges

    shared_records = {
        item["name"]: item for item in graph.to_dict()["shared_memory"]
    }
    assert shared_records["source_aux"]["consumers"] == ("camera",)
    assert shared_records["sink_aux"]["consumers"] == ("display",)


def test_validate_pipeline_config_rejects_source_and_kernel_conflict():
    class _ConflictSource(Source):
        kind = "test.conflict_source"
        storage = "cpu"

        def read(self):
            return None

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
                    "name": "shared_output",
                    "shape": [4],
                    "dtype": "float32",
                    "storage": "cpu",
                },
            ],
            "sources": [
                {
                    "name": "camera",
                    "kind": "test.conflict_source",
                    "stream": "shared_output",
                }
            ],
            "kernels": [
                {
                    "name": "copy_stage",
                    "kind": "cpu.copy",
                    "input": "input_frame",
                    "output": "shared_output",
                }
            ],
        }
    )

    errors = validate_pipeline_config(
        config,
        registry=get_default_registry().extended_sources(_ConflictSource),
    )

    assert any("multiple producers" in error for error in errors)
