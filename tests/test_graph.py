from __future__ import annotations

import pytest

from shmpipeline import PipelineConfig, PipelineGraph
from shmpipeline.graph import validate_pipeline_config

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
