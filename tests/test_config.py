from __future__ import annotations

import textwrap

import numpy as np
import pytest

from shmpipeline import PipelineConfig
from shmpipeline.errors import ConfigValidationError


pytestmark = pytest.mark.unit


def test_pipeline_config_loads_from_yaml(tmp_path):
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            shared_memory:
              - name: input_frame
                shape: [2, 2]
                dtype: float32
                storage: cpu
              - name: output_frame
                shape: [2, 2]
                dtype: float32
                storage: cpu
            kernels:
              - name: scale
                kind: cpu.scale
                inputs: [input_frame]
                outputs: [output_frame]
                parameters:
                  factor: 3.0
            """
        ),
        encoding="utf-8",
    )

    config = PipelineConfig.from_yaml(config_path)

    assert config.shared_memory[0].shape == (2, 2)
    assert config.shared_memory[0].dtype == np.dtype(np.float32)
    assert config.kernels[0].parameters["factor"] == 3.0


def test_pipeline_config_rejects_unknown_shared_memory_reference():
    with pytest.raises(ConfigValidationError, match="undefined shared memory"):
        PipelineConfig.from_dict(
            {
                "shared_memory": [
                    {
                        "name": "input_frame",
                        "shape": [4],
                        "dtype": "float32",
                        "storage": "cpu",
                    }
                ],
                "kernels": [
                    {
                        "name": "copy",
                        "kind": "cpu.copy",
                        "inputs": ["input_frame"],
                        "outputs": ["missing_output"],
                    }
                ],
            }
        )


def test_pipeline_config_rejects_same_input_and_output_name():
    with pytest.raises(
        ConfigValidationError,
        match="same shared memory for both input and output",
    ):
        PipelineConfig.from_dict(
            {
                "shared_memory": [
                    {
                        "name": "frame",
                        "shape": [4],
                        "dtype": "float32",
                        "storage": "cpu",
                    }
                ],
                "kernels": [
                    {
                        "name": "copy",
                        "kind": "cpu.copy",
                        "inputs": ["frame"],
                        "outputs": ["frame"],
                    }
                ],
            }
        )


def test_affine_kernel_rejects_incompatible_shapes():
    config = PipelineConfig.from_dict(
        {
            "shared_memory": [
                {
                    "name": "input_vector",
                    "shape": [3],
                    "dtype": "float32",
                    "storage": "cpu",
                },
                {
                    "name": "transform_matrix",
                    "shape": [2, 4],
                    "dtype": "float32",
                    "storage": "cpu",
                },
                {
                    "name": "offset_vector",
                    "shape": [2],
                    "dtype": "float32",
                    "storage": "cpu",
                },
                {
                    "name": "output_vector",
                    "shape": [2],
                    "dtype": "float32",
                    "storage": "cpu",
                },
            ],
            "kernels": [
                {
                    "name": "affine",
                    "kind": "cpu.affine_transform",
                    "inputs": [
                        "input_vector",
                        "transform_matrix",
                        "offset_vector",
                    ],
                    "outputs": ["output_vector"],
                }
            ],
        }
    )

    with pytest.raises(
        ConfigValidationError,
        match="matrix columns to match input vector length",
    ):
        from shmpipeline import PipelineManager

        manager = PipelineManager(config)
        manager.build()