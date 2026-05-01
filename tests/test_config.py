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
                input: input_frame
                output: output_frame
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


def test_pipeline_config_accepts_sources_and_sinks():
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
                    "parameters": {"device": "sim-0"},
                    "poll_interval": 0.05,
                }
            ],
            "kernels": [
                {
                    "name": "copy",
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
                    "parameters": {"window": "main"},
                    "read_timeout": 0.25,
                    "pause_sleep": 0.02,
                }
            ],
        }
    )

    assert config.sources[0].parameters == {"device": "sim-0"}
    assert config.sources[0].poll_interval == 0.05
    assert config.sinks[0].parameters == {"window": "main"}
    assert config.sinks[0].read_timeout == 0.25


def test_pipeline_config_allows_source_and_sink_only_pipeline():
    config = PipelineConfig.from_dict(
        {
            "shared_memory": [
                {
                    "name": "frame",
                    "shape": [4],
                    "dtype": "float32",
                    "storage": "cpu",
                }
            ],
            "sources": [
                {
                    "name": "camera",
                    "kind": "example.camera",
                    "stream": "frame",
                }
            ],
            "sinks": [
                {
                    "name": "display",
                    "kind": "example.display",
                    "stream": "frame",
                }
            ],
        }
    )

    assert config.kernels == ()
    assert config.sources[0].stream == "frame"
    assert config.sinks[0].stream == "frame"


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
                        "input": "input_frame",
                        "output": "missing_output",
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
                        "input": "frame",
                        "output": "frame",
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
                    "input": "input_vector",
                    "output": "output_vector",
                    "auxiliary": ["transform_matrix", "offset_vector"],
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


def test_kernel_config_rejects_legacy_inputs_outputs_keys():
    with pytest.raises(
        ConfigValidationError,
        match="must use 'input', 'output', and optional 'auxiliary'",
    ):
        PipelineConfig.from_dict(
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
                        "name": "copy",
                        "kind": "cpu.copy",
                        "inputs": ["input_frame"],
                        "outputs": ["output_frame"],
                    }
                ],
            }
        )


def test_kernel_config_rejects_unexpected_fields():
    with pytest.raises(
        ConfigValidationError,
        match="contains unsupported fields",
    ):
        PipelineConfig.from_dict(
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
                        "name": "copy",
                        "kind": "cpu.copy",
                        "input": "input_frame",
                        "output": "output_frame",
                        "unexpected": 123,
                    }
                ],
            }
        )


def test_kernel_config_accepts_named_auxiliary_bindings():
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
                    "name": "dark_frame",
                    "shape": [4],
                    "dtype": "float32",
                    "storage": "cpu",
                },
                {
                    "name": "flat_frame",
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
                    "name": "custom",
                    "kind": "cpu.custom_operation",
                    "operation": "(input - dark) / flat",
                    "input": "input_frame",
                    "output": "output_frame",
                    "auxiliary": {
                        "dark": "dark_frame",
                        "flat": "flat_frame",
                    },
                }
            ],
        }
    )

    assert config.kernels[0].auxiliary_aliases == ("dark", "flat")
    assert config.kernels[0].auxiliary_names == ("dark_frame", "flat_frame")
    assert config.kernels[0].auxiliary_by_alias == {
        "dark": "dark_frame",
        "flat": "flat_frame",
    }


def test_source_and_sink_configs_accept_named_auxiliary_bindings():
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
                    "name": "copy",
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

    assert config.sources[0].auxiliary_aliases == ("enabled",)
    assert config.sources[0].auxiliary_names == ("source_aux",)
    assert config.sources[0].auxiliary_by_alias == {"enabled": "source_aux"}
    assert config.sinks[0].auxiliary_aliases == ("mask",)
    assert config.sinks[0].auxiliary_names == ("sink_aux",)
    assert config.sinks[0].auxiliary_by_alias == {"mask": "sink_aux"}


def test_custom_operation_rejects_unsupported_function():
    from shmpipeline import PipelineManager

    with pytest.raises(
        ConfigValidationError,
        match="unsupported function",
    ):
        manager = PipelineManager(
            PipelineConfig.from_dict(
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
                            "name": "custom",
                            "kind": "cpu.custom_operation",
                            "operation": "sqrt(input)",
                            "input": "input_frame",
                            "output": "output_frame",
                        }
                    ],
                }
            )
        )
        manager.build()
