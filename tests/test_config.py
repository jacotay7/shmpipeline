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


def test_shared_memory_wait_and_lifecycle_options():
    config = PipelineConfig.from_dict(
        {
            "shared_memory": [
                {
                    "name": "input_frame",
                    "shape": [4],
                    "dtype": "float32",
                    "notify": True,
                    "mode": "attach",
                }
            ],
            "sources": [
                {
                    "name": "source",
                    "kind": "example.source",
                    "stream": "input_frame",
                }
            ],
        }
    )

    spec = config.shared_memory[0]
    assert spec.notify is True
    assert spec.mode == "attach"


@pytest.mark.parametrize("mode", ["bad", "reuse", "create-or-attach"])
def test_shared_memory_rejects_unknown_lifecycle_mode(mode):
    with pytest.raises(ConfigValidationError, match="mode"):
        PipelineConfig.from_dict(
            {
                "shared_memory": [
                    {
                        "name": "input_frame",
                        "shape": [4],
                        "dtype": "float32",
                        "mode": mode,
                    }
                ],
                "sources": [
                    {
                        "name": "source",
                        "kind": "example.source",
                        "stream": "input_frame",
                    }
                ],
            }
        )


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


def test_kernel_config_accepts_synchronized_inputs():
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
                    "name": "input_frame_b",
                    "shape": [4],
                    "dtype": "float32",
                    "storage": "cpu",
                },
                {
                    "name": "output_frame",
                    "shape": [8],
                    "dtype": "float32",
                    "storage": "cpu",
                },
            ],
            "kernels": [
                {
                    "name": "copy",
                    "kind": "cpu.concatenate",
                    "inputs": ["input_frame", "input_frame_b"],
                    "output": "output_frame",
                }
            ],
        }
    )

    kernel = config.kernels[0]
    assert kernel.trigger_inputs == ("input_frame", "input_frame_b")
    assert kernel.input == "input_frame"
    assert kernel.trigger_policy == "all_new"


def test_kernel_config_rejects_input_and_inputs_together():
    from shmpipeline.config import KernelConfig

    with pytest.raises(
        ConfigValidationError, match="either 'input' or 'inputs'"
    ):
        KernelConfig.from_dict(
            {
                "name": "bad",
                "kind": "cpu.concatenate",
                "input": "a",
                "inputs": ["a", "b"],
                "output": "out",
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


# ---------------------------------------------------------------------------
# KernelConfig.poll_interval
# ---------------------------------------------------------------------------


def _kernel_dict(**overrides):
    base = {
        "name": "k",
        "kind": "cpu.scale",
        "input": "i",
        "output": "o",
        "parameters": {"factor": 1.0},
    }
    base.update(overrides)
    # Allow callers to drop a default key by passing it as None (e.g. to
    # supply 'outputs' instead of 'output').
    return {key: value for key, value in base.items() if value is not None}


def test_kernel_config_poll_interval_default():
    from shmpipeline.config import KernelConfig

    config = KernelConfig.from_dict(_kernel_dict())
    assert config.poll_interval == pytest.approx(1e-5)


def test_kernel_config_poll_interval_custom():
    from shmpipeline.config import KernelConfig

    config = KernelConfig.from_dict(_kernel_dict(poll_interval=0.001))
    assert config.poll_interval == pytest.approx(0.001)


def test_kernel_config_poll_interval_must_be_positive():
    from shmpipeline.config import KernelConfig

    with pytest.raises(ConfigValidationError, match="poll_interval"):
        KernelConfig.from_dict(_kernel_dict(poll_interval=-0.001))


# ---------------------------------------------------------------------------
# Source/sink plugin timeout fields
# ---------------------------------------------------------------------------


def test_source_config_read_timeout_defaults_to_none():
    from shmpipeline.config import SourceConfig

    config = SourceConfig.from_dict(
        {"name": "s", "kind": "x.source", "stream": "i"}
    )
    assert config.read_timeout is None


def test_source_config_read_timeout_parsed_and_validated():
    from shmpipeline.config import SourceConfig

    config = SourceConfig.from_dict(
        {"name": "s", "kind": "x.source", "stream": "i", "read_timeout": 0.25}
    )
    assert config.read_timeout == pytest.approx(0.25)
    with pytest.raises(ConfigValidationError, match="read_timeout"):
        SourceConfig.from_dict(
            {
                "name": "s",
                "kind": "x.source",
                "stream": "i",
                "read_timeout": -1.0,
            }
        )


def test_sink_config_consume_timeout_defaults_to_none():
    from shmpipeline.config import SinkConfig

    config = SinkConfig.from_dict(
        {"name": "s", "kind": "x.sink", "stream": "o"}
    )
    assert config.consume_timeout is None


def test_sink_config_consume_timeout_parsed_and_validated():
    from shmpipeline.config import SinkConfig

    config = SinkConfig.from_dict(
        {"name": "s", "kind": "x.sink", "stream": "o", "consume_timeout": 0.5}
    )
    assert config.consume_timeout == pytest.approx(0.5)
    with pytest.raises(ConfigValidationError, match="consume_timeout"):
        SinkConfig.from_dict(
            {
                "name": "s",
                "kind": "x.sink",
                "stream": "o",
                "consume_timeout": 0.0,
            }
        )


# ---------------------------------------------------------------------------
# Multi-output kernels
# ---------------------------------------------------------------------------


def test_kernel_config_single_output_reports_all_outputs():
    from shmpipeline.config import KernelConfig

    config = KernelConfig.from_dict(_kernel_dict())
    assert config.output == "o"
    assert config.all_outputs == ("o",)


def test_kernel_config_accepts_outputs_list():
    from shmpipeline.config import KernelConfig

    config = KernelConfig.from_dict(
        _kernel_dict(output=None, outputs=["a", "b"])
    )
    assert config.output == "a"
    assert config.all_outputs == ("a", "b")


def test_kernel_config_rejects_both_output_and_outputs():
    from shmpipeline.config import KernelConfig

    with pytest.raises(ConfigValidationError, match="either 'output' or"):
        KernelConfig.from_dict(_kernel_dict(outputs=["a", "b"]))


def test_kernel_config_rejects_duplicate_outputs():
    from shmpipeline.config import KernelConfig

    with pytest.raises(ConfigValidationError, match="same output stream"):
        KernelConfig.from_dict(_kernel_dict(output=None, outputs=["a", "a"]))


def test_kernel_config_rejects_input_in_outputs():
    from shmpipeline.config import KernelConfig

    with pytest.raises(ConfigValidationError, match="both input and output"):
        KernelConfig.from_dict(
            _kernel_dict(input="a", output=None, outputs=["a", "b"])
        )


# ---------------------------------------------------------------------------
# YAML config error messages carry source file and line numbers
# ---------------------------------------------------------------------------


def test_from_yaml_error_includes_line_number_for_bad_reference(tmp_path):
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            shared_memory:
              - name: in
                shape: [4]
                dtype: float32
              - name: out
                shape: [4]
                dtype: float32
            kernels:
              - name: bad
                kind: cpu.scale
                input: in
                output: missing
                parameters: {factor: 2.0}
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigValidationError) as excinfo:
        PipelineConfig.from_yaml(config_path)
    message = str(excinfo.value)
    assert "pipeline.yaml" in message
    assert "line" in message


def test_from_yaml_error_includes_line_number_for_bad_dtype(tmp_path):
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            shared_memory:
              - name: in
                shape: [4]
                dtype: not_a_real_dtype
            kernels: []
            sources:
              - name: src
                kind: x.source
                stream: in
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigValidationError) as excinfo:
        PipelineConfig.from_yaml(config_path)
    message = str(excinfo.value)
    assert "pipeline.yaml" in message
    assert "line 3" in message
