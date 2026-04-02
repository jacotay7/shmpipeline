from __future__ import annotations

import json
import textwrap

import pytest

from shmpipeline.cli import main

pytestmark = [pytest.mark.unit, pytest.mark.integration]


def _write_valid_config(path) -> None:
    path.write_text(
        textwrap.dedent(
            """
            shared_memory:
              - name: input_frame
                shape: [4]
                dtype: float32
                storage: cpu
              - name: output_frame
                shape: [4]
                dtype: float32
                storage: cpu
            kernels:
              - name: scale_stage
                kind: cpu.scale
                input: input_frame
                output: output_frame
                parameters:
                  factor: 2.0
                read_timeout: 0.1
            """
        ),
        encoding="utf-8",
    )


def _write_invalid_config(path) -> None:
    path.write_text(
        textwrap.dedent(
            """
            shared_memory:
              - name: input_a
                shape: [4]
                dtype: float32
                storage: cpu
              - name: input_b
                shape: [4]
                dtype: float32
                storage: cpu
              - name: output_frame
                shape: [4]
                dtype: float32
                storage: cpu
            kernels:
              - name: copy_a
                kind: cpu.copy
                input: input_a
                output: output_frame
              - name: copy_b
                kind: cpu.copy
                input: input_b
                output: output_frame
            """
        ),
        encoding="utf-8",
    )


def test_cli_validate_passes_for_valid_config(tmp_path, capsys):
    config_path = tmp_path / "pipeline.yaml"
    _write_valid_config(config_path)

    exit_code = main(["validate", str(config_path)])

    assert exit_code == 0
    assert "Validation passed" in capsys.readouterr().out


def test_cli_validate_fails_for_invalid_config(tmp_path, capsys):
    config_path = tmp_path / "pipeline.yaml"
    _write_invalid_config(config_path)

    exit_code = main(["validate", str(config_path)])

    assert exit_code == 1
    assert "Validation failed" in capsys.readouterr().out


def test_cli_describe_json_emits_graph(tmp_path, capsys):
    config_path = tmp_path / "pipeline.yaml"
    _write_valid_config(config_path)

    exit_code = main(["describe", str(config_path), "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["source_streams"] == ["input_frame"]
    assert payload["sink_streams"] == ["output_frame"]


def test_cli_run_starts_and_stops_pipeline(tmp_path):
    config_path = tmp_path / "pipeline.yaml"
    _write_valid_config(config_path)

    exit_code = main(
        [
            "run",
            str(config_path),
            "--duration",
            "0.25",
            "--poll-interval",
            "0.05",
        ]
    )

    assert exit_code == 0
