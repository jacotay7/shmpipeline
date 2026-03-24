from __future__ import annotations

import logging
import time

import numpy as np
import pytest

from shmpipeline import PipelineConfig, PipelineManager, PipelineState
from shmpipeline.errors import WorkerProcessError


pytestmark = [pytest.mark.unit, pytest.mark.integration]


def _make_pipeline_config(shm_prefix: str, *, kind: str, parameters: dict):
    return PipelineConfig.from_dict(
        {
            "shared_memory": [
                {
                    "name": f"{shm_prefix}_input",
                    "shape": [4],
                    "dtype": "float32",
                    "storage": "cpu",
                },
                {
                    "name": f"{shm_prefix}_output",
                    "shape": [4],
                    "dtype": "float32",
                    "storage": "cpu",
                },
            ],
            "kernels": [
                {
                    "name": "stage",
                    "kind": kind,
                    "inputs": [f"{shm_prefix}_input"],
                    "outputs": [f"{shm_prefix}_output"],
                    "parameters": parameters,
                    "read_timeout": 0.1,
                }
            ],
        }
    )


def _make_affine_pipeline_config(shm_prefix: str):
    return PipelineConfig.from_dict(
        {
            "shared_memory": [
                {
                    "name": f"{shm_prefix}_input",
                    "shape": [3],
                    "dtype": "float32",
                    "storage": "cpu",
                },
                {
                    "name": f"{shm_prefix}_matrix",
                    "shape": [2, 3],
                    "dtype": "float32",
                    "storage": "cpu",
                },
                {
                    "name": f"{shm_prefix}_offset",
                    "shape": [2],
                    "dtype": "float32",
                    "storage": "cpu",
                },
                {
                    "name": f"{shm_prefix}_output",
                    "shape": [2],
                    "dtype": "float32",
                    "storage": "cpu",
                },
            ],
            "kernels": [
                {
                    "name": "affine_stage",
                    "kind": "cpu.affine_transform",
                    "inputs": [
                        f"{shm_prefix}_input",
                        f"{shm_prefix}_matrix",
                        f"{shm_prefix}_offset",
                    ],
                    "outputs": [f"{shm_prefix}_output"],
                    "parameters": {},
                    "read_timeout": 0.1,
                }
            ],
        }
    )


def test_manager_state_machine(shm_prefix):
    config = _make_pipeline_config(
        shm_prefix,
        kind="cpu.copy",
        parameters={},
    )
    manager = PipelineManager(config)

    manager.build()
    assert manager.state == PipelineState.BUILT

    manager.start()
    assert manager.state == PipelineState.RUNNING
    assert manager.status()["workers"]["stage"]["alive"] is True

    manager.pause()
    assert manager.state == PipelineState.PAUSED

    manager.resume()
    assert manager.state == PipelineState.RUNNING

    manager.stop()
    assert manager.state == PipelineState.BUILT

    manager.shutdown()
    assert manager.state == PipelineState.STOPPED


def test_manager_runs_cpu_scale_kernel_end_to_end(shm_prefix):
    config = _make_pipeline_config(
        shm_prefix,
        kind="cpu.scale",
        parameters={"factor": 2.5},
    )
    manager = PipelineManager(config)
    manager.build()
    manager.start()

    input_stream = manager.get_stream(f"{shm_prefix}_input")
    output_stream = manager.get_stream(f"{shm_prefix}_output")
    payload = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    input_stream.write(payload)

    received = output_stream.read_new(timeout=2.0)
    np.testing.assert_allclose(received, payload * 2.5)

    manager.stop()
    manager.shutdown()


def test_manager_surfaces_worker_failures(shm_prefix):
    config = _make_pipeline_config(
        shm_prefix,
        kind="cpu.raise_error",
        parameters={"message": "intentional failure"},
    )
    manager = PipelineManager(config)
    manager.build()
    manager.start()

    manager.get_stream(f"{shm_prefix}_input").write(
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    )

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and manager.state != PipelineState.FAILED:
        manager.poll_events()
        time.sleep(0.05)

    assert manager.state == PipelineState.FAILED
    with pytest.raises(WorkerProcessError, match="intentional failure"):
        manager.raise_if_failed()

    manager.shutdown(force=True)


def test_manager_runs_affine_transform_kernel_end_to_end(shm_prefix):
    config = _make_affine_pipeline_config(shm_prefix)
    manager = PipelineManager(config)
    manager.build()
    manager.start()

    manager.get_stream(f"{shm_prefix}_matrix").write(
        np.array(
            [
                [1.0, 2.0, 0.0],
                [0.0, -1.0, 3.0],
            ],
            dtype=np.float32,
        )
    )
    manager.get_stream(f"{shm_prefix}_offset").write(
        np.array([0.5, -2.0], dtype=np.float32)
    )
    manager.get_stream(f"{shm_prefix}_input").write(
        np.array([4.0, 1.0, 2.0], dtype=np.float32)
    )

    received = manager.get_stream(f"{shm_prefix}_output").read_new(timeout=2.0)
    expected = np.array([6.5, 3.0], dtype=np.float32)
    np.testing.assert_allclose(received, expected)

    manager.shutdown()


def test_manager_logs_state_transitions_and_validation(shm_prefix, caplog):
    caplog.set_level(logging.INFO)
    config = _make_pipeline_config(
        shm_prefix,
        kind="cpu.copy",
        parameters={},
    )
    manager = PipelineManager(config)

    manager.build()
    manager.start()
    manager.stop()
    manager.shutdown()

    messages = [record.getMessage() for record in caplog.records]
    assert any("validating pipeline config" in message for message in messages)
    assert any("state transition: initialized -> built" in message for message in messages)
    assert any("state transition: built -> running" in message for message in messages)
    assert any("worker started: kernel=stage" in message for message in messages)


def test_manager_runs_many_affine_vectors(shm_prefix):
    config = _make_affine_pipeline_config(shm_prefix)
    manager = PipelineManager(config)
    rng = np.random.default_rng(13)
    matrix = np.array(
        [
            [1.0, 2.0, 0.0],
            [0.0, -1.0, 3.0],
        ],
        dtype=np.float32,
    )
    offset = np.array([0.5, -2.0], dtype=np.float32)

    manager.build()
    manager.start()

    manager.get_stream(f"{shm_prefix}_matrix").write(matrix)
    manager.get_stream(f"{shm_prefix}_offset").write(offset)

    for _ in range(128):
        vector = rng.standard_normal(3, dtype=np.float32)
        expected = matrix @ vector + offset
        manager.get_stream(f"{shm_prefix}_input").write(vector)
        result = manager.get_stream(f"{shm_prefix}_output").read_new(timeout=2.0)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    manager.shutdown()