from __future__ import annotations

import io
import logging
import time

import numpy as np
import pytest

from shmpipeline import PipelineConfig, PipelineManager, PipelineState
from shmpipeline.errors import WorkerProcessError
from shmpipeline.logging_utils import ColorFormatter


pytestmark = [pytest.mark.unit, pytest.mark.integration]


def _wait_for_next_write(stream, previous_count: int, *, timeout: float = 2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if stream.count > previous_count:
            return stream.read()
        time.sleep(1e-4)
    raise TimeoutError(f"timed out waiting for a new write on {stream.name!r}")


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
                    "input": f"{shm_prefix}_input",
                    "output": f"{shm_prefix}_output",
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
                {"name": f"{shm_prefix}_input", "shape": [3], "dtype": "float32", "storage": "cpu"},
                {"name": f"{shm_prefix}_matrix", "shape": [2, 3], "dtype": "float32", "storage": "cpu"},
                {"name": f"{shm_prefix}_offset", "shape": [2], "dtype": "float32", "storage": "cpu"},
                {"name": f"{shm_prefix}_output", "shape": [2], "dtype": "float32", "storage": "cpu"},
            ],
            "kernels": [
                {
                    "name": "affine_stage",
                    "kind": "cpu.affine_transform",
                    "input": f"{shm_prefix}_input",
                    "output": f"{shm_prefix}_output",
                    "auxiliary": [f"{shm_prefix}_matrix", f"{shm_prefix}_offset"],
                    "parameters": {},
                    "read_timeout": 0.1,
                }
            ],
        }
    )


def _make_elementwise_pipeline_config(shm_prefix: str, *, kind: str):
    return PipelineConfig.from_dict(
        {
            "shared_memory": [
                {"name": f"{shm_prefix}_input", "shape": [4], "dtype": "float32", "storage": "cpu"},
                {"name": f"{shm_prefix}_aux", "shape": [4], "dtype": "float32", "storage": "cpu"},
                {"name": f"{shm_prefix}_output", "shape": [4], "dtype": "float32", "storage": "cpu"},
            ],
            "kernels": [
                {
                    "name": "binary_stage",
                    "kind": kind,
                    "input": f"{shm_prefix}_input",
                    "output": f"{shm_prefix}_output",
                    "auxiliary": [f"{shm_prefix}_aux"],
                    "parameters": {},
                    "read_timeout": 0.1,
                }
            ],
        }
    )


def _make_custom_operation_pipeline_config(shm_prefix: str, *, operation: str):
    return PipelineConfig.from_dict(
        {
            "shared_memory": [
                {"name": f"{shm_prefix}_input", "shape": [4, 4], "dtype": "float32", "storage": "cpu"},
                {"name": f"{shm_prefix}_dark", "shape": [4, 4], "dtype": "float32", "storage": "cpu"},
                {"name": f"{shm_prefix}_flat", "shape": [4, 4], "dtype": "float32", "storage": "cpu"},
                {"name": f"{shm_prefix}_output", "shape": [4, 4], "dtype": "float32", "storage": "cpu"},
            ],
            "kernels": [
                {
                    "name": "custom_stage",
                    "kind": "cpu.custom_operation",
                    "operation": operation,
                    "input": f"{shm_prefix}_input",
                    "output": f"{shm_prefix}_output",
                    "auxiliary": {
                        "dark": f"{shm_prefix}_dark",
                        "flat": f"{shm_prefix}_flat",
                    },
                    "parameters": {},
                    "read_timeout": 0.1,
                }
            ],
        }
    )


def _make_custom_intrinsic_pipeline_config(shm_prefix: str, *, operation: str):
    return PipelineConfig.from_dict(
        {
            "shared_memory": [
                {"name": f"{shm_prefix}_input", "shape": [4, 4], "dtype": "float32", "storage": "cpu"},
                {"name": f"{shm_prefix}_dark", "shape": [4, 4], "dtype": "float32", "storage": "cpu"},
                {"name": f"{shm_prefix}_flat", "shape": [4, 4], "dtype": "float32", "storage": "cpu"},
                {"name": f"{shm_prefix}_low", "shape": [4, 4], "dtype": "float32", "storage": "cpu"},
                {"name": f"{shm_prefix}_high", "shape": [4, 4], "dtype": "float32", "storage": "cpu"},
                {"name": f"{shm_prefix}_output", "shape": [4, 4], "dtype": "float32", "storage": "cpu"},
            ],
            "kernels": [
                {
                    "name": "custom_stage",
                    "kind": "cpu.custom_operation",
                    "operation": operation,
                    "input": f"{shm_prefix}_input",
                    "output": f"{shm_prefix}_output",
                    "auxiliary": {
                        "dark": f"{shm_prefix}_dark",
                        "flat": f"{shm_prefix}_flat",
                        "low": f"{shm_prefix}_low",
                        "high": f"{shm_prefix}_high",
                    },
                    "parameters": {},
                    "read_timeout": 0.1,
                }
            ],
        }
    )


def _make_custom_matmul_pipeline_config(shm_prefix: str):
    return PipelineConfig.from_dict(
        {
            "shared_memory": [
                {"name": f"{shm_prefix}_input", "shape": [3], "dtype": "float32", "storage": "cpu"},
                {"name": f"{shm_prefix}_matrix", "shape": [2, 3], "dtype": "float32", "storage": "cpu"},
                {"name": f"{shm_prefix}_bias", "shape": [2], "dtype": "float32", "storage": "cpu"},
                {"name": f"{shm_prefix}_output", "shape": [2], "dtype": "float32", "storage": "cpu"},
            ],
            "kernels": [
                {
                    "name": "custom_stage",
                    "kind": "cpu.custom_operation",
                    "operation": "matrix @ input + bias",
                    "input": f"{shm_prefix}_input",
                    "output": f"{shm_prefix}_output",
                    "auxiliary": {
                        "matrix": f"{shm_prefix}_matrix",
                        "bias": f"{shm_prefix}_bias",
                    },
                    "parameters": {},
                    "read_timeout": 0.1,
                }
            ],
        }
    )


def _make_ao_pipeline_config(shm_prefix: str):
    return PipelineConfig.from_dict(
        {
            "shared_memory": [
                {"name": f"{shm_prefix}_image", "shape": [8, 8], "dtype": "float32", "storage": "cpu"},
                {"name": f"{shm_prefix}_centroids", "shape": [4, 4, 2], "dtype": "float32", "storage": "cpu"},
                {"name": f"{shm_prefix}_centroid_offset", "shape": [4, 4, 2], "dtype": "float32", "storage": "cpu"},
                {"name": f"{shm_prefix}_corrected", "shape": [4, 4, 2], "dtype": "float32", "storage": "cpu"},
                {"name": f"{shm_prefix}_flattened", "shape": [32], "dtype": "float32", "storage": "cpu"},
                {"name": f"{shm_prefix}_reconstructor", "shape": [6, 32], "dtype": "float32", "storage": "cpu"},
                {"name": f"{shm_prefix}_affine_offset", "shape": [6], "dtype": "float32", "storage": "cpu"},
                {"name": f"{shm_prefix}_open_loop", "shape": [6], "dtype": "float32", "storage": "cpu"},
                {"name": f"{shm_prefix}_command", "shape": [6], "dtype": "float32", "storage": "cpu"},
            ],
            "kernels": [
                {
                    "name": "centroid_stage",
                    "kind": "cpu.shack_hartmann_centroid",
                    "input": f"{shm_prefix}_image",
                    "output": f"{shm_prefix}_centroids",
                    "parameters": {"tile_size": 2},
                    "read_timeout": 0.1,
                },
                {
                    "name": "gain_offset_stage",
                    "kind": "cpu.scale_offset",
                    "input": f"{shm_prefix}_centroids",
                    "output": f"{shm_prefix}_corrected",
                    "auxiliary": [f"{shm_prefix}_centroid_offset"],
                    "parameters": {"gain": 1.75},
                    "read_timeout": 0.1,
                },
                {
                    "name": "flatten_stage",
                    "kind": "cpu.flatten",
                    "input": f"{shm_prefix}_corrected",
                    "output": f"{shm_prefix}_flattened",
                    "parameters": {},
                    "read_timeout": 0.1,
                },
                {
                    "name": "reconstructor_stage",
                    "kind": "cpu.affine_transform",
                    "input": f"{shm_prefix}_flattened",
                    "output": f"{shm_prefix}_open_loop",
                    "auxiliary": [f"{shm_prefix}_reconstructor", f"{shm_prefix}_affine_offset"],
                    "parameters": {},
                    "read_timeout": 0.1,
                },
                {
                    "name": "control_stage",
                    "kind": "cpu.leaky_integrator",
                    "input": f"{shm_prefix}_open_loop",
                    "output": f"{shm_prefix}_command",
                    "parameters": {"leak": 0.92, "gain": 0.35},
                    "read_timeout": 0.1,
                },
            ],
        }
    )


def _compute_centroids(image: np.ndarray, tile_size: int) -> np.ndarray:
    centroids = np.zeros((4, 4, 2), dtype=np.float32)
    center = 0.5 * (tile_size - 1)
    for tile_y in range(4):
        row_start = tile_y * tile_size
        for tile_x in range(4):
            col_start = tile_x * tile_size
            patch = image[
                row_start : row_start + tile_size,
                col_start : col_start + tile_size,
            ]
            total = float(np.sum(patch))
            if total <= 0.0:
                continue
            y_coords, x_coords = np.indices(patch.shape, dtype=np.float32)
            centroids[tile_y, tile_x, 0] = np.sum(y_coords * patch) / total - center
            centroids[tile_y, tile_x, 1] = np.sum(x_coords * patch) / total - center
    return centroids


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
    baseline = output_stream.count
    input_stream.write(payload)

    received = _wait_for_next_write(output_stream, baseline, timeout=2.0)
    np.testing.assert_allclose(received, payload * 2.5)

    manager.stop()
    manager.shutdown()


def test_manager_runs_elementwise_subtract_kernel_end_to_end(shm_prefix):
    config = _make_elementwise_pipeline_config(
        shm_prefix,
        kind="cpu.elementwise_subtract",
    )
    manager = PipelineManager(config)
    manager.build()
    manager.start()

    input_stream = manager.get_stream(f"{shm_prefix}_input")
    aux_stream = manager.get_stream(f"{shm_prefix}_aux")
    output_stream = manager.get_stream(f"{shm_prefix}_output")
    input_payload = np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32)
    aux_payload = np.array([1.5, 0.5, 2.0, 3.0], dtype=np.float32)
    aux_stream.write(aux_payload)
    baseline = output_stream.count
    input_stream.write(input_payload)

    received = _wait_for_next_write(output_stream, baseline, timeout=2.0)
    np.testing.assert_allclose(received, input_payload - aux_payload)

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
    baseline = manager.get_stream(f"{shm_prefix}_output").count
    manager.get_stream(f"{shm_prefix}_input").write(
        np.array([4.0, 1.0, 2.0], dtype=np.float32)
    )

    received = _wait_for_next_write(
        manager.get_stream(f"{shm_prefix}_output"),
        baseline,
        timeout=2.0,
    )
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
        baseline = manager.get_stream(f"{shm_prefix}_output").count
        manager.get_stream(f"{shm_prefix}_input").write(vector)
        result = _wait_for_next_write(
            manager.get_stream(f"{shm_prefix}_output"),
            baseline,
            timeout=2.0,
        )
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    manager.shutdown()


def test_manager_runs_custom_dark_flat_operation(shm_prefix):
    config = _make_custom_operation_pipeline_config(
        shm_prefix,
        operation="(input - dark) / flat",
    )
    manager = PipelineManager(config)
    manager.build()
    manager.start()

    input_image = np.arange(16, dtype=np.float32).reshape(4, 4) + 10.0
    dark_image = np.full((4, 4), 2.0, dtype=np.float32)
    flat_image = np.full((4, 4), 4.0, dtype=np.float32)
    manager.get_stream(f"{shm_prefix}_dark").write(dark_image)
    manager.get_stream(f"{shm_prefix}_flat").write(flat_image)
    output_stream = manager.get_stream(f"{shm_prefix}_output")
    baseline = output_stream.count
    manager.get_stream(f"{shm_prefix}_input").write(input_image)

    received = _wait_for_next_write(output_stream, baseline, timeout=2.0)
    np.testing.assert_allclose(received, (input_image - dark_image) / flat_image)

    manager.shutdown()


def test_manager_runs_custom_matmul_operation(shm_prefix):
    config = _make_custom_matmul_pipeline_config(shm_prefix)
    manager = PipelineManager(config)
    manager.build()
    manager.start()

    matrix = np.array(
        [
            [1.0, 2.0, -1.0],
            [0.5, 0.0, 3.0],
        ],
        dtype=np.float32,
    )
    bias = np.array([1.0, -2.0], dtype=np.float32)
    vector = np.array([2.0, -1.0, 4.0], dtype=np.float32)
    manager.get_stream(f"{shm_prefix}_matrix").write(matrix)
    manager.get_stream(f"{shm_prefix}_bias").write(bias)
    output_stream = manager.get_stream(f"{shm_prefix}_output")
    baseline = output_stream.count
    manager.get_stream(f"{shm_prefix}_input").write(vector)

    received = _wait_for_next_write(output_stream, baseline, timeout=2.0)
    np.testing.assert_allclose(received, matrix @ vector + bias)

    manager.shutdown()


def test_manager_runs_custom_intrinsic_clip_abs_operation(shm_prefix):
    config = _make_custom_intrinsic_pipeline_config(
        shm_prefix,
        operation="clip(abs(input - dark), low, high)",
    )
    manager = PipelineManager(config)
    manager.build()
    manager.start()

    input_image = np.array(
        [
            [-3.0, -1.0, 1.0, 3.0],
            [-5.0, -2.0, 2.0, 5.0],
            [-7.0, -3.0, 3.0, 7.0],
            [-9.0, -4.0, 4.0, 9.0],
        ],
        dtype=np.float32,
    )
    dark_image = np.ones((4, 4), dtype=np.float32)
    low = np.full((4, 4), 1.5, dtype=np.float32)
    high = np.full((4, 4), 6.0, dtype=np.float32)
    manager.get_stream(f"{shm_prefix}_dark").write(dark_image)
    manager.get_stream(f"{shm_prefix}_flat").write(np.ones((4, 4), dtype=np.float32))
    manager.get_stream(f"{shm_prefix}_low").write(low)
    manager.get_stream(f"{shm_prefix}_high").write(high)
    output_stream = manager.get_stream(f"{shm_prefix}_output")
    baseline = output_stream.count
    manager.get_stream(f"{shm_prefix}_input").write(input_image)

    received = _wait_for_next_write(output_stream, baseline, timeout=2.0)
    expected = np.clip(np.abs(input_image - dark_image), low, high)
    np.testing.assert_allclose(received, expected)

    manager.shutdown()


def test_manager_runs_custom_intrinsic_minimum_maximum_operation(shm_prefix):
    config = _make_custom_intrinsic_pipeline_config(
        shm_prefix,
        operation="maximum(input, dark) - minimum(flat, high)",
    )
    manager = PipelineManager(config)
    manager.build()
    manager.start()

    input_image = np.arange(16, dtype=np.float32).reshape(4, 4) - 4.0
    dark_image = np.full((4, 4), 3.0, dtype=np.float32)
    flat = np.linspace(0.5, 8.0, 16, dtype=np.float32).reshape(4, 4)
    high = np.full((4, 4), 5.0, dtype=np.float32)
    manager.get_stream(f"{shm_prefix}_dark").write(dark_image)
    manager.get_stream(f"{shm_prefix}_flat").write(flat)
    manager.get_stream(f"{shm_prefix}_low").write(np.zeros((4, 4), dtype=np.float32))
    manager.get_stream(f"{shm_prefix}_high").write(high)
    output_stream = manager.get_stream(f"{shm_prefix}_output")
    baseline = output_stream.count
    manager.get_stream(f"{shm_prefix}_input").write(input_image)

    received = _wait_for_next_write(output_stream, baseline, timeout=2.0)
    expected = np.maximum(input_image, dark_image) - np.minimum(flat, high)
    np.testing.assert_allclose(received, expected)

    manager.shutdown()


def test_color_formatter_emits_ansi_sequences():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(ColorFormatter("%(levelname)s %(message)s", use_color=True))
    logger = logging.getLogger("shmpipeline.test.color")
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False

    logger.info("colored message")

    output = stream.getvalue()
    assert "\033[32m" in output
    assert "colored message" in output


def test_manager_runs_basic_ao_pipeline_and_verifies_all_stages(shm_prefix):
    config = _make_ao_pipeline_config(shm_prefix)
    manager = PipelineManager(config)
    rng = np.random.default_rng(21)
    centroid_offset = rng.normal(0.0, 0.03, size=(4, 4, 2)).astype(np.float32)
    reconstructor = rng.normal(0.0, 0.2, size=(6, 32)).astype(np.float32)
    affine_offset = rng.normal(0.0, 0.1, size=(6,)).astype(np.float32)
    controller_state = np.zeros(6, dtype=np.float32)

    manager.build()
    manager.start()

    manager.get_stream(f"{shm_prefix}_centroid_offset").write(centroid_offset)
    manager.get_stream(f"{shm_prefix}_reconstructor").write(reconstructor)
    manager.get_stream(f"{shm_prefix}_affine_offset").write(affine_offset)

    for _ in range(32):
        image = rng.uniform(0.05, 1.05, size=(8, 8)).astype(np.float32)
        expected_centroids = _compute_centroids(image, 2)
        expected_corrected = 1.75 * expected_centroids - centroid_offset
        expected_flattened = expected_corrected.reshape(-1)
        expected_open_loop = reconstructor @ expected_flattened + affine_offset
        controller_state = 0.92 * controller_state + 0.35 * expected_open_loop
        expected_command = controller_state.copy()

        baseline = manager.get_stream(f"{shm_prefix}_command").count
        manager.get_stream(f"{shm_prefix}_image").write(image)
        observed_command = _wait_for_next_write(
            manager.get_stream(f"{shm_prefix}_command"),
            baseline,
            timeout=2.0,
        )

        np.testing.assert_allclose(
            manager.get_stream(f"{shm_prefix}_centroids").read(),
            expected_centroids,
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            manager.get_stream(f"{shm_prefix}_corrected").read(),
            expected_corrected,
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            manager.get_stream(f"{shm_prefix}_flattened").read(),
            expected_flattened,
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            manager.get_stream(f"{shm_prefix}_open_loop").read(),
            expected_open_loop,
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            observed_command,
            expected_command,
            rtol=1e-5,
            atol=1e-5,
        )

    manager.shutdown()