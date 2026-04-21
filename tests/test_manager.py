from __future__ import annotations

import importlib
import io
import logging
import sys
import textwrap
import time

import numpy as np
import pyshmem
import pytest

from shmpipeline import (
    PipelineConfig,
    PipelineManager,
    PipelineState,
    get_default_registry,
)
from shmpipeline.errors import WorkerProcessError
from shmpipeline.logging_utils import ColorFormatter
from shmpipeline.shm_cleanup import close_stream

try:
    import torch
except Exception:  # pragma: no cover - exercised when torch is unavailable
    torch = None


pytestmark = [pytest.mark.unit, pytest.mark.integration]

CUDA_AVAILABLE = torch is not None and torch.cuda.is_available()


def _wait_for_next_write(
    stream,
    previous_count: int,
    *,
    timeout: float = 2.0,
    manager: PipelineManager | None = None,
):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if manager is not None:
            manager.poll_events()
            manager.raise_if_failed()
        if stream.count > previous_count:
            return stream.read()
        time.sleep(1e-4)
    raise TimeoutError(f"timed out waiting for a new write on {stream.name!r}")


def _to_host_array(value):
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()


def _wait_for_next_write_host(
    stream, previous_count: int, *, timeout: float = 2.0
):
    return _to_host_array(
        _wait_for_next_write(stream, previous_count, timeout=timeout)
    )


def _stream_payload(value, *, storage: str):
    if storage == "gpu":
        if not CUDA_AVAILABLE:
            pytest.skip("CUDA is not available")
        return torch.as_tensor(value, device="cuda:0")
    return np.asarray(value)


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


def _make_pipeline_config_for_storage(
    shm_prefix: str,
    *,
    kind: str,
    parameters: dict,
    storage: str,
):
    shared_memory = [
        {
            "name": f"{shm_prefix}_input",
            "shape": [4],
            "dtype": "float32",
            "storage": storage,
            **({"gpu_device": "cuda:0"} if storage == "gpu" else {}),
        },
        {
            "name": f"{shm_prefix}_output",
            "shape": [4],
            "dtype": "float32",
            "storage": storage,
            **({"gpu_device": "cuda:0"} if storage == "gpu" else {}),
        },
    ]
    return PipelineConfig.from_dict(
        {
            "shared_memory": shared_memory,
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
                    "input": f"{shm_prefix}_input",
                    "output": f"{shm_prefix}_output",
                    "auxiliary": [
                        f"{shm_prefix}_matrix",
                        f"{shm_prefix}_offset",
                    ],
                    "parameters": {},
                    "read_timeout": 0.1,
                }
            ],
        }
    )


def _make_affine_pipeline_config_for_storage(shm_prefix: str, *, storage: str):
    return PipelineConfig.from_dict(
        {
            "shared_memory": [
                {
                    "name": f"{shm_prefix}_input",
                    "shape": [3],
                    "dtype": "float32",
                    "storage": storage,
                    **({"gpu_device": "cuda:0"} if storage == "gpu" else {}),
                },
                {
                    "name": f"{shm_prefix}_matrix",
                    "shape": [2, 3],
                    "dtype": "float32",
                    "storage": storage,
                    **({"gpu_device": "cuda:0"} if storage == "gpu" else {}),
                },
                {
                    "name": f"{shm_prefix}_offset",
                    "shape": [2],
                    "dtype": "float32",
                    "storage": storage,
                    **({"gpu_device": "cuda:0"} if storage == "gpu" else {}),
                },
                {
                    "name": f"{shm_prefix}_output",
                    "shape": [2],
                    "dtype": "float32",
                    "storage": storage,
                    **({"gpu_device": "cuda:0"} if storage == "gpu" else {}),
                },
            ],
            "kernels": [
                {
                    "name": "affine_stage",
                    "kind": f"{storage}.affine_transform",
                    "input": f"{shm_prefix}_input",
                    "output": f"{shm_prefix}_output",
                    "auxiliary": [
                        f"{shm_prefix}_matrix",
                        f"{shm_prefix}_offset",
                    ],
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
                {
                    "name": f"{shm_prefix}_input",
                    "shape": [4],
                    "dtype": "float32",
                    "storage": "cpu",
                },
                {
                    "name": f"{shm_prefix}_aux",
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


def _make_elementwise_pipeline_config_for_storage(
    shm_prefix: str,
    *,
    kind: str,
    storage: str,
):
    return PipelineConfig.from_dict(
        {
            "shared_memory": [
                {
                    "name": f"{shm_prefix}_input",
                    "shape": [4],
                    "dtype": "float32",
                    "storage": storage,
                    **({"gpu_device": "cuda:0"} if storage == "gpu" else {}),
                },
                {
                    "name": f"{shm_prefix}_aux",
                    "shape": [4],
                    "dtype": "float32",
                    "storage": storage,
                    **({"gpu_device": "cuda:0"} if storage == "gpu" else {}),
                },
                {
                    "name": f"{shm_prefix}_output",
                    "shape": [4],
                    "dtype": "float32",
                    "storage": storage,
                    **({"gpu_device": "cuda:0"} if storage == "gpu" else {}),
                },
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
                {
                    "name": f"{shm_prefix}_input",
                    "shape": [4, 4],
                    "dtype": "float32",
                    "storage": "cpu",
                },
                {
                    "name": f"{shm_prefix}_dark",
                    "shape": [4, 4],
                    "dtype": "float32",
                    "storage": "cpu",
                },
                {
                    "name": f"{shm_prefix}_flat",
                    "shape": [4, 4],
                    "dtype": "float32",
                    "storage": "cpu",
                },
                {
                    "name": f"{shm_prefix}_output",
                    "shape": [4, 4],
                    "dtype": "float32",
                    "storage": "cpu",
                },
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


def _make_custom_operation_pipeline_config_for_storage(
    shm_prefix: str,
    *,
    operation: str,
    storage: str,
):
    return PipelineConfig.from_dict(
        {
            "shared_memory": [
                {
                    "name": f"{shm_prefix}_input",
                    "shape": [4, 4],
                    "dtype": "float32",
                    "storage": storage,
                    **({"gpu_device": "cuda:0"} if storage == "gpu" else {}),
                },
                {
                    "name": f"{shm_prefix}_dark",
                    "shape": [4, 4],
                    "dtype": "float32",
                    "storage": storage,
                    **({"gpu_device": "cuda:0"} if storage == "gpu" else {}),
                },
                {
                    "name": f"{shm_prefix}_flat",
                    "shape": [4, 4],
                    "dtype": "float32",
                    "storage": storage,
                    **({"gpu_device": "cuda:0"} if storage == "gpu" else {}),
                },
                {
                    "name": f"{shm_prefix}_output",
                    "shape": [4, 4],
                    "dtype": "float32",
                    "storage": storage,
                    **({"gpu_device": "cuda:0"} if storage == "gpu" else {}),
                },
            ],
            "kernels": [
                {
                    "name": "custom_stage",
                    "kind": f"{storage}.custom_operation",
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
                {
                    "name": f"{shm_prefix}_input",
                    "shape": [4, 4],
                    "dtype": "float32",
                    "storage": "cpu",
                },
                {
                    "name": f"{shm_prefix}_dark",
                    "shape": [4, 4],
                    "dtype": "float32",
                    "storage": "cpu",
                },
                {
                    "name": f"{shm_prefix}_flat",
                    "shape": [4, 4],
                    "dtype": "float32",
                    "storage": "cpu",
                },
                {
                    "name": f"{shm_prefix}_low",
                    "shape": [4, 4],
                    "dtype": "float32",
                    "storage": "cpu",
                },
                {
                    "name": f"{shm_prefix}_high",
                    "shape": [4, 4],
                    "dtype": "float32",
                    "storage": "cpu",
                },
                {
                    "name": f"{shm_prefix}_output",
                    "shape": [4, 4],
                    "dtype": "float32",
                    "storage": "cpu",
                },
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
                    "name": f"{shm_prefix}_bias",
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
                {
                    "name": f"{shm_prefix}_image",
                    "shape": [8, 8],
                    "dtype": "float32",
                    "storage": "cpu",
                },
                {
                    "name": f"{shm_prefix}_centroids",
                    "shape": [4, 4, 2],
                    "dtype": "float32",
                    "storage": "cpu",
                },
                {
                    "name": f"{shm_prefix}_centroid_offset",
                    "shape": [4, 4, 2],
                    "dtype": "float32",
                    "storage": "cpu",
                },
                {
                    "name": f"{shm_prefix}_corrected",
                    "shape": [4, 4, 2],
                    "dtype": "float32",
                    "storage": "cpu",
                },
                {
                    "name": f"{shm_prefix}_flattened",
                    "shape": [32],
                    "dtype": "float32",
                    "storage": "cpu",
                },
                {
                    "name": f"{shm_prefix}_reconstructor",
                    "shape": [6, 32],
                    "dtype": "float32",
                    "storage": "cpu",
                },
                {
                    "name": f"{shm_prefix}_affine_offset",
                    "shape": [6],
                    "dtype": "float32",
                    "storage": "cpu",
                },
                {
                    "name": f"{shm_prefix}_open_loop",
                    "shape": [6],
                    "dtype": "float32",
                    "storage": "cpu",
                },
                {
                    "name": f"{shm_prefix}_command",
                    "shape": [6],
                    "dtype": "float32",
                    "storage": "cpu",
                },
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
                    "auxiliary": [
                        f"{shm_prefix}_reconstructor",
                        f"{shm_prefix}_affine_offset",
                    ],
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
            centroids[tile_y, tile_x, 0] = (
                np.sum(y_coords * patch) / total - center
            )
            centroids[tile_y, tile_x, 1] = (
                np.sum(x_coords * patch) / total - center
            )
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


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_manager_runs_gpu_scale_kernel_end_to_end(shm_prefix):
    config = _make_pipeline_config_for_storage(
        shm_prefix,
        kind="gpu.scale",
        parameters={"factor": 2.5},
        storage="gpu",
    )
    manager = PipelineManager(config)
    manager.build()
    manager.start()

    input_stream = manager.get_stream(f"{shm_prefix}_input")
    output_stream = manager.get_stream(f"{shm_prefix}_output")
    payload = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    baseline = output_stream.count
    input_stream.write(_stream_payload(payload, storage="gpu"))

    received = _wait_for_next_write_host(output_stream, baseline, timeout=2.0)
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


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_manager_runs_gpu_elementwise_subtract_kernel_end_to_end(shm_prefix):
    config = _make_elementwise_pipeline_config_for_storage(
        shm_prefix,
        kind="gpu.elementwise_subtract",
        storage="gpu",
    )
    manager = PipelineManager(config)
    manager.build()
    manager.start()

    input_stream = manager.get_stream(f"{shm_prefix}_input")
    aux_stream = manager.get_stream(f"{shm_prefix}_aux")
    output_stream = manager.get_stream(f"{shm_prefix}_output")
    input_payload = np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32)
    aux_payload = np.array([1.5, 0.5, 2.0, 3.0], dtype=np.float32)
    aux_stream.write(_stream_payload(aux_payload, storage="gpu"))
    baseline = output_stream.count
    input_stream.write(_stream_payload(input_payload, storage="gpu"))

    received = _wait_for_next_write_host(output_stream, baseline, timeout=2.0)
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
    while (
        time.monotonic() < deadline and manager.state != PipelineState.FAILED
    ):
        manager.poll_events()
        time.sleep(0.05)

    assert manager.state == PipelineState.FAILED
    with pytest.raises(WorkerProcessError, match="intentional failure"):
        manager.raise_if_failed()

    manager.shutdown(force=True)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_manager_surfaces_gpu_worker_failures(shm_prefix):
    config = _make_pipeline_config_for_storage(
        shm_prefix,
        kind="gpu.raise_error",
        parameters={"message": "intentional failure"},
        storage="gpu",
    )
    manager = PipelineManager(config)
    manager.build()
    manager.start()

    manager.get_stream(f"{shm_prefix}_input").write(
        _stream_payload(
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), storage="gpu"
        )
    )

    deadline = time.monotonic() + 2.0
    while (
        time.monotonic() < deadline and manager.state != PipelineState.FAILED
    ):
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


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_manager_runs_gpu_affine_transform_kernel_end_to_end(shm_prefix):
    config = _make_affine_pipeline_config_for_storage(
        shm_prefix, storage="gpu"
    )
    manager = PipelineManager(config)
    manager.build()
    manager.start()

    manager.get_stream(f"{shm_prefix}_matrix").write(
        _stream_payload(
            np.array(
                [
                    [1.0, 2.0, 0.0],
                    [0.0, -1.0, 3.0],
                ],
                dtype=np.float32,
            ),
            storage="gpu",
        )
    )
    manager.get_stream(f"{shm_prefix}_offset").write(
        _stream_payload(np.array([0.5, -2.0], dtype=np.float32), storage="gpu")
    )
    baseline = manager.get_stream(f"{shm_prefix}_output").count
    manager.get_stream(f"{shm_prefix}_input").write(
        _stream_payload(
            np.array([4.0, 1.0, 2.0], dtype=np.float32), storage="gpu"
        )
    )

    received = _wait_for_next_write_host(
        manager.get_stream(f"{shm_prefix}_output"),
        baseline,
        timeout=2.0,
    )
    expected = np.array([6.5, 3.0], dtype=np.float32)
    np.testing.assert_allclose(received, expected)

    manager.shutdown()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_manager_runs_gpu_affine_transform_with_cpu_mirrors(shm_prefix):
    dimension = 256
    config = PipelineConfig.from_dict(
        {
            "shared_memory": [
                {
                    "name": f"{shm_prefix}_input",
                    "shape": [dimension],
                    "dtype": "float32",
                    "storage": "gpu",
                    "gpu_device": "cuda:0",
                },
                {
                    "name": f"{shm_prefix}_matrix",
                    "shape": [dimension, dimension],
                    "dtype": "float32",
                    "storage": "gpu",
                    "gpu_device": "cuda:0",
                    "cpu_mirror": True,
                },
                {
                    "name": f"{shm_prefix}_offset",
                    "shape": [dimension],
                    "dtype": "float32",
                    "storage": "gpu",
                    "gpu_device": "cuda:0",
                },
                {
                    "name": f"{shm_prefix}_output",
                    "shape": [dimension],
                    "dtype": "float32",
                    "storage": "gpu",
                    "gpu_device": "cuda:0",
                    "cpu_mirror": True,
                },
            ],
            "kernels": [
                {
                    "name": "affine_stage",
                    "kind": "gpu.affine_transform",
                    "input": f"{shm_prefix}_input",
                    "output": f"{shm_prefix}_output",
                    "auxiliary": [
                        f"{shm_prefix}_matrix",
                        f"{shm_prefix}_offset",
                    ],
                    "read_timeout": 0.1,
                }
            ],
        }
    )
    manager = PipelineManager(config)
    rng = np.random.default_rng(7)
    manager.build()
    manager.start()

    matrix = rng.standard_normal((dimension, dimension), dtype=np.float32)
    offset = rng.standard_normal(dimension, dtype=np.float32)
    vector = rng.standard_normal(dimension, dtype=np.float32)
    expected = matrix @ vector + offset

    manager.get_stream(f"{shm_prefix}_matrix").write(
        _stream_payload(matrix, storage="gpu")
    )
    manager.get_stream(f"{shm_prefix}_offset").write(
        _stream_payload(offset, storage="gpu")
    )
    output_stream = manager.get_stream(f"{shm_prefix}_output")
    baseline = output_stream.count
    manager.get_stream(f"{shm_prefix}_input").write(
        _stream_payload(vector, storage="gpu")
    )

    received_gpu = _wait_for_next_write_host(
        output_stream,
        baseline,
        timeout=5.0,
    )

    mirror_stream = pyshmem.open(f"{shm_prefix}_output")
    try:
        received_cpu = _wait_for_next_write(
            mirror_stream,
            baseline,
            timeout=5.0,
            manager=manager,
        )
    finally:
        mirror_stream.close()

    np.testing.assert_allclose(received_gpu, expected, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(received_cpu, expected, rtol=1e-4, atol=1e-4)

    manager.shutdown(force=True)


def test_manager_build_reuses_compatible_existing_shared_memory(shm_prefix):
    input_name = f"{shm_prefix}_input"
    output_name = f"{shm_prefix}_output"
    existing_input = pyshmem.create(
        input_name,
        shape=(4,),
        dtype=np.float32,
    )
    existing_output = pyshmem.create(
        output_name,
        shape=(4,),
        dtype=np.float32,
    )
    existing_input.write(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))

    manager = PipelineManager(
        _make_pipeline_config(shm_prefix, kind="cpu.copy", parameters={})
    )
    try:
        manager.build()

        reused_input = manager.get_stream(input_name)
        assert reused_input.count == 1
        np.testing.assert_allclose(
            reused_input.read(),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        )
    finally:
        existing_input.close()
        existing_output.close()
        manager.shutdown(force=True)


def test_manager_build_replaces_incompatible_existing_shared_memory(
    shm_prefix,
):
    input_name = f"{shm_prefix}_input"
    stale_stream = pyshmem.create(
        input_name,
        shape=(3,),
        dtype=np.float32,
    )
    stale_stream.close()

    manager = PipelineManager(
        _make_pipeline_config(shm_prefix, kind="cpu.copy", parameters={})
    )
    try:
        manager.build()

        rebuilt_input = manager.get_stream(input_name)
        assert rebuilt_input.shape == (4,)
        assert rebuilt_input.dtype == np.dtype(np.float32)
        assert rebuilt_input.count == 0
    finally:
        manager.shutdown(force=True)


def test_manager_build_reuses_stream_when_create_detects_existing_name(
    shm_prefix, monkeypatch
):
    manager = PipelineManager(
        _make_pipeline_config(shm_prefix, kind="cpu.copy", parameters={})
    )
    spec = manager.config.shared_memory_by_name[f"{shm_prefix}_input"]

    class _FakeStream:
        name = spec.name
        shape = (4,)
        dtype = np.dtype(np.float32)
        gpu_enabled = False

        def close(self) -> None:
            raise AssertionError("close should not be called")

    fake_stream = _FakeStream()
    open_calls = {"count": 0}

    def _fake_open_existing(_spec):
        open_calls["count"] += 1
        if open_calls["count"] == 1:
            return None
        return fake_stream

    monkeypatch.setattr(manager, "_open_existing_stream", _fake_open_existing)
    monkeypatch.setattr(
        pyshmem,
        "create",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            FileExistsError(spec.name)
        ),
    )

    stream = manager._build_stream(spec)

    assert stream is fake_stream
    assert open_calls["count"] == 2


def test_manager_build_recreates_stale_stream_after_duplicate_name(
    shm_prefix, monkeypatch
):
    manager = PipelineManager(
        _make_pipeline_config(shm_prefix, kind="cpu.copy", parameters={})
    )
    spec = manager.config.shared_memory_by_name[f"{shm_prefix}_input"]
    created_stream = object()
    create_calls: list[tuple[str, dict]] = []
    unlinked: list[str] = []

    monkeypatch.setattr(manager, "_open_existing_stream", lambda _spec: None)

    def _fake_create(name, **kwargs):
        create_calls.append((name, kwargs))
        if len(create_calls) == 1:
            raise FileExistsError(name)
        return created_stream

    monkeypatch.setattr(pyshmem, "create", _fake_create)
    monkeypatch.setattr(
        "shmpipeline.manager.unlink_stream_name",
        lambda name: unlinked.append(name),
    )

    stream = manager._build_stream(spec)

    assert stream is created_stream
    assert unlinked == [spec.name]
    assert len(create_calls) == 2


def test_close_stream_uses_direct_posix_unlink_when_available(monkeypatch):
    class _FakeSegment:
        def __init__(self, name: str) -> None:
            self._name = name

    class _FakeLockState:
        path = "/tmp/shmpipeline-fake.lock"

    class _FakeStream:
        def __init__(self) -> None:
            self.name = "demo"
            self.gpu_enabled = True
            self._data_shm = _FakeSegment("/ps_demo")
            self._metadata_shm = _FakeSegment("/ps_demo_meta")
            self._gpu_handle_shm = _FakeSegment("/ps_demo_gpu")
            self._lock_state = _FakeLockState()
            self.closed = False

        def close(self) -> None:
            self.closed = True

    import shmpipeline.shm_cleanup as shm_cleanup

    fake_stream = _FakeStream()
    unlinked: list[str] = []
    removed: list[str] = []
    dropped: list[str] = []

    monkeypatch.setattr(
        shm_cleanup,
        "_can_directly_unlink_posix_segments",
        lambda: True,
    )
    monkeypatch.setattr(shm_cleanup, "pyshmem_shared", None)
    monkeypatch.setattr(shm_cleanup, "_safe_posix_shm_unlink", unlinked.append)
    monkeypatch.setattr(shm_cleanup, "_safe_remove", removed.append)
    monkeypatch.setattr(shm_cleanup, "_drop_local_gpu_cache", dropped.append)
    monkeypatch.setattr(
        shm_cleanup.pyshmem,
        "unlink",
        lambda name: (_ for _ in ()).throw(AssertionError(name)),
    )

    close_stream(fake_stream, unlink=True)

    assert fake_stream.closed is True
    assert unlinked == ["/ps_demo", "/ps_demo_meta", "/ps_demo_gpu"]
    assert removed == ["/tmp/shmpipeline-fake.lock"]
    assert dropped == ["demo"]


def test_manager_does_not_reuse_compatible_gpu_shared_memory(shm_prefix):
    config = _make_pipeline_config_for_storage(
        shm_prefix,
        kind="gpu.copy",
        parameters={},
        storage="gpu",
    )
    manager = PipelineManager(config)

    class _FakeGpuStream:
        shape = (4,)
        dtype = np.dtype(np.float32)
        gpu_enabled = True
        cpu_mirror = False
        gpu_device = "cuda:0"

    assert (
        manager._stream_matches_spec(
            _FakeGpuStream(),
            config.shared_memory_by_name[f"{shm_prefix}_input"],
        )
        is False
    )


def test_manager_opens_existing_gpu_streams_without_cuda_attachment(
    shm_prefix,
    monkeypatch,
):
    config = _make_pipeline_config_for_storage(
        shm_prefix,
        kind="gpu.copy",
        parameters={},
        storage="gpu",
    )
    manager = PipelineManager(config)
    spec = config.shared_memory_by_name[f"{shm_prefix}_input"]
    open_calls: list[tuple[str, dict]] = []

    def _fake_open(name, **kwargs):
        open_calls.append((name, kwargs))
        raise FileNotFoundError(name)

    monkeypatch.setattr(pyshmem, "open", _fake_open)

    assert manager._open_existing_stream(spec) is None
    assert open_calls == [(spec.name, {})]


def test_manager_worker_retries_after_transient_lock_contention(shm_prefix):
    config = PipelineConfig.from_dict(
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
                    "kind": "cpu.copy",
                    "input": f"{shm_prefix}_input",
                    "output": f"{shm_prefix}_output",
                    "parameters": {},
                    "read_timeout": 0.01,
                }
            ],
        }
    )
    manager = PipelineManager(config)
    try:
        manager.build()
        manager.start()
        input_stream = manager.get_stream(f"{shm_prefix}_input")
        output_stream = manager.get_stream(f"{shm_prefix}_output")
        expected = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        baseline = output_stream.count

        with input_stream.locked():
            input_stream._mark_write_started()
            np.copyto(input_stream.read(safe=False), expected)
            input_stream._finish_write()
            time.sleep(0.05)

        observed = _wait_for_next_write(output_stream, baseline, timeout=2.0)
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            manager.poll_events()
            if manager.state == PipelineState.FAILED:
                break
            time.sleep(0.01)

        np.testing.assert_allclose(observed, expected)
        assert manager.state != PipelineState.FAILED
        assert manager.failures == ()
    finally:
        manager.shutdown(force=True)


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
    assert any(
        "state transition: initialized -> built" in message
        for message in messages
    )
    assert any(
        "state transition: built -> running" in message for message in messages
    )
    assert any(
        "worker started: kernel=stage" in message for message in messages
    )


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

    worker_status = manager.status()["workers"]["affine_stage"]
    assert worker_status["frames_processed"] >= 1
    assert worker_status["avg_exec_us"] is not None
    assert worker_status["jitter_us_rms"] is not None
    assert worker_status["metrics_window"] >= 1
    assert worker_status["throughput_hz"] >= 0.0
    assert worker_status["health"] == "active"
    assert worker_status["idle_s"] is not None

    manager.shutdown()


def test_manager_status_reports_waiting_input_health(shm_prefix):
    manager = PipelineManager(
        _make_pipeline_config(shm_prefix, kind="cpu.copy", parameters={})
    )
    manager.build()
    manager.start()

    status = manager.status()
    worker_status = status["workers"]["stage"]
    assert worker_status["health"] == "waiting-input"
    assert worker_status["idle_s"] is not None
    assert status["summary"]["waiting_workers"] == 1

    manager.shutdown()


def test_manager_can_run_with_extended_registry(
    tmp_path, shm_prefix, monkeypatch
):
    module_name = f"custom_bias_kernel_{shm_prefix}"
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        textwrap.dedent(
            """
            import numpy as np

            from shmpipeline.errors import ConfigValidationError
            from shmpipeline.kernel import Kernel


            class BiasCpuKernel(Kernel):
                kind = \"test.bias\"
                storage = \"cpu\"

                @classmethod
                def validate_config(cls, config, shared_memory):
                    super().validate_config(config, shared_memory)
                    if \"bias\" not in config.parameters:
                        raise ConfigValidationError(
                            \"test.bias requires a 'bias' parameter\"
                        )

                def compute_into(self, trigger_input, output, auxiliary_inputs):
                    output[...] = np.asarray(trigger_input) + float(
                        self.context.config.parameters[\"bias\"]
                    )
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)

    config = PipelineConfig.from_dict(
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
                    "name": "bias_stage",
                    "kind": "test.bias",
                    "input": f"{shm_prefix}_input",
                    "output": f"{shm_prefix}_output",
                    "parameters": {"bias": 1.5},
                    "read_timeout": 0.1,
                }
            ],
        }
    )
    registry = get_default_registry().extended(module.BiasCpuKernel)
    manager = PipelineManager(config, registry=registry)
    manager.build()
    manager.start()

    payload = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    output_stream = manager.get_stream(f"{shm_prefix}_output")
    baseline = output_stream.count
    manager.get_stream(f"{shm_prefix}_input").write(payload)
    received = _wait_for_next_write(
        output_stream,
        baseline,
        timeout=2.0,
        manager=manager,
    )

    np.testing.assert_allclose(received, payload + 1.5)

    manager.shutdown()


def test_manager_cpu_workers_can_compute_directly_into_shared_output(
    tmp_path, shm_prefix, monkeypatch
):
    module_name = f"direct_output_kernel_{shm_prefix}"
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        textwrap.dedent(
            """
            import numpy as np

            from shmpipeline.kernel import Kernel


            class DirectOutputProbeCpuKernel(Kernel):
                kind = \"test.direct_output\"
                storage = \"cpu\"

                def compute_into(self, trigger_input, output, auxiliary_inputs):
                    del auxiliary_inputs
                    if output is self.output_buffer:
                        raise RuntimeError(\"expected shared-memory output view\")
                    np.add(np.asarray(trigger_input), 1.0, out=output)
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)

    config = PipelineConfig.from_dict(
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
                    "name": "direct_output_stage",
                    "kind": "test.direct_output",
                    "input": f"{shm_prefix}_input",
                    "output": f"{shm_prefix}_output",
                    "read_timeout": 0.1,
                }
            ],
        }
    )
    registry = get_default_registry().extended(
        module.DirectOutputProbeCpuKernel
    )
    manager = PipelineManager(config, registry=registry)
    manager.build()
    manager.start()

    output_stream = manager.get_stream(f"{shm_prefix}_output")
    baseline = output_stream.count
    manager.get_stream(f"{shm_prefix}_input").write(
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    )
    received = _wait_for_next_write(
        output_stream,
        baseline,
        timeout=2.0,
        manager=manager,
    )

    np.testing.assert_allclose(
        received,
        np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32),
    )

    manager.shutdown()


def test_manager_runs_source_and_sink_plugins(
    tmp_path, shm_prefix, monkeypatch
):
    module_name = f"endpoint_plugins_{shm_prefix}"
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        textwrap.dedent(
            """
            import numpy as np

            from shmpipeline.sink import Sink
            from shmpipeline.source import Source


            CONSUMED = []


            class EmitsOnceSource(Source):
                kind = "test.once_source"
                storage = "cpu"

                def __init__(self, context):
                    super().__init__(context)
                    self._emitted = False

                def read(self):
                    if self._emitted:
                        return None
                    self._emitted = True
                    return np.asarray(
                        self.context.config.parameters["payload"],
                        dtype=np.float32,
                    )


            class RecordingSink(Sink):
                kind = "test.recording_sink"
                storage = "cpu"

                def consume(self, value):
                    CONSUMED.append(np.asarray(value).copy())
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)

    config = PipelineConfig.from_dict(
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
            "sources": [
                {
                    "name": "camera",
                    "kind": "test.once_source",
                    "stream": f"{shm_prefix}_input",
                    "parameters": {"payload": [1.0, 2.0, 3.0, 4.0]},
                    "poll_interval": 0.01,
                }
            ],
            "kernels": [
                {
                    "name": "copy_stage",
                    "kind": "cpu.copy",
                    "input": f"{shm_prefix}_input",
                    "output": f"{shm_prefix}_output",
                    "read_timeout": 0.1,
                }
            ],
            "sinks": [
                {
                    "name": "display",
                    "kind": "test.recording_sink",
                    "stream": f"{shm_prefix}_output",
                    "read_timeout": 0.1,
                    "pause_sleep": 0.01,
                }
            ],
        }
    )
    registry = get_default_registry().extended(
        sources=(module.EmitsOnceSource,),
        sinks=(module.RecordingSink,),
    )
    manager = PipelineManager(config, registry=registry)
    manager.build()
    manager.start()

    deadline = time.monotonic() + 2.0
    while not module.CONSUMED and time.monotonic() < deadline:
        manager.poll_events()
        manager.raise_if_failed()
        time.sleep(1e-3)

    assert module.CONSUMED
    np.testing.assert_allclose(
        module.CONSUMED[-1],
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
    )

    status = manager.status()
    assert status["sources"]["camera"]["frames_written"] >= 1
    assert status["sinks"]["display"]["frames_consumed"] >= 1

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
    np.testing.assert_allclose(
        received, (input_image - dark_image) / flat_image
    )

    manager.shutdown()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_manager_runs_gpu_custom_dark_flat_operation(shm_prefix):
    config = _make_custom_operation_pipeline_config_for_storage(
        shm_prefix,
        operation="(input - dark) / flat",
        storage="gpu",
    )
    manager = PipelineManager(config)
    manager.build()
    manager.start()

    input_image = np.arange(16, dtype=np.float32).reshape(4, 4) + 10.0
    dark_image = np.full((4, 4), 2.0, dtype=np.float32)
    flat_image = np.full((4, 4), 4.0, dtype=np.float32)
    manager.get_stream(f"{shm_prefix}_dark").write(
        _stream_payload(dark_image, storage="gpu")
    )
    manager.get_stream(f"{shm_prefix}_flat").write(
        _stream_payload(flat_image, storage="gpu")
    )
    output_stream = manager.get_stream(f"{shm_prefix}_output")
    baseline = output_stream.count
    manager.get_stream(f"{shm_prefix}_input").write(
        _stream_payload(input_image, storage="gpu")
    )

    received = _wait_for_next_write_host(output_stream, baseline, timeout=2.0)
    np.testing.assert_allclose(
        received, (input_image - dark_image) / flat_image
    )

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
    manager.get_stream(f"{shm_prefix}_flat").write(
        np.ones((4, 4), dtype=np.float32)
    )
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
    manager.get_stream(f"{shm_prefix}_low").write(
        np.zeros((4, 4), dtype=np.float32)
    )
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
    handler.setFormatter(
        ColorFormatter("%(levelname)s %(message)s", use_color=True)
    )
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
            timeout=5.0,
            manager=manager,
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
