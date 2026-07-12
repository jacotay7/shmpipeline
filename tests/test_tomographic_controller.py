"""Tests for the fused cpu.tomographic_controller kernel."""

from __future__ import annotations

import numpy as np
import pytest

from shmpipeline.config import (
    AuxiliaryBinding,
    KernelConfig,
    SharedMemoryConfig,
)
from shmpipeline.errors import ConfigValidationError
from shmpipeline.kernel import KernelContext
from shmpipeline.kernels.cpu._common import centroid_tiles
from shmpipeline.kernels.cpu.tomographic_controller import (
    TomographicControllerCpuKernel,
)

pytestmark = [pytest.mark.unit]

TILE = 4
ROWS = COLS = 8
TILES = ROWS // TILE
SLOPES_PER_WFS = TILES * TILES * 2
SLOPES = SLOPES_PER_WFS * 8
ACTUATORS = 3


def _spec(name, shape):
    return SharedMemoryConfig(
        name=name, shape=tuple(shape), dtype=np.dtype("float32"), storage="cpu"
    )


def _build(leak=0.0, control_gain=1.0):
    shared = {}
    inputs = [f"w{i}" for i in range(8)]
    aux = []
    for i in range(8):
        shared[f"w{i}"] = _spec(f"w{i}", (ROWS, COLS))
        shared[f"w{i}_dark"] = _spec(f"w{i}_dark", (ROWS, COLS))
        shared[f"w{i}_flat"] = _spec(f"w{i}_flat", (ROWS, COLS))
        shared[f"w{i}_soff"] = _spec(f"w{i}_soff", (TILES, TILES, 2))
        aux += [
            AuxiliaryBinding(alias=f"wfs{i}_dark", name=f"w{i}_dark"),
            AuxiliaryBinding(alias=f"wfs{i}_inverse_flat", name=f"w{i}_flat"),
            AuxiliaryBinding(alias=f"wfs{i}_slope_offset", name=f"w{i}_soff"),
        ]
    shared["recon"] = _spec("recon", (ACTUATORS, SLOPES))
    shared["bias"] = _spec("bias", (ACTUATORS,))
    shared["coff"] = _spec("coff", (ACTUATORS,))
    shared["clo"] = _spec("clo", (ACTUATORS,))
    shared["chi"] = _spec("chi", (ACTUATORS,))
    shared["dm"] = _spec("dm", (ACTUATORS,))
    aux += [
        AuxiliaryBinding(alias="reconstructor", name="recon"),
        AuxiliaryBinding(alias="reconstructor_bias", name="bias"),
        AuxiliaryBinding(alias="command_offset", name="coff"),
        AuxiliaryBinding(alias="command_low", name="clo"),
        AuxiliaryBinding(alias="command_high", name="chi"),
    ]
    config = KernelConfig(
        name="tomo",
        kind="cpu.tomographic_controller",
        input="w0",
        output="dm",
        inputs=tuple(inputs),
        trigger_policy="all_new",
        auxiliary=tuple(aux),
        parameters={
            "tile_size": TILE,
            "leak": leak,
            "control_gain": control_gain,
            "command_gain": 1.0,
        },
    )
    return config, shared


def test_cpu_tomographic_controller_matches_reference():
    config, shared = _build()
    TomographicControllerCpuKernel.validate_config(config, shared)
    kernel = TomographicControllerCpuKernel(
        KernelContext(config=config, shared_memory=shared)
    )
    rng = np.random.default_rng(0)
    images = [rng.random((ROWS, COLS), dtype=np.float32) for _ in range(8)]
    aux = {}
    for i in range(8):
        aux[f"wfs{i}_dark"] = np.zeros((ROWS, COLS), np.float32)
        aux[f"wfs{i}_inverse_flat"] = np.ones((ROWS, COLS), np.float32)
        aux[f"wfs{i}_slope_offset"] = np.zeros((TILES, TILES, 2), np.float32)
    matrix = rng.random((ACTUATORS, SLOPES), dtype=np.float32)
    aux["reconstructor"] = matrix
    aux["reconstructor_bias"] = np.zeros(ACTUATORS, np.float32)
    aux["command_offset"] = np.zeros(ACTUATORS, np.float32)
    aux["command_low"] = np.full(ACTUATORS, -1e9, np.float32)
    aux["command_high"] = np.full(ACTUATORS, 1e9, np.float32)
    out = np.zeros(ACTUATORS, np.float32)

    kernel.compute_into(tuple(images), out, aux)

    slopes = np.zeros(SLOPES, np.float32)
    for i, image in enumerate(images):
        centroids = np.zeros((TILES, TILES, 2), np.float32)
        centroid_tiles(image, centroids, TILE)
        slopes[i * SLOPES_PER_WFS : (i + 1) * SLOPES_PER_WFS] = (
            centroids.reshape(-1)
        )
    np.testing.assert_allclose(out, matrix @ slopes, atol=1e-4)


def test_cpu_tomographic_controller_clips_command():
    config, shared = _build()
    kernel = TomographicControllerCpuKernel(
        KernelContext(config=config, shared_memory=shared)
    )
    images = [np.ones((ROWS, COLS), np.float32) for _ in range(8)]
    aux = {}
    for i in range(8):
        aux[f"wfs{i}_dark"] = np.zeros((ROWS, COLS), np.float32)
        aux[f"wfs{i}_inverse_flat"] = np.ones((ROWS, COLS), np.float32)
        aux[f"wfs{i}_slope_offset"] = np.zeros((TILES, TILES, 2), np.float32)
    aux["reconstructor"] = np.full((ACTUATORS, SLOPES), 5.0, np.float32)
    aux["reconstructor_bias"] = np.zeros(ACTUATORS, np.float32)
    aux["command_offset"] = np.zeros(ACTUATORS, np.float32)
    aux["command_low"] = np.full(ACTUATORS, -0.5, np.float32)
    aux["command_high"] = np.full(ACTUATORS, 0.5, np.float32)
    out = np.zeros(ACTUATORS, np.float32)
    kernel.compute_into(tuple(images), out, aux)
    assert np.all(out <= 0.5) and np.all(out >= -0.5)


def test_cpu_tomographic_controller_rejects_bad_reconstructor():
    config, shared = _build()
    shared["recon"] = _spec("recon", (ACTUATORS, SLOPES + 1))
    with pytest.raises(ConfigValidationError, match="reconstructor columns"):
        TomographicControllerCpuKernel.validate_config(config, shared)
