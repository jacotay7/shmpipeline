from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_observatory_example_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "observatory_ao_system"
        / "run_example.py"
    )
    spec = importlib.util.spec_from_file_location(
        "observatory_ao_example",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_render_shack_hartmann_image_round_trips_offsets():
    module = _load_observatory_example_module()
    offsets = np.array(
        [
            [[0.40, -0.35], [-0.55, 0.25]],
            [[0.15, 0.60], [-0.30, -0.45]],
        ],
        dtype=np.float32,
    )
    flux_map = np.array(
        [
            [80.0, 120.0],
            [95.0, 140.0],
        ],
        dtype=np.float32,
    )

    image = module.render_shack_hartmann_image(
        offsets,
        flux_map,
        tile_size=8,
        spot_sigma_px=1.05,
    )
    observed = module.compute_centroids(image, 8)

    np.testing.assert_allclose(observed, offsets, rtol=0.0, atol=0.08)


def test_make_command_limits_returns_ordered_bounds():
    module = _load_observatory_example_module()
    low_limit, high_limit = module.make_command_limits(
        np.random.default_rng(7),
        64,
    )

    assert low_limit.shape == (64,)
    assert high_limit.shape == (64,)
    assert np.all(low_limit < high_limit)
