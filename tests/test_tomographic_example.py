"""Configuration-level checks for the self-contained tomography example."""

from __future__ import annotations

from pathlib import Path

import pytest

from shmpipeline import PipelineConfig

pytestmark = pytest.mark.unit

EXAMPLE = (
    Path(__file__).resolve().parents[1] / "examples" / "tomographic_ao_stress"
)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_tomographic_example_is_self_driving_and_initialized(backend):
    config = PipelineConfig.from_yaml(EXAMPLE / f"pipeline_{backend}.yaml")
    sources = {source.name: source for source in config.sources}
    assert sources["wfs_cameras"].parameters["rate_hz"] == 500.0
    assert sources["tt_a_source"].parameters["rate_hz"] == 100.0
    assert sources["tt_b_source"].parameters["rate_hz"] == 1000.0

    streams = config.shared_memory_by_name
    for index in range(8):
        assert streams[f"tomo_wfs{index}_inverse_flat"].initial == {
            "pattern": "constant",
            "value": 1.0,
        }
    assert streams["tomo_reconstructor"].initial["pattern"] == "normal"
    assert streams["tomo_reconstructor"].initial["std"] > 0.0
    assert streams["tomo_command_low"].initial["value"] < 0.0
    assert streams["tomo_command_high"].initial["value"] > 0.0
    assert streams["tt_a_rotation"].initial["pattern"] == "values"
    assert streams["tt_b_rotation"].initial["pattern"] == "values"
