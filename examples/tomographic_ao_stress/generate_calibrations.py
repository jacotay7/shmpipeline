"""Generate deterministic calibration arrays and the expected-dimensions map.

The default benchmark run generates zero/one calibrations in memory (see
``run_benchmark._load_calibrations``); this script materialises the same
deterministic arrays as ``.npy`` files for repeated memory-mappable runs, and
emits ``expected_dimensions.json`` describing every stream in the pipeline so
the dimensions in the plan can be checked mechanically.

Usage::

    python generate_calibrations.py --dimensions-only
    python generate_calibrations.py --out calibrations/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from shmpipeline import PipelineConfig

HERE = Path(__file__).resolve().parent
WFS_COUNT = 8


def expected_dimensions() -> dict:
    """Return every stream's shape/dtype/bytes from the checked-in CPU YAML."""
    config = PipelineConfig.from_yaml(HERE / "pipeline_cpu.yaml")
    streams = {}
    total = 0
    for spec in config.shared_memory:
        count = int(np.prod(spec.shape)) if spec.shape else 1
        nbytes = count * np.dtype(spec.dtype).itemsize
        total += nbytes
        streams[spec.name] = {
            "shape": list(spec.shape),
            "dtype": str(spec.dtype),
            "bytes": nbytes,
        }
    return {
        "wfs_count": WFS_COUNT,
        "slopes_per_wfs": 8192,
        "tomographic_vector": 65536,
        "actuators": 4096,
        "total_stream_bytes": total,
        "streams": streams,
    }


def calibration_arrays() -> dict[str, np.ndarray]:
    """Return the deterministic static calibration arrays keyed by stream."""
    rng = np.random.default_rng(2401)
    arrays: dict[str, np.ndarray] = {}
    for index in range(WFS_COUNT):
        arrays[f"tomo_wfs{index}_dark"] = np.zeros((256, 256), np.float32)
        arrays[f"tomo_wfs{index}_inverse_flat"] = np.ones(
            (256, 256), np.float32
        )
        arrays[f"tomo_wfs{index}_slope_offset"] = np.zeros(
            (64, 64, 2), np.float32
        )
    # A small random reconstructor keeps the multiply representative without
    # implying any physical meaning; the default runner uses zeros in memory.
    arrays["tomo_reconstructor"] = rng.standard_normal(
        (4096, 65536), dtype=np.float32
    ) * np.float32(1e-3)
    arrays["tomo_reconstructor_bias"] = np.zeros(4096, np.float32)
    arrays["tomo_command_offset"] = np.zeros(4096, np.float32)
    arrays["tomo_command_low"] = np.full(4096, -2.5, np.float32)
    arrays["tomo_command_high"] = np.full(4096, 2.5, np.float32)
    for loop, angle in (("a", np.deg2rad(12.0)), ("b", np.deg2rad(-7.0))):
        arrays[f"tt_{loop}_rotation"] = np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ],
            dtype=np.float32,
        )
        arrays[f"tt_{loop}_bias"] = np.zeros(2, np.float32)
    return arrays


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Directory to write .npy calibration files into.",
    )
    parser.add_argument(
        "--dimensions-only",
        action="store_true",
        help="Only (re)write expected_dimensions.json.",
    )
    args = parser.parse_args()

    dimensions_path = HERE / "expected_dimensions.json"
    dimensions_path.write_text(
        json.dumps(expected_dimensions(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {dimensions_path.name}")

    if args.dimensions_only or args.out is None:
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    for name, array in calibration_arrays().items():
        np.save(args.out / f"{name}.npy", array)
    print(
        f"wrote {len(calibration_arrays())} calibration arrays to {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
