"""Run the source/sink plugin package example pipeline."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import yaml

from shmpipeline import PipelineConfig, PipelineManager
from shmpipeline.logging_utils import configure_colored_logging


def _saved_frames(output_dir: Path) -> list[Path]:
    return sorted(output_dir.glob("processed_frame_*.npy"))


def main() -> None:
    """Run the packaged source/sink example and verify saved output."""
    configure_colored_logging()
    logger = logging.getLogger("shmpipeline.example.source_sink_plugins")
    example_dir = Path(__file__).resolve().parent
    config_path = example_dir / "pipeline.yaml"
    output_dir = example_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    for frame_path in _saved_frames(output_dir):
        frame_path.unlink()

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["sinks"][0]["parameters"]["output_dir"] = str(output_dir)
    config = PipelineConfig.from_dict(raw)
    manager = PipelineManager(config)

    logger.info("building source/sink example pipeline from %s", config_path)
    manager.build()
    manager.start()

    expected_saved_frames = int(
        raw["sinks"][0]["parameters"].get("max_saved_frames", 8)
    )
    deadline = time.monotonic() + 5.0
    try:
        while time.monotonic() < deadline:
            manager.raise_if_failed()
            saved_frames = _saved_frames(output_dir)
            if len(saved_frames) >= expected_saved_frames:
                break
            time.sleep(0.05)
        else:
            raise TimeoutError(
                "timed out waiting for the example sink to save frames"
            )

        saved_frames = _saved_frames(output_dir)
        sample = np.load(saved_frames[-1])
        np.testing.assert_equal(sample.shape, (64, 64))
        np.testing.assert_equal(sample.dtype, np.dtype(np.float32))

        status = manager.status()
        logger.info(
            "source frames=%s sink frames=%s saved_files=%s",
            status["sources"]["simulated_camera"]["frames_written"],
            status["sinks"]["frame_archive"]["frames_consumed"],
            len(saved_frames),
        )
        print(
            "Saved "
            f"{len(saved_frames)} processed frames to {output_dir} "
            f"using source={raw['sources'][0]['kind']} "
            f"and sink={raw['sinks'][0]['kind']}"
        )
    finally:
        manager.shutdown(force=True)


if __name__ == "__main__":
    main()
