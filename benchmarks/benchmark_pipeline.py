"""Run a reproducible shmpipeline throughput benchmark and write JSON."""

from __future__ import annotations

import argparse
import json
import platform
import socket
import time
from pathlib import Path
from typing import Any

from shmpipeline.config import PipelineConfig
from shmpipeline.manager import PipelineManager


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--warmup", type=float, default=0.5)
    parser.add_argument("--source", default=None)
    parser.add_argument("--output-stream", default=None)
    parser.add_argument("--poll-interval", type=float, default=1e-4)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write the report and environment metadata to this JSON path.",
    )
    return parser


def _source(value: str | None) -> dict[str, Any] | None:
    if value is None:
        return None
    parts = value.split(":")
    if len(parts) not in {2, 3}:
        raise ValueError("source must use STREAM:PATTERN[:RATE_HZ]")
    result: dict[str, Any] = {
        "stream_name": parts[0],
        "pattern": parts[1],
    }
    if len(parts) == 3:
        result["rate_hz"] = float(parts[2])
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = PipelineConfig.from_yaml(args.config)
    manager = PipelineManager(config)
    started = time.time()
    try:
        manager.build()
        manager.start()
        report = manager.benchmark(
            duration_s=args.duration,
            warmup_s=args.warmup,
            source=_source(args.source),
            output_stream=args.output_stream,
            poll_interval=args.poll_interval,
        )
    finally:
        manager.shutdown(force=True)
    return {
        "benchmark": "pipeline",
        "config": str(args.config),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "started_at_unix": started,
        "report": report,
    }


def main() -> int:
    args = _parser().parse_args()
    result = run(args)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
