"""Command-line entry points for shmpipeline."""

from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Sequence

from shmpipeline.config import PipelineConfig
from shmpipeline.graph import PipelineGraph, validate_pipeline_config
from shmpipeline.logging_utils import configure_colored_logging
from shmpipeline.manager import PipelineManager


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""
    parser = argparse.ArgumentParser(
        prog="shmpipeline",
        description="Shared-memory pipeline tools built on top of pyshmem.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Set the CLI logging verbosity.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a YAML pipeline configuration.",
    )
    validate_parser.add_argument(
        "config", help="Path to the YAML pipeline file."
    )

    describe_parser = subparsers.add_parser(
        "describe",
        help="Describe the pipeline graph without starting it.",
    )
    describe_parser.add_argument(
        "config", help="Path to the YAML pipeline file."
    )
    describe_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human text.",
    )

    run_parser = subparsers.add_parser(
        "run",
        help=(
            "Build and start a pipeline until interrupted or until duration "
            "elapses."
        ),
    )
    run_parser.add_argument("config", help="Path to the YAML pipeline file.")
    run_parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help=(
            "Optional run duration in seconds. Omit to run until interrupted."
        ),
    )
    run_parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.25,
        help="Runtime polling interval in seconds.",
    )
    run_parser.add_argument(
        "--json-status",
        action="store_true",
        help="Print the final runtime snapshot as JSON before exiting.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the shmpipeline CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_colored_logging(level=getattr(logging, args.log_level))

    if args.command == "validate":
        return _run_validate(args.config)
    if args.command == "describe":
        return _run_describe(args.config, as_json=args.json)
    if args.command == "run":
        return _run_pipeline(
            args.config,
            duration=args.duration,
            poll_interval=args.poll_interval,
            emit_json_status=args.json_status,
        )
    parser.error(f"unsupported command: {args.command}")
    return 2


def _run_validate(config_path: str) -> int:
    config = PipelineConfig.from_yaml(config_path)
    errors = validate_pipeline_config(config)
    if errors:
        print("Validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"Validation passed: {config_path}")
    return 0


def _run_describe(config_path: str, *, as_json: bool) -> int:
    config = PipelineConfig.from_yaml(config_path)
    errors = validate_pipeline_config(config)
    if errors:
        print("Validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    graph = PipelineGraph.from_config(config)
    if as_json:
        print(json.dumps(graph.to_dict(), indent=2, sort_keys=True))
    else:
        print(graph.describe())
    return 0


def _run_pipeline(
    config_path: str,
    *,
    duration: float | None,
    poll_interval: float,
    emit_json_status: bool,
) -> int:
    config = PipelineConfig.from_yaml(config_path)
    errors = validate_pipeline_config(config)
    if errors:
        print("Validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    manager = PipelineManager(config)
    started_at = time.monotonic()
    exit_code = 0
    try:
        manager.build()
        manager.start()
        while True:
            snapshot = manager.runtime_snapshot()
            if snapshot["state"] == "failed":
                manager.raise_if_failed()
            if (
                duration is not None
                and (time.monotonic() - started_at) >= duration
            ):
                break
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        exit_code = 130
    except Exception as exc:
        print(f"Pipeline run failed: {exc}")
        exit_code = 1
    finally:
        try:
            manager.shutdown(force=True)
        except Exception as exc:
            print(f"Pipeline shutdown failed: {exc}")
            exit_code = max(exit_code, 1)

    if emit_json_status:
        print(json.dumps(manager.runtime_snapshot(), indent=2, sort_keys=True))
    return exit_code
