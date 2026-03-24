"""Logging helpers for shmpipeline."""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a package logger with a stable namespace."""
    return logging.getLogger(f"shmpipeline.{name}")