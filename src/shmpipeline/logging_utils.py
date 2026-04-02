"""Logging helpers for shmpipeline."""

from __future__ import annotations

import logging
import sys


class ColorFormatter(logging.Formatter):
    """Formatter that adds ANSI colors based on the log level."""

    RESET = "\033[0m"
    COLORS = {
        logging.DEBUG: "\033[36m",
        logging.INFO: "\033[32m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[35m",
    }

    def __init__(self, fmt: str, *, use_color: bool = True) -> None:
        """Initialize the formatter with optional ANSI colors."""
        super().__init__(fmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        """Format one log record, optionally colorizing its level name."""
        if not self.use_color:
            return super().format(record)
        color = self.COLORS.get(record.levelno)
        if color is None:
            return super().format(record)
        original_levelname = record.levelname
        try:
            record.levelname = f"{color}{original_levelname}{self.RESET}"
            return super().format(record)
        finally:
            record.levelname = original_levelname


def get_logger(name: str) -> logging.Logger:
    """Return a package logger with a stable namespace."""
    return logging.getLogger(f"shmpipeline.{name}")


def configure_colored_logging(
    *,
    level: int = logging.INFO,
    fmt: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
) -> None:
    """Configure root logging with a color-aware stream handler."""
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(ColorFormatter(fmt, use_color=True))
    logging.basicConfig(level=level, handlers=[handler], force=True)
