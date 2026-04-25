"""Structured logging setup for Orinode.

Single entry point for all loggers in the package. Keeps output format
consistent across training, UI server, and eval scripts.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a named logger with stderr handler (idempotent).

    Args:
        name: Logger name — use ``__name__`` in every module.
        level: Logging level.

    Returns:
        Configured ``Logger``.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def configure_file_logging(log_path: Path, level: int = logging.DEBUG) -> None:
    """Attach a file handler to the root logger.

    Args:
        log_path: Output file path. Parent directory is created if missing.
        level: Logging level for the file handler.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE))
    handler.setLevel(level)
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(min(logging.getLogger().level or logging.DEBUG, level))
