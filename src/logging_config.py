"""Centralized logging configuration for Claude Researcher.

All modules should import their logger via:
    from src.logging_config import get_logger
    logger = get_logger(__name__)

Logs are written to a rotating file at:
    ~/.claude-researcher/researcher.log   (default)
    or $CLAUDE_RESEARCHER_LOG_FILE        (override)

Console output remains via Rich. This file logger captures
everything for post-hoc debugging, including errors that were
silently swallowed.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LOG_DIR = Path.home() / ".claude-researcher"
LOG_FILE = os.environ.get(
    "CLAUDE_RESEARCHER_LOG_FILE",
    str(LOG_DIR / "researcher.log"),
)
LOG_LEVEL = os.environ.get("CLAUDE_RESEARCHER_LOG_LEVEL", "DEBUG")
MAX_BYTES = 10 * 1024 * 1024  # 10 MB per file
BACKUP_COUNT = 3  # Keep 3 rotated files

_initialized = False


def _ensure_log_dir() -> None:
    """Create log directory if it doesn't exist."""
    log_path = Path(LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)


def setup_logging() -> None:
    """Initialize the centralized file logger (idempotent).

    Call once at startup (main.py / api server). Subsequent calls are no-ops.
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    _ensure_log_dir()

    # Root logger for the project namespace
    root = logging.getLogger("src")
    root.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.DEBUG))

    # Avoid duplicate handlers on re-import
    if any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        return

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    # Also configure the api namespace
    api_root = logging.getLogger("api")
    api_root.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.DEBUG))
    if not any(isinstance(h, RotatingFileHandler) for h in api_root.handlers):
        api_root.addHandler(file_handler)

    root.info("Logging initialized -> %s (level=%s)", LOG_FILE, LOG_LEVEL)


def get_logger(name: str) -> logging.Logger:
    """Get a logger for the given module name.

    Usage:
        logger = get_logger(__name__)
        logger.info("Something happened")
        logger.error("Failed to X", exc_info=True)
    """
    # Ensure logging is set up (safe to call multiple times)
    setup_logging()
    return logging.getLogger(name)
