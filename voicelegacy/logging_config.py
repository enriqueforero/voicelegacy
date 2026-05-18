"""voicelegacy — Logging configuration using loguru.

Loguru is used over the stdlib logging module because:
- Zero boilerplate to get colored, structured output.
- Builtin file rotation.
- Better default UX for notebooks (colors in Colab).
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

_CONFIGURED: bool = False


def configure_logging(
    log_file: Path | None = None,
    level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> None:
    """Configure loguru with a console sink and optionally a file sink.

    Idempotent: calling twice does nothing (avoids duplicate handlers in
    notebooks when cells are re-run).

    Args:
        log_file: Optional file path to log to. If None, only console is used.
        level: Minimum log level for both sinks. Default INFO.
        rotation: Loguru rotation directive for the file sink.
        retention: Loguru retention directive for the file sink.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    logger.remove()  # remove default handler

    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            level=level,
            rotation=rotation,
            retention=retention,
            encoding="utf-8",
            enqueue=True,  # safe for multiprocessing
        )

    _CONFIGURED = True
    logger.info("Logging configured (level={}, file={})", level, log_file)


def get_logger():  # type: ignore[no-untyped-def]
    """Return the configured loguru logger. Configures it on first call."""
    if not _CONFIGURED:
        configure_logging()
    return logger
