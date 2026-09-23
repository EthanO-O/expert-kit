"""Logging helpers for Ascend configuration generation."""

import logging

logger = logging.getLogger(__name__)


def long_banner(msg: str) -> None:
    """Log a wide progress banner."""

    banner(msg, width=50)


def short_banner(msg: str) -> None:
    """Log a compact progress banner."""

    banner(msg, width=40)


def banner(msg: str, width: int) -> None:
    """Log a banner with a fixed separator width."""

    logger.info("%s%s%s", "=" * width, msg, "=" * width)
