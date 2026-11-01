"""Command-line entry point for one Python Worker process."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from collections.abc import Mapping, Sequence
from typing import Any

import structlog

from expertkit_worker.config import ConfigFileError, WorkerConfig, read_config_document
from expertkit_worker.factory import build_worker_application
from expertkit_worker.observability import configure_logging
from expertkit_worker.weights.metadata import (
    ModelMetadataError,
    ResolvedWorkerConfig,
    resolve_worker_config,
)

logger = structlog.get_logger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ek-worker")
    parser.add_argument(
        "--config",
        help="Path to the Python Worker YAML configuration; overrides EK_CONFIG",
    )
    return parser


async def run_config(config: WorkerConfig | ResolvedWorkerConfig) -> None:
    """Build and run one already validated Worker configuration."""

    application = await build_worker_application(config)
    remove_signal_handlers = application.install_signal_handlers()
    try:
        await application.run()
    finally:
        remove_signal_handlers()


async def _run_document(document: Mapping[str, Any]) -> None:
    try:
        resolved = await resolve_worker_config(document)
    except (ValueError, ModelMetadataError) as error:
        raise ConfigFileError(str(error)) from error
    config = resolved.config
    configure_logging(config.logging)
    context = structlog.contextvars.bind_contextvars(
        service="worker",
        worker_id=config.worker.id,
        backend=config.worker.backend.value,
        device=config.worker.device,
    )
    try:
        await run_config(resolved)
    finally:
        structlog.contextvars.reset_contextvars(**context)


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve configuration, configure logging once, and run the Worker."""

    arguments = _parser().parse_args(argv)
    config_path = arguments.config or os.environ.get("EK_CONFIG")
    if not config_path:
        _parser().error("--config or EK_CONFIG is required")
    try:
        document = read_config_document(config_path)
        asyncio.run(_run_document(document))
    except ConfigFileError as error:
        logging.basicConfig(level=logging.ERROR)
        logging.getLogger(__name__).error("invalid Worker configuration: %s", error)
        return 2
    except KeyboardInterrupt:
        return 130
    except BaseException:
        logger.critical("worker_process_failed", exc_info=True)
        return 1
    return 0
