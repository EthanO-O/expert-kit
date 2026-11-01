"""Load the Worker YAML file before any runtime component is initialized."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from expertkit_worker.config.models import WorkerConfig


class ConfigFileError(ValueError):
    """Indicate that a Worker configuration file cannot be read as a YAML mapping."""


def read_config_document(path: str | Path) -> dict[str, Any]:
    """Read one Worker YAML mapping without validating runtime model facts.

    Args:
        path: UTF-8 YAML file selected by the Worker ``--config`` argument.

    Returns:
        The document to resolve with Weight Server metadata before full validation.

    Raises:
        ConfigFileError: The file cannot be read, parsed, or does not contain a mapping.
    """

    config_path = Path(path)
    try:
        document: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigFileError(f"cannot read Worker configuration {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigFileError(f"invalid YAML in Worker configuration {config_path}: {exc}") from exc

    if not isinstance(document, Mapping):
        raise ConfigFileError("Worker configuration must contain one YAML mapping")

    return dict(document)


def load_config(path: str | Path) -> WorkerConfig:
    """Load and fully validate an explicit Worker configuration without network IO.

    Configurations omitting model facts must use ``read_config_document`` and the
    asynchronous metadata resolver instead.

    Raises:
        ConfigFileError: The file cannot be read as one YAML mapping.
        pydantic.ValidationError: The document violates the complete Worker schema.
    """

    return WorkerConfig.model_validate(read_config_document(path))
