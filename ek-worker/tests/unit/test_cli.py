"""Tests for the Python Worker command-line entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from expertkit_worker import cli
from expertkit_worker.config.models import LoggingConfig
from expertkit_worker.weights.metadata import (
    ModelMetadata,
    ModelMetadataError,
    ModelMetadataUnavailable,
    ResolvedWorkerConfig,
)


class FakeApplication:
    def __init__(self, *, error: BaseException | None = None) -> None:
        self.error = error
        self.ran = False
        self.removed = False

    def install_signal_handlers(self) -> Any:
        def remove() -> None:
            self.removed = True

        return remove

    async def run(self) -> None:
        self.ran = True
        if self.error is not None:
            raise self.error


def _document() -> dict[str, Any]:
    return {
        "model": {
            "name": "fixture/model",
            "weight_version": "test",
            "activation_dtype": "bf16",
            "weight_dtype": "bf16",
        },
        "worker": {"id": "worker-0", "device": "cpu", "device_memory_limit": "1GiB"},
        "transport": {"type": "grpc", "listen": "localhost:50051", "advertise": "worker:50051"},
        "controller": {"endpoint": "localhost:50050"},
        "weight_manager": {
            "disk_cache": {"path": "/tmp/worker-test-weights"},
            "peer": {"listen": "localhost:50052", "advertise": "http://worker:50052"},
            "weight_server_endpoint": "http://weights:8080",
        },
    }


@pytest.fixture
def metadata_fetch(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    metadata = ModelMetadata(1, "qwen3_moe", 2, 0, 2, 4, 4, 8, 2, "bfloat16", None)
    fetch = AsyncMock(return_value=metadata)
    monkeypatch.setattr("expertkit_worker.weights.metadata.fetch_model_metadata", fetch)
    return fetch


def test_main_resolves_config_once_runs_application_and_removes_handlers(
    monkeypatch: pytest.MonkeyPatch, metadata_fetch: AsyncMock
) -> None:
    application = FakeApplication()
    configured: list[LoggingConfig] = []
    monkeypatch.setattr(cli, "read_config_document", lambda _path: _document())
    monkeypatch.setattr(cli, "configure_logging", configured.append)

    async def build(config: ResolvedWorkerConfig) -> FakeApplication:
        assert isinstance(config, ResolvedWorkerConfig)
        assert config.config.model.hidden_dim == 4
        assert config.metadata is metadata_fetch.return_value
        return application

    monkeypatch.setattr(cli, "build_worker_application", build)

    assert cli.main(["--config", "/tmp/worker.yaml"]) == 0
    assert application.ran is True
    assert application.removed is True
    assert configured == [LoggingConfig()]
    metadata_fetch.assert_awaited_once_with("http://weights:8080/", "fixture/model")


@pytest.mark.parametrize("explicit", [False, True])
def test_main_config_argument_overrides_environment(
    monkeypatch: pytest.MonkeyPatch, metadata_fetch: AsyncMock, explicit: bool
) -> None:
    application = FakeApplication()
    loaded_paths: list[str] = []
    monkeypatch.setenv("EK_CONFIG", "/tmp/from-environment.yaml")
    monkeypatch.setattr(
        cli,
        "read_config_document",
        lambda path: loaded_paths.append(path) or _document(),
    )
    monkeypatch.setattr(cli, "configure_logging", lambda _config: None)
    monkeypatch.setattr(cli, "build_worker_application", AsyncMock(return_value=application))

    arguments = ["--config", "/tmp/from-argument.yaml"] if explicit else []
    assert cli.main(arguments) == 0
    assert loaded_paths == ["/tmp/from-argument.yaml" if explicit else "/tmp/from-environment.yaml"]


@pytest.mark.parametrize(
    "error,code", [(RuntimeError("device failure"), 1), (KeyboardInterrupt(), 130)]
)
def test_main_runtime_failure_removes_handlers(
    monkeypatch: pytest.MonkeyPatch, metadata_fetch: AsyncMock, error: BaseException, code: int
) -> None:
    application = FakeApplication(error=error)
    monkeypatch.setattr(cli, "read_config_document", lambda _path: _document())
    monkeypatch.setattr(cli, "configure_logging", lambda _config: None)
    monkeypatch.setattr(cli, "build_worker_application", AsyncMock(return_value=application))

    assert cli.main(["--config", "/tmp/worker.yaml"]) == code
    assert application.removed is True


@pytest.mark.parametrize(
    "failure", [ModelMetadataError("invalid metadata"), ModelMetadataUnavailable("offline")]
)
def test_main_metadata_failure_is_configuration_error_before_build(
    monkeypatch: pytest.MonkeyPatch, metadata_fetch: AsyncMock, failure: Exception
) -> None:
    metadata_fetch.side_effect = failure
    build = AsyncMock()
    logging_setup = []
    monkeypatch.setattr(cli, "read_config_document", lambda _path: _document())
    monkeypatch.setattr(cli, "configure_logging", logging_setup.append)
    monkeypatch.setattr(cli, "build_worker_application", build)

    assert cli.main(["--config", "/tmp/worker.yaml"]) == 2
    assert logging_setup == []
    build.assert_not_awaited()


def test_main_model_conflict_is_configuration_error(
    monkeypatch: pytest.MonkeyPatch, metadata_fetch: AsyncMock
) -> None:
    document = _document()
    document["model"]["hidden_dim"] = 3
    build = AsyncMock()
    monkeypatch.setattr(cli, "read_config_document", lambda _path: document)
    monkeypatch.setattr(cli, "build_worker_application", build)

    assert cli.main(["--config", "/tmp/worker.yaml"]) == 2
    build.assert_not_awaited()


def test_main_returns_configuration_error_for_missing_file(tmp_path: Path) -> None:
    assert cli.main(["--config", str(tmp_path / "missing.yaml")]) == 2


def test_main_requires_config_argument_or_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EK_CONFIG", raising=False)
    with pytest.raises(SystemExit) as caught:
        cli.main([])

    assert caught.value.code == 2
