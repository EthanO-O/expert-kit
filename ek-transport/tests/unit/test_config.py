"""Optional frontend runtime configuration stays outside framework APIs."""

import sys
import types

import pytest
import torch

from expertkit_transport.config import runtime_from_environment


def test_absent_config_keeps_native_dependency_optional(monkeypatch) -> None:
    monkeypatch.delenv("EK_TRANSPORT_CONFIG", raising=False)
    imported_before = "mooncake.engine" in sys.modules
    assert runtime_from_environment(torch.device("cpu")) is None
    assert ("mooncake.engine" in sys.modules) == imported_before


@pytest.mark.parametrize("unknown", [False, True])
def test_environment_file_passes_only_supported_options_and_frontend_device(
    monkeypatch, tmp_path, unknown: bool
) -> None:
    import expertkit_transport.transports.transfer_engine.runtime as runtime_module

    captured = []
    from dataclasses import dataclass

    @dataclass
    class Config:
        segment_name: str
        device: torch.device
        max_registered_bytes: int | None = None

    def runtime(config):
        captured.append(config)
        return types.SimpleNamespace(config=config)

    monkeypatch.setattr(runtime_module, "TransferEngineRuntimeConfig", Config)
    monkeypatch.setattr(runtime_module, "TransferEngineRuntime", runtime)
    path = tmp_path / "transport.toml"
    path.write_text(
        '[transfer_engine]\nsegment_name="frontend:0"\n'
        + ('protocol="rdma"\n' if unknown else "max_registered_bytes=4096\n")
    )
    monkeypatch.setenv("EK_TRANSPORT_CONFIG", str(path))
    if unknown:
        with pytest.raises(ValueError, match=r"unsupported.*protocol"):
            runtime_from_environment(torch.device("cpu"))
        assert not captured
    else:
        selected = runtime_from_environment(torch.device("cpu"))
        assert selected.config.device == torch.device("cpu")
        assert selected.config.max_registered_bytes == 4096
