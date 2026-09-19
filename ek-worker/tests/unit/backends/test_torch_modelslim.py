"""ModelSlim W8A8_DYNAMIC validation and operator contract tests."""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save

from expertkit_worker.backends.torch.modelslim_w8a8 import (
    TorchModelSlimW8A8WeightAdapter,
    TorchModelSlimW8A8Weights,
    modelslim_w8a8_linear,
)
from expertkit_worker.config import WorkerConfig
from expertkit_worker.weights import parse_safetensors


class _FakeNpuRuntime:
    class _Device:
        type = "npu"

    device = _Device()

    def capture_current_work(self):
        return SimpleNamespace(wait_host=lambda: None)


def _values(prefix: str = "layers.0.ffn.experts.0") -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name, shape in (("w1", (8, 4)), ("w3", (8, 4)), ("w2", (4, 8))):
        result[f"{prefix}.{name}.weight"] = torch.ones(shape, dtype=torch.int8)
        result[f"{prefix}.{name}.weight_scale"] = torch.ones(shape[0], 1, dtype=torch.float32)
        result[f"{prefix}.{name}.weight_offset"] = torch.zeros(shape[0], 1, dtype=torch.float32)
    return result


def _adapter() -> TorchModelSlimW8A8WeightAdapter:
    return TorchModelSlimW8A8WeightAdapter(
        hidden_dim=4,
        intermediate_dim=8,
        runtime=_FakeNpuRuntime(),  # type: ignore[arg-type]
        compute_dtype=torch.float32,
    )


def test_modelslim_accepts_qwen_and_v4_projection_suffixes() -> None:
    a = _adapter()
    for values in (
        _values(),
        {
            k.replace("w1", "gate_proj").replace("w2", "down_proj").replace("w3", "up_proj"): v
            for k, v in _values().items()
        },
    ):
        cpu = a.make_cpu_weight(parse_safetensors(bytearray(save(values))))
        assert cpu.matrices[0].shape == (8, 4)
        assert a.ready_weight_bytes() == 3 * 4 * 8 + 3 * (2 * 8 + 4) * 4


@pytest.mark.parametrize("field", ["weight_scale", "weight_offset"])
def test_modelslim_rejects_missing_auxiliary(field: str) -> None:
    values = _values()
    del values[f"layers.0.ffn.experts.0.w1.{field}"]
    with pytest.raises(ValueError, match="ModelSlim"):
        _adapter().make_cpu_weight(parse_safetensors(bytearray(save(values))))


def test_modelslim_rejects_nonzero_offsets() -> None:
    values = _values()
    values["layers.0.ffn.experts.0.w1.weight_offset"][0] = 1
    with pytest.raises(ValueError, match="nonzero"):
        _adapter().make_cpu_weight(parse_safetensors(bytearray(save(values))))


def test_modelslim_rejects_cpu_startup() -> None:
    raw = {
        "model": {
            "name": "m",
            "weight_version": "v",
            "num_layers": 1,
            "experts_per_layer": 1,
            "hidden_dim": 4,
            "expert_intermediate_dim": 8,
            "top_k": 1,
            "activation_dtype": "bf16",
            "weight_dtype": "bf16",
            "quantization": {"type": "modelslim-w8a8-dynamic", "bits": 8, "group_size": None},
        },
        "worker": {"id": "w", "backend": "torch", "device": "cpu", "device_memory_limit": "1GiB"},
        "transport": {"type": "grpc", "listen": "127.0.0.1:1", "advertise": "w:1"},
        "controller": {"endpoint": "c:1"},
        "weight_manager": {
            "disk_cache": {"path": "/tmp/cache"},
            "peer": {"listen": "127.0.0.1:2", "advertise": "http://w:2"},
            "weight_server_endpoint": "http://s:1",
        },
    }
    with pytest.raises(ValueError, match="requires an indexed NPU"):
        WorkerConfig.model_validate(raw)


def test_modelslim_linear_uses_pinned_operator(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []
    module = SimpleNamespace(
        npu_dynamic_quant=lambda x, dst_type: (x.to(torch.int8), torch.ones(x.shape[0])),
        npu_quant_matmul=lambda *args, **kwargs: calls.append(args)
        or torch.zeros((args[0].shape[0], args[1].shape[1]), dtype=kwargs["output_dtype"]),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch_npu", module)
    weight = TorchModelSlimW8A8Weights(
        (
            torch.ones((4, 8), dtype=torch.int8),
            torch.ones((4, 8), dtype=torch.int8),
            torch.ones((8, 4), dtype=torch.int8),
        ),
        (torch.ones(8), torch.ones(8), torch.ones(4)),
        torch.float32,
    )
    output = modelslim_w8a8_linear(torch.ones((2, 4)), weight, 0)
    assert output.shape == (2, 8)
    assert calls and calls[0][1] is weight.matrices[0]
