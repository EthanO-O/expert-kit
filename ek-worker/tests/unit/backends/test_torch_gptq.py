"""Regression checks for the AutoGPTQ v1 packed layout."""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import save

from expertkit_worker.backends.torch import TorchGPTQWeightAdapter
from expertkit_worker.backends.torch.weights import dequantize_gptq
from expertkit_worker.weights import parse_safetensors


def _matrix(inputs: int, outputs: int, group: int) -> dict[str, torch.Tensor]:
    return {
        "qweight": torch.full((inputs // 8, outputs), -1717986919, dtype=torch.int32),
        "qzeros": torch.full((inputs // group, outputs // 8), 0x77777777, dtype=torch.int32),
        "scales": torch.ones(inputs // group, outputs, dtype=torch.float16),
        "g_idx": torch.arange(inputs, dtype=torch.int32) // group,
    }


def test_gptq_unpack_axes_and_zero_offset() -> None:
    inputs, outputs, group = 32, 16, 8
    generator = torch.Generator().manual_seed(57)
    qweight = torch.randint(
        -(2**31), 2**31, (inputs // 8, outputs), generator=generator, dtype=torch.int32
    )
    qzeros = torch.full((inputs // group, outputs // 8), 0x77777777, dtype=torch.int32)
    scales = torch.rand(inputs // group, outputs, generator=generator).half()
    expected = torch.empty(outputs, inputs, dtype=torch.float16)
    for o in range(outputs):
        for i in range(inputs):
            value = (int(qweight[i // 8, o]) >> (4 * (i % 8))) & 15
            expected[o, i] = (value - 8) * scales[i // group, o]
    actual = dequantize_gptq(
        qweight, qzeros, scales, in_features=inputs, out_features=outputs, group_size=group
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gptq_adapter_decodes_packed_expert(dtype: torch.dtype) -> None:
    hidden, intermediate, group = 16, 24, 8
    tensors = {}
    for role, inputs, outputs in (
        ("gate_proj", hidden, intermediate),
        ("up_proj", hidden, intermediate),
        ("down_proj", intermediate, hidden),
    ):
        tensors.update(
            {f"{role}.{name}": value for name, value in _matrix(inputs, outputs, group).items()}
        )
    adapter = TorchGPTQWeightAdapter(
        hidden_dim=hidden,
        intermediate_dim=intermediate,
        group_size=group,
        device="cpu",
        compute_dtype=dtype,
    )
    cpu = adapter.make_cpu_weight(parse_safetensors(bytearray(save(tensors))))
    ready = adapter.make_ready_weight(cpu, layer_id=0, expert_id=0)
    for value in ready.tensors:
        torch.testing.assert_close(value, torch.ones_like(value))
        assert value.dtype == dtype
    assert ready.storage_bytes == adapter.ready_weight_bytes()
    assert (
        sum(t.numel() * t.element_size() for t in tensors.values()) == adapter.source_tensor_bytes()
    )


@pytest.mark.parametrize("bad", ["g_idx", "qzeros", "qweight", "scales", "bias"])
def test_gptq_adapter_rejects_unsupported_tensors(bad: str) -> None:
    tensors = {
        f"{role}.{name}": tensor
        for role in ("gate_proj", "up_proj", "down_proj")
        for name, tensor in _matrix(16, 16, 8).items()
    }
    if bad == "g_idx":
        tensors["gate_proj.g_idx"] = torch.zeros(16, dtype=torch.int32)
    elif bad == "bias":
        tensors["gate_proj.bias"] = torch.ones(16, dtype=torch.float16)
    elif bad == "qzeros":
        tensors["gate_proj.qzeros"] = torch.zeros(1, 16, dtype=torch.int32)
    else:
        tensors[f"gate_proj.{bad}"] = tensors[f"gate_proj.{bad}"].to(torch.float32)
    adapter = TorchGPTQWeightAdapter(hidden_dim=16, intermediate_dim=16, group_size=8, device="cpu")
    with pytest.raises(ValueError):
        adapter.make_cpu_weight(parse_safetensors(bytearray(save(tensors))))
