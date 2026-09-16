"""Official V4 FP4 serialization, scale decoding, and activation rounding tests."""

import json

import pytest
import torch

from expertkit_worker.backends.torch import TorchFP4WeightAdapter
from expertkit_worker.backends.torch.fp4 import fp8_activation_reference
from expertkit_worker.weights import parse_safetensors


def bundle(scale_byte: int = 127, extra: bool = False) -> bytearray:
    """Encode all sixteen E2M1 values in their documented nibble order."""
    header, data = {}, bytearray()
    for role, rows, cols in (("w1", 256, 128), ("w3", 256, 128), ("w2", 128, 256)):
        packed = bytes([0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]) * (rows * cols // 16)
        for field, dtype, shape, raw in (
            ("weight", "I8", [rows, cols // 2], packed),
            ("scale", "F8_E8M0", [rows, cols // 32], bytes([scale_byte]) * (rows * cols // 32)),
        ):
            start = len(data)
            data.extend(raw)
            header[f"layers.0.ffn.experts.0.{role}.{field}"] = {
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [start, len(data)],
            }
    if extra:
        header["unknown"] = {
            "dtype": "I8",
            "shape": [1],
            "data_offsets": [len(data), len(data) + 1],
        }
        data.append(0)
    encoded = json.dumps(header).encode()
    return bytearray(len(encoded).to_bytes(8, "little") + encoded + data)


def test_fp4_values_scales_shapes_and_memory() -> None:
    a = TorchFP4WeightAdapter(hidden_dim=128, intermediate_dim=256, device="cpu")
    source = parse_safetensors(bundle(128))
    cpu = a.make_cpu_weight(source)
    assert a.cpu_extra_bytes() == 0
    assert sum(t.byte_count for t in source.tensors.values()) == a.source_tensor_bytes()
    assert cpu[0][0].dtype == torch.uint8
    ready = a.make_ready_weight(cpu, layer_id=0, expert_id=0)
    expected = (
        torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.bfloat16
        )
        * 2
    )
    torch.testing.assert_close(ready.gate_proj, expected.repeat(256, 8), rtol=0, atol=0)
    assert ready.down_proj.shape == (128, 256)
    assert ready.storage_bytes == a.ready_weight_bytes()


def test_fp4_preserves_bf16_range_without_fp16_intermediate() -> None:
    a = TorchFP4WeightAdapter(hidden_dim=128, intermediate_dim=256, device="cpu")
    ready = a.make_ready_weight(
        a.make_cpu_weight(parse_safetensors(bundle(145))), layer_id=0, expert_id=0
    )
    assert torch.isfinite(ready.gate_proj).all()
    assert ready.gate_proj.max() > 65504


@pytest.mark.parametrize("scale,extra", [(255, False), (127, True)])
def test_fp4_rejects_nan_scales_and_unrecognized_tensors(scale: int, extra: bool) -> None:
    a = TorchFP4WeightAdapter(hidden_dim=128, intermediate_dim=256, device="cpu")
    with pytest.raises(ValueError):
        a.make_cpu_weight(parse_safetensors(bundle(scale, extra)))


@pytest.mark.parametrize("dim", [0, 32, 129])
def test_fp4_rejects_invalid_activation_block_dimensions(dim: int) -> None:
    with pytest.raises(ValueError):
        TorchFP4WeightAdapter(hidden_dim=dim, intermediate_dim=256, device="cpu")


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)])
def test_fp8_activation_rounding_and_zero_rows(device: str) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    x = torch.zeros(2, 256, dtype=torch.bfloat16, device=device)
    x[0, 0:3] = torch.tensor([448, 1.125, -1.3125], device=device)
    x[0, 128:131] = torch.tensor([896, 2.25, -2.625], device=device)
    result = fp8_activation_reference(x)
    expected = x.clone()
    expected[0, 2], expected[0, 130] = -1.25, -2.5
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
