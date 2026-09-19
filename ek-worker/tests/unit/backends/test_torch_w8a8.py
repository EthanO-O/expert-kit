"""Compressed-tensors W8A8 validation and independent integer GEMM references."""

import pytest
import torch
from safetensors.torch import save

from expertkit_worker.backends import BackendBatch
from expertkit_worker.backends.torch import TorchBackend, TorchW8A8WeightAdapter
from expertkit_worker.backends.torch.w8a8 import w8a8_linear
from expertkit_worker.weights import ReadyWeightTable, parse_safetensors


def tensors(prefix: str = "layers.0.ffn.experts.0") -> dict[str, torch.Tensor]:
    """Build deterministic int8 weights with distinct projections and channel scales."""
    generator = torch.Generator().manual_seed(81)
    result = {}
    for name, shape in (("w1", (64, 32)), ("w3", (64, 32)), ("w2", (32, 64))):
        result[f"{prefix}.{name}.weight"] = torch.randint(
            -128, 128, shape, dtype=torch.int8, generator=generator
        )
        result[f"{prefix}.{name}.weight_scale"] = torch.linspace(0.001, 0.02, shape[0]).reshape(
            -1, 1
        )
    return result


def adapter(device: str = "cpu") -> TorchW8A8WeightAdapter:
    """Build an adapter matching the synthetic projection shapes."""
    return TorchW8A8WeightAdapter(
        hidden_dim=32, intermediate_dim=64, device=device, compute_dtype=torch.float32
    )


def test_packed_cache_memory_and_aliases() -> None:
    source = tensors()
    source = {
        k.replace("w1", "gate_proj").replace("w2", "down_proj").replace("w3", "up_proj"): v
        for k, v in source.items()
    }
    data = parse_safetensors(bytearray(save(source)))
    a = adapter()
    cpu = a.make_cpu_weight(data)
    ready = a.make_ready_weight(cpu, layer_id=0, expert_id=0)
    assert ready.storage_bytes == a.ready_weight_bytes()
    assert sum(r.byte_count for r in data.tensors.values()) <= a.source_tensor_bytes()
    assert a.cpu_extra_bytes() == 0
    assert ready.storage_bytes < 3 * 32 * 64 * 2
    for projection in ready.matrices:
        assert projection.dtype == torch.int8


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "shape",
        "dtype",
        "zero_scale",
        "nan",
        "negative",
        "offset",
        "nonzero_point",
        "duplicate",
        "wrong_prefix",
    ],
)
def test_rejects_unsupported_tensors(failure: str) -> None:
    values = tensors()
    base = "layers.0.ffn.experts.0.w1."
    match failure:
        case "missing":
            del values[base + "weight_scale"]
        case "shape":
            values[base + "weight_scale"] = torch.ones(64, 2)
        case "dtype":
            values[base + "weight"] = values[base + "weight"].float()
        case "zero_scale":
            values[base + "weight_scale"].zero_()
        case "nan":
            values[base + "weight_scale"][0] = float("nan")
        case "negative":
            values[base + "weight_scale"][0] = -1
        case "offset":
            values[base + "weight_offset"] = torch.zeros(64, 1)
        case "nonzero_point":
            values[base + "weight_zero_point"] = torch.ones(64, 1, dtype=torch.int8)
        case "duplicate":
            values[base.replace("w1", "gate_proj") + "weight"] = values[base + "weight"].clone()
        case "wrong_prefix":
            values[base.replace("experts.0", "experts.1") + "weight_scale"] = values.pop(
                base + "weight_scale"
            )
    with pytest.raises(ValueError):
        adapter().make_cpu_weight(parse_safetensors(bytearray(save(values))))


def test_accepts_zero_symmetric_zero_points() -> None:
    values = tensors()
    values["layers.0.ffn.experts.0.w1.weight_zero_point"] = torch.zeros(64, 1, dtype=torch.int8)
    adapter().make_cpu_weight(parse_safetensors(bytearray(save(values))))


def reference_linear(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Compute integer accumulation in float64 independently of torch._int_mm."""
    step = x.double().abs().amax(dim=1, keepdim=True) / 127
    quant = (x.double() / torch.where(step == 0, 1.0, step)).round().clamp(-128, 127)
    return ((quant @ weight.double().T) * step * scale.double().T).to(x.dtype)


@pytest.mark.parametrize("rows", [1, 7, 8, 17])
def test_integer_gemm_matches_reference(rows: int) -> None:
    torch.manual_seed(2)
    x = torch.randn(rows, 32)
    x[0].zero_()
    values = tensors()
    w = values["layers.0.ffn.experts.0.w1.weight"]
    s = values["layers.0.ffn.experts.0.w1.weight_scale"]
    torch.testing.assert_close(
        w8a8_linear(x, w, s), reference_linear(x, w, s), rtol=2e-5, atol=1e-5
    )


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)])
@pytest.mark.parametrize("expert_compute", ["deepseek_v4", "swiglu"])
def test_v4_w8a8_backend_weighting_clipping_and_completion(
    device: str, expert_compute: str
) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    a = adapter(device)
    ready = a.make_ready_weight(
        a.make_cpu_weight(parse_safetensors(bytearray(save(tensors())))), layer_id=0, expert_id=0
    )
    table = ReadyWeightTable(1, 1)
    table.publish(0, 0, ready)
    backend = TorchBackend(
        hidden_dim=32,
        intermediate_dim=64,
        top_k=1,
        dtype=torch.float32,
        device=device,
        acquire_many=table.acquire_many,
        expert_compute=expert_compute,
        swiglu_limit=1.0,
        linear_compute="w8a8",
    )
    torch.manual_seed(5)
    x = torch.randn(3, 32, device=device) * 4
    routing = torch.tensor([[0.12], [0.77], [0.0]], device=device)
    batch = BackendBatch(
        layer_id=0,
        hidden_states=x,
        expert_ids=torch.zeros(3, 1, dtype=torch.int32, device=device),
        routing_weights=routing,
        distinct_expert_ids=(0,),
    )
    output = torch.empty_like(x)
    completion = backend.submit(batch, output)
    completion.wait_host()
    matrices = [w.cpu() for w in ready.matrices]
    scales = [s.cpu() for s in ready.scales]
    gate = reference_linear(x.cpu(), matrices[0], scales[0])
    up = reference_linear(x.cpu(), matrices[1], scales[1])
    if expert_compute == "deepseek_v4":
        middle = torch.nn.functional.silu(gate.clamp(max=1)) * up.clamp(-1, 1) * routing.cpu()
        expected = reference_linear(middle, matrices[2], scales[2])
    else:
        middle = torch.nn.functional.silu(gate) * up
        expected = reference_linear(middle, matrices[2], scales[2]) * routing.cpu()
    torch.testing.assert_close(output.cpu(), expected, rtol=3e-5, atol=2e-5)
    completion.close()


def test_quantization_preserves_very_small_nonzero_rows() -> None:
    x = torch.full((1, 32), 1e-20)
    weight = torch.ones((64, 32), dtype=torch.int8)
    scale = torch.ones((64, 1))
    torch.testing.assert_close(
        w8a8_linear(x, weight, scale), x.sum().expand(1, 64), rtol=1e-6, atol=0
    )
