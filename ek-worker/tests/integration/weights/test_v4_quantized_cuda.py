"""A100 qualification with synthetic experts using the official V4 dimensions."""

import pytest
import torch
from safetensors.torch import save

from expertkit_worker.backends import BackendBatch
from expertkit_worker.backends.torch import (
    TorchBackend,
    TorchFP4WeightAdapter,
    TorchW8A8WeightAdapter,
)
from expertkit_worker.weights import ReadyWeightTable, parse_safetensors


@pytest.mark.cuda
@pytest.mark.parametrize("recipe", ["fp4", "w8a8"])
def test_v4_dimensions_place_and_execute_bf16(recipe: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    hidden, intermediate = 4096, 2048
    tensors = {}
    for role, rows, cols in (
        ("w1", intermediate, hidden),
        ("w3", intermediate, hidden),
        ("w2", hidden, intermediate),
    ):
        base = f"layers.0.ffn.experts.0.{role}"
        if recipe == "fp4":
            tensors[base + ".weight"] = torch.full((rows, cols // 2), 0x22, dtype=torch.int8)
            tensors[base + ".scale"] = torch.full((rows, cols // 32), 117, dtype=torch.uint8).view(
                torch.float8_e8m0fnu
            )
        else:
            tensors[base + ".weight"] = torch.ones(rows, cols, dtype=torch.int8)
            tensors[base + ".weight_scale"] = torch.full((rows, 1), 2.0**-10, dtype=torch.bfloat16)
    adapter_type = TorchFP4WeightAdapter if recipe == "fp4" else TorchW8A8WeightAdapter
    adapter = adapter_type(
        hidden_dim=hidden,
        intermediate_dim=intermediate,
        device="cuda:0",
        compute_dtype=torch.bfloat16,
    )
    source = parse_safetensors(bytearray(save(tensors)))
    ready = adapter.make_ready_weight(adapter.make_cpu_weight(source), layer_id=0, expert_id=0)
    assert ready.storage_bytes == adapter.ready_weight_bytes()
    table = ReadyWeightTable(1, 1)
    table.publish(0, 0, ready)
    backend = TorchBackend(
        hidden_dim=hidden,
        intermediate_dim=intermediate,
        top_k=1,
        dtype=torch.bfloat16,
        device="cuda:0",
        acquire_many=table.acquire_many,
        expert_compute="deepseek_v4",
        swiglu_limit=10.0,
        linear_compute="fp8_reference" if recipe == "fp4" else "w8a8",
    )
    x = torch.ones(1, hidden, dtype=torch.bfloat16, device="cuda:0")
    batch = BackendBatch(
        layer_id=0,
        hidden_states=x,
        expert_ids=torch.zeros(1, 1, dtype=torch.int32, device="cuda:0"),
        routing_weights=torch.full((1, 1), 0.5, device="cuda:0"),
        distinct_expert_ids=(0,),
    )
    output = torch.empty_like(x)
    completion = backend.submit(batch, output)
    completion.wait_host()
    # Both input projections equal 4096 / 1024 = 4 exactly.
    middle = (torch.nn.functional.silu(torch.tensor(4.0)) * 4 * 0.5).to(torch.bfloat16)
    if recipe == "fp4":
        # The reference V4 activation scale is ceil_pow2(middle / 448) = 1/32.
        middle = ((middle.float() * 32).to(torch.float8_e4m3fn).float() / 32).to(torch.bfloat16)
    expected = (middle.float() * 2).to(torch.bfloat16)
    torch.testing.assert_close(output, torch.full_like(output, expected.item()), rtol=0, atol=0)
    assert table.usage_count(0, 0) == 1
    completion.close()
    assert table.usage_count(0, 0) == 0
