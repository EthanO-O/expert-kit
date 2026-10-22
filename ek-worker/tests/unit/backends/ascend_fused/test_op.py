from types import ModuleType

import pytest
import torch

from expertkit_worker.backends.ascend_fused import op


@pytest.fixture(autouse=True)
def native_operators(monkeypatch):
    operators = ModuleType("torch_npu")
    monkeypatch.setattr(op, "torch_npu", operators)


def test_dynamic_w8a8_passes_list_weights_and_scales_to_gmm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def route(hidden: torch.Tensor, _ids: torch.Tensor, **kwargs: object):
        calls.append({"operator": "route", **kwargs})
        return (
            hidden.to(torch.int8),
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([1, 1], dtype=torch.int32),
            torch.ones(hidden.shape[0], dtype=torch.float32),
        )

    def grouped_matmul(**kwargs: object):
        calls.append({"operator": "gmm", **kwargs})
        x = kwargs["x"]
        weight = kwargs["weight"]
        assert isinstance(x, list) and isinstance(weight, list)
        return [torch.ones((x[0].shape[0], weight[0].shape[1]), dtype=torch.bfloat16)]

    def dynamic_quant(x: torch.Tensor, **_kwargs: object):
        return x.to(torch.int8), torch.ones(x.shape[0], dtype=torch.float32)

    def unpermute(**kwargs: object):
        calls.append({"operator": "unpermute", **kwargs})
        return kwargs["permuted_tokens"]

    monkeypatch.setattr(op.torch_npu, "npu_moe_init_routing_v2", route, raising=False)
    monkeypatch.setattr(op.torch_npu, "npu_grouped_matmul", grouped_matmul, raising=False)
    monkeypatch.setattr(
        op.torch_npu,
        "npu_swiglu",
        lambda x, dim: x.chunk(2, dim=dim)[0],
        raising=False,
    )
    monkeypatch.setattr(op.torch_npu, "npu_dynamic_quant", dynamic_quant, raising=False)
    monkeypatch.setattr(op.torch_npu, "npu_moe_token_unpermute", unpermute, raising=False)

    w13 = [torch.ones((2, 4), dtype=torch.int8) for _ in range(2)]
    w2 = [torch.ones((2, 2), dtype=torch.int8) for _ in range(2)]
    w13_scale = [torch.ones(4, dtype=torch.float32) for _ in range(2)]
    w2_scale = [torch.ones(2, dtype=torch.float32) for _ in range(2)]
    output = op.run_ascend_fused_moe(
        torch.ones((2, 2), dtype=torch.bfloat16),
        torch.tensor([[0], [1]], dtype=torch.int32),
        torch.ones((2, 1), dtype=torch.float32),
        w13=w13,
        w2=w2,
        compute_type=op.AscendFusedComputeType.DYNAMIC_W8A8,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
    )

    route_call = calls[0]
    first_gmm, second_gmm = (call for call in calls if call["operator"] == "gmm")
    assert route_call["quant_mode"] == 1
    assert first_gmm["weight"] is w13
    assert all(torch.equal(a, b) for a, b in zip(first_gmm["scale"], w13_scale, strict=True))
    assert second_gmm["weight"] is w2
    assert all(torch.equal(a, b) for a, b in zip(second_gmm["scale"], w2_scale, strict=True))
    assert all(scale.dtype is torch.bfloat16 for scale in first_gmm["scale"] + second_gmm["scale"])
    assert all(scale.dtype is torch.float32 for scale in w13_scale + w2_scale)
    assert first_gmm["output_dtype"] is torch.bfloat16
    assert second_gmm["output_dtype"] is torch.bfloat16
    assert output.shape == (2, 2)


def test_dynamic_w8a8_requires_both_scale_lists() -> None:
    with pytest.raises(ValueError, match="requires w13_scale and w2_scale"):
        op.run_ascend_fused_moe(
            torch.ones((1, 2), dtype=torch.bfloat16),
            torch.zeros((1, 1), dtype=torch.int32),
            torch.ones((1, 1), dtype=torch.float32),
            w13=[torch.ones((2, 4), dtype=torch.int8)],
            w2=[torch.ones((2, 2), dtype=torch.int8)],
            compute_type=op.AscendFusedComputeType.DYNAMIC_W8A8,
        )


@pytest.mark.parametrize("quantized", [False, True])
def test_clipped_swiglu_routes_after_down_projection(monkeypatch, quantized):
    calls = []
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    ids = torch.tensor([[1], [0]], dtype=torch.int32)
    routing = torch.tensor([[0.1234567], [0.7654321]])
    gate_up = torch.tensor([[12.0, -12.0, 14.0, -14.0], [2.0, 3.0, 4.0, 5.0]], dtype=torch.bfloat16)
    gate, up = gate_up.chunk(2, dim=-1)
    activated = torch.nn.functional.silu(gate.clamp(max=10)) * up.clamp(-10, 10)
    down = torch.tensor([[3.0, 7.0], [11.0, 13.0]], dtype=torch.bfloat16)

    def route(*args, **kwargs):
        x = hidden.flip(0).to(torch.int8) if quantized else hidden.flip(0)
        return x, torch.tensor([1, 0], dtype=torch.int32), torch.tensor([1, 1]), torch.ones(2)

    def gmm(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return [gate_up.clone()]
        expected = activated.to(torch.int8) if quantized else activated
        torch.testing.assert_close(kwargs["x"][0], expected, rtol=0, atol=0)
        return [down]

    def swiglu(x, *, dim):
        gate, up = x.chunk(2, dim=dim)
        assert x.dtype is torch.bfloat16
        assert gate.max() <= 10 and up.abs().max() <= 10
        return torch.nn.functional.silu(gate) * up

    def quantize(x, **kwargs):
        torch.testing.assert_close(x, activated, rtol=0, atol=0)
        return x.to(torch.int8), torch.ones(2)

    def restore(**kwargs):
        torch.testing.assert_close(kwargs["probs"], routing.bfloat16(), rtol=0, atol=0)
        return kwargs["permuted_tokens"].flip(0) * kwargs["probs"]

    monkeypatch.setattr(op.torch_npu, "npu_moe_init_routing_v2", route, raising=False)
    monkeypatch.setattr(op.torch_npu, "npu_grouped_matmul", gmm, raising=False)
    monkeypatch.setattr(op.torch_npu, "npu_swiglu", swiglu, raising=False)
    monkeypatch.setattr(op.torch_npu, "npu_dynamic_quant", quantize, raising=False)
    monkeypatch.setattr(op.torch_npu, "npu_moe_token_unpermute", restore, raising=False)
    options = dict(
        compute_type=(
            op.AscendFusedComputeType.DYNAMIC_W8A8
            if quantized
            else op.AscendFusedComputeType.UNQUANTIZED
        ),
        swiglu_limit=10.0,
    )
    if quantized:
        options.update(w13_scale=[torch.ones(4)] * 2, w2_scale=[torch.ones(2)] * 2)
    result = op.run_ascend_fused_moe(
        hidden,
        ids,
        routing,
        w13=[torch.ones(2, 4).bfloat16()] * 2,
        w2=[torch.ones(2, 2).bfloat16()] * 2,
        **options,
    )
    torch.testing.assert_close(result, down.flip(0) * routing.bfloat16(), rtol=0, atol=0)
