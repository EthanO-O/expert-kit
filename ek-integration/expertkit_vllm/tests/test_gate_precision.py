"""Keep the native internal gate precision identical in ordinary and graph execution."""

from types import SimpleNamespace

import pytest
import torch
from vllm.config import CUDAGraphMode

from expertkit_vllm.experts import remote_moe, selector


class _Router:
    """Capture gate logits at the native selector boundary."""

    def __init__(self):
        self.logits = []

    def select_experts(self, hidden_states, router_logits, **kwargs):
        self.logits.append(router_logits.detach().clone())
        values, ids = router_logits.float().softmax(-1).topk(2)
        return values / values.sum(-1, keepdim=True), ids.to(torch.int32)


def _runner(monkeypatch, gate):
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            enforce_eager=True, hf_text_config=SimpleNamespace(num_hidden_layers=1)
        ),
        compilation_config=SimpleNamespace(
            static_forward_context={},
            static_all_moe_layers=[],
            splitting_ops=[],
            cudagraph_mode=CUDAGraphMode.NONE,
        ),
    )
    monkeypatch.setattr(remote_moe, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(selector, "current_platform", SimpleNamespace(device_type="cpu"))
    monkeypatch.setenv("EK_ADDR", "127.0.0.1:50050")
    monkeypatch.delenv("EK_ASCEND_DECODE_GRAPH_CONFIG", raising=False)
    monkeypatch.delenv("EK_TRACE_ENDPOINT", raising=False)
    router = _Router()
    runner = remote_moe.RemoteMoERunner(
        "model.layers.0.mlp.experts",
        SimpleNamespace(num_logical_experts=3, experts_per_token=2, hidden_dim=4),
        router,
        SimpleNamespace(),
        gate=gate,
    )
    monkeypatch.setattr(
        remote_moe,
        "active_graph_session",
        lambda: SimpleNamespace(
            exchange=lambda layer_id, hidden_states, *args: torch.zeros_like(hidden_states)
        ),
    )
    return runner, router


def _gate():
    gate = torch.nn.Linear(4, 3, bias=False, dtype=torch.bfloat16)
    gate.weight.data.copy_(
        torch.tensor(
            [
                [0.12, 0.37, -0.81, 0.19],
                [0.21, -0.73, 0.11, 0.61],
                [-0.31, 0.24, 0.42, -0.53],
            ],
            dtype=torch.bfloat16,
        )
    )
    return gate


def test_internal_gate_requests_precast_before_weight_loading(monkeypatch):
    gate = _gate()
    _runner(monkeypatch, gate)
    assert gate.precast_fp32_weight is True
    assert not hasattr(gate, "weight_fp32")


def test_existing_fp32_gate_parameter_is_retained(monkeypatch):
    gate = _gate()
    gate.weight_fp32 = torch.nn.Parameter(gate.weight.float(), requires_grad=False)
    pointer = gate.weight_fp32.data_ptr()
    _runner(monkeypatch, gate)
    assert gate.weight_fp32.data_ptr() == pointer
    assert not hasattr(gate, "precast_fp32_weight")


@pytest.mark.parametrize("precast", [False, True])
def test_ordinary_and_graph_gate_logits_share_native_precision(monkeypatch, precast):
    gate = _gate()
    runner, router = _runner(monkeypatch, gate)
    if precast:
        gate.weight_fp32 = torch.nn.Parameter(gate.weight.float(), requires_grad=False)
    hidden = torch.tensor([[0.31, -0.17, 0.43, 0.71]], dtype=torch.bfloat16)
    expected = torch.nn.functional.linear(
        hidden.float() if precast else hidden,
        gate.weight_fp32 if precast else gate.weight,
    )
    for graph in (False, True):
        runner.graph_config = object() if graph else None
        runner._forward_impl(hidden, hidden, None)
        assert router.logits[-1].dtype == expected.dtype
        torch.testing.assert_close(router.logits[-1], expected, rtol=0, atol=0)


def test_ascend_unquantized_gate_requires_post_load_parameter_before_graph(monkeypatch):
    gate = _gate()
    method = type("AscendUnquantizedLinearMethod", (), {"__module__": "vllm_ascend.ops.linear"})
    gate.quant_method = method()
    runner, router = _runner(monkeypatch, gate)
    runner.graph_config = object()
    hidden = torch.ones((1, 4), dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="post-load FP32 weights before capture"):
        runner._forward_impl(hidden, hidden, None)
    assert router.logits == []


def test_v4_internal_gate_preserves_original_fp32_normalized_input(monkeypatch):
    gate = _gate()
    gate.weight_fp32 = torch.nn.Parameter(gate.weight.float(), requires_grad=False)
    runner, router = _runner(monkeypatch, gate)
    runner._is_deepseek_v4 = True
    runner._requires_fp32_gate = True
    original = torch.tensor([[0.31003, -0.17003, 0.43003, 0.71003]], dtype=torch.float32)
    hidden = original.to(torch.bfloat16)
    assert not torch.equal(original, hidden.float())
    for graph in (False, True):
        runner.graph_config = object() if graph else None
        runner._forward_impl(hidden, original, None)
        torch.testing.assert_close(router.logits[-1], original @ gate.weight_fp32.T, rtol=0, atol=0)
    for invalid in (hidden, original[:, :3], torch.empty(1, 3)):
        with pytest.raises(ValueError, match="original FP32"):
            runner._forward_impl(hidden, invalid, None)
    del gate.weight_fp32
    runner.graph_config = None
    with pytest.raises(RuntimeError, match="post-load FP32"):
        runner._forward_impl(hidden, original, None)
