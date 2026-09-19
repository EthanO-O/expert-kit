"""Regression tests for vLLM model-level routed-expert weight loading."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
import torch
from torch import nn

pytest.importorskip("vllm")

from vllm.config import CUDAGraphMode
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.expert_map_manager import ExpertMapManager
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter,
)
from vllm.model_executor.models.deepseek_v2 import DeepseekV2Model
from vllm.model_executor.models.utils import AutoWeightsLoader, is_pp_missing_parameter

from expertkit_vllm.experts import remote_moe, remote_routed_experts
from expertkit_vllm.experts.remote_routed_experts import RemoteRoutedExperts


class _MoELayer(nn.Module):
    def __init__(self, experts: nn.Module, gate: nn.Module, shared_experts: nn.Module) -> None:
        super().__init__()
        self.gate = gate
        self.shared_experts = shared_experts
        self.experts = experts


class _DecoderLayer(nn.Module):
    def __init__(self, mlp: nn.Module) -> None:
        super().__init__()
        self.mlp = mlp


class _DeepseekLoaderHarness(nn.Module):
    def __init__(self, remote_experts: nn.Module, gate: nn.Module, shared: nn.Module) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_DecoderLayer(nn.Identity()), _DecoderLayer(_MoELayer(remote_experts, gate, shared))]
        )
        self.config = SimpleNamespace(n_routed_experts=2, n_shared_experts=1)
        self.num_redundant_experts = 0
        self.use_mha = False


class _AutoLoaderHarness(nn.Module):
    def __init__(self, remote_experts: nn.Module, gate: nn.Module, shared: nn.Module) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [_DecoderLayer(nn.Identity()), _DecoderLayer(_MoELayer(remote_experts, gate, shared))]
        )


def _make_remote_runner(
    monkeypatch, quant_config=None
) -> tuple[remote_moe.RemoteMoERunner, nn.Linear, nn.Linear]:
    compilation = SimpleNamespace(
        static_forward_context={},
        static_all_moe_layers=[],
        splitting_ops=[],
        cudagraph_mode=CUDAGraphMode.NONE,
    )
    vllm_config = SimpleNamespace(
        compilation_config=compilation,
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(num_hidden_layers=27)),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            enable_expert_parallel=False,
            enable_eplb=False,
        ),
    )
    monkeypatch.setattr(remote_moe, "get_current_vllm_config", lambda: vllm_config)
    monkeypatch.setattr(
        remote_routed_experts,
        "get_current_vllm_config",
        lambda: vllm_config,
    )
    monkeypatch.setenv("EK_ADDR", "127.0.0.1:50050")
    monkeypatch.setenv("EK_INSTANCE_ID", "7")
    gate = nn.Linear(2, 2, bias=False)
    shared = nn.Linear(2, 2, bias=False)
    moe_config = cast(
        FusedMoEConfig,
        SimpleNamespace(
            num_experts=2,
            num_logical_experts=2,
            experts_per_token=1,
            hidden_dim=2,
            activation=MoEActivation.SILU,
            tp_size=1,
            pcp_size=1,
            is_sequence_parallel=False,
            has_bias=False,
            moe_parallel_config=SimpleNamespace(
                use_ep=False,
                enable_eplb=False,
            ),
        ),
    )
    expert_map_manager = cast(
        ExpertMapManager,
        SimpleNamespace(
            num_fused_shared_experts=0,
            placement_strategy="linear",
        ),
    )
    routed_experts = RemoteRoutedExperts(
        "layers.1.mlp.experts",
        torch.float32,
        moe_config,
        quant_config,
        expert_map_manager,
    )
    runner = remote_moe.RemoteMoERunner(
        layer_name="layers.1.mlp.experts",
        moe_config=moe_config,
        router=cast(FusedMoERouter, nn.Identity()),
        routed_experts=routed_experts,
        gate=gate,
        shared_experts=shared,
        routed_scaling_factor=1.0,
    )
    return runner, gate, shared


def test_deepseek_loader_discards_only_remote_routed_expert_weights(monkeypatch) -> None:
    runner, gate, shared = _make_remote_runner(monkeypatch)
    model = _DeepseekLoaderHarness(runner, gate, shared)

    routed_param = "layers.1.mlp.experts.routed_experts.w2_weight"
    assert not is_pp_missing_parameter(routed_param, model)
    assert not is_pp_missing_parameter("layers.1.mlp.gate.weight", model)
    assert not is_pp_missing_parameter("layers.1.mlp.shared_experts.weight", model)
    assert dict(model.named_parameters())[routed_param].numel() == 0

    gate_weight = torch.full_like(gate.weight, 3.0)
    loaded = DeepseekV2Model.load_weights(
        cast(DeepseekV2Model, model),
        [
            ("layers.1.mlp.experts.0.down_proj.weight", torch.zeros(2, 2)),
            ("layers.1.mlp.gate.weight", gate_weight),
        ],
    )

    assert routed_param in loaded
    assert "layers.1.mlp.gate.weight" in loaded
    assert dict(model.named_parameters())[routed_param].numel() == 0
    torch.testing.assert_close(gate.weight, gate_weight)


def test_runner_discards_auto_loader_expert_subtree(monkeypatch) -> None:
    runner, _, _ = _make_remote_runner(monkeypatch)
    weights = [
        ("0.gate_proj.weight", torch.ones(2, 2)),
        ("0.down_proj.weight", torch.ones(2, 2)),
        ("0.up_proj.weight", torch.ones(2, 2)),
    ]
    consumed: list[str] = []

    def tracked_weights():
        for name, weight in weights:
            consumed.append(name)
            yield name, weight

    assert runner.load_weights(tracked_weights()) == {
        "routed_experts.w13_weight",
        "routed_experts.w2_weight",
    }
    assert consumed == [name for name, _ in weights]
    assert all(parameter.numel() == 0 for parameter in runner.routed_experts.parameters())


def test_auto_loader_tracks_remote_expert_sinks_as_initialized(monkeypatch) -> None:
    runner, gate, shared = _make_remote_runner(monkeypatch)
    model = _AutoLoaderHarness(runner, gate, shared)
    weights = [
        ("model.layers.1.mlp.gate.weight", torch.ones_like(gate.weight)),
        (
            "model.layers.1.mlp.shared_experts.weight",
            torch.ones_like(shared.weight),
        ),
        (
            "model.layers.1.mlp.experts.0.gate_proj.weight",
            torch.ones(2, 2),
        ),
        (
            "model.layers.1.mlp.experts.0.down_proj.weight",
            torch.ones(2, 2),
        ),
        (
            "model.layers.1.mlp.experts.0.up_proj.weight",
            torch.ones(2, 2),
        ),
    ]

    loaded = AutoWeightsLoader(model).load_weights(iter(weights))

    assert loaded == {name for name, _ in model.named_parameters()}
    assert {
        "model.layers.1.mlp.experts.routed_experts.w13_weight",
        "model.layers.1.mlp.experts.routed_experts.w2_weight",
    } <= loaded


def test_layer_name_encoding_is_traceable_by_torch_compile() -> None:
    def encode_during_forward(value: torch.Tensor) -> torch.Tensor:
        remote_moe._encode_layer_name("layers.1.mlp.experts")
        return value + 1

    compiled = torch.compile(encode_during_forward, backend="eager", fullgraph=True)

    torch.testing.assert_close(compiled(torch.zeros(1)), torch.ones(1))


def test_client_creation_uses_model_metadata_captured_during_layer_init(monkeypatch) -> None:
    runner, _, _ = _make_remote_runner(monkeypatch)
    created: list[dict[str, object]] = []

    class FakeClient:
        def __init__(self, endpoint: str, **kwargs) -> None:
            created.append({"endpoint": endpoint, **kwargs})

        def start(self, *, timeout_seconds: float) -> None:
            created[-1]["timeout_seconds"] = timeout_seconds

        def close(self) -> None:
            return None

    def unavailable_config():
        raise AssertionError("vLLM config is unavailable during custom-op execution")

    remote_moe.close_clients()
    monkeypatch.setattr(remote_moe, "BlockingRoutedMoEClient", FakeClient)
    monkeypatch.setattr(remote_moe, "get_current_vllm_config", unavailable_config)

    client = remote_moe._client_for(runner, torch.zeros(1, 2))

    assert isinstance(client, FakeClient)
    assert created[0]["num_layers"] == 27
    remote_moe.close_clients()


@pytest.fixture
def v4_model(monkeypatch):
    """Provide a small module tree and single-rank distributed boundaries."""
    pytest.importorskip("vllm_ascend")
    from vllm_ascend.models import deepseek_v4
    from vllm_ascend.quantization.modelslim_config import AscendModelSlimConfig

    monkeypatch.setattr(
        deepseek_v4, "get_ascend_config", lambda: SimpleNamespace(mix_placement=False)
    )
    monkeypatch.setattr(deepseek_v4, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(deepseek_v4, "get_tensor_model_parallel_world_size", lambda: 1)
    runner, gate, _ = _make_remote_runner(monkeypatch, AscendModelSlimConfig())
    shared = nn.Module()
    shared.down_proj = nn.Linear(2, 2, bias=False)
    layer = nn.Module()
    layer.self_attn = nn.Linear(2, 2, bias=False)
    for module in (layer.self_attn, shared.down_proj):
        for suffix in ("weight_scale", "weight_offset"):
            module.register_parameter(suffix, nn.Parameter(torch.zeros(2, 1), requires_grad=False))
    layer.mlp = nn.Module()
    layer.mlp.gate = gate
    layer.mlp.shared_experts = shared
    layer.mlp.experts = runner
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([layer])
    model.config = SimpleNamespace(n_routed_experts=2, n_shared_experts=1, num_attention_heads=1)
    model.num_redundant_experts = 0
    return model


def _load(model, weights):
    from vllm_ascend.models import deepseek_v4

    return deepseek_v4.AscendDeepseekV4ForCausalLM.load_weights(model, iter(weights))


@pytest.mark.parametrize("projection,target", [("w1", "w13"), ("w3", "w13"), ("w2", "w2")])
@pytest.mark.parametrize("suffix", ["weight_offset", "weight_scale", "weight"])
def test_v4_loader_consumes_modelslim_routed_parameters(v4_model, projection, target, suffix):
    source = f"layers.0.ffn.experts.0.{projection}.{suffix}"
    destination = f"model.layers.0.mlp.experts.routed_experts.{target}_{suffix}"
    dtype = torch.int8 if suffix == "weight" else torch.float32
    loaded = _load(v4_model, [(source, torch.ones(2, 2, dtype=dtype))])
    assert loaded == {destination}
    parameters = dict(v4_model.named_parameters())
    assert parameters[destination].untyped_storage().nbytes() == 0


def test_v4_loader_preserves_all_local_weights(v4_model):
    local = {
        "layers.0.attn.weight": "model.layers.0.self_attn.weight",
        "layers.0.ffn.gate.weight": "model.layers.0.mlp.gate.weight",
        "layers.0.ffn.shared_experts.w2.weight": (
            "model.layers.0.mlp.shared_experts.down_proj.weight"
        ),
    }
    weights = [(source, torch.full((2, 2), float(index + 3))) for index, source in enumerate(local)]
    assert _load(v4_model, weights) == set(local.values())
    parameters = dict(v4_model.named_parameters())
    for source, value in weights:
        torch.testing.assert_close(parameters[local[source]], value)


@pytest.mark.parametrize("suffix", ["weight_scale", "weight_offset"])
def test_v4_loader_preserves_local_quantization_parameters(v4_model, suffix):
    weights = [
        (f"layers.0.attn.{suffix}", torch.full((2, 1), 2.0)),
        (f"layers.0.ffn.shared_experts.w2.{suffix}", torch.full((2, 1), 3.0)),
    ]
    names = [
        f"model.layers.0.self_attn.{suffix}",
        f"model.layers.0.mlp.shared_experts.down_proj.{suffix}",
    ]
    assert _load(v4_model, weights) == set(names)
    parameters = dict(v4_model.named_parameters())
    for name, (_, value) in zip(names, weights, strict=True):
        torch.testing.assert_close(parameters[name], value)


@pytest.mark.parametrize(
    "name",
    [
        "layers.0.ffn.experts.0.w1.unknown",
        "layers.0.attn.missing",
        "layers.0.ffn.shared_experts.w2.missing",
    ],
)
def test_v4_loader_rejects_unknown_parameters(v4_model, name):
    with pytest.raises(KeyError):
        _load(v4_model, [(name, torch.zeros(2, 2))])


def test_modelslim_auto_loader_reports_auxiliary_parameters(monkeypatch):
    pytest.importorskip("vllm_ascend")
    from vllm_ascend.quantization.modelslim_config import AscendModelSlimConfig

    runner, _, _ = _make_remote_runner(monkeypatch, AscendModelSlimConfig())
    weights = [
        (f"0.{projection}.{suffix}", torch.ones(2, 2))
        for projection in ("gate_proj", "up_proj", "down_proj")
        for suffix in ("weight", "weight_scale", "weight_offset")
    ]
    loaded = runner.load_weights(iter(weights))
    expected = {
        f"routed_experts.{projection}_{suffix}"
        for projection in ("w13", "w2")
        for suffix in ("weight", "weight_scale", "weight_offset")
    }
    assert loaded == expected
    assert all(
        parameter.untyped_storage().nbytes() == 0
        for parameter in runner.routed_experts.parameters()
    )


def test_remote_experts_reject_other_quantization(monkeypatch):
    with pytest.raises(ValueError, match="quantization"):
        _make_remote_runner(monkeypatch, SimpleNamespace())


@pytest.mark.parametrize(
    "name", ["0.gate_proj.unknown", "shared_experts.weight", "0.down_proj.weight_scale"]
)
def test_unquantized_auto_loader_rejects_unrecognized_parameters(monkeypatch, name):
    runner, _, _ = _make_remote_runner(monkeypatch)
    with pytest.raises(ValueError, match="unsupported remote expert checkpoint parameter"):
        runner.load_weights(iter([(name, torch.zeros(2, 2))]))


def test_auto_loader_reports_only_consumed_parameters(monkeypatch):
    runner, _, _ = _make_remote_runner(monkeypatch)
    assert runner.load_weights(iter([])) == set()
    assert runner.load_weights(iter([("0.down_proj.weight", torch.ones(2, 2))])) == {
        "routed_experts.w2_weight"
    }
