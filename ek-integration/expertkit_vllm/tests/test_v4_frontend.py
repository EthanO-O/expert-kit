"""Preserve V4 router inputs and exact local selection at the remote boundary."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from expertkit_vllm.experts import remote_routed_experts, selector
from expertkit_vllm.experts.remote_routed_experts import RemoteRoutedExperts


@pytest.fixture
def v4_placeholder(monkeypatch):
    """Use fake native configuration at the external Ascend dependency boundary."""
    module_name = "vllm_ascend.quantization.configs.modelslim_config"
    module = ModuleType(module_name)
    native_type = type("AscendModelSlimConfig", (), {"__module__": module_name})
    module.AscendModelSlimConfig = native_type
    monkeypatch.setitem(sys.modules, module_name, module)
    ascend_module = ModuleType("vllm_ascend.ascend_config")
    ascend_module.get_ascend_config = lambda: SimpleNamespace(mix_placement=False)
    monkeypatch.setitem(sys.modules, "vllm_ascend.ascend_config", ascend_module)
    hf = SimpleNamespace(
        model_type="deepseek_v4",
        num_hidden_layers=43,
        hidden_size=4096,
        moe_intermediate_size=2048,
        n_routed_experts=256,
        n_shared_experts=1,
        num_experts_per_tok=6,
        num_hash_layers=3,
        scoring_func="sqrtsoftplus",
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
        hidden_act="silu",
        vocab_size=129280,
    )
    native = native_type()
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=hf, quantization="ascend"),
        quant_config=native,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            enable_expert_parallel=False,
            enable_eplb=False,
        ),
    )
    monkeypatch.setattr(remote_routed_experts, "get_current_vllm_config", lambda: config)
    moe = SimpleNamespace(
        num_experts=256,
        num_logical_experts=256,
        experts_per_token=6,
        hidden_dim=4096,
        intermediate_size=2048,
        activation=MoEActivation.SILU,
        pcp_size=1,
        is_sequence_parallel=False,
        has_bias=False,
    )
    manager = SimpleNamespace(num_fused_shared_experts=0)
    table = torch.zeros((129280, 6), dtype=torch.int32)
    kwargs = dict(
        renormalize=True,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
        tid2eid=table,
    )

    def construct(**updates):
        return RemoteRoutedExperts(
            "model.layers.0.mlp.experts",
            torch.bfloat16,
            moe,
            native,
            manager,
            **{**kwargs, **updates},
        )

    return construct, config, moe, table


def test_v4_placeholder_admits_only_exact_recipe_and_preserves_native_table(v4_placeholder):
    construct, _, _, table = v4_placeholder
    experts = construct()
    assert experts.is_deepseek_v4
    assert experts.tid2eid is table
    assert all(p.numel() == 0 for name, p in experts.named_parameters() if name != "tid2eid")
    assert set(experts._parameters) == {
        f"{projection}_{suffix}"
        for projection in ("w13", "w2")
        for suffix in ("weight", "weight_scale", "weight_offset")
    }


@pytest.mark.parametrize(
    "target,name,value",
    [
        ("hf", "hidden_size", 2048),
        ("hf", "num_hash_layers", 2),
        ("hf", "swiglu_limit", 9.0),
        ("parallel", "data_parallel_size", 2),
        ("parallel", "pipeline_parallel_size", 2),
        ("model", "quantization", None),
        ("config", "quant_config", None),
        ("moe", "num_experts", 257),
    ],
)
def test_v4_placeholder_rejects_unqualified_configuration(v4_placeholder, target, name, value):
    construct, config, moe, _ = v4_placeholder
    objects = {
        "hf": config.model_config.hf_text_config,
        "parallel": config.parallel_config,
        "model": config.model_config,
        "config": config,
        "moe": moe,
    }
    setattr(objects[target], name, value)
    with pytest.raises(ValueError, match="V4"):
        construct()


@pytest.mark.parametrize(
    "updates",
    [
        {"swiglu_limit": None},
        {"swiglu_alpha": 1.0},
        {"renormalize": False},
        {"scoring_func": "softmax"},
        {"routed_scaling_factor": 1.0},
    ],
)
def test_v4_placeholder_rejects_wrong_expression(v4_placeholder, updates):
    with pytest.raises(ValueError):
        v4_placeholder[0](**updates)


@pytest.mark.parametrize("hashed", [False, True])
def test_qualified_v4_uses_native_hash_operator_on_local_rows(monkeypatch, hashed):
    calls = []

    def unavailable_legacy_selector():
        raise ImportError("The fixed Ascend version has no legacy selector module")

    monkeypatch.setattr(selector, "_load_ascend_select_experts", unavailable_legacy_selector)
    expected_weights = torch.ones(2, 6, dtype=torch.float32)
    expected_ids = torch.zeros(2, 6, dtype=torch.int32)

    def select(**kwargs):
        calls.append(kwargs)
        return expected_weights, expected_ids, None

    monkeypatch.setattr(torch.ops._C_ascend, "moe_gating_top_k_hash", select, raising=False)
    options = SimpleNamespace(
        is_deepseek_v4=True,
        moe_config=SimpleNamespace(),
        scoring_func="sqrtsoftplus",
        renormalize=True,
        top_k=6,
        topk_group=None,
        num_expert_group=None,
        routed_scaling_factor=1.5,
        e_score_correction_bias=None if hashed else torch.ones(256, dtype=torch.bfloat16),
    )
    table = torch.zeros(3, 6, dtype=torch.int32) if hashed else None

    class NativeRouter:
        __module__ = "vllm_ascend.ops.fused_moe.router.fused_topk_router"
        bias_vl = None

    monkeypatch.setattr(selector, "current_platform", SimpleNamespace(device_type="npu"))
    adapter = selector.create_expert_selector(NativeRouter(), options, tid2eid=table)
    assert isinstance(adapter, selector.AscendExpertSelector)
    logits = torch.zeros(2, 256, dtype=torch.float32)
    weights, ids = adapter.select_experts(
        torch.zeros(2, 4096), logits, input_ids=torch.tensor([-1, 2]) if hashed else None
    )
    assert weights is expected_weights and ids is expected_ids
    assert len(calls) == 1 and calls[0]["x"] is logits
    assert calls[0]["tid2eid"] is table
    if hashed:
        assert torch.equal(calls[0]["input_ids"], torch.tensor([0, 2]))
    else:
        assert calls[0]["input_ids"] is None
        assert calls[0]["bias"].dtype == logits.dtype
    assert calls[0]["routed_scaling_factor"] == 1.5
    assert calls[0]["renorm"] == 0 and calls[0]["norm_type"] == 2


def _native_runner(monkeypatch, v4_placeholder, *, layer=0, table=None, bias=None, **overrides):
    from vllm.config import CUDAGraphMode

    from expertkit_vllm.experts import remote_moe

    construct, config, moe, source_table = v4_placeholder
    table = source_table if layer < 3 and table is None else table
    bias = torch.zeros(256) if layer >= 3 and bias is None else bias
    experts = construct(tid2eid=table, e_score_correction_bias=bias)
    config.model_config.enforce_eager = True
    config.compilation_config = SimpleNamespace(
        static_forward_context={},
        static_all_moe_layers=[],
        splitting_ops=[],
        cudagraph_mode=CUDAGraphMode.NONE,
    )
    gate = torch.nn.Linear(4096, 256, bias=False, dtype=torch.bfloat16)
    method = type("AscendUnquantizedLinearMethod", (), {"__module__": "vllm_ascend.ops.linear"})
    gate.quant_method = method()
    gate.tid2eid = table
    router = type(
        "NativeRouter",
        (),
        {"__module__": "vllm_ascend.ops.fused_moe.router.fused_topk_router", "bias_vl": None},
    )()
    monkeypatch.setattr(remote_moe, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(selector, "_load_ascend_select_experts", lambda: None)
    monkeypatch.setattr(selector, "current_platform", SimpleNamespace(device_type="npu"))
    monkeypatch.setenv("EK_ADDR", "127.0.0.1:50050")
    monkeypatch.delenv("EK_ASCEND_DECODE_GRAPH_CONFIG", raising=False)
    monkeypatch.delenv("EK_TRACE_ENDPOINT", raising=False)
    runner = remote_moe.RemoteMoERunner(
        f"model.layers.{layer}.mlp.experts",
        moe,
        router,
        experts,
        **{"gate": gate, "shared_experts": torch.nn.Identity(), **overrides},
    )
    return runner, gate


def test_native_runner_keeps_gate_owned_hash_table_and_single_shared_module(
    monkeypatch, v4_placeholder
):
    runner, gate = _native_runner(monkeypatch, v4_placeholder)
    assert runner.expert_selector._tid2eid is gate.tid2eid
    assert runner.routed_experts.tid2eid is gate.tid2eid
    assert runner.routed_scaling_factor == 1.0
    assert gate.precast_fp32_weight
    assert "tid2eid" not in runner.routed_experts._parameters


@pytest.mark.parametrize(
    "overrides",
    [
        {"tid2eid": torch.empty(1, 6, dtype=torch.int32)},
        {"routed_scaling_factor": 1.5},
        {"shared_experts": None},
        {"gate": None},
    ],
)
def test_native_runner_rejects_table_conflicts_and_extra_output_scale(
    monkeypatch, v4_placeholder, overrides
):
    with pytest.raises(ValueError, match="V4"):
        _native_runner(monkeypatch, v4_placeholder, **overrides)


def test_native_runner_nonhash_requires_loaded_bias_and_no_hash_table(monkeypatch, v4_placeholder):
    runner, _ = _native_runner(monkeypatch, v4_placeholder, layer=3)
    assert runner.expert_selector._tid2eid is None
    assert runner.routed_experts.e_score_correction_bias is not None
    with pytest.raises(ValueError, match="nonhash"):
        _native_runner(monkeypatch, v4_placeholder, layer=3, table=v4_placeholder[3])
