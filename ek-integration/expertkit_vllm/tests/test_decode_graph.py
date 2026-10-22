"""Exercise native wrapper phase coordination with external CPU boundaries."""

import atexit
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from expertkit_transport.transports.ascend_graph.config import AscendGraphConfig
from vllm.config import CUDAGraphMode

from expertkit_vllm import decode_graph as graph


@pytest.fixture
def graph_runtime(monkeypatch, request):
    events = []
    context = SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.FULL, batch_descriptor=None)

    class Stream:
        def synchronize(self):
            events.append("synchronize")

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            current_device=lambda: 0,
            current_stream=Stream,
        ),
        raising=False,
    )

    class Control:
        def __init__(self, config):
            pass

        def begin(self, phase, capacity=0):
            events.append(("begin", phase, capacity))

        def finish(self):
            events.append("finish")

        def poison(self):
            events.append("poison")

        def close(self):
            events.append("control-close")

    class Data:
        def __init__(self, config, **options):
            self.profiles = {}

        def buffers(self, capacity):
            if capacity not in self.profiles:
                self.profiles[capacity] = SimpleNamespace(output=torch.zeros(capacity, 2048))
                events.append(("allocate", capacity))
            return self.profiles[capacity]

        def receive_inputs(self, buffers):
            events.append("warmup-input")

        def combine(self, buffers):
            events.append("warmup-output")

        def exchange(self, hidden, ids, routing):
            assert ids.dtype == torch.int32 and routing.dtype == torch.float32
            events.append("layer-exchange")
            return torch.zeros_like(hidden)

        def quarantine(self):
            events.append("quarantine")

        def close(self):
            events.append("data-close")

    monkeypatch.setattr(graph, "GraphStepControl", Control)
    monkeypatch.setattr(graph, "HcclGraphSession", Data)
    # The compatibility guard is tested separately against exact runtime metadata.
    monkeypatch.setattr(graph, "validate_graph_model", lambda *_: None)
    import vllm.forward_context

    monkeypatch.setattr(vllm.forward_context, "get_forward_context", lambda: context)

    class Wrapper:
        def __init__(self):
            self.runtime_mode = CUDAGraphMode.FULL
            self.vllm_config = object()
            self.concrete_aclgraph_entries = {}
            self.fail = False

        def __call__(self):
            if context.cudagraph_runtime_mode == CUDAGraphMode.NONE:
                events.append(("eager", graph.is_graph_profile()))
                return "eager"
            descriptor = context.batch_descriptor
            if descriptor not in self.concrete_aclgraph_entries:
                events.append("native-capture")
                session = graph.active_graph_session()
                assert session is not None
                for layer in config.layer_ids:
                    session.exchange(
                        layer,
                        torch.zeros(descriptor.num_tokens, 2048, dtype=torch.bfloat16),
                        torch.zeros(descriptor.num_tokens, config.model.top_k, dtype=torch.int32),
                        torch.full(
                            (descriptor.num_tokens, config.model.top_k),
                            0.1234567,
                            dtype=torch.float32,
                        ),
                    )
                self.concrete_aclgraph_entries[descriptor] = SimpleNamespace(aclgraph=object())
            else:
                events.append("native-replay")
            if self.fail:
                raise RuntimeError("native failure")
            return "result"

    class Runner:
        def __init__(self, wrapper):
            self.model = wrapper
            self.device = torch.device("cpu")
            self.fail_update = False

        def _model_forward(self):
            result = self.model()
            events.append("native-attention-update")
            if self.fail_update:
                raise RuntimeError("attention update failed")
            return result

        def profile_run(self):
            assert graph.is_graph_profile()
            return self.model()

    for name, attribute, value in (
        ("vllm_ascend.compilation.acl_graph", "ACLGraphWrapper", Wrapper),
        ("vllm_ascend.worker.model_runner_v1", "NPUModelRunner", Runner),
    ):
        module = ModuleType(name)
        setattr(module, attribute, value)
        monkeypatch.setitem(sys.modules, name, module)
    config = AscendGraphConfig(
        session_id="frontend-test-session-001",
        instance_id=1,
        worker_id="worker",
        weight_version="revision",
        control_endpoint="127.0.0.1:12345",
        rendezvous_host="127.0.0.1",
        rendezvous_port=12346,
        layer_ids=tuple(range(1, 27)),
        capture_sizes=(1, 4),
        timeout_seconds=2.0,
    )
    if getattr(request, "param", "decode") == "full":
        config = _full_qwen_config(config)
    graph.install_decode_graph_wrapper(config)
    wrapper = Wrapper()
    try:
        yield events, context, Runner(wrapper), wrapper
    finally:
        session = wrapper.__dict__.get("_expertkit_decode_session")
        if session is not None:
            session.close()
            atexit.unregister(session.close)


def descriptor(*, uniform=True, num_reqs=1, num_tokens=1, has_lora=False):
    """Return the pinned vLLM immutable batch descriptor contract."""
    from vllm.forward_context import BatchDescriptor

    return BatchDescriptor(
        num_tokens=num_tokens, num_reqs=num_reqs, uniform=uniform, has_lora=has_lora
    )


def _full_qwen_config(config):
    return AscendGraphConfig.model_validate(
        {
            **config.model_dump(exclude={"worker_id", "control_endpoint"}),
            "mode": "full",
            "model": {
                **config.model.model_dump(),
                "num_layers": 48,
                "experts_per_layer": 128,
                "expert_intermediate_dim": 768,
                "top_k": 8,
                "quantization": {
                    "type": "modelslim-w8a8-dynamic",
                    "bits": 8,
                    "group_size": None,
                    "symmetric": True,
                },
            },
            "layer_ids": list(range(48)),
            "workers": [
                dict(
                    worker_id=f"worker-{i}",
                    control_endpoint=f"127.0.0.1:{18100 + i}",
                    expert_start=i * 64,
                    expert_end=(i + 1) * 64,
                )
                for i in range(2)
            ],
        }
    )


@pytest.mark.parametrize("graph_runtime", ["full"], indirect=True)
def test_full_graph_accepts_generic_prefill_descriptor_and_replays_same_capacity(graph_runtime):
    events, context, runner, wrapper = graph_runtime
    runner.profile_run()
    events.clear()
    # Native FULL normalizes request count and uniformity into one generic token key.
    context.batch_descriptor = descriptor(uniform=False, num_reqs=4, num_tokens=4)
    assert runner._model_forward() == "result"
    assert events.count("layer-exchange") == 48
    assert ("begin", "capture", 4) in events
    events.clear()
    assert runner._model_forward() == "result"
    assert events == [
        ("begin", "replay", 4),
        "native-replay",
        "native-attention-update",
        "synchronize",
        "finish",
    ]
    assert len(wrapper.concrete_aclgraph_entries) == 1


def test_profile_allocates_before_probe_and_capture_replay_have_step_handshakes(graph_runtime):
    events, context, runner, wrapper = graph_runtime
    context.batch_descriptor = descriptor()
    assert runner.profile_run() == "eager"
    assert events.index(("allocate", 4)) < events.index(("eager", True))
    assert ("begin", "warmup", 4) in events
    assert not graph.is_graph_profile()
    events.clear()
    assert wrapper() == "result"
    assert events.index(("begin", "capture", 1)) < events.index("native-capture")
    assert events.count("layer-exchange") == 26
    assert events[-1] == "finish"
    events.clear()
    assert wrapper() == "result"
    assert events == [("begin", "replay", 1), "native-replay", "synchronize", "finish"]


def test_replay_completion_waits_for_native_attention_update(graph_runtime, monkeypatch):
    events, context, runner, _ = graph_runtime
    context.batch_descriptor = descriptor()
    runner.profile_run()
    assert runner._model_forward() == "result"
    assert events.count("layer-exchange") == 26
    assert events[-3:] == ["native-attention-update", "synchronize", "finish"]
    events.clear()

    class RequiredUpdateStream:
        def synchronize(self):
            assert "native-attention-update" in events, "wait blocks the required event record"
            events.append("synchronize")

    monkeypatch.setattr(torch.npu, "current_stream", RequiredUpdateStream)
    assert runner._model_forward() == "result"
    assert events == [
        ("begin", "replay", 1),
        "native-replay",
        "native-attention-update",
        "synchronize",
        "finish",
    ]


def test_native_post_forward_failure_poisoned_before_finish(graph_runtime):
    events, context, runner, wrapper = graph_runtime
    context.batch_descriptor = descriptor()
    runner.profile_run()
    wrapper()
    events.clear()
    runner.fail_update = True
    with pytest.raises(RuntimeError, match="attention update failed"):
        runner._model_forward()
    assert "poison" in events and "quarantine" in events
    assert "finish" not in events and "synchronize" not in events
    assert graph._FORWARD_SESSIONS.get() is None


@pytest.mark.parametrize(
    "options",
    [
        {"uniform": False},
        {"num_reqs": 1, "num_tokens": 4},
        {"num_reqs": 3, "num_tokens": 3},
        {"has_lora": True},
        {"num_reqs": 0, "num_tokens": 0},
    ],
)
def test_unqualified_step_uses_ordinary_path_without_control(graph_runtime, options):
    events, context, _, wrapper = graph_runtime
    context.batch_descriptor = descriptor(**options)
    assert wrapper() == "eager"
    assert events == [("eager", False)]
    assert context.cudagraph_runtime_mode == CUDAGraphMode.FULL


def test_prefill_and_mixed_steps_leave_captured_decode_reusable(graph_runtime):
    events, context, runner, wrapper = graph_runtime
    runner.profile_run()
    events.clear()
    context.batch_descriptor = descriptor(uniform=False, num_reqs=1, num_tokens=4)
    assert runner._model_forward() == "eager"
    assert wrapper.concrete_aclgraph_entries == {}
    assert events == [("eager", False), "native-attention-update"]

    context.batch_descriptor = descriptor()
    assert runner._model_forward() == "result"
    entry = wrapper.concrete_aclgraph_entries[context.batch_descriptor]
    assert events.count("native-capture") == 1
    events.clear()
    assert runner._model_forward() == "result"
    assert "native-replay" in events and "layer-exchange" not in events

    events.clear()
    context.batch_descriptor = descriptor(uniform=False, num_reqs=2, num_tokens=4)
    assert runner._model_forward() == "eager"
    assert events == [("eager", False), "native-attention-update"]

    events.clear()
    context.batch_descriptor = descriptor()
    assert runner._model_forward() == "result"
    assert wrapper.concrete_aclgraph_entries[context.batch_descriptor] is entry
    assert events == [
        ("begin", "replay", 1),
        "native-replay",
        "native-attention-update",
        "synchronize",
        "finish",
    ]


def test_partial_native_submission_poisoned_without_fallback(graph_runtime):
    events, context, runner, wrapper = graph_runtime
    context.batch_descriptor = descriptor()
    runner.profile_run()
    events.clear()
    wrapper.fail = True
    with pytest.raises(RuntimeError, match="native failure"):
        wrapper()
    assert "poison" in events and "quarantine" in events
    assert not any(isinstance(event, tuple) and event[0] == "eager" for event in events)
    with pytest.raises(RuntimeError, match="poisoned"):
        wrapper()


@pytest.mark.parametrize("max_sequences", [32, 64, 128])
@pytest.mark.parametrize("model_kind", ["v2", "qwen"])
def test_qualified_model_requires_every_capture_profile(monkeypatch, max_sequences, model_kind):
    monkeypatch.setattr(
        graph,
        "version",
        lambda name: {
            "vllm": "0.26.0+empty",
            "torch-npu": "2.10.0.post4",
            "vllm-ascend": "0.1.dev4525+g155f68974",
        }[name],
    )
    monkeypatch.setattr(torch, "__version__", "2.10.0+cpu")
    monkeypatch.setenv("EK_INSTANCE_ID", "7")
    config = AscendGraphConfig(
        session_id="qualified-session-v1",
        instance_id=7,
        worker_id="worker-0",
        weight_version="official",
        control_endpoint="127.0.0.1:18079",
        rendezvous_host="127.0.0.1",
        rendezvous_port=18080,
        layer_ids=tuple(range(1, 27)),
        capture_sizes=(1, 32, 64),
    )
    hf = SimpleNamespace(
        model_type="deepseek_v2",
        num_hidden_layers=27,
        first_k_dense_replace=1,
        hidden_size=2048,
        moe_intermediate_size=1408,
        n_routed_experts=64,
        num_experts_per_tok=6,
        n_shared_experts=2,
        scoring_func="softmax",
        norm_topk_prob=False,
        routed_scaling_factor=1.0,
        hidden_act="silu",
    )
    if model_kind == "qwen":
        hf = SimpleNamespace(
            model_type="qwen3_moe",
            num_hidden_layers=48,
            hidden_size=2048,
            moe_intermediate_size=768,
            num_experts=128,
            num_experts_per_tok=8,
            decoder_sparse_step=1,
            norm_topk_prob=True,
            mlp_only_layers=[],
            hidden_act="silu",
        )
        declaration = config.model.model_dump()
        declaration.update(
            num_layers=48, experts_per_layer=128, expert_intermediate_dim=768, top_k=8
        )
        config = AscendGraphConfig.model_validate(
            {
                **config.model_dump(),
                "model": declaration,
                "layer_ids": list(range(48)),
            }
        )
    runtime = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=hf, enforce_eager=False, dtype=torch.bfloat16, quantization=None
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            enable_dbo=False,
            enable_eplb=False,
            enable_expert_parallel=False,
        ),
        scheduler_config=SimpleNamespace(async_scheduling=False, max_num_seqs=max_sequences),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY, cudagraph_capture_sizes=[1, 32, 64]
        ),
        speculative_config=None,
        lora_config=None,
    )
    if max_sequences < 64:
        with pytest.raises(ValueError, match="max_num_seqs"):
            graph.validate_graph_model(runtime, config)
    else:
        graph.validate_graph_model(runtime, config)
        for field, value in (("hidden_dim", 4096), ("top_k", 1), ("num_layers", 49)):
            declaration = config.model.model_dump()
            declaration[field] = value
            conflicting = AscendGraphConfig.model_validate(
                {**config.model_dump(), "model": declaration}
            )
            with pytest.raises(ValueError, match="declaration does not match"):
                graph.validate_graph_model(runtime, conflicting)
        declaration = config.model.model_dump()
        declaration["quantization"] = {
            "type": "modelslim-w8a8-dynamic",
            "bits": 8,
            "group_size": None,
            "symmetric": True,
        }
        conflicting = AscendGraphConfig.model_validate(
            {**config.model_dump(), "model": declaration}
        )
        with pytest.raises(ValueError, match="declaration does not match"):
            graph.validate_graph_model(runtime, conflicting)


@pytest.fixture
def modelslim_graph_deployment(monkeypatch):
    monkeypatch.setattr(
        graph,
        "version",
        lambda name: {
            "vllm": "0.26.0+empty",
            "torch-npu": "2.10.0.post4",
            "vllm-ascend": "0.1.dev4525+g155f68974",
        }[name],
    )
    monkeypatch.setattr(torch, "__version__", "2.10.0+cpu")
    monkeypatch.setenv("EK_INSTANCE_ID", "7")
    geometry = dict(
        model_type="qwen3_moe",
        num_hidden_layers=48,
        hidden_size=2048,
        moe_intermediate_size=768,
        num_experts=128,
        num_experts_per_tok=8,
        decoder_sparse_step=1,
        norm_topk_prob=True,
        mlp_only_layers=[],
        hidden_act="silu",
    )
    description = {
        f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight": "W8A8_DYNAMIC"
        for layer in range(48)
        for expert in range(128)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    description.update({f"model.layers.{layer}.mlp.gate.weight": "FLOAT" for layer in range(48)})

    class NativeModelSlimConfig:
        def __init__(self):
            self.quant_description = description

    module_name = "vllm_ascend.quantization.modelslim_config"
    module = ModuleType(module_name)
    module.AscendModelSlimConfig = NativeModelSlimConfig
    monkeypatch.setitem(sys.modules, module_name, module)
    config = AscendGraphConfig(
        session_id="modelslim-native-v1",
        instance_id=7,
        worker_id="worker-0",
        weight_version="official",
        control_endpoint="127.0.0.1:18079",
        rendezvous_host="127.0.0.1",
        rendezvous_port=18080,
        layer_ids=tuple(range(48)),
        capture_sizes=(1, 32, 64),
        model=dict(
            num_layers=48,
            experts_per_layer=128,
            hidden_dim=2048,
            expert_intermediate_dim=768,
            top_k=8,
            activation_dtype="bf16",
            weight_dtype="bf16",
            activation="silu",
            expert_compute="swiglu",
            swiglu_limit=0.0,
            quantization=dict(
                type="modelslim-w8a8-dynamic", bits=8, group_size=None, symmetric=True
            ),
        ),
    )
    runtime = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(**geometry),
            enforce_eager=False,
            dtype=torch.bfloat16,
            quantization="ascend",
        ),
        quant_config=NativeModelSlimConfig(),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            enable_dbo=False,
            enable_eplb=False,
            enable_expert_parallel=False,
        ),
        scheduler_config=SimpleNamespace(async_scheduling=False, max_num_seqs=64),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            cudagraph_capture_sizes=[1, 32, 64],
        ),
        speculative_config=None,
        lora_config=None,
    )
    return runtime, config


def test_modelslim_native_recipe_matches_complete_shared_declaration(modelslim_graph_deployment):
    runtime, config = modelslim_graph_deployment
    graph.validate_graph_model(runtime, config)
    conflicting = AscendGraphConfig.model_validate(
        {
            **config.model_dump(),
            "model": {**config.model.model_dump(), "quantization": None},
        }
    )
    with pytest.raises(ValueError, match="declaration does not match"):
        graph.validate_graph_model(runtime, conflicting)
    runtime.parallel_config.tensor_parallel_size = 2
    with pytest.raises(ValueError, match="single-rank"):
        graph.validate_graph_model(runtime, config)


def test_full_graph_token_capacity_is_independent_of_sequence_limit(modelslim_graph_deployment):
    runtime, config = modelslim_graph_deployment
    config = _full_qwen_config(config).model_copy(update={"capture_sizes": (1, 64, 128, 256)})
    runtime.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
    runtime.compilation_config.cudagraph_capture_sizes = list(config.capture_sizes)
    runtime.scheduler_config.max_num_batched_tokens = 2048
    graph.validate_graph_model(runtime, config)
    runtime.scheduler_config.max_num_batched_tokens = 128
    with pytest.raises(ValueError, match="max_num_batched_tokens"):
        graph.validate_graph_model(runtime, config)
    runtime.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    with pytest.raises(ValueError, match="configured native graph mode"):
        graph.validate_graph_model(runtime, config)


@pytest.mark.parametrize("change", ["bf16", "one_worker", "layers", "capacity"])
def test_full_graph_rejects_unqualified_model_or_profile(modelslim_graph_deployment, change):
    _, previous = modelslim_graph_deployment
    values = _full_qwen_config(previous).model_dump()
    if change == "bf16":
        values["model"]["quantization"] = None
    elif change == "one_worker":
        values["workers"] = [{**values["workers"][0], "expert_end": 128}]
    elif change == "layers":
        values["layer_ids"] = [0]
    else:
        values["capture_sizes"] = [1, 4096]
    with pytest.raises(ValueError, match="full graph"):
        AscendGraphConfig.model_validate(values)


@pytest.mark.parametrize(
    "package,other_version",
    [
        ("vllm", "0.25.1"),
        ("vllm", "0.30.0"),
        ("vllm-ascend", "0.26.0rc1"),
        ("vllm-ascend", "0.1.dev4525+g155f68975"),
        ("vllm-ascend", "0.1.dev1+g8d4409d62.d20261013"),
        ("torch-npu", "2.10.0.post3"),
        ("torch", "2.11.0"),
    ],
)
def test_graph_rejects_other_runtime_revisions(
    modelslim_graph_deployment, monkeypatch, package, other_version
):
    runtime, config = modelslim_graph_deployment
    if package == "torch":
        monkeypatch.setattr(torch, "__version__", other_version)
    else:
        runtime_version = graph.version
        monkeypatch.setattr(
            graph,
            "version",
            lambda name: other_version if name == package else runtime_version(name),
        )
    with pytest.raises(ValueError, match="Ascend Decode graphs require"):
        graph.validate_graph_model(runtime, config)


@pytest.mark.parametrize("change", ["prefix_cache", "tracing"])
def test_graph_rejects_unqualified_runtime_features(
    modelslim_graph_deployment, monkeypatch, change
):
    runtime, config = modelslim_graph_deployment
    if change == "prefix_cache":
        runtime.cache_config.enable_prefix_caching = True
    else:
        monkeypatch.setenv("EK_TRACE_ENDPOINT", "http://127.0.0.1:4318/v1/traces")
    with pytest.raises(ValueError, match="prefix caching and tracing disabled"):
        graph.validate_graph_model(runtime, config)


@pytest.mark.parametrize(
    "name,value",
    [
        ("model.layers.47.mlp.gate.weight", "W8A8_DYNAMIC"),
        ("model.layers.47.mlp.experts.127.down_proj.weight", "W8A8"),
        ("model.layers.0.mlp.experts.0.up_proj.weight", None),
    ],
)
def test_modelslim_requires_every_exact_native_descriptor(modelslim_graph_deployment, name, value):
    runtime, config = modelslim_graph_deployment
    if value is None:
        del runtime.quant_config.quant_description[name]
    else:
        runtime.quant_config.quant_description[name] = value
    with pytest.raises(ValueError, match="requires"):
        graph.validate_graph_model(runtime, config)


@pytest.mark.parametrize(
    "change", ["native_type", "missing_native", "bf16_with_native", "other_quant"]
)
def test_modelslim_rejects_inconsistent_native_configuration(modelslim_graph_deployment, change):
    runtime, config = modelslim_graph_deployment
    if change == "native_type":
        runtime.quant_config = SimpleNamespace(
            quant_description=runtime.quant_config.quant_description
        )
    elif change == "missing_native":
        runtime.quant_config = None
    else:
        runtime.model_config.quantization = None if change == "bf16_with_native" else "gptq"
    with pytest.raises(ValueError, match="native"):
        graph.validate_graph_model(runtime, config)


@pytest.fixture
def v4_graph_deployment(modelslim_graph_deployment, monkeypatch):
    runtime, previous = modelslim_graph_deployment
    runtime.model_config.hf_text_config = SimpleNamespace(
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
    description = {
        f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight": "W8A8_DYNAMIC"
        for layer in range(43)
        for expert in range(256)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    for layer in range(43):
        prefix = f"model.layers.{layer}.mlp"
        description[f"{prefix}.gate.weight"] = "FLOAT"
        description[f"{prefix}.gate.{'tid2eid' if layer < 3 else 'bias'}"] = "FLOAT"
        for projection in ("gate_proj", "up_proj", "down_proj"):
            description[f"{prefix}.shared_experts.{projection}.weight"] = "W8A8_DYNAMIC"
    runtime.quant_config.quant_description = description
    ascend = ModuleType("vllm_ascend.ascend_config")
    ascend.get_ascend_config = lambda: SimpleNamespace(mix_placement=False)
    monkeypatch.setitem(sys.modules, ascend.__name__, ascend)
    declaration = {
        **previous.model.model_dump(),
        "num_layers": 43,
        "experts_per_layer": 256,
        "hidden_dim": 4096,
        "expert_intermediate_dim": 2048,
        "top_k": 6,
        "expert_compute": "deepseek_v4",
        "swiglu_limit": 10.0,
    }
    config = AscendGraphConfig.model_validate(
        {
            **previous.model_dump(exclude={"worker_id", "control_endpoint"}),
            "model": declaration,
            "layer_ids": tuple(range(43)),
            "workers": tuple(
                dict(
                    worker_id=f"worker-{rank}",
                    control_endpoint=f"127.0.0.1:{18100 + rank}",
                    expert_start=32 * rank,
                    expert_end=32 * (rank + 1),
                )
                for rank in range(8)
            ),
        }
    )
    return runtime, config


def test_v4_graph_requires_complete_native_declaration_and_layer_order(v4_graph_deployment):
    runtime, config = v4_graph_deployment
    graph.validate_graph_model(runtime, config)
    declaration = config.model.model_dump()
    declaration["expert_compute"], declaration["swiglu_limit"] = "swiglu", 0.0
    conflicting = config.model_copy(update={"model": type(config.model)(**declaration)})
    with pytest.raises(ValueError, match="declaration does not match"):
        graph.validate_graph_model(runtime, conflicting)
    with pytest.raises(ValueError, match="every routed model layer"):
        graph.validate_graph_model(runtime, config.model_copy(update={"layer_ids": (0,)}))


@pytest.mark.parametrize(
    "name,value",
    [
        ("model.layers.42.mlp.gate.weight", "W8A8_DYNAMIC"),
        ("model.layers.42.mlp.experts.255.down_proj.weight", "W8A8"),
        ("model.layers.42.mlp.shared_experts.up_proj.weight", None),
        ("model.layers.2.mlp.gate.tid2eid", None),
        ("model.layers.42.mlp.gate.bias", "W8A8_DYNAMIC"),
    ],
)
def test_v4_graph_requires_every_exact_native_descriptor(v4_graph_deployment, name, value):
    runtime, config = v4_graph_deployment
    if value is None:
        del runtime.quant_config.quant_description[name]
    else:
        runtime.quant_config.quant_description[name] = value
    with pytest.raises(ValueError, match="requires"):
        graph.validate_graph_model(runtime, config)


@pytest.mark.parametrize("change", ["unquantized", "wrong_native", "vision", "mix", "router"])
def test_v4_graph_rejects_unqualified_native_semantics(v4_graph_deployment, change):
    runtime, config = v4_graph_deployment
    if change == "unquantized":
        runtime.model_config.quantization = runtime.quant_config = None
    elif change == "wrong_native":
        runtime.quant_config = SimpleNamespace(
            quant_description=runtime.quant_config.quant_description
        )
    elif change == "vision":
        runtime.model_config.hf_text_config.vision_n_layers = 1
    elif change == "mix":
        sys.modules["vllm_ascend.ascend_config"].get_ascend_config = lambda: SimpleNamespace(
            mix_placement=True
        )
    else:
        runtime.model_config.hf_text_config.routed_scaling_factor = 1.0
    with pytest.raises(ValueError):
        graph.validate_graph_model(runtime, config)


def test_exchange_uses_shared_hidden_and_top_k_without_device_reads():
    config = AscendGraphConfig(
        session_id="frontend-buffer-0061",
        instance_id=1,
        worker_id="worker",
        weight_version="version",
        control_endpoint="127.0.0.1:18079",
        rendezvous_host="127.0.0.1",
        rendezvous_port=18080,
        layer_ids=(0,),
        model={
            "num_layers": 48,
            "experts_per_layer": 128,
            "hidden_dim": 16,
            "expert_intermediate_dim": 768,
            "top_k": 8,
            "activation_dtype": "bf16",
            "weight_dtype": "bf16",
            "activation": "silu",
            "expert_compute": "swiglu",
            "swiglu_limit": 0.0,
            "quantization": None,
        },
    )
    session = graph.DecodeGraphSession.__new__(graph.DecodeGraphSession)
    session.config, session._poisoned, session._layer_index, session._capacity = config, False, 0, 4
    session._data = SimpleNamespace(exchange=lambda hidden, *_: hidden)
    hidden = torch.zeros(4, 16, dtype=torch.bfloat16)
    ids = torch.zeros(4, 8, dtype=torch.int32)
    routing = torch.zeros(4, 8, dtype=torch.float32)
    assert session.exchange(0, hidden, ids, routing) is hidden
    session._layer_index = 0
    with pytest.raises(ValueError, match="Tensor contract"):
        session.exchange(0, hidden, ids[:, :6], routing[:, :6])
