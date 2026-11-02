"""Exercise native wrapper phase coordination with external CPU boundaries."""

import atexit
import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest
import torch
from expertkit_transport.transports.ascend_graph.config import AscendGraphConfig, AscendGraphOptions
from vllm.config import CUDAGraphMode

from expertkit_vllm import decode_graph as graph


def _workers(layers, experts, count=1):
    return tuple(
        dict(
            worker_id=f"worker-{rank}",
            expected_start_id=f"start-{rank}",
            control_endpoint=f"127.0.0.1:{18100 + rank}",
            device=f"npu:{rank}",
            experts_by_layer=tuple(
                tuple(range(rank * experts // count, (rank + 1) * experts // count)) for _ in layers
            ),
        )
        for rank in range(count)
    )


def _graph_config(**values):
    layers = values.get("layer_ids", tuple(range(1, 27)))
    model = values.get(
        "model",
        dict(
            num_layers=27,
            experts_per_layer=64,
            hidden_dim=2048,
            expert_intermediate_dim=1408,
            top_k=6,
            activation_dtype="bf16",
            weight_dtype="bf16",
            activation="silu",
            expert_compute="swiglu",
            swiglu_limit=0.0,
            quantization=None,
        ),
    )
    captures = values.get("capture_sizes", (1, 2, 4, 8, 16, 32, 64))
    return AscendGraphConfig(
        mode="decode",
        model_name="native-model",
        source_topology_version=1,
        model=model,
        workers=_workers(layers, model["experts_per_layer"]),
        decode_capture_sizes=captures,
        capture_sizes=captures,
        timeout_seconds=values.get("timeout_seconds", 120.0),
        **{
            key: value
            for key, value in values.items()
            if key
            not in {"worker_id", "control_endpoint", "model", "timeout_seconds", "capture_sizes"}
        },
    )


def _options(config):
    return AscendGraphOptions.model_validate(
        config.model_dump(include=set(AscendGraphOptions.model_fields))
    )


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
        def __init__(self, config, *, require_current=None):
            self.config = config
            self.require_current = require_current or (lambda: None)

        def begin(self, phase, capacity=0, *, graph_phase=None):
            if phase != "close":
                self.require_current()
            events.append(("begin", phase, capacity))
            if graph_phase is not None:
                events.append(("graph-phase", graph_phase))

        def finish(self):
            events.append("finish")

        def poison(self):
            events.append("poison")

        def close(self):
            events.append("control-close")

    class Data:
        def __init__(self, config, **options):
            self.config = config
            self.profiles = {}

        def buffers(self, capacity):
            if capacity not in self.profiles:
                self.profiles[capacity] = SimpleNamespace(
                    output=torch.zeros(capacity, config.model.hidden_dim)
                )
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
            self.capture_failure_factory = None

        def __call__(self):
            if context.cudagraph_runtime_mode == CUDAGraphMode.NONE:
                events.append(("eager", graph.is_graph_profile()))
                return "eager"
            descriptor = context.batch_descriptor
            if descriptor not in self.concrete_aclgraph_entries:
                events.append("native-capture")
                if self.capture_failure_factory is not None:
                    partial_graph = self.capture_failure_factory()
                    raise RuntimeError(f"partial native capture: {type(partial_graph).__name__}")
                session = graph.active_graph_session()
                assert session is not None
                for layer in config.layer_ids:
                    session.exchange(
                        layer,
                        torch.zeros(
                            descriptor.num_tokens, config.model.hidden_dim, dtype=torch.bfloat16
                        ),
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

        def _dummy_run(self, num_tokens, is_profile=False):
            return self.model()

        def execute_model(self, scheduler_output):
            return self._model_forward()

        def _capture_cudagraphs(self, descriptors, mode, profiler=None):
            if hasattr(self, "capture_callback"):
                return self.capture_callback()
            events.append(("capture-descriptors", tuple(descriptors), mode, profiler))

    for name, attribute, value in (
        ("vllm_ascend.compilation.acl_graph", "ACLGraphWrapper", Wrapper),
        ("vllm_ascend.worker.model_runner_v1", "NPUModelRunner", Runner),
    ):
        module = ModuleType(name)
        setattr(module, attribute, value)
        monkeypatch.setitem(sys.modules, name, module)
    module = ModuleType("expertkit_vllm.v4_full_attention")
    module.install_v4_memory_reservation = lambda: None
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(graph, "install_graph_scheduler", lambda _: None)
    config = _graph_config(
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
    if getattr(request, "param", "decode") == "v4-full":
        values = _full_qwen_config(config).model_dump()
        values["model"].update(
            num_layers=43,
            experts_per_layer=256,
            hidden_dim=4096,
            expert_intermediate_dim=2048,
            top_k=6,
            expert_compute="deepseek_v4",
            swiglu_limit=10.0,
        )
        values.update(
            layer_ids=tuple(range(43)),
            capture_sizes=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048),
            decode_capture_sizes=(1, 2, 4, 8, 16, 32, 64, 128, 256),
            workers=_workers(tuple(range(43)), 256, 8),
        )
        config = AscendGraphConfig.model_validate(values)

        class Adapter:
            phase = "decode"

            @contextmanager
            def capture_phase(self, phase):
                previous, self.phase = self.phase, phase
                events.append(("capture-phase", phase))
                try:
                    yield
                finally:
                    self.phase = previous

            def prepare_profile_metadata(self):
                events.append("profile-metadata")

            def close(self):
                assert all(not entries for entries in wrapper._expertkit_v4_phase_entries.values())
                events.append("adapter-close")

        adapter = Adapter()
        module = ModuleType("expertkit_vllm.v4_full_attention")

        def install(runner, _config):
            events.append("adapter-install")
            runner._expertkit_v4_full_attention = adapter
            adapter.graph_config = _config
            return adapter

        module.install_v4_memory_reservation = lambda: None
        module.install_v4_full_attention = install
        monkeypatch.setitem(sys.modules, module.__name__, module)
    topology = SimpleNamespace(
        require_current=lambda _: None, close=lambda: events.append("topology-close")
    )

    def resolve(*args):
        events.append("resolve-runtime")
        return config, topology, None, None

    monkeypatch.setattr(graph, "_resolve_graph_config", resolve)
    graph.install_decode_graph_wrapper(_options(config))
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
            "workers": _workers(tuple(range(48)), 128, 2),
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
    config = _graph_config(
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
                "workers": _workers(tuple(range(48)), 128),
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
        scheduler_config=SimpleNamespace(
            async_scheduling=False,
            max_num_seqs=max_sequences,
            max_num_batched_tokens=2048,
            max_num_scheduled_tokens=None,
            scheduler_cls=None,
            enable_chunked_prefill=True,
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY, cudagraph_capture_sizes=[1, 32, 64]
        ),
        speculative_config=None,
        lora_config=None,
    )
    if max_sequences < 64:
        with pytest.raises(ValueError, match="allocated request buffers"):
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
    config = _graph_config(
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
        scheduler_config=SimpleNamespace(
            async_scheduling=False,
            max_num_seqs=64,
            max_num_batched_tokens=2048,
            max_num_scheduled_tokens=None,
            scheduler_cls=None,
            enable_chunked_prefill=True,
        ),
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
    config = _full_qwen_config(config).model_copy(
        update={"capture_sizes": (1, 64, 128, 256), "decode_capture_sizes": (1, 64)}
    )
    runtime.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
    runtime.compilation_config.cudagraph_capture_sizes = list(config.capture_sizes)
    runtime.scheduler_config.max_num_batched_tokens = 2048
    graph.validate_graph_model(runtime, config)
    runtime.scheduler_config.max_num_batched_tokens = 128
    with pytest.raises(ValueError, match="allocated token buffers"):
        graph.validate_graph_model(runtime, config)
    runtime.scheduler_config.max_num_batched_tokens = 2048
    runtime.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    with pytest.raises(ValueError, match="configured native graph mode"):
        graph.validate_graph_model(runtime, config)


def test_full_graph_has_no_experimental_geometry_or_participant_whitelist(
    modelslim_graph_deployment,
):
    runtime, previous = modelslim_graph_deployment
    config = _full_qwen_config(previous)
    config = AscendGraphConfig.model_validate(
        {**config.model_dump(), "workers": _workers(config.layer_ids, 128)}
    )
    runtime.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
    graph.validate_graph_model(runtime, config)


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
            "workers": _workers(tuple(range(43)), 256, 8),
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
        runtime.model_config.hf_text_config.scoring_func = "softmax"
    with pytest.raises(ValueError):
        graph.validate_graph_model(runtime, config)


def test_exchange_uses_shared_hidden_and_top_k_without_device_reads():
    config = _graph_config(
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


@pytest.mark.parametrize("graph_runtime", ["v4-full"], indirect=True)
def test_v4_capture_preserves_native_descriptors_and_phase_scope(graph_runtime):
    events, _context, runner, wrapper = graph_runtime
    runner.profile_run()
    adapter = runner._expertkit_v4_full_attention
    items = [
        descriptor(uniform=False, num_tokens=t)
        for t in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
    ]
    marker = object()
    events.clear()
    runner._capture_cudagraphs(items, CUDAGraphMode.FULL, marker)
    captures = [
        event for event in events if isinstance(event, tuple) and event[0] == "capture-descriptors"
    ]
    assert [event[1] for event in captures] == [tuple(items[:9]), tuple(items[1:])]
    assert all(event[2:] == (CUDAGraphMode.FULL, marker) for event in captures)
    assert adapter.phase == "decode"
    assert wrapper.concrete_aclgraph_entries == {}


@pytest.mark.parametrize("graph_runtime", ["v4-full"], indirect=True)
def test_v4_selects_phase_map_before_session_lookup_and_clears_both(graph_runtime):
    events, context, runner, wrapper = graph_runtime
    runner.profile_run()
    adapter = runner._expertkit_v4_full_attention
    context.batch_descriptor = descriptor(uniform=False, num_reqs=4, num_tokens=4)
    original_map = wrapper.concrete_aclgraph_entries
    for phase in ("decode", "mixed"):
        adapter.phase = phase
        events.clear()
        assert runner._model_forward() == "result"
        assert "native-capture" in events and "native-replay" not in events
        assert ("graph-phase", phase) in events
        assert wrapper.concrete_aclgraph_entries is original_map
    for phase in ("mixed", "decode"):
        adapter.phase = phase
        events.clear()
        assert runner._model_forward() == "result"
        assert "native-replay" in events and "native-capture" not in events
        assert ("graph-phase", phase) in events
    session = wrapper._expertkit_decode_session
    assert all(len(entries) == 1 for entries in session._v4_phase_entries.values())
    session.close()
    assert all(not entries for entries in session._v4_phase_entries.values())
    assert events.index("adapter-close") < events.index("data-close")


@pytest.mark.parametrize("graph_runtime", ["v4-full"], indirect=True)
def test_v4_failure_restores_phase_map_and_retains_adapter_storage(graph_runtime):
    events, context, runner, wrapper = graph_runtime
    runner.profile_run()
    adapter = runner._expertkit_v4_full_attention
    adapter.phase = "mixed"
    context.batch_descriptor = descriptor(uniform=False, num_reqs=4, num_tokens=4)
    previous = wrapper.concrete_aclgraph_entries
    wrapper.fail = True
    with pytest.raises(RuntimeError, match="native failure"):
        runner._model_forward()
    assert wrapper.concrete_aclgraph_entries is previous
    session = wrapper._expertkit_decode_session
    assert session._poisoned
    session.close()
    assert "adapter-close" not in events and "data-close" not in events


@pytest.mark.parametrize("graph_runtime", ["v4-full"], indirect=True)
def test_v4_capture_scope_failure_poisons_session_without_releasing_adapter(graph_runtime):
    events, _context, runner, wrapper = graph_runtime
    runner.profile_run()
    adapter = runner._expertkit_v4_full_attention
    original_scope = adapter.capture_phase

    @contextmanager
    def failed_scope(phase):
        with original_scope(phase):
            yield
            raise RuntimeError("capture scope failed")

    adapter.capture_phase = failed_scope
    with pytest.raises(RuntimeError, match="capture scope failed"):
        runner._capture_cudagraphs([descriptor(num_tokens=4)], CUDAGraphMode.FULL)
    assert adapter.phase == "decode"
    session = wrapper._expertkit_decode_session
    assert session._poisoned
    session.close()
    assert "adapter-close" not in events


@pytest.mark.parametrize("graph_runtime", ["decode", "full", "v4-full"], indirect=True)
def test_capture_failure_traceback_holds_unpublished_native_graph(graph_runtime):
    import gc
    import weakref

    events, context, runner, wrapper = graph_runtime
    runner.profile_run()
    context.batch_descriptor = descriptor(num_reqs=4, num_tokens=4)
    references = []

    class PartialGraph:
        pass

    def create_graph():
        value = PartialGraph()
        references.append(weakref.ref(value))
        return value

    wrapper.capture_failure_factory = create_graph
    try:
        runner._model_forward()
    except RuntimeError as error:
        assert "partial native capture" in str(error)
    else:
        pytest.fail("capture must fail before publishing its native graph")
    session = wrapper._expertkit_decode_session
    assert session._poisoned and session._native_capture_failure is not None
    assert not wrapper.concrete_aclgraph_entries
    if session._v4_phase_entries is not None:
        assert not any(session._v4_phase_entries.values())
    gc.collect()
    assert references[0]() is not None
    session.close()
    gc.collect()
    assert references[0]() is not None
    assert "adapter-close" not in events and "data-close" not in events


@pytest.mark.parametrize("graph_runtime", ["decode", "full"], indirect=True)
@pytest.mark.parametrize("change", ["none", "descriptor", "capacity", "lora", "wrapper"])
def test_real_graph_required_steps_reject_before_worker_or_model_submission(graph_runtime, change):
    events, context, runner, wrapper = graph_runtime
    runner.profile_run()
    context.batch_descriptor = descriptor()
    if change == "none":
        context.cudagraph_runtime_mode = CUDAGraphMode.NONE
    elif change == "descriptor":
        context.batch_descriptor = None
    elif change == "capacity":
        context.batch_descriptor = descriptor(num_tokens=3, num_reqs=3)
    elif change == "lora":
        context.batch_descriptor = descriptor(has_lora=True)
    else:
        wrapper.runtime_mode = CUDAGraphMode.NONE
    events.clear()
    with pytest.raises(ValueError, match="no eligible native FULL graph"):
        runner.execute_model(SimpleNamespace(num_scheduled_tokens={"request": 1}))
    assert events == []
    assert graph._REAL_STEP.get() is None


def test_decode_prefill_classification_uses_host_scheduling_metadata(graph_runtime):
    events, context, runner, _wrapper = graph_runtime
    runner.profile_run()
    # The native descriptor can look uniform after padding; Host query counts
    # still identify this as an ordinary Decode-only mixed/Prefill step.
    context.batch_descriptor = descriptor(num_tokens=4, num_reqs=4)
    context.cudagraph_runtime_mode = CUDAGraphMode.NONE
    events.clear()
    assert (
        runner.execute_model(SimpleNamespace(num_scheduled_tokens={"prefill": 3, "decode": 1}))
        == "eager"
    )
    assert events == [("eager", False), "native-attention-update"]


@pytest.mark.parametrize("graph_runtime", ["decode", "full"], indirect=True)
def test_native_compile_dummy_endpoint_is_not_a_real_request(graph_runtime):
    events, context, runner, _wrapper = graph_runtime
    runner.profile_run()
    context.batch_descriptor = descriptor(num_tokens=3, num_reqs=3)
    events.clear()
    assert runner._dummy_run(3, False) == "eager"
    assert events == [("eager", False)]
    assert not graph._SYNTHETIC.get()


@pytest.mark.parametrize("graph_runtime", ["v4-full"], indirect=True)
def test_v4_profile_metadata_prepared_inside_native_profile_window(graph_runtime):
    events, _context, runner, _wrapper = graph_runtime
    runner.profile_run()
    assert events.index("adapter-install") < events.index(("begin", "prepare", 0))
    assert events.index("profile-metadata") < events.index(("eager", True))


def test_session_checks_topology_before_coarse_begin_and_retains_failed_owners(graph_runtime):
    events, context, runner, wrapper = graph_runtime
    runner.profile_run()
    session = wrapper._expertkit_decode_session
    context.batch_descriptor = descriptor()
    events.clear()

    def changed():
        raise RuntimeError("selected dependency changed")

    session._control.require_current = changed
    with pytest.raises(RuntimeError, match="dependency changed"):
        runner.execute_model(SimpleNamespace(num_scheduled_tokens={"request": 1}))
    assert "native-capture" not in events and "warmup-input" not in events
    assert session._poisoned
    session.close()
    assert "data-close" not in events
    assert "topology-close" in events and "control-close" in events
    assert not session._watchdog.is_alive()


def test_startup_prepare_failure_keeps_attention_and_closes_host_owners(graph_runtime, monkeypatch):
    events, _context, runner, _wrapper = graph_runtime
    retained = []
    monkeypatch.setattr(graph, "retain_until_process_exit", retained.append)
    original_control = graph.GraphStepControl

    class FailingControl(original_control):
        def begin(self, *args, **kwargs):
            raise RuntimeError("partial PREPARE")

    monkeypatch.setattr(graph, "GraphStepControl", FailingControl)
    attention = object()
    topology = SimpleNamespace(
        require_current=lambda _: None, close=lambda: events.append("topology-close")
    )
    runtime = graph._resolve_graph_config(None, None)[0]
    with pytest.raises(RuntimeError, match="partial PREPARE"):
        graph.DecodeGraphSession(runtime, runner.device, topology=topology, attention=attention)
    assert len(retained) == 1
    session = retained[0]
    assert session._v4_attention is attention and session._poisoned and session._closed
    assert not session._watchdog.is_alive()
    assert "topology-close" in events and "control-close" in events
    assert "data-close" not in events


@pytest.mark.parametrize("mode,expected_tokens", [("full", 12), ("decode", 24)])
def test_scheduler_hook_changes_runtime_fields_without_mutating_allocations(
    monkeypatch, mode, expected_tokens
):
    import vllm.config
    import vllm.v1.core.sched.scheduler as native

    class Scheduler:
        def __init__(self, vllm_config):
            self.max_num_running_reqs = vllm_config.scheduler_config.max_num_seqs
            self.max_num_scheduled_tokens = vllm_config.scheduler_config.max_num_scheduled_tokens

    class SchedulerConfig:
        def get_scheduler_cls(self):
            return Scheduler

    monkeypatch.setattr(native, "Scheduler", Scheduler)
    monkeypatch.setattr(vllm.config, "SchedulerConfig", SchedulerConfig)
    options = AscendGraphOptions(
        mode=mode, capture_sizes=(3, 7, 12), rendezvous_host="host", rendezvous_port=1234
    )
    graph.install_graph_scheduler(options)
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_seqs=8 if mode == "full" else 16,
            max_num_batched_tokens=32,
            max_num_scheduled_tokens=24,
            scheduler_cls=None,
        ),
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[12, 3, 7, 3]),
    )
    before = vars(config.scheduler_config).copy()
    scheduler = Scheduler(vllm_config=config)
    assert scheduler.max_num_running_reqs == (7 if mode == "full" else 12)
    assert scheduler.max_num_scheduled_tokens == expected_tokens
    assert vars(config.scheduler_config) == before
    assert SchedulerConfig().get_scheduler_cls() is Scheduler
    config.compilation_config.cudagraph_capture_sizes = [3, 7]
    with pytest.raises(ValueError, match="capture sizes must match"):
        Scheduler(config)


def test_explicit_custom_scheduler_cannot_silently_miss_graph_limits(monkeypatch):
    import vllm.config
    import vllm.v1.core.sched.scheduler as native

    class CustomScheduler:
        pass

    class SchedulerConfig:
        def get_scheduler_cls(self):
            return CustomScheduler

    class Scheduler:
        def __init__(self, vllm_config):
            pass

    monkeypatch.setattr(native, "Scheduler", Scheduler)
    monkeypatch.setattr(vllm.config, "SchedulerConfig", SchedulerConfig)
    options = AscendGraphOptions(rendezvous_host="host", rendezvous_port=1234)
    graph.install_graph_scheduler(options)
    with pytest.raises(ValueError, match="default native Scheduler"):
        SchedulerConfig().get_scheduler_cls()


def test_options_environment_rejects_old_placement_document(tmp_path, monkeypatch):
    path = tmp_path / "options.json"
    path.write_text('{"capture_sizes":[12,3,7,3],"rendezvous_host":"host","rendezvous_port":1234}')
    monkeypatch.setenv("EK_ASCEND_DECODE_GRAPH_CONFIG", str(path))
    options = graph.graph_config_from_environment()
    assert isinstance(options, AscendGraphOptions) and options.capture_sizes == (3, 7, 12)
    path.write_text('{"workers":[]}')
    with pytest.raises(ValueError, match="ek-cli schedule"):
        graph.graph_config_from_environment()


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"rendezvous_host": "host"},
        {"rendezvous_port": 1234},
        {"rendezvous_host": "host", "rendezvous_port": 1234},
    ],
)
@pytest.mark.parametrize("control_host", ["127.0.0.1", "[::1]"])
def test_startup_resolves_native_facts_and_ready_assignments_once(
    modelslim_graph_deployment, monkeypatch, overrides, control_host
):
    runtime, existing = modelslim_graph_deployment
    calls = []
    runtime.scheduler_config.max_num_seqs = 8
    runtime.scheduler_config.max_num_scheduled_tokens = 24
    runtime.compilation_config.cudagraph_capture_sizes = [3, 7, 12]
    runtime.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
    options = AscendGraphOptions(mode="full", capture_sizes=(3, 7, 12), **overrides)

    def resolve(endpoint, *, requested_instance_id, monotonic_deadline):
        calls.append(("instance", endpoint, requested_instance_id, monotonic_deadline))
        return SimpleNamespace(instance_id=7, model_name="controller-model")

    class Topology:
        def __init__(self, endpoint, **kwargs):
            calls.append(("topology", endpoint, kwargs))

        def start(self, **kwargs):
            calls.append(("start", kwargs))
            workers = tuple(
                worker.model_copy(update={"control_endpoint": f"{control_host}:12345"})
                for worker in existing.workers
            )
            return 9, "ready-version", workers

        def require_current(self, config):
            calls.append(("current", config))

        def close(self):
            calls.append("close")

    monkeypatch.setattr(graph, "resolve_default_instance_sync", resolve)
    monkeypatch.setattr(graph, "GraphTopologySubscription", Topology)
    monkeypatch.setenv("EK_ADDR", "controller:5002")
    runner = SimpleNamespace(vllm_config=runtime, max_num_reqs=8, max_num_tokens=32)

    def reserve(facts, *, monotonic_deadline, require_current):
        assert facts["rendezvous_host"] == overrides.get(
            "rendezvous_host", control_host.strip("[]")
        )
        assert facts["rendezvous_port"] == overrides.get("rendezvous_port", 0)
        config = AscendGraphConfig.model_validate(
            {**facts, "rendezvous_port": facts["rendezvous_port"] or 32123}
        )
        require_current(config)
        return SimpleNamespace(config=config)

    monkeypatch.setattr(graph.GraphStepControl, "reserve", reserve)
    config, topology, deadline, control = graph._resolve_graph_config(runner, options)
    assert control.config is config
    assert config.model == existing.model and config.layer_ids == existing.layer_ids
    assert config.instance_id == 7 and config.model_name == "controller-model"
    assert config.weight_version == "ready-version" and config.source_topology_version == 9
    assert config.decode_capture_sizes == (3, 7)
    assert len(config.session_id) == 32
    assert calls[0] == ("instance", "controller:5002", 7, deadline)
    assert calls[2][1] == {
        "layer_ids": config.layer_ids,
        "max_capture_size": 12,
        "monotonic_deadline": deadline,
    }
    assert calls[-1] == ("current", config)
    topology.close()


def test_native_hf_geometry_and_sparse_layers_are_not_experimental_constants(
    modelslim_graph_deployment,
):
    runtime, _ = modelslim_graph_deployment
    runtime.model_config.quantization = None
    runtime.quant_config = None
    hf = runtime.model_config.hf_text_config
    hf.num_hidden_layers, hf.hidden_size, hf.moe_intermediate_size = 7, 1536, 640
    hf.num_experts, hf.num_experts_per_tok = 11, 3
    hf.decoder_sparse_step, hf.mlp_only_layers = 2, [3]
    model, layers = graph.resolve_graph_model(runtime)
    assert (model.num_layers, model.hidden_dim, model.experts_per_layer, model.top_k) == (
        7,
        1536,
        11,
        3,
    )
    assert layers == (1, 5)


def test_installed_native_scheduler_chunks_300_tokens_and_dispatches_smallest_capture(monkeypatch):
    """Exercise installed CPU policy; this is not pinned Ascend 0.26 qualification."""
    from vllm import SamplingParams
    from vllm.config import CacheConfig, SchedulerConfig
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
    from vllm.v1.request import Request

    monkeypatch.setattr(Scheduler, "__init__", Scheduler.__init__)
    monkeypatch.setattr(SchedulerConfig, "get_scheduler_cls", SchedulerConfig.get_scheduler_cls)
    options = AscendGraphOptions(
        mode="full", capture_sizes=(64, 128, 256), rendezvous_host="host", rendezvous_port=1234
    )
    graph.install_graph_scheduler(options)
    cache = CacheConfig(enable_prefix_caching=False)
    cache.num_gpu_blocks = 128
    scheduling = SchedulerConfig(
        max_model_len=4096,
        is_encoder_decoder=False,
        max_num_seqs=128,
        max_num_batched_tokens=2048,
        max_num_scheduled_tokens=2048,
        async_scheduling=False,
    )
    config = SimpleNamespace(
        scheduler_config=scheduling,
        cache_config=cache,
        lora_config=None,
        kv_events_config=None,
        parallel_config=SimpleNamespace(
            data_parallel_index=0,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        observability_config=SimpleNamespace(kv_cache_metrics=False, enable_mfu_metrics=False),
        model_config=SimpleNamespace(
            is_encoder_decoder=False,
            max_model_len=4096,
            is_diffusion=False,
            enable_return_routed_experts=False,
        ),
        kv_transfer_config=None,
        ec_transfer_config=None,
        speculative_config=None,
        num_speculative_tokens=0,
        use_v2_model_runner=False,
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL,
            cudagraph_capture_sizes=[64, 128, 256],
            max_cudagraph_capture_size=256,
            compile_sizes=[],
        ),
    )
    kv = KVCacheConfig(
        num_blocks=128,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["layer"],
                kv_cache_spec=FullAttentionSpec(
                    16, num_kv_heads=1, head_size=16, dtype=torch.bfloat16
                ),
            )
        ],
    )
    scheduler = Scheduler(
        config,
        kv,
        SimpleNamespace(),
        16,
        mm_registry=SimpleNamespace(supports_multimodal_inputs=lambda _: False),
    )
    scheduler.add_request(Request("request", [1] * 300, SamplingParams(max_tokens=1), None))
    first, second = scheduler.schedule(), scheduler.schedule()
    assert first.num_scheduled_tokens == {"request": 256}
    assert second.num_scheduled_tokens == {"request": 44}
    assert scheduling.max_num_batched_tokens == 2048
    dispatcher = CudagraphDispatcher(config)
    dispatcher.initialize_cudagraph_keys(CUDAGraphMode.FULL)
    mode, batch = dispatcher.dispatch(second.total_num_scheduled_tokens)
    assert mode == CUDAGraphMode.FULL and batch.num_tokens == 64
    assert config.compilation_config.cudagraph_capture_sizes == [64, 128, 256]


def test_native_dummy_positional_profile_flag_does_not_admit_workers(graph_runtime):
    events, context, runner, _wrapper = graph_runtime
    runner.profile_run()
    context.batch_descriptor = descriptor()
    events.clear()
    assert runner._dummy_run(1, True) == "eager"
    assert events == [("eager", True)]
    assert not graph.is_graph_profile() and not graph._SYNTHETIC.get()


@pytest.mark.parametrize("native_mode", [CUDAGraphMode.NONE, CUDAGraphMode.FULL])
def test_capture_scope_allows_native_eager_warmup_but_rejects_invalid_full_capture(
    graph_runtime, native_mode
):
    events, context, runner, _wrapper = graph_runtime
    runner.profile_run()
    context.batch_descriptor = descriptor(num_tokens=3, num_reqs=3)
    context.cudagraph_runtime_mode = native_mode
    runner.capture_callback = lambda: runner._dummy_run(3)
    events.clear()
    if native_mode == CUDAGraphMode.FULL:
        with pytest.raises(ValueError, match="no eligible native FULL graph"):
            runner._capture_cudagraphs([context.batch_descriptor], CUDAGraphMode.FULL)
        assert ("eager", False) not in events
    else:
        assert runner._capture_cudagraphs([context.batch_descriptor], CUDAGraphMode.FULL) == "eager"
        assert events == [("eager", False)]
    assert not graph._CAPTURE.get() and not graph._SYNTHETIC.get()


@pytest.mark.parametrize("graph_runtime", ["v4-full"], indirect=True)
def test_profile_metadata_failure_poisons_without_releasing_attention(graph_runtime):
    events, _context, runner, wrapper = graph_runtime
    runner.profile_run()
    session = wrapper._expertkit_decode_session

    def fail_metadata():
        raise RuntimeError("metadata allocation failed")

    session._v4_attention.prepare_profile_metadata = fail_metadata
    with pytest.raises(RuntimeError, match="metadata allocation failed"):
        runner.profile_run()
    assert session._poisoned
    session.close()
    assert "adapter-close" not in events and "data-close" not in events


def test_failed_hccl_constructor_retains_unpublished_resources(graph_runtime, monkeypatch):
    import gc
    import weakref

    _events, _context, runner, _wrapper = graph_runtime
    retained, references = [], []
    monkeypatch.setattr(graph, "retain_until_process_exit", retained.append)

    class Resource:
        pass

    class FailedData:
        def __init__(self, *args, **kwargs):
            self.resource = Resource()
            references.append(weakref.ref(self.resource))
            raise RuntimeError("partially created HCCL")

    monkeypatch.setattr(graph, "HcclGraphSession", FailedData)
    config, topology, _deadline, _control = graph._resolve_graph_config(None, None)

    def fail():
        with pytest.raises(RuntimeError, match="partially created HCCL"):
            graph.DecodeGraphSession(config, runner.device, topology=topology)

    fail()
    gc.collect()
    assert references[0]() is not None
    assert retained[0]._startup_failure is not None and retained[0]._closed
    assert not retained[0]._watchdog.is_alive()


@pytest.mark.parametrize("graph_runtime", ["decode", "v4-full"], indirect=True)
def test_profile_consumers_share_one_resolved_runtime_for_entire_lifetime(graph_runtime):
    events, _context, runner, wrapper = graph_runtime
    runner.profile_run()
    session = wrapper._expertkit_decode_session
    runner.profile_run()
    assert events.count("resolve-runtime") == 1
    assert session.config is session._control.config is session._data.config
    if session._v4_attention is not None:
        assert session._v4_attention.graph_config is session.config


@pytest.mark.parametrize("change", ["decode_descriptor", "padded_total"])
def test_host_phase_does_not_bypass_native_descriptor_eligibility(graph_runtime, change):
    events, context, runner, _wrapper = graph_runtime
    runner.profile_run()
    context.batch_descriptor = descriptor(uniform=change != "decode_descriptor")
    scheduled = {"request": 1 if change == "decode_descriptor" else 3}
    events.clear()
    with pytest.raises(
        ValueError, match=r"no eligible native FULL graph|Host scheduled token total"
    ):
        runner.execute_model(SimpleNamespace(num_scheduled_tokens=scheduled))
    assert events == []


@pytest.mark.parametrize("graph_runtime", ["v4-full"], indirect=True)
def test_attention_setup_failure_releases_bootstrap_before_prepare(graph_runtime, monkeypatch):
    events, _context, runner, _wrapper = graph_runtime
    config, topology, deadline, _control = graph._resolve_graph_config(None, None)
    control = SimpleNamespace(close=lambda: events.append("reservation-release"))
    monkeypatch.setattr(
        graph, "_resolve_graph_config", lambda *_: (config, topology, deadline, control)
    )

    def fail(*_):
        raise RuntimeError("attention setup failed")

    monkeypatch.setattr(
        sys.modules["expertkit_vllm.v4_full_attention"], "install_v4_full_attention", fail
    )
    with pytest.raises(RuntimeError, match="attention setup failed"):
        runner.profile_run()
    assert events.count("reservation-release") == 1
    assert "topology-close" in events
    assert not any(
        isinstance(event, tuple) and event[:2] == ("begin", "prepare") for event in events
    )
