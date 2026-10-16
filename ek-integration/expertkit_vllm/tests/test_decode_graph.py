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
def graph_runtime(monkeypatch):
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
                        torch.zeros(1, 2048, dtype=torch.bfloat16),
                        torch.zeros(1, 6, dtype=torch.int32),
                        torch.full((1, 6), 0.1234567, dtype=torch.float32),
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
            "vllm": "0.30.0",
            "torch-npu": "2.10.0.post4",
            "vllm-ascend": "0.1.dev1+g8d4409d62.d20261013",
        }[name],
    )
    monkeypatch.setattr(torch, "__version__", "2.10.0")
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
