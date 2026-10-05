"""Idle DP forwards must skip RPC while startup model runs retain it."""

from __future__ import annotations

import asyncio
import inspect
import sys
from contextvars import Context
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn
from vllm import forward_context, platforms
from vllm.model_executor.layers import fused_moe
from vllm.model_executor.layers.fused_moe import layer as fused_moe_layer

from expertkit_vllm import dummy_batch, plugin
from expertkit_vllm.experts import remote_moe


def _forward(layer, hidden_states):
    # Exercise the real operation on CPU regardless of the installed dispatch key.
    return remote_moe._remote_moe_impl(
        hidden_states,
        torch.zeros(hidden_states.shape[0], 2),
        None,
        remote_moe._encode_layer_name(layer.layer_name),
    )


def _remote_layer(monkeypatch, *, shared: bool, scaling: float):
    calls = []

    class Selector:
        def select_experts(self, hidden_states, router_logits, *, input_ids):
            calls.append("route")
            rows = hidden_states.shape[0]
            return torch.ones(rows, 1), torch.zeros(rows, 1, dtype=torch.int32)

    class Shared(nn.Module):
        def forward(self, hidden_states):
            calls.append("shared")
            return hidden_states * 3, None

    class Client:
        def execute(self, **kwargs):
            calls.append("execute")
            return torch.full_like(kwargs["hidden_states"], 2)

    def client_for(*args):
        calls.append("client")
        return Client()

    monkeypatch.setattr(remote_moe, "_client_for", client_for)
    layer = remote_moe.RemoteMoERunner.__new__(remote_moe.RemoteMoERunner)
    nn.Module.__init__(layer)
    layer.layer_name = "layers.0.mlp.experts"
    layer._layer_id = 0
    layer.num_experts = 2
    layer.gate = None
    layer.expert_selector = Selector()
    layer._shared_experts_module = Shared() if shared else None
    layer.routed_scaling_factor = scaling
    layer.client_config = SimpleNamespace(timeout_seconds=1)
    return layer, calls


def _register_dummy_worker(monkeypatch, *, device_type, layer, hidden_states):
    runs = []
    context = forward_context.ForwardContext(
        no_compile_layers={layer.layer_name: layer},
        attn_metadata=None,
        slot_mapping={},
    )

    class ModelRunner:
        uniform_decode_query_len = hidden_states.shape[0]

        def _dummy_run(self, num_tokens, **kwargs):
            runs.append((num_tokens, kwargs))
            context.attn_metadata = {} if kwargs.get("force_attention") else None
            with forward_context.override_forward_context(context):
                output = _forward(layer, hidden_states)
            return output, output

    class Worker:
        def __init__(self):
            self.model_runner = ModelRunner()

        def execute_dummy_batch(self):
            num_tokens = getattr(self.model_runner, "uniform_decode_query_len", 1)
            self.model_runner._dummy_run(num_tokens, uniform_decode=True)

    module_name, class_name = (
        ("vllm_ascend.worker.worker", "NPUWorker")
        if device_type == "npu"
        else ("vllm.v1.worker.gpu_worker", "Worker")
    )
    module = ModuleType(module_name)
    setattr(module, class_name, Worker)
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setattr(
        platforms,
        "current_platform",
        SimpleNamespace(device_type=device_type, pre_register_and_update=lambda: None),
    )
    monkeypatch.setattr(fused_moe, "FusedMoE", fused_moe.FusedMoE)
    monkeypatch.setattr(fused_moe_layer, "FusedMoE", fused_moe_layer.FusedMoE)
    monkeypatch.setenv("EK_ENABLE", "1")
    plugin.register()
    return Worker(), runs, context


@pytest.mark.parametrize("device_type", ["cuda", "npu", "cpu"])
def test_idle_dummy_batch_preserves_runner_and_never_dispatches(monkeypatch, device_type):
    layer, calls = _remote_layer(monkeypatch, shared=False, scaling=1)
    hidden_states = torch.ones(2, 4)
    worker, runs, context = _register_dummy_worker(
        monkeypatch, device_type=device_type, layer=layer, hidden_states=hidden_states
    )

    assert worker.execute_dummy_batch() is None

    assert runs == [(2, {"uniform_decode": True})]
    assert not calls
    torch.testing.assert_close(hidden_states, torch.zeros_like(hidden_states))

    context.attn_metadata = None
    with forward_context.override_forward_context(context):
        _forward(layer, hidden_states)
    assert calls == ["route", "client", "execute"]
    torch.testing.assert_close(hidden_states, torch.full_like(hidden_states, 2))


@pytest.mark.parametrize("device_type", ["cuda", "npu"])
@pytest.mark.parametrize(
    "options",
    [{}, {"is_profile": True}, {"is_graph_capturing": True, "force_attention": True}],
)
def test_startup_dummy_runs_still_dispatch(monkeypatch, device_type, options):
    layer, calls = _remote_layer(monkeypatch, shared=False, scaling=1)
    hidden_states = torch.ones(2, 4)
    worker, runs, _ = _register_dummy_worker(
        monkeypatch, device_type=device_type, layer=layer, hidden_states=hidden_states
    )

    output, last_output = worker.model_runner._dummy_run(2, **options)

    assert runs == [(2, options)]
    assert calls == ["route", "client", "execute"]
    assert output is last_output is hidden_states
    torch.testing.assert_close(output, torch.full_like(output, 2))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scaling", [1.0, 2.0])
def test_dummy_forward_retains_shared_experts_and_scaling(monkeypatch, dtype, scaling):
    layer, calls = _remote_layer(monkeypatch, shared=True, scaling=scaling)
    hidden_states = torch.ones(2, 4, dtype=dtype)
    worker, _, context = _register_dummy_worker(
        monkeypatch, device_type="cuda", layer=layer, hidden_states=hidden_states
    )

    worker.execute_dummy_batch()
    output = hidden_states

    assert calls == ["shared"]
    expected_shared = 3 / scaling if dtype == torch.float16 else 3
    assert output is hidden_states
    assert output.shape == (2, 4)
    assert output.dtype == dtype
    torch.testing.assert_close(output, torch.full_like(output, expected_shared))

    calls.clear()
    hidden_states.fill_(1)
    with forward_context.override_forward_context(context):
        output = _forward(layer, hidden_states)
    assert calls == ["route", "client", "execute", "shared"]
    expected = 2 + expected_shared if dtype == torch.float16 else 2 * scaling + expected_shared
    torch.testing.assert_close(output, torch.full_like(output, expected))


@pytest.mark.parametrize("shape", [(0, 4), (1, 1), (4, 8)])
def test_dummy_forward_keeps_input_shape_dtype_and_device(monkeypatch, shape):
    layer, calls = _remote_layer(monkeypatch, shared=False, scaling=2)
    hidden_states = torch.ones(shape, dtype=torch.bfloat16)
    worker, _, _ = _register_dummy_worker(
        monkeypatch, device_type="cuda", layer=layer, hidden_states=hidden_states
    )

    worker.execute_dummy_batch()
    output = hidden_states

    assert not calls
    assert output is hidden_states
    assert output.shape == shape
    assert output.dtype == hidden_states.dtype
    assert output.device == hidden_states.device
    torch.testing.assert_close(output, torch.zeros_like(hidden_states))


def test_guard_restores_nested_context_after_failure_and_is_idempotent(monkeypatch):
    observations = []

    class Worker:
        def execute_dummy_batch(self, *, nested=False):
            observations.append(dummy_batch.is_dummy_forward())
            assert not Context().run(dummy_batch.is_dummy_forward)
            if not nested:
                with pytest.raises(ValueError, match="dummy failure"):
                    self.execute_dummy_batch(nested=True)
                observations.append(dummy_batch.is_dummy_forward())
            raise ValueError("dummy failure")

    module = ModuleType("vllm.v1.worker.gpu_worker")
    module.Worker = Worker
    monkeypatch.setitem(sys.modules, module.__name__, module)
    original = Worker.execute_dummy_batch
    dummy_batch.install_dummy_batch_guard("cpu")
    wrapped = Worker.execute_dummy_batch
    dummy_batch.install_dummy_batch_guard("cpu")
    assert Worker.execute_dummy_batch is wrapped
    assert inspect.signature(wrapped) == inspect.signature(original)

    with pytest.raises(ValueError, match="dummy failure"):
        Worker().execute_dummy_batch()

    assert observations == [True, True, True]
    assert not dummy_batch.is_dummy_forward()


def test_dummy_failure_does_not_mark_other_async_tasks(monkeypatch):
    layer, calls = _remote_layer(monkeypatch, shared=False, scaling=1)
    worker, _, _ = _register_dummy_worker(
        monkeypatch, device_type="cuda", layer=layer, hidden_states=torch.ones(2, 4)
    )

    class Failure(nn.Module):
        def forward(self, hidden_states):
            assert dummy_batch.is_dummy_forward()
            raise ValueError("shared failure")

    layer._shared_experts_module = Failure()

    async def check_other_task():
        assert not dummy_batch.is_dummy_forward()

    async def run():
        other = asyncio.create_task(check_other_task())
        with pytest.raises(ValueError, match="shared failure"):
            worker.execute_dummy_batch()
        assert not dummy_batch.is_dummy_forward()
        await other

    asyncio.run(run())
    assert not calls
