"""Qualify ModelSlim V4 computation through a real gRPC NPU Worker."""

import asyncio
import json
import os
from pathlib import Path

import pytest
import torch
from expertkit_transport.batches import RoutedLayerBatch
from expertkit_transport.buffers import OutputPool
from expertkit_transport.routing import (
    RoundRobinSelector,
    TopologySnapshot,
    WorkerConnection,
    WorkerIdentity,
    execute_routed_layer,
)
from expertkit_transport.transports import WorkerEndpointConfig
from expertkit_transport.transports.base import BatchBufferConfig
from expertkit_transport.transports.grpc import GrpcWorkerBatchReceiver, GrpcWorkerTransport
from safetensors.torch import save

from expertkit_worker.backends.torch import TorchBackend
from expertkit_worker.backends.torch.modelslim_w8a8 import TorchModelSlimW8A8WeightAdapter
from expertkit_worker.execution import WorkerExecutor
from expertkit_worker.factory import _create_device_wiring
from expertkit_worker.weights import ReadyWeightTable, parse_safetensors


class _Topology:
    def __init__(self, snapshot):
        self.snapshot = snapshot

    def current(self, instance_id):
        assert instance_id == self.snapshot.instance_id
        return self.snapshot

    async def refresh(self, *args, **kwargs):
        raise AssertionError("the healthy local Worker must not require topology refresh")


def _synthetic_source():
    tensors = {}
    for projection, sign in (("w1", 1), ("w3", -1), ("w2", 1)):
        prefix = f"layers.0.ffn.experts.0.{projection}"
        matrix = torch.eye(64, dtype=torch.int8) * sign
        if projection == "w1":
            matrix[:32] *= -1
        tensors[f"{prefix}.weight"] = matrix
        tensors[f"{prefix}.weight_scale"] = torch.ones(64, 1)
        tensors[f"{prefix}.weight_offset"] = torch.zeros(64, 1)
    return parse_safetensors(bytearray(save(tensors)))


def _reference(x, ids, routes, matrices, scales):
    """Compute independent FP32 dequantized projections and V4 activation."""
    dequantized = [m.float() * s.float() for m, s in zip(matrices, scales, strict=True)]
    output = torch.zeros_like(x, dtype=torch.float32)
    clipped_gate = clipped_up = negative_gate = 0
    for token in range(x.shape[0]):
        gate = (x[token].float() @ dequantized[0].T).to(x.dtype).float()
        up = (x[token].float() @ dequantized[1].T).to(x.dtype).float()
        clipped_gate += int((gate > 10).sum())
        negative_gate += int((gate < -10).sum())
        clipped_up += int((up.abs() > 10).sum())
        gate = gate.clamp(max=10)
        up = up.clamp(-10, 10)
        for route in range(ids.shape[1]):
            if ids[token, route] < 0:
                continue
            middle = (torch.nn.functional.silu(gate) * up * routes[token, route]).to(x.dtype)
            output[token] += (middle.float() @ dequantized[2].T).to(x.dtype).float()
    return output.to(x.dtype), (clipped_gate, clipped_up, negative_gate)


def _native_reference(x, routes, matrices, scales):
    """Use native dynamic-I8 operators independently of Worker grouping and helpers."""
    import torch_npu

    device_matrices = [matrix.T.contiguous().npu() for matrix in matrices]
    device_scales = [scale.flatten().float().npu() for scale in scales]

    def projection(value, index):
        quantized, token_scales = torch_npu.npu_dynamic_quant(value)
        return torch_npu.npu_quant_matmul(
            quantized,
            device_matrices[index],
            device_scales[index],
            pertoken_scale=token_scales,
            output_dtype=torch.bfloat16,
        )

    values = x.npu()
    gate = projection(values, 0).float().clamp(max=10)
    up = projection(values, 1).float().clamp(-10, 10)
    output = torch.zeros_like(values, dtype=torch.float32)
    for route in range(routes.shape[1]):
        middle = (torch.nn.functional.silu(gate) * up * routes[:, route : route + 1].npu()).to(
            torch.bfloat16
        )
        output += projection(middle, 2).float()
    return output.to(torch.bfloat16).cpu()


@pytest.mark.skipif(os.getenv("EK_TEST_NPU") != "1", reason="requires an assigned Ascend NPU")
@pytest.mark.parametrize("source_kind", ["synthetic", "checkpoint"])
def test_v4_modelslim_grpc_npu(source_kind):
    pytest.importorskip("torch_npu")
    checkpoint = os.getenv("EK_TEST_V4_EXPERT")
    if source_kind == "checkpoint" and not checkpoint:
        pytest.skip("requires a read-only real V4 expert bundle")
    source = (
        parse_safetensors(bytearray(Path(checkpoint).read_bytes()))
        if source_kind == "checkpoint"
        else _synthetic_source()
    )
    hidden = 4096 if source_kind == "checkpoint" else 64
    intermediate = 2048 if source_kind == "checkpoint" else 64
    torch.npu.set_device(0)

    async def scenario():
        wiring = _create_device_wiring("npu:0")
        adapter = TorchModelSlimW8A8WeightAdapter(
            hidden_dim=hidden, intermediate_dim=intermediate, runtime=wiring.runtime
        )
        cpu_weight = adapter.make_cpu_weight(source)
        weight = adapter.make_ready_weight(cpu_weight, layer_id=0, expert_id=0)
        ready = ReadyWeightTable(1, 3)
        ready.publish(0, 0, weight)
        ready.publish(0, 1, weight)
        spec = WorkerEndpointConfig(
            instance_id=7,
            num_layers=1,
            experts_per_layer=3,
            max_batch_tokens=4,
            hidden_dim=hidden,
            top_k=3,
            dtype=torch.bfloat16,
        )
        receiver = GrpcWorkerBatchReceiver(
            "127.0.0.1:0",
            spec,
            max_active_batches=1,
            max_pending_batches=1,
        )
        backend = TorchBackend(
            hidden_dim=hidden,
            intermediate_dim=intermediate,
            top_k=3,
            dtype=torch.bfloat16,
            runtime=wiring.runtime,
            acquire_many=ready.acquire_many,
            expert_compute="deepseek_v4",
            swiglu_limit=10.0,
            linear_compute="modelslim_w8a8_dynamic",
        )
        execution = WorkerExecutor(
            receiver,
            backend,
            create_slot=wiring.create_slot,
            instance_id=7,
            buffer_config=BatchBufferConfig(
                max_batch_tokens=4,
                hidden_dim=hidden,
                top_k=3,
                dtype=torch.bfloat16,
                device=torch.device("npu:0"),
            ),
            slot_count=1,
        )
        await execution.start()
        transport = GrpcWorkerTransport(
            f"127.0.0.1:{receiver.bound_port}",
            spec,
            max_in_flight=2,
            device="cpu",
        )
        pool = OutputPool(
            max_batch_tokens=4,
            hidden_dim=hidden,
            dtype=torch.bfloat16,
            device="cpu",
            capacity=2,
        )
        metrics = []
        try:
            await transport.start()
            identity = WorkerIdentity("v4-npu-test", "isolated-start")
            target = WorkerConnection(
                identity=identity,
                transport=transport,
                max_batch_tokens=4,
                max_active_batches=1,
                max_pending_batches=1,
            )
            topology = _Topology(
                TopologySnapshot(
                    instance_id=7, version=1, routes={(0, 0): (target,), (0, 1): (target,)}
                )
            )
            for label, count, amplitude in (
                ("zero", 1, 0),
                ("one", 1, 1),
                ("multi", 3, 1),
                ("clamp", 3, 100),
            ):
                generator = torch.Generator().manual_seed(21)
                x = (torch.randn(count, hidden, generator=generator) * amplitude).to(torch.bfloat16)
                ids = torch.tensor([[0, 1, 0]] * count, dtype=torch.int32)
                routes = torch.tensor([[0.15, 0.8, 0.55]] * count)
                batch = RoutedLayerBatch(
                    instance_id=7,
                    layer_id=0,
                    hidden_states=x,
                    expert_ids=ids,
                    routing_weights=routes,
                    distinct_expert_ids=(0, 1),
                )
                async with asyncio.timeout(60):
                    result = await execute_routed_layer(
                        batch,
                        topology,
                        RoundRobinSelector(),
                        {identity: pool},
                        monotonic_deadline=asyncio.get_running_loop().time() + 55,
                    )
                reference, clipped = _reference(
                    x, ids, routes, cpu_weight.matrices, cpu_weight.scales
                )
                native = _native_reference(x, routes, cpu_weight.matrices, cpu_weight.scales)
                native_relative_l2 = float(
                    (result.float() - native.float()).norm()
                    / native.float().norm().clamp_min(1e-10)
                )
                relative_l2 = float(
                    (result.float() - reference.float()).norm()
                    / reference.float().norm().clamp_min(1e-10)
                )
                cosine = (
                    float(
                        torch.nn.functional.cosine_similarity(
                            result.float().flatten(), reference.float().flatten(), dim=0
                        )
                    )
                    if amplitude
                    else 1.0
                )
                metrics.append(
                    dict(
                        case=label,
                        tokens=count,
                        relative_l2=relative_l2,
                        native_relative_l2=native_relative_l2,
                        cosine=cosine,
                        clipped_gate=clipped[0],
                        clipped_up=clipped[1],
                        negative_gate=clipped[2],
                    )
                )
                assert torch.isfinite(result).all()
                assert native_relative_l2 < 0.01
                assert relative_l2 < 0.06
                assert cosine > 0.995
                if label == "clamp":
                    assert all(value > 0 for value in clipped)
                if not amplitude:
                    assert torch.count_nonzero(result) == 0
                assert ready.usage_count(0, 0) == ready.usage_count(0, 1) == 0
                assert receiver.active_count == 0
        finally:
            evidence = os.getenv("EK_TEST_EVIDENCE")
            if evidence:
                Path(evidence, f"v4-worker-grpc-{source_kind}.json").write_text(
                    json.dumps(metrics, indent=2)
                )
            await transport.close()
            await pool.close()
            await execution.close()

    asyncio.run(scenario())
