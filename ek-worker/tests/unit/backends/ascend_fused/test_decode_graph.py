"""Resident fused computation follows eager operators, mapping and lease ownership."""

from __future__ import annotations

import pytest
import torch

from expertkit_worker.backends.ascend_fused import op
from expertkit_worker.backends.ascend_fused.decode_graph import (
    ResidentAscendFusedGraphWeights,
    create_modelslim_graph_weights,
)
from expertkit_worker.backends.ascend_fused.op import AscendFusedComputeType, run_ascend_fused_moe
from expertkit_worker.backends.ascend_fused.weights import (
    AscendFusedExpertWeights,
    AscendFusedModelSlimW8A8ExpertWeights,
)
from expertkit_worker.weights import ReadyWeightTable, WeightsNotReady


class CpuFusedOperators:
    """Model native routing and scaled GMM, including unused fixed-shape tail rows."""

    def __init__(self):
        self.routes = []
        self.gmms = []

    def npu_dynamic_quant(self, inputs, *, dst_type):
        assert dst_type is torch.int8
        scale = inputs.float().abs().amax(-1).clamp(min=1e-6) / 127
        return (inputs.float() / scale[:, None]).round().to(torch.int8), scale

    def npu_moe_init_routing_v2(self, hidden, ids, **options):
        self.routes.append(ids.clone())
        local_experts = options["expert_num"]
        flat = ids.flatten().long()
        order = flat.argsort(stable=True)
        inverse = torch.empty_like(order, dtype=torch.int32)
        inverse[order] = torch.arange(order.numel(), dtype=torch.int32)
        counts = torch.bincount(flat, minlength=local_experts + 1)[:local_experts]
        packed = hidden[order // ids.shape[1]].clone()
        packed[int(counts.sum()) :] = 0
        if options["quant_mode"] == 1:
            packed, scale = self.npu_dynamic_quant(packed, dst_type=torch.int8)
        else:
            scale = None
        return packed, inverse, counts, scale

    def npu_grouped_matmul(self, *, x, weight, group_list, **options):
        self.gmms.append((weight, options))
        matrices = [w[0] if w.ndim == 3 else w for w in weight]
        result = torch.zeros(x[0].shape[0], matrices[0].shape[-1], dtype=torch.bfloat16)
        offset = 0
        for i, (matrix, count) in enumerate(zip(matrices, group_list.tolist(), strict=True)):
            rows = slice(offset, offset + count)
            value = x[0][rows].float() @ matrix.float()
            if "scale" in options:
                value = (
                    value
                    * options["per_token_scale"][0][rows, None]
                    * options["scale"][i].flatten()
                )
            result[rows] = value.bfloat16()
            offset += count
        return [result]

    def npu_swiglu(self, inputs, *, dim):
        gate, up = inputs.chunk(2, dim=dim)
        return torch.nn.functional.silu(gate) * up

    def npu_moe_token_unpermute(self, *, permuted_tokens, sorted_indices, probs):
        rows = permuted_tokens[sorted_indices.long()].float()
        weighted = rows * probs.flatten()[:, None].float()
        return weighted.reshape(*probs.shape, -1).sum(1).bfloat16()


@pytest.fixture
def cpu_operators(monkeypatch):
    """Simulate an NPU binding without adding test-only production parameters."""
    operators = CpuFusedOperators()
    monkeypatch.setattr(op, "torch_npu", operators)

    return operators


def weights(
    quantized,
    *,
    device="cpu",
    hidden=4,
    intermediate=6,
    experts=(1, 2),
    layers=1,
    experts_per_layer=4,
):
    table = ReadyWeightTable(layers, experts_per_layer)
    generator = torch.Generator().manual_seed(196)
    for layer in range(layers):
        for expert in experts:
            if quantized:
                weight = AscendFusedModelSlimW8A8ExpertWeights(
                    torch.randint(
                        -6, 7, (hidden, 2 * intermediate), generator=generator, dtype=torch.int8
                    ).to(device),
                    torch.randint(
                        -6, 7, (intermediate, hidden), generator=generator, dtype=torch.int8
                    ).to(device),
                    # Native eager rounds these non-BF16-exact scales too.
                    torch.full((2 * intermediate,), 0.1, device=device),
                    torch.full((hidden,), 0.07, device=device),
                    torch.bfloat16,
                )
            else:
                weight = AscendFusedExpertWeights(
                    (torch.randn(hidden, 2 * intermediate, generator=generator) / 4)
                    .bfloat16()
                    .to(device),
                    (torch.randn(intermediate, hidden, generator=generator) / 4)
                    .bfloat16()
                    .to(device),
                )
            table.publish(layer, expert, weight)
    return table


def resident(
    table,
    quantized,
    *,
    interval=(1, 3),
    layers=(0,),
    v4=False,
    hidden=4,
    intermediate=6,
    experts_per_layer=4,
):
    factory = create_modelslim_graph_weights if quantized else ResidentAscendFusedGraphWeights
    return factory(
        layer_ids=layers,
        experts_per_layer=experts_per_layer,
        hidden_dim=hidden,
        intermediate_dim=intermediate,
        top_k=2,
        expert_range=interval,
        expert_compute="deepseek_v4" if v4 else "swiglu",
        swiglu_limit=10.0 if v4 else 0.0,
        acquire_many=table.acquire_many,
        ready_revision=lambda: table.revision,
    )


def eager_reference(table, quantized, hidden, ids, routing, *, interval=(1, 3), v4=False):
    # Independent host mapping to check the provider's subtraction/range-mask implementation.
    start, end = interval
    local = torch.tensor(
        [[int(e) - start if start <= int(e) < end else end - start for e in row] for row in ids],
        dtype=torch.int32,
        device=ids.device,
    )
    probabilities = routing.bfloat16().clone()
    probabilities[(ids < start) | (ids >= end)] = 0
    lease = table.acquire_many(0, tuple(range(start, end)))
    try:
        return run_ascend_fused_moe(
            hidden,
            local,
            probabilities,
            w13=[w.w13 for w in lease.objects],
            w2=[w.w2 for w in lease.objects],
            compute_type=AscendFusedComputeType.DYNAMIC_W8A8
            if quantized
            else AscendFusedComputeType.UNQUANTIZED,
            w13_scale=[w.w13_scale for w in lease.objects] if quantized else None,
            w2_scale=[w.w2_scale for w in lease.objects] if quantized else None,
            swiglu_limit=10.0 if v4 else 0.0,
        )
    finally:
        lease.close()


@pytest.mark.parametrize("quantized,v4", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("capacity", [1, 4, 32])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32])
def test_resident_matches_eager_with_changing_routes(
    monkeypatch, cpu_operators, quantized, v4, capacity, output_dtype
):
    table = weights(quantized)
    operators = cpu_operators
    provider = resident(table, quantized, v4=v4)
    hidden = torch.linspace(-2, 2, capacity * 4).reshape(capacity, 4).bfloat16()
    routing = torch.tensor([[0.1234567, 0.7654321]] * capacity)
    prepared = provider._prepared[0]
    lists = (prepared.w13, prepared.w2, prepared.w13_scale, prepared.w2_scale)

    def no_search(*args, **kwargs):
        raise AssertionError("contiguous shard mapping must not search expert IDs")

    monkeypatch.setattr(torch, "searchsorted", no_search)
    try:
        for pair in ((1, 2), (3, 0), (2, 2), (0, 1)):
            ids = torch.tensor([pair] * capacity, dtype=torch.int32)
            output = torch.full(hidden.shape, float("nan"), dtype=output_dtype)
            provider.compute(0, hidden, ids, routing, output)
            expected = eager_reference(table, quantized, hidden, ids, routing, v4=v4)
            torch.testing.assert_close(output, expected.to(output_dtype), rtol=0, atol=0)
            if pair == (3, 0):
                assert not output.count_nonzero()
            assert all(
                a is b
                for a, b in zip(
                    lists,
                    (prepared.w13, prepared.w2, prepared.w13_scale, prepared.w2_scale),
                    strict=True,
                )
            )
        assert len(operators.gmms) == 16  # W13 + W2 for four resident and four eager calls.
        assert torch.equal(
            operators.routes[-1], torch.tensor([[2, 0]] * capacity, dtype=torch.int32)
        )
        assert table.usage_count(0, 1) == table.usage_count(0, 2) == 1
    finally:
        provider.close(wait_for_completion=lambda: None)
    assert table.usage_count(0, 1) == table.usage_count(0, 2) == 0
    assert not provider._prepared


@pytest.mark.parametrize("quantized", [False, True])
def test_singleton_shard_and_ready_revision(cpu_operators, quantized):
    table = weights(quantized, experts=(3,))
    provider = resident(table, quantized, interval=(3, 4))
    hidden = torch.ones(3, 4, dtype=torch.bfloat16)
    ids = torch.tensor([[3, 0], [1, 2], [3, 3]], dtype=torch.int32)
    routing = torch.tensor([[0.25, 0.75]] * 3)
    output = torch.empty_like(hidden)
    provider.compute(0, hidden, ids, routing, output)
    expected = eager_reference(table, quantized, hidden, ids, routing, interval=(3, 4))
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    table.begin_withdrawal(0, 3)
    with pytest.raises(RuntimeError, match="invalidated"):
        provider.require_current()
    provider.close(wait_for_completion=lambda: None)
    assert table.usage_count(0, 3) == 0


@pytest.mark.parametrize("quantized", [False, True])
def test_failed_completion_retains_full_provider(monkeypatch, cpu_operators, quantized):
    retained = []
    monkeypatch.setattr(
        "expertkit_worker.backends.ascend_fused.decode_graph.retain_until_process_exit",
        retained.append,
    )
    table = weights(quantized)
    provider = resident(table, quantized)

    def failed():
        raise RuntimeError("completion unknown")

    with pytest.raises(RuntimeError, match="completion unknown"):
        provider.close(wait_for_completion=failed)
    assert retained == [provider]
    assert table.quarantined and table.usage_count(0, 1) == 1
    provider.close(wait_for_completion=lambda: pytest.fail("must not wait twice"))
    assert provider._prepared


def test_partial_acquisition_failure_releases_leases():
    table = weights(False, layers=2)

    def acquire(layer, ids):
        if layer == 1:
            raise WeightsNotReady(ids)
        return table.acquire_many(layer, ids)

    with pytest.raises(WeightsNotReady):
        ResidentAscendFusedGraphWeights(
            layer_ids=(0, 1),
            experts_per_layer=4,
            hidden_dim=4,
            intermediate_dim=6,
            top_k=2,
            expert_range=(1, 3),
            acquire_many=acquire,
            ready_revision=lambda: table.revision,
        )
    assert table.usage_count(0, 1) == 0


def test_scale_preparation_failure_retains_published_allocations(monkeypatch, cpu_operators):
    retained = []
    monkeypatch.setattr(
        "expertkit_worker.backends.ascend_fused.decode_graph.retain_until_process_exit",
        retained.append,
    )
    original = torch.Tensor.to
    calls = 0

    def fail_second_scale(tensor, *args, **kwargs):
        nonlocal calls
        if tensor.dtype == torch.float32 and args == (torch.bfloat16,):
            calls += 1
            if calls == 2:
                raise RuntimeError("scale allocation failed")
        return original(tensor, *args, **kwargs)

    table = weights(True)
    monkeypatch.setattr(torch.Tensor, "to", fail_second_scale)
    with pytest.raises(RuntimeError, match="scale allocation failed"):
        resident(table, True)
    assert table.quarantined and table.usage_count(0, 1) == 1
    assert len(retained[0]._prepared[0].w13_scale) == 1


@pytest.mark.parametrize("wait_first", [False, True])
def test_eager_completion_failure_keeps_original_tensor_owners(wait_first):
    from expertkit_worker.backends.ascend_fused.backend import _AscendFusedCompletion
    from expertkit_worker.backends.base import BackendFatalError

    table = weights(False)
    lease = table.acquire_many(0, (1, 2))
    temporary = torch.ones(3)

    class FailedWork:
        def wait_host(self):
            raise RuntimeError("device failed")

    completion = _AscendFusedCompletion(lease, FailedWork(), (temporary,))
    if wait_first:
        with pytest.raises(BackendFatalError):
            completion.wait_host()
        completion.close()
    else:
        with pytest.raises(BackendFatalError):
            completion.close()
    assert table.quarantined
    assert table.usage_count(0, 1) == 1
    assert completion._retained_tensors[0] is temporary


@pytest.mark.parametrize("quantized", [False, True])
def test_native_capture_replays_changing_routes_like_eager(quantized):
    """Run on an NPU host: one capture, changing routes, duplicates and zero local tokens."""
    npu_ops = pytest.importorskip("torch_npu")
    if not npu_ops.npu.is_available():
        pytest.skip("requires an available Ascend NPU")
    device = torch.device("npu:0")
    npu_ops.npu.set_device(device)
    with torch.inference_mode():
        table = weights(quantized, device=device, hidden=128, intermediate=256)
        provider = resident(table, quantized, hidden=128, intermediate=256)
        hidden = torch.randn(4, 128, device=device, dtype=torch.bfloat16)
        ids = torch.tensor([[1, 2]] * 4, device=device, dtype=torch.int32)
        routing = torch.full((4, 2), 0.5, device=device)
        output = torch.empty_like(hidden)
        stream = npu_ops.npu.Stream(device=device)
        stream.wait_stream(npu_ops.npu.current_stream())
        graph = None
        try:
            with npu_ops.npu.stream(stream):
                for _ in range(3):
                    provider.compute(0, hidden, ids, routing, output)
                stream.synchronize()
                graph = npu_ops.npu.NPUGraph()
                with npu_ops.npu.graph(graph, stream=stream):
                    provider.compute(0, hidden, ids, routing, output)
                for pair in ((1, 2), (3, 0), (2, 2), (0, 1)):
                    ids.copy_(torch.tensor([pair] * 4, device=device, dtype=torch.int32))
                    hidden.add_(0.125)
                    routing.copy_(torch.tensor([[0.1234567, 0.7654321]] * 4, device=device))
                    graph.replay()
                    stream.synchronize()
                    expected = eager_reference(table, quantized, hidden, ids, routing)
                    stream.synchronize()
                    torch.testing.assert_close(output, expected, rtol=0, atol=0)
        except BaseException:
            provider.quarantine()
            raise
        finally:
            if not table.quarantined:
                stream.synchronize()
                graph = None
                provider.close(wait_for_completion=stream.synchronize)


@pytest.mark.parametrize("interval", [(0, 16), (48, 64), (112, 128)])
def test_modelslim_sixteen_expert_shards_mask_global_routes(cpu_operators, interval):
    start, end = interval
    table = weights(True, experts=range(start, end), experts_per_layer=128)
    provider = resident(table, True, interval=interval, experts_per_layer=128)
    hidden = torch.ones(3, 4, dtype=torch.bfloat16)
    routing = torch.tensor([[0.25, 0.75]] * 3)
    outside = 127 if start == 0 else start - 1
    try:
        for pair in ((start, end - 1), (outside, outside), (end - 1, end % 128)):
            ids = torch.tensor([pair] * 3, dtype=torch.int32)
            output = torch.full_like(hidden, float("nan"))
            provider.compute(0, hidden, ids, routing, output)
            expected = eager_reference(table, True, hidden, ids, routing, interval=interval)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            if pair == (outside, outside):
                assert not output.count_nonzero()
        assert table.usage_count(0, start) == table.usage_count(0, end - 1) == 1
    finally:
        provider.close(wait_for_completion=lambda: None)
    assert table.usage_count(0, start) == table.usage_count(0, end - 1) == 0
