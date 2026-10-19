"""Resident graph computation and weight lifetime tests on a CPU operator boundary."""

from __future__ import annotations

import threading

import pytest
import torch
import torch.nn.functional as functional

from expertkit_worker.backends.torch.decode_graph import ResidentExpertGraphWeights
from expertkit_worker.backends.torch.weights import TorchExpertWeights
from expertkit_worker.weights import ReadyWeightTable, WeightsNotReady


class CpuExpertOperators:
    """Implement the external device-operator contract with explicit CPU grouping."""

    def __init__(self) -> None:
        self.weight_addresses: list[list[int]] = []
        self.counts: list[list[int]] = []
        self.gmm_counts: list[list[int]] = []

    def npu_moe_init_routing_v2(self, hidden_states, expert_ids, **options):
        assert options["row_idx_type"] == 1
        assert options["active_num"] == expert_ids.numel()
        assert bool(((expert_ids >= 0) & (expert_ids < options["expert_num"])).all())
        original_rows = expert_ids.flatten().argsort(stable=True)
        token_indices = original_rows // expert_ids.shape[1]
        counts = torch.bincount(expert_ids.flatten().long(), minlength=options["expert_num"])
        self.counts.append(counts.tolist())
        return hidden_states[token_indices], original_rows, counts, None

    def npu_grouped_matmul(self, inputs, weights, **options):
        assert options["split_item"] == 2
        assert options["group_type"] == 0
        assert options["group_list_type"] == 1
        self.weight_addresses.append([weight.data_ptr() for weight in weights])
        self.gmm_counts.append(options["group_list"].tolist())
        output = []
        start = 0
        for weight, count in zip(weights, options["group_list"].tolist(), strict=True):
            output.append(inputs[0][start : start + count] @ weight)
            start += count
        assert start == inputs[0].shape[0]
        return [torch.cat(output)]


def resident_case(layer_ids=(0, 1)):
    """Create three resident experts including one with no selected assignments."""
    table = ReadyWeightTable(2, 3)
    generator = torch.Generator().manual_seed(914)
    weights = {}
    for layer_id in layer_ids:
        for expert_id in range(3):
            weight = TorchExpertWeights(
                *(
                    torch.randn(shape, generator=generator).to(torch.bfloat16) / 4
                    for shape in [(6, 4), (6, 4), (4, 6)]
                )
            )
            table.publish(layer_id, expert_id, weight)
            weights[layer_id, expert_id] = weight
    operators = CpuExpertOperators()
    resident = ResidentExpertGraphWeights(
        layer_ids=layer_ids,
        experts_per_layer=3,
        hidden_dim=4,
        intermediate_dim=6,
        top_k=2,
        acquire_many=table.acquire_many,
        ready_revision=lambda: table.revision,
        operators=operators,
    )
    return table, operators, resident, weights


def reference(hidden, ids, routing, weights, layer_id):
    """Project each assignment independently and weight its BF16 output in FP32."""
    result = torch.zeros_like(hidden, dtype=torch.float32)
    for token in range(hidden.shape[0]):
        for column in range(ids.shape[1]):
            weight = weights[layer_id, ids[token, column].item()]
            x = hidden[token : token + 1]
            gate = functional.linear(x, weight.gate_proj)
            up = functional.linear(x, weight.up_proj)
            down = functional.linear(functional.silu(gate) * up, weight.down_proj)
            result[token] += down[0].float() * routing[token, column]
    return result.to(torch.bfloat16)


@pytest.mark.parametrize("capacity", [1, 3, 32])
def test_dynamic_routing_zero_count_and_padded_rows_match_fp32_reference(capacity):
    table, operators, resident, weights = resident_case()
    for step in range(3):
        hidden = (torch.arange(capacity * 4).reshape(capacity, 4) / 64 + step / 8).to(
            torch.bfloat16
        )
        pairs = [([0, 1],), ([1, 2],), ([0, 2], [1, 2])][step]
        ids = torch.tensor([pairs[row % len(pairs)] for row in range(capacity)], dtype=torch.int32)
        routing = torch.tensor([[0.1234567, 0.7654321]] * capacity, dtype=torch.float32)
        output = torch.empty_like(hidden)
        resident.compute(1, hidden, ids, routing, output)
        torch.testing.assert_close(
            output, reference(hidden, ids, routing, weights, 1), rtol=0, atol=0
        )
        if step < 2:
            assert operators.counts[-1][2 if step == 0 else 0] == 0
    assert operators.counts[0] != operators.counts[1]
    expected = [weights[1, expert].gate_proj.data_ptr() for expert in range(3)]
    assert operators.weight_addresses[0] == expected
    assert all(operators.weight_addresses[step * 3] == expected for step in range(3))
    assert table.usage_count(1, 2) == 1
    resident.close(wait_for_completion=lambda: None)
    assert table.usage_count(1, 2) == 0


def test_withdrawal_invalidates_before_wait_and_completion_precedes_release():
    table, _, resident, _ = resident_case()
    before = resident.revision
    table.begin_withdrawal(1, 2)
    assert table.revision > before
    with pytest.raises(RuntimeError, match="invalidated"):
        resident.require_current()
    waiting = threading.Event()
    completed = threading.Event()

    def withdraw():
        waiting.set()
        table.finish_withdrawal(1, 2, timeout=2)
        completed.set()

    thread = threading.Thread(target=withdraw)
    thread.start()
    assert waiting.wait(2)

    def finish_device():
        assert table.usage_count(1, 2) == 1
        assert not completed.is_set()

    resident.close(wait_for_completion=finish_device)
    assert completed.wait(2)
    thread.join(timeout=2)
    assert not thread.is_alive()
    with pytest.raises(RuntimeError, match="closed"):
        resident.require_current()


def test_failed_completion_quarantines_original_weights():
    table, _, resident, _ = resident_case()

    def failed_device():
        raise RuntimeError("device failed")

    with pytest.raises(RuntimeError, match="device failed"):
        resident.close(wait_for_completion=failed_device)
    assert table.quarantined
    assert table.usage_count(1, 2) == 1
    with pytest.raises(RuntimeError, match="quarantined"):
        resident.require_current()


def test_partial_prepare_failure_releases_previously_acquired_layers():
    table = ReadyWeightTable(2, 1)
    weight = TorchExpertWeights(*(torch.ones(4, 4, dtype=torch.bfloat16) for _ in range(3)))
    table.publish(0, 0, weight)
    with pytest.raises(WeightsNotReady):
        ResidentExpertGraphWeights(
            layer_ids=(0, 1),
            experts_per_layer=1,
            hidden_dim=4,
            intermediate_dim=4,
            top_k=1,
            acquire_many=table.acquire_many,
            ready_revision=lambda: table.revision,
            operators=CpuExpertOperators(),
        )
    assert table.usage_count(0, 0) == 0


@pytest.mark.parametrize("capacity,dtype", [(0, torch.bfloat16), (1, torch.float32)])
def test_invalid_input_metadata_rejected_without_device_operators(capacity, dtype):
    _, operators, resident, _ = resident_case()
    hidden = torch.empty(capacity, 4, dtype=dtype)
    with pytest.raises(ValueError, match=r"nonempty|BF16"):
        resident.compute(
            0,
            hidden,
            torch.zeros(capacity, 2, dtype=torch.int32),
            torch.zeros(capacity, 2),
            torch.empty_like(hidden),
        )
    assert not operators.counts
    resident.close(wait_for_completion=lambda: None)


def test_withdrawal_during_prepare_releases_every_acquired_layer():
    table, _, previous, _ = resident_case()
    previous.close(wait_for_completion=lambda: None)

    def racing_acquire(layer_id, expert_ids):
        lease = table.acquire_many(layer_id, expert_ids)
        if layer_id == 1:
            table.begin_withdrawal(0, 2)
        return lease

    with pytest.raises(RuntimeError, match="invalidated"):
        ResidentExpertGraphWeights(
            layer_ids=(0, 1),
            experts_per_layer=3,
            hidden_dim=4,
            intermediate_dim=6,
            top_k=2,
            acquire_many=racing_acquire,
            ready_revision=lambda: table.revision,
            operators=CpuExpertOperators(),
        )
    assert table.usage_count(0, 2) == 0
    assert table.usage_count(1, 2) == 0


@pytest.mark.parametrize("capacity", [1, 3, 32])
def test_sharded_weights_cover_global_routes_and_mask_padding_before_weighting(capacity):
    generator = torch.Generator().manual_seed(43)
    tables = [ReadyWeightTable(1, 6), ReadyWeightTable(1, 6)]
    weights = {}
    residents = []
    operators = []
    for rank, (start, end) in enumerate(((0, 3), (3, 6))):
        for expert in range(start, end):
            weight = TorchExpertWeights(
                *(
                    torch.randn(shape, generator=generator).bfloat16() / 4
                    for shape in [(6, 4), (6, 4), (4, 6)]
                )
            )
            weights[0, expert] = weight
            tables[rank].publish(0, expert, weight)
        op = CpuExpertOperators()
        operators.append(op)
        residents.append(
            ResidentExpertGraphWeights(
                layer_ids=(0,),
                experts_per_layer=6,
                hidden_dim=4,
                intermediate_dim=6,
                top_k=2,
                acquire_many=tables[rank].acquire_many,
                ready_revision=lambda table=tables[rank]: table.revision,
                operators=op,
                expert_range=(start, end),
            )
        )
    try:
        for pair in ([0, 2], [3, 5], [0, 5], [2, 3]):
            hidden = torch.randn(capacity, 4, generator=generator).bfloat16()
            ids = torch.tensor([pair] * capacity, dtype=torch.int32)
            routing = torch.tensor([[0.1234567, 0.7654321]] * capacity)
            partials = [torch.full((capacity, 4), float("nan")) for _ in residents]
            for rank, (resident, op, partial) in enumerate(
                zip(residents, operators, partials, strict=True)
            ):
                local_count = int(((ids >= rank * 3) & (ids < (rank + 1) * 3)).sum())
                original = op.npu_grouped_matmul
                calls = [0]

                def poison_padding(
                    inputs,
                    matrices,
                    original=original,
                    calls=calls,
                    local_count=local_count,
                    **options,
                ):
                    result = original(inputs, matrices, **options)
                    calls[0] += 1
                    if calls[0] == 3:
                        result[0][local_count:] = float("nan")
                    return result

                op.npu_grouped_matmul = poison_padding
                try:
                    resident.compute(0, hidden, ids, routing, partial)
                finally:
                    op.npu_grouped_matmul = original
                assert torch.isfinite(partial).all()
                assert all(sum(counts) == capacity * 2 for counts in op.gmm_counts)
                assert all(len(addresses) == 3 for addresses in op.weight_addresses)
                if local_count == 0:
                    assert torch.count_nonzero(partial) == 0
                    assert op.gmm_counts[-1] == [0, 0, capacity * 2]
            combined = (partials[0] + partials[1]).bfloat16()
            torch.testing.assert_close(
                combined, reference(hidden, ids, routing, weights, 0), rtol=0, atol=0
            )
        for rank, table in enumerate(tables):
            assert [table.usage_count(0, expert) for expert in range(6)] == (
                [1, 1, 1, 0, 0, 0] if rank == 0 else [0, 0, 0, 1, 1, 1]
            )
    finally:
        for resident in residents:
            resident.close(wait_for_completion=lambda: None)


@pytest.mark.parametrize(
    "expert_compute,swiglu_limit,experts,expert_range,error",
    [
        ("deepseek_v4", 10.0, 2, (0, 2), "ModelSlim"),
        ("deepseek_v4", 0.0, 2, (0, 2), "limit"),
        ("swiglu", 10.0, 2, (0, 2), "limit"),
        ("swiglu", 0.0, 256, (224, 256), "128"),
        ("deepseek_v4", 10.0, 256, (0, 129), "128"),
        ("deepseek_v4", 10.0, 257, (224, 256), "256"),
        ("deepseek_v4", 10.0, 256, (255, 256), "two"),
    ],
)
def test_recipe_and_local_group_limits_reject_before_computation(
    expert_compute, swiglu_limit, experts, expert_range, error
):
    table = ReadyWeightTable(1, experts)
    for expert in range(*expert_range):
        table.publish(
            0, expert, TorchExpertWeights(*(torch.ones(4, 4).bfloat16() for _ in range(3)))
        )
    with pytest.raises(ValueError, match=error):
        ResidentExpertGraphWeights(
            layer_ids=(0,),
            experts_per_layer=experts,
            hidden_dim=4,
            intermediate_dim=4,
            top_k=2,
            acquire_many=table.acquire_many,
            ready_revision=lambda: table.revision,
            operators=CpuExpertOperators(),
            expert_range=expert_range,
            expert_compute=expert_compute,
            swiglu_limit=swiglu_limit,
        )
    assert all(table.usage_count(0, expert) == 0 for expert in range(*expert_range))
