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
        singleton = len(weights) == 1
        assert options["group_list_type"] == (0 if singleton else 1)
        assert all(weight.ndim == (3 if singleton else 2) for weight in weights)
        self.weight_addresses.append([weight.data_ptr() for weight in weights])
        self.gmm_counts.append(options["group_list"].tolist())
        output = []
        start = 0
        for weight, count in zip(weights, options["group_list"].tolist(), strict=True):
            if singleton:
                assert weight.shape[0] == 1
                weight = weight[0]
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
        ("deepseek_v4", 0.0, 2, (0, 2), "ModelSlim"),
        ("swiglu", 10.0, 2, (0, 2), "limit"),
        ("deepseek_v4", 10.0, 256, (0, 129), "128"),
        ("deepseek_v4", 10.0, 257, (224, 256), "ModelSlim"),
        ("deepseek_v4", 10.0, 256, (255, 256), "ModelSlim"),
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


def test_per_layer_noncontiguous_and_empty_shards_keep_exact_weights_and_fresh_output():
    generator = torch.Generator().manual_seed(72)
    table = ReadyWeightTable(3, 5)
    weights = {}
    for layer in range(3):
        for expert in range(5):
            weight = TorchExpertWeights(
                *(
                    (torch.randn(shape, generator=generator) / 4).bfloat16()
                    for shape in ((6, 4), (6, 4), (4, 6))
                )
            )
            table.publish(layer, expert, weight)
            weights[layer, expert] = weight
    # Mapping insertion order is unrelated to the model's execution order.
    selection = {2: (1,), 1: (0, 2, 4), 0: ()}
    operators = CpuExpertOperators()
    resident = ResidentExpertGraphWeights(
        layer_ids=(0, 1, 2),
        experts_per_layer=5,
        hidden_dim=4,
        intermediate_dim=6,
        top_k=2,
        acquire_many=table.acquire_many,
        ready_revision=lambda: table.revision,
        expert_ids_by_layer=selection,
        operators=operators,
    )
    hidden = torch.randn(4, 4, generator=generator).bfloat16()
    ids = torch.tensor([[4, 0], [1, 3], [2, 2], [3, 4]], dtype=torch.int32)
    routing = torch.tensor([[0.125, 0.75]] * 4)
    output = torch.full((4, 4), float("nan"))
    try:
        assert resident.layer_ids == (0, 1, 2)
        assert resident._routing_maps[2] is None
        for layer in resident.layer_ids:
            assert resident._layers[layer].expert_ids == selection[layer]
            for expert in range(5):
                assert table.usage_count(layer, expert) == int(expert in selection[layer])
            for stale in (float("nan"), 17.0):
                output.fill_(stale)
                before = len(operators.counts)
                resident.compute(layer, hidden, ids, routing, output)
                local_routing = routing.clone()
                for row in range(4):
                    for column in range(2):
                        if int(ids[row, column]) not in selection[layer]:
                            local_routing[row, column] = 0
                torch.testing.assert_close(
                    output.bfloat16(),
                    reference(hidden, ids, local_routing, weights, layer),
                    rtol=0,
                    atol=0,
                )
                if not selection[layer]:
                    assert len(operators.counts) == before
                    assert torch.equal(output, torch.zeros_like(output))
    finally:
        resident.close(wait_for_completion=lambda: None)
    assert all(table.usage_count(layer, expert) == 0 for layer in range(3) for expert in range(5))


@pytest.mark.parametrize("per_layer", [False, True])
def test_unrelated_ready_changes_do_not_change_handshake_or_invalidate_selected_leases(per_layer):
    table = ReadyWeightTable(2, 4)
    weight = TorchExpertWeights(*(torch.ones(4, 4).bfloat16() for _ in range(3)))
    table.publish(0, 0, weight)
    table.publish(0, 1, weight)
    resident = ResidentExpertGraphWeights(
        layer_ids=(0,),
        experts_per_layer=4,
        hidden_dim=4,
        intermediate_dim=4,
        top_k=2,
        acquire_many=table.acquire_many,
        ready_revision=lambda: table.revision,
        **({"expert_ids_by_layer": {0: (0, 1)}} if per_layer else {"expert_range": (0, 2)}),
        operators=CpuExpertOperators(),
    )
    frozen_revision = resident.revision
    try:
        table.publish(1, 3, weight)
        resident.require_current()
        table.begin_withdrawal(1, 3)
        resident.require_current()
        table.finish_withdrawal(1, 3, timeout=0)
        table.publish(1, 3, weight)
        resident.require_current()
        assert resident.revision == frozen_revision < table.revision
        table.begin_withdrawal(0, 1)
        with pytest.raises(RuntimeError, match="invalidated"):
            resident.require_current()
        table.publish(1, 0, weight)
        with pytest.raises(RuntimeError, match="invalidated"):
            resident.require_current()
    finally:
        resident.close(wait_for_completion=lambda: None)


@pytest.mark.parametrize(
    "selection,error",
    [
        ({0: (0,)}, "exactly"),
        ({False: (0,), 1: ()}, "exactly"),
        ({0: (), 1: (0,), 2: ()}, "exactly"),
        ({0: (), 1: ()}, "at least one"),
        ({0: (2, 0), 1: ()}, "sorted"),
        ({0: (0, 0), 1: ()}, "sorted"),
        ({0: [0, 1], 1: ()}, "tuples"),
        ({0: (True,), 1: ()}, "integer"),
        ({0: (1.0,), 1: ()}, "integer"),
        ({0: (-1,), 1: ()}, "integer"),
        ({0: (3,), 1: ()}, "integer"),
    ],
)
def test_per_layer_expert_metadata_rejected_before_acquisition(selection, error):
    from expertkit_worker.backends.graph_common import ResidentGraphProfile

    with pytest.raises(ValueError, match=error):
        ResidentGraphProfile((0, 1), 3, 4, 6, 2, expert_ids_by_layer=selection)


def test_per_layer_profile_freezes_mapping_and_retains_kernel_group_limits():
    from expertkit_worker.backends.graph_common import ResidentGraphProfile

    selection = {1: (), 0: (0, 2)}
    profile = ResidentGraphProfile((0, 1), 3, 4, 6, 2, expert_ids_by_layer=selection)
    selection[0] = (1,)
    assert tuple(profile.expert_ids_by_layer) == (0, 1)
    assert profile.expert_ids(0) == (0, 2)
    with pytest.raises(TypeError):
        profile.expert_ids_by_layer[0] = (1,)
    with pytest.raises(ValueError, match="mutually exclusive"):
        ResidentGraphProfile((0,), 3, 4, 6, 2, (0, 2), expert_ids_by_layer={0: (0, 2)})
    with pytest.raises(ValueError, match="128 local"):
        ResidentGraphProfile(
            (0,),
            256,
            4,
            6,
            2,
            expert_compute="deepseek_v4",
            swiglu_limit=10,
            expert_ids_by_layer={0: tuple(range(129))},
        )
    singleton = ResidentGraphProfile(
        (0, 1),
        256,
        4,
        6,
        2,
        expert_compute="deepseek_v4",
        swiglu_limit=10,
        expert_ids_by_layer={0: (), 1: (255,)},
    )
    assert singleton.expert_ids(0) == () and singleton.expert_ids(1) == (255,)


def test_acquisition_cannot_substitute_a_different_ready_expert_list():
    table = ReadyWeightTable(1, 3)
    weight = TorchExpertWeights(*(torch.ones(4, 4).bfloat16() for _ in range(3)))
    table.publish(0, 0, weight)
    table.publish(0, 1, weight)

    def wrong_experts(layer, requested):
        assert requested == (0, 2)
        return table.acquire_many(layer, (0, 1))

    with pytest.raises(ValueError, match="exact per-layer"):
        ResidentExpertGraphWeights(
            layer_ids=(0,),
            experts_per_layer=3,
            hidden_dim=4,
            intermediate_dim=4,
            top_k=2,
            acquire_many=wrong_experts,
            ready_revision=lambda: table.revision,
            expert_ids_by_layer={0: (0, 2)},
            operators=CpuExpertOperators(),
        )
    assert table.usage_count(0, 0) == table.usage_count(0, 1) == 0


def test_routing_map_prepare_failure_quarantines_leases_and_prepared_maps(monkeypatch):
    from expertkit_worker.backends.graph_common import ResidentGraphProfile

    table = ReadyWeightTable(2, 3)
    weight = TorchExpertWeights(*(torch.ones(4, 4).bfloat16() for _ in range(3)))
    for layer in range(2):
        for expert in (0, 2):
            table.publish(layer, expert, weight)
    original = ResidentGraphProfile.routing_map
    retained = []

    def fail_later(profile, layer, device):
        if layer == 1:
            raise RuntimeError("mapping device failure")
        return original(profile, layer, device)

    monkeypatch.setattr(ResidentGraphProfile, "routing_map", fail_later)
    monkeypatch.setattr(
        "expertkit_worker.backends.torch.decode_graph.retain_until_process_exit", retained.append
    )
    with pytest.raises(RuntimeError, match="mapping device failure"):
        ResidentExpertGraphWeights(
            layer_ids=(0, 1),
            experts_per_layer=3,
            hidden_dim=4,
            intermediate_dim=4,
            top_k=2,
            acquire_many=table.acquire_many,
            ready_revision=lambda: table.revision,
            expert_ids_by_layer={0: (0, 2), 1: (0, 2)},
            operators=CpuExpertOperators(),
        )
    assert table.quarantined
    assert all(table.usage_count(layer, expert) == 1 for layer in range(2) for expert in (0, 2))
    assert len(retained) == 1
    assert retained[0]._routing_maps[0] is not None


@pytest.mark.parametrize("layers", [(1, 0), (0, 0)])
def test_graph_profile_rejects_nonascending_or_duplicate_layers_without_sorting(layers):
    from expertkit_worker.backends.graph_common import ResidentGraphProfile

    with pytest.raises(ValueError, match="sorted, unique"):
        ResidentGraphProfile(layers, 3, 4, 6, 2, expert_ids_by_layer={0: (0,), 1: (1,)})


@pytest.mark.parametrize("global_experts", [129, 257, 10240])
def test_sparse_global_routing_uses_native_bound_not_model_experiment_count(global_experts):
    table = ReadyWeightTable(1, global_experts)
    weight = TorchExpertWeights(*(torch.ones(4, 4).bfloat16() / 8 for _ in range(3)))
    selected = (0, global_experts - 1)
    for expert in selected:
        table.publish(0, expert, weight)
    operators = CpuExpertOperators()
    provider = ResidentExpertGraphWeights(
        layer_ids=(0,),
        experts_per_layer=global_experts,
        hidden_dim=4,
        intermediate_dim=4,
        top_k=2,
        acquire_many=table.acquire_many,
        ready_revision=lambda: table.revision,
        expert_ids_by_layer={0: selected},
        operators=operators,
    )
    hidden = torch.arange(16).reshape(4, 4).bfloat16() / 16
    ids = torch.tensor([[selected[1], 0], [1, 1], [0, 0], [1, selected[1]]], dtype=torch.int32)
    routing = torch.tensor([[0.125, 0.75]] * 4)
    output = torch.full_like(hidden, float("nan"), dtype=torch.float32)
    local_mask = (ids == 0) | (ids == selected[1])
    try:
        provider.compute(0, hidden, ids, routing, output)
        expected = reference(
            hidden,
            ids.masked_fill(~local_mask, 0),
            routing.masked_fill(~local_mask, 0),
            {(0, expert): weight for expert in selected},
            0,
        )
        torch.testing.assert_close(output.bfloat16(), expected, rtol=0, atol=0)
        assert len(operators.counts[-1]) == global_experts
        assert all(len(counts) == 2 for counts in operators.gmm_counts)
    finally:
        provider.close(wait_for_completion=lambda: None)


def test_torch_global_routing_bound_rejects_before_weight_acquisition():
    def unexpected_acquisition(*args):
        raise AssertionError("routing limit must be checked before weight acquisition")

    with pytest.raises(ValueError, match="10240"):
        ResidentExpertGraphWeights(
            layer_ids=(0,),
            experts_per_layer=10241,
            hidden_dim=4,
            intermediate_dim=4,
            top_k=2,
            acquire_many=unexpected_acquisition,
            ready_revision=lambda: 0,
            expert_ids_by_layer={0: (0, 10240)},
            operators=CpuExpertOperators(),
        )


@pytest.mark.parametrize(
    "recipe,clip,top_k,error",
    [
        ("deepseek_v4", -1.0, 2, "limit"),
        ("deepseek_v4", float("nan"), 2, "limit"),
        ("deepseek_v4", float("inf"), 2, "limit"),
        ("deepseek_v4", float("-inf"), 2, "limit"),
        ("unknown", 0.0, 2, "computation"),
        ("swiglu", 2.5, 2, "limit"),
        ("swiglu", 0.0, 4, "top_k"),
    ],
)
def test_graph_recipe_and_routing_semantic_constraints_remain(recipe, clip, top_k, error):
    from expertkit_worker.backends.graph_common import ResidentGraphProfile

    with pytest.raises(ValueError, match=error):
        ResidentGraphProfile((0,), 3, 4, 6, top_k, expert_compute=recipe, swiglu_limit=clip)
