"""CPU checks for ModelSlim graph operator inputs and shared resident routing."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as functional

from expertkit_worker.backends.torch.modelslim_decode_graph import (
    ResidentModelSlimExpertGraphWeights,
)
from expertkit_worker.backends.torch.modelslim_w8a8 import TorchModelSlimW8A8Weights
from expertkit_worker.backends.torch.weights import TorchExpertWeights
from expertkit_worker.weights import ReadyWeightTable


def quantize(inputs):
    """Provide deterministic row quantization at the external operator boundary."""
    scale = inputs.float().abs().amax(dim=-1).clamp(min=1e-6) / 127
    return (inputs.float() / scale.unsqueeze(-1)).round().to(torch.int8), scale


def projection(inputs, matrix, scale):
    """Compute exact integer dots before the fake operator's FP32 dequantization."""
    quantized, token_scale = quantize(inputs)
    dots = quantized.long() @ matrix.long()
    return ((scale * token_scale.unsqueeze(1)) * dots.float()).to(inputs.dtype)


class CpuModelSlimOperators:
    """Model the device boundary without claiming torch-npu numerical qualification."""

    def __init__(self):
        self.calls = []

    def npu_moe_init_routing_v2(self, hidden, ids, **options):
        order = ids.flatten().argsort(stable=True)
        counts = torch.bincount(ids.flatten().long(), minlength=options["expert_num"])
        indices = order
        if options["row_idx_type"] == 0:
            indices = torch.empty_like(order, dtype=torch.int32)
            indices[order] = torch.arange(order.numel(), dtype=torch.int32)
        return hidden[order // ids.shape[1]], indices, counts, None

    def npu_dynamic_quant(self, inputs, *, dst_type):
        assert dst_type is torch.int8
        return quantize(inputs)

    def npu_grouped_matmul(self, inputs, weights, **options):
        self.calls.append((weights, options))
        assert inputs[0].dtype is torch.int8
        assert options["output_dtype"] is torch.int32
        assert options["split_item"] == 3
        assert "scale" not in options and "per_token_scale" not in options
        assert options["group_list_type"] == (0 if len(weights) == 1 else 1)
        counts = options["group_list"].tolist()
        outputs, start = [], 0
        for matrix, count in zip(weights, counts, strict=True):
            assert matrix.ndim == (3 if len(weights) == 1 else 2)
            if len(weights) == 1:
                assert matrix.shape[0] == 1
                matrix = matrix[0]
            end = start + count
            dots = inputs[0][start:end].long() @ matrix.long()
            outputs.append(dots.to(torch.int32))
            start = end
        assert start == inputs[0].shape[0]
        return [torch.cat(outputs)]


def cpu_ordered_combine(
    down, ids, routing, inverse, offset, count, output, *, expert_start, expert_end
):
    """Model the independent kernel boundary with stable expert-order FP32 sums."""
    accumulator = torch.zeros_like(output, dtype=torch.float32)
    for row in range(ids.shape[0]):
        order = sorted(range(ids.shape[1]), key=lambda column: (ids[row, column].item(), column))
        for column in order:
            expert = ids[row, column].item()
            position = int(inverse[row * ids.shape[1] + column] - offset)
            if expert_start <= expert < expert_end and 0 <= position < count:
                accumulator[row] += down[position].float() * routing[row, column]
    output.copy_(accumulator.bfloat16())


@pytest.mark.parametrize("direct_combine", [False, True])
@pytest.mark.parametrize("expert_range", [(0, 2), (2, 3), (0, 16), (48, 64), (112, 128)])
def test_original_i8_storage_and_lossless_scales_follow_shared_routing(
    expert_range, direct_combine
):
    generator = torch.Generator().manual_seed(61)
    experts_per_layer = 128 if expert_range[1] - expert_range[0] == 16 else 3
    table = ReadyWeightTable(1, experts_per_layer)
    weights = {}
    for expert in range(*expert_range):
        weight = TorchModelSlimW8A8Weights(
            tuple(
                torch.randint(-8, 8, shape, generator=generator, dtype=torch.int8)
                for shape in ((4, 6), (4, 6), (6, 4))
            ),
            tuple(
                (torch.rand(width, generator=generator) / 8 + 0.001).bfloat16().float()
                for width in (6, 6, 4)
            ),
            torch.bfloat16,
        )
        table.publish(0, expert, weight)
        weights[expert] = weight
    operators = CpuModelSlimOperators()
    resident = ResidentModelSlimExpertGraphWeights(
        layer_ids=(0,),
        experts_per_layer=experts_per_layer,
        hidden_dim=4,
        intermediate_dim=6,
        top_k=2,
        acquire_many=table.acquire_many,
        ready_revision=lambda: table.revision,
        operators=operators,
        expert_range=expert_range,
    )
    if direct_combine:
        resident._ordered_combine = cpu_ordered_combine
    try:
        pairs = ((0, 1), (2, 2), (1, 2), (2, 1))
        if experts_per_layer == 128:
            start, end = expert_range
            nonlocal_id = 127 if start == 0 else start - 1
            pairs = (
                (start, end - 1),
                (nonlocal_id, nonlocal_id),
                (nonlocal_id, start),
                (end - 1, end % 128),
            )
        for pair in pairs:
            hidden = torch.randn(3, 4, generator=generator).bfloat16()
            hidden[0].zero_()
            ids = torch.tensor([pair] * 3, dtype=torch.int32)
            routing = torch.tensor([[0.0, 0.25], [0.5, 0.75], [1.0, 0.125]])
            output = torch.full((3, 4), float("nan"))
            resident.compute(0, hidden, ids, routing, output)
            reference = torch.zeros_like(output)
            for row in range(3):
                for column in sorted(range(2), key=lambda column: (pair[column], column)):
                    expert = pair[column]
                    if expert not in weights:
                        continue
                    weight = weights[expert]
                    gate, up = (
                        projection(hidden[row : row + 1], weight.matrices[i], weight.scales[i])
                        for i in (0, 1)
                    )
                    down = projection(
                        functional.silu(gate) * up, weight.matrices[2], weight.scales[2]
                    )
                    reference[row] += down[0].float() * routing[row, column]
            torch.testing.assert_close(output, reference.bfloat16().float(), rtol=0, atol=0)
        for call, (matrices, _options) in enumerate(operators.calls):
            for index, weight in enumerate(weights.values()):
                matrix = matrices[index]
                original = weight.matrices[call % 3]
                assert matrix is operators.calls[call % 3][0][index]
                assert matrix.data_ptr() == original.data_ptr()
                assert matrix.untyped_storage().data_ptr() == original.untyped_storage().data_ptr()
                assert matrix.storage_offset() == original.storage_offset()
                if len(weights) == 1:
                    assert matrix.shape == (1, *original.shape)
                else:
                    assert matrix is original
                lease = next(iter(resident._layers.values()))
                assert torch.equal(
                    resident._fp32_scales[lease][call % 3][index], weight.scales[call % 3]
                )
        assert table.usage_count(0, expert_range[0]) == 1
        assert resident._scales
        assert resident._matrices
        assert resident._fp32_scales
    finally:
        resident.close(wait_for_completion=lambda: None)
    assert table.usage_count(0, expert_range[0]) == 0
    assert not resident._scales
    assert not resident._matrices
    assert not resident._fp32_scales
    assert resident._ordered_combine is None
    with pytest.raises(RuntimeError, match="closed or quarantined"):
        resident.compute(0, hidden, ids, routing, output)


def test_modelslim_graph_rejects_float_ready_weights_and_releases_lease():
    table = ReadyWeightTable(1, 1)
    table.publish(0, 0, TorchExpertWeights(*(torch.ones(4, 4).bfloat16() for _ in range(3))))
    with pytest.raises(ValueError, match="ModelSlim I8"):
        ResidentModelSlimExpertGraphWeights(
            layer_ids=(0,),
            experts_per_layer=1,
            hidden_dim=4,
            intermediate_dim=4,
            top_k=1,
            acquire_many=table.acquire_many,
            ready_revision=lambda: table.revision,
            operators=CpuModelSlimOperators(),
        )
    assert table.usage_count(0, 0) == 0


def single_expert_table(scale=1.0):
    """Create one ready expert for preparation and completion failure checks."""
    table = ReadyWeightTable(1, 1)
    table.publish(
        0,
        0,
        TorchModelSlimW8A8Weights(
            tuple(torch.ones(4, 4, dtype=torch.int8) for _ in range(3)),
            tuple(torch.full((4,), scale) for _ in range(3)),
            torch.bfloat16,
        ),
    )
    return table


def single_expert_resident(table):
    """Prepare one resident expert through the production lifecycle."""
    return ResidentModelSlimExpertGraphWeights(
        layer_ids=(0,),
        experts_per_layer=1,
        hidden_dim=4,
        intermediate_dim=4,
        top_k=1,
        acquire_many=table.acquire_many,
        ready_revision=lambda: table.revision,
        operators=CpuModelSlimOperators(),
    )


def test_nonrepresentable_fp32_scale_rejects_preparation_and_releases_lease():
    table = single_expert_table(1.001)
    with pytest.raises(ValueError, match="preserve every FP32 bit"):
        single_expert_resident(table)
    assert table.usage_count(0, 0) == 0
    assert not table.quarantined


@pytest.mark.parametrize("failure_phase", ["prepare", "completion"])
def test_device_failure_retains_prepared_scales_with_original_lease(monkeypatch, failure_phase):
    retained = []
    monkeypatch.setattr(
        "expertkit_worker.backends.torch.decode_graph.retain_until_process_exit", retained.append
    )
    table = single_expert_table()
    if failure_phase == "prepare":
        original_to = torch.Tensor.to
        conversions = 0

        def fail_second_conversion(tensor, *args, **kwargs):
            nonlocal conversions
            if args == (torch.bfloat16,) and tensor.dtype is torch.float32:
                conversions += 1
                if conversions == 2:
                    raise RuntimeError("device conversion failed")
            return original_to(tensor, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, "to", fail_second_conversion)
        with pytest.raises(RuntimeError, match="device conversion failed"):
            single_expert_resident(table)
        resident = retained[0]
        assert sum(len(scales) for layer in resident._scales.values() for scales in layer) == 1
    else:
        resident = single_expert_resident(table)

        def failed_completion():
            raise RuntimeError("device completion failed")

        with pytest.raises(RuntimeError, match="device completion failed"):
            resident.close(wait_for_completion=failed_completion)
        assert retained == [resident]
        assert len(next(iter(resident._fp32_scales.values()))) == 3
    prepared = [
        scale for layer in resident._scales.values() for scales in layer for scale in scales
    ]
    matrices = [
        matrix
        for layer in resident._matrices.values()
        for projection in layer
        for matrix in projection
    ]
    assert prepared
    assert matrices
    assert table.quarantined
    assert table.usage_count(0, 0) == 1
    resident.close(wait_for_completion=lambda: None)
    assert prepared == [
        scale for layer in resident._scales.values() for scales in layer for scale in scales
    ]
    assert matrices == [
        matrix
        for layer in resident._matrices.values()
        for projection in layer
        for matrix in projection
    ]
    assert table.usage_count(0, 0) == 1


def v4_case(experts=3, expert_range=(0, 2), scale=1.0):
    """Keep original FP32 scales that preserve all bits through BF16."""
    table = ReadyWeightTable(1, experts)
    generator = torch.Generator().manual_seed(81)
    weights = {}
    for expert in range(*expert_range):
        weight = TorchModelSlimW8A8Weights(
            tuple(
                torch.randint(-2, 3, shape, generator=generator, dtype=torch.int8)
                for shape in ((4, 6), (4, 6), (6, 4))
            ),
            tuple(
                torch.full((width,), scale + expert / 1000).bfloat16().float()
                for width in (6, 6, 4)
            ),
            torch.bfloat16,
        )
        table.publish(0, expert, weight)
        weights[expert] = weight
    operators = CpuModelSlimOperators()
    return table, operators, weights


def v4_resident(table, operators, experts=3, expert_range=(0, 2)):
    """Construct the candidate directly without opening Worker factory admission."""
    return ResidentModelSlimExpertGraphWeights(
        layer_ids=(0,),
        experts_per_layer=experts,
        hidden_dim=4,
        intermediate_dim=6,
        top_k=2,
        acquire_many=table.acquire_many,
        ready_revision=lambda: table.revision,
        operators=operators,
        expert_range=expert_range,
        expert_compute="deepseek_v4",
        swiglu_limit=10.0,
    )


@pytest.mark.parametrize("experts,expert_range", [(3, (0, 2)), (256, (224, 256))])
def test_v4_lossless_scales_route_before_down_and_mask_nonlocal_tail(experts, expert_range):
    table, operators, weights = v4_case(experts, expert_range)
    resident = v4_resident(table, operators, experts, expert_range)
    start, end = expert_range
    nonlocal_id = 0 if start else experts - 1
    hidden = torch.tensor([[16, -8, 4, 2], [0, 0, 0, 0], [-12, 2, 4, 8]], dtype=torch.bfloat16)
    routing = torch.tensor(
        [[0.1234567, 0.7654321], [0.25, 0.75], [1.25, 0.125]], dtype=torch.float32
    )
    states = [
        [[start, start + 1], [nonlocal_id, start], [start + 1, nonlocal_id]],
        [[nonlocal_id, nonlocal_id]] * 3,
        [[start + 1, start + 1]] * 3,
    ]

    def linear(x, weight, index):
        return projection(x, weight.matrices[index], weight.scales[index])

    try:
        for state in states:
            ids = torch.tensor(state, dtype=torch.int32)
            output = torch.full((3, 4), float("nan"))
            local_count = int(((ids >= start) & (ids < end)).sum())
            original = operators.npu_dynamic_quant

            def corrupt_tail(inputs, original=original, local_count=local_count, **options):
                result = original(inputs, **options)
                if len(operators.calls) % 3 == 2:
                    # Invalid padding must be masked after the BF16 projection.
                    result[1][local_count:] = float("nan")
                return result

            operators.npu_dynamic_quant = corrupt_tail
            try:
                resident.compute(0, hidden, ids, routing, output)
            finally:
                operators.npu_dynamic_quant = original
            reference = torch.zeros((3, 4))
            for row in range(3):
                for column in range(2):
                    weight = weights.get(int(ids[row, column]))
                    if weight is None:
                        continue
                    gate = linear(hidden[row : row + 1], weight, 0).float().clamp(max=10)
                    up = linear(hidden[row : row + 1], weight, 1).float().clamp(-10, 10)
                    intermediate = (functional.silu(gate) * up * routing[row, column]).bfloat16()
                    reference[row] += linear(intermediate, weight, 2)[0].float()
            torch.testing.assert_close(output, reference.bfloat16().float(), rtol=0, atol=0)
            assert torch.isfinite(output).all()
            for call, (matrices, _) in enumerate(operators.calls):
                assert all(
                    matrix is weight.matrices[call % 3]
                    for matrix, weight in zip(matrices, weights.values(), strict=True)
                )
        lease = next(iter(resident._layers.values()))
        for index, scales in enumerate(resident._scales[lease]):
            for converted, weight in zip(scales, weights.values(), strict=True):
                assert converted.dtype is torch.bfloat16
                assert torch.equal(
                    converted.float().view(torch.int32), weight.scales[index].view(torch.int32)
                )
        assert table.usage_count(0, start) == 1
    finally:
        resident.close(wait_for_completion=lambda: None)
    assert not resident._scales and not resident._matrices and not resident._fp32_scales
    assert table.usage_count(0, start) == 0


@pytest.mark.parametrize("failure_phase", ["invalid", "nonrepresentable", "prepare", "completion"])
def test_v4_lossless_scale_failure_follows_shared_lease_lifetime(monkeypatch, failure_phase):
    retained = []
    monkeypatch.setattr(
        "expertkit_worker.backends.torch.decode_graph.retain_until_process_exit", retained.append
    )
    table, operators, weights = v4_case(scale=-1 if failure_phase == "invalid" else 1.0)
    if failure_phase == "nonrepresentable":
        weights[0].scales[0][0] = 1.001
    if failure_phase in ("invalid", "nonrepresentable"):
        with pytest.raises(ValueError, match=r"finite positive|preserve every FP32 bit"):
            v4_resident(table, operators)
        assert table.usage_count(0, 0) == 0 and not table.quarantined
        assert not operators.calls
        return
    if failure_phase == "prepare":
        original = torch.Tensor.to
        calls = 0

        def fail_second_conversion(tensor, *args, **kwargs):
            nonlocal calls
            if args == (torch.bfloat16,) and tensor.dtype is torch.float32:
                calls += 1
                if calls == 2:
                    raise RuntimeError("device scale conversion failed")
            return original(tensor, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, "to", fail_second_conversion)
        with pytest.raises(RuntimeError, match="scale conversion failed"):
            v4_resident(table, operators)
        resident = retained[0]
        assert sum(len(scales) for layer in resident._scales.values() for scales in layer) == 1
    else:
        resident = v4_resident(table, operators)

        def failed_completion():
            raise RuntimeError("device completion failed")

        with pytest.raises(RuntimeError, match="completion failed"):
            resident.close(wait_for_completion=failed_completion)
    prepared = [
        value for layer in resident._scales.values() for scales in layer for value in scales
    ]
    assert retained == [resident] and table.quarantined
    assert table.usage_count(0, 0) == 1 and prepared
    resident.close(wait_for_completion=lambda: None)
    assert all(
        left is right
        for left, right in zip(
            prepared,
            [value for layer in resident._scales.values() for scales in layer for value in scales],
            strict=True,
        )
    )
    assert table.usage_count(0, 0) == 1


@pytest.mark.parametrize("capacity", [1, 3])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32])
def test_fp32_transport_preserves_ordinary_bf16_local_output(capacity, output_dtype):
    table = single_expert_table()
    resident = single_expert_resident(table)
    hidden = torch.ones(capacity, 4, dtype=torch.bfloat16)
    ids = torch.zeros(capacity, 1, dtype=torch.int32)
    routing = torch.full((capacity, 1), 0.1234567)
    output = torch.full((capacity, 4), float("nan"), dtype=output_dtype)
    pointer = output.data_ptr()
    weight = next(iter(resident._layers.values())).objects[0]
    gate, up = (
        projection(hidden, weight.matrices[index], weight.scales[index]) for index in (0, 1)
    )
    down = projection(functional.silu(gate) * up, weight.matrices[2], weight.scales[2])
    unrounded = down.float() * routing
    expected = unrounded.bfloat16().to(output_dtype)
    assert not torch.equal(unrounded, unrounded.bfloat16().float())
    try:
        resident.compute(0, hidden, ids, routing, output)
        assert output.data_ptr() == pointer
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
    finally:
        resident.close(wait_for_completion=lambda: None)


@pytest.mark.parametrize(
    "output",
    [
        torch.empty(1, 5),
        torch.empty(1, 4, dtype=torch.float16),
        torch.empty(1, 8)[:, ::2],
        torch.empty(1, 4, device="meta"),
    ],
)
def test_fp32_transport_validates_original_destination_before_projection(output):
    table = single_expert_table()
    resident = single_expert_resident(table)
    try:
        with pytest.raises(ValueError, match=r"nonempty|BF16|contiguous"):
            resident.compute(
                0,
                torch.ones(1, 4, dtype=torch.bfloat16),
                torch.zeros(1, 1, dtype=torch.int32),
                torch.ones(1, 1),
                output,
            )
        assert not resident._operators.calls
    finally:
        resident.close(wait_for_completion=lambda: None)


@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32])
def test_local_output_rounding_rejects_work_after_close(output_dtype):
    resident = single_expert_resident(single_expert_table())
    resident.close(wait_for_completion=lambda: None)
    with pytest.raises(RuntimeError, match="closed"):
        resident.compute(
            0,
            torch.ones(1, 4, dtype=torch.bfloat16),
            torch.zeros(1, 1, dtype=torch.int32),
            torch.ones(1, 1),
            torch.empty(1, 4, dtype=output_dtype),
        )
    assert not resident._operators.calls


def test_local_sum_follows_expert_order_when_index_add_uses_another_valid_order(monkeypatch):
    # These real layer13 contributions straddle a BF16 midpoint after FP32 folding.
    contributions = torch.tensor(
        [
            0.003758637933060527,
            -0.004008466377854347,
            0.0040187956765294075,
            -0.019419409334659576,
            0.0009282701648771763,
            -0.04179638251662254,
        ]
    )
    table = ReadyWeightTable(1, 6)
    weight = TorchModelSlimW8A8Weights(
        tuple(torch.ones(4, 4, dtype=torch.int8) for _ in range(3)),
        tuple(torch.ones(4) for _ in range(3)),
        torch.bfloat16,
    )
    for expert in range(6):
        table.publish(0, expert, weight)

    class Operators(CpuModelSlimOperators):
        def npu_dynamic_quant(self, inputs, **options):
            quantized, scale = super().npu_dynamic_quant(inputs, **options)
            return quantized, torch.ones_like(scale) if len(self.calls) % 3 == 2 else scale

        def npu_grouped_matmul(self, inputs, weights, **options):
            result = super().npu_grouped_matmul(inputs, weights, **options)
            if len(self.calls) % 3 == 0:
                return [contributions.sign().int().unsqueeze(1).expand(-1, 4).clone()]
            return result

    resident = ResidentModelSlimExpertGraphWeights(
        layer_ids=(0,),
        experts_per_layer=6,
        hidden_dim=4,
        intermediate_dim=4,
        top_k=6,
        acquire_many=table.acquire_many,
        ready_revision=lambda: table.revision,
        operators=Operators(),
    )
    original = torch.Tensor.index_add_
    calls = []

    def alternate_order(destination, dim, indices, source, **kwargs):
        calls.append(source)
        order = torch.tensor([0, 1, 2, 3, 5, 4])
        return original(destination, dim, indices[order], source[order], **kwargs)

    monkeypatch.setattr(torch.Tensor, "index_add_", alternate_order)
    try:
        output = torch.empty(1, 4)
        resident.compute(
            0,
            torch.ones(1, 4, dtype=torch.bfloat16),
            torch.arange(6, dtype=torch.int32).reshape(1, 6),
            contributions.abs().reshape(1, 6),
            output,
        )
        expected = torch.zeros(())
        for contribution in contributions:
            expected.add_(contribution)
        assert torch.equal(output, expected.bfloat16().float().expand(1, 4))
        assert not calls
    finally:
        resident.close(wait_for_completion=lambda: None)


@pytest.mark.parametrize("capacity", [1, 3, 64])
@pytest.mark.parametrize("expert_range", [(0, 6), (0, 3), (3, 6)])
def test_ordered_local_sum_maps_global_assignments_and_ignores_nan_padding(capacity, expert_range):
    table = ReadyWeightTable(1, 6)
    weight = TorchModelSlimW8A8Weights(
        tuple(torch.ones(4, 4, dtype=torch.int8) for _ in range(3)),
        tuple(torch.ones(4) for _ in range(3)),
        torch.bfloat16,
    )
    for expert in range(*expert_range):
        table.publish(0, expert, weight)
    resident = ResidentModelSlimExpertGraphWeights(
        layer_ids=(0,),
        experts_per_layer=6,
        hidden_dim=4,
        intermediate_dim=4,
        top_k=3,
        acquire_many=table.acquire_many,
        ready_revision=lambda: table.revision,
        operators=CpuModelSlimOperators(),
        expert_range=expert_range,
    )
    ids = torch.tensor([[5, 5, 5], [5, 2, 2], [0, 4, 1]] * capacity, dtype=torch.int32)[:capacity]
    full = torch.tensor([2**25, -(2**25), 1.0]).repeat(capacity).unsqueeze(1).expand(-1, 4).clone()
    full[-3:] = 0.0
    # Equal-expert packed rows need not retain their original column order.
    global_rows = torch.cat(
        [(ids.flatten() == expert).nonzero().flatten().flip(0) for expert in range(6)]
    )
    counts = torch.bincount(ids.flatten().long(), minlength=6)
    offset = counts[: expert_range[0]].sum()
    local_count = counts[expert_range[0] : expert_range[1]].sum()
    positions = (torch.arange(ids.numel()) + offset).clamp(max=ids.numel() - 1)
    packed = full[global_rows][positions]
    packed[local_count:] = float("nan")
    expected = torch.zeros(capacity, 4)
    for token in range(capacity):
        for column in sorted(range(3), key=lambda column: (int(ids[token, column]), column)):
            if expert_range[0] <= ids[token, column] < expert_range[1]:
                expected[token].add_(full[token * 3 + column])
    try:
        result = resident._accumulate(
            torch.zeros(capacity, 4, dtype=torch.bfloat16),
            global_rows[positions] // 3,
            packed,
            expert_ids=ids,
            original_rows=global_rows,
            local_offset=offset,
            local_count=local_count,
        )
        assert result.dtype is torch.float32 and torch.isfinite(result).all()
        assert torch.equal(result, expected)
    finally:
        resident.close(wait_for_completion=lambda: None)


def test_scale_first_preserves_real_quantized_projection_below_bf16_midpoint():
    # This real layer5 dot and scale pair separates the two FP32 evaluation orders.
    token_scale = torch.tensor([0.004336860030889511])
    channel_scale = 27 / 32768
    dot = torch.tensor([[17695] * 4], dtype=torch.int32)

    class Operators(CpuModelSlimOperators):
        def npu_dynamic_quant(self, inputs, **options):
            return torch.ones_like(inputs, dtype=torch.int8), token_scale

        def npu_grouped_matmul(self, inputs, weights, **options):
            self.calls.append((weights, options))
            assert len(weights) == 1 and options["group_list_type"] == 0
            if options["output_dtype"] == torch.int32:
                assert options["split_item"] == 3
                assert "scale" not in options and "per_token_scale" not in options
                return [dot]
            return [(dot.float() * channel_scale * token_scale[:, None]).bfloat16()]

    table = single_expert_table(channel_scale)
    operators = Operators()
    resident = ResidentModelSlimExpertGraphWeights(
        layer_ids=(0,),
        experts_per_layer=1,
        hidden_dim=4,
        intermediate_dim=4,
        top_k=1,
        acquire_many=table.acquire_many,
        ready_revision=lambda: table.revision,
        operators=operators,
    )
    try:
        lease = next(iter(resident._layers.values()))
        actual = resident._project(
            torch.ones(1, 4, dtype=torch.bfloat16), lease, 2, torch.tensor([1])
        )
        assert torch.equal(actual, torch.full((1, 4), 0.06298828125, dtype=torch.bfloat16))
        assert not torch.equal(
            actual, (dot.float() * channel_scale * token_scale[:, None]).bfloat16()
        )
    finally:
        resident.close(wait_for_completion=lambda: None)


@pytest.mark.parametrize("counts", [[0, 2, 0, 3, 0], [0, 0, 0, 0, 5], [5]])
def test_int32_projection_maps_zero_counts_and_charged_padding(counts):
    table = ReadyWeightTable(1, len(counts))
    for expert in range(len(counts)):
        table.publish(
            0,
            expert,
            TorchModelSlimW8A8Weights(
                tuple(torch.ones(4, 4, dtype=torch.int8) for _ in range(3)),
                tuple(torch.full((4,), float(expert + 1)) for _ in range(3)),
                torch.bfloat16,
            ),
        )
    resident = ResidentModelSlimExpertGraphWeights(
        layer_ids=(0,),
        experts_per_layer=len(counts),
        hidden_dim=4,
        intermediate_dim=4,
        top_k=1,
        acquire_many=table.acquire_many,
        ready_revision=lambda: table.revision,
        operators=CpuModelSlimOperators(),
    )
    try:
        lease = next(iter(resident._layers.values()))
        inputs = torch.arange(1, 6).bfloat16().unsqueeze(1).expand(-1, 4).contiguous()
        output = resident._project(inputs, lease, 2, torch.tensor(counts))
        scales = torch.repeat_interleave(torch.arange(1, len(counts) + 1), torch.tensor(counts))
        expected = inputs.float() * 4 * scales[:, None]
        assert torch.equal(output, expected.bfloat16())
    finally:
        resident.close(wait_for_completion=lambda: None)


def test_fp32_scale_table_failure_retains_published_tables_and_lease(monkeypatch):
    retained = []
    monkeypatch.setattr(
        "expertkit_worker.backends.torch.decode_graph.retain_until_process_exit", retained.append
    )
    original = torch.stack
    calls = 0

    def fail_second_table(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("FP32 table allocation failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "stack", fail_second_table)
    table = single_expert_table()
    with pytest.raises(RuntimeError, match="FP32 table allocation failed"):
        single_expert_resident(table)
    resident = retained[0]
    assert sum(len(values) for values in resident._fp32_scales.values()) == 1
    assert table.quarantined and table.usage_count(0, 0) == 1
    resident.close(wait_for_completion=lambda: None)
    assert sum(len(values) for values in resident._fp32_scales.values()) == 1


@pytest.mark.parametrize("missing", ["triton", "ascend"])
def test_optional_ordered_combine_reports_missing_ascend_dependency(monkeypatch, missing):
    table = single_expert_table()
    resident = single_expert_resident(table)
    if missing == "triton":
        monkeypatch.setitem(sys.modules, "triton", None)
        monkeypatch.setitem(sys.modules, "triton.backends", None)
    else:
        monkeypatch.setitem(sys.modules, "triton.backends", SimpleNamespace(backends={}))
    try:
        with pytest.raises(RuntimeError, match="requires Triton-Ascend with the Ascend backend"):
            resident._prepare_ordered_combine()
        assert resident._combine_temporaries == []
    finally:
        resident.close(wait_for_completion=lambda: None)


def test_ordered_combine_precompile_failure_quarantines_allocations_and_lease(monkeypatch):
    retained = []
    monkeypatch.setattr(
        "expertkit_worker.backends.torch.decode_graph.retain_until_process_exit", retained.append
    )

    class NpuDevice:
        type = "npu"

        def __eq__(self, other):
            return other is self or other == torch.device("cpu")

    device = NpuDevice()
    monkeypatch.setattr(TorchModelSlimW8A8Weights, "device", property(lambda self: device))
    monkeypatch.setitem(sys.modules, "torch_npu", CpuModelSlimOperators())
    monkeypatch.setitem(
        sys.modules, "triton.backends", SimpleNamespace(backends={"ascend": object()})
    )

    def failed_compile(*args, **kwargs):
        raise RuntimeError("ordered combine compilation failed")

    monkeypatch.setitem(
        sys.modules,
        "expertkit_worker.backends.torch.ascend_moe_ops",
        SimpleNamespace(ordered_combine=failed_compile),
    )
    original_empty = torch.empty

    def allocate_on_cpu(*args, **kwargs):
        if kwargs.get("device") is device:
            kwargs["device"] = "cpu"
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", allocate_on_cpu)
    gate = torch.zeros(2048, 768, dtype=torch.int8)
    down = torch.zeros(768, 2048, dtype=torch.int8)
    weight = TorchModelSlimW8A8Weights(
        (gate, gate, down),
        (torch.ones(768), torch.ones(768), torch.ones(2048)),
        torch.bfloat16,
    )
    table = ReadyWeightTable(1, 128)
    for expert in range(64):
        table.publish(0, expert, weight)
    with pytest.raises(RuntimeError, match="ordered combine compilation failed"):
        ResidentModelSlimExpertGraphWeights(
            layer_ids=(0,),
            experts_per_layer=128,
            hidden_dim=2048,
            intermediate_dim=768,
            top_k=8,
            acquire_many=table.acquire_many,
            ready_revision=lambda: table.revision,
            expert_range=(0, 64),
        )
    resident = retained[0]
    assert resident._quarantined and table.quarantined
    assert table.usage_count(0, 0) == table.usage_count(0, 63) == 1
    assert len(resident._combine_temporaries) == 7
    resident.close(wait_for_completion=lambda: None)
    assert len(resident._combine_temporaries) == 7
