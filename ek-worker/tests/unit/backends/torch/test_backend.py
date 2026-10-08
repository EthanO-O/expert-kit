"""Numerical and lifetime tests for the Torch eager Backend."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as functional
from expertkit_transport.tracing import use_tracer

from expertkit_worker.backends import (
    BackendBatch,
    BackendFatalError,
    BackendFatalReason,
    BackendWeightUnavailable,
    InvalidBackendInput,
)
from expertkit_worker.backends.torch import TorchBackend, TorchExpertWeights
from expertkit_worker.device import CpuWorkerRuntime, CudaWorkerRuntime
from expertkit_worker.device.runtime import DeviceWork
from expertkit_worker.weights import ReadyWeightTable

_HIDDEN_DIM = 4
_INTERMEDIATE_DIM = 6
_TOP_K = 3


def make_weight(seed: int, dtype: torch.dtype, device: torch.device) -> TorchExpertWeights:
    """Return a deterministic small expert fixture."""

    generator = torch.Generator(device="cpu").manual_seed(seed)

    def matrix(rows: int, columns: int) -> torch.Tensor:
        value = torch.randn(rows, columns, generator=generator, dtype=torch.float32) / 4
        return value.to(device=device, dtype=dtype)

    return TorchExpertWeights(
        gate_proj=matrix(_INTERMEDIATE_DIM, _HIDDEN_DIM),
        up_proj=matrix(_INTERMEDIATE_DIM, _HIDDEN_DIM),
        down_proj=matrix(_HIDDEN_DIM, _INTERMEDIATE_DIM),
    )


def make_case(
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[BackendBatch, dict[int, TorchExpertWeights]]:
    """Return routing with multiple experts, duplicate Tokens, and invalid slots."""

    hidden = torch.tensor(
        [
            [0.2, -0.4, 0.8, 0.5],
            [-0.1, 0.7, 0.3, -0.6],
            [0.9, 0.1, -0.2, 0.4],
        ],
        dtype=dtype,
        device=device,
    )
    expert_ids = torch.tensor(
        [[0, 2, -1], [1, 0, -1], [2, 1, 0]],
        dtype=torch.int32,
        device=device,
    )
    routing = torch.tensor(
        [[0.35, 0.65, 0.0], [0.8, 0.2, 0.0], [0.15, 0.25, 0.6]],
        dtype=torch.float32,
        device=device,
    )
    batch = BackendBatch(
        layer_id=0,
        hidden_states=hidden,
        expert_ids=expert_ids,
        routing_weights=routing,
        distinct_expert_ids=(0, 1, 2),
    )
    return batch, {expert_id: make_weight(17 + expert_id, dtype, device) for expert_id in range(3)}


def reference_output(
    batch: BackendBatch,
    weights: Mapping[int, TorchExpertWeights],
) -> torch.Tensor:
    """Compute an independent assignment-by-assignment FP32 reference."""

    result = torch.zeros(batch.token_count, batch.hidden_dim, dtype=torch.float32)
    host_hidden = batch.hidden_states.detach().cpu()
    host_ids = batch.expert_ids.detach().cpu()
    host_routing = batch.routing_weights.detach().cpu()
    host_weights = {
        expert_id: tuple(tensor.detach().cpu() for tensor in weight.tensors)
        for expert_id, weight in weights.items()
    }
    for token_index in range(batch.token_count):
        for route_index in range(batch.top_k):
            expert_id = int(host_ids[token_index, route_index])
            if expert_id < 0:
                continue
            gate_weight, up_weight, down_weight = host_weights[expert_id]
            hidden = host_hidden[token_index : token_index + 1]
            gate = hidden @ gate_weight.transpose(0, 1)
            up = hidden @ up_weight.transpose(0, 1)
            output = (functional.silu(gate) * up) @ down_weight.transpose(0, 1)
            result[token_index] += (
                output.float().squeeze(0) * host_routing[token_index, route_index]
            )
    return result.to(batch.hidden_states.dtype)


def make_backend(
    table: ReadyWeightTable[TorchExpertWeights],
    dtype: torch.dtype,
    device: torch.device,
) -> TorchBackend:
    """Build a Backend over the ready-weight table fixture."""

    return TorchBackend(
        hidden_dim=_HIDDEN_DIM,
        intermediate_dim=_INTERMEDIATE_DIM,
        top_k=_TOP_K,
        dtype=dtype,
        runtime=_device_runtime(device),
        acquire_many=table.acquire_many,
    )


def _device_runtime(device: torch.device):
    if device.type == "npu":
        from expertkit_worker.device.ascend import AscendWorkerRuntime

        return AscendWorkerRuntime(device)
    return CpuWorkerRuntime(device) if device.type == "cpu" else CudaWorkerRuntime(device)


def single_token_case(
    dtype: torch.dtype,
    routes: tuple[int, ...] = (2, 0, -1),
) -> tuple[BackendBatch, dict[int, TorchExpertWeights]]:
    """Build a decode batch with route order distinct from weight acquisition order."""
    batch, weights = make_case(dtype, torch.device("cpu"))
    return replace(
        batch,
        hidden_states=batch.hidden_states[:1],
        expert_ids=torch.tensor([routes], dtype=torch.int32),
        routing_weights=torch.tensor(
            [
                [
                    value if expert >= 0 else 0.0
                    for expert, value in zip(routes, (0.2, 0.0, 0.7), strict=True)
                ]
            ],
            dtype=torch.float32,
        ),
        distinct_expert_ids=tuple(sorted({expert for expert in routes if expert >= 0})),
    ), weights


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("routes", [(2, 0, -1), (2, 0, 1), (2, 0, 2)])
def test_single_token_preserves_route_order_weighting_and_weight_lifetime(
    dtype: torch.dtype, routes: tuple[int, ...]
) -> None:
    batch, weights = single_token_case(dtype, routes)
    table: ReadyWeightTable[TorchExpertWeights] = ReadyWeightTable(1, 3)
    for expert_id, weight in weights.items():
        table.publish(0, expert_id, weight)
    backend = make_backend(table, dtype, torch.device("cpu"))
    output = torch.full_like(batch.hidden_states, torch.nan)
    expected = reference_output(batch, weights)

    completion = backend.submit(batch, output)
    assert [table.usage_count(0, i) for i in range(3)] == [
        int(i in batch.distinct_expert_ids) for i in range(3)
    ]
    completion.wait_host()
    torch.testing.assert_close(output, expected)
    completion.close()
    completion.close()
    assert [table.usage_count(0, i) for i in range(3)] == [0, 0, 0]


def test_single_token_avoids_dynamic_indexing() -> None:
    batch, weights = single_token_case(torch.bfloat16)
    table: ReadyWeightTable[TorchExpertWeights] = ReadyWeightTable(1, 3)
    for expert_id, weight in weights.items():
        table.publish(0, expert_id, weight)
    backend = make_backend(table, torch.bfloat16, torch.device("cpu"))
    output = torch.empty_like(batch.hidden_states)

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        completion = backend.submit(batch, output)
        completion.wait_host()
        completion.close()

    operators = {event.key for event in profile.key_averages()}
    assert operators.isdisjoint({"aten::nonzero", "aten::index_select", "aten::index_add_"})
    torch.testing.assert_close(output, reference_output(batch, weights))


@pytest.mark.parametrize(
    ("device_name", "dtype", "linear", "expert", "available", "native_available", "enabled"),
    [
        ("Ascend910_9362", torch.bfloat16, "float", "swiglu", True, True, True),
        ("Ascend910B2", torch.float16, "float", "swiglu", True, True, True),
        ("Ascend310P3", torch.float16, "float", "swiglu", True, True, False),
        ("unknown", torch.bfloat16, "float", "swiglu", True, True, False),
        ("Ascend910_9362", torch.float32, "float", "swiglu", True, True, False),
        ("Ascend910_9362", torch.bfloat16, "w8a8", "swiglu", True, True, False),
        ("Ascend910_9362", torch.bfloat16, "float", "deepseek_v4", True, True, False),
        ("Ascend910_9362", torch.bfloat16, "float", "swiglu", False, True, False),
        ("Ascend910_9362", torch.bfloat16, "float", "swiglu", True, False, True),
    ],
)
def test_ascend_grouped_selection_and_per_batch_memory(
    monkeypatch, device_name, dtype, linear, expert, available, native_available, enabled
) -> None:
    operator = object()
    npu = SimpleNamespace(npu=SimpleNamespace(get_device_name=lambda _: device_name))
    if available:
        npu.npu_grouped_matmul = operator
    route_operator = object()
    if native_available:
        npu.npu_moe_init_routing_v2 = route_operator
    monkeypatch.setitem(sys.modules, "torch_npu", npu)
    options = dict(
        hidden_dim=_HIDDEN_DIM,
        intermediate_dim=_INTERMEDIATE_DIM,
        top_k=_TOP_K,
        dtype=dtype,
        acquire_many=ReadyWeightTable(1, 3).acquire_many,
        linear_compute=linear,
        expert_compute=expert,
    )
    native = enabled and native_available
    for top_k in (1, 2, 8):
        options["top_k"] = top_k
        cpu = TorchBackend(**options, device="cpu")
        backend = TorchBackend(
            **options, runtime=SimpleNamespace(device=SimpleNamespace(type="npu"))
        )
        assert backend._grouped_matmul is (operator if enabled else None)
        assert backend._moe_init_routing is (route_operator if native else None)
        assert cpu._moe_init_routing is None
        for tokens in (1, 32):
            extra = backend.estimate_resources(tokens).total_bytes(2) - cpu.estimate_resources(
                tokens
            ).total_bytes(2)
            scratch = 16 * 1024 * 1024 if enabled else 0
            route_extra = 0
            if native and tokens > 1:
                counts = 8 * 10240 + 4 * 128 * 8 + 3 * 8
                indices_and_routes = 28 * tokens * top_k
                packed = tokens * top_k * _HIDDEN_DIM * 2
                route_extra = counts + indices_and_routes + packed
            assert extra == 2 * (scratch + route_extra)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("routes", [(2, 0, -1), (2, 0, 1), (2, 0, 2), (2, -1, -1), (-1, -1, -1)])
def test_grouped_single_token_preserves_routes_and_weight_views(dtype, routes) -> None:
    batch, weights = single_token_case(dtype, routes)
    table: ReadyWeightTable[TorchExpertWeights] = ReadyWeightTable(1, 3)
    for expert_id, weight in weights.items():
        table.publish(0, expert_id, weight)
    backend = make_backend(table, dtype, torch.device("cpu"))
    calls = []
    leaf_outputs = []

    def grouped(inputs, matrices, *, split_item, group_type, group_list_type=None):
        assert (split_item, group_type, group_list_type) == (2, 0, 0)
        projection = len(calls)
        for matrix, expert_id in zip(matrices, batch.distinct_expert_ids, strict=True):
            original = weights[expert_id].tensors[projection]
            assert matrix.data_ptr() == original.data_ptr()
            assert matrix.stride() == original.t().stride()
        if projection < 2:
            assert all(value is batch.hidden_states for value in inputs)
        else:
            for row, value in enumerate(inputs):
                expected = functional.silu(leaf_outputs[0][row]) * leaf_outputs[1][row]
                torch.testing.assert_close(value, expected, rtol=0, atol=0)
                assert value.shape == (1, _INTERMEDIATE_DIM)
                assert value.storage_offset() == row * _INTERMEDIATE_DIM
                assert value.untyped_storage().data_ptr() == inputs[0].untyped_storage().data_ptr()
        calls.append(projection)
        leaf_outputs.append([x @ matrix for x, matrix in zip(inputs, matrices, strict=True)])
        return [torch.cat(leaf_outputs[-1])]

    backend._grouped_matmul = grouped
    parent = torch.full((3, _HIDDEN_DIM), 7, dtype=dtype)
    output = parent[1:2]
    storage_pointer = parent.untyped_storage().data_ptr()
    output_pointer = output.data_ptr()
    attributes = {}
    span = SimpleNamespace(is_recording=lambda: True, set_attribute=attributes.__setitem__)
    tracer = SimpleNamespace(start_as_current_span=lambda *args, **kwargs: nullcontext(span))
    previous = None
    for scale in (1.0, 0.5):
        batch = replace(batch, routing_weights=batch.routing_weights * scale)
        calls.clear()
        leaf_outputs.clear()
        attributes.clear()
        with use_tracer(tracer):
            completion = backend.submit(batch, output)
        assert [table.usage_count(0, i) for i in batch.distinct_expert_ids] == [1] * len(
            batch.distinct_expert_ids
        )
        completion.wait_host()
        completion.close()
        assert [table.usage_count(0, i) for i in range(3)] == [0, 0, 0]
        assert calls == ([0, 1, 2] if routes in ((2, 0, -1), (2, 0, 1)) else [])
        assert attributes == (
            {"expertkit.expert_ids": ",".join(map(str, batch.distinct_expert_ids))} if calls else {}
        )
        assert output.data_ptr() == output_pointer
        if calls:
            expected = torch.zeros_like(batch.hidden_states, dtype=torch.float32)
            for expert_id, leaf in zip(batch.distinct_expert_ids, leaf_outputs[2], strict=True):
                column = routes.index(expert_id)
                expected.add_(leaf.float() * batch.routing_weights[:, column : column + 1])
            torch.testing.assert_close(output, expected.to(dtype), rtol=0, atol=0)
        torch.testing.assert_close(output, reference_output(batch, weights))

        if previous is not None and batch.distinct_expert_ids:
            assert not torch.equal(output, previous)
        previous = output.clone()
        assert output.storage_offset() == _HIDDEN_DIM
        assert output.untyped_storage().data_ptr() == storage_pointer
        assert torch.equal(parent[[0, 2]], torch.full((2, _HIDDEN_DIM), 7, dtype=dtype))


@pytest.mark.parametrize("failed_projection", [1, 2])
def test_grouped_projection_failure_drains_work_before_releasing_weights(failed_projection) -> None:
    batch, weights = single_token_case(torch.bfloat16)
    table: ReadyWeightTable[TorchExpertWeights] = ReadyWeightTable(1, 3)
    for expert_id, weight in weights.items():
        table.publish(0, expert_id, weight)
    events = []

    class Work:
        def wait_host(self):
            assert [table.usage_count(0, i) for i in (0, 2)] == [1, 1]
            events.append("waited")

    class Runtime(CpuWorkerRuntime):
        def capture_current_work(self):
            return Work()

    backend = TorchBackend(
        hidden_dim=_HIDDEN_DIM,
        intermediate_dim=_INTERMEDIATE_DIM,
        top_k=_TOP_K,
        dtype=torch.bfloat16,
        runtime=Runtime(torch.device("cpu")),
        acquire_many=table.acquire_many,
    )

    def grouped(inputs, matrices, **kwargs):
        assert kwargs == {"split_item": 2, "group_type": 0, "group_list_type": 0}
        if len(events) == failed_projection:
            raise RuntimeError("grouped projection failed")
        events.append("submitted")
        return [torch.cat([x @ matrix for x, matrix in zip(inputs, matrices, strict=True)])]

    backend._grouped_matmul = grouped
    with pytest.raises(RuntimeError, match="grouped projection failed"):
        backend.submit(batch, torch.empty_like(batch.hidden_states))
    assert events == ["submitted"] * failed_projection + ["waited"]
    assert [table.usage_count(0, i) for i in range(3)] == [0, 0, 0]


class _PrefillPacking:
    """Model native initialization, including a poisoned unused packed tail."""

    def __init__(self, batch, *, zero_counts=False):
        self.batch = batch
        self.zero_counts = zero_counts
        self.pack_calls = 0
        self.packed = None

    def __call__(self, hidden, expert_ids, **kwargs):
        assert hidden is self.batch.hidden_states
        assert expert_ids is self.batch.expert_ids
        expert_num = self.batch.distinct_expert_ids[-1] + 1
        assignments = self.batch.token_count * self.batch.top_k
        assert kwargs == dict(
            active_num=assignments,
            expert_num=expert_num,
            drop_pad_mode=0,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[0, expert_num],
            quant_mode=-1,
            row_idx_type=1,
        )
        self.pack_calls += 1
        flat = expert_ids.flatten()
        order = flat.argsort(stable=True)
        order = order[(flat[order] >= 0) & (flat[order] < expert_num)]
        self.packed = torch.full((assignments, hidden.shape[1]), torch.nan, dtype=hidden.dtype)
        rows = torch.full((assignments,), -999, dtype=torch.int32)
        counts = torch.zeros(expert_num, dtype=torch.int64)
        if not self.zero_counts:
            self.packed[: len(order)] = hidden[order // self.batch.top_k]
            rows[: len(order)] = order.int()
            counts = torch.bincount(flat[order].long(), minlength=expert_num)
        return self.packed, rows, counts, torch.zeros(assignments, dtype=torch.float32)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_prefill_packing_preserves_linear_inputs_routes_output_and_fallback(monkeypatch, dtype):
    linear = functional.linear
    for kind in (
        "normal",
        "sparse",
        "duplicate",
        "invalid_nan",
        "zero_counts",
        "missing",
        "unleased",
        "outside_range",
    ):
        batch, weights = make_case(dtype, torch.device("cpu"))
        batch.expert_ids[0, 2] = 1
        batch.expert_ids[1, 2] = 2
        if kind == "sparse":
            ids = batch.expert_ids.clone()
            ids[ids == 1] = 10239
            weights = {0: weights[0], 2: weights[2], 10239: weights[1]}
            batch = replace(batch, expert_ids=ids, distinct_expert_ids=(0, 2, 10239))
        elif kind == "duplicate":
            batch.expert_ids[0] = torch.tensor([2, 2, 0], dtype=torch.int32)
        elif kind in {"invalid_nan", "zero_counts"}:
            batch.hidden_states[2].fill_(torch.nan)
            batch.expert_ids[2].fill_(-1)
            batch.routing_weights[2].zero_()
        elif kind == "missing":
            batch.expert_ids[batch.expert_ids == 1] = -1
        elif kind == "unleased":
            batch = replace(batch, distinct_expert_ids=(0, 2))
        elif kind == "outside_range":
            batch.expert_ids[0, 2] = 77
        batch.routing_weights[0, 0] = 0
        table: ReadyWeightTable[TorchExpertWeights] = ReadyWeightTable(1, max(weights) + 1)
        for expert_id, weight in weights.items():
            table.publish(0, expert_id, weight)
        backend = make_backend(table, dtype, torch.device("cpu"))
        packing = _PrefillPacking(batch, zero_counts=kind == "zero_counts")
        backend._moe_init_routing = packing
        parent = torch.full((batch.token_count + 2, batch.hidden_dim), 7, dtype=dtype)
        output = parent[1:-1]
        storage_pointer = parent.untyped_storage().data_ptr()
        pointer = output.data_ptr()
        routes = batch.expert_ids.clone()
        calls = []
        handled = kind in {"normal", "sparse", "duplicate", "invalid_nan"}

        def checked_linear(
            inputs,
            matrix,
            bias=None,
            *,
            batch=batch,
            weights=weights,
            handled=handled,
            packing=packing,
            calls=calls,
        ):
            expert_id, projection = next(
                (i, p)
                for i, weight in weights.items()
                for p, value in enumerate(weight.tensors)
                if matrix is value
            )
            coordinates = torch.nonzero(batch.expert_ids == expert_id)
            if projection < 2:
                expected = batch.hidden_states.index_select(0, coordinates[:, 0])
                torch.testing.assert_close(inputs, expected, rtol=0, atol=0, equal_nan=True)
                if handled:
                    start = sum(
                        int((batch.expert_ids == i).sum())
                        for i in batch.distinct_expert_ids
                        if i < expert_id
                    )
                    assert (
                        inputs.data_ptr()
                        == packing.packed.data_ptr()
                        + start * batch.hidden_dim * inputs.element_size()
                    )
                    assert inputs.is_contiguous()
            calls.append((expert_id, projection))
            return linear(inputs, matrix, bias)

        with monkeypatch.context() as context:
            context.setattr(functional, "linear", checked_linear)
            if kind == "missing":
                with pytest.raises(InvalidBackendInput, match="distinct expert metadata"):
                    backend.submit(batch, output)
            else:
                completion = backend.submit(batch, output)
                assert all(table.usage_count(0, i) == 1 for i in batch.distinct_expert_ids)
                completion.wait_host()
                completion.close()
        assert packing.pack_calls == 1
        if kind != "missing":
            reference_batch = replace(batch, expert_ids=batch.expert_ids.clone())
            for i in set(batch.expert_ids.flatten().tolist()) - set(batch.distinct_expert_ids):
                reference_batch.expert_ids[reference_batch.expert_ids == i] = -1
            torch.testing.assert_close(output, reference_output(reference_batch, weights))
            assert calls == [(i, p) for i in batch.distinct_expert_ids for p in range(3)]
        if kind in {"invalid_nan", "zero_counts"}:
            assert torch.equal(output[2], torch.zeros_like(output[2]))
        assert output.data_ptr() == pointer
        assert torch.equal(batch.expert_ids, routes)
        assert all(table.usage_count(0, i) == 0 for i in batch.distinct_expert_ids)

        if kind == "normal":
            previous = output.clone()
            batch = replace(batch, routing_weights=batch.routing_weights * 0.5)
            completion = backend.submit(batch, output)
            completion.wait_host()
            completion.close()
            torch.testing.assert_close(output, reference_output(batch, weights))
            assert not torch.equal(output, previous)
            assert packing.pack_calls == 2
            assert all(table.usage_count(0, i) == 0 for i in batch.distinct_expert_ids)
        assert output.data_ptr() == pointer
        assert output.storage_offset() == batch.hidden_dim
        assert output.untyped_storage().data_ptr() == storage_pointer
        assert torch.equal(parent[[0, -1]], torch.full((2, batch.hidden_dim), 7, dtype=dtype))


@pytest.mark.parametrize("failed_projection", [1, 2])
def test_prefill_linear_failure_drains_native_work_before_releasing_weights(
    monkeypatch, failed_projection
):
    batch, weights = make_case(torch.bfloat16, torch.device("cpu"))
    table: ReadyWeightTable[TorchExpertWeights] = ReadyWeightTable(1, 3)
    for expert_id, weight in weights.items():
        table.publish(0, expert_id, weight)
    events = []

    class Work:
        def wait_host(self):
            assert [table.usage_count(0, i) for i in range(3)] == [1, 1, 1]
            events.append("drained")

    class Runtime(CpuWorkerRuntime):
        def capture_current_work(self):
            return Work()

    backend = TorchBackend(
        hidden_dim=_HIDDEN_DIM,
        intermediate_dim=_INTERMEDIATE_DIM,
        top_k=_TOP_K,
        dtype=torch.bfloat16,
        runtime=Runtime(torch.device("cpu")),
        acquire_many=table.acquire_many,
    )
    packing = _PrefillPacking(batch)
    backend._moe_init_routing = packing
    linear = functional.linear

    def fail_projection(*args, **kwargs):
        if len(events) == failed_projection:
            raise RuntimeError("prefill projection failed")
        result = linear(*args, **kwargs)
        events.append("submitted")
        return result

    monkeypatch.setattr(functional, "linear", fail_projection)
    with pytest.raises(RuntimeError, match="prefill projection failed"):
        backend.submit(batch, torch.empty_like(batch.hidden_states))
    assert packing.pack_calls == 1
    assert events == ["submitted"] * failed_projection + ["drained"]
    assert [table.usage_count(0, i) for i in range(3)] == [0, 0, 0]


@pytest.mark.parametrize("grouped", [False, True])
def test_single_token_rejects_absent_expert_and_releases_weights(grouped: bool) -> None:
    batch, weights = single_token_case(torch.float32)
    batch = replace(batch, distinct_expert_ids=(0, 1, 2))
    table: ReadyWeightTable[TorchExpertWeights] = ReadyWeightTable(1, 3)
    for expert_id, weight in weights.items():
        table.publish(0, expert_id, weight)
    backend = make_backend(table, torch.float32, torch.device("cpu"))
    if grouped:

        def unexpected_projection(*args, **kwargs):
            raise AssertionError("invalid metadata must fail before grouped projection")

        backend._grouped_matmul = unexpected_projection

    with pytest.raises(InvalidBackendInput, match="distinct expert metadata"):
        backend.submit(batch, torch.empty_like(batch.hidden_states))
    assert [table.usage_count(0, i) for i in range(3)] == [0, 0, 0]


@pytest.mark.skipif(os.getenv("EK_TEST_NPU") != "1", reason="requires an assigned Ascend NPU")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("routes", [(2, 0, -1), (2, 0, 1), (2, 0, 2), (-1, -1, -1)])
def test_ascend_grouped_single_token_matches_independent_reference(dtype, routes) -> None:
    pytest.importorskip("torch_npu")
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    generator = torch.Generator().manual_seed(82)
    weights = {
        i: TorchExpertWeights(
            *(
                torch.randn(shape, generator=generator).to(dtype) / 8
                for shape in ((64, 32),) * 2 + ((32, 64),)
            )
        )
        for i in range(3)
    }
    batch = BackendBatch(
        layer_id=0,
        hidden_states=torch.randn(1, 32, generator=generator).to(dtype),
        expert_ids=torch.tensor([routes], dtype=torch.int32),
        routing_weights=torch.tensor(
            [[value if i >= 0 else 0.0 for i, value in zip(routes, (0.2, 0.0, 0.7), strict=True)]],
            dtype=torch.float32,
        ),
        distinct_expert_ids=tuple(sorted({i for i in routes if i >= 0})),
    )
    expected = reference_output(batch, weights)
    table: ReadyWeightTable[TorchExpertWeights] = ReadyWeightTable(1, 3)
    for expert_id, weight in weights.items():
        table.publish(0, expert_id, weight.to(device))
    batch = replace(
        batch,
        hidden_states=batch.hidden_states.to(device),
        expert_ids=batch.expert_ids.to(device),
        routing_weights=batch.routing_weights.to(device),
    )
    backend = TorchBackend(
        hidden_dim=32,
        intermediate_dim=64,
        top_k=3,
        dtype=dtype,
        device=device,
        acquire_many=table.acquire_many,
    )
    assert backend._grouped_matmul is not None
    parent = torch.full((3, 32), 7, dtype=dtype, device=device)
    output = parent[1:2]
    pointer = output.data_ptr()
    storage_pointer = parent.untyped_storage().data_ptr()
    stream = torch.npu.Stream(device=device)
    stream.wait_stream(torch.npu.current_stream(device))
    previous = None
    for scale in (1.0, 0.5):
        with torch.npu.stream(stream):
            batch = replace(batch, routing_weights=batch.routing_weights * scale)
            completion = backend.submit(batch, output)
        assert all(table.usage_count(0, i) == 1 for i in batch.distinct_expert_ids)
        try:
            completion.wait_host()
            actual = output.cpu()
            torch.testing.assert_close(actual, expected * scale, atol=2e-3, rtol=2e-2)
            if previous is not None and batch.distinct_expert_ids:
                assert not torch.equal(actual, previous)
            previous = actual
        finally:
            completion.close()
        assert all(table.usage_count(0, i) == 0 for i in range(3))
        assert output.data_ptr() == pointer
        assert output.storage_offset() == 32
        assert output.untyped_storage().data_ptr() == storage_pointer
        assert parent[[0, 2]].cpu().eq(7).all()


@pytest.mark.skipif(os.getenv("EK_TEST_NPU") != "1", reason="requires an assigned Ascend NPU")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("top_k", [1, 2, 8])
def test_ascend_packed_prefill_preserves_linear_views_fallback_and_completion(
    monkeypatch, dtype, top_k
):
    torch_npu = pytest.importorskip("torch_npu")
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    generator = torch.Generator().manual_seed(40)
    routes = (
        [[1], [0], [1]]
        if top_k == 1
        else [[2, 0], [1, 0], [2, 1]]
        if top_k == 2
        else [[7, 0, 4, 2, 6, 1, 5, 3], list(range(8)), list(range(8))]
    )
    ids = torch.tensor(routes, dtype=torch.int32)
    distinct = tuple(sorted({i for row in routes for i in row}))
    hidden = torch.randn(3, 32, generator=generator).to(dtype) / 8
    routing = torch.full(ids.shape, 1 / top_k)
    routing[0, 0] = 0
    cpu_batch = BackendBatch(
        layer_id=0,
        hidden_states=hidden,
        expert_ids=ids,
        routing_weights=routing,
        distinct_expert_ids=distinct,
    )
    weights = {
        i: TorchExpertWeights(
            *(
                torch.randn(shape, generator=generator).to(dtype) / 8
                for shape in ((64, 32), (64, 32), (32, 64))
            )
        )
        for i in distinct
    }
    device_weights = {i: weight.to(device) for i, weight in weights.items()}
    table: ReadyWeightTable[TorchExpertWeights] = ReadyWeightTable(1, max(distinct) + 1)
    for i, weight in device_weights.items():
        table.publish(0, i, weight)
    backend = TorchBackend(
        hidden_dim=32,
        intermediate_dim=64,
        top_k=top_k,
        dtype=dtype,
        device=device,
        acquire_many=table.acquire_many,
    )
    assert backend._moe_init_routing is not None
    packed_buffers = []

    def pack(*args, **kwargs):
        assert kwargs["row_idx_type"] == 1
        result = torch_npu.npu_moe_init_routing_v2(*args, **kwargs)
        packed_buffers.append(result[0])
        return result

    backend._moe_init_routing = pack
    linear = functional.linear
    calls = []
    check_views = False
    active_cpu = cpu_batch

    def checked_linear(inputs, matrix, bias=None):
        expert_id, projection = next(
            (i, p)
            for i, weight in device_weights.items()
            for p, value in enumerate(weight.tensors)
            if matrix is value
        )
        if check_views and projection < 2:
            start = sum(int((active_cpu.expert_ids == i).sum()) for i in distinct if i < expert_id)
            assert (
                inputs.data_ptr()
                == packed_buffers[-1].data_ptr() + start * 32 * inputs.element_size()
            )
            assert inputs.is_contiguous()
        calls.append((expert_id, projection))
        return linear(inputs, matrix, bias)

    monkeypatch.setattr(functional, "linear", checked_linear)
    stream = torch.npu.Stream(device=device)
    stream.wait_stream(torch.npu.current_stream(device))

    with torch.npu.stream(stream):
        parent = torch.full((5, 32), 7, dtype=dtype, device=device)
    output = parent[1:4]
    pointer = output.data_ptr()
    storage_pointer = parent.untyped_storage().data_ptr()

    def run(cpu_case, *, packed_views=False):
        nonlocal check_views, active_cpu
        check_views, active_cpu = packed_views, cpu_case
        calls.clear()
        before = len(packed_buffers)
        with torch.npu.stream(stream):
            batch = replace(
                cpu_case,
                hidden_states=cpu_case.hidden_states.to(device),
                expert_ids=cpu_case.expert_ids.to(device),
                routing_weights=cpu_case.routing_weights.to(device),
            )
            completion = backend.submit(batch, output)
        assert len(packed_buffers) == before + 1
        assert all(table.usage_count(0, i) == 1 for i in distinct)
        completion.wait_host()
        completion.close()
        actual = output.cpu()
        assert calls == [(i, p) for i in distinct for p in range(3)]
        assert output.data_ptr() == pointer
        assert output.storage_offset() == 32
        assert output.untyped_storage().data_ptr() == storage_pointer
        assert parent[[0, 4]].cpu().eq(7).all()
        assert all(table.usage_count(0, i) == 0 for i in distinct)
        check_views = False
        torch.testing.assert_close(
            actual, reference_output(cpu_case, weights), atol=2e-3, rtol=2e-2
        )
        return batch, output

    batch, output = run(cpu_batch, packed_views=True)
    previous = output.cpu()
    run(replace(cpu_batch, routing_weights=routing * 0.5), packed_views=True)
    assert not torch.equal(output.cpu(), previous)
    masked_hidden, masked_ids, masked_routing = hidden.clone(), ids.clone(), routing.clone()
    masked_hidden[2].fill_(torch.nan)
    masked_ids[2].fill_(-1)
    masked_routing[2].zero_()
    masked = replace(
        cpu_batch,
        hidden_states=masked_hidden,
        expert_ids=masked_ids,
        routing_weights=masked_routing,
    )
    _, masked_output = run(masked)
    assert masked_output[2].cpu().eq(0).all()
    if top_k > 1:
        duplicate_ids = ids.clone()
        duplicate_ids[0, 1] = duplicate_ids[0, 0]
        run(replace(cpu_batch, expert_ids=duplicate_ids), packed_views=True)
    calls.clear()

    def fail_projection(*args, **kwargs):
        if calls:
            assert all(table.usage_count(0, i) == 1 for i in distinct)
            raise RuntimeError("prefill projection failed")
        return checked_linear(*args, **kwargs)

    monkeypatch.setattr(functional, "linear", fail_projection)
    with torch.npu.stream(stream), pytest.raises(RuntimeError, match="prefill projection failed"):
        backend.submit(batch, output)
    assert len(calls) == 1
    assert all(table.usage_count(0, i) == 0 for i in distinct)


@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [
        (torch.float32, 1e-6, 1e-5),
        (torch.float16, 2e-3, 2e-3),
        (torch.bfloat16, 2e-2, 2e-2),
    ],
)
def test_torch_backend_matches_weighted_reference_and_releases_weights(
    dtype: torch.dtype,
    atol: float,
    rtol: float,
) -> None:
    device = torch.device("cpu")
    batch, weights = make_case(dtype, device)
    table: ReadyWeightTable[TorchExpertWeights] = ReadyWeightTable(1, 3)
    for expert_id, weight in weights.items():
        table.publish(0, expert_id, weight)
    backend = make_backend(table, dtype, device)
    output = torch.full_like(batch.hidden_states, torch.nan)

    completion = backend.submit(batch, output)

    completion.wait_host()
    torch.testing.assert_close(output, reference_output(batch, weights), atol=atol, rtol=rtol)
    assert [table.usage_count(0, expert_id) for expert_id in range(3)] == [1, 1, 1]
    completion.close()
    completion.close()
    assert [table.usage_count(0, expert_id) for expert_id in range(3)] == [0, 0, 0]


def test_torch_backend_acquires_all_experts_once() -> None:
    device = torch.device("cpu")
    batch, weights = make_case(torch.float32, device)
    table: ReadyWeightTable[TorchExpertWeights] = ReadyWeightTable(1, 3)
    for expert_id, weight in weights.items():
        table.publish(0, expert_id, weight)
    calls: list[tuple[int, tuple[int, ...]]] = []

    def acquire(layer_id: int, expert_ids: tuple[int, ...]):
        calls.append((layer_id, expert_ids))
        return table.acquire_many(layer_id, expert_ids)

    backend = TorchBackend(
        hidden_dim=_HIDDEN_DIM,
        intermediate_dim=_INTERMEDIATE_DIM,
        top_k=_TOP_K,
        dtype=torch.float32,
        runtime=CpuWorkerRuntime(device),
        acquire_many=acquire,
    )
    completion = backend.submit(batch, torch.empty_like(batch.hidden_states))
    completion.wait_host()
    completion.close()

    assert calls == [(0, (0, 1, 2))]


class _RecordingWork:
    def __init__(self) -> None:
        self.wait_count = 0

    def wait_host(self) -> None:
        self.wait_count += 1


class _RecordingRuntime:
    def __init__(self, device: torch.device, work: DeviceWork) -> None:
        self.device = device
        self.work = work
        self.capture_count = 0

    def capture_current_work(self) -> DeviceWork:
        self.capture_count += 1
        return self.work


def test_torch_completion_retains_weights_until_device_work_finishes() -> None:
    device = torch.device("cpu")
    batch, weights = make_case(torch.float32, device)
    table: ReadyWeightTable[TorchExpertWeights] = ReadyWeightTable(1, 3)
    for expert_id, weight in weights.items():
        table.publish(0, expert_id, weight)
    work = _RecordingWork()
    runtime = _RecordingRuntime(device, work)
    backend = TorchBackend(
        hidden_dim=_HIDDEN_DIM,
        intermediate_dim=_INTERMEDIATE_DIM,
        top_k=_TOP_K,
        dtype=torch.float32,
        runtime=runtime,  # type: ignore[arg-type]
        acquire_many=table.acquire_many,
    )

    completion = backend.submit(batch, torch.empty_like(batch.hidden_states))

    assert runtime.capture_count == 1
    assert work.wait_count == 0
    assert [table.usage_count(0, expert_id) for expert_id in range(3)] == [1, 1, 1]
    completion.close()
    assert work.wait_count == 1
    assert [table.usage_count(0, expert_id) for expert_id in range(3)] == [0, 0, 0]


def test_torch_backend_reports_missing_ready_weights_without_partial_retention() -> None:
    device = torch.device("cpu")
    batch, weights = make_case(torch.float32, device)
    table: ReadyWeightTable[TorchExpertWeights] = ReadyWeightTable(1, 3)
    table.publish(0, 0, weights[0])
    table.publish(0, 2, weights[2])
    backend = make_backend(table, torch.float32, device)

    with pytest.raises(BackendWeightUnavailable) as caught:
        backend.submit(batch, torch.empty_like(batch.hidden_states))

    assert caught.value.unavailable_expert_ids == (1,)
    assert table.usage_count(0, 0) == 0
    assert table.usage_count(0, 2) == 0


def test_torch_backend_treats_invalid_ready_weight_as_fatal_and_releases_lease() -> None:
    device = torch.device("cpu")
    batch, weights = make_case(torch.float32, device)
    weights[0] = TorchExpertWeights(
        gate_proj=torch.ones(_INTERMEDIATE_DIM, _HIDDEN_DIM - 1),
        up_proj=torch.ones(_INTERMEDIATE_DIM, _HIDDEN_DIM - 1),
        down_proj=torch.ones(_HIDDEN_DIM - 1, _INTERMEDIATE_DIM),
    )
    table: ReadyWeightTable[TorchExpertWeights] = ReadyWeightTable(1, 3)
    for expert_id, weight in weights.items():
        table.publish(0, expert_id, weight)
    backend = make_backend(table, torch.float32, device)

    with pytest.raises(BackendFatalError) as caught:
        backend.submit(batch, torch.empty_like(batch.hidden_states))

    assert caught.value.reason is BackendFatalReason.UNEXPECTED
    assert [table.usage_count(0, expert_id) for expert_id in range(3)] == [0, 0, 0]


@pytest.mark.parametrize("token_count", [1, 2])
def test_torch_backend_zeroes_output_when_worker_has_no_local_assignments(token_count: int) -> None:
    device = torch.device("cpu")
    hidden = torch.ones(token_count, _HIDDEN_DIM)
    batch = BackendBatch(
        layer_id=0,
        hidden_states=hidden,
        expert_ids=torch.full((token_count, _TOP_K), -1, dtype=torch.int32),
        routing_weights=torch.zeros(token_count, _TOP_K, dtype=torch.float32),
        distinct_expert_ids=(),
    )
    table: ReadyWeightTable[TorchExpertWeights] = ReadyWeightTable(1, 3)
    backend = make_backend(table, torch.float32, device)
    output = torch.full_like(hidden, 7)

    completion = backend.submit(batch, output)
    completion.wait_host()
    completion.close()

    torch.testing.assert_close(output, torch.zeros_like(output))


def test_torch_backend_rejects_wrong_prepared_output_before_weight_acquisition() -> None:
    device = torch.device("cpu")
    batch, _ = make_case(torch.float32, device)
    calls = 0

    def acquire(
        _layer_id: int,
        _expert_ids: tuple[int, ...],
    ) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("acquisition must not run")

    backend = TorchBackend(
        hidden_dim=_HIDDEN_DIM,
        intermediate_dim=_INTERMEDIATE_DIM,
        top_k=_TOP_K,
        dtype=torch.float32,
        runtime=CpuWorkerRuntime(device),
        acquire_many=acquire,  # type: ignore[arg-type]
    )

    with pytest.raises(InvalidBackendInput, match="prepared output shape"):
        backend.submit(batch, torch.empty(2, _HIDDEN_DIM))
    assert calls == 0


def test_torch_backend_resource_estimate_scales_with_token_bound() -> None:
    table: ReadyWeightTable[TorchExpertWeights] = ReadyWeightTable(1, 3)
    backend = make_backend(table, torch.float16, torch.device("cpu"))

    one = backend.estimate_resources(1).temporary_bytes_per_active_batch
    eight = backend.estimate_resources(8).temporary_bytes_per_active_batch

    assert one > 0
    assert eight == one * 8


@pytest.mark.parametrize("device_type", ["cuda", "npu"])
def test_torch_backend_completes_repeated_device_submissions(device_type: str) -> None:
    if device_type == "npu":
        pytest.importorskip("torch_npu")
    if not torch.get_device_module(device_type).is_available():
        pytest.skip(f"{device_type} is unavailable")
    device = torch.device(f"{device_type}:0")
    batch, weights = make_case(torch.float16, device)
    table: ReadyWeightTable[TorchExpertWeights] = ReadyWeightTable(1, 3)
    for expert_id, weight in weights.items():
        table.publish(0, expert_id, weight)
    backend = make_backend(table, torch.float16, device)
    expected = reference_output(batch, weights)

    for _ in range(3):
        output = torch.empty_like(batch.hidden_states)
        completion = backend.submit(batch, output)
        completion.wait_host()
        completion.close()
        torch.testing.assert_close(output.cpu(), expected.cpu(), atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("wait_first", [False, True])
def test_failed_completion_keeps_weights_until_process_exit(wait_first: bool) -> None:
    import gc
    import weakref

    from expertkit_worker.backends.torch.backend import _TorchCompletion

    class FailedWork:
        def wait_host(self) -> None:
            raise RuntimeError("unprovable device completion")

    table: ReadyWeightTable[TorchExpertWeights] = ReadyWeightTable(1, 1)
    table.publish(0, 0, make_weight(0, torch.float32, torch.device("cpu")))
    lease = table.acquire_many(0, (0,))
    retained = weakref.ref(lease)
    completion = _TorchCompletion(lease, FailedWork())  # type: ignore[arg-type]
    if wait_first:
        with pytest.raises(RuntimeError, match="unprovable"):
            completion.wait_host()
        completion.close()
    else:
        with pytest.raises(RuntimeError, match="unprovable"):
            completion.close()
    assert table.quarantined
    assert table.usage_count(0, 0) == 1
    del lease, completion
    gc.collect()
    assert retained() is not None
