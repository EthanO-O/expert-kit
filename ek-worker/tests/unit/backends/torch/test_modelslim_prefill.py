"""Check grouped ModelSlim arithmetic, allocation bounds and lease completion."""

import os

import pytest
import torch

from expertkit_worker.backends.base import BackendBatch
from expertkit_worker.backends.torch.backend import TorchBackend
from expertkit_worker.backends.torch.modelslim_w8a8 import (
    TorchModelSlimW8A8Weights,
)
from expertkit_worker.weights import ReadyWeightTable


def test_cpu_and_v4_keep_the_ordinary_backend():
    for expert_compute in ["swiglu", "deepseek_v4"]:
        backend = TorchBackend(
            hidden_dim=2048,
            intermediate_dim=768,
            top_k=8,
            dtype=torch.bfloat16,
            device="cpu",
            acquire_many=lambda *_: None,
            linear_compute="modelslim_w8a8_dynamic",
            expert_compute=expert_compute,
        )
        assert backend._modelslim_prefill is None


def test_projection_views_borrow_singleton_and_sparse_weight_storage():
    pytest.importorskip("triton")
    from expertkit_worker.backends.torch.modelslim_prefill import projection_views, temporary_bytes

    weight = TorchModelSlimW8A8Weights(
        (
            torch.zeros((4, 8), dtype=torch.int8),
            torch.zeros((4, 8), dtype=torch.int8),
            torch.zeros((8, 4), dtype=torch.int8),
        ),
        (torch.ones(8), torch.ones(8), torch.ones(4)),
        torch.bfloat16,
    )
    assert projection_views((weight,), 0)[0].shape == (1, 4, 8)
    assert projection_views((weight,), 0)[0].data_ptr() == weight.matrices[0].data_ptr()
    assert projection_views((weight, weight), 2)[1] is weight.matrices[2]
    assert temporary_bytes(4096, 2048, 768, 8) > temporary_bytes(2048, 2048, 768, 8)
    # The bound includes simultaneous full-capacity INT32 products and BF16 rows.
    assert temporary_bytes(4096, 2048, 768, 8) > 4096 * 8 * (2048 + 768) * 12


@pytest.mark.parametrize("failure", [None, "launch", "drain"])
def test_warmup_fixed_signatures_and_failed_startup_drain(monkeypatch, failure):
    from contextlib import nullcontext

    pytest.importorskip("triton")
    from expertkit_worker.backends.torch import modelslim_prefill

    events = []
    retained = []
    original = RuntimeError("launch failed")

    class Launcher:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                events.append((self.name, grid, args, kwargs))
                if self.name == "inverse" and failure:
                    raise original

            return launch

    class Runtime:
        device = torch.device("cpu")

        def device_context(self):
            return nullcontext()

        def capture_current_work(self):
            return self

        def wait_host(self):
            events.append(("wait",))
            if failure == "drain":
                raise RuntimeError("drain failed")

    for name, label in [
        ("_copy_prefix", "copy"),
        ("_inverse_prefix", "inverse"),
        ("_dequantize", "dequant"),
    ]:
        monkeypatch.setattr(modelslim_prefill, name, Launcher(label))
    monkeypatch.setattr(
        modelslim_prefill, "ordered_combine", lambda *a, **k: events.append(("combine", a, k))
    )
    monkeypatch.setattr(modelslim_prefill, "retain_until_process_exit", retained.append)
    if failure:
        with pytest.raises(RuntimeError) as caught:
            modelslim_prefill.warmup(Runtime())
        assert caught.value is original
        assert events[-1] == ("wait",)
        assert bool(retained) == (failure == "drain")
        if retained:
            assert all(isinstance(t, torch.Tensor) for t in retained[0])
            assert "drain failed" in original.__notes__[0]
    else:
        result = modelslim_prefill.warmup(Runtime())
        assert [event[0] for event in events] == [
            "copy",
            "inverse",
            "dequant",
            "dequant",
            "combine",
            "wait",
        ]
        assert len(events[1][2]) == 4
        for event in events[2:4]:
            assert event[2][5].dtype == torch.bfloat16
            assert event[2][7].dtype == torch.int32 and event[2][7].tolist() == [2]
            assert event[3] == {"enable_fp_fusion": False}
        assert events[-2][2] == {"expert_start": 0, "expert_end": 128}
        assert 0 < result["buffer_bytes"] < 1024 * 1024 and result["duration_seconds"] >= 0
        assert not retained


def test_eligible_backend_warms_before_becoming_available(monkeypatch):
    from types import SimpleNamespace

    from expertkit_worker.backends.torch import backend as module

    calls = []
    helper = SimpleNamespace(warmup=lambda runtime: calls.append(runtime) or {})
    native = SimpleNamespace(
        npu=SimpleNamespace(get_device_name=lambda _: "Ascend910_9362"),
        npu_moe_init_routing_v2=lambda: None,
        npu_dynamic_quant=lambda: None,
        npu_grouped_matmul=lambda: None,
    )
    monkeypatch.setattr(
        module, "import_module", lambda name: native if name == "torch_npu" else helper
    )
    runtime = SimpleNamespace(device=SimpleNamespace(type="npu"))
    instance = TorchBackend(
        hidden_dim=2048,
        intermediate_dim=768,
        top_k=8,
        dtype=torch.bfloat16,
        runtime=runtime,
        acquire_many=lambda *_: None,
        linear_compute="modelslim_w8a8_dynamic",
    )
    assert calls == [runtime] and instance._modelslim_prefill is helper


@pytest.mark.parametrize("missing_module", ["triton", "triton.language"])
def test_missing_triton_keeps_ordinary_backend_without_hiding_broken_imports(
    monkeypatch, missing_module
):
    from types import SimpleNamespace

    from expertkit_worker.backends.torch import backend as module

    native = SimpleNamespace(
        npu=SimpleNamespace(get_device_name=lambda _: "Ascend910_9362"),
        npu_moe_init_routing_v2=lambda: None,
        npu_dynamic_quant=lambda: None,
        npu_grouped_matmul=lambda: None,
    )
    missing = ModuleNotFoundError(f"No module named {missing_module!r}", name=missing_module)

    def load(name):
        if name == "torch_npu":
            return native
        assert name == "expertkit_worker.backends.torch.modelslim_prefill"
        raise missing

    monkeypatch.setattr(module, "import_module", load)
    runtime = SimpleNamespace(device=SimpleNamespace(type="npu"))
    options = dict(
        hidden_dim=2048,
        intermediate_dim=768,
        top_k=8,
        dtype=torch.bfloat16,
        runtime=runtime,
        acquire_many=lambda *_: None,
        linear_compute="modelslim_w8a8_dynamic",
    )
    if missing_module == "triton":
        instance = TorchBackend(**options)
        assert instance._modelslim_prefill is None
        assert instance.estimate_resources(32).temporary_bytes_per_active_batch > 0
    else:
        with pytest.raises(ModuleNotFoundError) as caught:
            TorchBackend(**options)
        assert caught.value is missing


def _assert_bits(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


def _cpu_integer_dot(quantized, matrix):
    """Compute an exact I8 dot independently of the grouped NPU implementation."""
    assert quantized.device.type == matrix.device.type == "cpu"
    assert quantized.dtype == matrix.dtype == torch.int8
    dot = quantized.double() @ matrix.double()
    assert bool(torch.isfinite(dot).all()) and torch.equal(dot, dot.round())
    assert bool((dot.abs() <= quantized.shape[1] * 128**2).all())
    assert bool(((dot >= -(2**31)) & (dot < 2**31)).all())
    integer = dot.to(torch.int32)
    assert torch.equal(integer.double(), dot)
    return integer


def _cpu_projection_formulas(integer, channel, token):
    """Keep both FP32 multiplication orders explicit before their BF16 boundary."""
    assert channel.dtype == token.dtype == torch.float32
    dot = integer.float()
    combined = channel * token[:, None]
    scale_first = combined * dot
    dot_channel = dot * channel
    dot_first = dot_channel * token[:, None]
    assert bool(torch.isfinite(scale_first).all()) and bool(torch.isfinite(dot_first).all())
    return scale_first.bfloat16(), dot_first.bfloat16()


def _formula_membership(actual, first, second):
    """Accept a whole matrix's ordering, never an elementwise mixture of both."""
    return [
        torch.equal(actual.contiguous().view(torch.uint8), form.contiguous().view(torch.uint8))
        for form in (first, second)
    ]


def _add_ordered_reference(accumulator, down, token, route, routing):
    """Add one expert's rows in slot order without an atomic reduction or FMA."""
    for slot in range(8):
        selected = route == slot
        targets = token[selected]
        contribution = down[selected].float() * routing[targets, slot, None]
        added = accumulator[targets] + contribution
        accumulator[targets] = added


def test_independent_integer_dot_formulas_and_ordered_sum():
    q = torch.tensor([[-128, 127, -3], [127, 127, 127]], dtype=torch.int8)
    weight = torch.tensor([[127, -128], [-128, 127], [5, -7]], dtype=torch.int8)
    integer = _cpu_integer_dot(q, weight)
    expected = torch.tensor(
        [
            [sum(int(a) * int(b) for a, b in zip(row, col, strict=True)) for col in weight.T]
            for row in q
        ],
        dtype=torch.int32,
    )
    assert torch.equal(integer, expected)
    first, second = _cpu_projection_formulas(
        integer, torch.tensor([0.003, 0.007]), torch.tensor([0.01, 0.02])
    )
    assert first.dtype == second.dtype == torch.bfloat16
    first = torch.tensor([[1, 2]], dtype=torch.bfloat16)
    second = torch.tensor([[3, 4]], dtype=torch.bfloat16)
    assert _formula_membership(first.clone(), first, second) == [True, False]
    assert not any(_formula_membership(torch.tensor([[1, 4]], dtype=torch.bfloat16), first, second))
    accumulator = torch.zeros((1, 1))
    # The two small contributions would survive if added before the large one.
    for value, slots in [(2**25, [0]), (1, [1, 2])]:
        _add_ordered_reference(
            accumulator,
            torch.full((len(slots), 1), value, dtype=torch.bfloat16),
            torch.zeros(len(slots), dtype=torch.long),
            torch.tensor(slots),
            torch.ones((1, 8)),
        )
    assert accumulator.item() == 2**25


@pytest.fixture
def eight_cpu_threads():
    """Bound qualification BLAS work without changing other tests' thread settings."""
    previous = torch.get_num_threads()
    torch.set_num_threads(8)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.mark.skipif(os.environ.get("EK_TEST_NPU") != "1", reason="requires an Ascend NPU")
def test_grouped_modelslim_actual_projections_tail_stream_output_and_failure(
    monkeypatch, tmp_path, eight_cpu_threads
):
    native = pytest.importorskip("torch_npu")
    from expertkit_worker.device.ascend import AscendWorkerRuntime

    torch.npu.set_device(0)
    device = torch.device("npu:0")
    table = ReadyWeightTable[TorchModelSlimW8A8Weights](1, 128)
    generator = torch.Generator().manual_seed(68)
    weights = {}
    for expert in range(64):
        matrices = tuple(
            torch.randint(-32, 32, shape, generator=generator, dtype=torch.int8).to(device)
            for shape in [(2048, 768), (2048, 768), (768, 2048)]
        )
        scales = tuple(
            (torch.rand(width, generator=generator) * 0.002 + 0.001).to(device)
            for width in [768, 768, 2048]
        )
        weights[expert] = TorchModelSlimW8A8Weights(matrices, scales, torch.bfloat16)
        table.publish(0, expert, weights[expert])
    weights[127] = weights[0]
    table.publish(0, 127, weights[127])
    for expert in (65, 73, 98, 99, 100, 114):
        weights[expert] = weights[expert % 64]
        table.publish(0, expert, weights[expert])
    backend = TorchBackend(
        hidden_dim=2048,
        intermediate_dim=768,
        top_k=8,
        dtype=torch.bfloat16,
        runtime=AscendWorkerRuntime(device),
        acquire_many=table.acquire_many,
        linear_compute="modelslim_w8a8_dynamic",
    )
    assert backend._modelslim_prefill is not None
    grouped = native.npu_grouped_matmul
    calls = []

    def observe_grouped(*args, **kwargs):
        result = grouped(*args, **kwargs)
        calls.append((kwargs["group_list"], kwargs["output_dtype"]))
        return result

    monkeypatch.setattr(native, "npu_grouped_matmul", observe_grouped)
    init = native.npu_moe_init_routing_v2
    valid_rows = 0

    def poison_tail(*args, **kwargs):
        result = init(*args, **kwargs)
        result[0][valid_rows:].fill_(float("nan"))
        result[1][valid_rows:].fill_(-2147483648)
        return result

    monkeypatch.setattr(native, "npu_moe_init_routing_v2", poison_tail)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    from expertkit_worker.backends.torch import modelslim_prefill

    dequantize = modelslim_prefill._dequantize
    capturing = False
    captured_quant = []
    captured_projection = []
    dynamic_quant = native.npu_dynamic_quant

    def observe_quant(*args, **kwargs):
        result = dynamic_quant(*args, **kwargs)
        if capturing:
            captured_quant.append((args[0], *result))
        return result

    class ObserveDequantize:
        def __getitem__(self, grid):
            launch = dequantize[grid]

            def invoke(*args, **kwargs):
                result = launch(*args, **kwargs)
                if capturing:
                    captured_projection.append(args[:6])
                return result

            return invoke

    monkeypatch.setattr(native, "npu_dynamic_quant", observe_quant)
    monkeypatch.setattr(modelslim_prefill, "_dequantize", ObserveDequantize())

    with torch.npu.stream(stream):
        for width in [768, 2048]:
            dot = torch.randint(-(2**24), 2**24, (24, width), generator=generator).to(device)
            dot[8:].fill_(-2147483648)
            channels = (torch.rand((3, width), generator=generator) * 0.003).to(device)
            tokens = (torch.rand(24, generator=generator) * 0.1).to(device)
            result = torch.empty((24, width), dtype=torch.bfloat16, device=device)
            dequantize[(24, (width + 255) // 256)](
                dot,
                tokens,
                channels,
                torch.tensor([3, 1, 4], device=device),
                torch.tensor([8], device=device),
                result,
                width,
                torch.tensor([3], device=device, dtype=torch.int32),
                256,
                enable_fp_fusion=False,
            )
            expected = torch.zeros_like(result)
            for row, expert in enumerate([0, 0, 0, 1, 2, 2, 2, 2]):
                expected[row] = ((channels[expert] * tokens[row]) * dot[row].float()).bfloat16()
            stream.synchronize()
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
    peaks = []

    def make_batch(tokens, shift):
        host = torch.arange(tokens * 8, dtype=torch.int32).reshape(tokens, 8) % 128
        if tokens == 3:
            host = torch.tensor(
                [[7, 0, -1, 7, 3, -1, 0, -1], [3, -1, 0, 7, -1, 0, -1, 3], [-1] * 8],
                dtype=torch.int32,
            )
        host[host >= 64] = -1
        if tokens == 3 and shift in (1.0, 2.0):
            host[host >= 0] = 0 if shift == 1.0 else 127
        if tokens == 64:
            sparse = [40, 45] if shift == 1.0 else [65, 73, 98, 99, 100, 114]
            positions = torch.arange(tokens * 8).reshape(tokens, 8)
            host = torch.tensor(sparse, dtype=torch.int32)[positions % len(sparse)]
            host[positions % 3 == 0] = -1
            host[-1] = -1
        ids = tuple(sorted(set(host[host >= 0].tolist())))
        hidden = (torch.randn((tokens, 2048), generator=generator) + shift).bfloat16().to(device)
        if tokens == 3:
            hidden[-1] = float("nan")
        batch = BackendBatch(
            0,
            hidden,
            host.to(device),
            torch.linspace(0.01, 0.99, tokens * 8).reshape(tokens, 8).to(device),
            ids,
        )
        return batch, host

    import json

    boundary_results = []
    samples = {}
    for tokens, shift in [
        (3, 0.0),
        (3, 0.5),
        (3, 1.0),
        (3, 2.0),
        (2048, 0.0),
        (4096, 0.5),
        (64, 1.0),
        (64, 2.0),
    ]:
        with torch.npu.stream(stream):
            batch, host = make_batch(tokens, shift)
            valid_rows = int((host >= 0).sum())
            storage = torch.full((tokens + 2, 2048), 31, dtype=torch.bfloat16, device=device)
            torch.npu.reset_peak_memory_stats()
            before = torch.npu.memory_allocated()
            calls.clear()
            completion = backend.submit(batch, storage[1:-1])
        completion.wait_host()
        assert table.usage_count(0, batch.distinct_expert_ids[0]) == 1
        assert bool((storage[[0, -1]] == 31).all())
        expected = [int((host == e).sum()) for e in batch.distinct_expert_ids]
        assert [(counts.cpu().tolist(), dtype) for counts, dtype in calls] == [
            (expected, torch.int32)
        ] * 3
        peak = torch.npu.max_memory_allocated() - before
        bound = backend.estimate_resources(tokens).temporary_bytes_per_active_batch
        assert peak <= bound
        peaks.append(
            dict(tokens=tokens, assignments=tokens * 8, allocator_peak=peak, estimate=bound)
        )
        measured_output = storage[1:-1].cpu()
        completion.close()
        assert all(table.usage_count(0, e) == 0 for e in batch.distinct_expert_ids)

        # This second pass retains boundaries only after the independent peak pass.
        calls.clear()
        capturing = True
        with torch.npu.stream(stream):
            completion = backend.submit(batch, storage[1:-1])
        completion.wait_host()
        capturing = False
        assert len(captured_quant) == len(captured_projection) == 3
        assert [(counts.cpu().tolist(), dtype) for counts, dtype in calls] == [
            (expected, torch.int32)
        ] * 3
        assert all(table.usage_count(0, e) == 1 for e in batch.distinct_expert_ids)
        _assert_bits(storage[1:-1].cpu(), measured_output)
        assert bool((storage[[0, -1]] == 31).all())
        reference = torch.zeros((tokens, 2048), dtype=torch.float32)
        routing = batch.routing_weights.cpu()
        offset = 0
        for index, expert in enumerate(batch.distinct_expert_ids):
            coordinates = torch.nonzero(host == expert)
            token, route = coordinates[:, 0], coordinates[:, 1]
            count = token.numel()
            weight = weights[expert]

            def reference_projection(
                inputs,
                projection,
                *,
                weight=weight,
                offset=offset,
                count=count,
                index=index,
                tokens=tokens,
                expert=expert,
                shift=shift,
            ):
                with torch.npu.stream(stream):
                    quantized, token_scale = dynamic_quant(inputs, dst_type=torch.int8)
                    native_output = native.npu_quant_matmul(
                        quantized,
                        weight.matrices[projection],
                        weight.scales[projection],
                        pertoken_scale=token_scale,
                        output_dtype=torch.bfloat16,
                    )
                # Reference-only Host reads wait for their non-default producer stream.
                stream.synchronize()
                q, pertoken = quantized.cpu(), token_scale.cpu().reshape(-1)
                matrix, channel = weight.matrices[projection].cpu(), weight.scales[projection].cpu()
                observed_input, observed_q, observed_token = captured_quant[projection]
                observed_dot, _, observed_channel, _, _, observed_output = captured_projection[
                    projection
                ]
                region = slice(offset, offset + count)
                _assert_bits(observed_input[region].cpu(), inputs.cpu())
                _assert_bits(observed_q[region].cpu(), q)
                _assert_bits(observed_token[region].cpu().reshape(-1), pertoken)
                _assert_bits(observed_channel[index].cpu(), channel)
                integer = _cpu_integer_dot(q, matrix)
                _assert_bits(observed_dot[region].cpu(), integer)
                fixed, alternate = _cpu_projection_formulas(integer, channel, pertoken)
                actual_native = native_output.cpu()
                matches = _formula_membership(actual_native, fixed, alternate)
                assert any(matches), (tokens, expert, projection, "unmodeled native formula")
                _assert_bits(observed_output[region].cpu(), fixed)
                boundary_results.append(
                    dict(
                        tokens=tokens,
                        shift=shift,
                        expert=expert,
                        projection=projection,
                        rows=count,
                        elements=fixed.numel(),
                        q_scale_dot_bf16_bitwise=True,
                        native_scale_first=matches[0],
                        native_dot_first=matches[1],
                        native_difference_count=int((actual_native != fixed).sum()),
                        native_max_abs=float((actual_native.float() - fixed.float()).abs().max()),
                    )
                )
                if index == 0:
                    samples[f"{tokens}-{shift}-{projection}"] = dict(
                        q=q[:16].clone(),
                        matrix=matrix[:, :16].clone(),
                        channel=channel[:16].clone(),
                        token=pertoken[:16].clone(),
                        integer=integer[:16, :16].clone(),
                        fixed=fixed[:16, :16].clone(),
                        native=actual_native[:16, :16].clone(),
                    )
                return fixed

            with torch.npu.stream(stream):
                inputs = batch.hidden_states.index_select(0, token.to(device))
            gate = reference_projection(inputs, 0)
            up = reference_projection(inputs, 1)
            with torch.npu.stream(stream):
                activation = torch.nn.functional.silu(gate.to(device)) * up.to(device)
            down = reference_projection(activation, 2)
            _add_ordered_reference(reference, down, token, route, routing)
            offset += count
        assert offset == valid_rows
        _assert_bits(storage[1:-1].cpu(), reference.bfloat16())
        completion.close()
        assert all(table.usage_count(0, e) == 0 for e in batch.distinct_expert_ids)
        captured_quant.clear()
        captured_projection.clear()
        (tmp_path / "resources.json").write_text(json.dumps(peaks, indent=2))
        (tmp_path / "boundaries.json").write_text(json.dumps(boundary_results, indent=2))
        torch.save(samples, tmp_path / "oracle-samples.pt")

    with torch.npu.stream(stream):
        empty = BackendBatch(
            0,
            torch.full((3, 2048), float("nan"), dtype=torch.bfloat16, device=device),
            torch.full((3, 8), -1, dtype=torch.int32, device=device),
            torch.zeros((3, 8), device=device),
            (),
        )
        output = torch.empty_like(empty.hidden_states)
        calls.clear()
        completion = backend.submit(empty, output)
    completion.wait_host()
    completion.close()
    assert bool((output == 0).all()) and not calls

    def fail_after_launch(*args, **kwargs):
        grouped(*args, **kwargs)
        raise RuntimeError("grouped projection failure after launch")

    monkeypatch.setattr(native, "npu_grouped_matmul", fail_after_launch)
    with torch.npu.stream(stream):
        batch, host = make_batch(3, 1.0)
        valid_rows = int((host >= 0).sum())
        with pytest.raises(RuntimeError, match="failure after launch"):
            backend.submit(batch, torch.empty_like(batch.hidden_states))
    assert all(table.usage_count(0, e) == 0 for e in batch.distinct_expert_ids)
