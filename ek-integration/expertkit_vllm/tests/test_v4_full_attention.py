"""Check private V4 attention capacity without importing the native runtime."""

from types import SimpleNamespace

import pytest

from expertkit_vllm.v4_full_attention import attention_capacity, local_builder_config

BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)


@pytest.mark.parametrize("tokens", BUCKETS)
def test_mixed_capacity_covers_reachable_phase_arithmetic(tokens):
    if tokens == 1:
        with pytest.raises(ValueError):
            attention_capacity(tokens, mixed=True)
        return
    capacity = attention_capacity(tokens, mixed=True)
    for decodes in range(min(255, tokens - 2) + 1):
        remaining = tokens - decodes
        for prefills in range(1, min(256 - decodes, remaining // 2) + 1):
            lengths = [1] * decodes + [2] * (prefills - 1)
            lengths.append(remaining - 2 * (prefills - 1))
            capacity.validate_population(lengths, decodes)
    assert capacity.tokens <= 2304
    assert capacity.requests <= 512


@pytest.mark.parametrize("tokens", BUCKETS[:9])
def test_pure_decode_keeps_native_geometry(tokens):
    capacity = attention_capacity(tokens, mixed=False)
    capacity.validate_population([1] * tokens, tokens)
    assert capacity.tokens == capacity.requests == tokens
    assert capacity.prefill_tokens == capacity.prefill_requests == 0


def test_population_rejects_classifier_and_capacity_changes():
    capacity = attention_capacity(8, mixed=True)
    for lengths, decodes in (([1], 1), ([1, 1], 0), ([2], 1), ([0, 2], 1), ([9], 0)):
        with pytest.raises(ValueError):
            capacity.validate_population(lengths, decodes)
    with pytest.raises(ValueError):
        attention_capacity(512, mixed=False)


def test_builder_private_start_and_slot_buffers_do_not_mutate_outer_config():
    compilation = SimpleNamespace(static_forward_context={"layer": object()})
    outer = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=256, max_num_batched_tokens=2048),
        cache_config=SimpleNamespace(block_size=128, num_gpu_blocks_override=2048),
        compilation_config=compilation,
    )
    for tokens in BUCKETS[1:]:
        capacity = attention_capacity(tokens, mixed=True)
        local = local_builder_config(outer, capacity)
        assert local.scheduler_config.max_num_seqs == capacity.requests
        assert local.scheduler_config.max_num_batched_tokens == capacity.tokens
        assert local.compilation_config is compilation
        assert local.cache_config is not outer.cache_config
        assert local.cache_config.num_gpu_blocks_override == 2048
    assert outer.scheduler_config.max_num_seqs == 256
    assert outer.scheduler_config.max_num_batched_tokens == 2048
    assert not hasattr(outer, "topk_indices_buffer")


@pytest.mark.parametrize("tokens", BUCKETS[1:])
def test_pack_gather_preserves_every_real_row_and_isolates_virtual_rows(tokens):
    from expertkit_vllm.v4_full_attention import attention_population

    capacity = attention_capacity(tokens, mixed=True)
    decodes = min(255, tokens - 2)
    lengths = [1] * decodes + [tokens - decodes]
    population = attention_population(capacity, lengths, decodes)
    assert sum(population.physical_lengths) == tokens
    assert len(population.physical_lengths) == capacity.requests
    assert population.request_sources.count(None) == capacity.requests - len(lengths)
    assert len(set(population.valid_rows)) == tokens
    for outer, physical in enumerate(population.valid_rows):
        assert population.pack_sources[physical] == outer
        assert population.gather_sources[outer] == physical
    assert all(
        source == tokens
        for physical, source in enumerate(population.pack_sources)
        if physical not in set(population.valid_rows)
    )


def test_short_population_gathers_outer_tail_from_dedicated_sentinel():
    from expertkit_vllm.v4_full_attention import attention_population

    capacity = attention_capacity(2048, mixed=True)
    population = attention_population(capacity, [1, 1, 1, 128], 3)
    assert population.valid_rows == tuple(range(3)) + tuple(range(256, 384))
    assert population.gather_sources[131:] == (2304,) * (2048 - 131)
    assert population.physical_lengths[3:256] == (0,) * 253
    assert population.request_sources[256] == 3


def test_dummy_pages_are_disjoint_from_all_logical_pages_and_other_owners():
    from expertkit_vllm.v4_full_attention import independent_dummy_page

    pages = {independent_dummy_page(group, owner) for group in range(6) for owner in range(512)}
    assert pages == set(range(2048, 5120))
    for group, owner in ((-1, 0), (6, 0), (0, -1), (0, 512)):
        with pytest.raises(ValueError):
            independent_dummy_page(group, owner)


def test_metadata_snapshot_detaches_builder_reused_nested_tensors():
    from dataclasses import dataclass
    from enum import Enum

    import torch

    from expertkit_vllm.v4_full_attention import _clone_metadata, _update_metadata

    class Phase(Enum):
        MIXED = "mixed"

    @dataclass
    class Metadata:
        phase: Phase
        slots: object
        opaque: object

    source = Metadata(Phase.MIXED, torch.arange(4), {"sas": [torch.arange(3), (None,)]})
    frozen = _clone_metadata(source)
    pointers = (frozen.slots.data_ptr(), frozen.opaque["sas"][0].data_ptr())
    source.slots.fill_(7)
    source.opaque["sas"][0].fill_(8)
    assert frozen.slots.tolist() == list(range(4))
    assert frozen.opaque["sas"][0].tolist() == list(range(3))
    _update_metadata(frozen, source)
    assert pointers == (frozen.slots.data_ptr(), frozen.opaque["sas"][0].data_ptr())
    assert frozen.slots.tolist() == [7] * 4
    assert frozen.opaque["sas"][0].tolist() == [8] * 3
    with pytest.raises(ValueError):
        _update_metadata(frozen, Metadata(Phase.MIXED, torch.zeros(5), source.opaque))
    with pytest.raises(TypeError):
        _clone_metadata(SimpleNamespace(unowned=source.slots))


def test_stable_metadata_storage_keeps_each_role_address_and_isolates_builder():
    import torch

    from expertkit_vllm.v4_full_attention import _NativeMetadataStorage

    store = _NativeMetadataStorage()
    shared = torch.tensor([1, 2])
    first = store.update("swa", {"slots": shared})
    pointer = first["slots"].data_ptr()
    shared.fill_(3)
    second = store.update("compressed", {"slots": shared})
    assert first["slots"].tolist() == [1, 2]
    assert first["slots"].data_ptr() != second["slots"].data_ptr()
    assert store.update("swa", {"slots": shared}) is first
    assert first["slots"].data_ptr() == pointer
    assert first["slots"].tolist() == [3, 3]
    store.close()
    assert not store.values


def test_owned_bindings_restore_inherited_and_local_methods_on_failure():
    from expertkit_vllm.v4_full_attention import _OwnedBindings

    class Runner:
        def build(self):
            return "native"

    runner = Runner()
    runner.local = object()
    original_local = runner.local
    bindings = _OwnedBindings()

    def replacement():
        return "private"

    try:
        bindings.bind(runner, "build", replacement)
        bindings.bind(runner, "local", replacement)
        assert runner.build() == "private"
        raise ValueError("fake native failure")
    except ValueError:
        bindings.close()
    assert runner.build() == "native"
    assert "build" not in vars(runner)
    assert runner.local is original_local


def test_owned_bindings_do_not_remove_foreign_replacements():
    from expertkit_vllm.v4_full_attention import _OwnedBindings

    runner = SimpleNamespace(build=lambda: None)
    bindings = _OwnedBindings()
    bindings.bind(runner, "build", lambda: None)
    runner.build = lambda: "foreign"
    with pytest.raises(RuntimeError, match="lost ownership"):
        bindings.close()
    assert runner.build() == "foreign"


def test_attention_buffers_pack_gather_never_propagate_dummy_nan():
    import torch

    from expertkit_vllm.v4_full_attention import _AttentionBuffers, attention_population

    capacity = attention_capacity(8, mixed=True)
    population = attention_population(capacity, [1, 3], 1)
    buffers = _AttentionBuffers(capacity, 2, torch.float32, "cpu")
    buffers.update_population(population)
    original = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    packed = buffers.pack(original)
    for outer, physical in enumerate(population.valid_rows):
        assert torch.equal(packed[physical], original[outer])
    buffers.output[:-1].fill_(float("nan"))
    for outer, physical in enumerate(population.valid_rows):
        buffers.output[physical].copy_(original[outer])
    target = torch.empty_like(original)
    buffers.gather(target)
    assert torch.equal(target[:4], original[:4])
    assert torch.equal(target[4:], torch.zeros_like(target[4:]))
    assert torch.isfinite(target).all()
    assert buffers.output[-1].data_ptr() != buffers.output[0].data_ptr()


def test_attention_context_preserves_outer_fields_and_restores_on_native_error():
    from expertkit_vllm.v4_full_attention import _attention_context

    outer = SimpleNamespace(
        num_tokens=2048,
        attn_metadata={},
        additional_kwargs={"sentinel": object()},
        capturing=True,
        communication=object(),
    )
    module = SimpleNamespace(_forward_context=outer)
    module.get_forward_context = lambda: module._forward_context
    metadata = {"layer": object()}
    private_kwargs = dict(outer.additional_kwargs)
    with (
        pytest.raises(RuntimeError, match="native"),
        _attention_context(module, metadata, 2304, private_kwargs) as local,
    ):
        assert module.get_forward_context() is local
        assert local.num_tokens == 2304
        assert local.communication is outer.communication
        assert local.capturing is outer.capturing
        assert local.additional_kwargs is private_kwargs
        assert outer.num_tokens == 2048
        raise RuntimeError("native failure")
    assert module.get_forward_context() is outer
    assert outer.attn_metadata == {}


def test_physical_pool_clone_preserves_logical_object_and_every_alias():
    from copy import deepcopy

    from expertkit_vllm.v4_full_attention import physical_cache_config

    groups = []
    for group in range(6):
        names = [f"layer{index}.group{group}" for index in range(28 if group < 5 else 27)]
        groups.append(
            SimpleNamespace(
                layer_names=names,
                kv_cache_spec=SimpleNamespace(kv_cache_specs={name: object() for name in names}),
            )
        )
    roles = [name for group in groups for name in group.layer_names]
    logical = SimpleNamespace(
        num_blocks=2048,
        kv_cache_groups=groups,
        kv_cache_tensors=[
            SimpleNamespace(
                size=2048 * (index + 1),
                offset=0,
                block_stride=0,
                shared_by=[roles[index], roles[index + 1]],
            )
            for index in range(44)
        ],
    )
    before_sizes = [raw.size for raw in logical.kv_cache_tensors]
    before_aliases = [tuple(raw.shared_by) for raw in logical.kv_cache_tensors]
    physical = physical_cache_config(logical)
    assert logical.num_blocks == 2048
    assert [raw.size for raw in logical.kv_cache_tensors] == before_sizes
    assert physical.num_blocks == 5120
    assert [raw.size for raw in physical.kv_cache_tensors] == [
        size // 2048 * 5120 for size in before_sizes
    ]
    assert [tuple(raw.shared_by) for raw in physical.kv_cache_tensors] == before_aliases
    physical.kv_cache_tensors[0].shared_by.clear()
    assert logical.kv_cache_tensors[0].shared_by
    wrong = deepcopy(logical)
    wrong.kv_cache_tensors[0].shared_by.append("unknown")
    with pytest.raises(ValueError, match="allocation layout"):
        physical_cache_config(wrong)


def test_actual_adapter_preparation_uses_native_cpu_metadata_and_fresh_step_dicts():
    import torch

    from expertkit_vllm.v4_full_attention import V4FullAttention

    adapter = object.__new__(V4FullAttention)
    adapter.runner = SimpleNamespace(
        model_config=SimpleNamespace(get_hidden_size=lambda: 2, dtype=torch.float32), device="cpu"
    )
    adapter.graph_config = SimpleNamespace(capture_sizes=BUCKETS)
    adapter._pending_tokens = 8
    adapter._capture_phase = None
    adapter._phase = None
    adapter._closed = False
    adapter._buckets = {}
    adapter._immutable = {}
    adapter._step = None
    common = SimpleNamespace(
        query_start_loc_cpu=torch.tensor([0, 1, 4]),
        seq_lens_cpu=torch.tensor([128, 130]),
        positions_cpu=torch.tensor([127, 127, 128, 129]),
    )
    adapter._prepare_step(common, 2)
    assert adapter.phase == "mixed"
    first = adapter._step
    assert first["population"].valid_rows == (0, 8, 9, 10)
    assert first["starts_cpu"].tolist() == [127] + [0] * 7 + [127] + [0] * 3
    assert first["positions_cpu"].tolist() == [127] + [0] * 7 + [127, 128, 129] + [0] * 5
    first["common_shared"]["stale"] = object()
    adapter._step = None
    adapter._prepare_step(common, 2)
    assert adapter._step["common_shared"] == {}
    assert adapter._step["prefill_shared"] is not first["prefill_shared"]
    assert adapter._step["decode_shared"] is not first["decode_shared"]


def test_capture_scope_prepares_one_positive_prefill_for_warmup_too():
    import torch

    from expertkit_vllm.v4_full_attention import V4FullAttention

    adapter = object.__new__(V4FullAttention)
    adapter.runner = SimpleNamespace(
        model_config=SimpleNamespace(get_hidden_size=lambda: 2, dtype=torch.float32), device="cpu"
    )
    adapter.graph_config = SimpleNamespace(capture_sizes=BUCKETS)
    adapter._pending_tokens = 8
    adapter._capture_phase = None
    adapter._phase = None
    adapter._closed = False
    adapter._buckets = {}
    adapter._immutable = {}
    adapter._step = None
    common = SimpleNamespace(query_start_loc_cpu=torch.arange(9))
    with adapter.capture_phase("mixed"):
        # Warmup has no for_cudagraph_capture flag and still needs its carrier.
        adapter._prepare_step(common, 8)
        assert adapter.phase == "mixed"
        assert adapter._step["population"].query_lengths == (8,)
        assert adapter._step["population"].num_decodes == 0
        assert adapter._step["capturing_mixed"]
    assert adapter._capture_phase is None


def test_native_impl_wrapper_restores_outer_context_and_gathers_real_rows(monkeypatch):
    import sys
    import types

    import torch

    from expertkit_vllm.v4_full_attention import (
        V4FullAttention,
        _AttentionBuffers,
        _OwnedBindings,
        attention_population,
    )

    outer = SimpleNamespace(
        num_tokens=8,
        additional_kwargs={
            "dsv4_compressor_metadata_cache": {"old": object()},
            "sentinel": object(),
        },
        attn_metadata={"native": object()},
        capturing=True,
        communication=object(),
    )
    root = types.ModuleType("vllm")
    context = types.ModuleType("vllm.forward_context")
    context._forward_context = outer
    context.get_forward_context = lambda: context._forward_context
    root.forward_context = context
    monkeypatch.setitem(sys.modules, "vllm", root)
    monkeypatch.setitem(sys.modules, "vllm.forward_context", context)
    capacity = attention_capacity(8, mixed=True)
    buffers = _AttentionBuffers(capacity, 2, torch.float32, "cpu")
    buffers.update_population(attention_population(capacity, [1, 3], 1))
    calls = []

    def native(layer, hidden, cache, metadata, gather, output):
        local = context.get_forward_context()
        assert local.num_tokens == 16
        assert local.communication is outer.communication
        assert "dsv4_compressor_metadata_cache" not in local.additional_kwargs
        assert outer.additional_kwargs["dsv4_compressor_metadata_cache"]
        output.copy_(hidden * 2)
        calls.append((layer, cache, metadata, gather))

    impl = SimpleNamespace(forward=native)
    adapter = object.__new__(V4FullAttention)
    adapter._bindings = _OwnedBindings()
    adapter._phase = "mixed"
    adapter._capture_phase = None
    adapter._closed = False
    adapter._step = {
        "bucket": {"capacity": capacity, "buffers": buffers},
        "additional_kwargs": None,
    }
    adapter._bind_implementation(impl)
    hidden = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    output = torch.empty_like(hidden)
    cache, metadata = object(), object()
    assert impl.forward("layer", hidden, cache, metadata, False, output) is output
    assert calls == [("layer", cache, metadata, False)]
    assert torch.equal(output[:4], hidden[:4] * 2)
    assert torch.equal(output[4:], torch.zeros_like(output[4:]))
    assert context.get_forward_context() is outer
    adapter._bindings.close()
    assert impl.forward is native


def test_only_registered_global_tensor_identity_is_shared_not_dynamic_rope():
    import torch

    from expertkit_vllm.v4_full_attention import _clone_metadata, _update_metadata

    class RopeDataProxy:
        def __init__(self, data, is_cos):
            self._data = data
            self.idx = 0 if is_cos else 1

    RopeDataProxy.__module__ = "vllm_ascend.ops.rope_dsv4"
    full_cos = torch.arange(8)
    dynamic = torch.arange(4)
    source = {"full": full_cos, "dynamic": RopeDataProxy({"c4": (dynamic, dynamic + 1)}, True)}
    registry = {id(full_cos): full_cos}
    frozen = _clone_metadata(source, registry)
    assert frozen["full"] is full_cos
    assert frozen["dynamic"] is not source["dynamic"]
    assert frozen["dynamic"]._data["c4"][0].data_ptr() != dynamic.data_ptr()
    dynamic.fill_(9)
    assert frozen["dynamic"]._data["c4"][0].tolist() == list(range(4))
    _update_metadata(frozen, source, registry)
    assert frozen["dynamic"]._data["c4"][0].tolist() == [9] * 4
    with pytest.raises(ValueError, match="immutable metadata identity"):
        _update_metadata(frozen, {**source, "full": full_cos.clone()}, registry)


def test_mixed_builder_publishes_fresh_dicts_to_original_runner_postbuild_consumer():
    from expertkit_vllm.v4_full_attention import V4FullAttention, _OwnedBindings

    class NativeBuilder:
        def build(self, common_prefix_len, common_attn_metadata, fast_build=False, **kwargs):
            raise AssertionError("mixed preparation must delegate to its private native builder")

    class PrivateNativeBoundary(V4FullAttention):
        def _prepare_step(self, common, requests):
            assert common is marker and requests == 1
            self._phase = "mixed"
            self._step = {"prefill_shared": {}, "decode_shared": {}, "common_shared": {}}

        def _build_local(self, builder, group_id, layer_names, common):
            self._step["prefill_shared"]["native_sas"] = marker
            return marker

    marker = object()
    adapter = object.__new__(PrivateNativeBoundary)
    adapter._bindings = _OwnedBindings()
    adapter._closed = False
    adapter._capture_phase = None
    builder = NativeBuilder()
    adapter._bind_builder(builder, 0, ["layer"])
    # The exact NPU runner reads these original builder properties afterwards,
    # including the first mixed build before any original Decode build ran.
    assert builder.build(0, marker, num_reqs_actual=1) is marker
    first = builder.prefill_ratio_to_sas_metadata
    assert first["native_sas"] is marker
    assert builder.decode_ratio_to_sas_metadata is adapter._step["decode_shared"]
    assert builder.common_ratio_to_sas_metadata is adapter._step["common_shared"]
    assert builder.build(0, marker, num_reqs_actual=1) is marker
    assert builder.prefill_ratio_to_sas_metadata is not first
    assert builder.prefill_ratio_to_sas_metadata is adapter._step["prefill_shared"]
    adapter._bindings.close()


@pytest.mark.parametrize("tokens", (2, 64, 2048))
@pytest.mark.parametrize(
    "group,block,ratio,width",
    (
        (0, 128, 4, 16),
        (1, 128, 128, 1),
        (2, 128, 1, 64),
        (3, 128, 1, 64),
        (4, 8, 1, 1024),
        (5, 32, 1, 256),
    ),
)
def test_native_capture_carrier_keeps_group_width_and_runtime_slots(
    monkeypatch, tokens, group, block, ratio, width
):
    import sys

    import torch

    from expertkit_vllm.v4_full_attention import V4FullAttention, capture_carrier_pages

    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.attention.attention_v1",
        SimpleNamespace(AscendAttentionState=SimpleNamespace(ChunkedPrefill=1)),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ops.rope_dsv4",
        SimpleNamespace(get_cos_and_sin_dsa=lambda positions: (positions, positions)),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.attention.dsa_v1",
        SimpleNamespace(AscendDSAMetadataBuilder=SimpleNamespace(hadamard=None)),
    )
    received = []

    class Builder:
        def __init__(self, spec, *args):
            self.kv_cache_spec = spec

        def build(self, prefix, common, **kwargs):
            received.append(common)
            return SimpleNamespace(
                prefill=SimpleNamespace(), decode=SimpleNamespace(), hadamard=None
            )

    adapter = object.__new__(V4FullAttention)
    adapter.runner = SimpleNamespace(
        model_config=SimpleNamespace(get_hidden_size=lambda: 2, dtype=torch.float32), device="cpu"
    )
    adapter.graph_config = SimpleNamespace(capture_sizes=BUCKETS)
    adapter._pending_tokens = tokens
    adapter._capture_phase = "mixed"
    adapter._phase = None
    adapter._closed = False
    adapter._buckets = {}
    adapter._immutable = {}
    adapter._step = None
    adapter._capture_page_starts = [5120] * 6
    adapter._capture_group_ratios = [4, 128, 1, 1, 1, 1]
    common = SimpleNamespace(
        query_start_loc_cpu=torch.tensor([0, tokens]),
        seq_lens_cpu=torch.tensor([127 + tokens]),
        positions_cpu=torch.arange(127, 127 + tokens),
        block_table_tensor=torch.arange(width, dtype=torch.int32).reshape(1, width),
        slot_mapping=torch.arange(tokens, dtype=torch.int64) + 123,
    )
    builder = Builder(SimpleNamespace(block_size=block, compress_ratio=ratio))
    adapter._prepare_step(common, 1)
    bucket = adapter._step["bucket"]
    bucket["builders"]["layer"] = builder
    bucket["metadata"] = SimpleNamespace(update=lambda key, value: value)
    adapter._build_local(builder, group, ["layer"], common)
    captured = received[-1]
    assert captured.block_table_tensor.shape == (bucket["capacity"].requests, width)
    pages = capture_carrier_pages(block, ratio, tokens)
    owner = bucket["capacity"].decode_requests
    assert captured.block_table_tensor[owner, :pages].tolist() == list(range(5120, 5120 + pages))
    if ratio > 1:
        assert (captured.slot_mapping == -1).all()
    else:
        positions = torch.arange(127, 127 + tokens)
        expected = (5120 + positions // block) * block + positions % block
        assert torch.equal(
            captured.slot_mapping[list(adapter._step["population"].valid_rows)], expected
        )
    adapter._capture_phase = None
    adapter._step = None
    adapter._prepare_step(common, 1)
    adapter._build_local(builder, group, ["layer"], common)
    runtime = received[-1]
    assert runtime.block_table_tensor.shape == captured.block_table_tensor.shape
    assert torch.equal(runtime.block_table_tensor[owner], common.block_table_tensor[0])
    assert torch.equal(
        runtime.slot_mapping[list(adapter._step["population"].valid_rows)], common.slot_mapping
    )


def test_capture_group_ratio_rejects_nonuniform_native_group():
    from expertkit_vllm.v4_full_attention import capture_carrier_pages, capture_group_ratio

    with pytest.raises(ValueError, match="inconsistent"):
        capture_group_ratio(
            SimpleNamespace(
                kv_cache_specs={
                    "a": SimpleNamespace(compress_ratio=4),
                    "b": SimpleNamespace(compress_ratio=128),
                }
            )
        )
    for invalid in (True, 1.0):
        with pytest.raises(ValueError, match="positive integer"):
            capture_group_ratio(
                SimpleNamespace(
                    kv_cache_specs={
                        "a": SimpleNamespace(compress_ratio=1),
                        "b": SimpleNamespace(compress_ratio=invalid),
                    }
                )
            )
    assert capture_carrier_pages(128, 128, 2) == 1
    assert capture_carrier_pages(128, 4, 2048) == 5
    assert capture_carrier_pages(128, 128, 2048) == 1


def test_carrier_pages_cover_every_mixed_bucket_native_compressor_rows():
    from expertkit_vllm.v4_full_attention import capture_carrier_pages

    for tokens in BUCKETS[1:]:
        for block, ratio, width in (
            (128, 4, 16),
            (128, 128, 1),
            (128, 1, 64),
            (8, 1, 1024),
            (32, 1, 256),
        ):
            start = 127 // ratio
            rows = (127 + tokens) // ratio - start
            assert rows > 0
            pages = capture_carrier_pages(block, ratio, tokens)
            assert (start + rows - 1) // block < pages <= width


@pytest.mark.parametrize("actual,bucket", ((194, 256), (255, 256), (1, 2), (256, 256)))
def test_decode_padding_uses_uniform_native_geometry_and_restores_owner(
    monkeypatch, actual, bucket
):
    import sys

    import numpy as np

    from expertkit_vllm.v4_full_attention import V4FullAttention, _OwnedBindings

    full = object()
    monkeypatch.setitem(
        sys.modules, "vllm.config", SimpleNamespace(CUDAGraphMode=SimpleNamespace(FULL=full))
    )
    calls = []
    uploads = []
    array = np.arange(257, dtype=np.int32)
    array[actual + 1 :] = -777
    if actual < bucket:
        array[actual + 1] = bucket
    qloc = SimpleNamespace(np=array, copy_to_gpu=lambda: uploads.append(True))

    def native(
        query_start_loc,
        num_tokens_padded,
        num_reqs_padded,
        num_reqs,
        cudagraph_runtime_mode=None,
        batch_desc_num_reqs=None,
    ):
        calls.append(
            (
                num_tokens_padded,
                num_reqs_padded,
                num_reqs,
                cudagraph_runtime_mode,
                batch_desc_num_reqs,
            )
        )
        if array[num_reqs] < num_tokens_padded:
            array[num_reqs + 1] = num_tokens_padded
            num_reqs += 1
        query_start_loc.copy_to_gpu()
        return num_reqs

    runner = SimpleNamespace(
        _pad_query_start_loc_for_fia=native, arange_np=np.arange(257, dtype=np.int32)
    )
    adapter = object.__new__(V4FullAttention)
    adapter.runner = runner
    adapter._capture_phase = None
    adapter._bindings = _OwnedBindings()
    adapter._closed = False
    adapter._buckets = {}
    adapter._immutable = {}
    adapter._bind_decode_padding()
    result = runner._pad_query_start_loc_for_fia(qloc, bucket, bucket, actual, full, bucket)
    assert result == bucket
    assert array[: bucket + 1].tolist() == list(range(bucket + 1))
    assert uploads == [True]
    assert calls == []
    adapter.close()
    assert runner._pad_query_start_loc_for_fia is native


def test_decode_padding_delegates_mixed_capture_and_nonuniform_population(monkeypatch):
    import sys

    import numpy as np

    from expertkit_vllm.v4_full_attention import V4FullAttention, _OwnedBindings

    full = object()
    monkeypatch.setitem(
        sys.modules, "vllm.config", SimpleNamespace(CUDAGraphMode=SimpleNamespace(FULL=full))
    )
    calls = []

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return "native"

    runner = SimpleNamespace(_pad_query_start_loc_for_fia=original, arange_np=np.arange(257))
    adapter = object.__new__(V4FullAttention)
    adapter.runner = runner
    adapter._bindings = _OwnedBindings()
    adapter._capture_phase = None
    adapter._bind_decode_padding()
    for phase, mode, actual, bucket, values in (
        ("mixed", full, 2, 4, [0, 1, 2]),
        (None, full, 2, 4, [0, 1, 3]),
        (None, None, 2, 4, [0, 1, 2]),
        (None, full, 256, 512, list(range(257))),
        (None, full, 0, 1, [0]),
    ):
        adapter._capture_phase = phase
        qloc = SimpleNamespace(np=np.array(values))
        arguments = (qloc, bucket, bucket, actual, mode, bucket)
        assert runner._pad_query_start_loc_for_fia(*arguments) == "native"
        assert calls[-1] == (arguments, {})
    adapter._bindings.close()


@pytest.mark.parametrize(
    "lengths,swap",
    [
        ([1] * 22 + [184, 85, 380, 76, 50, 82, 14, 29, 20, 27, 1, 642, 32, 80, 324], (22, 32)),
        ([1] * 194, None),
        ([1, 1, 4, 8], None),
    ],
)
def test_batch_reorder_delegates_native_dsa_and_restores_identity(monkeypatch, lengths, swap):
    import sys

    from expertkit_vllm.v4_full_attention import V4FullAttention, _OwnedBindings

    calls = []

    class NativeBuilder:
        decode_threshold = 1

        def reorder_batch(self, batch, scheduled):
            calls.append((batch, scheduled))
            if swap is not None:
                batch.swap_states(*swap)

    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.attention.dsa_v1",
        SimpleNamespace(AscendDSAMetadataBuilder=NativeBuilder),
    )
    ids = list(range(len(lengths)))
    states = [(rid, f"slot-{rid}", f"cache-{rid}") for rid in ids]
    batch = SimpleNamespace(req_ids=ids, states=states)

    def swap_states(left, right):
        ids[left], ids[right] = ids[right], ids[left]
        states[left], states[right] = states[right], states[left]

    batch.swap_states = swap_states
    scheduled = SimpleNamespace(num_scheduled_tokens=dict(enumerate(lengths)))
    before = list(states)

    def original(output):
        assert output is scheduled
        assert calls == []
        return "native-result"

    builder = NativeBuilder()
    runner = SimpleNamespace(
        _may_reorder_batch=original,
        input_batch=batch,
        attn_groups=[[SimpleNamespace(get_metadata_builder=lambda index: builder)]],
    )
    adapter = object.__new__(V4FullAttention)
    adapter.runner = runner
    adapter._bindings = _OwnedBindings()
    adapter._closed = False
    adapter._buckets = {}
    adapter._immutable = {}

    def validate():
        ordered = [scheduled.num_scheduled_tokens[rid] for rid in ids]
        decodes = next((i for i, n in enumerate(ordered) if n != 1), len(ordered))
        capacity = attention_capacity(2048 if swap else sum(lengths), mixed=decodes < len(ids))
        capacity.validate_population(ordered, decodes)

    if swap:
        with pytest.raises(ValueError, match="native phase capacity"):
            validate()
    else:
        validate()
    adapter._bind_batch_reorder()
    assert runner._may_reorder_batch(scheduled) == "native-result"
    assert calls == [(batch, scheduled)]
    validate()
    assert sorted(states) == before
    assert all(state[0] == rid for rid, state in zip(ids, states, strict=True))
    if swap is None:
        assert states == before
    adapter.close()
    assert runner._may_reorder_batch is original
