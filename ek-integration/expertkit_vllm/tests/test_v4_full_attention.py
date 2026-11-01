"""Check private V4 attention capacity without importing the native runtime."""

from types import SimpleNamespace

import pytest

from expertkit_vllm.v4_full_attention import (
    _AttentionBuffers,
    _NativeMetadataStorage,
    attention_capacity,
    local_builder_config,
)

BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)


def _test_buckets():
    import torch

    return {
        c: {
            "capacity": (capacity := attention_capacity(c, max_requests=256, mixed=True)),
            "buffers": _AttentionBuffers(capacity, 2, torch.float32, "cpu"),
            "metadata": _NativeMetadataStorage(),
            "builders": {},
        }
        for c in BUCKETS[1:]
    }


@pytest.mark.parametrize("tokens", BUCKETS)
def test_mixed_capacity_covers_reachable_phase_arithmetic(tokens):
    if tokens == 1:
        with pytest.raises(ValueError):
            attention_capacity(tokens, max_requests=256, mixed=True)
        return
    capacity = attention_capacity(tokens, max_requests=256, mixed=True)
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
    capacity = attention_capacity(tokens, max_requests=256, mixed=False)
    capacity.validate_population([1] * tokens, tokens)
    assert capacity.tokens == capacity.requests == tokens
    assert capacity.prefill_tokens == capacity.prefill_requests == 0


def test_population_rejects_classifier_and_capacity_changes():
    capacity = attention_capacity(8, max_requests=256, mixed=True)
    for lengths, decodes in (([1], 1), ([1, 1], 0), ([2], 1), ([0, 2], 1), ([9], 0)):
        with pytest.raises(ValueError):
            capacity.validate_population(lengths, decodes)
    with pytest.raises(ValueError):
        attention_capacity(512, max_requests=256, mixed=False)


def test_builder_private_start_and_slot_buffers_do_not_mutate_outer_config():
    compilation = SimpleNamespace(static_forward_context={"layer": object()})
    outer = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=256, max_num_batched_tokens=2048),
        cache_config=SimpleNamespace(block_size=128, num_gpu_blocks_override=2048),
        compilation_config=compilation,
    )
    for tokens in BUCKETS[1:]:
        capacity = attention_capacity(tokens, max_requests=256, mixed=True)
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

    capacity = attention_capacity(tokens, max_requests=256, mixed=True)
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

    capacity = attention_capacity(2048, max_requests=256, mixed=True)
    population = attention_population(capacity, [1, 1, 1, 128], 3)
    assert population.valid_rows == tuple(range(3)) + tuple(range(256, 384))
    assert population.gather_sources[131:] == (2304,) * (2048 - 131)
    assert population.physical_lengths[3:256] == (0,) * 253
    assert population.request_sources[256] == 3


@pytest.mark.parametrize("logical_blocks", (113, 2048, 4097))
@pytest.mark.parametrize("groups,owners", ((2, 7), (6, 512), (3, 19)))
def test_dummy_pages_are_disjoint_from_all_logical_pages_and_other_owners(
    logical_blocks, groups, owners
):
    from expertkit_vllm.v4_full_attention import V4KVScratchLayout, independent_dummy_page

    layout = V4KVScratchLayout(owners, (32,) * groups, (1,) * groups, (5,) * groups)
    pages = {
        independent_dummy_page(layout, logical_blocks=logical_blocks, group=g, owner=o)
        for g in range(groups)
        for o in range(owners)
    }
    assert pages == set(range(logical_blocks, logical_blocks + groups * owners))
    assert min(layout.carrier_start(logical_blocks, g) for g in range(groups)) > max(pages)
    for group, owner in ((-1, 0), (groups, 0), (0, -1), (0, owners)):
        with pytest.raises(ValueError):
            independent_dummy_page(layout, logical_blocks=logical_blocks, group=group, owner=owner)


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

    capacity = attention_capacity(8, max_requests=256, mixed=True)
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


@pytest.mark.parametrize("padding", (False, True))
def test_physical_pool_clone_preserves_logical_object_and_every_alias(padding):
    from copy import deepcopy

    from expertkit_vllm.v4_full_attention import kv_scratch_layout, physical_cache_config

    groups = [
        SimpleNamespace(
            layer_names=names,
            kv_cache_spec=SimpleNamespace(
                kv_cache_specs={name: SimpleNamespace(block_size=32) for name in names}
            ),
        )
        for names in (("a", "b"), ("c", "d"))
    ]
    logical = SimpleNamespace(
        num_blocks=113,
        kv_cache_groups=groups,
        kv_cache_tensors=[
            SimpleNamespace(size=113 * stride, offset=0, block_stride=0, shared_by=names)
            for stride, names in ((2, ["a", "c"]), (3, ["b", "d"]))
        ],
    )
    if padding:
        # Native V4 planners retain empty page-size slots in the allocation plan.
        logical.kv_cache_tensors.append(
            SimpleNamespace(size=113 * 3, offset=0, block_stride=0, shared_by=[])
        )
    before = deepcopy(logical)
    layout = kv_scratch_layout(groups, max_virtual_owners=7, mixed_capture_sizes=(3, 7))
    physical = physical_cache_config(logical, layout)
    assert logical == before
    assert physical.num_blocks == logical.num_blocks + layout.extra_pages
    assert [r.size for r in physical.kv_cache_tensors] == [
        2 * physical.num_blocks,
        3 * physical.num_blocks,
        *([3 * physical.num_blocks] if padding else []),
    ]
    assert [r.shared_by for r in physical.kv_cache_tensors] == [
        r.shared_by for r in logical.kv_cache_tensors
    ]
    physical.kv_cache_tensors[0].shared_by.clear()
    assert logical == before
    for change in ("unknown", "missing", "duplicate", "stride", "offset", "length"):
        wrong = deepcopy(logical)
        if change == "unknown":
            wrong.kv_cache_tensors[0].shared_by.append("unknown")
        elif change == "missing":
            wrong.kv_cache_tensors[0].shared_by.pop()
        elif change == "duplicate":
            wrong.kv_cache_groups[0].layer_names += ("a",)
        elif change == "stride":
            wrong.kv_cache_tensors[-1 if padding else 0].block_stride = 2
        elif change == "offset":
            wrong.kv_cache_tensors[-1 if padding else 0].offset = 1
        else:
            wrong.kv_cache_tensors[-1 if padding else 0].size += 1
        with pytest.raises(ValueError):
            physical_cache_config(wrong, layout)


def test_actual_adapter_preparation_uses_native_cpu_metadata_and_fresh_step_dicts():
    import torch

    from expertkit_vllm.v4_full_attention import V4FullAttention

    adapter = object.__new__(V4FullAttention)
    adapter._max_requests = 256
    adapter._profile_preparing = False
    adapter.graph_config = SimpleNamespace(capture_sizes=BUCKETS, decode_capture_sizes=BUCKETS[:9])
    adapter.runner = SimpleNamespace(
        model_config=SimpleNamespace(get_hidden_size=lambda: 2, dtype=torch.float32), device="cpu"
    )
    adapter.graph_config = SimpleNamespace(capture_sizes=BUCKETS, decode_capture_sizes=BUCKETS[:9])
    adapter._pending_tokens = 8
    adapter._capture_phase = None
    adapter._phase = None
    adapter._closed = False
    adapter._buckets = _test_buckets()
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
    adapter._max_requests = 256
    adapter._profile_preparing = False
    adapter.graph_config = SimpleNamespace(capture_sizes=BUCKETS, decode_capture_sizes=BUCKETS[:9])
    adapter.runner = SimpleNamespace(
        model_config=SimpleNamespace(get_hidden_size=lambda: 2, dtype=torch.float32), device="cpu"
    )
    adapter.graph_config = SimpleNamespace(capture_sizes=BUCKETS, decode_capture_sizes=BUCKETS[:9])
    adapter._pending_tokens = 8
    adapter._capture_phase = None
    adapter._phase = None
    adapter._closed = False
    adapter._buckets = _test_buckets()
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
    capacity = attention_capacity(8, max_requests=256, mixed=True)
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
    adapter._max_requests = 256
    adapter._profile_preparing = False
    adapter.graph_config = SimpleNamespace(capture_sizes=BUCKETS, decode_capture_sizes=BUCKETS[:9])
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
    builder.kv_cache_spec = object()
    adapter._builder_signatures = {(0, "layer"): (type(builder), builder.kv_cache_spec, ("layer",))}
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
    adapter._max_requests = 256
    adapter._profile_preparing = False
    adapter.graph_config = SimpleNamespace(capture_sizes=BUCKETS, decode_capture_sizes=BUCKETS[:9])
    adapter.runner = SimpleNamespace(
        model_config=SimpleNamespace(get_hidden_size=lambda: 2, dtype=torch.float32), device="cpu"
    )
    adapter.graph_config = SimpleNamespace(capture_sizes=BUCKETS, decode_capture_sizes=BUCKETS[:9])
    adapter._pending_tokens = tokens
    adapter._capture_phase = "mixed"
    adapter._phase = None
    adapter._closed = False
    adapter._buckets = _test_buckets()
    adapter._immutable = {}
    adapter._step = None
    from expertkit_vllm.v4_full_attention import V4KVScratchLayout

    adapter._model_len = 4096
    adapter.runner.kv_cache_config = SimpleNamespace(num_blocks=113)
    blocks = [128, 128, 128, 128, 8, 32]
    adapter._layout = V4KVScratchLayout(
        512,
        tuple(blocks),
        (4, 128, 1, 1, 1, 1),
        tuple(
            capture_carrier_pages(b, r, 2048)
            for b, r in zip(blocks, (4, 128, 1, 1, 1, 1), strict=True)
        ),
    )
    adapter._profile_table_widths = tuple(width if i == group else 0 for i in range(6))
    carrier_start = adapter._layout.carrier_start(113, group)
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
    bucket["metadata"] = SimpleNamespace(values={"layer": None}, update=lambda key, value: value)
    adapter._build_local(builder, group, ["layer"], common)
    captured = received[-1]
    assert captured.block_table_tensor.shape == (bucket["capacity"].requests, width)
    pages = capture_carrier_pages(block, ratio, tokens)
    owner = bucket["capacity"].decode_requests
    assert captured.block_table_tensor[owner, :pages].tolist() == list(
        range(carrier_start, carrier_start + pages)
    )
    if ratio > 1:
        assert (captured.slot_mapping == -1).all()
    else:
        positions = torch.arange(127, 127 + tokens)
        expected = (carrier_start + positions // block) * block + positions % block
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
    adapter._max_requests = 256
    adapter._profile_preparing = False
    adapter.graph_config = SimpleNamespace(capture_sizes=BUCKETS, decode_capture_sizes=BUCKETS[:9])
    adapter.runner = runner
    adapter._capture_phase = None
    adapter._bindings = _OwnedBindings()
    adapter._closed = False
    adapter._buckets = _test_buckets()
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
    adapter._max_requests = 256
    adapter._profile_preparing = False
    adapter.graph_config = SimpleNamespace(capture_sizes=BUCKETS, decode_capture_sizes=BUCKETS[:9])
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
    adapter._max_requests = 256
    adapter._profile_preparing = False
    adapter.graph_config = SimpleNamespace(capture_sizes=BUCKETS, decode_capture_sizes=BUCKETS[:9])
    adapter.runner = runner
    adapter._bindings = _OwnedBindings()
    adapter._closed = False
    adapter._buckets = _test_buckets()
    adapter._immutable = {}

    def validate():
        ordered = [scheduled.num_scheduled_tokens[rid] for rid in ids]
        decodes = next((i for i, n in enumerate(ordered) if n != 1), len(ordered))
        capacity = attention_capacity(
            2048 if swap else sum(lengths), max_requests=256, mixed=decodes < len(ids)
        )
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


@pytest.mark.parametrize("requests,tokens", ((1, 3), (3, 7), (9, 13), (300, 513)))
def test_configured_non_power_of_two_mixed_capacities(requests, tokens):
    capacity = attention_capacity(tokens, mixed=True, max_requests=requests)
    for decodes in range(min(requests - 1, tokens - 2) + 1):
        for prefills in range(1, min(requests - decodes, (tokens - decodes) // 2) + 1):
            lengths = [1] * decodes + [2] * (prefills - 1)
            lengths.append(tokens - decodes - 2 * (prefills - 1))
            capacity.validate_population(lengths, decodes)
    assert capacity.tokens == tokens + min(requests, tokens)
    assert capacity.requests == min(requests, tokens) + min(requests, tokens // 2)
    with pytest.raises(ValueError):
        capacity.validate_population([2] * (requests + 1), 0)


@pytest.fixture
def native_profile_runtime(monkeypatch, request):
    """Supply fake native API boundaries without replacing adapter policy."""
    import sys
    from copy import deepcopy
    from dataclasses import dataclass

    import numpy as np
    import torch

    from expertkit_vllm.v4_full_attention import V4FullAttention

    events = []

    @dataclass(frozen=True)
    class Spec:
        block_size: int
        compress_ratio: int

    @dataclass
    class Prefill:
        max_query_len: int
        max_seq_lens: int
        table: object
        slots: object
        positions: object

    @dataclass
    class Decode:
        max_seq_lens: object = 0
        max_seqlen_kv: object = 0
        max_seqlen_q: object = 0
        seq_lens_list: object = None

    @dataclass
    class Metadata:
        prefill: object
        decode: object
        hadamard: object

    class Builder:
        hadamard = torch.ones(2, 2)
        decode_threshold = 1

        def __init__(self, spec, names, config, device):
            self.kv_cache_spec = spec
            self.names = names
            self.start = torch.zeros(config.scheduler_config.max_num_seqs, dtype=torch.int32)
            events.append(("builder", names[0], config.scheduler_config.max_num_batched_tokens))

        def build(self, prefix, common, **kwargs):
            self.source = Metadata(
                Prefill(
                    common.max_query_len,
                    common.max_seq_len,
                    common.block_table_tensor,
                    common.slot_mapping,
                    common.positions,
                ),
                Decode(),
                self.hadamard,
            )
            events.append(("metadata", self.names[0], common.block_table_tensor.shape))
            return self.source

    class Backend:
        @staticmethod
        def get_builder_cls():
            return Builder

        @staticmethod
        def full_cls_name():
            return "native.DSA"

    class Impl:
        skip_topk = False
        multistream_dsv4_dsa_overlap = True

        def forward(self, *args, **kwargs):
            raise AssertionError("profile metadata must not run attention")

    specs = {"a": Spec(128, 4), "b": Spec(128, 4), "c": Spec(32, 1)}
    original_specs = deepcopy(specs)

    def native_groups(config, copied_specs):
        assert copied_specs is not specs
        copied_specs["a"] = deepcopy(copied_specs["a"])
        return [
            SimpleNamespace(
                layer_names=names,
                kv_cache_spec=SimpleNamespace(
                    kv_cache_specs={name: copied_specs[name] for name in names}
                ),
            )
            for names in (("a", "b"), ("c",))
        ]

    padding = getattr(request, "param", False)

    def native_plan(config, groups, available):
        assert config.cache_config.num_gpu_blocks_override == 1
        events.append(("native_plan", available))
        return SimpleNamespace(
            num_blocks=1,
            kv_cache_groups=groups,
            kv_cache_tensors=[
                SimpleNamespace(size=stride, shared_by=names, offset=0, block_stride=0)
                for stride, names in ((16, ["a", "c"]), (8, ["b"]), *([(8, [])] if padding else []))
            ],
        )

    def native_layers(config, base, names):
        return {name: config.compilation_config.static_forward_context[name] for name in names}

    modules = {
        "vllm.config": SimpleNamespace(
            CUDAGraphMode=SimpleNamespace(FULL=1), get_layers_from_vllm_config=native_layers
        ),
        "vllm.model_executor.layers.attention_layer_base": SimpleNamespace(
            AttentionLayerBase=object
        ),
        "vllm.v1.attention.backend": SimpleNamespace(AttentionCGSupport=SimpleNamespace(ALWAYS=1)),
        "vllm.v1.core.kv_cache_utils": SimpleNamespace(
            get_kv_cache_groups=native_groups, get_kv_cache_config_from_groups=native_plan
        ),
        "vllm.v1.worker.utils": SimpleNamespace(select_common_block_size=lambda b, backends: b),
        "vllm_ascend.attention.dsa_v1": SimpleNamespace(
            AscendDSAMetadataBuilder=Builder, AscendDSAImpl=Impl
        ),
        "vllm_ascend.attention.utils": SimpleNamespace(
            AscendCommonAttentionMetadata=SimpleNamespace
        ),
        "vllm_ascend.attention.attention_v1": SimpleNamespace(
            AscendAttentionState=SimpleNamespace(ChunkedPrefill=1)
        ),
        "vllm_ascend.ops.rope_dsv4": SimpleNamespace(
            _ROPE_STATE=SimpleNamespace(full_rope_cache={"c4": (torch.zeros(16384, 1),)}),
            get_cos_and_sin_dsa=lambda p: (p, p),
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    context = {
        name: SimpleNamespace(get_attn_backend=lambda: Backend, kv_cache=[]) for name in specs
    }
    context["a"].impl, context["b"].impl = Impl(), Impl()
    compilation = SimpleNamespace(static_forward_context=context)
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=32),
        cache_config=SimpleNamespace(num_gpu_blocks_override=None, kv_cache_memory_bytes=None),
        model_config=SimpleNamespace(max_model_len=4096),
        compilation_config=compilation,
    )
    graph = SimpleNamespace(
        capture_sizes=(3, 7, 12), decode_capture_sizes=(3, 7), model=SimpleNamespace(num_layers=2)
    )

    def build_metadata(
        num_tokens,
        num_reqs,
        max_query_len,
        num_tokens_padded=None,
        ubatch_slices=None,
        use_spec_decode=False,
    ):
        raise AssertionError("profile metadata must not use the live native metadata runner")

    runner = SimpleNamespace(
        max_num_reqs=8,
        max_num_tokens=32,
        max_model_len=4096,
        max_encoder_len=0,
        model_config=SimpleNamespace(
            max_model_len=4096, get_hidden_size=lambda: 2, dtype=torch.float32
        ),
        vllm_config=config,
        compilation_config=compilation,
        device=torch.device("cpu"),
        get_kv_cache_spec=lambda: specs,
        arange_np=np.arange(9),
        _build_attention_metadata=build_metadata,
        _pad_query_start_loc_for_fia=lambda *a: None,
        _may_reorder_batch=lambda *a: None,
        _check_and_update_cudagraph_mode=lambda *a: None,
    )

    def allocate(plan):
        events.append(("allocate", plan.num_blocks))
        views = {}
        for raw in plan.kv_cache_tensors:
            if not raw.shared_by:
                continue
            storage = torch.zeros(raw.size, dtype=torch.uint8)
            for role in raw.shared_by:
                views[role] = storage.view(-1)
                context[role].kv_cache = [views[role]]
        return views

    runner.initialize_kv_cache_tensors = allocate
    adapter = V4FullAttention(runner, graph)
    runner._expertkit_v4_full_attention = adapter
    return SimpleNamespace(
        adapter=adapter,
        runner=runner,
        config=config,
        events=events,
        specs=specs,
        original_specs=original_specs,
        planned_bytes_per_block=32 if padding else 24,
        modules=modules,
        Builder=Builder,
    )


@pytest.mark.parametrize("native_profile_runtime", (False, True), indirect=True)
def test_profile_materializes_exact_configured_buckets_and_reuses_native_storage(
    native_profile_runtime,
):
    import torch

    runtime = native_profile_runtime
    adapter, runner = runtime.adapter, runtime.runner
    context = runner.compilation_config.static_forward_context
    assert not hasattr(runner, "kv_cache_config")
    assert all(not layer.kv_cache for layer in context.values())
    adapter.prepare_profile_metadata()
    assert tuple(adapter._buckets) == (3, 7, 12)
    assert len([event for event in runtime.events if event[0] == "builder"]) == 6
    assert runtime.specs == runtime.original_specs
    assert not hasattr(runner, "kv_cache_config")
    assert all(not layer.kv_cache for layer in context.values())
    identities = {
        (c, role): (id(builder), bucket["metadata"].values[role].prefill.table.data_ptr())
        for c, bucket in adapter._buckets.items()
        for role, builder in bucket["builders"].items()
    }
    events = list(runtime.events)
    adapter.prepare_profile_metadata()
    assert runtime.events == events
    adapter._pending_tokens = 7
    adapter._capture_phase = "mixed"
    adapter._prepare_step(SimpleNamespace(), 1)
    runner.kv_cache_config = SimpleNamespace(num_blocks=113)
    for group, role in ((0, "a"), (1, "c")):
        builder = adapter._buckets[7]["builders"][role]
        common = SimpleNamespace(
            block_table_tensor=torch.zeros(
                1, adapter._profile_table_widths[group], dtype=torch.int32
            )
        )
        adapter._build_local(builder, group, builder.names, common)
        stored = adapter._buckets[7]["metadata"].values[role]
        assert (id(builder), stored.prefill.table.data_ptr()) == identities[7, role]
    assert not any(event[0] == "allocate" for event in runtime.events)


def test_profile_detects_native_context_or_builder_geometry_drift(native_profile_runtime):
    runtime = native_profile_runtime
    runtime.adapter.prepare_profile_metadata()
    runtime.runner.max_model_len = 2048
    with pytest.raises(RuntimeError, match="context auto-fit"):
        runtime.adapter.prepare_profile_metadata()
    runtime.runner.max_model_len = 4096
    with pytest.raises(RuntimeError, match="builder grouping"):
        runtime.adapter._bind_builder(
            runtime.Builder(runtime.specs["c"], ["other"], runtime.config, runtime.runner.device),
            1,
            ["other"],
        )
    assert not any(event[0] == "allocate" for event in runtime.events)


@pytest.mark.parametrize("block", (2, 8, 32, 64, 128))
def test_profile_accepts_native_non_split_block_sizes(native_profile_runtime, block):
    from dataclasses import replace

    runtime = native_profile_runtime
    runtime.specs["c"] = replace(runtime.specs["c"], block_size=block)
    runtime.adapter.prepare_profile_metadata()
    assert runtime.adapter._layout.block_sizes[1] == block


def test_profile_rejects_virtual_split_before_allocation(native_profile_runtime):
    runtime = native_profile_runtime
    runtime.modules["vllm.v1.worker.utils"].select_common_block_size = lambda b, _: b // 2
    with pytest.raises(ValueError, match=r"physical_block_size=128.*kernel_block_size=64"):
        runtime.adapter.prepare_profile_metadata()
    assert not any(event[0] == "allocate" for event in runtime.events)


@pytest.mark.parametrize("native_profile_runtime", (False, True), indirect=True)
@pytest.mark.parametrize("manual", (False, True))
def test_native_memory_hook_reserves_after_profile_in_both_budget_modes(
    native_profile_runtime, monkeypatch, manual
):
    import sys

    from expertkit_vllm.v4_full_attention import install_v4_memory_reservation

    runtime = native_profile_runtime
    budget = 10000
    runtime.config.cache_config.kv_cache_memory_bytes = budget if manual else None

    class Worker:
        def __init__(self):
            self.model_runner = runtime.runner
            if not manual:
                self.available_kv_cache_memory_bytes = budget

        def determine_available_memory(self):
            runtime.events.append(("profile_enter", manual))
            self.model_runner._expertkit_v4_full_attention.prepare_profile_metadata()
            runtime.events.append(("native_budget", budget))
            return budget

    monkeypatch.setitem(sys.modules, "vllm_ascend.worker.worker", SimpleNamespace(NPUWorker=Worker))
    install_v4_memory_reservation()
    hook = Worker.determine_available_memory
    install_v4_memory_reservation()
    assert Worker.determine_available_memory is hook
    worker = Worker()
    result = worker.determine_available_memory()
    extension = runtime.adapter._layout.extra_pages * runtime.planned_bytes_per_block
    assert result == budget - extension
    assert runtime.events[0] == ("profile_enter", manual)
    assert runtime.events[-1] == ("native_budget", budget)
    assert runtime.config.cache_config.kv_cache_memory_bytes == (budget if manual else None)
    assert runtime.config.cache_config.num_gpu_blocks_override is None
    if manual:
        assert not hasattr(worker, "available_kv_cache_memory_bytes")
    else:
        assert worker.available_kv_cache_memory_bytes == result
    assert runtime.adapter._kv_budget == budget
    assert runtime.adapter._kv_extension_bytes == extension
    assert worker.determine_available_memory() == result


def test_memory_reservation_requires_profile_and_positive_remaining_budget(native_profile_runtime):
    runtime = native_profile_runtime
    with pytest.raises(RuntimeError, match="prepared inside native profiling"):
        runtime.adapter._reserve_kv_memory(10000)
    runtime.adapter.prepare_profile_metadata()
    extension = runtime.adapter._layout.extra_pages * 24
    for budget in (0, -1, True, extension):
        with pytest.raises(ValueError):
            runtime.adapter._reserve_kv_memory(budget)
    assert not any(event[0] == "allocate" for event in runtime.events)


@pytest.mark.parametrize("native_profile_runtime", (False, True), indirect=True)
@pytest.mark.parametrize("logical_blocks", (13, 113, 4097))
@pytest.mark.parametrize("override", (False, True))
def test_native_allocation_keeps_logical_plan_and_raw_storage_aliases(
    native_profile_runtime, logical_blocks, override
):
    from copy import deepcopy

    runtime = native_profile_runtime
    adapter, runner = runtime.adapter, runtime.runner
    adapter.prepare_profile_metadata()
    logical = deepcopy(adapter._profile_plan)
    logical.num_blocks = logical_blocks
    for raw in logical.kv_cache_tensors:
        raw.size *= logical_blocks
    runner.kv_cache_config = logical
    runtime.config.cache_config.num_gpu_blocks_override = logical_blocks if override else None
    before = deepcopy(logical)
    total = runtime.planned_bytes_per_block * (logical_blocks + adapter._layout.extra_pages)
    assert adapter._reserve_kv_memory(total) == runtime.planned_bytes_per_block * logical_blocks
    views = runner.initialize_kv_cache_tensors(logical)
    assert runner.kv_cache_config is logical and logical == before
    assert views["a"].untyped_storage().data_ptr() == views["c"].untyped_storage().data_ptr()
    assert views["a"].untyped_storage().data_ptr() != views["b"].untyped_storage().data_ptr()
    assert runtime.events[-1] == ("allocate", logical_blocks + adapter._layout.extra_pages)
    assert (
        adapter._physical_config.kv_cache_tensors[-1].shared_by
        == logical.kv_cache_tensors[-1].shared_by
    )
    assert sum(raw.size for raw in adapter._physical_config.kv_cache_tensors) == total
    assert runtime.config.cache_config.num_gpu_blocks_override == (
        logical_blocks if override else None
    )
    for role, view in views.items():
        assert runner.compilation_config.static_forward_context[role].kv_cache[0] is view


@pytest.mark.parametrize("failure", ("unreserved", "budget", "override", "aliases", "context"))
def test_native_allocation_rejects_unbudgeted_or_changed_plan_before_submission(
    native_profile_runtime, failure
):
    from copy import deepcopy

    runtime = native_profile_runtime
    adapter, runner = runtime.adapter, runtime.runner
    adapter.prepare_profile_metadata()
    logical = deepcopy(adapter._profile_plan)
    logical.num_blocks = 113
    for raw in logical.kv_cache_tensors:
        raw.size *= 113
    runner.kv_cache_config = logical
    if failure != "unreserved":
        adapter._reserve_kv_memory(24 * (113 + adapter._layout.extra_pages))
    if failure == "budget":
        adapter._kv_budget -= 1
    elif failure == "override":
        runtime.config.cache_config.num_gpu_blocks_override = 112
    elif failure == "aliases":
        logical.kv_cache_tensors[0].shared_by.reverse()
    elif failure == "context":
        runner.max_model_len = 2048
    with pytest.raises((ValueError, RuntimeError)):
        runner.initialize_kv_cache_tensors(logical)
    assert not any(event[0] == "allocate" for event in runtime.events)


def test_profile_builder_failure_is_not_retried_or_left_eligible_for_reservation(
    native_profile_runtime, monkeypatch
):
    runtime = native_profile_runtime
    native_build = runtime.Builder.build
    calls = []

    def fail_on_second_group(builder, *args, **kwargs):
        calls.append(builder.names[0])
        if builder.names[0] == "c":
            raise RuntimeError("native metadata operation failed")
        return native_build(builder, *args, **kwargs)

    monkeypatch.setattr(runtime.Builder, "build", fail_on_second_group)
    with pytest.raises(RuntimeError, match="native metadata operation failed"):
        runtime.adapter.prepare_profile_metadata()
    assert calls == ["a", "c"]
    assert not runtime.adapter._profile_preparing
    assert not runtime.adapter._profile_metadata_prepared
    with pytest.raises(RuntimeError, match="cannot be retried"):
        runtime.adapter.prepare_profile_metadata()
    with pytest.raises(RuntimeError, match="prepared inside native profiling"):
        runtime.adapter._reserve_kv_memory(10000)
    assert not any(event[0] == "allocate" for event in runtime.events)


def test_profile_rejects_short_rope_before_materializing_builders(native_profile_runtime):
    import torch

    runtime = native_profile_runtime
    runtime.modules["vllm_ascend.ops.rope_dsv4"]._ROPE_STATE.full_rope_cache = {
        "c4": (torch.zeros(138, 1),)
    }
    with pytest.raises(ValueError, match="exceeds native RoPE storage"):
        runtime.adapter.prepare_profile_metadata()
    assert not any(event[0] in {"builder", "allocate"} for event in runtime.events)


def test_profile_rejects_short_native_table_before_materializing_builders(native_profile_runtime):
    runtime = native_profile_runtime
    runtime.adapter.close()
    runtime.runner.max_model_len = 128
    runtime.runner.model_config.max_model_len = 128
    runtime.config.model_config.max_model_len = 128
    from expertkit_vllm.v4_full_attention import V4FullAttention

    adapter = V4FullAttention(runtime.runner, runtime.adapter.graph_config)
    with pytest.raises(ValueError, match="table is too short"):
        adapter.prepare_profile_metadata()
    assert not any(event[0] in {"builder", "allocate"} for event in runtime.events)
    adapter.close()


@pytest.mark.parametrize(
    "captures,decode", (((33,), (33,)), ((3, 7, 12), ()), ((3, 7, 12), (3, 12)))
)
def test_adapter_rejects_incompatible_padded_capture_allocations(
    native_profile_runtime, captures, decode
):
    from expertkit_vllm.v4_full_attention import V4FullAttention

    runtime = native_profile_runtime
    runtime.adapter.close()
    graph = SimpleNamespace(capture_sizes=captures, decode_capture_sizes=decode)
    with pytest.raises(ValueError, match="exceed the native padded allocations"):
        V4FullAttention(runtime.runner, graph)
    assert not any(event[0] == "allocate" for event in runtime.events)


def test_runtime_missing_prepared_builder_is_an_error_not_a_late_allocation(native_profile_runtime):
    import torch

    runtime = native_profile_runtime
    adapter = runtime.adapter
    adapter.prepare_profile_metadata()
    builder = adapter._buckets[3]["builders"].pop("a")
    adapter._pending_tokens = 3
    adapter._capture_phase = "mixed"
    adapter._prepare_step(SimpleNamespace(), 1)
    common = SimpleNamespace(block_table_tensor=torch.zeros(1, 8, dtype=torch.int32))
    events = list(runtime.events)
    with pytest.raises(RuntimeError, match="not prepared before KV planning"):
        adapter._build_local(builder, 0, builder.names, common)
    assert runtime.events == events


@pytest.fixture
def native_install_runtime(native_profile_runtime, monkeypatch):
    import sys

    runtime = native_profile_runtime
    runtime.adapter.close()
    del runtime.runner._expertkit_v4_full_attention
    runtime.adapter.graph_config.mode = "full"
    runtime.adapter.graph_config.model.hidden_dim = 2
    runtime.config.model_config.hf_text_config = SimpleNamespace(
        model_type="deepseek_v4",
        num_hidden_layers=2,
        hidden_size=2,
        compress_ratios=(4, 0),
        num_attention_heads=64,
        index_topk=512,
        use_index_cache=False,
    )
    runtime.config.parallel_config = SimpleNamespace(
        tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1
    )
    runtime.config.speculative_config = None
    runtime.config.lora_config = None
    runtime.config.kv_transfer_config = None
    device_types = SimpleNamespace(A3=object())
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.utils",
        SimpleNamespace(
            AscendDeviceType=device_types,
            get_ascend_device_type=lambda: device_types.A3,
            enable_custom_op=lambda: True,
            enable_sp=lambda _: False,
            olora_tp_enable=lambda: False,
            oproj_tp_enable=lambda: False,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ascend_config",
        SimpleNamespace(
            get_ascend_config=lambda: SimpleNamespace(multistream_dsv4_dsa_overlap=True)
        ),
    )
    return runtime


def test_native_install_uses_resolved_model_shape_not_historical_geometry(native_install_runtime):
    from expertkit_vllm.v4_full_attention import install_v4_full_attention

    runtime = native_install_runtime
    adapter = install_v4_full_attention(runtime.runner, runtime.adapter.graph_config)
    assert adapter.graph_config.model.num_layers == 2
    assert adapter.graph_config.model.hidden_dim == 2
    assert adapter._buckets.keys() == {3, 7, 12}
    adapter.close()


@pytest.mark.parametrize(
    "field,value,error",
    (
        ("num_hidden_layers", 3, "model geometry"),
        ("hidden_size", 4, "model geometry"),
        ("model_type", "other_architecture", "model geometry"),
        ("compress_ratios", (4, 8), "compression ratios 4/128"),
        ("num_attention_heads", 32, "64 query heads"),
        ("index_topk", 256, "index_topk=512"),
    ),
)
def test_native_install_retains_concrete_model_or_operator_constraints(
    native_install_runtime, field, value, error
):
    from expertkit_vllm.v4_full_attention import install_v4_full_attention

    runtime = native_install_runtime
    setattr(runtime.config.model_config.hf_text_config, field, value)
    with pytest.raises(ValueError, match=error):
        install_v4_full_attention(runtime.runner, runtime.adapter.graph_config)
    assert not any(event[0] == "allocate" for event in runtime.events)


def test_native_install_does_not_fix_unused_sparse_selection_width(native_install_runtime):
    from expertkit_vllm.v4_full_attention import install_v4_full_attention

    runtime = native_install_runtime
    runtime.config.model_config.hf_text_config.compress_ratios = (0, 128)
    runtime.config.model_config.hf_text_config.index_topk = 256
    adapter = install_v4_full_attention(runtime.runner, runtime.adapter.graph_config)
    adapter.close()
