"""Bound the private native V4 attention geometry without enlarging the scheduler.

Private metadata and buffers follow the native graph lifetime. Decode keeps
the native geometry; mixed attention uses bounded local packing and gathering.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from copy import copy, deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class V4AttentionCapacity:
    """Physical attention capacities for one native token bucket and phase."""

    outer_tokens: int
    decode_tokens: int
    prefill_tokens: int
    decode_requests: int
    prefill_requests: int
    max_requests: int

    @property
    def tokens(self) -> int:
        """Return the private attention slot-mapping length."""
        return self.decode_tokens + self.prefill_tokens

    @property
    def requests(self) -> int:
        """Return the private attention start-position buffer length."""
        return self.decode_requests + self.prefill_requests

    def validate_population(self, query_lengths: Sequence[int], num_decodes: int) -> None:
        """Reject populations outside the native qlen-one Decode classification.

        Args:
            query_lengths: Real requests in native Decode-then-Prefill order,
                without the private zero-query padding slots.
            num_decodes: Number of leading requests with exactly one query.
        """
        lengths = tuple(query_lengths)
        if (
            not lengths
            or len(lengths) > self.max_requests
            or type(num_decodes) is not int
            or not 0 <= num_decodes <= len(lengths)
            or any(type(n) is not int or n <= 0 for n in lengths)
            or any(n != 1 for n in lengths[:num_decodes])
            or any(n < 2 for n in lengths[num_decodes:])
            or sum(lengths) > self.outer_tokens
            or num_decodes > self.decode_requests
            or len(lengths) - num_decodes > self.prefill_requests
            or sum(lengths[num_decodes:]) > self.prefill_tokens
        ):
            raise ValueError("V4 attention population exceeds its native phase capacity")
        if self.prefill_tokens and num_decodes == len(lengths):
            raise ValueError("pure Decode must retain the native Decode branch")


def attention_capacity(token_bucket: int, *, mixed: bool, max_requests: int) -> V4AttentionCapacity:
    """Bound private slots by the native single-query Decode classifier."""
    if any(type(n) is not int or n <= 0 for n in (token_bucket, max_requests)):
        raise ValueError("V4 token and request capacities must be positive integers")
    if not mixed:
        if token_bucket > max_requests:
            raise ValueError("pure Decode exceeds the native request allocation")
        return V4AttentionCapacity(token_bucket, token_bucket, 0, token_bucket, 0, max_requests)
    if token_bucket < 2:
        raise ValueError("mixed attention requires a Prefill query of length at least two")
    return V4AttentionCapacity(
        token_bucket,
        min(max_requests, token_bucket),
        token_bucket,
        min(max_requests, token_bucket),
        min(max_requests, token_bucket // 2),
        max_requests,
    )


@dataclass(frozen=True)
class V4CaptureLayout:
    """Bound positive-query capture owners; remaining token rows use sentinels."""

    query_lengths: tuple[int, ...]
    starts: tuple[int, ...]

    def page_counts(self, block_size: int, compress_ratio: int) -> tuple[int, ...]:
        """Return isolated page-span sizes in native request order."""
        return tuple(
            capture_carrier_pages(block_size, compress_ratio, length, start=start)
            for start, length in zip(self.starts, self.query_lengths, strict=True)
        )


def capture_layout(
    capacity: V4AttentionCapacity, model_len: int, compress_ratios: Sequence[int]
) -> V4CaptureLayout:
    """Fill legal Prefill owners and leave an unfillable tail as zero padding.

    Starts cross the largest feasible native compression boundary. A context
    shorter than a compression ratio cannot produce a row for that ratio, but
    native padded compressor metadata still retains its fixed output capacity.
    """
    if type(model_len) is not int or model_len < 2:
        raise ValueError("V4 mixed capture requires a context of at least two tokens")
    boundary = max((ratio for ratio in compress_ratios if ratio <= model_len), default=1)
    remaining = capacity.outer_tokens
    lengths = []
    while remaining >= 2 and len(lengths) < capacity.max_requests:
        length = min(remaining, model_len)
        lengths.append(length)
        remaining -= length
    capacity.validate_population(lengths, 0)
    return V4CaptureLayout(
        tuple(lengths), tuple(min(boundary - 1, model_len - length) for length in lengths)
    )


def capture_carrier_pages(block_size: int, compress_ratio: int, tokens: int, *, start: int) -> int:
    """Bound one capture owner's native compressed positions.

    The native compressor emits floor((start + qlen) / ratio) minus
    floor(start / ratio) rows, starting at floor(start / ratio).
    """
    if any(type(n) is not int or n <= 0 for n in (block_size, compress_ratio, tokens)) or (
        type(start) is not int or start < 0
    ):
        raise ValueError("V4 capture geometry must contain positive integers")
    end = (start + tokens) // compress_ratio
    return max(1, (end + block_size - 1) // block_size)


def capture_group_ratio(group_spec: Any) -> int:
    """Require the uniform compression ratio used by the native block table."""
    specs = tuple(group_spec.kv_cache_specs.values())
    values = tuple(getattr(spec, "compress_ratio", 1) for spec in specs)
    if any(type(ratio) is not int or ratio <= 0 for ratio in values):
        raise ValueError("V4 capture group compression ratio must be a positive integer")
    ratios = set(values)
    if not specs or len(ratios) != 1:
        raise ValueError("V4 capture group has inconsistent compression ratios")
    return next(iter(ratios))


def local_builder_config(native_config: Any, capacity: V4AttentionCapacity) -> Any:
    """Copy only configuration owned by the private native attention builder.

    The original compilation configuration contains live layer objects and must
    retain identity; deep-copying the full VllmConfig could duplicate weights.
    Scheduler and cache settings have separate ownership so native builder buffer
    sizing cannot mutate the outer scheduler or its allocation capacities.
    """
    if (
        capacity.max_requests != native_config.scheduler_config.max_num_seqs
        or capacity.outer_tokens > native_config.scheduler_config.max_num_batched_tokens
    ):
        raise ValueError("V4 private capacity exceeds the native outer allocations")
    local = copy(native_config)
    local.scheduler_config = copy(native_config.scheduler_config)
    local.cache_config = copy(native_config.cache_config)
    local.scheduler_config.max_num_seqs = capacity.requests
    local.scheduler_config.max_num_batched_tokens = capacity.tokens
    return local


@dataclass(frozen=True)
class V4AttentionPopulation:
    """Host-owned routing for one scheduler step, without a device synchronization.

    Real requests retain native Decode-then-Prefill ordering. Virtual request
    owners are separate from logical scheduler request identities.
    """

    capacity: V4AttentionCapacity
    query_lengths: tuple[int, ...]
    num_decodes: int
    physical_lengths: tuple[int, ...]
    request_sources: tuple[int | None, ...]
    valid_rows: tuple[int, ...]
    pack_sources: tuple[int, ...]
    gather_sources: tuple[int, ...]


def attention_population(
    capacity: V4AttentionCapacity,
    query_lengths: Sequence[int],
    num_decodes: int,
) -> V4AttentionPopulation:
    """Build bounded host maps; sentinel indices address dedicated zero rows.

    The caller adds one zero input row after the outer token bucket and one
    zero output row after the attention capacity. Unscheduled outer rows gather
    that output sentinel, never an uninitialized virtual attention row.
    """
    lengths = tuple(query_lengths)
    capacity.validate_population(lengths, num_decodes)
    if not capacity.prefill_tokens:
        physical_lengths = lengths + (0,) * (capacity.requests - len(lengths))
        sources = tuple(range(len(lengths))) + (None,) * (capacity.requests - len(lengths))
        valid = tuple(range(sum(lengths)))
    else:
        prefill_lengths = lengths[num_decodes:]
        physical_lengths = (
            lengths[:num_decodes]
            + (0,) * (capacity.decode_requests - num_decodes)
            + prefill_lengths
            + (0,) * (capacity.prefill_requests - len(prefill_lengths))
        )
        sources = (
            tuple(range(num_decodes))
            + (None,) * (capacity.decode_requests - num_decodes)
            + tuple(range(num_decodes, len(lengths)))
            + (None,) * (capacity.prefill_requests - len(prefill_lengths))
        )
        valid = tuple(range(num_decodes)) + tuple(
            range(capacity.decode_tokens, capacity.decode_tokens + sum(prefill_lengths))
        )
    pack = [capacity.outer_tokens] * capacity.tokens
    gather = [capacity.tokens] * capacity.outer_tokens
    for outer, physical in enumerate(valid):
        pack[physical] = outer
        gather[outer] = physical
    return V4AttentionPopulation(
        capacity,
        lengths,
        num_decodes,
        physical_lengths,
        sources,
        valid,
        tuple(pack),
        tuple(gather),
    )


@dataclass(frozen=True)
class V4KVScratchLayout:
    """Private page domains shared across native raw-allocation aliases."""

    owners: int
    block_sizes: tuple[int, ...]
    compress_ratios: tuple[int, ...]
    carrier_pages: tuple[int, ...]

    @property
    def extra_pages(self) -> int:
        """Return pages excluded from the logical scheduler pool."""
        return len(self.block_sizes) * self.owners + sum(self.carrier_pages)

    def carrier_start(self, logical_blocks: int, group: int) -> int:
        """Return this group's positive-query carrier outside all dummy pages."""
        if type(group) is not int or not 0 <= group < len(self.block_sizes):
            raise ValueError("V4 cache group is outside the native layout")
        return (
            logical_blocks + len(self.block_sizes) * self.owners + sum(self.carrier_pages[:group])
        )


def kv_scratch_layout(
    groups: Sequence[Any],
    *,
    max_virtual_owners: int,
    mixed_capture_layouts: Sequence[V4CaptureLayout],
) -> V4KVScratchLayout:
    """Derive isolated scratch domains from native block and compression specs."""
    if type(max_virtual_owners) is not int or max_virtual_owners < 0 or not groups:
        raise ValueError("V4 scratch requires native groups and a nonnegative owner count")
    layouts = tuple(mixed_capture_layouts)
    if bool(layouts) != bool(max_virtual_owners):
        raise ValueError("V4 mixed scratch owners and capture sizes are inconsistent")
    blocks, ratios, carriers = [], [], []
    for group in groups:
        specs = group.kv_cache_spec.kv_cache_specs
        values = {spec.block_size for spec in specs.values()}
        if len(values) != 1 or any(type(n) is not int or n <= 0 for n in values):
            raise ValueError("V4 capture group has inconsistent block sizes")
        block = next(iter(values))
        ratio = capture_group_ratio(group.kv_cache_spec)
        blocks.append(block)
        ratios.append(ratio)
        carriers.append(
            max((sum(layout.page_counts(block, ratio)) for layout in layouts), default=0)
        )
    return V4KVScratchLayout(max_virtual_owners, tuple(blocks), tuple(ratios), tuple(carriers))


def independent_dummy_page(
    layout: V4KVScratchLayout, *, logical_blocks: int, group: int, owner: int
) -> int:
    """Address one zero-query-only page without aliasing logical KV or other owners."""
    if type(logical_blocks) is not int or logical_blocks <= 0:
        raise ValueError("V4 logical block count must be positive")
    if type(group) is not int or not 0 <= group < len(layout.block_sizes):
        raise ValueError("V4 cache group is outside the native layout")
    if type(owner) is not int or not 0 <= owner < layout.owners:
        raise ValueError("V4 virtual owner is outside the private capacity")
    return logical_blocks + group * layout.owners + owner


def _allocation_strides(logical: Any) -> tuple[int, ...]:
    """Validate the native role join and the currently supported unpacked strides."""
    if type(logical.num_blocks) is not int or logical.num_blocks <= 0:
        raise ValueError("V4 logical block count must be positive")
    roles = []
    for group in logical.kv_cache_groups:
        if not group.layer_names:
            raise ValueError("V4 native cache group must contain roles")
        for name in group.layer_names:
            if name not in group.kv_cache_spec.kv_cache_specs:
                raise ValueError("V4 native cache role join is ambiguous")
            roles.append(name)
    allocated = []
    strides = []
    for raw in logical.kv_cache_tensors:
        if (
            raw.offset != 0
            or raw.block_stride != 0
            or type(raw.size) is not int
            or raw.size <= 0
            or raw.size % logical.num_blocks
        ):
            raise ValueError("V4 native allocation layout has unsupported byte strides")
        # Native empty padding slots still contribute to the planned byte budget.
        allocated.extend(raw.shared_by)
        strides.append(raw.size // logical.num_blocks)
    if (
        not roles
        or any(n != 1 for n in Counter(roles).values())
        or Counter(allocated) != Counter(roles)
    ):
        raise ValueError("V4 native allocation layout must cover every role exactly once")
    return tuple(strides)


def physical_cache_config(logical: Any, layout: V4KVScratchLayout) -> Any:
    """Clone the native plan and aliases; never allocate or bind role views here."""
    strides = _allocation_strides(logical)
    if len(logical.kv_cache_groups) != len(layout.block_sizes):
        raise ValueError("V4 scratch group count differs from the native plan")
    physical = deepcopy(logical)
    physical.num_blocks += layout.extra_pages
    for raw, stride in zip(physical.kv_cache_tensors, strides, strict=True):
        raw.size = stride * physical.num_blocks
    return physical


def _same_native_plan(left: Any, right: Any) -> bool:
    return (
        left.kv_cache_groups == right.kv_cache_groups
        and _allocation_strides(left) == _allocation_strides(right)
        and tuple(tuple(raw.shared_by) for raw in left.kv_cache_tensors)
        == tuple(tuple(raw.shared_by) for raw in right.kv_cache_tensors)
    )


def install_v4_memory_reservation() -> None:
    """Reserve future private KV pages at the pinned native Worker budget boundary.

    The frontend must prepare persistent metadata inside its profile_run hook.
    Native manual-memory mode intentionally bypasses automatic non-KV accounting.
    """
    from functools import wraps

    from vllm_ascend.worker.worker import NPUWorker

    original = NPUWorker.determine_available_memory
    if getattr(original, "_expertkit_v4_reservation", False):
        return

    @wraps(original)
    def determine(worker: Any) -> int:
        budget = original(worker)
        adapter = getattr(worker.model_runner, "_expertkit_v4_full_attention", None)
        if adapter is None:
            return budget
        logical_budget = adapter._reserve_kv_memory(budget)
        if hasattr(worker, "available_kv_cache_memory_bytes"):
            worker.available_kv_cache_memory_bytes = logical_budget
        return logical_budget

    determine._expertkit_v4_reservation = True
    NPUWorker.determine_available_memory = determine


def _clone_metadata(value: Any, immutable: dict[int, Any] | None = None) -> Any:
    """Detach every native metadata tensor from reusable builder storage."""
    import dataclasses

    import torch

    if isinstance(value, torch.Tensor):
        if immutable is not None and immutable.get(id(value)) is value:
            return value
        return value.clone()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.replace(
            value,
            **{
                field.name: _clone_metadata(getattr(value, field.name), immutable)
                for field in dataclasses.fields(value)
            },
        )
    if isinstance(value, dict):
        return {key: _clone_metadata(item, immutable) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_metadata(item, immutable) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_metadata(item, immutable) for item in value)
    # The installed native proxy deliberately exposes its tensor tuple tree.
    if type(value).__name__ == "RopeDataProxy" and type(value).__module__.startswith(
        "vllm_ascend."
    ):
        return type(value)(_clone_metadata(value._data, immutable), is_cos=value.idx == 0)
    if value is None or isinstance(value, (bool, int, float, str, Enum)):
        return value
    raise TypeError(f"Unsupported mutable V4 metadata leaf: {type(value)!r}")


def _update_metadata(
    destination: Any, source: Any, immutable: dict[int, Any] | None = None
) -> None:
    """Update a captured metadata tree without replacing any tensor address.

    Scalar and container geometry must stay unchanged within a native bucket.
    Unknown leaves fail closed instead of retaining an unowned mutable alias.
    """
    import dataclasses

    import torch

    if isinstance(destination, torch.Tensor):
        if immutable is not None and immutable.get(id(destination)) is destination:
            if source is not destination:
                raise ValueError("V4 global immutable metadata identity changed")
            return
        if not isinstance(source, torch.Tensor) or (
            destination.shape != source.shape
            or destination.dtype != source.dtype
            or destination.device != source.device
        ):
            raise ValueError("V4 metadata tensor geometry changed within a capture bucket")
        destination.copy_(source)
    elif dataclasses.is_dataclass(destination) and not isinstance(destination, type):
        if type(destination) is not type(source):
            raise ValueError("V4 metadata dataclass type changed")
        for field in dataclasses.fields(destination):
            _update_metadata(
                getattr(destination, field.name), getattr(source, field.name), immutable
            )
    elif isinstance(destination, dict):
        if not isinstance(source, dict) or destination.keys() != source.keys():
            raise ValueError("V4 metadata dictionary geometry changed")
        for key in destination:
            _update_metadata(destination[key], source[key], immutable)
    elif isinstance(destination, (list, tuple)):
        if type(source) is not type(destination) or len(source) != len(destination):
            raise ValueError("V4 metadata sequence geometry changed")
        for left, right in zip(destination, source, strict=True):
            _update_metadata(left, right, immutable)
    elif type(destination).__name__ == "RopeDataProxy" and type(destination).__module__.startswith(
        "vllm_ascend."
    ):
        if type(source) is not type(destination) or source.idx != destination.idx:
            raise ValueError("V4 rotary proxy geometry changed")
        _update_metadata(destination._data, source._data, immutable)
    elif destination is None or isinstance(destination, (bool, int, float, str, Enum)):
        if type(destination) is not type(source) or destination != source:
            raise ValueError("V4 metadata scalar changed within a capture bucket")
    else:
        raise TypeError(f"Unsupported mutable V4 metadata leaf: {type(destination)!r}")


class _NativeMetadataStorage:
    """Own one native bucket's stable graph inputs independently of its builder."""

    def __init__(self, immutable: dict[int, Any] | None = None) -> None:
        self.values: dict[str, Any] = {}
        self.immutable = immutable if immutable is not None else {}

    def update(self, role: str, value: Any) -> Any:
        """Snapshot immediately before another native builder can reuse buffers."""
        if role not in self.values:
            self.values[role] = _clone_metadata(value, self.immutable)
        else:
            _update_metadata(self.values[role], value, self.immutable)
        return self.values[role]

    def close(self) -> None:
        """Release metadata only after every graph referring to it is cleared."""
        self.values.clear()


class _OwnedBindings:
    """Restore only this adapter's instance attributes, including failure paths."""

    def __init__(self) -> None:
        self._bindings: list[tuple[Any, str, bool, Any, Any]] = []

    def bind(self, owner: Any, name: str, replacement: Any) -> None:
        """Remember inherited attributes without turning them into local fields."""
        local = name in vars(owner)
        previous = getattr(owner, name)
        self._bindings.append((owner, name, local, previous, replacement))
        setattr(owner, name, replacement)

    def close(self) -> None:
        """Reject foreign replacement instead of removing another owner's hook."""
        conflicts = []
        for owner, name, local, previous, replacement in reversed(self._bindings):
            if getattr(owner, name) is not replacement:
                conflicts.append(name)
                continue
            if local:
                setattr(owner, name, previous)
            else:
                delattr(owner, name)
        self._bindings.clear()
        if conflicts:
            raise RuntimeError(f"V4 attention lost ownership of {conflicts!r}")


class _AttentionBuffers:
    """Own private attention-only token storage; outer Worker buffers stay fixed."""

    def __init__(
        self, capacity: V4AttentionCapacity, hidden_size: int, dtype: Any, device: Any
    ) -> None:
        import torch

        self.capacity = capacity
        self.outer_input = torch.zeros(
            (capacity.outer_tokens + 1, hidden_size), dtype=dtype, device=device
        )
        self.hidden = torch.empty((capacity.tokens, hidden_size), dtype=dtype, device=device)
        self.output = torch.zeros((capacity.tokens + 1, hidden_size), dtype=dtype, device=device)
        self.pack_map = torch.full(
            (capacity.tokens,), capacity.outer_tokens, dtype=torch.int64, device=device
        )
        self.gather_map = torch.full(
            (capacity.outer_tokens,), capacity.tokens, dtype=torch.int64, device=device
        )

    def update_population(self, population: V4AttentionPopulation) -> None:
        """Upload bounded scheduler-derived maps outside graph capture/replay."""
        import torch

        if population.capacity != self.capacity:
            raise ValueError("V4 attention buffer population belongs to a different bucket")
        self.pack_map.copy_(torch.tensor(population.pack_sources, dtype=torch.int64, device="cpu"))
        self.gather_map.copy_(
            torch.tensor(population.gather_sources, dtype=torch.int64, device="cpu")
        )

    def pack(self, hidden_states: Any) -> Any:
        """Copy real outer slots and gather private slots entirely on device."""
        if tuple(hidden_states.shape) != tuple(self.outer_input[:-1].shape):
            raise ValueError("V4 attention input differs from its outer native token bucket")
        self.outer_input[:-1].copy_(hidden_states)
        self.hidden.copy_(self.outer_input.index_select(0, self.pack_map))
        return self.hidden

    def gather(self, output: Any) -> Any:
        """Gather only mapped real rows; padding reads a separate zero sentinel."""
        if tuple(output.shape) != tuple(self.outer_input[:-1].shape):
            raise ValueError("V4 attention output differs from its outer native token bucket")
        output.copy_(self.output.index_select(0, self.gather_map))
        return output


def _attention_context(
    context_module: Any, metadata: Any, num_tokens: int, additional_kwargs: dict[str, Any]
) -> Any:
    """Scope native attention geometry while preserving outer communication state.

    The supplied private kwargs dictionary is shared by this step's attention
    layers, so native compressor metadata reuse retains its original group and
    phase semantics. It must be created once per step, without the previous
    step's compressor cache. Restoring the outer context precedes residual/MoE.
    """
    from contextlib import contextmanager

    @contextmanager
    def scope():
        outer = context_module.get_forward_context()
        local = copy(outer)
        local.num_tokens = num_tokens
        local.attn_metadata = metadata
        local.additional_kwargs = additional_kwargs
        context_module._forward_context = local
        try:
            yield local
        finally:
            context_module._forward_context = outer

    return scope()


class V4FullAttention:
    """Own the exact V4 attention-only geometry and native hook lifetime.

    Captured graphs and this object have the same lifetime; a
    poisoned graph session must retain this object instead of calling close.
    """

    def __init__(self, runner: Any, graph_config: Any) -> None:
        self.runner = runner
        self.graph_config = graph_config
        self._bindings = _OwnedBindings()
        self._immutable: dict[int, Any] = {}
        from vllm_ascend.ops.rope_dsv4 import _ROPE_STATE

        for pair in _ROPE_STATE.full_rope_cache.values():
            for tensor in pair:
                self._immutable[id(tensor)] = tensor
        self._capture_phase: str | None = None
        self._phase: str | None = None
        self._closed = False
        self._step: dict[str, Any] | None = None
        self._buckets: dict[int, dict[str, Any]] = {}
        self._wrapped_builders: set[int] = set()
        self._physical_config: Any = None
        self._group_roles: dict[str, int] = {}
        self._max_requests = runner.max_num_reqs
        self._max_tokens = runner.max_num_tokens
        self._model_len = runner.max_model_len
        self._profile_metadata_prepared = False
        self._profile_preparing = False
        self._profile_plan: Any = None
        self._kv_budget: int | None = None
        self._kv_extension_bytes: int | None = None
        self._profile_table_widths: tuple[int, ...] = ()
        self._builder_signatures: dict[tuple[int, str], tuple[Any, ...]] = {}
        if (
            any(
                type(n) is not int or n <= 0
                for n in (self._max_requests, self._max_tokens, self._model_len)
            )
            or max(graph_config.capture_sizes) > self._max_tokens
            or tuple(graph_config.decode_capture_sizes)
            != tuple(n for n in graph_config.capture_sizes if n <= self._max_requests)
            or not graph_config.decode_capture_sizes
        ):
            raise ValueError("V4 capture sizes exceed the native padded allocations")
        try:
            for tokens in graph_config.capture_sizes:
                if tokens < 2:
                    continue
                capacity = attention_capacity(tokens, mixed=True, max_requests=self._max_requests)
                self._buckets[tokens] = {
                    "capacity": capacity,
                    "buffers": _AttentionBuffers(
                        capacity,
                        runner.model_config.get_hidden_size(),
                        runner.model_config.dtype,
                        runner.device,
                    ),
                    "metadata": _NativeMetadataStorage(self._immutable),
                    "builders": {},
                }
            self._install()
        except BaseException:
            self._bindings.close()
            self._buckets.clear()
            raise

    def _check_profile_geometry(self) -> None:
        if (
            self.runner.max_num_reqs != self._max_requests
            or self.runner.max_num_tokens != self._max_tokens
            or self.runner.max_model_len != self._model_len
            or self.runner.model_config.max_model_len != self._model_len
        ):
            raise RuntimeError(
                "V4 native allocation/context geometry changed after profiling; "
                "context auto-fit requires rebuilding and budgeting private metadata"
            )

    def prepare_profile_metadata(self) -> None:
        """Materialize persistent private builders and clones inside native profiling.

        The frontend must call this from its profile_run wrapper, before the
        original profile forward. No KV allocation or runner binding occurs.
        Native context/table geometry must remain unchanged through allocation.
        """
        import torch
        from vllm.config import get_layers_from_vllm_config
        from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
        from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups, get_kv_cache_groups
        from vllm.v1.worker.utils import select_common_block_size
        from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder
        from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
        from vllm_ascend.ops.rope_dsv4 import _ROPE_STATE

        self._check_profile_geometry()
        if self._closed:
            raise RuntimeError("V4 attention profile lifetime has ended")
        if self._profile_metadata_prepared:
            return
        if self._profile_preparing or self._profile_plan is not None:
            raise RuntimeError("V4 private metadata preparation cannot be retried after failure")
        config = self.runner.vllm_config
        specs = deepcopy(self.runner.get_kv_cache_spec())
        groups = get_kv_cache_groups(config, specs)
        probe_config = copy(config)
        probe_config.cache_config = copy(config.cache_config)
        probe_config.cache_config.num_gpu_blocks_override = 1
        plan = get_kv_cache_config_from_groups(probe_config, groups, 0)
        _allocation_strides(plan)
        ratios = tuple(capture_group_ratio(group.kv_cache_spec) for group in groups)
        for bucket in self._buckets.values():
            bucket["capture"] = capture_layout(bucket["capacity"], self._model_len, ratios)
        captures = tuple(bucket["capture"] for bucket in self._buckets.values())
        owners = max((bucket["capacity"].requests for bucket in self._buckets.values()), default=0)
        self._layout = kv_scratch_layout(
            groups, max_virtual_owners=owners, mixed_capture_layouts=captures
        )
        max_capture_end = max(
            (
                start + length
                for layout in captures
                for start, length in zip(layout.starts, layout.query_lengths, strict=True)
            ),
            default=0,
        )
        for pair in _ROPE_STATE.full_rope_cache.values():
            for tensor in pair:
                if tensor.shape[0] < max_capture_end:
                    raise ValueError("V4 capture carrier exceeds native RoPE storage")
                self._immutable[id(tensor)] = tensor
        builder_groups = []
        widths = []
        for group_id, group in enumerate(groups):
            layers = get_layers_from_vllm_config(config, AttentionLayerBase, group.layer_names)
            selected = {}
            backends = {}
            for role in group.layer_names:
                spec = group.kv_cache_spec.kv_cache_specs[role]
                backend = layers[role].get_attn_backend()
                builder_cls = backend.get_builder_cls()
                if not issubclass(builder_cls, AscendDSAMetadataBuilder):
                    raise ValueError(
                        "V4 profile encountered an unsupported native metadata builder"
                    )
                key = (backend.full_cls_name(), spec)
                selected.setdefault(key, []).append(role)
                backends[key] = backend
            block = self._layout.block_sizes[group_id]
            kernel_block = select_common_block_size(block, list(backends.values()))
            if kernel_block != block:
                raise ValueError(
                    "V4 private carrier cannot map raw physical pages with "
                    f"physical_block_size={block} and kernel_block_size={kernel_block}; "
                    "virtual splitting requires expanded page IDs"
                )
            ratio = self._layout.compress_ratios[group_id]
            model_len = max(self._model_len, self.runner.max_encoder_len)
            width = max((((model_len + block - 1) // block) + ratio - 1) // ratio, 1)
            if any(max(layout.page_counts(block, ratio)) > width for layout in captures):
                raise ValueError("V4 native table is too short for a capture owner")
            widths.append(width)
            for key, names in selected.items():
                builder_cls = backends[key].get_builder_cls()
                names = tuple(names)
                builder_groups.append((group_id, builder_cls, key[1], names))
                self._builder_signatures[group_id, names[0]] = (builder_cls, key[1], names)
        self._profile_table_widths = tuple(widths)
        self._profile_plan = plan
        previous = self._step, self._phase, self._capture_phase
        self._profile_preparing = True
        try:
            for tokens, bucket in self._buckets.items():
                self._pending_tokens = tokens
                self._step = None
                self._capture_phase = "mixed"
                layout = bucket["capture"]
                query = [0]
                for length in layout.query_lengths:
                    query.append(query[-1] + length)
                common = AscendCommonAttentionMetadata(
                    query_start_loc=torch.tensor(
                        query, dtype=torch.int32, device=self.runner.device
                    ),
                    query_start_loc_cpu=torch.tensor(query, dtype=torch.int32),
                    seq_lens=torch.zeros(
                        len(layout.query_lengths), dtype=torch.int32, device=self.runner.device
                    ),
                    seq_lens_cpu=torch.tensor(
                        [
                            start + length
                            for start, length in zip(
                                layout.starts, layout.query_lengths, strict=True
                            )
                        ],
                        dtype=torch.int32,
                    ),
                    num_reqs=len(layout.query_lengths),
                    num_actual_tokens=sum(layout.query_lengths),
                    max_query_len=max(layout.query_lengths),
                    max_seq_len=self._model_len,
                    block_table_tensor=None,
                    slot_mapping=None,
                    causal=True,
                    num_input_tokens=tokens,
                )
                self._prepare_step(common, len(layout.query_lengths))
                for group_id, builder_cls, spec, names in builder_groups:
                    builder = builder_cls(
                        spec,
                        list(names),
                        local_builder_config(config, bucket["capacity"]),
                        self.runner.device,
                    )
                    bucket["builders"][names[0]] = builder
                    common.block_table_tensor = torch.zeros(
                        (len(layout.query_lengths), widths[group_id]),
                        dtype=torch.int32,
                        device=self.runner.device,
                    )
                    self._build_local(builder, group_id, list(names), common)
            if self.runner.device.type == "npu":
                torch.npu.synchronize(self.runner.device)
            self._profile_metadata_prepared = True
        finally:
            self._profile_preparing = False
            self._step, self._phase, self._capture_phase = previous

    def _reserve_kv_memory(self, budget: int) -> int:
        self._check_profile_geometry()
        if not self._profile_metadata_prepared:
            raise RuntimeError("V4 private metadata must be prepared inside native profiling")
        if type(budget) is not int or budget <= 0:
            raise ValueError("V4 native KV memory budget must be positive")
        extension = self._layout.extra_pages * sum(_allocation_strides(self._profile_plan))
        if budget <= extension:
            raise ValueError("V4 native KV budget cannot hold the mandatory private pages")
        self._kv_budget = budget
        self._kv_extension_bytes = extension
        logger.info(
            "v4_graph_kv_memory_reserved",
            native_kv_budget_bytes=budget,
            private_kv_bytes=extension,
            logical_kv_budget_bytes=budget - extension,
            manual_kv_memory=getattr(
                self.runner.vllm_config.cache_config, "kv_cache_memory_bytes", None
            )
            is not None,
            capture_sizes=self.graph_config.capture_sizes,
            decode_capture_sizes=self.graph_config.decode_capture_sizes,
            max_num_requests=self._max_requests,
            max_num_tokens=self._max_tokens,
            max_model_len=self._model_len,
        )
        return budget - extension

    @property
    def phase(self) -> str:
        """Return the already prepared phase without reading a device tensor."""
        phase = self._capture_phase or self._phase
        if self._closed or phase not in ("decode", "mixed"):
            raise RuntimeError("V4 attention phase has not been prepared")
        return phase

    def capture_phase(self, phase: str) -> Any:
        """Scope native capture preparation without changing its descriptors."""
        from contextlib import contextmanager

        @contextmanager
        def scope():
            if self._closed or phase not in ("decode", "mixed"):
                raise ValueError("V4 capture phase must be decode or mixed")
            previous = self._capture_phase
            self._capture_phase = phase
            try:
                yield
            finally:
                self._capture_phase = previous

        return scope()

    def close(self) -> None:
        """Restore owned hooks after both phase graph dictionaries are cleared."""
        if self._closed:
            return
        self._bindings.close()
        self._buckets.clear()
        self._step = None
        self._physical_config = None
        self._profile_plan = None
        self._builder_signatures = {}
        self._profile_metadata_prepared = False
        self._immutable.clear()
        self._closed = True

    def _install(self) -> None:
        import inspect
        from functools import wraps

        runner = self.runner
        self._bind_decode_padding()
        self._bind_batch_reorder()
        build = runner._build_attention_metadata
        signature = inspect.signature(build)

        @wraps(build)
        def build_step(*args: Any, **kwargs: Any) -> Any:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            values = bound.arguments
            if values.get("ubatch_slices") is not None or values.get("use_spec_decode"):
                raise ValueError("V4 Full does not support microbatching or speculation")
            self._step = None
            self._phase = None
            self._pending_tokens = values["num_tokens_padded"] or values["num_tokens"]
            self._wrap_builders()
            return build(*args, **kwargs)

        self._bindings.bind(runner, "_build_attention_metadata", build_step)
        allocate = runner.initialize_kv_cache_tensors

        @wraps(allocate)
        def allocate_pool(logical: Any) -> Any:
            self._check_profile_geometry()
            if not self._profile_metadata_prepared or self._kv_budget is None:
                raise RuntimeError(
                    "V4 metadata and KV memory must be reserved before native allocation"
                )
            if not _same_native_plan(logical, self._profile_plan):
                raise RuntimeError("V4 native KV layout changed after profile reservation")
            override = runner.vllm_config.cache_config.num_gpu_blocks_override
            if override is not None and logical.num_blocks != override:
                raise RuntimeError("V4 native logical block override changed before allocation")
            physical = physical_cache_config(logical, self._layout)
            extension = sum(raw.size for raw in physical.kv_cache_tensors) - sum(
                raw.size for raw in logical.kv_cache_tensors
            )
            if extension != self._kv_extension_bytes:
                raise RuntimeError("V4 private KV byte reservation changed before allocation")
            if sum(raw.size for raw in physical.kv_cache_tensors) > self._kv_budget:
                raise ValueError(
                    "V4 logical KV pool plus private pages exceeds the native memory budget"
                )
            for group_id, group in enumerate(logical.kv_cache_groups):
                for role in group.layer_names:
                    self._group_roles[role] = group_id
            logical_identity = runner.kv_cache_config
            logical_before = deepcopy(logical)
            views = allocate(physical)
            if runner.kv_cache_config is not logical_identity or logical_identity is not logical:
                raise RuntimeError("V4 native allocator changed logical pool ownership")
            if logical != logical_before or set(views) != set(self._group_roles):
                raise RuntimeError("V4 native allocator changed logical roles or capacity")
            context = runner.compilation_config.static_forward_context
            if any(context[role].kv_cache[0] is not views[role] for role in views):
                raise RuntimeError("V4 native allocator did not bind its original role views")
            self._physical_config = physical
            logger.info(
                "v4_graph_kv_pool_allocated",
                logical_num_blocks=logical.num_blocks,
                physical_num_blocks=physical.num_blocks,
                physical_kv_bytes=sum(raw.size for raw in physical.kv_cache_tensors),
                private_kv_bytes=extension,
            )
            return views

        self._bindings.bind(runner, "initialize_kv_cache_tensors", allocate_pool)
        self._wrap_support_check()
        self._wrap_implementations()

    def _wrap_support_check(self) -> None:
        from vllm.v1.attention.backend import AttentionCGSupport
        from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder

        original = self.runner._check_and_update_cudagraph_mode

        def check(backends: Any, groups: Any) -> Any:
            proxies = []
            for backend_set in backends:
                local = set()
                for backend in backend_set:
                    builder = backend.get_builder_cls()
                    if not issubclass(builder, AscendDSAMetadataBuilder):
                        raise ValueError("V4 Full encountered an unqualified attention backend")
                    private_builder = type(
                        "V4Local" + builder.__name__,
                        (builder,),
                        {
                            "get_cudagraph_support": classmethod(
                                lambda cls, config, spec: AttentionCGSupport.ALWAYS
                            )
                        },
                    )
                    proxy = type(
                        "V4Local" + backend.__name__,
                        (backend,),
                        {"get_builder_cls": staticmethod(lambda result=private_builder: result)},
                    )
                    local.add(proxy)
                proxies.append(local)
            return original(proxies, groups)

        self._bindings.bind(self.runner, "_check_and_update_cudagraph_mode", check)

    def _wrap_implementations(self) -> None:
        from vllm_ascend.attention.dsa_v1 import AscendDSAImpl

        implementations = {}
        for owner in self.runner.compilation_config.static_forward_context.values():
            impl = getattr(owner, "impl", None)
            if isinstance(impl, AscendDSAImpl):
                implementations[id(impl)] = impl
        if len(implementations) != self.graph_config.model.num_layers:
            raise ValueError(
                "V4 Full requires one original native attention implementation per model layer"
            )
        for impl in implementations.values():
            if impl.skip_topk or not impl.multistream_dsv4_dsa_overlap:
                raise ValueError(
                    "V4 Full requires the qualified native learned-indexer/overlap path"
                )
            self._bind_implementation(impl)

    def _bind_implementation(self, impl: Any) -> None:
        original = impl.forward

        def forward(
            layer_name: str,
            hidden_states: Any,
            kv_cache: Any,
            attn_metadata: Any,
            need_gather_q_kv: bool = False,
            output: Any = None,
        ) -> Any:
            if attn_metadata is None or self.phase == "decode":
                return original(
                    layer_name, hidden_states, kv_cache, attn_metadata, need_gather_q_kv, output
                )
            if need_gather_q_kv or output is None or self._step is None:
                raise ValueError("V4 Full local attention requires the qualified TP1 step")
            import vllm.forward_context as context_module

            step = self._step
            bucket = step["bucket"]
            outer = context_module.get_forward_context()
            if step["additional_kwargs"] is None:
                step["additional_kwargs"] = dict(outer.additional_kwargs)
                step["additional_kwargs"].pop("dsv4_compressor_metadata_cache", None)
            buffers = bucket["buffers"]
            packed = buffers.pack(hidden_states)
            with _attention_context(
                context_module,
                outer.attn_metadata,
                bucket["capacity"].tokens,
                step["additional_kwargs"],
            ):
                original(layer_name, packed, kv_cache, attn_metadata, False, buffers.output[:-1])
            return buffers.gather(output)

        self._bindings.bind(impl, "forward", forward)

    def _bind_decode_padding(self) -> None:
        """Use native uniform padding for this adapter's single-query Decode.

        Native FULL appends one long dummy query. That geometry cannot replay
        the separate Decode graph captured with one query per padded request.
        Actual request/token counts and native slot/table handling stay intact.
        """
        from vllm.config import CUDAGraphMode

        runner = self.runner
        original = runner._pad_query_start_loc_for_fia

        def pad(
            query_start_loc: Any,
            num_tokens_padded: int,
            num_reqs_padded: int,
            num_reqs: int,
            cudagraph_runtime_mode: Any = None,
            batch_desc_num_reqs: int | None = None,
        ) -> int:
            if (
                self._capture_phase != "mixed"
                and cudagraph_runtime_mode == CUDAGraphMode.FULL
                and type(num_reqs) is int
                and type(num_tokens_padded) is int
                and 1 <= num_reqs <= num_tokens_padded <= self._max_requests
                and num_tokens_padded in self.graph_config.decode_capture_sizes
                and (query_start_loc.np[: num_reqs + 1] == runner.arange_np[: num_reqs + 1]).all()
            ):
                query_start_loc.np[num_reqs + 1 : num_tokens_padded + 1] = (
                    runner.arange_np[1 : num_tokens_padded + 1 - num_reqs] + num_reqs
                )
                query_start_loc.copy_to_gpu()
                return num_tokens_padded
            return original(
                query_start_loc,
                num_tokens_padded,
                num_reqs_padded,
                num_reqs,
                cudagraph_runtime_mode,
                batch_desc_num_reqs,
            )

        self._bindings.bind(runner, "_pad_query_start_loc_for_fia", pad)

    def _bind_batch_reorder(self) -> None:
        """Restore native DSA's single-query prefix after generic reordering."""
        from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder

        runner = self.runner
        original = runner._may_reorder_batch

        def reorder(scheduler_output: Any) -> Any:
            result = original(scheduler_output)
            builder = runner.attn_groups[0][0].get_metadata_builder(0)
            if not isinstance(builder, AscendDSAMetadataBuilder) or builder.decode_threshold != 1:
                raise ValueError("V4 Full requires the native single-query DSA reorder policy")
            builder.reorder_batch(runner.input_batch, scheduler_output)
            return result

        self._bindings.bind(runner, "_may_reorder_batch", reorder)

    def _wrap_builders(self) -> None:
        for group_id, groups in enumerate(self.runner.attn_groups):
            for group in groups:
                builder = group.get_metadata_builder(0)
                if id(builder) in self._wrapped_builders:
                    continue
                self._bind_builder(builder, group_id, group.layer_names)
                self._wrapped_builders.add(id(builder))

    def _bind_builder(self, original_builder: Any, group_id: int, layer_names: list[str]) -> None:
        signature = self._builder_signatures.get((group_id, layer_names[0]))
        if signature != (
            type(original_builder),
            original_builder.kv_cache_spec,
            tuple(layer_names),
        ):
            raise RuntimeError("V4 native metadata builder grouping changed after profiling")
        original = original_builder.build

        def build(
            common_prefix_len: int,
            common_attn_metadata: Any,
            fast_build: bool = False,
            **kwargs: Any,
        ) -> Any:
            if common_prefix_len or fast_build:
                raise ValueError("V4 Full does not support cascade or fast-build metadata")
            self._prepare_step(common_attn_metadata, kwargs["num_reqs_actual"])
            if self.phase == "decode":
                return original(
                    common_prefix_len, common_attn_metadata, fast_build=fast_build, **kwargs
                )
            # The native runner reads these attributes after build returns.
            # Publish the same fresh step dictionaries used by the delegate.
            original_builder.prefill_ratio_to_sas_metadata = self._step["prefill_shared"]
            original_builder.decode_ratio_to_sas_metadata = self._step["decode_shared"]
            original_builder.common_ratio_to_sas_metadata = self._step["common_shared"]
            return self._build_local(original_builder, group_id, layer_names, common_attn_metadata)

        self._bindings.bind(original_builder, "build", build)

    def _prepare_step(self, common: Any, num_requests: int) -> None:
        if self._step is not None:
            return
        import torch

        if self._pending_tokens not in self.graph_config.capture_sizes:
            raise ValueError("V4 token geometry is absent from the frozen native bucket list")
        if type(num_requests) is not int or not 1 <= num_requests <= self._max_requests:
            raise ValueError("V4 actual native request count exceeds its allocation")
        capturing_mixed = self._capture_phase == "mixed"
        if capturing_mixed:
            layout = self._buckets[self._pending_tokens]["capture"]
            lengths = layout.query_lengths
            decodes = 0
        else:
            qcpu = common.query_start_loc_cpu
            if qcpu.device.type != "cpu":
                raise ValueError("V4 phase routing requires native host query metadata")
            lengths = tuple(
                int(value) for value in (qcpu[1 : num_requests + 1] - qcpu[:num_requests]).tolist()
            )
            decodes = 0
            for length in lengths:
                if length != 1:
                    break
                decodes += 1
        phase = self._capture_phase or ("decode" if decodes == len(lengths) else "mixed")
        self._phase = phase
        if phase == "decode":
            if self._pending_tokens not in self.graph_config.decode_capture_sizes or any(
                length != 1 for length in lengths
            ):
                raise ValueError("V4 native Decode phase is not uniform single-query")
            self._step = {"phase": phase}
            return
        capacity = attention_capacity(
            self._pending_tokens, mixed=True, max_requests=self._max_requests
        )
        population = attention_population(capacity, lengths, decodes)
        if capacity.outer_tokens not in self._buckets:
            raise RuntimeError("V4 private bucket was not allocated before profiling")
        bucket = self._buckets[capacity.outer_tokens]
        bucket["buffers"].update_population(population)
        if capturing_mixed:
            starts = layout.starts
            original_positions = [
                position
                for start, length in zip(starts, lengths, strict=True)
                for position in range(start, start + length)
            ]
        else:
            scpu = common.seq_lens_cpu
            if scpu is None:
                scpu = common._seq_lens_cpu
            if scpu.device.type != "cpu" or common.positions_cpu.device.type != "cpu":
                raise ValueError(
                    "V4 Full requires synchronous native host sequence/position metadata"
                )
            starts = tuple(
                int(value) - length
                for value, length in zip(scpu[:num_requests].tolist(), lengths, strict=True)
            )
            original_positions = common.positions_cpu[: sum(lengths)].tolist()
        physical_starts = [
            0 if source is None else starts[source] for source in population.request_sources
        ]
        physical_positions = [0] * capacity.tokens
        for source, target in enumerate(population.valid_rows):
            physical_positions[target] = int(original_positions[source])
        qloc = [0]
        for length in population.physical_lengths:
            qloc.append(qloc[-1] + length)
        sequences = [
            start + length if length else 0
            for start, length in zip(physical_starts, population.physical_lengths, strict=True)
        ]
        self._step = {
            "phase": phase,
            "bucket": bucket,
            "population": population,
            "capturing_mixed": capturing_mixed,
            "positions_cpu": torch.tensor(physical_positions, dtype=torch.int64, device="cpu"),
            "query_cpu": torch.tensor(qloc, dtype=torch.int32, device="cpu"),
            "seq_cpu": torch.tensor(sequences, dtype=torch.int32, device="cpu"),
            "starts_cpu": torch.tensor(physical_starts, dtype=torch.int32, device="cpu"),
            "prefill_shared": {},
            "decode_shared": {},
            "common_shared": {},
            "additional_kwargs": None,
        }

    def _build_local(
        self, original_builder: Any, group_id: int, layer_names: list[str], common: Any
    ) -> Any:
        import torch
        from vllm_ascend.attention.attention_v1 import AscendAttentionState
        from vllm_ascend.ops.rope_dsv4 import get_cos_and_sin_dsa

        step = self._step
        bucket = step["bucket"]
        capacity = bucket["capacity"]
        population = step["population"]
        key = layer_names[0]
        if key not in bucket["builders"] or (
            not self._profile_preparing and key not in bucket["metadata"].values
        ):
            raise RuntimeError("V4 private builder/metadata was not prepared before KV planning")
        builder = bucket["builders"][key]
        local = copy(common)
        positions = step["positions_cpu"].to(self.runner.device)
        query = step["query_cpu"].to(self.runner.device)
        sequences = step["seq_cpu"].to(self.runner.device)
        width = common.block_table_tensor.shape[1]
        if width != self._profile_table_widths[group_id]:
            raise RuntimeError("V4 native table geometry changed after profiling")
        table = torch.zeros(
            (capacity.requests, width),
            dtype=common.block_table_tensor.dtype,
            device=self.runner.device,
        )
        if not self._profile_preparing:
            for owner in range(capacity.requests):
                table[owner].fill_(
                    independent_dummy_page(
                        self._layout,
                        logical_blocks=self.runner.kv_cache_config.num_blocks,
                        group=group_id,
                        owner=owner,
                    )
                )
        real_rows = [
            (owner, source)
            for owner, source in enumerate(population.request_sources)
            if source is not None
        ]
        ratio = self._layout.compress_ratios[group_id]
        if self._profile_preparing:
            pass
        elif step["capturing_mixed"]:
            block_size = builder.kv_cache_spec.block_size
            if getattr(builder.kv_cache_spec, "compress_ratio", 1) != ratio:
                raise ValueError("V4 builder compression differs from its native group")
            counts = bucket["capture"].page_counts(block_size, ratio)
            reserved = self._layout.carrier_pages[group_id]
            if sum(counts) > reserved:
                raise ValueError("V4 capture carrier exceeds its reserved physical pages")
            if max(counts) > width:
                raise ValueError("V4 native table is too short for the capture carrier")
            page = self._layout.carrier_start(self.runner.kv_cache_config.num_blocks, group_id)
            for (owner, _), count in zip(real_rows, counts, strict=True):
                table[owner, :count].copy_(
                    torch.arange(page, page + count, dtype=table.dtype, device=table.device)
                )
                page += count
        else:
            owners = torch.tensor(
                [owner for owner, _ in real_rows], dtype=torch.int64, device=self.runner.device
            )
            sources = torch.tensor(
                [source for _, source in real_rows], dtype=torch.int64, device=self.runner.device
            )
            table.index_copy_(0, owners, common.block_table_tensor.index_select(0, sources))
        slots = torch.full((capacity.tokens,), -1, dtype=torch.int64, device=self.runner.device)
        valid_rows = torch.tensor(
            population.valid_rows, dtype=torch.int64, device=self.runner.device
        )
        if self._profile_preparing:
            pass
        elif step["capturing_mixed"] and ratio > 1:
            # Native compressed Prefill ignores common slots and derives its
            # real scatter slots from compressor metadata. Decode owners have
            # zero query length in this synthetic population.
            pass
        elif step["capturing_mixed"]:
            block_size = builder.kv_cache_spec.block_size
            for owner, _ in real_rows:
                begin, end = (
                    capacity.decode_tokens + offset
                    for offset in step["query_cpu"][owner : owner + 2].tolist()
                )
                logical_positions = positions[begin:end]
                pages = table[owner].index_select(0, logical_positions // block_size)
                slots[begin:end] = (
                    pages.to(torch.int64) * block_size + logical_positions % block_size
                )
        else:
            slots.index_copy_(
                0, valid_rows, common.slot_mapping[: len(population.valid_rows)].to(torch.int64)
            )
        local.query_start_loc = query
        local.query_start_loc_cpu = step["query_cpu"]
        local.seq_lens = sequences
        local.seq_lens_cpu = step["seq_cpu"]
        local._seq_lens_cpu = step["seq_cpu"]
        local.seq_lens_cpu_upper_bound = step["seq_cpu"]
        local.num_computed_tokens_cpu = step["starts_cpu"]
        local.num_reqs = capacity.requests
        local.num_actual_tokens = capacity.tokens
        local.num_input_tokens = capacity.tokens
        local.max_query_len = capacity.prefill_tokens
        local.max_seq_len = self._model_len
        local.block_table_tensor = table
        local.slot_mapping = slots
        local.positions = positions
        local.positions_cpu = step["positions_cpu"]
        local.attn_state = AscendAttentionState.ChunkedPrefill
        local.graph_pad_size = -1
        if not step["common_shared"]:
            cos, sin = get_cos_and_sin_dsa(positions)
            step["common_shared"].update(
                num_decodes=capacity.decode_requests,
                num_prefills=capacity.prefill_requests,
                num_decode_tokens=capacity.decode_tokens,
                num_prefill_tokens=capacity.prefill_tokens,
                input_positions=positions,
                cos=cos,
                sin=sin,
                seq_lens=sequences,
                query_lens=step["query_cpu"][1:] - step["query_cpu"][:-1],
            )
        value = builder.build(
            0,
            local,
            num_reqs_actual=capacity.requests,
            block_size=builder.kv_cache_spec.block_size,
            prefill_ratio_to_sas_metadata=step["prefill_shared"],
            decode_ratio_to_sas_metadata=step["decode_shared"],
            common_ratio_to_sas_metadata=step["common_shared"],
        )
        value.prefill.max_query_len = capacity.prefill_tokens
        value.prefill.max_seq_lens = self._model_len
        value.decode.max_seq_lens = None
        value.decode.max_seqlen_kv = None
        value.decode.max_seqlen_q = None
        value.decode.seq_lens_list = None
        from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder

        hadamard = AscendDSAMetadataBuilder.hadamard
        if value.hadamard is not None:
            if value.hadamard is not hadamard:
                raise ValueError("V4 native Hadamard global identity changed")
            self._immutable[id(hadamard)] = hadamard
        return bucket["metadata"].update(key, value)


def install_v4_full_attention(runner: Any, graph_config: Any) -> V4FullAttention:
    """Install the native V4 attention boundary before the original profile run.

    Shared model facts are resolved before this device/backend-specific check.
    """
    from vllm_ascend.ascend_config import get_ascend_config
    from vllm_ascend.utils import (
        AscendDeviceType,
        enable_custom_op,
        enable_sp,
        get_ascend_device_type,
        olora_tp_enable,
        oproj_tp_enable,
    )

    existing = getattr(runner, "_expertkit_v4_full_attention", None)
    if existing is not None:
        if existing.graph_config != graph_config or existing._closed:
            raise RuntimeError("V4 Full runner already belongs to another graph lifetime")
        return existing
    config = runner.vllm_config
    hf = config.model_config.hf_text_config
    if (
        hf.model_type != "deepseek_v4"
        or hf.num_hidden_layers != graph_config.model.num_layers
        or hf.hidden_size != graph_config.model.hidden_dim
    ):
        raise ValueError("V4 native model geometry differs from the resolved graph model")
    active_ratios = tuple(hf.compress_ratios[: hf.num_hidden_layers])
    if len(active_ratios) != hf.num_hidden_layers or any(
        type(ratio) is not int or ratio not in (0, 1, 4, 128) for ratio in active_ratios
    ):
        raise ValueError(
            "V4 native metadata/forward supports uncompressed layers or compression ratios 4/128"
        )
    # Pinned SparseAttnSharedkv accepts 64 query heads; its sparse-index axis
    # is 512 when the ratio-4 branch passes learned indexer selections.
    if hf.num_attention_heads != 64:
        raise ValueError("A3 npu_sparse_attn_sharedkv requires 64 query heads")
    if 4 in active_ratios and hf.index_topk != 512:
        raise ValueError("A3 npu_sparse_attn_sharedkv requires index_topk=512 for ratio-4 layers")
    if (
        get_ascend_device_type() is not AscendDeviceType.A3
        or not enable_custom_op()
        or getattr(hf, "use_index_cache", False)
        or graph_config.mode != "full"
        or config.parallel_config.tensor_parallel_size != 1
        or config.parallel_config.pipeline_parallel_size != 1
        or config.parallel_config.data_parallel_size != 1
        or config.speculative_config is not None
        or config.lora_config is not None
        or config.kv_transfer_config is not None
        or enable_sp(config)
        or olora_tp_enable()
        or oproj_tp_enable()
        or not get_ascend_config().multistream_dsv4_dsa_overlap
    ):
        raise ValueError("V4 Full native attention configuration is outside the qualified profile")
    adapter = V4FullAttention(runner, graph_config)
    runner._expertkit_v4_full_attention = adapter
    return adapter
