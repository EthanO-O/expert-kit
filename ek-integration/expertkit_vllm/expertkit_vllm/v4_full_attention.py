"""Bound the private native V4 attention geometry without enlarging the scheduler.

Private metadata and buffers follow the native graph lifetime. Decode keeps
the native geometry; mixed attention uses bounded local packing and gathering.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass
from enum import Enum
from typing import Any


@dataclass(frozen=True)
class V4AttentionCapacity:
    """Physical attention capacities for one native token bucket and phase."""

    outer_tokens: int
    decode_tokens: int
    prefill_tokens: int
    decode_requests: int
    prefill_requests: int

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
            or len(lengths) > 256
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


def attention_capacity(token_bucket: int, *, mixed: bool) -> V4AttentionCapacity:
    """Derive local capacities from a validated native capture bucket.

    The mixed request bound uses the exact native threshold of one query per
    Decode request. Other classifier policies must be rejected by installation.
    """
    if type(token_bucket) is not int or not 1 <= token_bucket <= 2048:
        raise ValueError("V4 native token bucket must be within 1..2048")
    if not mixed:
        if token_bucket > 256:
            raise ValueError("pure Decode cannot exceed the 256-request scheduler")
        return V4AttentionCapacity(token_bucket, token_bucket, 0, token_bucket, 0)
    if token_bucket < 2:
        raise ValueError("mixed attention requires a Prefill query of length at least two")
    return V4AttentionCapacity(
        token_bucket,
        min(256, token_bucket),
        token_bucket,
        min(256, token_bucket),
        min(256, token_bucket // 2),
    )


def capture_carrier_pages(block_size: int, compress_ratio: int, tokens: int) -> int:
    """Bound native compressed positions for the start-127 capture carrier.

    The native compressor emits floor((start + qlen) / ratio) minus
    floor(start / ratio) rows, starting at floor(start / ratio).
    """
    if any(type(n) is not int or n <= 0 for n in (block_size, compress_ratio, tokens)):
        raise ValueError("V4 capture geometry must contain positive integers")
    end = (127 + tokens) // compress_ratio
    return (end + block_size - 1) // block_size


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
    sizing cannot mutate the outer scheduler's 256 requests and 2048 tokens.
    """
    if (
        native_config.scheduler_config.max_num_seqs != 256
        or native_config.scheduler_config.max_num_batched_tokens != 2048
    ):
        raise ValueError("V4 Full requires the qualified outer scheduler capacities")
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


def independent_dummy_page(group: int, owner: int) -> int:
    """Return an isolated zero-query scratch page outside logical N2048.

    A page is not a ring buffer: this mapping is valid only for zero-query
    requests with start position zero. Real requests retain their native table.
    """
    if type(group) is not int or not 0 <= group < 6:
        raise ValueError("V4 cache group must be within 0..5")
    if type(owner) is not int or not 0 <= owner < 512:
        raise ValueError("V4 virtual owner must be within 0..511")
    return 2048 + group * 512 + owner


def physical_cache_config(logical: Any) -> Any:
    """Clone only the native KV allocation plan, preserving logical capacity.

    Every raw allocation retains its native shared-by aliases and byte stride.
    The scheduler continues to receive the original N2048 object. The caller
    must allocate this plan through the original native runner, never manually
    reshape or bind role views.
    """
    from copy import deepcopy

    if logical.num_blocks != 2048 or len(logical.kv_cache_groups) != 6:
        raise ValueError("V4 Full requires the qualified N2048 six-group pool")
    if len(logical.kv_cache_tensors) != 44:
        raise ValueError("V4 Full requires all 44 native raw allocation slots")
    roles: set[str] = set()
    for group in logical.kv_cache_groups:
        for name in group.layer_names:
            if name in roles or name not in group.kv_cache_spec.kv_cache_specs:
                raise ValueError("V4 native cache role join is ambiguous")
            roles.add(name)
    if len(roles) != 167:
        raise ValueError("V4 Full requires all 167 native cache roles")
    for raw in logical.kv_cache_tensors:
        if (
            raw.offset != 0
            or raw.block_stride != 0
            or raw.size <= 0
            or raw.size % 2048
            or any(role not in roles for role in raw.shared_by)
        ):
            raise ValueError("V4 native allocation layout differs from qualification")
    physical = deepcopy(logical)
    physical.num_blocks = 5120
    for raw in physical.kv_cache_tensors:
        raw.size = raw.size // 2048 * physical.num_blocks
    return physical


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
        try:
            for tokens in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048):
                capacity = attention_capacity(tokens, mixed=True)
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
        self._immutable.clear()
        self._closed = True

    def _install(self) -> None:
        import inspect
        from functools import wraps

        runner = self.runner
        self._bind_decode_padding()
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
            physical = physical_cache_config(logical)
            for group_id, group in enumerate(logical.kv_cache_groups):
                for role in group.layer_names:
                    self._group_roles[role] = group_id
            # One positive capture carrier per group; zero-query pages keep
            # their separate N2048..N5120 domain and never serve long queries.
            block_counts = []
            self._capture_page_starts = []
            self._capture_group_ratios = []
            next_page = 5120
            for group in logical.kv_cache_groups:
                self._capture_group_ratios.append(capture_group_ratio(group.kv_cache_spec))
                block_sizes = {
                    spec.block_size for spec in group.kv_cache_spec.kv_cache_specs.values()
                }
                if len(block_sizes) != 1:
                    raise ValueError("V4 capture group has inconsistent block sizes")
                count = (127 + 2048 + next(iter(block_sizes)) - 1) // next(iter(block_sizes))
                self._capture_page_starts.append(next_page)
                next_page += count
                block_counts.append(count)
            if block_counts != [17, 17, 17, 17, 272, 68] or next_page != 5528:
                raise ValueError("V4 native capture carrier differs from its source contract")
            physical.num_blocks = next_page
            for raw, original in zip(
                physical.kv_cache_tensors, logical.kv_cache_tensors, strict=True
            ):
                raw.size = original.size // 2048 * next_page
            from copy import deepcopy

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
        if len(implementations) != 43:
            raise ValueError(
                "V4 Full requires exactly 43 original native attention implementations"
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
                and 1 <= num_reqs <= num_tokens_padded <= 256
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

    def _wrap_builders(self) -> None:
        for group_id, groups in enumerate(self.runner.attn_groups):
            for group in groups:
                builder = group.get_metadata_builder(0)
                if id(builder) in self._wrapped_builders:
                    continue
                self._bind_builder(builder, group_id, group.layer_names)
                self._wrapped_builders.add(id(builder))

    def _bind_builder(self, original_builder: Any, group_id: int, layer_names: list[str]) -> None:
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
        if type(num_requests) is not int or not 1 <= num_requests <= 256:
            raise ValueError("V4 Full requires 1..256 actual native requests")
        capturing_mixed = self._capture_phase == "mixed"
        if capturing_mixed:
            lengths = (self._pending_tokens,)
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
            if self._pending_tokens > 256 or any(length != 1 for length in lengths):
                raise ValueError("V4 native Decode phase is not uniform single-query")
            self._step = {"phase": phase}
            return
        capacity = attention_capacity(self._pending_tokens, mixed=True)
        population = attention_population(capacity, lengths, decodes)
        if capacity.outer_tokens not in self._buckets:
            self._buckets[capacity.outer_tokens] = {
                "capacity": capacity,
                "buffers": _AttentionBuffers(
                    capacity,
                    self.runner.model_config.get_hidden_size(),
                    self.runner.model_config.dtype,
                    self.runner.device,
                ),
                "metadata": _NativeMetadataStorage(self._immutable),
                "builders": {},
            }
        bucket = self._buckets[capacity.outer_tokens]
        bucket["buffers"].update_population(population)
        if capturing_mixed:
            starts = (127,)
            original_positions = list(range(127, 127 + sum(lengths)))
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
        if key not in bucket["builders"]:
            bucket["builders"][key] = type(original_builder)(
                original_builder.kv_cache_spec,
                layer_names,
                local_builder_config(self.runner.vllm_config, capacity),
                self.runner.device,
            )
        builder = bucket["builders"][key]
        local = copy(common)
        positions = step["positions_cpu"].to(self.runner.device)
        query = step["query_cpu"].to(self.runner.device)
        sequences = step["seq_cpu"].to(self.runner.device)
        width = common.block_table_tensor.shape[1]
        table = torch.empty(
            (capacity.requests, width),
            dtype=common.block_table_tensor.dtype,
            device=self.runner.device,
        )
        for owner in range(capacity.requests):
            table[owner].fill_(independent_dummy_page(group_id, owner))
        real_rows = [
            (owner, source)
            for owner, source in enumerate(population.request_sources)
            if source is not None
        ]
        if step["capturing_mixed"]:
            block_size = builder.kv_cache_spec.block_size
            ratio = self._capture_group_ratios[group_id]
            if getattr(builder.kv_cache_spec, "compress_ratio", 1) != ratio:
                raise ValueError("V4 builder compression differs from its native group")
            page_count = capture_carrier_pages(block_size, ratio, capacity.outer_tokens)
            reserved = (127 + 2048 + block_size - 1) // block_size
            if page_count > reserved:
                raise ValueError("V4 capture carrier exceeds its reserved physical pages")
            if page_count > width:
                raise ValueError("V4 native table is too short for the capture carrier")
            owner = capacity.decode_requests
            table[owner].zero_()
            table[owner, :page_count].copy_(
                torch.arange(
                    self._capture_page_starts[group_id],
                    self._capture_page_starts[group_id] + page_count,
                    dtype=table.dtype,
                    device=table.device,
                )
            )
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
        if step["capturing_mixed"] and ratio > 1:
            # Native compressed Prefill ignores common slots and derives its
            # real scatter slots from compressor metadata. Decode owners have
            # zero query length in this synthetic population.
            pass
        elif step["capturing_mixed"]:
            block_size = builder.kv_cache_spec.block_size
            logical_positions = torch.arange(
                127, 127 + capacity.outer_tokens, dtype=torch.int64, device=self.runner.device
            )
            pages = table[capacity.decode_requests].index_select(0, logical_positions // block_size)
            slots.index_copy_(
                0, valid_rows, pages.to(torch.int64) * block_size + logical_positions % block_size
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
        local.max_seq_len = 8192
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
        value.prefill.max_seq_lens = 8192
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
    """Install one exact-profile adapter before the original native profile run.

    The adapter admits no additional runtime options. The caller performs
    the shared graph-profile validation before reaching this native boundary.
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
    expected_ratios = (0, 0) + (4, 128) * 20 + (4, 0, 0, 0)
    if (
        get_ascend_device_type() is not AscendDeviceType.A3
        or not enable_custom_op()
        or tuple(hf.compress_ratios) != expected_ratios
        or hf.num_hidden_layers != 43
        or hf.hidden_size != 4096
        or hf.num_attention_heads != 64
        or hf.index_topk != 512
        or getattr(hf, "use_index_cache", False)
        or graph_config.mode != "full"
        or tuple(graph_config.capture_sizes) != (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
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
