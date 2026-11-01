"""Shared graph profile validation and resident lease ownership.

Providers compose these helpers with independent computations. Capture, replay,
communication, routing, projections and reduction are owned by their callers.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import torch

from expertkit_worker.weights import ReadyWeightLease

# These native count-mode routing and weight-list GMM limits are not model profiles.
ASCEND_ROUTING_MAX_EXPERTS = 10240
ASCEND_GROUPED_MAX_EXPERTS = 128


@dataclass(frozen=True)
class ResidentGraphProfile:
    """Freeze ordered model layers and their exact resident expert IDs."""

    layer_ids: Sequence[int]
    experts_per_layer: int
    hidden_dim: int
    intermediate_dim: int
    top_k: int
    expert_range: tuple[int, int] | None = None
    expert_compute: str = "swiglu"
    swiglu_limit: float = 0.0
    expert_ids_by_layer: Mapping[int, tuple[int, ...]] | None = None

    def __post_init__(self) -> None:
        for name in ("experts_per_layer", "hidden_dim", "intermediate_dim", "top_k"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.expert_compute not in ("swiglu", "deepseek_v4"):
            raise ValueError("unsupported resident expert computation")
        if (
            not math.isfinite(self.swiglu_limit)
            or self.swiglu_limit < 0
            or (self.expert_compute == "swiglu" and self.swiglu_limit != 0)
        ):
            raise ValueError("swiglu_limit must be finite nonnegative and zero for ordinary swiglu")
        if self.top_k > self.experts_per_layer:
            raise ValueError("resident graph requires top_k <= experts_per_layer")
        layers = tuple(self.layer_ids)
        if (
            not layers
            or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in layers)
            or tuple(sorted(set(layers))) != layers
        ):
            raise ValueError("layer_ids must be nonempty, sorted, unique and nonnegative")
        object.__setattr__(self, "layer_ids", layers)
        if self.expert_range is not None and self.expert_ids_by_layer is not None:
            raise ValueError("expert_range and expert_ids_by_layer are mutually exclusive")
        if self.expert_ids_by_layer is None:
            start, end = self.interval
            if not 0 <= start < end <= self.experts_per_layer:
                raise ValueError("resident expert interval must be within the model")
            selections = {layer: tuple(range(start, end)) for layer in layers}
        else:
            if any(
                isinstance(layer, bool) or not isinstance(layer, int)
                for layer in self.expert_ids_by_layer
            ) or set(self.expert_ids_by_layer) != set(layers):
                raise ValueError("expert_ids_by_layer must contain exactly layer_ids")
            selections = {layer: self.expert_ids_by_layer[layer] for layer in layers}
        for ids in selections.values():
            if (
                not isinstance(ids, tuple)
                or any(
                    isinstance(i, bool)
                    or not isinstance(i, int)
                    or not 0 <= i < self.experts_per_layer
                    for i in ids
                )
                or tuple(sorted(set(ids))) != ids
            ):
                raise ValueError("per-layer expert IDs must be sorted unique valid integer tuples")
            # The existing GMM path is bounded to 128 local weight groups.
            if len(ids) > ASCEND_GROUPED_MAX_EXPERTS:
                raise ValueError(
                    f"resident GMM requires at most {ASCEND_GROUPED_MAX_EXPERTS} local experts"
                )
        if not any(selections.values()):
            raise ValueError("a resident graph participant must own at least one expert")
        object.__setattr__(self, "expert_ids_by_layer", MappingProxyType(selections))

    @property
    def interval(self) -> tuple[int, int]:
        """Return the legacy uniform interval for callers that configured one."""
        return self.expert_range if self.expert_range is not None else (0, self.experts_per_layer)

    def expert_ids(self, layer_id: int) -> tuple[int, ...]:
        """Return the exact immutable global expert order for one routed layer."""
        assert self.expert_ids_by_layer is not None
        return self.expert_ids_by_layer[layer_id]

    def layer_interval(self, layer_id: int) -> tuple[int, int] | None:
        """Return a nonempty contiguous layer interval, or no interval."""
        ids = self.expert_ids(layer_id)
        if ids and ids == tuple(range(ids[0], ids[-1] + 1)):
            return ids[0], ids[-1] + 1
        return None

    def routing_map(self, layer_id: int, device: torch.device) -> torch.Tensor | None:
        """Prepare a global-ID permutation with selected experts first.

        Contiguous layers need no lookup. Noncontiguous layers retain all global
        IDs in the permutation so the routing operator never sees an invalid ID.
        """
        if not self.expert_ids(layer_id) or self.layer_interval(layer_id) is not None:
            return None
        selected = self.expert_ids(layer_id)
        order = (*selected, *(i for i in range(self.experts_per_layer) if i not in selected))
        mapping = [0] * self.experts_per_layer
        for local_id, global_id in enumerate(order):
            mapping[global_id] = local_id
        return torch.tensor(mapping, dtype=torch.int32, device=device)

    def validate_inputs(
        self,
        device: torch.device,
        hidden: torch.Tensor,
        ids: torch.Tensor,
        routing: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        if (
            hidden.ndim != 2
            or hidden.shape[0] <= 0
            or hidden.shape[1] != self.hidden_dim
            or output.shape != hidden.shape
        ):
            raise ValueError("resident graph requires nonempty hidden/output matrices")
        if ids.shape != (hidden.shape[0], self.top_k):
            raise ValueError("resident graph expert IDs have the wrong shape")
        if routing.shape != ids.shape:
            raise ValueError("resident graph routing weights have the wrong shape")
        if (
            hidden.dtype != torch.bfloat16
            or output.dtype not in (torch.bfloat16, torch.float32)
            or ids.dtype != torch.int32
            or routing.dtype != torch.float32
        ):
            raise ValueError("resident graph requires BF16 activations, int32 IDs and FP32 routing")
        if any(t.device != device or not t.is_contiguous() for t in (hidden, ids, routing, output)):
            raise ValueError("resident graph inputs must be contiguous on the weights' device")


class ResidentWeightOwner[T]:
    """Own all local leases; callers retain their full provider after quarantine."""

    def __init__(self, ready_revision: Callable[[], int]) -> None:
        self.layers: dict[int, ReadyWeightLease[T]] = {}
        self._ready_revision = ready_revision
        self.revision = ready_revision()
        self._checked_revision = self.revision
        self._invalidated = False
        self.closed = False
        self.quarantined = False

    def acquire(
        self,
        profile: ResidentGraphProfile,
        acquire_many: Callable[[int, tuple[int, ...]], ReadyWeightLease[T]],
        validate: Callable[[ReadyWeightLease[T]], None],
    ) -> None:
        try:
            for layer in profile.layer_ids:
                expert_ids = profile.expert_ids(layer)
                lease = acquire_many(layer, expert_ids)
                self.layers[layer] = lease
                if lease.expert_ids != expert_ids or len(lease.objects) != len(expert_ids):
                    raise ValueError("resident graph requires its exact per-layer expert order")
                validate(lease)
            self.require_current()
        except BaseException:
            self.close(wait_for_completion=lambda: None)
            raise

    def require_current(self) -> None:
        if self.closed or self.quarantined:
            raise RuntimeError("resident graph weights are closed or quarantined")
        if self._invalidated:
            raise RuntimeError("resident graph weights were invalidated by a ready-storage change")
        revision = self._ready_revision()
        if revision != self._checked_revision:
            try:
                for lease in self.layers.values():
                    lease.require_current()
            except RuntimeError:
                self._invalidated = True
                raise
            self._checked_revision = revision

    def close(self, *, wait_for_completion: Callable[[], None]) -> None:
        if self.closed or self.quarantined:
            return
        wait_for_completion()
        for lease in self.layers.values():
            lease.close()
        self.layers.clear()
        self.closed = True

    def quarantine(self) -> None:
        if self.closed or self.quarantined:
            return
        self.quarantined = True
        for lease in self.layers.values():
            lease.quarantine()
