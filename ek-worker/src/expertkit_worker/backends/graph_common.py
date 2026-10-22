"""Shared graph profile validation and resident lease ownership.

Providers compose these helpers with independent computations. Capture, replay,
communication, routing, projections and reduction are owned by their callers.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from expertkit_worker.weights import ReadyWeightLease


@dataclass(frozen=True)
class ResidentGraphProfile:
    layer_ids: Sequence[int]
    experts_per_layer: int
    hidden_dim: int
    intermediate_dim: int
    top_k: int
    expert_range: tuple[int, int] | None = None
    expert_compute: str = "swiglu"
    swiglu_limit: float = 0.0

    def __post_init__(self) -> None:
        for name in ("experts_per_layer", "hidden_dim", "intermediate_dim", "top_k"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not (
            (self.expert_compute == "swiglu" and self.swiglu_limit == 0)
            or (self.expert_compute == "deepseek_v4" and self.swiglu_limit == 10)
        ):
            raise ValueError("resident graph requires swiglu limit0 or deepseek_v4 limit10")
        maximum = 256 if self.expert_compute == "deepseek_v4" else 128
        if self.experts_per_layer > maximum or self.top_k > self.experts_per_layer:
            raise ValueError(f"resident graph requires top_k <= experts_per_layer <= {maximum}")
        layers = tuple(self.layer_ids)
        if (
            not layers
            or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in layers)
            or tuple(sorted(set(layers))) != layers
        ):
            raise ValueError("layer_ids must be nonempty, sorted, unique and nonnegative")
        object.__setattr__(self, "layer_ids", layers)
        start, end = self.interval
        if not 0 <= start < end <= self.experts_per_layer:
            raise ValueError("resident expert interval must be within the model")
        if end - start > 128:
            raise ValueError("resident GMM requires at most 128 local experts")
        if self.expert_compute == "deepseek_v4" and end - start < 2:
            raise ValueError("resident V4 INT32 GMM requires at least two local experts")

    @property
    def interval(self) -> tuple[int, int]:
        return self.expert_range if self.expert_range is not None else (0, self.experts_per_layer)

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
        self.closed = False
        self.quarantined = False

    def acquire(
        self,
        profile: ResidentGraphProfile,
        acquire_many: Callable[[int, tuple[int, ...]], ReadyWeightLease[T]],
        validate: Callable[[ReadyWeightLease[T]], None],
    ) -> None:
        start, end = profile.interval
        try:
            for layer in profile.layer_ids:
                lease = acquire_many(layer, tuple(range(start, end)))
                self.layers[layer] = lease
                validate(lease)
            self.require_current()
        except BaseException:
            self.close(wait_for_completion=lambda: None)
            raise

    def require_current(self) -> None:
        if self.closed or self.quarantined:
            raise RuntimeError("resident graph weights are closed or quarantined")
        if self._ready_revision() != self.revision:
            raise RuntimeError("resident graph weights were invalidated by a ready-storage change")

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
