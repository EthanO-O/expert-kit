"""Resident packed weights for capturing the ordinary Ascend fused MoE pipeline.

Preparation freezes all local weight lists and converts ModelSlim scales once.
Computation maps global routes to this shard, then calls run_ascend_fused_moe.
Capture, HCCL and replay remain the executor's responsibility.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import NotRequired, TypedDict, Unpack

import torch
from expertkit_transport.quarantine import retain_until_process_exit

from expertkit_worker.backends.graph_common import ResidentGraphProfile, ResidentWeightOwner
from expertkit_worker.weights import ReadyWeightLease

from .op import AscendFusedComputeType, run_ascend_fused_moe
from .weights import AscendFusedExpertWeights, AscendFusedModelSlimW8A8ExpertWeights

type PackedReadyWeights = AscendFusedExpertWeights | AscendFusedModelSlimW8A8ExpertWeights


class _GraphWeightsOptions(TypedDict):
    layer_ids: Sequence[int]
    experts_per_layer: int
    hidden_dim: int
    intermediate_dim: int
    top_k: int
    acquire_many: Callable[[int, tuple[int, ...]], ReadyWeightLease[PackedReadyWeights]]
    ready_revision: Callable[[], int]
    expert_range: NotRequired[tuple[int, int] | None]
    expert_compute: NotRequired[str]
    swiglu_limit: NotRequired[float]


_COMPUTE_TYPE_TO_WEIGHTS = {
    AscendFusedComputeType.UNQUANTIZED: AscendFusedExpertWeights,
    AscendFusedComputeType.DYNAMIC_W8A8: AscendFusedModelSlimW8A8ExpertWeights,
}


@dataclass
class _PreparedLayer:
    w13: list[torch.Tensor] = field(default_factory=list)
    w2: list[torch.Tensor] = field(default_factory=list)
    w13_scale: list[torch.Tensor] | None = None
    w2_scale: list[torch.Tensor] | None = None


class ResidentAscendFusedGraphWeights:
    """Compose resident ownership with the same operator path used by eager submit."""

    def __init__(
        self,
        *,
        layer_ids: Sequence[int],
        experts_per_layer: int,
        hidden_dim: int,
        intermediate_dim: int,
        top_k: int,
        acquire_many: Callable[[int, tuple[int, ...]], ReadyWeightLease[PackedReadyWeights]],
        ready_revision: Callable[[], int],
        expert_range: tuple[int, int] | None = None,
        expert_compute: str = "swiglu",
        swiglu_limit: float = 0.0,
        compute_type: AscendFusedComputeType = AscendFusedComputeType.UNQUANTIZED,
    ) -> None:
        self._profile = ResidentGraphProfile(
            layer_ids,
            experts_per_layer,
            hidden_dim,
            intermediate_dim,
            top_k,
            expert_range,
            expert_compute,
            swiglu_limit,
        )
        self._compute_type = compute_type
        self._weight_type = _COMPUTE_TYPE_TO_WEIGHTS[compute_type]
        if expert_compute == "deepseek_v4" and compute_type is AscendFusedComputeType.UNQUANTIZED:
            raise ValueError("resident V4 graph requires ModelSlim I8 weights")
        self._owner = ResidentWeightOwner[PackedReadyWeights](ready_revision)
        self._device: torch.device | None = None
        self._prepared: dict[int, _PreparedLayer] = {}
        try:
            self._owner.acquire(self._profile, acquire_many, self._validate_weights)
        except BaseException:
            self._owner.close(wait_for_completion=lambda: None)
            raise
        try:
            self._prepare_layers(compute_type)
            self.require_current()
        except BaseException:
            self.quarantine()
            raise

    def _prepare_layers(self, compute_type: AscendFusedComputeType) -> None:
        for layer_id, lease in self._owner.layers.items():
            prepared = _PreparedLayer()
            self._prepared[layer_id] = prepared
            if compute_type is AscendFusedComputeType.DYNAMIC_W8A8:
                prepared.w13_scale, prepared.w2_scale = [], []
            for weight in lease.objects:
                prepared.w13.append(weight.w13)
                prepared.w2.append(weight.w2)
                if not isinstance(weight, AscendFusedModelSlimW8A8ExpertWeights):
                    continue
                assert prepared.w13_scale is not None
                assert prepared.w2_scale is not None
                # Publish each allocation before a later device operation can fail.
                prepared.w13_scale.append(weight.w13_scale.to(torch.bfloat16))
                prepared.w2_scale.append(weight.w2_scale.to(torch.bfloat16))

    def _validate_weights(self, lease: ReadyWeightLease[PackedReadyWeights]) -> None:
        start, end = self._profile.interval
        if lease.expert_ids != tuple(range(start, end)) or len(lease.objects) != end - start:
            raise ValueError(
                "resident Ascend fused graph requires the complete local expert interval"
            )
        for weight in lease.objects:
            if not isinstance(weight, self._weight_type):
                raise ValueError("resident Ascend fused graph received the wrong ready weight type")
            if (weight.hidden_dim, weight.intermediate_dim) != (
                self._profile.hidden_dim,
                self._profile.intermediate_dim,
            ):
                raise ValueError("resident Ascend fused graph dimensions do not match the model")
            dtype = (
                weight.dtype
                if isinstance(weight, AscendFusedModelSlimW8A8ExpertWeights)
                else weight.w13.dtype
            )
            if dtype != torch.bfloat16:
                raise ValueError("resident Ascend fused graph requires BF16 computation")
            if self._device is None:
                self._device = weight.device
            tensors = [weight.w13, weight.w2]
            if isinstance(weight, AscendFusedModelSlimW8A8ExpertWeights):
                tensors.extend((weight.w13_scale, weight.w2_scale))
            if any(t.device != self._device for t in tensors):
                raise ValueError("resident Ascend fused graph weights must share one device")

    @property
    def revision(self) -> int:
        return self._owner.revision

    @property
    def layer_ids(self) -> tuple[int, ...]:
        return tuple(self._owner.layers)

    def require_current(self) -> None:
        self._owner.require_current()

    def compute(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
        prepared_output: torch.Tensor,
    ) -> None:
        if self._owner.closed or self._owner.quarantined:
            raise RuntimeError("resident graph weights are closed or quarantined")
        self._profile.validate_inputs(
            self._device,
            hidden_states,
            expert_ids,
            routing_weights,
            prepared_output,
        )
        prepared = self._prepared[layer_id]
        start, end = self._profile.interval
        # Identify routes assigned to this Worker's contiguous shard [start, end).
        local_mask = (expert_ids >= start) & (expert_ids < end)
        # Map global IDs to local weight-list indices. Nonlocal routes use the
        # inactive sentinel E = end - start, one past the last local expert.
        local_ids = torch.where(local_mask, expert_ids - start, end - start)
        # Cast routing weights to the activation dtype and zero nonlocal contributions.
        local_weights = routing_weights.to(hidden_states.dtype).masked_fill(~local_mask, 0)
        result = run_ascend_fused_moe(
            hidden_states,
            local_ids,
            local_weights,
            w13=prepared.w13,
            w2=prepared.w2,
            compute_type=self._compute_type,
            w13_scale=prepared.w13_scale,
            w2_scale=prepared.w2_scale,
            swiglu_limit=self._profile.swiglu_limit,
        )
        # FP32 transport carries the same BF16 partial result as ordinary eager execution.
        prepared_output.copy_(result)

    def close(self, *, wait_for_completion: Callable[[], None]) -> None:
        try:
            self._owner.close(wait_for_completion=wait_for_completion)
        except BaseException:
            self.quarantine()
            raise
        if self._owner.closed:
            self._prepared.clear()

    def quarantine(self) -> None:
        if self._owner.closed or self._owner.quarantined:
            return
        self._owner.quarantine()
        retain_until_process_exit(self)


def create_modelslim_graph_weights(
    **options: Unpack[_GraphWeightsOptions],
) -> ResidentAscendFusedGraphWeights:
    """Select the ModelSlim recipe without inheriting another backend's computation."""
    return ResidentAscendFusedGraphWeights(
        compute_type=AscendFusedComputeType.DYNAMIC_W8A8,
        **options,
    )
