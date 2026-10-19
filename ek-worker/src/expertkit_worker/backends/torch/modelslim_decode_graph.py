"""Resident ModelSlim I8 projections using dynamic quantization and Ascend GMM.

Routing and ready-weight lifetime use the shared resident computation. Local
sums preserve the ordinary BF16 output boundary before FP32 transport. Original
FP32 channel scales must preserve every bit through BF16 before prepared BF16
scales can be admitted. Projections use INT32 grouped dots and multiply channel
and token scales in FP32 before applying them to the dot and rounding to BF16.
Device graph execution and numerical equivalence still require qualification.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, cast

import torch

from expertkit_worker.backends.torch.decode_graph import (
    ExpertGraphOperators,
    ResidentExpertGraphWeights,
)
from expertkit_worker.backends.torch.modelslim_w8a8 import TorchModelSlimW8A8Weights
from expertkit_worker.weights import ReadyWeightLease


class ModelSlimGraphOperators(ExpertGraphOperators, Protocol):
    """Add the pinned dynamic-quantization operator to resident graph operators."""

    def npu_dynamic_quant(
        self, inputs: torch.Tensor, *, dst_type: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return I8 rows and their FP32 per-token scales."""
        ...


class ResidentModelSlimExpertGraphWeights(ResidentExpertGraphWeights):
    """Use original ModelSlim ready weights with the shared resident lifecycle."""

    def __init__(
        self,
        *,
        layer_ids: Sequence[int],
        experts_per_layer: int,
        hidden_dim: int,
        intermediate_dim: int,
        top_k: int,
        acquire_many: Callable[[int, tuple[int, ...]], ReadyWeightLease[TorchModelSlimW8A8Weights]],
        ready_revision: Callable[[], int],
        operators: ModelSlimGraphOperators | None = None,
        expert_range: tuple[int, int] | None = None,
        expert_compute: str = "swiglu",
        swiglu_limit: float = 0.0,
    ) -> None:
        self._matrices: dict[
            ReadyWeightLease[TorchModelSlimW8A8Weights],
            tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]],
        ] = {}
        self._scales: dict[
            ReadyWeightLease[TorchModelSlimW8A8Weights],
            tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]],
        ] = {}
        self._fp32_scales: dict[
            ReadyWeightLease[TorchModelSlimW8A8Weights], list[torch.Tensor]
        ] = {}
        self._scale_temporaries: list[torch.Tensor] = []
        super().__init__(
            layer_ids=layer_ids,
            experts_per_layer=experts_per_layer,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            top_k=top_k,
            acquire_many=acquire_many,
            ready_revision=ready_revision,
            operators=operators,
            expert_range=expert_range,
            expert_compute=expert_compute,
            swiglu_limit=swiglu_limit,
        )
        try:
            exact = self._prepare_projections()
        except BaseException:
            self.quarantine()
            raise
        if not exact:
            # The Host guard completed all preparation work before rejecting it.
            self.close(wait_for_completion=lambda: None)
            raise ValueError(
                "ModelSlim graph scales must be finite positive and "
                "preserve every FP32 bit through BF16"
            )
        try:
            self.require_current()
        except BaseException:
            self.close(wait_for_completion=lambda: None)
            raise

    def compute(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
        prepared_output: torch.Tensor,
    ) -> None:
        """Write the ordinary BF16 local result into the prepared destination.

        FP32 destinations carry BF16-rounded partial sums for transport. Local
        FP32 accumulation follows expert order before this output boundary.
        """
        if prepared_output.dtype == torch.bfloat16:
            super().compute(layer_id, hidden_states, expert_ids, routing_weights, prepared_output)
            return
        self._validate_inputs(hidden_states, expert_ids, routing_weights, prepared_output)
        local_output = torch.empty_like(hidden_states)
        super().compute(layer_id, hidden_states, expert_ids, routing_weights, local_output)
        prepared_output.copy_(local_output)

    def _accumulate(
        self,
        hidden_states: torch.Tensor,
        token_indices: torch.Tensor,
        contribution: torch.Tensor,
        *,
        expert_ids: torch.Tensor,
        original_rows: torch.Tensor,
        local_offset: int | torch.Tensor,
        local_count: int | torch.Tensor,
    ) -> torch.Tensor:
        """Fold FP32 contributions by expert ID and original route column.

        The full assignment permutation is unique before local padding. Its
        inverse maps each route to a real local packed row without scattering
        repeated padding indices. Capture records a fixed top-k sequence of
        device additions; replay makes no Host routing decisions.
        """
        capacity = hidden_states.shape[0]
        rows = torch.arange(capacity * self._top_k, device=contribution.device)
        inverse = torch.empty_like(rows)
        inverse.scatter_(0, original_rows.long(), rows)
        columns = torch.arange(self._top_k, device=expert_ids.device)
        order = (expert_ids.long() * self._top_k + columns).argsort(dim=1)
        assignments = rows.reshape(capacity, self._top_k).gather(1, order)
        positions = inverse.index_select(0, assignments.flatten()) - local_offset
        sorted_ids = expert_ids.gather(1, order)
        valid = (
            (sorted_ids >= self._expert_start)
            & (sorted_ids < self._expert_end)
            & (positions.reshape(capacity, self._top_k) >= 0)
            & (positions.reshape(capacity, self._top_k) < local_count)
        )
        ordered = contribution.index_select(0, positions.clamp(0, rows.numel() - 1))
        ordered = ordered.reshape(capacity, self._top_k, self._hidden_dim)
        ordered = torch.where(valid.unsqueeze(-1), ordered, 0.0)
        accumulator = torch.zeros_like(hidden_states, dtype=torch.float32)
        for column in range(self._top_k):
            accumulator = torch.where(
                valid[:, column : column + 1],
                accumulator + ordered[:, column],
                accumulator,
            )
        return accumulator

    def close(self, *, wait_for_completion: Callable[[], None]) -> None:
        """Release prepared scales after the shared lifecycle proves completion."""
        super().close(wait_for_completion=wait_for_completion)
        if self._closed and not self._quarantined:
            self._matrices.clear()
            self._scales.clear()
            self._scale_temporaries.clear()
            self._fp32_scales.clear()

    def _prepare_projections(self) -> bool:
        """Retain GMM views and synchronously guard scales before capture begins."""
        for lease in self._layers.values():
            self._matrices[lease] = ([], [], [])
            self._scales[lease] = ([], [], [])
            singleton = len(lease.objects) == 1
            for projection in range(3):
                for weight in lease.objects:
                    matrix = weight.matrices[projection]
                    self._matrices[lease][projection].append(
                        matrix.unsqueeze(0) if singleton else matrix
                    )
                    source = weight.scales[projection]
                    converted = source.to(torch.bfloat16)
                    # Publish each allocation before another device operation can fail.
                    self._scales[lease][projection].append(converted)
                    if singleton:
                        # Split-M with one weight requires [1,K,N] and [1,N] views.
                        self._scales[lease][projection][-1] = converted.unsqueeze(0)
                    round_trip = converted.float()
                    self._scale_temporaries.append(round_trip)
                    same_bits = torch.equal(source.view(torch.int32), round_trip.view(torch.int32))
                    valid = bool(torch.all(torch.isfinite(source) & (source > 0)))
                    self._scale_temporaries.clear()
                    if not same_bits or not valid:
                        return False
        for lease in self._layers.values():
            self._fp32_scales[lease] = []
            for projection in range(3):
                # Publish each table before a later device allocation can fail.
                self._fp32_scales[lease].append(
                    torch.stack([weight.scales[projection] for weight in lease.objects])
                )
        return True

    def _validate_weights(self, lease: ReadyWeightLease[TorchModelSlimW8A8Weights]) -> None:
        if lease.expert_ids != tuple(range(self._expert_start, self._expert_end)):
            raise ValueError("resident ModelSlim graph requires local experts in logical order")
        for weight in lease.objects:
            if not isinstance(weight, TorchModelSlimW8A8Weights):
                raise ValueError("resident ModelSlim graph requires ModelSlim I8 ready weights")
            if (
                weight.hidden_dim != self._hidden_dim
                or weight.intermediate_dim != self._intermediate_dim
            ):
                raise ValueError("resident ModelSlim graph dimensions do not match the model")
            if weight.dtype != torch.bfloat16:
                raise ValueError("resident ModelSlim graph requires BF16 computation")
            if self._device is None:
                self._device = weight.device
            if any(tensor.device != self._device for tensor in (*weight.matrices, *weight.scales)):
                raise ValueError("resident ModelSlim graph weights must share one device")

    def _project(
        self,
        inputs: torch.Tensor,
        lease: ReadyWeightLease[TorchModelSlimW8A8Weights],
        projection: int,
        group_list: torch.Tensor,
    ) -> torch.Tensor:
        operators = cast(ModelSlimGraphOperators, self._operators)
        quantized, token_scale = operators.npu_dynamic_quant(inputs, dst_type=torch.int8)
        # A singleton count is also its cumulative endpoint; count mode rejects it.
        group_list_type = 0 if len(lease.objects) == 1 else 1
        dot = operators.npu_grouped_matmul(
            [quantized],
            self._matrices[lease][projection],
            group_list=group_list,
            split_item=3,
            group_type=0,
            group_list_type=group_list_type,
            output_dtype=torch.int32,
        )[0]
        # Charged counts include padding; repeated endpoints skip zero-count experts.
        ends = group_list.cumsum(0)
        rows = torch.arange(inputs.shape[0], device=group_list.device)
        row_group = (rows[:, None] >= ends[None, :]).sum(1)
        channel_scale = self._fp32_scales[lease][projection].index_select(0, row_group)
        combined_scale = channel_scale * token_scale[:, None]
        return (combined_scale * dot.float()).to(inputs.dtype)
