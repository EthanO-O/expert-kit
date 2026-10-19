"""Resident BF16 expert computation using device routing and Ascend GMM.

The owner acquires all local experts before capture and retains them through every
replay. This module owns no communication or device graph; its caller orders
device completion before closing the resident weights.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from importlib import import_module
from typing import Protocol

import torch
import torch.nn.functional as functional
from expertkit_transport.quarantine import retain_until_process_exit

from expertkit_worker.backends.torch.weights import TorchExpertWeights
from expertkit_worker.weights import ReadyWeightLease


class ExpertGraphOperators(Protocol):
    """Device operators supplied by torch-npu at the computation boundary."""

    def npu_moe_init_routing_v2(
        self, hidden_states: torch.Tensor, expert_ids: torch.Tensor, **options: object
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, object]:
        """Return packed rows, original assignment indices and expert counts."""
        ...

    def npu_grouped_matmul(
        self, inputs: list[torch.Tensor], weights: list[torch.Tensor], **options: object
    ) -> list[torch.Tensor]:
        """Multiply device-grouped rows against fixed resident weight views."""
        ...


class ResidentExpertGraphWeights:
    """Retain all local routed experts and compute without Host routing decisions.

    Inputs contain valid expert IDs for every row, including padded rows. The
    model's native router guarantees IDs and finite FP32 routing weights; this
    boundary validates metadata without copying Tensor values to the Host.
    """

    def __init__(
        self,
        *,
        layer_ids: Sequence[int],
        experts_per_layer: int,
        hidden_dim: int,
        intermediate_dim: int,
        top_k: int,
        acquire_many: Callable[[int, tuple[int, ...]], ReadyWeightLease[TorchExpertWeights]],
        ready_revision: Callable[[], int],
        operators: ExpertGraphOperators | None = None,
        expert_range: tuple[int, int] | None = None,
        expert_compute: str = "swiglu",
        swiglu_limit: float = 0.0,
    ) -> None:
        for name, value in (
            ("experts_per_layer", experts_per_layer),
            ("hidden_dim", hidden_dim),
            ("intermediate_dim", intermediate_dim),
            ("top_k", top_k),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not (
            (expert_compute == "swiglu" and swiglu_limit == 0)
            or (expert_compute == "deepseek_v4" and swiglu_limit == 10)
        ):
            raise ValueError("resident graph requires swiglu limit0 or deepseek_v4 limit10")
        max_experts = 256 if expert_compute == "deepseek_v4" else 128
        if experts_per_layer > max_experts or top_k > experts_per_layer:
            raise ValueError(f"resident graph requires top_k <= experts_per_layer <= {max_experts}")
        layers = tuple(layer_ids)
        if (
            not layers
            or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in layers)
            or tuple(sorted(set(layers))) != layers
        ):
            raise ValueError("layer_ids must be nonempty, sorted, unique and nonnegative")
        start, end = expert_range if expert_range is not None else (0, experts_per_layer)
        if not 0 <= start < end <= experts_per_layer:
            raise ValueError("resident expert interval must be within the model")
        if end - start > 128:
            raise ValueError("resident GMM requires at most 128 local experts")
        if expert_compute == "deepseek_v4" and end - start < 2:
            raise ValueError("resident V4 INT32 GMM requires at least two local experts")
        self._expert_compute = expert_compute
        self._swiglu_limit = swiglu_limit
        self._expert_start = start
        self._expert_end = end
        self._operators = operators
        self._ready_revision = ready_revision
        self._revision = ready_revision()
        self._hidden_dim = hidden_dim
        self._intermediate_dim = intermediate_dim
        self._top_k = top_k
        self._experts_per_layer = experts_per_layer
        self._layers: dict[int, ReadyWeightLease[TorchExpertWeights]] = {}
        self._device: torch.device | None = None
        self._closed = False
        self._quarantined = False
        expert_ids = tuple(range(start, end))
        try:
            for layer_id in layers:
                lease = acquire_many(layer_id, expert_ids)
                try:
                    self._validate_weights(lease)
                except BaseException:
                    lease.close()
                    raise
                self._layers[layer_id] = lease
            self.require_current()
            if operators is None:
                if self._device is None or self._device.type != "npu":
                    raise ValueError("resident Ascend graph weights require an NPU device")
                self._operators = import_module("torch_npu")
        except BaseException:
            for lease in self._layers.values():
                lease.close()
            self._layers.clear()
            raise

    @property
    def revision(self) -> int:
        """Return the ready-storage revision frozen during preparation."""
        return self._revision

    @property
    def layer_ids(self) -> tuple[int, ...]:
        """Return the fixed ordered routed layers."""
        return tuple(self._layers)

    def require_current(self) -> None:
        """Reject work after closure, quarantine or ready-storage invalidation."""
        if self._closed or self._quarantined:
            raise RuntimeError("resident graph weights are closed or quarantined")
        if self._ready_revision() != self._revision:
            raise RuntimeError("resident graph weights were invalidated by a ready-storage change")

    def compute(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
        prepared_output: torch.Tensor,
    ) -> None:
        """Write the local FP32-weighted sum into prepared BF16 or FP32 output.

        Args:
            hidden_states: BF16 device matrix shaped [capacity, hidden_dim].
            expert_ids: Int32 device matrix shaped [capacity, top_k], all IDs valid.
            routing_weights: FP32 device matrix aligned with expert_ids.
            prepared_output: Caller-owned BF16/FP32 output with hidden_states' shape.

        Note:
            The caller checks require_current before admitting a whole step and
            retains graph intermediates and leases until device completion.
        """
        if self._closed or self._quarantined:
            raise RuntimeError("resident graph weights are closed or quarantined")
        weights = self._layers[layer_id]
        self._validate_inputs(hidden_states, expert_ids, routing_weights, prepared_output)
        operators = self._operators
        assert operators is not None
        capacity = hidden_states.shape[0]
        packed, original_rows, counts, _ = operators.npu_moe_init_routing_v2(
            hidden_states,
            expert_ids,
            active_num=capacity * self._top_k,
            expert_num=self._experts_per_layer,
            drop_pad_mode=0,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[0, self._experts_per_layer],
            quant_mode=-1,
            row_idx_type=1,
        )
        global_original_rows = original_rows
        group_list = counts[self._expert_start : self._expert_end].to(torch.int64)
        local_rows = None
        offset: int | torch.Tensor = 0
        local_count: int | torch.Tensor = capacity * self._top_k
        if self._expert_start != 0 or self._expert_end != self._experts_per_layer:
            # All global assignments are packed, so every source index is defined.
            # The local prefix is useful; fixed tail gathers stay in bounds.
            rows = torch.arange(capacity * self._top_k, device=hidden_states.device)
            offset = counts[: self._expert_start].sum()
            indices = (rows + offset).clamp(max=capacity * self._top_k - 1).long()
            packed = packed.index_select(0, indices)
            original_rows = original_rows.index_select(0, indices)
            local_count = group_list.sum()
            local_rows = rows < local_count
            # GMM requires group counts to cover its fixed input shape. Charge
            # harmless padding rows to the last real local expert, then mask them.
            group_list = group_list.clone()
            group_list[-1] += capacity * self._top_k - local_count

        original_rows = original_rows.to(torch.int64)
        token_indices = original_rows.div(self._top_k, rounding_mode="floor")
        routing = routing_weights.flatten().index_select(0, original_rows).unsqueeze(1)
        gate = self._project(packed, weights, 0, group_list)
        up = self._project(packed, weights, 1, group_list)
        if self._expert_compute == "deepseek_v4":
            gate = gate.float().clamp(max=self._swiglu_limit)
            up = up.float().clamp(min=-self._swiglu_limit, max=self._swiglu_limit)
            intermediate = (functional.silu(gate) * up * routing).to(torch.bfloat16)
        else:
            intermediate = functional.silu(gate) * up
        down = self._project(intermediate, weights, 2, group_list)
        if local_rows is not None:
            # Padding must contribute zero, even if its projections contain NaNs.
            down_fp32 = torch.where(local_rows.unsqueeze(1), down.float(), 0.0)
        else:
            down_fp32 = down.float()
        contribution = down_fp32 if self._expert_compute == "deepseek_v4" else down_fp32 * routing
        accumulator = self._accumulate(
            hidden_states,
            token_indices,
            contribution,
            expert_ids=expert_ids,
            original_rows=global_original_rows,
            local_offset=offset,
            local_count=local_count,
        )
        prepared_output.copy_(accumulator)

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
        """Return the FP32 local sum using the original indexed reduction.

        Args:
            original_rows: Complete global assignment permutation before local
                gathers duplicate any padding rows.
            local_offset: Device-computed start of this shard's packed rows.
            local_count: Real assignment count before charging padding to GMM.
        """
        accumulator = torch.zeros_like(hidden_states, dtype=torch.float32)
        accumulator.index_add_(0, token_indices, contribution)
        return accumulator

    def close(self, *, wait_for_completion: Callable[[], None]) -> None:
        """Release leases only after the owning device work has completed.

        A failed completion wait quarantines the original weights and this owner
        until process exit. Callers must destroy device graphs before this method
        returns successfully so no replay can reference released weights.
        """
        if self._closed or self._quarantined:
            return
        try:
            wait_for_completion()
        except BaseException:
            self.quarantine()
            raise
        for lease in self._layers.values():
            lease.close()
        self._layers.clear()
        self._closed = True

    def quarantine(self) -> None:
        """Retain graph storage when failed device work may still read it."""
        if self._closed or self._quarantined:
            return
        self._quarantined = True
        for lease in self._layers.values():
            lease.quarantine()
        retain_until_process_exit(self)

    def _project(
        self,
        inputs: torch.Tensor,
        lease: ReadyWeightLease[TorchExpertWeights],
        projection: int,
        group_list: torch.Tensor,
    ) -> torch.Tensor:
        assert self._operators is not None
        return self._operators.npu_grouped_matmul(
            [inputs],
            [weight.tensors[projection].t() for weight in lease.objects],
            group_list=group_list,
            split_item=2,
            group_type=0,
            group_list_type=1,
        )[0]

    def _validate_weights(self, lease: ReadyWeightLease[TorchExpertWeights]) -> None:
        if self._expert_compute == "deepseek_v4":
            raise ValueError("resident V4 graph requires ModelSlim I8 weights")
        if lease.expert_ids != tuple(range(self._expert_start, self._expert_end)):
            raise ValueError("resident graph requires its local experts in logical order")
        for weight in lease.objects:
            if not isinstance(weight, TorchExpertWeights):
                raise ValueError("resident graph requires unquantized Torch expert weights")
            if weight.gate_proj.shape != (self._intermediate_dim, self._hidden_dim):
                raise ValueError("resident graph weight dimensions do not match the model")
            if weight.gate_proj.dtype != torch.bfloat16:
                raise ValueError("resident graph weights must use BF16")
            if self._device is None:
                self._device = weight.gate_proj.device
            if weight.gate_proj.device != self._device:
                raise ValueError("resident graph weights must share one device")

    def _validate_inputs(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        if (
            hidden_states.ndim != 2
            or hidden_states.shape[0] <= 0
            or hidden_states.shape[1] != self._hidden_dim
            or output.shape != hidden_states.shape
        ):
            raise ValueError("resident graph requires nonempty hidden/output matrices")
        if expert_ids.shape != (hidden_states.shape[0], self._top_k):
            raise ValueError("resident graph expert IDs have the wrong shape")
        if routing_weights.shape != expert_ids.shape:
            raise ValueError("resident graph routing weights have the wrong shape")
        if (
            hidden_states.dtype != torch.bfloat16
            or output.dtype not in (torch.bfloat16, torch.float32)
            or expert_ids.dtype != torch.int32
            or routing_weights.dtype != torch.float32
        ):
            raise ValueError("resident graph requires BF16 activations, int32 IDs and FP32 routing")
        if any(
            tensor.device != self._device or not tensor.is_contiguous()
            for tensor in (hidden_states, expert_ids, routing_weights, output)
        ):
            raise ValueError("resident graph inputs must be contiguous on the weights' device")
