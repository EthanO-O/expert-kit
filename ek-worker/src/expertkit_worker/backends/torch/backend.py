"""Torch eager implementation of one complete Worker-batch computation."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from importlib import import_module

import torch
import torch.nn.functional as functional
from expertkit_transport.batches import ACTIVATION_DTYPES
from expertkit_transport.quarantine import retain_until_process_exit
from expertkit_transport.tracing import trace_span

from expertkit_worker.backends.base import (
    BackendBatch,
    BackendCapabilities,
    BackendCompletion,
    BackendFatalError,
    BackendFatalReason,
    BackendResourceEstimate,
    BackendWeightUnavailable,
    ComputeBackend,
    InvalidBackendInput,
)
from expertkit_worker.backends.torch.fp4 import fp8_activation_reference
from expertkit_worker.backends.torch.modelslim_w8a8 import (
    TorchModelSlimW8A8Weights,
    modelslim_w8a8_linear,
)
from expertkit_worker.backends.torch.w8a8 import TorchW8A8Weights, w8a8_linear
from expertkit_worker.backends.torch.weights import TorchExpertWeights
from expertkit_worker.device import CpuWorkerRuntime, CudaWorkerRuntime, WorkerDeviceRuntime
from expertkit_worker.device.runtime import DeviceWork
from expertkit_worker.weights import ReadyWeightLease, WeightsNotReady

type TorchReadyWeights = TorchExpertWeights | TorchW8A8Weights | TorchModelSlimW8A8Weights

type AcquireTorchWeights = Callable[
    [int, tuple[int, ...]],
    ReadyWeightLease[TorchReadyWeights],
]

# Reserve the measured NPU temporary peak in addition to explicit expert tensors.
_ASCEND_GROUPED_TEMPORARY_BYTES = 16 * 1024 * 1024
_ASCEND_ROUTING_MAX_EXPERTS = 10240
_ASCEND_GROUPED_MAX_EXPERTS = 128


class _TorchCompletion(BackendCompletion):
    """Retain ready weights and the submitting device stream until result release."""

    def __init__(
        self,
        lease: ReadyWeightLease[TorchReadyWeights],
        work: DeviceWork,
    ) -> None:
        self._lease = lease
        self._work = work
        self._lock = threading.Lock()
        self._waited = False
        self._failed = False
        self._closed = False

    def wait_host(self) -> None:
        """Wait for submitted device work and surface asynchronous failures."""
        try:
            self._work.wait_host()
        except BaseException:
            with self._lock:
                self._failed = True
            raise
        with self._lock:
            self._waited = True

    def close(self) -> None:
        """Release ready weights after submitted Torch work is complete."""
        with self._lock:
            if self._closed:
                return
            if self._failed:
                self._closed = True
                self._lease.quarantine()
                retain_until_process_exit(self)
                return
            self._closed = True
            waited = self._waited

        close_error: BaseException | None = None
        if not waited:
            try:
                self._work.wait_host()
            except BaseException as error:
                close_error = error
        if close_error is not None:
            self._lease.quarantine()
            retain_until_process_exit(self)
            raise close_error
        self._lease.close()


class TorchBackend(ComputeBackend):
    """Execute gated experts with eager Torch operations and FP32 reduction."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        intermediate_dim: int,
        top_k: int,
        dtype: torch.dtype,
        runtime: WorkerDeviceRuntime | None = None,
        device: torch.device | str | None = None,
        acquire_many: AcquireTorchWeights,
        expert_compute: str = "swiglu",
        swiglu_limit: float = 0.0,
        linear_compute: str = "float",
    ) -> None:
        for name, value in (
            ("hidden_dim", hidden_dim),
            ("intermediate_dim", intermediate_dim),
            ("top_k", top_k),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if dtype not in ACTIVATION_DTYPES:
            raise ValueError("Torch Backend dtype must be FP16, BF16, or FP32")
        if not callable(acquire_many):
            raise TypeError("acquire_many must be callable")
        if runtime is None:
            if device is None:
                raise TypeError("TorchBackend requires runtime or device")
            parsed_device = torch.device(device)
            if parsed_device.type == "cpu":
                runtime = CpuWorkerRuntime(parsed_device)
            elif parsed_device.type == "cuda":
                runtime = CudaWorkerRuntime(parsed_device)
            elif parsed_device.type == "npu":
                from expertkit_worker.device.ascend import AscendWorkerRuntime

                runtime = AscendWorkerRuntime(parsed_device)
            else:
                raise ValueError(f"unsupported Torch device: {parsed_device}")

        if (
            expert_compute not in {"swiglu", "deepseek_v4"}
            or not math.isfinite(swiglu_limit)
            or swiglu_limit < 0
        ):
            raise ValueError("unsupported expert computation policy")
        linear_implementations = {
            "float": self._float_linear,
            "fp8_reference": self._fp8_linear,
            "w8a8": self._w8a8_linear,
            "modelslim_w8a8_dynamic": self._modelslim_w8a8_linear,
        }
        if linear_compute not in linear_implementations:
            raise ValueError("unsupported linear computation recipe")
        if linear_compute == "fp8_reference" and (hidden_dim % 128 or intermediate_dim % 128):
            raise ValueError("V4 FP8 activation widths must be divisible by 128")
        self._expert_compute = expert_compute
        self._swiglu_limit = swiglu_limit
        self._linear = linear_implementations[linear_compute]
        self._weight_type = {
            "w8a8": TorchW8A8Weights,
            "modelslim_w8a8_dynamic": TorchModelSlimW8A8Weights,
        }.get(linear_compute, TorchExpertWeights)
        self._finish_expert = (
            self._v4_output if expert_compute == "deepseek_v4" else self._swiglu_output
        )
        self._quantized_linear = linear_compute != "float"
        self._reuse_single_token = linear_compute == "float" and expert_compute == "swiglu"
        self._grouped_matmul: Callable[..., list[torch.Tensor]] | None = None
        self._moe_init_routing: Callable[..., tuple[torch.Tensor, ...]] | None = None
        if (
            self._reuse_single_token
            and runtime.device.type == "npu"
            and dtype in (torch.float16, torch.bfloat16)
            and max(hidden_dim, intermediate_dim) < 65536
        ):
            torch_npu = import_module("torch_npu")
            device_name = torch_npu.npu.get_device_name(runtime.device)
            # Other Ascend products may expose the API without supporting tensor lists.
            if device_name.startswith("Ascend910B") or device_name == "Ascend910_9362":
                self._grouped_matmul = getattr(torch_npu, "npu_grouped_matmul", None)
                if self._grouped_matmul is not None:
                    self._moe_init_routing = getattr(torch_npu, "npu_moe_init_routing_v2", None)
        self._hidden_dim = hidden_dim
        self._intermediate_dim = intermediate_dim
        self._top_k = top_k
        self._dtype = dtype
        self._acquire_many = acquire_many
        self._runtime = runtime
        self._unsafe_leases: list[ReadyWeightLease[TorchReadyWeights]] = []
        self._capabilities = BackendCapabilities(
            supports_dynamic_tokens=True,
            supports_concurrent_batches=True,
        )

    @property
    def capabilities(self) -> BackendCapabilities:
        """Return dynamic-token and bounded-concurrency support."""

        return self._capabilities

    def estimate_resources(self, max_batch_tokens: int) -> BackendResourceEstimate:
        """Estimate eager grouping, FFN intermediates, and FP32 accumulation."""

        if (
            isinstance(max_batch_tokens, bool)
            or not isinstance(max_batch_tokens, int)
            or max_batch_tokens <= 0
        ):
            raise ValueError("max_batch_tokens must be a positive integer")
        assignments = max_batch_tokens * self._top_k
        element_bytes = torch.empty((), dtype=self._dtype).element_size()
        index_bytes = assignments * 2 * 8
        mask_bytes = max_batch_tokens * self._top_k
        gathered_and_output = assignments * self._hidden_dim * element_bytes * 2
        intermediate = assignments * self._intermediate_dim * element_bytes * 3
        weighted_output = assignments * self._hidden_dim * 4
        accumulator = max_batch_tokens * self._hidden_dim * 4
        native_routing = 0
        if self._moe_init_routing is not None and max_batch_tokens > 1:
            # Add the fixed packed allocation and native indices/routes to the eager-loop bound.
            # Non-quantized initialization also retains an FP32 scale per assignment.
            native_routing = (
                assignments * self._hidden_dim * element_bytes
                + assignments * (4 + 4 + 8 + 8 + 4)
                + _ASCEND_ROUTING_MAX_EXPERTS * 8
                + 4 * _ASCEND_GROUPED_MAX_EXPERTS * 8
                + 3 * 8
            )
        return BackendResourceEstimate(
            temporary_bytes_per_active_batch=(
                index_bytes
                + mask_bytes
                + gathered_and_output
                + intermediate
                + weighted_output
                + accumulator
                + native_routing
                + (_ASCEND_GROUPED_TEMPORARY_BYTES if self._grouped_matmul is not None else 0)
                # Bound quantization, INT32 products, FP32 activation work, and row padding.
                + (
                    (max_batch_tokens + 32) * (self._hidden_dim + self._intermediate_dim) * 64
                    if self._quantized_linear or self._expert_compute == "deepseek_v4"
                    else 0
                )
            )
        )

    def submit(
        self,
        batch: BackendBatch,
        prepared_output: torch.Tensor,
    ) -> BackendCompletion:
        """Compute every local expert assignment into caller-owned output."""

        self._validate_batch(batch, prepared_output)
        lease = self._acquire(batch)

        try:
            self._validate_weights(lease, batch)

            with torch.inference_mode():
                self._compute(batch, prepared_output, lease)

            work = self._runtime.capture_current_work()
            return _TorchCompletion(lease, work)
        except BaseException:
            self._finish_failed_submission(lease)
            raise

    def _validate_batch(
        self,
        batch: BackendBatch,
        prepared_output: torch.Tensor,
    ) -> None:
        device = self._runtime.device
        if batch.hidden_dim != self._hidden_dim:
            raise InvalidBackendInput("batch hidden dimension does not match the Torch Backend")
        if batch.top_k != self._top_k:
            raise InvalidBackendInput("batch top_k does not match the Torch Backend")
        if batch.hidden_states.dtype != self._dtype:
            raise InvalidBackendInput("batch dtype does not match the Torch Backend")
        if batch.hidden_states.device != device:
            raise InvalidBackendInput("batch device does not match the Torch Backend")
        if prepared_output.shape != batch.hidden_states.shape:
            raise InvalidBackendInput("prepared output shape does not match hidden states")
        if prepared_output.dtype != self._dtype:
            raise InvalidBackendInput("prepared output dtype does not match the Torch Backend")
        if prepared_output.device != device:
            raise InvalidBackendInput("prepared output device does not match the Torch Backend")
        if not prepared_output.is_contiguous():
            raise InvalidBackendInput("prepared output must be contiguous")

    def _acquire(
        self,
        batch: BackendBatch,
    ) -> ReadyWeightLease[TorchReadyWeights]:
        try:
            return self._acquire_many(batch.layer_id, batch.distinct_expert_ids)
        except WeightsNotReady as error:
            raise BackendWeightUnavailable(error.expert_ids) from error

    def _validate_weights(
        self,
        lease: ReadyWeightLease[TorchReadyWeights],
        batch: BackendBatch,
    ) -> None:
        device = self._runtime.device
        if lease.expert_ids != batch.distinct_expert_ids:
            raise RuntimeError("ready weight lookup returned different expert IDs")
        if len(lease.objects) != len(batch.distinct_expert_ids):
            raise RuntimeError("ready weight lookup returned the wrong object count")
        for weight in lease.objects:
            if not isinstance(
                weight, TorchExpertWeights | TorchW8A8Weights | TorchModelSlimW8A8Weights
            ):
                raise RuntimeError("ready weight lookup returned a non-Torch object")
            if not isinstance(weight, self._weight_type):
                raise RuntimeError("ready weight quantization differs from the Backend recipe")
            if (
                weight.hidden_dim != self._hidden_dim
                or weight.intermediate_dim != self._intermediate_dim
            ):
                raise BackendFatalError(
                    BackendFatalReason.UNEXPECTED,
                    "ready expert weight shape does not match the Torch Backend",
                )
            if weight.dtype != self._dtype:
                raise BackendFatalError(
                    BackendFatalReason.UNEXPECTED,
                    "ready expert weight dtype does not match the Torch Backend",
                )
            if weight.device != device:
                raise BackendFatalError(
                    BackendFatalReason.UNEXPECTED,
                    "ready expert weight device does not match the Torch Backend",
                )

    @staticmethod
    def _float_linear(x: torch.Tensor, weight: TorchExpertWeights, projection: int) -> torch.Tensor:
        return functional.linear(x, weight.tensors[projection])

    @staticmethod
    def _fp8_linear(x: torch.Tensor, weight: TorchExpertWeights, projection: int) -> torch.Tensor:
        return functional.linear(fp8_activation_reference(x), weight.tensors[projection])

    @staticmethod
    def _w8a8_linear(x: torch.Tensor, weight: TorchW8A8Weights, projection: int) -> torch.Tensor:
        return w8a8_linear(x, weight.matrices[projection], weight.scales[projection])

    @staticmethod
    def _modelslim_w8a8_linear(
        x: torch.Tensor, weight: TorchModelSlimW8A8Weights, projection: int
    ) -> torch.Tensor:
        return modelslim_w8a8_linear(x, weight, projection)

    def _swiglu_output(
        self, gate: torch.Tensor, up: torch.Tensor, routing: torch.Tensor, weight: TorchReadyWeights
    ) -> torch.Tensor:
        intermediate = functional.silu(gate) * up
        return self._linear(intermediate, weight, 2).float() * routing

    def _v4_output(
        self, gate: torch.Tensor, up: torch.Tensor, routing: torch.Tensor, weight: TorchReadyWeights
    ) -> torch.Tensor:
        dtype = gate.dtype
        gate, up = gate.float(), up.float()
        if self._swiglu_limit > 0:
            gate = gate.clamp(max=self._swiglu_limit)
            up = up.clamp(min=-self._swiglu_limit, max=self._swiglu_limit)
        intermediate = (functional.silu(gate) * up * routing).to(dtype)
        return self._linear(intermediate, weight, 2).float()

    def _compute(
        self,
        batch: BackendBatch,
        prepared_output: torch.Tensor,
        lease: ReadyWeightLease[TorchReadyWeights],
    ) -> None:
        if not lease.objects:
            prepared_output.zero_()
            return

        columns = (
            self._single_token_columns(batch)
            if self._reuse_single_token and batch.token_count == 1
            else None
        )
        if (
            columns is not None
            and self._grouped_matmul is not None
            and 1 < len(lease.objects) <= 128
        ):
            self._compute_grouped_single_token(batch, prepared_output, lease, columns)
            return
        if (
            batch.token_count > 1
            and self._moe_init_routing is not None
            and self._compute_packed_prefill(batch, prepared_output, lease)
        ):
            return
        accumulator = torch.zeros_like(batch.hidden_states, dtype=torch.float32)
        for expert_id, weight in zip(lease.expert_ids, lease.objects, strict=True):
            with trace_span(
                "worker.expert.submit",
                attributes={
                    "expertkit.layer_id": batch.layer_id,
                    "expertkit.expert_id": expert_id,
                    "expertkit.timing": "host_submission",
                },
            ):
                if columns is None:
                    coordinates = torch.nonzero(batch.expert_ids == expert_id, as_tuple=False)
                    if coordinates.shape[0] == 0:
                        raise InvalidBackendInput(
                            "distinct expert metadata does not match the routing tensor"
                        )
                    token_indices = coordinates[:, 0]
                    route_indices = coordinates[:, 1]
                    expert_input = torch.index_select(batch.hidden_states, 0, token_indices)
                else:
                    column = columns.get(expert_id)
                    if column is None:
                        raise InvalidBackendInput(
                            "distinct expert metadata does not match the routing tensor"
                        )
                    expert_input = batch.hidden_states
                gate = self._linear(expert_input, weight, 0)
                up = self._linear(expert_input, weight, 1)
                if columns is None:
                    routing = batch.routing_weights[token_indices, route_indices].unsqueeze(1)
                else:
                    routing = batch.routing_weights[:, column : column + 1]
                weighted = self._finish_expert(gate, up, routing, weight)
                if columns is None:
                    accumulator.index_add_(0, token_indices, weighted)
                else:
                    accumulator.add_(weighted)
        prepared_output.copy_(accumulator.to(batch.hidden_states.dtype))

    def _compute_packed_prefill(
        self,
        batch: BackendBatch,
        prepared_output: torch.Tensor,
        lease: ReadyWeightLease[TorchReadyWeights],
    ) -> bool:
        """Pack initialized rows and retain ordinary projections and accumulation.

        Native type-1 row indices must preserve token-major order within each expert
        exactly as the baseline nonzero/gather path. One Host transfer obtains the
        selected counts and original valid-assignment total. A failed coverage check
        returns before reading packed data or the potentially uninitialized tail.
        The caller retains the lease and drains submitted work on failure.

        Returns:
            Whether computation wrote prepared output. False retains the general path
            before any potentially uninitialized packed input or index is read.
        """
        if batch.token_count <= 1 or not 1 < len(lease.objects) <= _ASCEND_GROUPED_MAX_EXPERTS:
            return False
        expert_num = lease.expert_ids[-1] + 1
        if expert_num > _ASCEND_ROUTING_MAX_EXPERTS:
            return False
        init_routing = self._moe_init_routing
        if init_routing is None:
            return False

        packed, original_rows, counts, _ = init_routing(
            batch.hidden_states,
            batch.expert_ids,
            active_num=batch.token_count * batch.top_k,
            expert_num=expert_num,
            drop_pad_mode=0,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[0, expert_num],
            quant_mode=-1,
            row_idx_type=1,
        )
        indices = torch.tensor(lease.expert_ids, dtype=torch.int64, device=batch.expert_ids.device)
        selected_counts = counts.index_select(0, indices)
        original_valid_count = (batch.expert_ids >= 0).sum().reshape(1)
        summary = torch.cat((selected_counts, original_valid_count)).cpu().tolist()
        expert_counts, valid_count = summary[:-1], summary[-1]
        if any(count <= 0 for count in expert_counts) or sum(expert_counts) != valid_count:
            return False

        # Only the initialized prefix contains defined type-1 row indices and inputs.
        original_rows = original_rows[:valid_count].long()
        token_indices = original_rows.div(batch.top_k, rounding_mode="floor")
        routing = batch.routing_weights.flatten().index_select(0, original_rows).unsqueeze(1)
        accumulator = torch.zeros_like(batch.hidden_states, dtype=torch.float32)
        start = 0
        for expert_id, weight, count in zip(
            lease.expert_ids, lease.objects, expert_counts, strict=True
        ):
            with trace_span(
                "worker.expert.submit",
                attributes={
                    "expertkit.layer_id": batch.layer_id,
                    "expertkit.expert_id": expert_id,
                    "expertkit.timing": "host_submission",
                },
            ):
                stop = start + count
                expert_input = packed[start:stop]
                gate = self._linear(expert_input, weight, 0)
                up = self._linear(expert_input, weight, 1)
                weighted = self._finish_expert(gate, up, routing[start:stop], weight)
                accumulator.index_add_(0, token_indices[start:stop], weighted)
                start = stop
        prepared_output.copy_(accumulator.to(batch.hidden_states.dtype))
        return True

    def _compute_grouped_single_token(
        self,
        batch: BackendBatch,
        prepared_output: torch.Tensor,
        lease: ReadyWeightLease[TorchReadyWeights],
        columns: dict[int, int],
    ) -> None:
        """Group projections and postprocessing without changing reduction order.

        Only fresh activation and FP32 weighting tensors are modified in place.
        """
        if any(expert_id not in columns for expert_id in lease.expert_ids):
            raise InvalidBackendInput("distinct expert metadata does not match the routing tensor")
        grouped_matmul = self._grouped_matmul
        assert grouped_matmul is not None
        with trace_span(
            "worker.backend.grouped_experts",
            attributes={
                "expertkit.layer_id": batch.layer_id,
                "expertkit.expert_count": len(lease.expert_ids),
                "expertkit.backend": "torch",
                "expertkit.timing": "host_submission",
            },
        ) as span:
            if span is not None and span.is_recording():
                span.set_attribute("expertkit.expert_ids", ",".join(map(str, lease.expert_ids)))
            inputs = [batch.hidden_states] * len(lease.objects)
            gates = grouped_matmul(
                inputs,
                [w.gate_proj.t() for w in lease.objects],
                split_item=2,
                group_type=0,
                group_list_type=0,
            )[0]
            ups = grouped_matmul(
                inputs,
                [w.up_proj.t() for w in lease.objects],
                split_item=2,
                group_type=0,
                group_list_type=0,
            )[0]
            intermediate = functional.silu(gates)
            intermediate.mul_(ups)
            outputs = grouped_matmul(
                [intermediate[row : row + 1] for row in range(len(lease.objects))],
                [w.down_proj.t() for w in lease.objects],
                split_item=2,
                group_type=0,
                group_list_type=0,
            )[0]
            routing = batch.routing_weights[:, [columns[i] for i in lease.expert_ids]].t()
            weighted = outputs.float()
            weighted.mul_(routing)
            accumulator = torch.zeros_like(batch.hidden_states, dtype=torch.float32)
            for row in range(len(lease.objects)):
                accumulator.add_(weighted[row : row + 1])
            prepared_output.copy_(accumulator.to(batch.hidden_states.dtype))

    @staticmethod
    def _single_token_columns(batch: BackendBatch) -> dict[int, int] | None:
        """Read small routing IDs once instead of synchronizing for every expert.

        Duplicate assignments retain the general path and its per-assignment
        projection, weighting, and accumulation order.
        """
        columns: dict[int, int] = {}
        for column, expert_id in enumerate(batch.expert_ids[0].cpu().tolist()):
            if expert_id < 0:
                continue
            if expert_id in columns:
                return None
            columns[expert_id] = column
        return columns

    def _finish_failed_submission(
        self,
        lease: ReadyWeightLease[TorchReadyWeights],
    ) -> None:
        synchronization_error: BaseException | None = None
        try:
            self._runtime.capture_current_work().wait_host()
        except BaseException as error:
            synchronization_error = error
        if synchronization_error is not None:
            lease.quarantine()
            self._unsafe_leases.append(lease)
            retain_until_process_exit(lease)
            raise BackendFatalError(
                BackendFatalReason.ASYNC_EXECUTION,
                str(synchronization_error),
            ) from synchronization_error
        lease.close()
