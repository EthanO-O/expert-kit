"""Symmetric per-channel INT8 weights with dynamic per-token INT8 activations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from expertkit_transport.batches import ACTIVATION_DTYPES

from expertkit_worker.weights.adapter import (
    WeightAdapter,
    WeightPlacementFatalError,
    WeightPlacementFatalReason,
)
from expertkit_worker.weights.format import SafeTensorData, SafeTensorDType

_FLOAT_DTYPES = {
    SafeTensorDType.FP16: torch.float16,
    SafeTensorDType.BF16: torch.bfloat16,
    SafeTensorDType.FP32: torch.float32,
}


@dataclass(frozen=True, slots=True)
class TorchW8A8Weights:
    """Own or view three INT8 matrices and per-output scales in gate/up/down order.

    Matrices have shapes [intermediate, hidden], [intermediate, hidden], and
    [hidden, intermediate]. Scales are [output, 1]. CPU views retain source storage;
    ready tensors own device storage. Neither kind is mutated during execution.
    """

    matrices: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    scales: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    dtype: torch.dtype

    def __post_init__(self) -> None:
        if self.dtype not in ACTIVATION_DTYPES:
            raise ValueError("W8A8 output dtype must be a floating activation dtype")
        if len(self.matrices) != 3 or len(self.scales) != 3:
            raise ValueError("W8A8 requires three projections and scales")
        gate, up, down = self.matrices
        if gate.ndim != 2 or up.shape != gate.shape or down.shape != gate.T.shape:
            raise ValueError("W8A8 projection shapes are inconsistent")
        for weight, scale in zip(self.matrices, self.scales, strict=True):
            if min(weight.shape) <= 0 or weight.dtype != torch.int8:
                raise ValueError("W8A8 weights must be nonempty INT8 matrices")
            if scale.shape != (weight.shape[0], 1) or scale.dtype not in ACTIVATION_DTYPES:
                raise ValueError("W8A8 scales must be floating per-channel column vectors")
            if any(
                t.device != gate.device or not t.is_contiguous() or t.requires_grad
                for t in (weight, scale)
            ):
                raise ValueError("W8A8 tensors must be contiguous on one device without gradients")

    @property
    def hidden_dim(self) -> int:
        """Return the input and output width."""
        return self.matrices[0].shape[1]

    @property
    def intermediate_dim(self) -> int:
        """Return the intermediate width."""
        return self.matrices[0].shape[0]

    @property
    def device(self) -> torch.device:
        """Return the common tensor device."""
        return self.matrices[0].device

    @property
    def storage_bytes(self) -> int:
        """Return the owned tensor byte count used for placement accounting."""
        return sum(t.numel() * t.element_size() for t in (*self.matrices, *self.scales))


def w8a8_linear(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Quantize each input row, accumulate INT8 products in INT32, and rescale.

    Input is [tokens, input] FP16/BF16/FP32; output is [tokens, output] in the
    input dtype on the same device. Rows are padded for CUDA INT8 GEMM alignment.
    Weight is contiguous [output, input] INT8; scale is [output, 1] FP32.
    """
    values = x.float()
    activation_scale = values.abs().amax(dim=1, keepdim=True) / 127.0
    divisor = torch.where(activation_scale == 0, 1.0, activation_scale)
    quantized = (values / divisor).round().clamp(-128, 127).to(torch.int8)
    padded_rows = max(32, ((x.shape[0] + 31) // 32) * 32) - x.shape[0]
    if padded_rows:
        quantized = functional.pad(quantized, (0, 0, 0, padded_rows))
    product = torch._int_mm(quantized.contiguous(), weight.T)
    return (product[: x.shape[0]].float() * activation_scale * scale.T).to(x.dtype)


class TorchW8A8WeightAdapter(WeightAdapter[TorchW8A8Weights, TorchW8A8Weights]):
    """Load compressed-tensors symmetric channel weights for dynamic-token INT8 GEMM."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        intermediate_dim: int,
        device: torch.device | str,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if any(
            isinstance(v, bool) or not isinstance(v, int) or v <= 0 or v % 32
            for v in (hidden_dim, intermediate_dim)
        ):
            raise ValueError("W8A8 dimensions must be positive multiples of 32")
        if compute_dtype not in ACTIVATION_DTYPES:
            raise ValueError("unsupported W8A8 computation dtype")
        self._device = torch.device(device)
        if self._device.type not in {"cpu", "cuda"} or (
            self._device.type == "cuda" and self._device.index is None
        ):
            raise ValueError("W8A8 requires CPU or an indexed CUDA device")
        self._hidden_dim, self._intermediate_dim = hidden_dim, intermediate_dim
        self._compute_dtype = compute_dtype

    @property
    def backend_name(self) -> str:
        """Identify the packed ready-cache layout in diagnostics."""
        return "torch-w8a8-token-channel"

    def make_cpu_weight(self, source: SafeTensorData) -> TorchW8A8Weights:
        """Validate the exact supported recipe and retain zero-copy CPU views."""
        matrices, scales, consumed = [], [], set()
        for roles, inputs, outputs in (
            (("gate_proj", "w1"), self._hidden_dim, self._intermediate_dim),
            (("up_proj", "w3"), self._hidden_dim, self._intermediate_dim),
            (("down_proj", "w2"), self._intermediate_dim, self._hidden_dim),
        ):
            weight = source.find_unique_suffix(tuple(f"{r}.weight" for r in roles))
            base = weight.name.removesuffix("weight")
            scale = source.tensors.get(base + "weight_scale")
            if weight.dtype is not SafeTensorDType.INT8 or weight.shape != (outputs, inputs):
                raise ValueError("unexpected W8A8 weight dtype or shape")
            if scale is None or scale.dtype not in _FLOAT_DTYPES or scale.shape != (outputs, 1):
                raise ValueError("W8A8 requires per-channel floating weight_scale [output, 1]")
            s = torch.frombuffer(scale.data, dtype=_FLOAT_DTYPES[scale.dtype]).reshape(scale.shape)
            if not torch.all(torch.isfinite(s) & (s > 0)):
                raise ValueError("W8A8 scales must be finite and positive")
            matrices.append(torch.frombuffer(weight.data, dtype=torch.int8).reshape(weight.shape))
            scales.append(s)
            consumed.update((weight.name, scale.name))
            zero = source.tensors.get(base + "weight_zero_point")
            if zero is not None:
                if (
                    zero.dtype is not SafeTensorDType.INT8
                    or zero.shape != (outputs, 1)
                    or torch.any(torch.frombuffer(zero.data, dtype=torch.int8) != 0)
                ):
                    raise ValueError("symmetric W8A8 zero points must be INT8 zeros [output, 1]")
                consumed.add(zero.name)
        if consumed != set(source.tensors):
            raise ValueError("unsupported or ambiguous Tensor in W8A8 expert bundle")
        return TorchW8A8Weights(tuple(matrices), tuple(scales), self._compute_dtype)

    def make_ready_weight(
        self, cpu_weight: TorchW8A8Weights, *, layer_id: int, expert_id: int
    ) -> TorchW8A8Weights:
        """Place INT8 matrices and FP32 scales; synchronize before READY publication."""
        del layer_id, expert_id
        if cpu_weight.device.type != "cpu":
            raise ValueError("W8A8 cached tensors must be on CPU")
        try:
            ready = TorchW8A8Weights(
                tuple(w.to(self._device, copy=True) for w in cpu_weight.matrices),
                tuple(
                    s.to(device=self._device, dtype=torch.float32, copy=True)
                    for s in cpu_weight.scales
                ),
                self._compute_dtype,
            )
            if self._device.type == "cuda":
                torch.cuda.current_stream(self._device).synchronize()
            return ready
        except torch.OutOfMemoryError as error:
            raise WeightPlacementFatalError(
                WeightPlacementFatalReason.DEVICE_OOM, str(error)
            ) from error
        except RuntimeError as error:
            if self._device.type == "cuda":
                raise WeightPlacementFatalError(
                    WeightPlacementFatalReason.DEVICE_FAILURE, str(error)
                ) from error
            raise

    def cpu_extra_bytes(self) -> int:
        """Return zero because the CPU cache retains source views."""
        return 0

    def source_tensor_bytes(self) -> int:
        """Bound INT8 matrices, FP32 scales, and optional INT8 zero points."""
        return 3 * self._hidden_dim * self._intermediate_dim + 5 * (
            2 * self._intermediate_dim + self._hidden_dim
        )

    def ready_weight_bytes(self) -> int:
        """Count INT8 matrices and FP32 per-channel scales."""
        return 3 * self._hidden_dim * self._intermediate_dim + 4 * (
            2 * self._intermediate_dim + self._hidden_dim
        )

    def conversion_temporary_bytes(self) -> int:
        """Return zero because copies allocate directly into final storage."""
        return 0
