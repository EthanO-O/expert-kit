"""Ascend ModelSlim W8A8_DYNAMIC expert weights and linear operation."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

import torch
from expertkit_transport.batches import ACTIVATION_DTYPES

from expertkit_worker.device import WorkerDeviceRuntime
from expertkit_worker.weights.adapter import (
    WeightAdapter,
    WeightPlacementFatalError,
    WeightPlacementFatalReason,
)
from expertkit_worker.weights.format import SafeTensorData, SafeTensorDType

_SCALE_DTYPES = {SafeTensorDType.BF16: torch.bfloat16, SafeTensorDType.FP32: torch.float32}
_ROLES = (("gate_proj", "w1"), ("up_proj", "w3"), ("down_proj", "w2"))


@dataclass(frozen=True, slots=True)
class TorchModelSlimW8A8CpuWeight:
    """CPU views of three ModelSlim I8 matrices and their channel scales."""

    matrices: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    scales: tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True, slots=True)
class TorchModelSlimW8A8Weights:
    """NPU-ready ModelSlim matrices in operator layout ``[input, output]``."""

    matrices: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    scales: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    dtype: torch.dtype

    def __post_init__(self) -> None:
        if self.dtype not in ACTIVATION_DTYPES:
            raise ValueError("ModelSlim output dtype must be a floating activation dtype")
        if len(self.matrices) != 3 or len(self.scales) != 3:
            raise ValueError("ModelSlim W8A8 requires three projections and scales")
        gate, up, down = self.matrices
        if gate.ndim != 2 or up.shape != gate.shape or down.shape != gate.T.shape:
            raise ValueError("ModelSlim projection shapes are inconsistent")
        for matrix, scale in zip(self.matrices, self.scales, strict=True):
            if matrix.dtype != torch.int8 or not matrix.is_contiguous():
                raise ValueError("ModelSlim matrices must be contiguous I8 tensors")
            if scale.dtype != torch.float32 or scale.shape != (matrix.shape[1],):
                raise ValueError("ModelSlim scales must be flattened FP32 output vectors")
            if matrix.device != scale.device or matrix.requires_grad or scale.requires_grad:
                raise ValueError("ModelSlim tensors must share a device without gradients")

    @property
    def hidden_dim(self) -> int:
        """Return the routed expert input and output width."""

        return self.matrices[0].shape[0]

    @property
    def intermediate_dim(self) -> int:
        """Return the routed expert intermediate width."""

        return self.matrices[0].shape[1]

    @property
    def device(self) -> torch.device:
        """Return the common NPU device."""

        return self.matrices[0].device

    @property
    def storage_bytes(self) -> int:
        """Return resident matrix and scale bytes."""

        return sum(
            tensor.numel() * tensor.element_size() for tensor in (*self.matrices, *self.scales)
        )


class TorchModelSlimW8A8WeightAdapter(
    WeightAdapter[TorchModelSlimW8A8CpuWeight, TorchModelSlimW8A8Weights]
):
    """Validate ModelSlim expert blobs and place them on an Ascend device."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        intermediate_dim: int,
        runtime: WorkerDeviceRuntime,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if min(hidden_dim, intermediate_dim) <= 0:
            raise ValueError("ModelSlim dimensions must be positive")
        if compute_dtype not in ACTIVATION_DTYPES:
            raise ValueError("unsupported ModelSlim computation dtype")
        if runtime.device.type != "npu":
            raise ValueError("ModelSlim W8A8_DYNAMIC requires an NPU runtime")
        self._hidden_dim = hidden_dim
        self._intermediate_dim = intermediate_dim
        self._runtime = runtime
        self._compute_dtype = compute_dtype

    @property
    def backend_name(self) -> str:
        """Identify the Ascend ModelSlim storage format."""

        return "ascend-modelslim-w8a8-dynamic"

    def make_cpu_weight(self, source: SafeTensorData) -> TorchModelSlimW8A8CpuWeight:
        """Validate exact Qwen or V4 projection tensors and retain source views."""

        matrices: list[torch.Tensor] = []
        scales: list[torch.Tensor] = []
        consumed: set[str] = set()
        for roles, inputs, outputs in (
            (_ROLES[0], self._hidden_dim, self._intermediate_dim),
            (_ROLES[1], self._hidden_dim, self._intermediate_dim),
            (_ROLES[2], self._intermediate_dim, self._hidden_dim),
        ):
            weight = source.find_unique_suffix(tuple(f"{role}.weight" for role in roles))
            scale = source.tensors.get(weight.name.removesuffix("weight") + "weight_scale")
            offset = source.tensors.get(weight.name.removesuffix("weight") + "weight_offset")
            if weight.dtype is not SafeTensorDType.INT8 or weight.shape != (outputs, inputs):
                raise ValueError("unexpected ModelSlim W8A8 weight dtype or shape")
            if scale is None or scale.dtype not in _SCALE_DTYPES or scale.shape != (outputs, 1):
                raise ValueError("ModelSlim scales must be BF16 or F32 with shape [output, 1]")
            if offset is None or offset.dtype is not scale.dtype or offset.shape != scale.shape:
                raise ValueError("ModelSlim offsets must match scale dtype and shape")
            scale_tensor = torch.frombuffer(scale.data, dtype=_SCALE_DTYPES[scale.dtype]).reshape(
                scale.shape
            )
            offset_tensor = torch.frombuffer(
                offset.data, dtype=_SCALE_DTYPES[offset.dtype]
            ).reshape(offset.shape)
            if not torch.all(torch.isfinite(scale_tensor) & (scale_tensor > 0)):
                raise ValueError("ModelSlim scales must be finite and positive")
            if torch.any(offset_tensor != 0):
                raise ValueError("nonzero ModelSlim weight offsets are unsupported")
            matrices.append(torch.frombuffer(weight.data, dtype=torch.int8).reshape(weight.shape))
            scales.append(scale_tensor)
            consumed.update((weight.name, scale.name, offset.name))
        if consumed != set(source.tensors):
            raise ValueError("unsupported or ambiguous Tensor in ModelSlim expert bundle")
        return TorchModelSlimW8A8CpuWeight(tuple(matrices), tuple(scales))

    def make_ready_weight(
        self,
        cpu_weight: TorchModelSlimW8A8CpuWeight,
        *,
        layer_id: int,
        expert_id: int,
    ) -> TorchModelSlimW8A8Weights:
        """Transpose matrices, move final storage, and synchronize publication."""

        del layer_id, expert_id
        if any(matrix.device.type != "cpu" for matrix in cpu_weight.matrices):
            raise ValueError("ModelSlim cached matrices must be on CPU")
        device = self._runtime.device
        try:
            ready = TorchModelSlimW8A8Weights(
                tuple(
                    matrix.transpose(0, 1).contiguous().to(device=device, copy=True)
                    for matrix in cpu_weight.matrices
                ),
                tuple(
                    scale.reshape(-1).to(device=device, dtype=torch.float32, copy=True)
                    for scale in cpu_weight.scales
                ),
                self._compute_dtype,
            )
            self._runtime.capture_current_work().wait_host()
            return ready
        except torch.OutOfMemoryError as error:
            raise WeightPlacementFatalError(
                WeightPlacementFatalReason.DEVICE_OOM, str(error)
            ) from error
        except RuntimeError as error:
            raise WeightPlacementFatalError(
                WeightPlacementFatalReason.DEVICE_FAILURE, str(error)
            ) from error

    def cpu_extra_bytes(self) -> int:
        """Return zero because CPU tensors view the retained source buffer."""

        return 0

    def source_tensor_bytes(self) -> int:
        """Return a conservative bound covering I8 matrices and F32 auxiliaries."""

        elements = 3 * self._hidden_dim * self._intermediate_dim
        channels = 2 * (2 * self._intermediate_dim + self._hidden_dim)
        return elements + channels * 4

    def ready_weight_bytes(self) -> int:
        """Return exact final I8 matrix and FP32 scale storage."""

        elements = 3 * self._hidden_dim * self._intermediate_dim
        channels = 2 * self._intermediate_dim + self._hidden_dim
        return elements + channels * 4

    def conversion_temporary_bytes(self) -> int:
        """Return zero because final tensors are allocated directly."""

        return 0

    def host_conversion_temporary_bytes(self) -> int:
        """Return zero because CPU views and direct device copies are retained."""

        return 0


def modelslim_w8a8_linear(
    x: torch.Tensor,
    weight: TorchModelSlimW8A8Weights,
    projection: int,
) -> torch.Tensor:
    """Run one ModelSlim dynamic-I8 projection with the pinned torch-npu API."""

    torch_npu = import_module("torch_npu")
    quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(x, dst_type=torch.int8)
    squeeze = pertoken_scale.ndim == 2
    if squeeze:
        quantized_x = quantized_x.squeeze(1)
        pertoken_scale = pertoken_scale.squeeze(1)
    output = torch_npu.npu_quant_matmul(
        quantized_x,
        weight.matrices[projection],
        weight.scales[projection],
        pertoken_scale=pertoken_scale,
        output_dtype=x.dtype,
    )
    return output.unsqueeze(1) if squeeze else output
