"""Computation-ready Torch expert weight objects."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from expertkit_transport.batches import ACTIVATION_DTYPES


@dataclass(frozen=True, slots=True)
class TorchExpertWeights:
    """Hold one gated FFN's final Torch tensors on its computation device.

    Attributes:
        gate_proj: Contiguous matrix shaped ``[intermediate_dim, hidden_dim]``.
        up_proj: Contiguous matrix shaped ``[intermediate_dim, hidden_dim]``.
        down_proj: Contiguous matrix shaped ``[hidden_dim, intermediate_dim]``.

    Note:
        These tensors are already converted to the compute dtype and device. The
        Backend never moves or converts them on the request path.
    """

    gate_proj: torch.Tensor
    up_proj: torch.Tensor
    down_proj: torch.Tensor

    def __post_init__(self) -> None:
        tensors = (self.gate_proj, self.up_proj, self.down_proj)
        if any(tensor.ndim != 2 for tensor in tensors):
            raise ValueError("Torch expert weights must be two-dimensional matrices")
        if any(tensor.dtype not in ACTIVATION_DTYPES for tensor in tensors):
            raise ValueError("Torch expert weights must use FP16, BF16, or FP32")
        if any(tensor.device != self.gate_proj.device for tensor in tensors[1:]):
            raise ValueError("Torch expert weights must be on one device")
        if any(tensor.dtype != self.gate_proj.dtype for tensor in tensors[1:]):
            raise ValueError("Torch expert weights must use one dtype")
        if any(not tensor.is_contiguous() for tensor in tensors):
            raise ValueError("Torch expert weights must be contiguous")
        if any(tensor.requires_grad for tensor in tensors):
            raise ValueError("Torch expert weights must not require gradients")

        intermediate_dim, hidden_dim = self.gate_proj.shape
        if min(intermediate_dim, hidden_dim) <= 0:
            raise ValueError("Torch expert weight dimensions must be positive")
        if self.up_proj.shape != self.gate_proj.shape:
            raise ValueError("gate and up projection shapes must match")
        if self.down_proj.shape != (hidden_dim, intermediate_dim):
            raise ValueError("down projection shape must reverse gate and up dimensions")

    @property
    def hidden_dim(self) -> int:
        """Return the FFN input and output width."""

        return self.gate_proj.shape[1]

    @property
    def intermediate_dim(self) -> int:
        """Return the gated FFN intermediate width."""

        return self.gate_proj.shape[0]

    @property
    def dtype(self) -> torch.dtype:
        """Return the already prepared computation dtype."""

        return self.gate_proj.dtype

    @property
    def device(self) -> torch.device:
        """Return the already prepared computation device."""

        return self.gate_proj.device

    @property
    def storage_bytes(self) -> int:
        """Return logical Tensor storage bytes for device-capacity accounting."""

        return sum(tensor.numel() * tensor.element_size() for tensor in self.tensors)

    @property
    def tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the projection tensors in gate, up, and down order."""

        return self.gate_proj, self.up_proj, self.down_proj


@dataclass(frozen=True, slots=True)
class TorchGPTQCpuWeight:
    """Packed GPTQ tensors retained on CPU until a weight becomes ready."""

    gate: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    up: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    down: tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def dequantize_gptq(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    *,
    in_features: int,
    out_features: int,
    group_size: int,
) -> torch.Tensor:
    """Decode one AutoGPTQ 4-bit weight matrix into ``[out, in]`` FP16.

    GPTQ stores eight 4-bit values in each signed 32-bit word. This reference
    decoder intentionally favors clear validation over speed; a packed CUDA
    kernel can replace it without changing the Worker contract.
    """

    if group_size <= 0 or min(in_features, out_features) <= 0:
        raise ValueError("GPTQ dimensions and group_size must be positive")
    if in_features % 8 or out_features % 8 or in_features % group_size:
        raise ValueError("GPTQ input width must be divisible by 8 and group_size")
    if tuple(qweight.shape) != (in_features // 8, out_features):
        raise ValueError("unexpected GPTQ qweight shape")
    groups = in_features // group_size
    if tuple(qzeros.shape) != (groups, out_features // 8):
        raise ValueError("unexpected GPTQ qzeros shape")
    if tuple(scales.shape) != (groups, out_features):
        raise ValueError("unexpected GPTQ scales shape")
    shifts = torch.arange(8, device=qweight.device, dtype=torch.int32) * 4
    values = (qweight.to(torch.int32).unsqueeze(-1) >> shifts) & 0xF
    values = values.permute(0, 2, 1).reshape(in_features, out_features)
    zero_values = (qzeros.to(torch.int32).unsqueeze(-1) >> shifts) & 0xF
    zero_values = zero_values.expand(-1, -1, 8).reshape(groups, out_features) + 1
    group_ids = torch.arange(in_features, device=qweight.device) // group_size
    return ((values - zero_values[group_ids]).to(scales.dtype) * scales[group_ids]).T.contiguous()
