"""Reference decoder for DeepSeek-V4 packed FP4 routed experts."""

from __future__ import annotations

import torch

from expertkit_worker.backends.torch.adapter import TorchWeightAdapter
from expertkit_worker.backends.torch.weights import TorchExpertWeights
from expertkit_worker.weights.adapter import WeightAdapter
from expertkit_worker.weights.format import SafeTensorData, SafeTensorDType


def dequantize_fp4(packed: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Decode low-nibble-first E2M1 weights and E8M0 scale bytes on CPU.

    Packed uint8 is [output, input/2]; uint8 scales are [output, input/32].
    Return a new contiguous [output, input] tensor in the requested floating dtype.
    """
    outputs, columns = packed.shape
    nibble = torch.stack((packed & 15, packed >> 4), dim=-1).long()
    magnitude = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    values = magnitude[nibble & 7] * torch.where((nibble & 8) != 0, -1.0, 1.0)
    factors = torch.ldexp(torch.ones_like(scales, dtype=torch.float32), scales.int() - 127)
    result = (
        (values.reshape(outputs, -1, 32) * factors.unsqueeze(-1))
        .reshape(outputs, columns * 2)
        .to(dtype)
    )
    if not torch.all(torch.isfinite(result)):
        raise ValueError("FP4 weights overflow the selected compute dtype")
    return result.contiguous()


def fp8_activation_reference(x: torch.Tensor) -> torch.Tensor:
    """Emulate V4 dynamic E4M3 activation rounding with power-of-two block scales.

    Input and output have the same floating dtype, shape, and device. Each row
    uses one scale per 128 columns. No native FP8 matrix instruction is required.
    """
    grouped = x.float().reshape(x.shape[0], -1, 128)
    maximum = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(maximum / 448.0)))
    rounded = (grouped / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    return (rounded * scale).reshape_as(x).to(x.dtype)


class TorchFP4WeightAdapter(
    WeightAdapter[tuple[tuple[torch.Tensor, torch.Tensor], ...], TorchExpertWeights]
):
    """Retain packed CPU views and expand FP4 weights only during placement."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        intermediate_dim: int,
        device: torch.device | str,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if any(
            isinstance(v, bool) or not isinstance(v, int) or v <= 0 or v % 128
            for v in (hidden_dim, intermediate_dim)
        ):
            raise ValueError("V4 FP4 dimensions must be positive multiples of 128")
        self._hidden_dim, self._intermediate_dim = hidden_dim, intermediate_dim
        self._compute_dtype = compute_dtype
        self._float_adapter = TorchWeightAdapter(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            source_dtype=compute_dtype,
            compute_dtype=compute_dtype,
            device=device,
        )

    @property
    def backend_name(self) -> str:
        """Identify the floating ready-cache compatibility path."""
        return "torch-mxfp4-dequantize"

    def make_cpu_weight(
        self, source: SafeTensorData
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """Validate packed E2M1 bytes and finite E8M0 scales without expanding weights."""
        matrices, consumed = [], set()
        for role, inputs, outputs in (
            ("w1", self._hidden_dim, self._intermediate_dim),
            ("w3", self._hidden_dim, self._intermediate_dim),
            ("w2", self._intermediate_dim, self._hidden_dim),
        ):
            weight = source.find_unique_suffix((f"{role}.weight",))
            scale = source.tensors.get(weight.name.removesuffix("weight") + "scale")
            if weight.dtype is not SafeTensorDType.INT8 or weight.shape != (outputs, inputs // 2):
                raise ValueError("unexpected FP4 packed weight dtype or shape")
            if (
                scale is None
                or scale.dtype is not SafeTensorDType.F8_E8M0
                or scale.shape != (outputs, inputs // 32)
            ):
                raise ValueError("unexpected FP4 scale dtype or shape")
            packed = torch.frombuffer(weight.data, dtype=torch.uint8).reshape(weight.shape)
            scales = torch.frombuffer(scale.data, dtype=torch.uint8).reshape(scale.shape)
            if torch.any(scales == 255):
                raise ValueError("FP4 E8M0 scales must be finite")
            matrices.append((packed, scales))
            consumed.update((weight.name, scale.name))
        if consumed != set(source.tensors):
            raise ValueError("unsupported or ambiguous Tensor in FP4 expert bundle")
        return tuple(matrices)

    def make_ready_weight(
        self,
        cpu_weight: tuple[tuple[torch.Tensor, torch.Tensor], ...],
        *,
        layer_id: int,
        expert_id: int,
    ) -> TorchExpertWeights:
        """Decode into the compute dtype and synchronize device placement."""
        decoded = TorchExpertWeights(*(dequantize_fp4(*m, self._compute_dtype) for m in cpu_weight))
        return self._float_adapter.make_ready_weight(
            decoded, layer_id=layer_id, expert_id=expert_id
        )

    def cpu_extra_bytes(self) -> int:
        """Return zero because CPU weights are views of retained source bytes."""
        return 0

    def source_tensor_bytes(self) -> int:
        """Count packed nibbles and one-byte scales for each group of 32."""
        elements = 3 * self._hidden_dim * self._intermediate_dim
        return elements // 2 + elements // 32

    def ready_weight_bytes(self) -> int:
        """Count the expanded floating device matrices."""
        return self._float_adapter.ready_weight_bytes()

    def conversion_temporary_bytes(self) -> int:
        """Return device conversion scratch, excluding transient CPU decoding."""
        return self._float_adapter.conversion_temporary_bytes()
