"""Ascend-only ordered Qwen reduction with the ordinary BF16 output boundary.

Import this module only when preparing the optional Ascend graph implementation.
The caller warms both destination dtypes before capturing device operations.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _ordered_combine_kernel(
    down,
    expert_ids,
    weights,
    inverse,
    offset,
    count,
    output,
    H: tl.constexpr,
    START: tl.constexpr,
    END: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    route = tl.arange(0, 8)
    keys = tl.load(expert_ids + token * 8 + route) * 8 + route
    local_offset = tl.load(offset)
    local_count = tl.load(count)
    accumulator = tl.full((BLOCK,), 0.0, tl.float32)
    for _ in tl.static_range(8):
        key = tl.min(keys, 0)
        selected = key % 8
        expert = key // 8
        position = tl.load(inverse + token * 8 + selected) - local_offset
        valid = (expert >= START) & (expert < END) & (position >= 0) & (position < local_count)
        value = tl.load(down + position * H + columns, mask=valid & (columns < H), other=0).to(
            tl.float32
        )
        weight = tl.load(weights + token * 8 + selected)
        product = value * weight
        accumulator = tl.where(valid, accumulator + product, accumulator)
        keys = tl.where(keys == key, 2147483647, keys)
    rounded = accumulator.to(tl.bfloat16).to(tl.float32)
    tl.store(output + token * H + columns, rounded, mask=columns < H)


def ordered_combine(
    down_bf16: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights_fp32: torch.Tensor,
    inverse_rows: torch.Tensor,
    local_offset: torch.Tensor,
    local_count: torch.Tensor,
    prepared_output: torch.Tensor,
    *,
    expert_start: int,
    expert_end: int,
) -> None:
    """Write the expert-ordered local sum into a caller-owned fixed destination.

    The caller supplies expert IDs in [0, 128), a complete assignment inverse,
    and valid local offset/count values. Validation reads only Tensor metadata.

    Args:
        down_bf16: Contiguous BF16 packed rows, shaped [capacity * 8, hidden].
        expert_ids: Contiguous INT32 [capacity, 8] original expert IDs.
        routing_weights_fp32: Contiguous FP32 weights aligned with expert_ids.
        inverse_rows: INT32 global assignment-to-packed-row permutation.
        local_offset: Device scalar start of the local packed rows.
        local_count: Device scalar number of actual local rows.
        prepared_output: Contiguous BF16/FP32 [capacity, hidden] destination.
            FP32 carries the same BF16-rounded sum. Padding is never read.
    """
    tensors = (
        down_bf16,
        expert_ids,
        routing_weights_fp32,
        inverse_rows,
        local_offset,
        local_count,
        prepared_output,
    )
    if any(t.device != down_bf16.device or not t.is_contiguous() for t in tensors):
        raise ValueError("ordered combine requires contiguous tensors on one device")
    if down_bf16.device.type != "npu":
        raise ValueError("ordered combine requires an Ascend NPU")
    if expert_ids.ndim != 2 or expert_ids.shape[1] != 8 or expert_ids.dtype != torch.int32:
        raise ValueError("expert IDs must be INT32 [capacity, 8]")
    batch = expert_ids.shape[0]
    if down_bf16.ndim != 2 or down_bf16.shape[0] != batch * 8 or down_bf16.dtype != torch.bfloat16:
        raise ValueError("down rows must be BF16 [capacity * 8, hidden]")
    if (
        routing_weights_fp32.shape != expert_ids.shape
        or routing_weights_fp32.dtype != torch.float32
    ):
        raise ValueError("routing weights must be FP32 and aligned with expert IDs")
    if inverse_rows.shape != (batch * 8,) or inverse_rows.dtype != torch.int32:
        raise ValueError("inverse rows must be a flat INT32 permutation")
    if any(t.numel() != 1 or t.dtype != torch.int64 for t in (local_offset, local_count)):
        raise ValueError("local offset and count must be INT64 device scalars")
    if prepared_output.shape != (batch, down_bf16.shape[1]) or prepared_output.dtype not in (
        torch.bfloat16,
        torch.float32,
    ):
        raise ValueError("prepared output must be BF16/FP32 [capacity, hidden]")
    if not batch or not down_bf16.shape[1] or not 0 <= expert_start < expert_end <= 128:
        raise ValueError("ordered combine requires nonempty dimensions and a valid expert range")
    if any(
        prepared_output.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
        for t in tensors[:-1]
    ):
        raise ValueError("prepared output must not alias any input storage")
    hidden = down_bf16.shape[1]
    _ordered_combine_kernel[(expert_ids.shape[0], triton.cdiv(hidden, 256))](
        down_bf16,
        expert_ids,
        routing_weights_fp32,
        inverse_rows,
        local_offset,
        local_count,
        prepared_output,
        hidden,
        expert_start,
        expert_end,
        256,
        enable_fp_fusion=False,
    )
