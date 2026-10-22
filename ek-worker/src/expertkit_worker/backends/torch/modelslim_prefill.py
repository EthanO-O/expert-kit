"""Grouped Qwen ModelSlim projections on the submitting Ascend stream.

The receiver validates distinct expert metadata before admission. Ready weights
are borrowed from the caller's lease; no weight cache or secondary stream exists.
All temporary allocations use the current Torch stream, whose allocator delays
reuse until their submitted consumers finish. The caller retains the lease and
uses its normal completion/failure drain for all native and Triton launches.
"""

from __future__ import annotations

from importlib import import_module
from time import perf_counter

import torch
import torch.nn.functional as functional
import triton
import triton.language as tl
from expertkit_transport.quarantine import retain_until_process_exit

from expertkit_worker.backends.base import BackendBatch
from expertkit_worker.backends.torch.ascend_moe_ops import ordered_combine
from expertkit_worker.backends.torch.modelslim_w8a8 import TorchModelSlimW8A8Weights
from expertkit_worker.device.runtime import WorkerDeviceRuntime

# This additional allocator/workspace allowance is checked by the large-shape
# hardware regression; explicit Tensor storage is accounted for separately.
_WORKSPACE_BYTES = 128 * 1024 * 1024


@triton.jit
def _copy_prefix(source, valid_count, destination, H: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    offsets = row * H + columns
    valid = row < tl.load(valid_count)
    values = tl.load(source + offsets, mask=valid & (columns < H), other=0)
    tl.store(destination + offsets, values, mask=columns < H)


@triton.jit
def _inverse_prefix(indices, valid_count, inverse, BLOCK: tl.constexpr):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = rows < tl.load(valid_count)
    assignments = tl.load(indices + rows, mask=valid, other=0)
    tl.store(inverse + assignments, rows, mask=valid)


@triton.jit
def _dequantize(
    dot,
    token_scales,
    channel_scales,
    counts,
    valid_count,
    output,
    D: tl.constexpr,
    expert_count,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    experts = tl.arange(0, 128)
    count = tl.load(expert_count)
    sizes = tl.load(counts + experts, mask=experts < count, other=0)
    ends = tl.cumsum(sizes, 0)
    valid = row < tl.load(valid_count)
    expert = tl.sum(((row >= ends) & (experts < count)).to(tl.int32), 0)
    expert = tl.where(valid, expert, 0)
    # Mask the native GMM tail before converting an undefined INT32 product.
    product = tl.load(dot + row * D + columns, mask=valid & (columns < D), other=0)
    channel = tl.load(channel_scales + expert * D + columns, mask=columns < D, other=0)
    token = tl.load(token_scales + row)
    scale = channel * token
    value = scale * product.to(tl.float32)
    tl.store(output + row * D + columns, value, mask=columns < D)


def warmup(runtime: WorkerDeviceRuntime) -> dict[str, float | int]:
    """Compile the fixed BF16 helper signatures before Worker registration.

    All buffers belong to the startup current stream and are drained before
    release. A failed drain retains only these buffers until fatal process exit.
    No ready weights or native grouped projections are needed for compilation.
    """
    started = perf_counter()
    buffers: list[torch.Tensor] = []

    def keep(tensor: torch.Tensor) -> torch.Tensor:
        buffers.append(tensor)
        return tensor

    with runtime.device_context():
        try:
            raw = keep(torch.zeros((16, 2048), device=runtime.device, dtype=torch.bfloat16))
            packed = keep(torch.empty_like(raw))
            valid = keep(torch.tensor([16], device=runtime.device, dtype=torch.int64))
            indices = keep(torch.arange(16, device=runtime.device, dtype=torch.int32))
            inverse = keep(torch.full_like(indices, -1))
            counts = keep(torch.tensor([8, 8], device=runtime.device, dtype=torch.int64))
            expert_count = keep(torch.tensor([2], device=runtime.device, dtype=torch.int32))
            tokens = keep(torch.ones(16, device=runtime.device, dtype=torch.float32))
            _copy_prefix[(16, 8)](raw, valid, packed, 2048, 256)
            _inverse_prefix[(1,)](indices, valid, inverse, 256)
            for width in (768, 2048):
                dot = keep(torch.zeros((16, width), device=runtime.device, dtype=torch.int32))
                channels = keep(torch.ones((2, width), device=runtime.device, dtype=torch.float32))
                output = keep(torch.empty((16, width), device=runtime.device, dtype=torch.bfloat16))
                _dequantize[(16, triton.cdiv(width, 256))](
                    dot,
                    tokens,
                    channels,
                    counts,
                    valid,
                    output,
                    width,
                    expert_count,
                    256,
                    enable_fp_fusion=False,
                )
            ids = keep(torch.zeros((2, 8), device=runtime.device, dtype=torch.int32))
            routing = keep(torch.ones((2, 8), device=runtime.device, dtype=torch.float32))
            offset = keep(torch.zeros(1, device=runtime.device, dtype=torch.int64))
            combined = keep(torch.empty((2, 2048), device=runtime.device, dtype=torch.bfloat16))
            ordered_combine(
                output,
                ids,
                routing,
                inverse,
                offset,
                valid,
                combined,
                expert_start=0,
                expert_end=128,
            )
            runtime.capture_current_work().wait_host()
        except BaseException as error:
            try:
                runtime.capture_current_work().wait_host()
            except BaseException as drain_error:
                retain_until_process_exit(buffers)
                error.add_note(f"Grouped Prefill warmup drain failed: {drain_error!r}")
            raise
    return {
        "duration_seconds": perf_counter() - started,
        "buffer_bytes": sum(t.numel() * t.element_size() for t in buffers),
    }


def temporary_bytes(max_tokens: int, hidden: int, intermediate: int, top_k: int) -> int:
    """Bound explicit live Tensor storage plus a separately checked workspace reserve.

    The bound deliberately retains all three projection inputs, INT32 products,
    BF16 outputs and FP32 conversions together, although dequantization is fused
    and Python releases each product before the next projection. Routing includes
    the raw and initialized packs, inverse/indices, masks and native scales.
    """
    assignments = max_tokens * top_k
    return (
        assignments * (hidden * 20 + intermediate * 28 + 64)
        + 128 * (hidden + 2 * intermediate) * 4
        + 10240 * 8
        + _WORKSPACE_BYTES
    )


def projection_views(weights: tuple[TorchModelSlimW8A8Weights, ...], projection: int):
    """Borrow original matrices and expose the native singleton group dimension."""
    matrices = [weight.matrices[projection] for weight in weights]
    if len(matrices) == 1:
        matrices = [matrices[0].unsqueeze(0)]
    return matrices


def compute(
    batch: BackendBatch,
    prepared_output: torch.Tensor,
    weights: tuple[TorchModelSlimW8A8Weights, ...],
) -> None:
    """Write a validated Qwen batch with three grouped BF16-boundary projections.

    Distinct expert IDs are the receiver-validated, nonempty sorted lease order.
    Therefore at least one assignment exists; an empty lease is handled by the
    Backend before this function. Counts remain on device and follow this exact
    order, including sparse IDs. Inputs and caller-owned output are not replaced.
    """
    native = import_module("torch_npu")
    expert_ids = batch.distinct_expert_ids
    end = expert_ids[-1] + 1
    capacity = batch.token_count * batch.top_k
    supplied = batch.expert_ids.clone()
    supplied.masked_fill_((supplied < 0) | (supplied >= end), end)
    raw, indices, counts, _ = native.npu_moe_init_routing_v2(
        batch.hidden_states,
        supplied,
        active_num=capacity,
        expert_num=end + 1,
        drop_pad_mode=0,
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        active_expert_range=[0, end],
        quant_mode=-1,
        row_idx_type=1,
    )
    selected = counts.index_select(
        0, torch.tensor(expert_ids, device=counts.device, dtype=torch.int64)
    )
    valid = selected.sum().reshape(1)
    packed = torch.empty_like(raw)
    _copy_prefix[(raw.shape[0], triton.cdiv(batch.hidden_dim, 256))](
        raw, valid, packed, batch.hidden_dim, 256
    )
    inverse = torch.full((capacity,), -1, device=indices.device, dtype=torch.int32)
    _inverse_prefix[(triton.cdiv(capacity, 256),)](indices, valid, inverse, 256)

    expert_count = torch.tensor([len(weights)], device=selected.device, dtype=torch.int32)

    def project(inputs: torch.Tensor, projection: int) -> torch.Tensor:
        quantized, token_scales = native.npu_dynamic_quant(inputs, dst_type=torch.int8)
        dot = native.npu_grouped_matmul(
            [quantized],
            projection_views(weights, projection),
            split_item=3,
            group_type=0,
            group_list=selected,
            group_list_type=0 if len(weights) == 1 else 1,
            output_dtype=torch.int32,
        )[0]
        channels = torch.stack([weight.scales[projection] for weight in weights])
        output = torch.empty(dot.shape, dtype=torch.bfloat16, device=dot.device)
        _dequantize[(dot.shape[0], triton.cdiv(dot.shape[1], 256))](
            dot,
            token_scales,
            channels,
            selected,
            valid,
            output,
            dot.shape[1],
            expert_count,
            256,
            enable_fp_fusion=False,
        )
        return output

    gate = project(packed, 0)
    up = project(packed, 1)
    activation = functional.silu(gate) * up
    del gate, up
    down = project(activation, 2)
    ordered_combine(
        down,
        supplied,
        batch.routing_weights,
        inverse,
        torch.zeros((1,), device=valid.device, dtype=torch.int64),
        valid,
        prepared_output,
        expert_start=0,
        expert_end=128,
    )
