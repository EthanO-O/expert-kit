"""Exercise exact ordered reduction and graph replay on a real Ascend device."""

import importlib.util

import pytest
import torch

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch_npu") is None, reason="Ascend runtime required"
)


def reference(down, ids, weights, inverse, offset, count, start, end):
    """Evaluate each ordered FP32 multiply/add separately on the CPU."""
    d = down.cpu().float()
    e = ids.cpu()
    w = weights.cpu()
    inv = inverse.cpu()
    result = torch.zeros((ids.shape[0], down.shape[1]), dtype=torch.float32)
    for token in range(ids.shape[0]):
        for column in sorted(range(8), key=lambda j: (int(e[token, j]), j)):
            position = int(inv[token * 8 + column]) - offset
            if start <= int(e[token, column]) < end and 0 <= position < count:
                product = d[position] * w[token, column]
                result[token] = result[token] + product
    return result.bfloat16()


def route(ids):
    """Get real global inverse positions and counts from the frozen operator."""
    import torch_npu

    x = torch.zeros((ids.shape[0], 32), dtype=torch.bfloat16, device="npu")
    return torch_npu.npu_moe_init_routing_v2(
        x,
        ids,
        active_num=ids.numel(),
        expert_num=128,
        drop_pad_mode=0,
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        active_expert_range=[0, 128],
        quant_mode=-1,
        row_idx_type=0,
    )


def operands(batch, pattern, start):
    """Include local zero/full rows, repeated IDs and poisoned unused capacity."""
    ids = (torch.arange(batch * 8).reshape(batch, 8) * 17 % 128).int()
    if pattern == "zero":
        ids = ids % 64 + (64 if start == 0 else 0)
    elif pattern == "full":
        ids = ids % 64 + start
    if pattern == "mixed":
        ids[:, 1] = ids[:, 0]
    ids = ids.to("npu")
    _, inverse, counts, _ = route(ids)
    offset = counts[:start].sum()
    count = counts[start : start + 64].sum()
    generator = torch.Generator().manual_seed(61 + batch)
    down = torch.randn((batch * 8, 2048), generator=generator).bfloat16().to("npu")
    weights = torch.randn((batch, 8), generator=generator).to("npu")
    weights[0] = torch.tensor([1.0, -1.0, 1.00001, 0.25, -0.5, 2.0, 1e-5, -2.0], device="npu")
    down[0, :8] = torch.tensor(
        [16777216.0, -16777216.0, 1.0, 257.0, -1.0, 1.0078125, 128.0, 1e-8],
        dtype=torch.bfloat16,
        device="npu",
    )
    rows = torch.arange(batch * 8, device="npu")
    down = torch.where((rows < count)[:, None], down, float("nan"))
    return down, ids, weights, inverse, offset, count


@pytest.mark.parametrize("batch", [1, 32, 64])
@pytest.mark.parametrize("pattern", ["mixed", "zero", "full"])
@pytest.mark.parametrize("start", [0, 64])
def test_ordered_combine_bit_patterns(batch, pattern, start):
    """Both destination dtypes retain exact BF16 bits and skip poisoned rows."""
    import torch_npu  # noqa: F401

    from expertkit_worker.backends.torch.ascend_moe_ops import ordered_combine

    torch.npu.set_device(0)
    down, ids, weights, inverse, offset, count = operands(batch, pattern, start)
    expected = reference(down, ids, weights, inverse, int(offset), int(count), start, start + 64)
    for dtype in [torch.bfloat16, torch.float32]:
        output = torch.full((batch, 2048), float("nan"), dtype=dtype, device="npu")
        ordered_combine(
            down,
            ids,
            weights,
            inverse,
            offset,
            count,
            output,
            expert_start=start,
            expert_end=start + 64,
        )
        torch.npu.synchronize()
        assert torch.equal(output.cpu(), expected.to(dtype))
        assert torch.equal(output.cpu().bfloat16().view(torch.int16), expected.view(torch.int16))


def test_ordered_combine_replay_changes_device_counts():
    """A single graph reuses addresses across mixed, no-local and all-local routing."""
    import torch_npu  # noqa: F401

    from expertkit_worker.backends.torch.ascend_moe_ops import ordered_combine

    torch.npu.set_device(0)
    values = operands(32, "mixed", 64)
    output = torch.empty((32, 2048), dtype=torch.float32, device="npu")
    for _ in range(3):
        ordered_combine(*values, output, expert_start=64, expert_end=128)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        ordered_combine(*values, output, expert_start=64, expert_end=128)
    for pattern in ["zero", "full", "mixed"]:
        replacement = operands(32, pattern, 64)
        for target, source in zip(values, replacement, strict=True):
            target.copy_(source)
        graph.replay()
        torch.npu.synchronize()
        expected = reference(*values[:4], int(values[4]), int(values[5]), 64, 128)
        assert torch.equal(output.cpu(), expected.float())


def test_ordered_combine_real_operands():
    """Reuse actual layer5 down outputs and original FP32 routing weights."""
    import os
    from pathlib import Path

    import torch_npu  # noqa: F401

    from expertkit_worker.backends.torch.ascend_moe_ops import ordered_combine

    path = Path(os.environ.get("EK_TEST_MOE_OPERANDS", "/nonexistent"))
    if not path.is_file():
        pytest.skip("Set EK_TEST_MOE_OPERANDS to the frozen real layer5 input subset.")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    ids = saved["route_ids"].to("npu")
    weights = saved["route_weights"].to("npu")
    for start in [0, 64]:
        down = saved[f"{start}/graph/project/2/output"].to("npu")
        forward = saved[f"{start}/graph/original_rows"].to("npu").long()
        counts = saved[f"{start}/graph/counts"].to("npu")
        inverse = torch.empty(8, dtype=torch.int32, device="npu")
        inverse.scatter_(0, forward, torch.arange(8, dtype=torch.int32, device="npu"))
        offset, count = counts[:start].sum(), counts[start : start + 64].sum()
        rows = torch.arange(8, device="npu")
        down = torch.where((rows < count)[:, None], down, float("nan"))
        expected = reference(
            down, ids, weights, inverse, int(offset), int(count), start, start + 64
        )
        for dtype in [torch.bfloat16, torch.float32]:
            output = torch.full((1, 2048), float("nan"), dtype=dtype, device="npu")
            ordered_combine(
                down,
                ids,
                weights,
                inverse,
                offset,
                count,
                output,
                expert_start=start,
                expert_end=start + 64,
            )
            torch.npu.synchronize()
            assert torch.equal(
                output.cpu().bfloat16().view(torch.int16), expected.view(torch.int16)
            )
            assert torch.equal(output.cpu(), expected.to(dtype))


@pytest.mark.parametrize(
    "invalid", ["ids", "weights", "inverse", "offset", "output", "layout", "alias"]
)
def test_ordered_combine_rejects_invalid_metadata(invalid):
    """Reject incompatible storage before any kernel can read the tensors."""
    import torch_npu  # noqa: F401

    from expertkit_worker.backends.torch.ascend_moe_ops import ordered_combine

    values = list(operands(1, "mixed", 0))
    output = torch.empty((1, 2048), dtype=torch.bfloat16, device="npu")
    if invalid == "ids":
        values[1] = values[1][:, :7].contiguous()
    elif invalid == "weights":
        values[2] = values[2].bfloat16()
    elif invalid == "inverse":
        values[3] = values[3].long()
    elif invalid == "offset":
        values[4] = torch.zeros(2, dtype=torch.int64, device="npu")
    elif invalid == "output":
        output = output[:, :1024]
    elif invalid == "layout":
        values[0] = values[0].t().contiguous().t()
    elif invalid == "alias":
        output = values[0][:1]
    with pytest.raises(ValueError):
        ordered_combine(*values, output, expert_start=0, expert_end=64)
