"""Benchmark one real Qwen1.5-MoE expert before and after temporary W4A16 packing.

The quantized representation is generated locally from the BF16 checkpoint. It is
useful for measuring the Worker adapter and memory trade-off, but it is not an
official GPTQ checkpoint and its accuracy must not be presented as model quality.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
from safetensors.torch import save
from safetensors import safe_open

from expertkit_worker.backends.torch import TorchGPTQWeightAdapter
from expertkit_worker.weights import parse_safetensors


def _pack_symmetric(weight: torch.Tensor, group_size: int) -> tuple[torch.Tensor, ...]:
    output, input_width = weight.shape
    grouped = weight.reshape(output, input_width // group_size, group_size)
    scales = grouped.abs().amax(-1).clamp_min(1e-8) / 7
    values = (
        (grouped / scales.unsqueeze(-1)).round().clamp(-8, 7).add(8).to(torch.int64)
    )
    values = values.reshape(output, input_width // 8, 8).permute(1, 0, 2)
    shifts = torch.arange(8, dtype=torch.int64) * 4
    qweight = (values << shifts).sum(-1).to(torch.int32)
    groups = input_width // group_size
    qzeros = torch.full((groups, output // 8), int("77777777", 16), dtype=torch.int32)
    return qweight, qzeros, scales.reshape(output, -1).T.contiguous().half()


def _ffn(hidden: torch.Tensor, weights: tuple[torch.Tensor, ...]) -> torch.Tensor:
    gate, up, down = weights
    return torch.nn.functional.linear(
        torch.nn.functional.silu(torch.nn.functional.linear(hidden, gate))
        * torch.nn.functional.linear(hidden, up),
        down,
    )


def _median_ms(
    function, hidden: torch.Tensor, weights: tuple[torch.Tensor, ...], repeats: int
) -> float:
    for _ in range(20):
        function(hidden, weights)
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function(hidden, weights)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="local Qwen1.5-MoE BF16 checkpoint directory")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    device = torch.device("cuda:0")
    shard = f"{args.model}/model-00001-of-00008.safetensors"
    with safe_open(shard, framework="pt", device="cpu") as source:
        weights = tuple(
            source.get_tensor(
                f"model.layers.{args.layer}.mlp.experts.{args.expert}.{role}_proj.weight"
            )
            for role in ("gate", "up", "down")
        )
    baseline = tuple(
        weight.to(device=device, dtype=torch.float16).contiguous() for weight in weights
    )
    encoded: dict[str, torch.Tensor] = {}
    for role, weight in zip(("gate", "up", "down"), weights, strict=True):
        qweight, qzeros, scales = _pack_symmetric(weight.float(), 128)
        encoded.update(
            {
                f"{role}_proj.qweight": qweight,
                f"{role}_proj.qzeros": qzeros,
                f"{role}_proj.scales": scales,
                f"{role}_proj.g_idx": torch.arange(weight.shape[1], dtype=torch.int32)
                // 128,
            }
        )
    adapter = TorchGPTQWeightAdapter(
        hidden_dim=2048, intermediate_dim=1408, group_size=128, device=device
    )
    source_blob = parse_safetensors(save(encoded))
    cpu_weight = adapter.make_cpu_weight(source_blob)
    start = time.perf_counter()
    quantized = adapter.make_ready_weight(
        cpu_weight, layer_id=args.layer, expert_id=args.expert
    )
    torch.cuda.synchronize()
    dequant_ms = (time.perf_counter() - start) * 1000

    print(
        f"gpu={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}"
    )
    print(f"checkpoint={args.model} layer={args.layer} expert={args.expert}")
    print("representation,rows,median_ms,tokens_per_second,ready_bytes,source_bytes")
    for rows in (1, 8, 32, 128, 512):
        hidden = torch.randn(rows, 2048, device=device, dtype=torch.float16)
        baseline_ms = _median_ms(_ffn, hidden, baseline, args.repeats)
        quantized_ms = _median_ms(_ffn, hidden, quantized.tensors, args.repeats)
        for name, latency, ready_bytes, source_bytes in (
            (
                "bf16-checkpoint-cast-fp16",
                baseline_ms,
                sum(t.numel() * t.element_size() for t in baseline),
                sum(t.numel() * t.element_size() for t in weights),
            ),
            (
                "temporary-w4a16",
                quantized_ms,
                quantized.storage_bytes,
                sum(t.numel() * t.element_size() for t in encoded.values()),
            ),
        ):
            print(
                f"{name},{rows},{latency:.3f},{rows * 1000 / latency:.1f},{ready_bytes},{source_bytes}"
            )
    print(
        f"dequantize_ms={dequant_ms:.3f} quantized_ready_bytes={quantized.storage_bytes}"
    )


if __name__ == "__main__":
    main()
