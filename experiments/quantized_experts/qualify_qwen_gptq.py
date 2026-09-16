"""Qualify the GPTQ decoder with one locally available Qwen BF16 expert.

This creates a temporary symmetric W4A16 representation for plumbing and kernel
checks. It is not an official quantized checkpoint or an accuracy benchmark.
"""

from __future__ import annotations

import argparse

import torch
from safetensors import safe_open
from safetensors.torch import save

from expertkit_worker.backends.torch import TorchGPTQWeightAdapter
from expertkit_worker.weights import parse_safetensors


def _pack_symmetric(weight: torch.Tensor, group_size: int) -> tuple[torch.Tensor, ...]:
    output, input_width = weight.shape
    scales = (
        weight.reshape(output, input_width // group_size, group_size).abs().amax(-1)
    )
    scales = scales.clamp_min(1e-8) / 7
    values = weight.reshape(output, -1, group_size) / scales.unsqueeze(-1)
    values = values.round().clamp(-8, 7).add(8).to(torch.int64)
    values = values.reshape(output, input_width // 8, 8).permute(1, 0, 2)
    shifts = torch.arange(8, dtype=torch.int64) * 4
    qweight = (values << shifts).sum(-1).to(torch.int32)
    qzeros = torch.full(
        (input_width // group_size, output // 8),
        int("77777777", 16),
        dtype=torch.int32,
    )
    return qweight, qzeros, scales.reshape(output, -1).T.contiguous().half()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="local Qwen1.5-MoE-BF16 checkpoint directory")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    args = parser.parse_args()
    shard = f"{args.model}/model-00001-of-00008.safetensors"
    with safe_open(shard, framework="pt", device="cpu") as source:
        weights = {
            role: source.get_tensor(
                f"model.layers.{args.layer}.mlp.experts.{args.expert}.{role}_proj.weight"
            ).float()
            for role in ("gate", "up", "down")
        }
    encoded: dict[str, torch.Tensor] = {}
    for role, weight in weights.items():
        qweight, qzeros, scales = _pack_symmetric(weight, 128)
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
        hidden_dim=2048, intermediate_dim=1408, group_size=128, device="cuda:0"
    )
    ready = adapter.make_ready_weight(
        adapter.make_cpu_weight(parse_safetensors(save(encoded))),
        layer_id=args.layer,
        expert_id=args.expert,
    )
    hidden = torch.randn(3, 2048, device="cuda:0", dtype=torch.float16)
    reference = torch.nn.functional.linear(
        torch.nn.functional.silu(
            torch.nn.functional.linear(hidden, weights["gate"].cuda().half())
        )
        * torch.nn.functional.linear(hidden, weights["up"].cuda().half()),
        weights["down"].cuda().half(),
    )
    actual = torch.nn.functional.linear(
        torch.nn.functional.silu(torch.nn.functional.linear(hidden, ready.gate_proj))
        * torch.nn.functional.linear(hidden, ready.up_proj),
        ready.down_proj,
    )
    error = (actual - reference).abs()
    print(
        f"gpu={torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)} "
        f"max_abs={error.max().item():.5f} ready_bytes={ready.storage_bytes}"
    )


if __name__ == "__main__":
    main()
