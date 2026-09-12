from __future__ import annotations

import torch
from safetensors.torch import save

from expertkit_worker.backends.torch import TorchGPTQWeightAdapter
from expertkit_worker.weights import parse_safetensors


def _pack(values: torch.Tensor) -> torch.Tensor:
    values = values.to(torch.int32)
    if values.ndim == 2:
        values = values.reshape(values.shape[0], -1, 8).permute(1, 0, 2)
    else:
        values = values.reshape(-1, 8)
    shifts = torch.arange(8, dtype=torch.int32) * 4
    return (values << shifts).sum(dim=-1)


def test_gptq_adapter_decodes_packed_expert() -> None:
    hidden, intermediate, group = 8, 8, 8
    expected = torch.ones(intermediate, hidden, dtype=torch.float16)

    def tensors(prefix: str) -> dict[str, torch.Tensor]:
        q = torch.full((hidden // 8, intermediate), 0x11111111, dtype=torch.int64)
        return {
            f"{prefix}.qweight": q,
            f"{prefix}.qzeros": torch.zeros(1, intermediate, dtype=torch.int64),
            f"{prefix}.scales": torch.ones(1, intermediate, dtype=torch.float16),
        }

    payload = save({**tensors("gate_proj"), **tensors("up_proj"), **tensors("down_proj")})
    ready = TorchGPTQWeightAdapter(
        hidden_dim=hidden, intermediate_dim=intermediate, group_size=group, device="cpu"
    ).make_ready_weight(
        TorchGPTQWeightAdapter(
            hidden_dim=hidden, intermediate_dim=intermediate, group_size=group, device="cpu"
        ).make_cpu_weight(parse_safetensors(payload)),
        layer_id=0,
        expert_id=0,
    )
    torch.testing.assert_close(ready.gate_proj, expected)
