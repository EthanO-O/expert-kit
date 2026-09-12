"""Qwen2-MoE routed block for the pinned Transformers implementation."""

from __future__ import annotations

import torch
from torch import nn

from expertkit_torch.client import RoutedMoEClient
from expertkit_torch.models._common import RoutedLayerIds


def create_routed_moe_class(client: RoutedMoEClient, layer_ids: RoutedLayerIds) -> type[nn.Module]:
    """Create a Qwen2-MoE block that keeps routing local and experts remote."""

    class RoutedQwen2MoeSparseMoeBlock(nn.Module):
        def __init__(self, config) -> None:
            super().__init__()
            self.layer_id = layer_ids.take()
            self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
            self.top_k = config.num_experts_per_tok
            self.norm_topk_prob = config.norm_topk_prob

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            flattened = hidden_states.reshape(-1, hidden_dim)
            router_logits = self.gate(flattened)
            weights = torch.softmax(router_logits, dim=1, dtype=torch.float32)
            weights, expert_ids = torch.topk(weights, self.top_k, dim=-1)
            if self.norm_topk_prob:
                weights = weights / weights.sum(dim=-1, keepdim=True)
            routed = client.forward_layer(
                layer_id=self.layer_id,
                hidden_states=flattened,
                expert_ids=expert_ids,
                routing_weights=weights.to(hidden_states.dtype),
            )
            return routed.reshape(batch_size, sequence_length, hidden_dim)

    return RoutedQwen2MoeSparseMoeBlock
