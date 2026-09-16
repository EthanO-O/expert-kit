"""Qwen2-MoE routed block for the pinned Transformers implementation."""

from __future__ import annotations

import torch
from torch import nn
from transformers.models.qwen2_moe import modeling_qwen2_moe

from expertkit_torch.client import RoutedMoEClient
from expertkit_torch.models._common import RoutedLayerIds


def create_routed_moe_class(client: RoutedMoEClient, layer_ids: RoutedLayerIds) -> type[nn.Module]:
    """Create a Qwen2-MoE block that keeps routing local and experts remote."""

    class RoutedQwen2MoeSparseMoeBlock(nn.Module):
        def __init__(self, config) -> None:
            super().__init__()
            self.layer_id = layer_ids.take()
            self.gate = modeling_qwen2_moe.Qwen2MoeTopKRouter(config)
            self.shared_expert = modeling_qwen2_moe.Qwen2MoeMLP(
                config, intermediate_size=config.shared_expert_intermediate_size
            )
            self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            flattened = hidden_states.reshape(-1, hidden_dim)
            shared = self.shared_expert(flattened)
            _, weights, expert_ids = self.gate(flattened)
            routed = client.forward_layer(
                layer_id=self.layer_id,
                hidden_states=flattened,
                expert_ids=expert_ids,
                routing_weights=weights,
            )
            shared = torch.sigmoid(self.shared_expert_gate(flattened)) * shared
            return (routed + shared).reshape(batch_size, sequence_length, hidden_dim)

    return RoutedQwen2MoeSparseMoeBlock
