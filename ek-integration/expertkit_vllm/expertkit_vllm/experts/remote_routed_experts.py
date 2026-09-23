"""No-local-weight routed-expert placeholder for vLLM's MoE factory."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
import torch.nn as nn
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.expert_map_manager import (
    ExpertMapManager,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.utils import set_weight_attrs

type RoutingFunction = Callable[..., tuple[torch.Tensor, torch.Tensor]]


class RemoteRoutedExperts(nn.Module):
    """Carry routing metadata while making routed checkpoint weights absent."""

    def __init__(
        self,
        layer_name: str,
        params_dtype: torch.dtype,
        moe_config: FusedMoEConfig,
        quant_config: QuantizationConfig | None,
        expert_map_manager: ExpertMapManager,
        ckpt_gate_proj_name: str = "gate_proj",
        ckpt_down_proj_name: str = "down_proj",
        ckpt_up_proj_name: str = "up_proj",
        *,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: int | None = None,
        topk_group: int | None = None,
        custom_routing_function: RoutingFunction | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        swiglu_limit: float | None = None,
        swiglu_alpha: float | None = None,
        swiglu_beta: float | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
    ) -> None:
        super().__init__()
        self.layer_name = layer_name
        self.params_dtype = params_dtype
        self.moe_config = moe_config
        self.quant_config = quant_config
        self.expert_map_manager = expert_map_manager
        self.ckpt_gate_proj_name = ckpt_gate_proj_name
        self.ckpt_down_proj_name = ckpt_down_proj_name
        self.ckpt_up_proj_name = ckpt_up_proj_name
        self.top_k = moe_config.experts_per_token
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.custom_routing_function = custom_routing_function
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.e_score_correction_bias = e_score_correction_bias

        vllm_config = get_current_vllm_config()

        # ModelSlim quantizes local attention and checkpoint experts. Remote
        # experts are discarded and recomputed by EK, so this quantization
        # configuration is compatible with the remote placeholder.
        modelslim_quantization = (
            quant_config is not None and quant_config.__class__.__name__ == "AscendModelSlimConfig"
        )

        unsupported = {
            "quantization": quant_config is not None and not modelslim_quantization,
            "prefill_context_parallel": moe_config.pcp_size != 1,
            "sequence_parallel": moe_config.is_sequence_parallel,
            # The wrapper forces the injected MoE's parallel sizes to one so
            # each vLLM DP rank delegates independently to EK. Read native
            # TP/EP settings from the global config because `moe_config` no
            # longer preserves them.
            "tensor_parallel": vllm_config.parallel_config.tensor_parallel_size != 1,
            "expert_parallel": vllm_config.parallel_config.enable_expert_parallel,
            "eplb": vllm_config.parallel_config.enable_eplb,
            "expert_bias": moe_config.has_bias,
            "fused_shared_experts": expert_map_manager.num_fused_shared_experts != 0,
            "custom_swiglu": any(
                value is not None for value in (swiglu_limit, swiglu_alpha, swiglu_beta)
            ),
            "router_weight_on_input": apply_router_weight_on_input,
            "activation": moe_config.activation is not MoEActivation.SILU,
        }
        enabled = sorted(name for name, value in unsupported.items() if value)
        if enabled:
            raise ValueError(f"Expert Kit remote MoE does not support: {', '.join(enabled)}")

        self._load_weight_sink(modelslim_quantization=modelslim_quantization)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise AssertionError("RemoteRoutedExperts must be executed through RemoteMoERunner")

    def _load_weight_sink(self, *, modelslim_quantization: bool) -> None:
        suffixes = (
            ("weight", "weight_scale", "weight_offset") if modelslim_quantization else ("weight",)
        )
        for name in (
            f"{projection}_{suffix}" for projection in ("w13", "w2") for suffix in suffixes
        ):
            param = nn.Parameter(torch.empty(0), requires_grad=False)
            # Model loaders resolve parameters before invoking their loaders.
            # Empty parameters acknowledge remote weights without allocating storage.
            set_weight_attrs(param, {"weight_loader": _discard_weight})
            self.register_parameter(name, param)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Consume remote expert checkpoint entries without retaining local weights."""
        projections = {
            self.ckpt_gate_proj_name: "w13",
            self.ckpt_up_proj_name: "w13",
            self.ckpt_down_proj_name: "w2",
        }
        loaded = set()
        for name, _ in weights:
            parts = name.split(".")
            if len(parts) != 3 or not parts[0].isdigit() or parts[1] not in projections:
                raise ValueError(f"unsupported remote expert checkpoint parameter: {name}")
            destination = f"{projections[parts[1]]}_{parts[2]}"
            if destination not in self._parameters:
                raise ValueError(f"unsupported remote expert checkpoint parameter: {name}")
            loaded.add(destination)
        return loaded


def _discard_weight(
    param: nn.Parameter,
    loaded_weight: torch.Tensor,
    weight_name: str,
    *,
    shard_id: str,
    expert_id: int,
    return_success: bool = False,
) -> bool | None:
    del param, loaded_weight, weight_name, shard_id, expert_id
    return True if return_success else None
