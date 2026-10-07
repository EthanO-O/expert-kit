"""Validate Frontend router output before narrowing it to wire dtypes."""

from __future__ import annotations

import torch


def _validate_single_token_routing(
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    experts_per_layer: int,
) -> tuple[int, ...]:
    """Check the original small row before narrowing IDs or routing weights."""
    host_ids = expert_ids.detach().cpu().tolist()[0]
    if any(expert_id < -1 or expert_id >= experts_per_layer for expert_id in host_ids):
        raise ValueError("expert_ids must be within the configured expert range [-1, count)")
    if -1 in host_ids:
        host_weights = routing_weights.detach().cpu().tolist()[0]
        if any(
            expert_id == -1 and weight != 0
            for expert_id, weight in zip(host_ids, host_weights, strict=True)
        ):
            raise ValueError("an expert ID of -1 must have zero routing weight")
    return tuple(sorted({expert_id for expert_id in host_ids if expert_id >= 0}))


def validate_and_convert_routing(
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    *,
    experts_per_layer: int,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    """Return validated routing Tensors and their sorted distinct expert IDs.

    A single Token's routing values are checked on the Host; larger batches
    copy a fixed-size validation summary instead. Both paths may synchronize
    an accelerator. Call once on final router output, before integer narrowing
    could change an invalid expert ID into a valid one.
    """

    if (
        isinstance(experts_per_layer, bool)
        or not isinstance(experts_per_layer, int)
        or not 0 < experts_per_layer <= torch.iinfo(torch.int32).max + 1
    ):
        raise ValueError("experts_per_layer must fit the int32 expert-ID range")
    if expert_ids.ndim != 2:
        raise ValueError("expert_ids must have shape [token_count, top_k]")
    if routing_weights.shape != expert_ids.shape:
        raise ValueError("routing_weights must match expert_ids")
    if routing_weights.device != expert_ids.device:
        raise ValueError("routing_weights and expert_ids must use the same device")
    try:
        integer_info = torch.iinfo(expert_ids.dtype)
    except TypeError as error:
        raise ValueError("expert_ids must use an integer dtype") from error

    if expert_ids.shape[0] == 1:
        distinct_expert_ids = _validate_single_token_routing(
            expert_ids, routing_weights, experts_per_layer
        )
        return (
            expert_ids.to(dtype=torch.int32),
            routing_weights.to(dtype=torch.float32),
            distinct_expert_ids,
        )

    checked_ids = expert_ids
    if integer_info.min >= 0:
        checked_ids = expert_ids.to(
            torch.float64 if integer_info.max > torch.iinfo(torch.int64).max else torch.int64
        )
    invalid_range = (checked_ids < -1) | (checked_ids >= experts_per_layer)
    invalid_weight = (checked_ids == -1) & (routing_weights != 0)
    valid = (checked_ids >= 0) & (checked_ids < experts_per_layer)
    safe_ids = torch.where(valid, checked_ids, 0).to(dtype=torch.int64).flatten()

    summary = torch.zeros(
        experts_per_layer + 2,
        dtype=torch.int32,
        device=expert_ids.device,
    )
    summary[:2].copy_(
        torch.stack((invalid_range.any(), invalid_weight.any())).to(dtype=torch.int32)
    )
    summary[2:].scatter_add_(
        0,
        safe_ids,
        valid.flatten().to(dtype=torch.int32),
    )
    host_summary = summary.detach().to(device="cpu").tolist()
    range_error, weight_error = host_summary[:2]
    if range_error:
        raise ValueError("expert_ids must be within the configured expert range [-1, count)")
    if weight_error:
        raise ValueError("an expert ID of -1 must have zero routing weight")
    distinct_expert_ids = tuple(
        expert_id for expert_id, assignment_count in enumerate(host_summary[2:]) if assignment_count
    )
    return (
        expert_ids.to(dtype=torch.int32),
        routing_weights.to(dtype=torch.float32),
        distinct_expert_ids,
    )
