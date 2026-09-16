"""Tests for Weight Server model metadata validation."""

from __future__ import annotations

import pytest

from expertkit_worker.weights.metadata import ModelMetadataError, _parse_payload


def test_parse_gptq_metadata() -> None:
    metadata = _parse_payload(
        {
            "schema_version": 1,
            "model_type": "qwen2_moe",
            "num_layers": 24,
            "moe_layer_start": 0,
            "moe_layer_end": 24,
            "experts_per_layer": 60,
            "hidden_dim": 2048,
            "expert_intermediate_dim": 1408,
            "top_k": 4,
            "activation_dtype": "bfloat16",
            "quantization": {
                "method": "gptq",
                "bits": 4,
                "group_size": 128,
                "symmetric": True,
                "desc_act": False,
            },
        }
    )
    assert metadata.quantization is not None
    assert metadata.quantization.group_size == 128
    assert metadata.top_k == 4


def test_parse_rejects_invalid_dimensions() -> None:
    with pytest.raises(ModelMetadataError, match="hidden_dim"):
        _parse_payload(
            {
                "schema_version": 1,
                "model_type": "qwen2_moe",
                "num_layers": 24,
                "moe_layer_start": 0,
                "moe_layer_end": 24,
                "experts_per_layer": 60,
                "hidden_dim": 0,
                "expert_intermediate_dim": 1408,
            }
        )
