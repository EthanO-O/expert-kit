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


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2),
        ("swiglu_limit", -1),
        ("swiglu_limit", float("nan")),
        ("swiglu_limit", False),
        ("expert_compute", "unknown"),
    ],
)
def test_rejects_invalid_expert_metadata(field: str, value: object) -> None:
    payload = {
        "schema_version": 1,
        "model_type": "deepseek_v4",
        "num_layers": 1,
        "moe_layer_start": 0,
        "moe_layer_end": 1,
        "experts_per_layer": 2,
        "hidden_dim": 128,
        "expert_intermediate_dim": 256,
        field: value,
    }
    with pytest.raises(ModelMetadataError):
        _parse_payload(payload)


@pytest.mark.parametrize(
    "status,invalid_json", [(404, False), (501, False), (400, False), (500, False), (200, True)]
)
def test_unavailable_metadata_is_distinct_from_invalid_response(
    monkeypatch: pytest.MonkeyPatch, status: int, invalid_json: bool
) -> None:
    import asyncio
    from contextlib import AbstractAsyncContextManager

    import aiohttp

    from expertkit_worker.weights.metadata import ModelMetadataUnavailable, fetch_model_metadata

    class Response(AbstractAsyncContextManager):
        def __init__(self) -> None:
            self.status = status

        async def __aenter__(self) -> object:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def json(self) -> object:
            if invalid_json:
                raise aiohttp.ContentTypeError(None, (), message="unexpected content type")
            return {}

    class Session(Response):
        def __init__(self, **kwargs: object) -> None:
            super().__init__()

        def get(self, url: str) -> Response:
            return Response()

    monkeypatch.setattr(aiohttp, "ClientSession", Session)
    with pytest.raises(ModelMetadataError) as caught:
        asyncio.run(fetch_model_metadata("http://fixture", "model"))
    assert isinstance(caught.value, ModelMetadataUnavailable) == (status in {404, 501})
