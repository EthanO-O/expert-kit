"""Model metadata discovery from the Weight Server."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import aiohttp


class ModelMetadataError(RuntimeError):
    """The Weight Server returned unavailable or invalid model metadata."""


@dataclass(frozen=True)
class QuantizationMetadata:
    """Normalized quantization recipe advertised by the Weight Server."""

    method: str
    bits: int | None
    group_size: int | None
    symmetric: bool | None
    desc_act: bool | None


@dataclass(frozen=True)
class ModelMetadata:
    """Normalized model dimensions and optional quantization recipe."""

    schema_version: int
    model_type: str
    num_layers: int
    moe_layer_start: int
    moe_layer_end: int
    experts_per_layer: int
    hidden_dim: int
    expert_intermediate_dim: int
    top_k: int | None
    activation_dtype: str | None
    quantization: QuantizationMetadata | None


def _positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ModelMetadataError(f"model metadata field {field!r} must be a positive integer")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ModelMetadataError(f"model metadata field {field!r} must be a nonnegative integer")
    return value


def _parse_payload(payload: Any) -> ModelMetadata:
    if not isinstance(payload, dict):
        raise ModelMetadataError("model metadata response must be a JSON object")
    schema_version = _positive_int(payload.get("schema_version"), "schema_version")
    model_type = payload.get("model_type")
    if not isinstance(model_type, str) or not model_type:
        raise ModelMetadataError("model metadata field 'model_type' must be a non-empty string")
    top_k = payload.get("top_k")
    if top_k is not None:
        top_k = _positive_int(top_k, "top_k")
    activation_dtype = payload.get("activation_dtype")
    if activation_dtype is not None and not isinstance(activation_dtype, str):
        raise ModelMetadataError("model metadata field 'activation_dtype' must be a string")
    quantization_payload = payload.get("quantization")
    quantization = None
    if quantization_payload is not None:
        if not isinstance(quantization_payload, dict):
            raise ModelMetadataError(
                "model metadata field 'quantization' must be an object or null"
            )
        method = quantization_payload.get("method")
        if not isinstance(method, str) or not method:
            raise ModelMetadataError("quantization.method must be a non-empty string")
        quantization = QuantizationMetadata(
            method=method,
            bits=quantization_payload.get("bits"),
            group_size=quantization_payload.get("group_size"),
            symmetric=quantization_payload.get("symmetric"),
            desc_act=quantization_payload.get("desc_act"),
        )
    return ModelMetadata(
        schema_version=schema_version,
        model_type=model_type,
        num_layers=_positive_int(payload.get("num_layers"), "num_layers"),
        moe_layer_start=_nonnegative_int(payload.get("moe_layer_start"), "moe_layer_start"),
        moe_layer_end=_positive_int(payload.get("moe_layer_end"), "moe_layer_end"),
        experts_per_layer=_positive_int(payload.get("experts_per_layer"), "experts_per_layer"),
        hidden_dim=_positive_int(payload.get("hidden_dim"), "hidden_dim"),
        expert_intermediate_dim=_positive_int(
            payload.get("expert_intermediate_dim"), "expert_intermediate_dim"
        ),
        top_k=top_k,
        activation_dtype=activation_dtype,
        quantization=quantization,
    )


async def fetch_model_metadata(
    endpoint: str,
    model_name: str,
    *,
    timeout_seconds: float = 5.0,
) -> ModelMetadata:
    """Fetch and validate one model's normalized metadata.

    Args:
        endpoint: Weight Server base URL.
        model_name: Model identifier understood by the Weight Server.
        timeout_seconds: Bound for the one startup request.

    Raises:
        ModelMetadataError: If the endpoint is unavailable or returns an invalid schema.
    """

    url = f"{str(endpoint).rstrip('/')}/meta/model/{quote(model_name, safe='')}"
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session, session.get(url) as response:
            if response.status != 200:
                raise ModelMetadataError(
                    f"Weight Server metadata request returned HTTP {response.status}"
                )
            try:
                payload = await response.json()
            except (TypeError, ValueError) as error:
                raise ModelMetadataError("Weight Server metadata response is not JSON") from error
    except ModelMetadataError:
        raise
    except (aiohttp.ClientError, TimeoutError) as error:
        raise ModelMetadataError(f"cannot fetch model metadata from {url}") from error
    return _parse_payload(payload)
