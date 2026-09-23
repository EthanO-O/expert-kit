"""Model metadata discovery from the Weight Server."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import aiohttp
import structlog

from expertkit_worker.config import QuantizationConfig, QuantizationType, WorkerConfig

logger = structlog.get_logger(__name__)


class ModelMetadataError(RuntimeError):
    """The Weight Server returned unavailable or invalid model metadata."""


class ModelMetadataUnavailable(ModelMetadataError):
    """A legacy or unreachable server cannot supply model metadata."""


@dataclass(frozen=True)
class QuantizationMetadata:
    """Closed quantization recipes advertised by the Weight Server.

    W8A8 means symmetric channel INT8 weights and dynamic token INT8 activations
    with FP32 scales. MXFP4 means E2M1/E8M0 weights in groups of 32 and E4M3
    activation rounding in groups of 128 with power-of-two scales. These method
    names do not imply support for arbitrary exports with the same bit widths.
    """

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
    expert_compute: str = "swiglu"
    swiglu_limit: float = 0.0


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
    if schema_version != 1:
        raise ModelMetadataError("unsupported model metadata schema version")
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
        for field in ("bits", "group_size"):
            if quantization_payload.get(field) is not None:
                _positive_int(quantization_payload[field], f"quantization.{field}")
        for field in ("symmetric", "desc_act"):
            value = quantization_payload.get(field)
            if value is not None and not isinstance(value, bool):
                raise ModelMetadataError(f"quantization.{field} must be a boolean")
        quantization = QuantizationMetadata(
            method=method,
            bits=quantization_payload.get("bits"),
            group_size=quantization_payload.get("group_size"),
            symmetric=quantization_payload.get("symmetric"),
            desc_act=quantization_payload.get("desc_act"),
        )
    expert_compute = payload.get("expert_compute", "swiglu")
    if not isinstance(expert_compute, str) or expert_compute not in {"swiglu", "deepseek_v4"}:
        raise ModelMetadataError("unsupported expert_compute metadata")
    limit = payload.get("swiglu_limit")
    if limit is None:
        limit = 0.0
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int | float)
        or not math.isfinite(limit)
        or limit < 0
    ):
        raise ModelMetadataError("swiglu_limit must be finite and nonnegative")
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
        expert_compute=expert_compute,
        swiglu_limit=float(limit),
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
            if response.status in {404, 501}:
                raise ModelMetadataUnavailable("Weight Server has no model metadata endpoint")
            if response.status != 200:
                raise ModelMetadataError(
                    f"Weight Server metadata request returned HTTP {response.status}"
                )
            try:
                payload = await response.json()
            except (TypeError, ValueError, aiohttp.ContentTypeError) as error:
                raise ModelMetadataError("Weight Server metadata response is not JSON") from error
    except ModelMetadataError:
        raise
    except (aiohttp.ClientError, TimeoutError) as error:
        raise ModelMetadataUnavailable(f"cannot fetch model metadata from {url}") from error
    return _parse_payload(payload)


def _quantization_from_metadata(metadata: ModelMetadata) -> QuantizationConfig | None:
    """Convert a normalized server recipe into the Worker's validated recipe."""

    if metadata.quantization is None:
        return None
    recipe = metadata.quantization
    method = recipe.method.lower()
    if (
        method == "mxfp4"
        and recipe.bits == 4
        and recipe.group_size == 32
        and recipe.symmetric is True
        and recipe.desc_act is not True
    ):
        return QuantizationConfig(type=QuantizationType.FP4, bits=4, group_size=32, symmetric=True)
    if (
        method == "w8a8"
        and recipe.bits == 8
        and recipe.symmetric is True
        and recipe.desc_act is not True
    ):
        return QuantizationConfig(
            type=QuantizationType.W8A8,
            bits=8,
            group_size=recipe.group_size,
            symmetric=True,
        )
    if (
        method == QuantizationType.MODELSLIM_W8A8_DYNAMIC.value
        and recipe.bits == 8
        and recipe.group_size is None
        and recipe.symmetric is True
        and recipe.desc_act is not True
    ):
        return QuantizationConfig(
            type=QuantizationType.MODELSLIM_W8A8_DYNAMIC,
            bits=8,
            group_size=None,
            symmetric=True,
        )
    if method != QuantizationType.GPTQ.value:
        raise ValueError(f"unsupported model quantization method: {recipe.method}")
    if recipe.bits != 4 or recipe.group_size is None:
        raise ValueError("Weight Server advertised unsupported GPTQ parameters")
    if recipe.desc_act is True:
        raise ValueError("GPTQ activation-order groups are unsupported")
    if recipe.symmetric is False:
        raise ValueError("the Torch GPTQ Backend requires symmetric quantization")
    return QuantizationConfig(
        type=QuantizationType.GPTQ,
        bits=recipe.bits,
        group_size=recipe.group_size,
        symmetric=True,
    )


def _validate_model_metadata(config: WorkerConfig, metadata: ModelMetadata) -> None:
    """Reject metadata whose dimensions disagree with Worker routing."""

    expected = config.model
    fields = (
        ("num_layers", expected.num_layers, metadata.num_layers),
        ("experts_per_layer", expected.experts_per_layer, metadata.experts_per_layer),
        ("hidden_dim", expected.hidden_dim, metadata.hidden_dim),
        (
            "expert_intermediate_dim",
            expected.expert_intermediate_dim,
            metadata.expert_intermediate_dim,
        ),
    )
    for name, configured, advertised in fields:
        if configured != advertised:
            raise ValueError(
                f"Weight Server model metadata mismatch for {name}: "
                f"Worker has {configured}, server has {advertised}"
            )
    if metadata.top_k is not None and expected.top_k != metadata.top_k:
        raise ValueError(
            f"Weight Server model metadata mismatch for top_k: "
            f"Worker has {expected.top_k}, server has {metadata.top_k}"
        )


async def resolve_model_metadata(config: WorkerConfig) -> WorkerConfig:
    """Resolve model recipes from Weight Server metadata with legacy fallback."""

    if not config.weight_manager.auto_model_metadata:
        return config
    try:
        metadata = await fetch_model_metadata(
            str(config.weight_manager.weight_server_endpoint),
            config.model.name,
        )
        _validate_model_metadata(config, metadata)
        discovered = _quantization_from_metadata(metadata)
        configured = config.model.quantization
        if configured is not None and discovered != configured:
            raise ValueError("Worker model.quantization conflicts with Weight Server metadata")
        for field in ("expert_compute", "swiglu_limit"):
            if field in config.model.model_fields_set and getattr(config.model, field) != getattr(
                metadata, field
            ):
                raise ValueError(f"Worker model.{field} conflicts with Weight Server metadata")
        model_data = config.model.model_dump()
        model_data.update(
            quantization=discovered,
            expert_compute=metadata.expert_compute,
            swiglu_limit=metadata.swiglu_limit,
        )
        if metadata.model_type == "deepseek_v4" and metadata.expert_compute != "deepseek_v4":
            raise ValueError(
                "V4 requires expert computation metadata from an updated Weight Server"
            )
        data = config.model_dump()
        data["model"] = model_data
        return WorkerConfig.model_validate(data)
    except ModelMetadataUnavailable as error:
        if config.weight_manager.metadata_required:
            raise RuntimeError("required Weight Server model metadata is unavailable") from error
        logger.warning("model_metadata_unavailable", error=str(error))
        return config
