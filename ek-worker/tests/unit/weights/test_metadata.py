"""Tests for Weight Server model metadata validation."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from expertkit_worker.config import WorkerConfig
from expertkit_worker.weights.metadata import (
    ModelMetadataError,
    ModelMetadataUnavailable,
    _parse_payload,
    resolve_worker_config,
)


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


def _payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "model_type": "qwen3_moe",
        "num_layers": 2,
        "moe_layer_start": 0,
        "moe_layer_end": 2,
        "experts_per_layer": 4,
        "hidden_dim": 4,
        "expert_intermediate_dim": 8,
        "top_k": 2,
        "activation_dtype": "bfloat16",
    }


def _document(tmp_path: Path, *, complete: bool = False, graph: bool = False) -> dict[str, Any]:
    document: dict[str, Any] = {
        "model": {
            "name": "fixture/model",
            "weight_version": "test",
            "activation_dtype": "bf16",
            "weight_dtype": "bf16",
        },
        "worker": {
            "id": "worker-0",
            "device": "npu:0" if graph else "cpu",
            "max_batch_tokens": 32,
            "device_memory_limit": "1GiB",
        },
        "transport": {"type": "grpc", "listen": "localhost:50051", "advertise": "worker:50051"},
        "controller": {"endpoint": "localhost:50050"},
        "weight_manager": {
            "disk_cache": {"path": str(tmp_path)},
            "peer": {"listen": "localhost:50052", "advertise": "http://worker:50052"},
            "weight_server_endpoint": "http://weights:8080",
        },
    }
    if complete:
        document["model"].update({field: _payload()[field] for field in _GEOMETRY_FIELDS})
    if graph:
        document["worker"]["ascend_decode_graph"] = {
            "control_listen": "0.0.0.0:18079",
            "control_advertise": "worker:18079",
        }
    return document


_GEOMETRY_FIELDS = (
    "num_layers",
    "experts_per_layer",
    "hidden_dim",
    "expert_intermediate_dim",
    "top_k",
)


@pytest.fixture
def metadata_fetch(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:

    fetch = AsyncMock(return_value=_parse_payload(_payload()))
    monkeypatch.setattr("expertkit_worker.weights.metadata.fetch_model_metadata", fetch)
    return fetch


def test_resolves_omitted_structure_without_mutating_document(tmp_path, metadata_fetch):

    document = _document(tmp_path)
    original = deepcopy(document)
    resolved = asyncio.run(resolve_worker_config(document))

    assert isinstance(resolved.config, WorkerConfig)
    for field in _GEOMETRY_FIELDS:
        assert getattr(resolved.config.model, field) == _payload()[field]
    assert resolved.config.worker.max_batch_tokens == 32
    assert resolved.config.model.activation_dtype.value == "bf16"
    assert resolved.metadata is metadata_fetch.return_value
    assert document == original
    metadata_fetch.assert_awaited_once_with("http://weights:8080/", "fixture/model")


@pytest.mark.parametrize("field", (*_GEOMETRY_FIELDS, "expert_compute", "swiglu_limit"))
def test_rejects_explicit_model_fact_conflicts(tmp_path, metadata_fetch, field):

    document = _document(tmp_path)
    document["model"][field] = "deepseek_v4" if field == "expert_compute" else 10
    with pytest.raises(ValueError, match=field):
        asyncio.run(resolve_worker_config(document))


@pytest.mark.parametrize("value", [None, 0, -1, "invalid"])
def test_rejects_invalid_supplied_dimension(tmp_path, metadata_fetch, value):

    document = _document(tmp_path)
    document["model"]["hidden_dim"] = value
    with pytest.raises(ValueError):
        asyncio.run(resolve_worker_config(document))


def test_validates_supplied_dimension_using_existing_coercion(tmp_path, metadata_fetch):

    document = _document(tmp_path)
    document["model"]["hidden_dim"] = "4"
    resolved = asyncio.run(resolve_worker_config(document))
    assert resolved.config.model.hidden_dim == 4


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("explicit_top_k", [False, True])
def test_missing_server_top_k_requires_explicit_ordinary_value(
    tmp_path, metadata_fetch, graph, explicit_top_k
):

    payload = _payload()
    payload["top_k"] = None
    metadata_fetch.return_value = _parse_payload(payload)
    document = _document(tmp_path, graph=graph)
    if explicit_top_k:
        document["model"]["top_k"] = 2
    if graph or not explicit_top_k:
        with pytest.raises(ModelMetadataError, match="top_k"):
            asyncio.run(resolve_worker_config(document))
    else:
        assert asyncio.run(resolve_worker_config(document)).config.model.top_k == 2


@pytest.mark.parametrize("complete", [False, True])
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("required", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_metadata_unavailability_and_disabled_discovery(
    tmp_path, metadata_fetch, complete, graph, required, enabled
):

    metadata_fetch.side_effect = ModelMetadataUnavailable("offline")
    document = _document(tmp_path, complete=complete, graph=graph)
    document["weight_manager"].update(auto_model_metadata=enabled, metadata_required=required)
    if not graph and complete and (not enabled or not required):
        resolved = asyncio.run(resolve_worker_config(document))
        assert resolved.config.model.hidden_dim == 4
        assert resolved.metadata is None
    else:
        with pytest.raises((ModelMetadataError, ValueError)):
            asyncio.run(resolve_worker_config(document))
    assert metadata_fetch.await_count == int(enabled)


@pytest.mark.parametrize("graph", [False, True])
def test_invalid_metadata_never_uses_explicit_config_fallback(tmp_path, metadata_fetch, graph):

    metadata_fetch.side_effect = ModelMetadataError("invalid response")
    with pytest.raises(ModelMetadataError, match="invalid response"):
        asyncio.run(resolve_worker_config(_document(tmp_path, complete=True, graph=graph)))


@pytest.mark.parametrize("explicit", [False, True])
def test_quantization_comes_from_storage_recipe_not_checkpoint_dtype(
    tmp_path, metadata_fetch, explicit
):

    payload = _payload()
    payload["quantization"] = {
        "method": "modelslim-w8a8-dynamic",
        "bits": 8,
        "group_size": None,
        "symmetric": True,
        "desc_act": False,
    }
    metadata_fetch.return_value = _parse_payload(payload)
    document = _document(tmp_path)
    document["worker"]["device"] = "npu:0"
    if explicit:
        document["model"]["quantization"] = {
            "type": "modelslim-w8a8-dynamic",
            "bits": 8,
            "group_size": None,
            "symmetric": True,
        }
    resolved = asyncio.run(resolve_worker_config(document))
    assert resolved.config.model.quantization.type.value == "modelslim-w8a8-dynamic"
    assert resolved.config.model.weight_dtype.value == "bf16"


@pytest.mark.parametrize("typed", [False, True])
def test_explicit_null_recipe_conflicts_with_quantized_metadata(tmp_path, metadata_fetch, typed):

    payload = _payload()
    payload["quantization"] = {
        "method": "gptq",
        "bits": 4,
        "group_size": 128,
        "symmetric": True,
    }
    metadata_fetch.return_value = _parse_payload(payload)
    document = _document(tmp_path, complete=True)
    document["model"]["quantization"] = None
    config = WorkerConfig.model_validate(document) if typed else document
    with pytest.raises(ValueError, match="quantization"):
        asyncio.run(resolve_worker_config(config))


def test_graph_startup_resolves_worker_model_without_injecting_graph_facts(
    tmp_path, metadata_fetch
):
    resolved = asyncio.run(resolve_worker_config(_document(tmp_path, graph=True)))
    assert resolved.config.model.hidden_dim == 4
    assert resolved.config.model.num_layers == 2
    assert resolved.config.worker.ascend_decode_graph.model_dump() == {
        "control_listen": "0.0.0.0:18079",
        "control_advertise": "worker:18079",
    }


def test_graph_supplied_model_is_rejected_as_a_second_authority(tmp_path, metadata_fetch):
    document = _document(tmp_path, graph=True)
    document["worker"]["ascend_decode_graph"]["model"] = {"hidden_dim": 2048}
    with pytest.raises(ValueError, match="Extra inputs"):
        asyncio.run(resolve_worker_config(document))


@pytest.mark.parametrize("graph", [False, True])
def test_missing_v4_limit_is_required_only_by_graph_startup(tmp_path, metadata_fetch, graph):

    payload = _payload()
    payload.update(model_type="deepseek_v4", expert_compute="deepseek_v4")
    metadata_fetch.return_value = _parse_payload(payload)
    document = _document(tmp_path, graph=graph)
    if graph:
        with pytest.raises(ModelMetadataError, match="swiglu_limit"):
            asyncio.run(resolve_worker_config(document))
    else:
        resolved = asyncio.run(resolve_worker_config(document))
        assert resolved.config.model.swiglu_limit == 0
        assert resolved.config.model.expert_compute == "deepseek_v4"


@pytest.mark.parametrize(
    "changes",
    [
        {"moe_layer_start": 2},
        {"moe_layer_end": 3},
        {"moe_layer_end": 1, "moe_layer_start": 1},
        {"top_k": 5},
    ],
)
def test_rejects_inconsistent_routing_metadata(changes):
    payload = _payload()
    payload.update(changes)
    with pytest.raises(ModelMetadataError):
        _parse_payload(payload)


def test_resolution_keeps_final_unknown_field_validation(tmp_path, metadata_fetch):

    document = _document(tmp_path)
    document["unexpected"] = True
    with pytest.raises(ValueError, match="Extra inputs"):
        asyncio.run(resolve_worker_config(document))


@pytest.mark.parametrize("field", ["name", "weight_server_endpoint"])
def test_invalid_lookup_selectors_fail_before_fetch(tmp_path, metadata_fetch, field):
    document = _document(tmp_path)
    section = "model" if field == "name" else "weight_manager"
    document[section][field] = None
    with pytest.raises(ValueError):
        asyncio.run(resolve_worker_config(document))
    metadata_fetch.assert_not_awaited()


def test_model_name_uses_normal_configuration_whitespace_rules(tmp_path, metadata_fetch):
    document = _document(tmp_path)
    document["model"]["name"] = " fixture/model "
    resolved = asyncio.run(resolve_worker_config(document))
    assert resolved.config.model.name == "fixture/model"
    metadata_fetch.assert_awaited_once_with("http://weights:8080/", "fixture/model")


@pytest.mark.parametrize("typed", [False, True])
def test_quantization_parameters_cannot_override_server_recipe(tmp_path, metadata_fetch, typed):
    payload = _payload()
    payload["quantization"] = {"method": "gptq", "bits": 4, "group_size": 128, "symmetric": True}
    metadata_fetch.return_value = _parse_payload(payload)
    document = _document(tmp_path, complete=True)
    document["model"]["quantization"] = {"type": "gptq", "group_size": 32}
    config = WorkerConfig.model_validate(document) if typed else document
    with pytest.raises(ValueError, match="quantization"):
        asyncio.run(resolve_worker_config(config))


def test_graph_endpoint_configuration_remains_strict(tmp_path, metadata_fetch):
    document = _document(tmp_path, graph=True)
    document["worker"]["ascend_decode_graph"]["control_listen"] = "http://worker:18079"
    with pytest.raises(ValueError, match="network address"):
        asyncio.run(resolve_worker_config(document))


def test_typed_graph_configuration_reuses_resolved_worker_facts(tmp_path, metadata_fetch):
    original = WorkerConfig.model_validate(_document(tmp_path, complete=True, graph=True))
    resolved = asyncio.run(resolve_worker_config(original))
    assert resolved.config.worker.ascend_decode_graph == original.worker.ascend_decode_graph
    assert resolved.config.model.hidden_dim == original.model.hidden_dim
