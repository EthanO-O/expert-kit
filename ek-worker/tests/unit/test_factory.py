"""Tests for concrete Worker component construction."""

from __future__ import annotations

import asyncio
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest
import torch
from expertkit_transport.controller import ResolvedDefaultInstance

from expertkit_worker.config import WorkerConfig
from expertkit_worker.factory import _resolve_model_metadata, build_worker_application
from expertkit_worker.weights import DirectIOWeightDiskCache
from expertkit_worker.weights.metadata import ModelMetadata, QuantizationMetadata


def _config(
    cache_path: Path,
    *,
    backend: str = "torch",
    instance_id: int | None = 7,
) -> WorkerConfig:
    worker_device = "cpu" if backend == "ggml" else "cuda:0"
    document: dict[str, object] = {
        "model": {
            "name": "fixture/model",
            "weight_version": "test",
            "num_layers": 2,
            "experts_per_layer": 4,
            "hidden_dim": 4,
            "expert_intermediate_dim": 8,
            "top_k": 2,
            "activation_dtype": "fp16",
            "weight_dtype": "fp16",
        },
        "worker": {
            "id": "worker-0",
            "backend": backend,
            "device": worker_device,
            "max_batch_tokens": 4,
            "max_active_batches_per_device": 1,
            "device_memory_limit": "513MiB" if backend == "fused" else "1GiB",
        },
        "transport": {
            "type": "grpc",
            "max_pending_batches_per_device": 1,
            "listen": "127.0.0.1:50051",
            "advertise": "worker-0:50051",
        },
        "controller": {"endpoint": "127.0.0.1:50050"},
        "weight_manager": {
            "max_concurrent_loads": 2,
            "disk_cache": {"path": str(cache_path)},
            "peer": {
                "listen": "[::1]:50052",
                "advertise": "http://worker-0:50052",
            },
            "weight_server_endpoint": "http://127.0.0.1:50053",
        },
    }
    if instance_id is not None:
        document["model"]["instance_id"] = instance_id
    if backend == "ggml":
        document["worker"]["ggml"] = {"cpu_threads": 2}
    return WorkerConfig.model_validate(document)


async def _resolve_instance(
    endpoint: str,
    *,
    requested_instance_id: int | None,
    timeout_seconds: float,
) -> ResolvedDefaultInstance:
    assert endpoint == "127.0.0.1:50050"
    assert requested_instance_id in (None, 7)
    assert timeout_seconds == 10
    return ResolvedDefaultInstance(7, "fixture/model", "default")


def test_factory_builds_and_closes_ggml_worker(tmp_path: Path) -> None:
    try:
        version("ggml-python")
    except PackageNotFoundError:
        pytest.skip("the GGML extra is not installed")

    async def scenario() -> None:
        application = await build_worker_application(
            _config(tmp_path, backend="ggml"),
            instance_resolver=_resolve_instance,
        )
        await application.close()

    asyncio.run(scenario())


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_factory_builds_and_closes_fused_worker(tmp_path: Path) -> None:
    async def scenario() -> None:
        application = await build_worker_application(
            _config(tmp_path, backend="fused"),
            instance_resolver=_resolve_instance,
        )
        await application.close()

    asyncio.run(scenario())


def test_factory_probes_direct_io_before_returning_application(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialized = False

    async def fail_initialize(_cache: DirectIOWeightDiskCache) -> None:
        nonlocal initialized
        initialized = True
        raise OSError("direct I/O probe failed")

    monkeypatch.setattr(DirectIOWeightDiskCache, "initialize", fail_initialize)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(
        "expertkit_worker.factory._memory_info",
        lambda _device: (2**40, 2**40),
    )

    async def scenario() -> None:
        with pytest.raises(OSError, match="direct I/O probe failed"):
            await build_worker_application(
                _config(tmp_path),
                instance_resolver=_resolve_instance,
            )

    asyncio.run(scenario())
    assert initialized is True


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_factory_builds_and_closes_torch_worker(tmp_path: Path) -> None:
    async def scenario() -> None:
        application = await build_worker_application(
            _config(tmp_path),
            instance_resolver=_resolve_instance,
        )
        await application.close()

    asyncio.run(scenario())


def test_factory_resolves_an_omitted_instance_before_device_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int | None] = []

    async def fail_resolution(
        _endpoint: str,
        *,
        requested_instance_id: int | None,
        timeout_seconds: float,
    ) -> ResolvedDefaultInstance:
        calls.append(requested_instance_id)
        assert timeout_seconds == 10
        raise RuntimeError("Controller resolution failed")

    monkeypatch.setattr(
        "expertkit_worker.factory.torch_dtype",
        lambda _dtype: pytest.fail("device setup started before instance resolution"),
    )

    async def scenario() -> None:
        with pytest.raises(RuntimeError, match="Controller resolution failed"):
            await build_worker_application(
                _config(tmp_path, instance_id=None),
                instance_resolver=fail_resolution,
            )

    asyncio.run(scenario())
    assert calls == [None]


def test_factory_discovers_gptq_from_weight_server_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = ModelMetadata(
        schema_version=1,
        model_type="qwen2_moe",
        num_layers=2,
        moe_layer_start=0,
        moe_layer_end=2,
        experts_per_layer=4,
        hidden_dim=4,
        expert_intermediate_dim=8,
        top_k=2,
        activation_dtype="bfloat16",
        quantization=QuantizationMetadata("gptq", 4, 128, True, False),
    )

    async def discover(*_args: object, **_kwargs: object) -> ModelMetadata:
        return metadata

    monkeypatch.setattr("expertkit_worker.factory.fetch_model_metadata", discover)

    async def scenario() -> None:
        resolved = await _resolve_model_metadata(_config(tmp_path))
        assert resolved.model.quantization is not None
        assert resolved.model.quantization.type.value == "gptq"
        assert resolved.model.quantization.group_size == 128

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "method,bits,group,expected", [("mxfp4", 4, 32, "fp4"), ("w8a8", 8, None, "w8a8")]
)
def test_v4_metadata_selects_recipe_and_expert_math(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    bits: int,
    group: int | None,
    expected: str,
) -> None:
    metadata = ModelMetadata(
        1,
        "deepseek_v4",
        2,
        0,
        2,
        4,
        4,
        8,
        2,
        "bfloat16",
        QuantizationMetadata(method, bits, group, True, False),
        "deepseek_v4",
        10.0,
    )

    async def discover(*args: object, **kwargs: object) -> ModelMetadata:
        return metadata

    monkeypatch.setattr("expertkit_worker.factory.fetch_model_metadata", discover)
    resolved = asyncio.run(_resolve_model_metadata(_config(tmp_path)))
    assert resolved.model.quantization.type.value == expected
    assert resolved.model.expert_compute == "deepseek_v4"
    assert resolved.model.swiglu_limit == 10


def test_invalid_metadata_never_falls_back_to_unquantized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from expertkit_worker.weights import ModelMetadataError

    async def discover(*args: object, **kwargs: object) -> ModelMetadata:
        raise ModelMetadataError("unsupported quantization recipe")

    monkeypatch.setattr("expertkit_worker.factory.fetch_model_metadata", discover)
    with pytest.raises(ModelMetadataError, match="unsupported quantization"):
        asyncio.run(_resolve_model_metadata(_config(tmp_path)))


@pytest.mark.parametrize("method", ["fp8", "blockwise_int8", "int8", "compressed-tensors"])
def test_worker_rejects_unnormalized_quantization(method: str) -> None:
    from expertkit_worker.factory import _quantization_from_metadata

    metadata = ModelMetadata(
        1,
        "deepseek_v4",
        1,
        0,
        1,
        2,
        128,
        256,
        1,
        "bfloat16",
        QuantizationMetadata(method, 8, None, True, False),
        "deepseek_v4",
        10.0,
    )
    with pytest.raises(ValueError, match="unsupported model quantization"):
        _quantization_from_metadata(metadata)
