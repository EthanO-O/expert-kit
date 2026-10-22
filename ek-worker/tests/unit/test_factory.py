"""Tests for concrete Worker component construction."""

from __future__ import annotations

import asyncio
import builtins
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
import torch
from expertkit_transport.controller import ResolvedDefaultInstance
from expertkit_transport.transports.base import BatchBufferConfig
from expertkit_transport.transports.grpc.worker_buffers import GrpcWorkerBatchBuffers

from expertkit_worker.config import WorkerConfig
from expertkit_worker.execution import AsyncExecutionSlot, CpuExecutionSlot
from expertkit_worker.factory import (
    _async_wiring,
    _create_device_wiring,
    build_worker_application,
)
from expertkit_worker.weights import DirectIOWeightDiskCache
from expertkit_worker.weights.metadata import (
    ModelMetadata,
    QuantizationMetadata,
    _quantization_from_metadata,
    resolve_model_metadata,
)


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


@pytest.mark.parametrize("backend_name", ["torch", "ascend_fused"])
@pytest.mark.parametrize(
    ("recipe", "expert_compute", "fail_after_graph"),
    [
        (None, "swiglu", False),
        ("modelslim-w8a8-dynamic", "swiglu", False),
        ("modelslim-w8a8-dynamic", "swiglu", True),
        ("modelslim-w8a8-dynamic", "deepseek_v4", False),
    ],
    ids=["bf16", "modelslim", "graph-close-failure", "v4"],
)
def test_factory_selects_graph_resident_from_validated_recipe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recipe: str | None,
    expert_compute: str,
    fail_after_graph: bool,
    backend_name: str,
) -> None:
    import expertkit_transport.transports.ascend_graph.control as graph_control

    import expertkit_worker.backends.torch.decode_graph as bf16_graph
    import expertkit_worker.backends.torch.modelslim_decode_graph as modelslim_graph
    import expertkit_worker.execution.decode_graph as graph_execution
    import expertkit_worker.factory as factory
    from expertkit_worker.backends import BackendResourceEstimate

    if backend_name == "ascend_fused":
        import expertkit_worker.backends.ascend_fused.decode_graph as packed_graph

    raw = _config(tmp_path).model_dump(mode="json")
    raw["worker"]["backend"] = backend_name
    raw["model"].update(activation_dtype="bf16", weight_dtype="bf16")
    if recipe is not None:
        raw["model"]["quantization"] = {
            "type": recipe,
            "bits": 8,
            "group_size": None,
            "symmetric": True,
        }
    is_v4 = expert_compute == "deepseek_v4"
    if is_v4:
        raw["model"].update(
            num_layers=43,
            experts_per_layer=256,
            hidden_dim=4096,
            expert_intermediate_dim=2048,
            top_k=6,
            expert_compute=expert_compute,
            swiglu_limit=10.0,
        )
    shared_model = {
        key: value
        for key, value in raw["model"].items()
        if key not in {"instance_id", "name", "weight_version"}
    }
    raw["worker"].update(
        device="npu:0",
        ascend_decode_graph={
            "session_id": "worker-graph-factory-0061",
            "instance_id": 7,
            "weight_version": "test",
            "workers": [
                {
                    "worker_id": "other",
                    "control_endpoint": "127.0.0.1:18078",
                    "expert_start": 0,
                    "expert_end": 2,
                },
                {
                    "worker_id": "worker-0",
                    "control_endpoint": "127.0.0.1:18079",
                    "expert_start": 2,
                    "expert_end": 4,
                },
            ],
            "rendezvous_host": "127.0.0.1",
            "rendezvous_port": 18080,
            "layer_ids": [0, 1],
            "capture_sizes": [1],
            "model": shared_model,
        },
    )
    if is_v4:
        raw["worker"]["ascend_decode_graph"]["workers"] = [
            {
                "worker_id": "worker-0" if rank == 1 else f"other-{rank}",
                "control_endpoint": f"127.0.0.1:{18100 + rank}",
                "expert_start": rank * 32,
                "expert_end": (rank + 1) * 32,
            }
            for rank in range(8)
        ]
    config = WorkerConfig.model_validate(raw)
    runtime = Mock(device=torch.device("cpu"))
    runtime.memory_info.return_value = (2 << 30, 2 << 30)
    adapter = Mock(backend_name="fixture")
    adapter.conversion_temporary_bytes.return_value = 0
    backend = Mock()
    backend.estimate_resources.return_value = BackendResourceEstimate(0)
    manager = Mock(ready_revision=17)
    services = Mock(manager=manager)
    services.close = AsyncMock()
    ordinary = Mock(fixed_device_bytes=0, fixed_host_staging_bytes=0)
    resident_types = [Mock(), Mock()]
    graph_factory = Mock()
    graph_factory.return_value.close = AsyncMock(side_effect=RuntimeError("graph close failed"))
    control_factory = Mock()
    monkeypatch.setattr(graph_control, "GraphControlServer", control_factory)
    monkeypatch.setattr(factory, "resolve_model_metadata", AsyncMock(return_value=config))
    monkeypatch.setattr(
        factory, "_create_device_wiring", lambda _device: Mock(runtime=runtime, create_slot=Mock())
    )
    observability = Mock(close=AsyncMock())
    monkeypatch.setattr(factory, "create_observability", Mock(return_value=observability))
    monkeypatch.setattr(factory, "create_weight_disk_cache", AsyncMock())
    monkeypatch.setattr(factory, "create_weight_adapter", Mock(return_value=adapter))
    monkeypatch.setattr(factory, "create_compute_backend", Mock(return_value=backend))
    monkeypatch.setattr(
        factory, "GrpcWorkerBatchReceiver", Mock(return_value=Mock(fixed_device_bytes=0))
    )
    monkeypatch.setattr(factory, "WorkerExecutor", Mock(return_value=ordinary))
    monkeypatch.setattr(factory, "create_weight_services", AsyncMock(return_value=services))
    monkeypatch.setattr(
        factory,
        "create_controller_supervisor",
        Mock(side_effect=ValueError("composition failed") if fail_after_graph else None),
    )
    monkeypatch.setattr(bf16_graph, "ResidentExpertGraphWeights", resident_types[0])
    monkeypatch.setattr(modelslim_graph, "ResidentModelSlimExpertGraphWeights", resident_types[1])
    if backend_name == "ascend_fused":
        monkeypatch.setattr(packed_graph, "ResidentAscendFusedGraphWeights", resident_types[0])
        monkeypatch.setattr(
            packed_graph, "create_modelslim_graph_weights", resident_types[1]
        )
    monkeypatch.setattr(graph_execution, "AscendDecodeGraphExecution", graph_factory)

    if fail_after_graph:
        with pytest.raises(RuntimeError, match="graph close failed"):
            asyncio.run(build_worker_application(config, instance_resolver=_resolve_instance))
        control_factory.return_value.stop.assert_called_once()
        control_factory.return_value.close.assert_called_once()
        services.close.assert_awaited_once()
        observability.close.assert_awaited_once()
        return

    application = asyncio.run(build_worker_application(config, instance_resolver=_resolve_instance))

    assert application._execution._ordinary is ordinary
    assert application._execution._graph is graph_factory.return_value
    assert application._execution._control is control_factory.return_value
    control_factory.assert_called_once_with(
        "127.0.0.1:18101" if is_v4 else "127.0.0.1:18079",
        begin=graph_factory.return_value.begin,
        finish=graph_factory.return_value.finish,
    )
    for resident in resident_types:
        resident.assert_not_called()
    options = graph_factory.call_args.kwargs
    assert options["worker_id"] == "worker-0"
    selected, other = resident_types if recipe is None else resident_types[::-1]
    assert options["prepare_weights"]() is selected.return_value
    other.assert_not_called()
    selected.assert_called_once()
    prepared = selected.call_args.kwargs
    assert prepared == {
        "layer_ids": (0, 1),
        "experts_per_layer": config.model.experts_per_layer,
        "hidden_dim": config.model.hidden_dim,
        "intermediate_dim": config.model.expert_intermediate_dim,
        "top_k": config.model.top_k,
        "acquire_many": manager.acquire_many,
        "ready_revision": prepared["ready_revision"],
        "expert_range": (32, 64) if is_v4 else (2, 4),
        "expert_compute": expert_compute,
        "swiglu_limit": 10.0 if is_v4 else 0.0,
    }
    assert prepared["ready_revision"]() == 17
    manager.ready_revision = 18
    assert prepared["ready_revision"]() == 18


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


def _buffer_config() -> BatchBufferConfig:
    return BatchBufferConfig(
        max_batch_tokens=2,
        hidden_dim=4,
        top_k=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


def test_cpu_wiring_builds_cpu_slot() -> None:
    wiring = _create_device_wiring("cpu")
    spec = _buffer_config()

    slot = wiring.create_slot(spec, GrpcWorkerBatchBuffers(spec))

    assert isinstance(slot, CpuExecutionSlot)
    assert wiring.runtime.device == torch.device("cpu")
    slot.close()


class _FakeAsyncRuntime:
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.current_device_set = False

    def device_context(self):
        from contextlib import nullcontext

        return nullcontext()

    def create_stream(self, *, priority: int = 0) -> object:
        assert priority == 0
        return object()

    def create_event(self, *, enable_timing: bool = False) -> object:
        assert not enable_timing
        return object()

    def set_current_device(self) -> None:
        self.current_device_set = True


def test_async_wiring_captures_runtime_in_slot_factory() -> None:
    runtime = _FakeAsyncRuntime()
    wiring = _async_wiring(runtime)  # type: ignore[arg-type]
    spec = _buffer_config()

    slot = wiring.create_slot(
        spec,
        GrpcWorkerBatchBuffers(spec),
        enable_device_timing=False,
    )

    assert runtime.current_device_set is True
    assert isinstance(slot, AsyncExecutionSlot)
    slot.close()


def test_cpu_and_cuda_selection_do_not_import_ascend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import expertkit_worker.factory as factory

    real_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any):
        if name == "expertkit_worker.device.ascend":
            raise AssertionError("Ascend runtime imported for a non-NPU device")
        return real_import(name, *args, **kwargs)

    class FakeCudaRuntime:
        def __init__(self, device: torch.device) -> None:
            self.device = device

    cuda_wiring = object()
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(factory, "CudaWorkerRuntime", FakeCudaRuntime)
    monkeypatch.setattr(factory, "_async_wiring", lambda _runtime: cuda_wiring)

    cpu_wiring = _create_device_wiring("cpu")
    assert isinstance(cpu_wiring.runtime, factory.CpuWorkerRuntime)
    assert _create_device_wiring("cuda:0") is cuda_wiring


def test_npu_selection_lazily_loads_ascend_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import expertkit_worker.factory as factory

    selected: list[object] = []

    class FakeAscendRuntime:
        def __init__(self, device: object) -> None:
            selected.append(device)

    module = ModuleType("expertkit_worker.device.ascend")
    module.AscendWorkerRuntime = FakeAscendRuntime  # type: ignore[attr-defined]
    wiring = object()
    monkeypatch.setitem(sys.modules, "expertkit_worker.device.ascend", module)
    monkeypatch.setattr(factory.torch, "device", lambda name: f"parsed:{name}")
    monkeypatch.setattr(factory, "_async_wiring", lambda _runtime: wiring)

    assert _create_device_wiring("npu:2") is wiring
    assert selected == ["parsed:npu:2"]


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

    monkeypatch.setattr("expertkit_worker.weights.metadata.fetch_model_metadata", discover)

    async def scenario() -> None:
        resolved = await resolve_model_metadata(_config(tmp_path))
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

    monkeypatch.setattr("expertkit_worker.weights.metadata.fetch_model_metadata", discover)
    resolved = asyncio.run(resolve_model_metadata(_config(tmp_path)))
    assert resolved.model.quantization.type.value == expected
    assert resolved.model.expert_compute == "deepseek_v4"
    assert resolved.model.swiglu_limit == 10


def test_invalid_metadata_never_falls_back_to_unquantized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from expertkit_worker.weights import ModelMetadataError

    async def discover(*args: object, **kwargs: object) -> ModelMetadata:
        raise ModelMetadataError("unsupported quantization recipe")

    monkeypatch.setattr("expertkit_worker.weights.metadata.fetch_model_metadata", discover)
    with pytest.raises(ModelMetadataError, match="unsupported quantization"):
        asyncio.run(resolve_model_metadata(_config(tmp_path)))


@pytest.mark.parametrize("method", ["fp8", "blockwise_int8", "int8", "compressed-tensors"])
def test_worker_rejects_unnormalized_quantization(method: str) -> None:
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


@pytest.mark.parametrize("concurrency", [1, 2, 4])
@pytest.mark.parametrize("recipe", ["fp4", "gptq"])
def test_host_budget_reserves_conversion_capacity(
    tmp_path: Path, concurrency: int, recipe: str
) -> None:
    from expertkit_worker.backends.torch import TorchFP4WeightAdapter, TorchGPTQWeightAdapter
    from expertkit_worker.weights.factory import _dram_cache_limit
    from expertkit_worker.weights.format import max_safetensors_file_bytes

    if recipe == "fp4":
        adapter = TorchFP4WeightAdapter(hidden_dim=128, intermediate_dim=256, device="cpu")
    else:
        adapter = TorchGPTQWeightAdapter(
            hidden_dim=128, intermediate_dim=256, group_size=128, device="cpu"
        )
    document = _config(tmp_path).model_dump()
    document["weight_manager"]["max_concurrent_loads"] = concurrency
    entry = max_safetensors_file_bytes(adapter.source_tensor_bytes()) + adapter.cpu_extra_bytes()
    conversion = concurrency * adapter.host_conversion_temporary_bytes()
    assert conversion > 0
    document["weight_manager"]["dram_cache"]["max_bytes"] = entry + conversion
    assert _dram_cache_limit(WorkerConfig.model_validate(document), adapter) == entry
    document["weight_manager"]["dram_cache"]["max_bytes"] -= 1
    with pytest.raises(ValueError, match="concurrent Host conversion"):
        _dram_cache_limit(WorkerConfig.model_validate(document), adapter)
    document["weight_manager"]["dram_cache"]["max_bytes"] = None
    config = WorkerConfig.model_validate(document)
    assert (
        _dram_cache_limit(config, adapter)
        == entry * config.model.num_layers * config.model.experts_per_layer
    )


def test_failed_receiver_cleanup_still_closes_disk_cache_and_observability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from expertkit_transport.transports.grpc import GrpcWorkerBatchReceiver

    import expertkit_worker.factory as factory
    from expertkit_worker.observability import create_observability

    raw = _config(tmp_path).model_dump()
    raw["worker"]["device"] = "cpu"
    config = WorkerConfig.model_validate(raw)
    observability = create_observability(config.observability, worker_id=config.worker.id)
    closed = []
    original_cache_close = DirectIOWeightDiskCache.close

    def cache_close(cache):
        closed.append("disk")
        original_cache_close(cache)

    async def initialize(cache):
        pass

    async def receiver_close(receiver):
        raise RuntimeError("native close failed")

    async def observability_close():
        closed.append("observability")

    def backend(*args, **kwargs):
        raise RuntimeError("backend startup failed")

    monkeypatch.setattr(DirectIOWeightDiskCache, "initialize", initialize)
    monkeypatch.setattr(DirectIOWeightDiskCache, "close", cache_close)
    monkeypatch.setattr(GrpcWorkerBatchReceiver, "close", receiver_close)
    monkeypatch.setattr(observability, "close", observability_close)
    monkeypatch.setattr(factory, "create_observability", lambda *args, **kwargs: observability)
    monkeypatch.setattr(factory, "create_compute_backend", backend)
    with pytest.raises(RuntimeError, match="native close failed"):
        asyncio.run(build_worker_application(config, instance_resolver=_resolve_instance))
    assert closed == ["disk", "observability"]
