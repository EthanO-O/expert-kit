"""Tests for Backend-specific Worker configuration constraints."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from expertkit_worker.config import ShmTransportConfig, WorkerConfig


def _config(cache_path: Path, *, backend: str, device: str) -> dict[str, object]:
    return {
        "model": {
            "instance_id": 1,
            "name": "test",
            "weight_version": "v1",
            "num_layers": 1,
            "experts_per_layer": 4,
            "hidden_dim": 16,
            "expert_intermediate_dim": 32,
            "top_k": 2,
            "activation_dtype": "fp16",
            "weight_dtype": "fp16",
        },
        "worker": {
            "id": "worker-0",
            "backend": backend,
            "device": device,
            "device_memory_limit": "2GiB",
        },
        "transport": {
            "type": "grpc",
            "listen": "127.0.0.1:50051",
            "advertise": "worker:50051",
        },
        "controller": {"endpoint": "controller:50050"},
        "weight_manager": {
            "disk_cache": {"path": cache_path},
            "peer": {
                "listen": "127.0.0.1:50052",
                "advertise": "http://worker:50052",
            },
            "weight_server_endpoint": "http://weights:8080",
        },
    }


def test_ggml_requires_cpu_threads(tmp_path: Path) -> None:
    raw = _config(tmp_path, backend="ggml", device="cpu")

    with pytest.raises(ValidationError, match=r"worker\.ggml configuration is required"):
        WorkerConfig.model_validate(raw)

    worker = raw["worker"]
    assert isinstance(worker, dict)
    worker["ggml"] = {"cpu_threads": 8}
    config = WorkerConfig.model_validate(raw)
    assert config.worker.ggml is not None
    assert config.worker.ggml.cpu_threads == 8


def test_non_ggml_backend_rejects_ggml_settings(tmp_path: Path) -> None:
    raw = _config(tmp_path, backend="torch", device="cuda:0")
    worker = raw["worker"]
    assert isinstance(worker, dict)
    worker["ggml"] = {"cpu_threads": 8}

    with pytest.raises(ValidationError, match=r"valid only when worker\.backend is ggml"):
        WorkerConfig.model_validate(raw)


def test_fused_rejects_fp32(tmp_path: Path) -> None:
    raw = _config(tmp_path, backend="fused", device="cuda:1")
    model = raw["model"]
    assert isinstance(model, dict)
    model["activation_dtype"] = "fp32"

    with pytest.raises(ValidationError, match="supports only FP16 or BF16"):
        WorkerConfig.model_validate(raw)


def test_paths_must_be_absolute(tmp_path: Path) -> None:
    raw = _config(tmp_path, backend="torch", device="cuda:0")
    weight_manager = raw["weight_manager"]
    assert isinstance(weight_manager, dict)
    weight_manager["disk_cache"] = {"path": "relative/cache"}

    with pytest.raises(ValidationError, match="path must be absolute"):
        WorkerConfig.model_validate(raw)


def test_device_memory_limit_must_be_positive(tmp_path: Path) -> None:
    raw = _config(tmp_path, backend="torch", device="cuda:0")
    worker = raw["worker"]
    assert isinstance(worker, dict)
    worker["device_memory_limit"] = 0

    with pytest.raises(ValidationError, match="greater than 0"):
        WorkerConfig.model_validate(raw)


def test_torch_backend_accepts_indexed_npu_device(tmp_path: Path) -> None:
    config = WorkerConfig.model_validate(_config(tmp_path, backend="torch", device="npu:0"))

    assert config.worker.device == "npu:0"


@pytest.mark.parametrize(
    "field,value",
    [
        (None, None),
        (
            "matching_quantization",
            {"type": "modelslim-w8a8-dynamic", "bits": 8, "group_size": None, "symmetric": True},
        ),
        ("num_layers", 2),
        ("experts_per_layer", 5),
        ("hidden_dim", 32),
        ("expert_intermediate_dim", 64),
        ("top_k", 3),
        ("activation_dtype", "fp16"),
        ("weight_dtype", "fp16"),
        ("expert_compute", "deepseek_v4"),
        ("swiglu_limit", 1.0),
        (
            "quantization",
            {"type": "modelslim-w8a8-dynamic", "bits": 8, "group_size": None, "symmetric": True},
        ),
        ("quantization", {"type": "w8a8", "bits": 8, "group_size": None}),
        ("quantization", {"type": "gptq", "bits": 4, "group_size": 128}),
        ("quantization", {"type": "fp4", "bits": 4, "group_size": 32}),
        ("quantization", {"type": "modelslim-w8a8-dynamic", "bits": 4, "group_size": None}),
        ("quantization", {"type": "modelslim-w8a8-dynamic", "bits": 8, "group_size": 128}),
        (
            "quantization",
            {"type": "modelslim-w8a8-dynamic", "bits": 8, "group_size": None, "symmetric": False},
        ),
    ],
)
def test_graph_worker_checks_complete_shared_model(tmp_path, field, value):
    raw = _config(tmp_path, backend="torch", device="npu:0")
    raw["model"].update(activation_dtype="bf16", weight_dtype="bf16")
    shared_model = {
        key: value
        for key, value in raw["model"].items()
        if key not in {"instance_id", "name", "weight_version"}
    }
    shared_model.update(
        activation="silu", expert_compute="swiglu", swiglu_limit=0.0, quantization=None
    )
    raw["worker"].update(
        max_active_batches_per_device=1,
        ascend_decode_graph={
            "session_id": "worker-graph-model-0061",
            "instance_id": 1,
            "weight_version": "v1",
            "worker_id": "worker-0",
            "control_endpoint": "127.0.0.1:18079",
            "rendezvous_host": "127.0.0.1",
            "rendezvous_port": 18080,
            "layer_ids": [0],
            "capture_sizes": [1],
            "model": shared_model,
        },
    )
    if field == "matching_quantization":
        raw["model"]["quantization"] = value
        shared_model["quantization"] = value
    if field is None or field == "matching_quantization":
        assert WorkerConfig.model_validate(raw).worker.ascend_decode_graph is not None
    else:
        raw["model"][field] = value
        with pytest.raises(ValidationError):
            WorkerConfig.model_validate(raw)


@pytest.mark.parametrize("device", ["npu", "npu:x", "cuda", "cuda:x"])
def test_torch_backend_rejects_unindexed_or_invalid_accelerator(
    tmp_path: Path,
    device: str,
) -> None:
    with pytest.raises(ValidationError, match=r"cpu, cuda:<id> or npu:<id>"):
        WorkerConfig.model_validate(_config(tmp_path, backend="torch", device=device))


def test_fused_backend_remains_cuda_only(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match=r"requires cuda:<id>"):
        WorkerConfig.model_validate(_config(tmp_path, backend="fused", device="npu:0"))


def test_ascend_fused_backend_requires_indexed_npu(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match=r"requires npu:<id>"):
        WorkerConfig.model_validate(_config(tmp_path, backend="ascend_fused", device="cuda:0"))


def test_ascend_fused_accepts_modelslim_and_v4(tmp_path: Path) -> None:
    raw = _config(tmp_path, backend="ascend_fused", device="npu:0")
    model = raw["model"]
    assert isinstance(model, dict)
    model["activation_dtype"] = "bf16"
    model["weight_dtype"] = "bf16"
    model["expert_compute"] = "deepseek_v4"
    model["swiglu_limit"] = 7.0
    model["quantization"] = {
        "type": "modelslim-w8a8-dynamic",
        "bits": 8,
        "group_size": None,
    }

    config = WorkerConfig.model_validate(raw)

    assert config.model.quantization is not None
    assert config.model.expert_compute == "deepseek_v4"


def test_ascend_fused_rejects_other_quantization(tmp_path: Path) -> None:
    raw = _config(tmp_path, backend="ascend_fused", device="npu:0")
    model = raw["model"]
    assert isinstance(model, dict)
    model["quantization"] = {"type": "w8a8", "bits": 8, "group_size": None}

    with pytest.raises(ValidationError, match="supports only ModelSlim W8A8_DYNAMIC"):
        WorkerConfig.model_validate(raw)


def test_npu_worker_rejects_shm_transport(tmp_path: Path) -> None:
    raw = _config(tmp_path, backend="torch", device="npu:0")
    raw["transport"] = {
        "type": "shm",
        "rpc_listen": "127.0.0.1:50051",
        "rpc_advertise": "worker:50051",
        "shared_memory_dir": "/dev/shm",
    }

    with pytest.raises(
        ValidationError, match="shared-memory transport is unavailable for NPU workers"
    ):
        WorkerConfig.model_validate(raw)


def test_gptq_quantization_requires_4_bit_grouped_weights(tmp_path: Path) -> None:
    raw = _config(tmp_path, backend="torch", device="cuda:0")
    model = raw["model"]
    assert isinstance(model, dict)
    model["quantization"] = {"type": "gptq", "bits": 4, "group_size": 128}

    config = WorkerConfig.model_validate(raw)

    assert config.model.quantization is not None
    assert config.model.quantization.group_size == 128


def test_shm_transport_uses_only_notification_rpc_fields(tmp_path: Path) -> None:
    raw = _config(tmp_path, backend="torch", device="cuda:0")
    raw["transport"] = {
        "type": "shm",
        "rpc_listen": "127.0.0.1:50051",
        "rpc_advertise": "worker:50051",
        "shared_memory_dir": "/dev/shm",
    }

    config = WorkerConfig.model_validate(raw)

    assert isinstance(config.transport, ShmTransportConfig)
    assert config.transport.max_pending_batches_per_device == 1
    assert config.transport.shared_memory_dir == "/dev/shm"


@pytest.mark.parametrize(
    "transport",
    [
        {"grpc": {"listen": "127.0.0.1:50051", "advertise": "worker:50051"}},
        {
            "type": "grpc",
            "listen": "127.0.0.1:50051",
            "advertise": "worker:50051",
            "rpc_listen": "127.0.0.1:50052",
        },
        {
            "type": "shm",
            "rpc_listen": "127.0.0.1:50051",
            "rpc_advertise": "worker:50051",
            "listen": "127.0.0.1:50052",
        },
    ],
)
def test_transport_rejects_old_or_mixed_fields(
    tmp_path: Path,
    transport: dict[str, object],
) -> None:
    raw = _config(tmp_path, backend="torch", device="cuda:0")
    raw["transport"] = transport

    with pytest.raises(ValidationError):
        WorkerConfig.model_validate(raw)


def test_tracing_requires_plaintext_endpoint_and_bounded_sampling(tmp_path: Path) -> None:
    raw = _config(tmp_path, backend="torch", device="cuda:0")
    raw["observability"] = {
        "tracing": {
            "enabled": True,
            "endpoint": "https://collector:4317",
            "sample_ratio": 1,
        }
    }

    with pytest.raises(ValidationError, match="plaintext HTTP endpoint"):
        WorkerConfig.model_validate(raw)

    raw["observability"] = {
        "tracing": {
            "enabled": True,
            "endpoint": "http://collector:4317",
            "sample_ratio": 0,
        }
    }
    with pytest.raises(ValidationError, match="greater than 0"):
        WorkerConfig.model_validate(raw)


@pytest.mark.parametrize(
    "device, pending, valid",
    [
        ("npu:0", 4095, True),
        ("npu:0", 4096, False),
        ("cpu", 1, False),
    ],
)
def test_ascend_transport_requires_npu_and_bounded_registered_capacity(
    tmp_path: Path, device: str, pending: int, valid: bool
) -> None:
    raw = _config(tmp_path, backend="torch", device=device)
    raw["transport"] = dict(
        type="transfer_engine",
        control_listen="0.0.0.0:50051",
        control_advertise="worker:50051",
        segment_advertise="worker:0",
        max_pending_batches_per_device=pending,
    )
    if valid:
        config = WorkerConfig.model_validate(raw)
        assert config.transport.segment_advertise == "worker:0"
        assert config.transport.ascend_receive_fence == "stream"
    else:
        with pytest.raises(ValidationError, match=r"capacity|indexed NPU"):
            WorkerConfig.model_validate(raw)


@pytest.mark.parametrize(
    "field,value",
    [(None, None), ("quantization", None), ("swiglu_limit", 0.0), ("hidden_dim", 2048)],
)
def test_graph_worker_requires_exact_v4_computation(tmp_path, field, value):
    raw = _config(tmp_path, backend="torch", device="npu:0")
    raw["model"].update(
        num_layers=43,
        experts_per_layer=256,
        hidden_dim=4096,
        expert_intermediate_dim=2048,
        top_k=6,
        activation_dtype="bf16",
        weight_dtype="bf16",
        activation="silu",
        expert_compute="deepseek_v4",
        swiglu_limit=10.0,
        quantization={
            "type": "modelslim-w8a8-dynamic",
            "bits": 8,
            "group_size": None,
            "symmetric": True,
        },
    )
    shared_model = {
        key: value
        for key, value in raw["model"].items()
        if key not in {"instance_id", "name", "weight_version"}
    }
    raw["worker"].update(
        max_active_batches_per_device=1,
        ascend_decode_graph={
            "session_id": "worker-graph-v4-model",
            "instance_id": 1,
            "weight_version": "v1",
            "workers": [
                {
                    "worker_id": f"worker-{rank}",
                    "control_endpoint": f"127.0.0.1:{18100 + rank}",
                    "expert_start": rank * 32,
                    "expert_end": (rank + 1) * 32,
                }
                for rank in range(8)
            ],
            "rendezvous_host": "127.0.0.1",
            "rendezvous_port": 18080,
            "layer_ids": [0],
            "capture_sizes": [1],
            "model": shared_model,
        },
    )
    if field is None:
        assert WorkerConfig.model_validate(raw).worker.ascend_decode_graph is not None
    else:
        raw["model"][field] = value
        with pytest.raises(ValidationError, match="exact V4"):
            WorkerConfig.model_validate(raw)
