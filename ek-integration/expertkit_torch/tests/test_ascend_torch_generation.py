"""Structural tests for generated Ascend deployment configuration."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
ASCEND = ROOT / "dev" / "ascend"


def _inputs(
    tmp_path: Path,
    *,
    model_config: str = "qwen3-30b-a3b",
    dataset_type: str = "sharegpt",
) -> tuple[Path, Path]:
    cluster = tmp_path / "cluster.yaml"
    experiment = tmp_path / "experiment.yaml"
    example_dir = ASCEND / "configs" / model_config
    shutil.copy(example_dir / "cluster.example.yaml", cluster)
    shutil.copy(example_dir / "experiment.example.yaml", experiment)

    cluster_data = yaml.safe_load(cluster.read_text(encoding="utf-8"))
    cluster_data["nodes"]["node-a"]["address"] = "192.0.2.10"
    cluster_data["nodes"]["node-b"]["address"] = "192.0.2.11"
    cluster_data["nodes"]["node-c"]["address"] = "192.0.2.12"
    cluster.write_text(yaml.safe_dump(cluster_data), encoding="utf-8")

    data = yaml.safe_load(experiment.read_text(encoding="utf-8"))
    data["dataset"] = {
        "type": dataset_type,
        "name": dataset_type.title(),
    }
    if dataset_type == "sharegpt":
        data["dataset"].update(
            {
                "path_ref": "sharegpt",
                "mounted_path": "/dataset/sharegpt",
                "file": "ShareGPT.json",
            }
        )
        data["run"].pop("input_len", None)
    experiment.write_text(yaml.safe_dump(data), encoding="utf-8")
    return cluster, experiment


def _generate(
    cluster: Path, experiment: Path, output: Path, *, expect_success: bool = True
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ASCEND)
    completed = subprocess.run(
        [
            sys.executable,
            str(ASCEND / "cli.py"),
            "generate",
            "--cluster",
            str(cluster),
            "--experiment",
            str(experiment),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if expect_success:
        assert completed.returncode == 0, completed.stderr
    return completed


@pytest.mark.parametrize(
    ("model_config", "expected_model_path", "expected_memory_limit"),
    [
        (
            "qwen3-30b-a3b",
            "<QWEN3_30B_A3B_MODEL_DIR>",
            "20GiB",
        ),
        (
            "deepseek-v3",
            "<DEEPSEEK_V3_BF16_DIR>",
            "52GiB",
        ),
    ],
)
def test_sharegpt_generation_emits_host_native_torch_config(
    tmp_path: Path,
    model_config: str,
    expected_model_path: str,
    expected_memory_limit: str,
) -> None:
    cluster, experiment = _inputs(
        tmp_path,
        model_config=model_config,
    )
    output = tmp_path / "generated"

    _generate(cluster, experiment, output)
    config = yaml.safe_load((output / "torch-bench.yaml").read_text())

    assert config["controller-endpoint"] == "192.0.2.10:15002"
    assert config["device-platform"] == "npu"
    assert config["device-ids"] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert config["num-prompts"] == 16
    assert config["max-concurrency"] == 1
    assert config["dataset-path"] == ("<SHAREGPT_DATASET_DIR>/ShareGPT.json")
    assert config["model-path"] == expected_model_path

    worker = yaml.safe_load((output / "worker-pool1" / "workers" / "worker-00.yaml").read_text())
    assert worker["worker"]["device_memory_limit"] == expected_memory_limit


def test_generation_splits_control_and_attention_bundles(tmp_path: Path) -> None:
    cluster, experiment = _inputs(tmp_path)
    output = tmp_path / "generated"

    _generate(cluster, experiment, output)
    control = yaml.safe_load((output / "compose.control.yaml").read_text())
    attention = yaml.safe_load((output / "compose.attention.yaml").read_text())

    assert control["name"] == "expert-kit-ascend-control"
    assert set(control["services"]) == {
        "admin",
        "controller",
        "migrations",
        "model-init",
        "postgres",
        "weight-server",
    }
    assert attention["name"] == "expert-kit-ascend-attention"
    assert set(attention["services"]) == {"attention", "benchmark"}
    assert attention["services"]["attention"]["environment"]["EK_ADDR"] == ("192.0.2.10:15002")
    frontend = attention["services"]["attention"]
    assert frontend["networks"] == ["expert-kit-network"]
    assert "EK_TRANSPORT_CONFIG" not in frontend["environment"]
    assert not (output / "a-transport.toml").exists()
    benchmark = yaml.safe_load((output / "vllm-bench.yaml").read_text())
    assert benchmark["base-url"] == "http://attention:8000"
    worker = yaml.safe_load((output / "worker-pool1/workers/worker-00.yaml").read_text())
    assert worker["transport"] == {
        "type": "grpc",
        "max_pending_batches_per_device": 8,
        "listen": "0.0.0.0:51234",
        "advertise": "192.0.2.11:53234",
    }


@pytest.mark.parametrize(
    "settings",
    [
        {},
        {
            "max_workers": 3,
            "client_max_in_flight": 2,
            "max_registered_bytes": 1048576,
            "ascend_receive_fence": "device",
        },
    ],
)
def test_ascend_transport_generation_connects_frontend_and_worker_config(
    tmp_path: Path, settings: dict[str, object]
) -> None:
    cluster, experiment = _inputs(tmp_path)
    data = yaml.safe_load(cluster.read_text())
    data["transport"] = {"type": "transfer_engine", **settings}
    data["attention"]["vllm_port"] = 18765
    cluster.write_text(yaml.safe_dump(data))
    output = tmp_path / "generated"

    _generate(cluster, experiment, output)

    limits = {
        "max_workers": settings.get("max_workers", 2),
        "ascend_receive_fence": settings.get("ascend_receive_fence", "stream"),
    }
    if "max_registered_bytes" in settings:
        limits["max_registered_bytes"] = settings["max_registered_bytes"]
    frontend_config = tomllib.loads((output / "a-transport.toml").read_text())
    assert frontend_config == {
        "transfer_engine": {
            "segment_name": "192.0.2.10:0",
            "client_max_in_flight": settings.get("client_max_in_flight", 1),
            **limits,
        }
    }
    attention = yaml.safe_load((output / "compose.attention.yaml").read_text())
    frontend = attention["services"]["attention"]
    assert frontend["network_mode"] == "host"
    assert "networks" not in frontend
    assert "./a-transport.toml:/configs/a-transport.toml:ro" in frontend["volumes"]
    assert "/etc/hccn.conf:/etc/hccn.conf:ro" in frontend["volumes"]
    assert frontend["environment"]["EK_TRANSPORT_CONFIG"] == "/configs/a-transport.toml"
    assert "EK_TRANSPORT_CONFIG" not in attention["services"]["benchmark"]["environment"]
    assert attention["services"]["benchmark"]["networks"] == ["expert-kit-network"]
    benchmark = yaml.safe_load((output / "vllm-bench.yaml").read_text())
    assert benchmark["base-url"] == "http://192.0.2.10:18765"
    assert "18765/health" in frontend["healthcheck"]["test"][-1]
    serve = yaml.safe_load((output / "vllm-serve.yaml").read_text())
    assert serve["port"] == 18765
    attention_overlay = yaml.safe_load((output / "compose.attention.dev.yaml").read_text())
    assert attention_overlay["services"]["attention"] == {"network_mode": "host"}
    for pool, address in (("worker-pool1", "192.0.2.11"), ("worker-pool2", "192.0.2.12")):
        compose = yaml.safe_load((output / pool / "compose.expert.yaml").read_text())
        overlay = yaml.safe_load((output / pool / "compose.expert.dev.yaml").read_text())
        for index, (name, service) in enumerate(compose["services"].items()):
            worker = yaml.safe_load((output / pool / "workers" / f"{name}.yaml").read_text())
            assert worker["transport"] == {
                "type": "transfer_engine",
                "max_pending_batches_per_device": 8,
                "control_listen": f"0.0.0.0:{53234 + index}",
                "control_advertise": f"{address}:{53234 + index}",
                "segment_advertise": f"{address}:0",
                **limits,
            }
            assert worker["worker"]["max_active_batches_per_device"] == 8
            assert worker["weight_manager"]["peer"]["listen"] == f"0.0.0.0:{54234 + index}"
            assert service["network_mode"] == "host"
            assert "networks" not in service
            assert "/etc/hccn.conf:/etc/hccn.conf:ro" in service["volumes"]
            assert overlay["services"][name] == {"network_mode": "host"}


@pytest.mark.parametrize(
    ("settings", "error"),
    [
        ({"type": "rdma"}, "type"),
        ({"max_workers": 0}, "max_workers"),
        ({"client_max_in_flight": 0}, "client_max_in_flight"),
        ({"max_registered_bytes": 0}, "max_registered_bytes"),
        ({"ascend_receive_fence": "none"}, "ascend_receive_fence"),
        ({"protocol": "ascend_direct"}, "Extra inputs are not permitted"),
    ],
)
def test_transport_generation_rejects_unsupported_settings(
    tmp_path: Path, settings: dict[str, object], error: str
) -> None:
    cluster, experiment = _inputs(tmp_path)
    data = yaml.safe_load(cluster.read_text())
    data["transport"] = {"type": "transfer_engine", **settings}
    cluster.write_text(yaml.safe_dump(data))
    output = tmp_path / "generated"

    result = _generate(cluster, experiment, output, expect_success=False)

    assert result.returncode != 0
    assert error in result.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    "model_config",
    [
        "qwen3-30b-a3b",
        "deepseek-v3",
    ],
)
def test_random_generation_does_not_emit_torch_config(
    tmp_path: Path,
    model_config: str,
) -> None:
    cluster, experiment = _inputs(
        tmp_path,
        model_config=model_config,
        dataset_type="random",
    )
    output = tmp_path / "generated"

    _generate(cluster, experiment, output)

    assert not (output / "torch-bench.yaml").exists()


@pytest.mark.parametrize("model_config", ["qwen3-30b-a3b", "deepseek-v3"])
def test_tracing_defaults_off_in_every_generated_role(tmp_path: Path, model_config: str) -> None:
    cluster, experiment = _inputs(tmp_path, model_config=model_config)
    output = tmp_path / "generated"

    _generate(cluster, experiment, output)

    attention = yaml.safe_load((output / "compose.attention.yaml").read_text())
    serve = yaml.safe_load((output / "vllm-serve.yaml").read_text())
    assert "EK_TRACE_ENDPOINT" not in attention["services"]["attention"]["environment"]
    assert "EK_TRACE_SAMPLE_RATIO" not in attention["services"]["attention"]["environment"]
    assert "enforce-eager" not in serve
    workers = sorted(output.glob("worker-pool*/workers/worker-*.yaml"))
    assert workers
    assert all(
        yaml.safe_load(path.read_text())["observability"]["tracing"]["enabled"] is False
        for path in workers
    )


def test_tracing_enabled_reaches_attention_and_every_worker(tmp_path: Path) -> None:
    cluster, experiment = _inputs(tmp_path)
    data = yaml.safe_load(experiment.read_text())
    data["tracing"] = {
        "enabled": True,
        "endpoint": "http://192.0.2.40:4317",
        "sample_ratio": 0.25,
    }
    experiment.write_text(yaml.safe_dump(data))
    output = tmp_path / "generated"

    _generate(cluster, experiment, output)

    attention = yaml.safe_load((output / "compose.attention.yaml").read_text())
    env = attention["services"]["attention"]["environment"]
    assert env["EK_TRACE_ENDPOINT"] == "http://192.0.2.40:4317/"
    assert env["EK_TRACE_SAMPLE_RATIO"] == "0.25"
    assert "EK_TRACE_ENDPOINT" not in attention["services"]["benchmark"]["environment"]
    serve = yaml.safe_load((output / "vllm-serve.yaml").read_text())
    assert serve["enforce-eager"] is True
    workers = sorted(output.glob("worker-pool*/workers/worker-*.yaml"))
    assert len(workers) == 32
    for path in workers:
        tracing = yaml.safe_load(path.read_text())["observability"]["tracing"]
        assert tracing == {
            "enabled": True,
            "endpoint": "http://192.0.2.40:4317/",
            "sample_ratio": 0.25,
        }


def test_tracing_off_can_retain_endpoint_and_eager_mode(tmp_path: Path) -> None:
    cluster, experiment = _inputs(tmp_path)
    data = yaml.safe_load(experiment.read_text())
    data["tracing"] = {
        "enabled": False,
        "endpoint": "http://192.0.2.40:4317",
        "sample_ratio": 0.25,
    }
    data["serve"]["enforce_eager"] = True
    experiment.write_text(yaml.safe_dump(data))
    output = tmp_path / "generated"

    _generate(cluster, experiment, output)

    attention = yaml.safe_load((output / "compose.attention.yaml").read_text())
    assert "EK_TRACE_ENDPOINT" not in attention["services"]["attention"]["environment"]
    assert yaml.safe_load((output / "vllm-serve.yaml").read_text())["enforce-eager"] is True
    worker = yaml.safe_load((output / "worker-pool1/workers/worker-00.yaml").read_text())
    assert worker["observability"]["tracing"] == {"enabled": False}


@pytest.mark.parametrize(
    ("tracing", "error"),
    [
        ({"enabled": True}, "endpoint is required"),
        ({"enabled": True, "endpoint": "https://collector:4317"}, "plaintext HTTP"),
        ({"enabled": True, "endpoint": "http://collector:4317", "sample_ratio": 0}, "sample_ratio"),
        (
            {"enabled": True, "endpoint": "http://collector:4317", "sample_ratio": 1.1},
            "sample_ratio",
        ),
    ],
)
def test_tracing_rejects_invalid_enabled_config(
    tmp_path: Path, tracing: dict[str, object], error: str
) -> None:
    cluster, experiment = _inputs(tmp_path)
    data = yaml.safe_load(experiment.read_text())
    data["tracing"] = tracing
    experiment.write_text(yaml.safe_dump(data))
    output = tmp_path / "generated"

    result = _generate(cluster, experiment, output, expect_success=False)

    assert result.returncode != 0
    assert error in result.stderr
    assert not output.exists()
