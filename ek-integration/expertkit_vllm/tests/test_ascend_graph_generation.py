"""Validate the public fixed Worker graph deployment materialization."""

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[3]
ASCEND = ROOT / "dev/ascend"


def _inputs(tmp_path):
    examples = ASCEND / "configs/qwen3-30b-a3b"
    cluster = yaml.safe_load((examples / "cluster.example.yaml").read_text())
    experiment = yaml.safe_load((examples / "experiment.example.yaml").read_text())
    cluster["nodes"]["node-a"]["address"] = "192.0.2.10"
    cluster["attention"]["devices"] = [0]
    cluster["pools"] = [{**cluster["pools"][0], "node": "node-a", "devices": [1]}]
    cluster["expert"]["runtime"]["max_active_batches_per_device"] = 1
    experiment["model"].update(
        name="DeepSeek-V2-Lite-Chat",
        weight_version="85864749",
        num_layers=27,
        experts_per_layer=64,
        hidden_dim=2048,
        intermediate_dim=1408,
        topk=6,
        weight_dtype="bf16",
        activation_dtype="bf16",
    )
    experiment["ascend_decode_graph"] = {
        "session_id": "deepseek-resident-v1",
        "instance_id": 7,
        "worker_id": "worker-pool1-worker-00",
        "weight_version": "85864749",
        "control_endpoint": "192.0.2.10:18079",
        "rendezvous_host": "192.0.2.10",
        "rendezvous_port": 18080,
        "layer_ids": list(range(1, 27)),
        "capture_sizes": [1, 32, 64],
    }
    return cluster, experiment


def _generate(tmp_path, cluster, experiment):
    cluster_path, experiment_path = tmp_path / "cluster.yaml", tmp_path / "experiment.yaml"
    cluster_path.write_text(yaml.safe_dump(cluster))
    experiment_path.write_text(yaml.safe_dump(experiment))
    return subprocess.run(
        [
            sys.executable,
            str(ASCEND / "cli.py"),
            "generate",
            "--cluster",
            str(cluster_path),
            "--experiment",
            str(experiment_path),
            "--output",
            str(tmp_path / "generated"),
        ],
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                (str(ASCEND), str(ROOT / "ek-transport/src"), os.environ.get("PYTHONPATH", ""))
            ),
        },
        text=True,
        capture_output=True,
        timeout=30,
    )


def _v4_inputs(tmp_path):
    cluster, experiment = _inputs(tmp_path)
    cluster["pools"][0]["devices"] = list(range(1, 9))
    experiment["model"].update(
        name="DeepSeek-V4-Flash-0731-w8a8",
        weight_version="0731",
        num_layers=43,
        experts_per_layer=256,
        hidden_dim=4096,
        intermediate_dim=2048,
        topk=6,
        expert_compute="deepseek_v4",
        swiglu_limit=10.0,
        quantization={
            "type": "modelslim-w8a8-dynamic",
            "bits": 8,
            "group_size": None,
            "symmetric": True,
        },
    )
    graph = experiment["ascend_decode_graph"]
    del graph["worker_id"], graph["control_endpoint"]
    graph.update(
        weight_version="0731",
        layer_ids=list(range(43)),
        rendezvous_port=18090,
        workers=[
            {
                "worker_id": f"worker-pool1-worker-{index:02d}",
                "control_endpoint": f"192.0.2.10:{18079 + index}",
                "expert_start": index * 32,
                "expert_end": (index + 1) * 32,
            }
            for index in range(8)
        ],
    )
    return cluster, experiment


@pytest.mark.parametrize(
    ("profile", "worker_count", "expert_count"),
    [("qwen3-30b-a3b-w8a8", 2, 6144), ("deepseek-v4-flash-w8a8", 8, 11008)],
)
def test_checked_in_modelslim_graph_examples_generate_complete_deployments(
    tmp_path, profile, worker_count, expert_count
):
    from expertkit_worker.config import WorkerConfig

    examples = ASCEND / "configs" / profile
    cluster = yaml.safe_load((examples / "cluster.example.yaml").read_text())
    experiment = yaml.safe_load((examples / "experiment.example.yaml").read_text())
    result = _generate(tmp_path, cluster, experiment)
    assert result.returncode == 0, result.stderr
    output = tmp_path / "generated"
    graph = json.loads((output / "ascend-decode-graph.json").read_text())
    worker_paths = sorted((output / "worker-pool1/workers").glob("*.yaml"))
    assert len(worker_paths) == worker_count == len(graph["workers"])
    assert len(graph["layer_ids"]) * graph["model"]["experts_per_layer"] == expert_count
    serve = yaml.safe_load((output / "vllm-serve.yaml").read_text())
    assert serve["compilation-config"]["cudagraph_mode"] == "FULL_DECODE_ONLY"
    assert "enforce-eager" not in serve
    attention = yaml.safe_load((output / "compose.attention.yaml").read_text())["services"][
        "attention"
    ]
    assert "TORCHINDUCTOR_AUTOGRAD_CACHE" not in attention["environment"]
    assert "VLLM_USE_AOT_COMPILE" not in attention["environment"]
    for worker_path, participant in zip(worker_paths, graph["workers"], strict=True):
        worker = WorkerConfig.model_validate(yaml.safe_load(worker_path.read_text()))
        assert worker.worker.id == participant["worker_id"]
        assert worker.worker.backend.value == "ascend_fused"
        assert worker.worker.ascend_decode_graph.model_dump(mode="json") == graph
        assert worker.model.quantization.model_dump(mode="json") == graph["model"]["quantization"]


def test_v4_generation_produces_shared_manifest_and_eight_valid_workers(tmp_path):
    from expertkit_worker.config import WorkerConfig

    cluster, experiment = _v4_inputs(tmp_path)
    result = _generate(tmp_path, cluster, experiment)
    assert result.returncode == 0, result.stderr
    output = tmp_path / "generated"
    shared = json.loads((output / "ascend-decode-graph.json").read_text())
    assert shared["model"]["expert_compute"] == "deepseek_v4"
    assert shared["model"]["swiglu_limit"] == 10.0
    assert shared["model"]["quantization"] == experiment["model"]["quantization"]
    assert shared["layer_ids"] == list(range(43))
    configs = sorted((output / "worker-pool1/workers").glob("*.yaml"))
    assert len(configs) == 8
    for index, path in enumerate(configs):
        worker = WorkerConfig.model_validate(yaml.safe_load(path.read_text()))
        assert worker.model.weight_version == "0731"
        assert worker.model.expert_compute == "deepseek_v4"
        assert worker.model.swiglu_limit == 10.0
        assert worker.model.quantization.model_dump(mode="json") == shared["model"]["quantization"]
        assert worker.worker.ascend_decode_graph.model_dump(mode="json") == shared
        assert worker.worker.id == shared["workers"][index]["worker_id"]
        assert (
            shared["workers"][index]["expert_start"],
            shared["workers"][index]["expert_end"],
        ) == (
            index * 32,
            (index + 1) * 32,
        )
    serve = yaml.safe_load((output / "vllm-serve.yaml").read_text())
    assert serve["data-parallel-size"] == 1
    assert serve["quantization"] == "ascend"
    assert serve["dtype"] == "bfloat16"
    assert serve["trust-remote-code"] is False
    assert serve["load-format"] == "safetensors"
    assert serve["safetensors-load-strategy"] == "lazy"
    assert serve["distributed-executor-backend"] == "mp"
    assert serve["max-model-len"] == experiment["serve"]["max_model_len"]
    assert serve["max-num-seqs"] == 64
    assert serve["max-num-batched-tokens"] == 2048
    assert serve["compilation-config"]["mode"] == "VLLM_COMPILE"
    assert serve["compilation-config"]["cudagraph_mode"] == "FULL_DECODE_ONLY"
    assert serve["compilation-config"]["cudagraph_capture_sizes"] == [1, 32, 64]
    attention = yaml.safe_load((output / "compose.attention.yaml").read_text())["services"][
        "attention"
    ]
    assert {"ascend", "ascend_model", "register_expertkit"} <= set(
        attention["environment"]["VLLM_PLUGINS"].split(",")
    )


@pytest.mark.parametrize("case", ["wrong-limit", "missing-quantization", "partition-gap"])
def test_v4_generation_rejects_unsupported_computation_or_partition(tmp_path, case):
    cluster, experiment = _v4_inputs(tmp_path)
    if case == "wrong-limit":
        experiment["model"]["swiglu_limit"] = 9.0
    elif case == "missing-quantization":
        del experiment["model"]["quantization"]
    else:
        experiment["ascend_decode_graph"]["workers"][1]["expert_start"] = 33
    result = _generate(tmp_path, cluster, experiment)
    assert result.returncode != 0
    assert not (tmp_path / "generated").exists()


def test_v4_ordinary_generation_keeps_native_modelslim_loader(tmp_path):
    cluster, experiment = _v4_inputs(tmp_path)
    del experiment["ascend_decode_graph"]
    result = _generate(tmp_path, cluster, experiment)
    assert result.returncode == 0, result.stderr
    output = tmp_path / "generated"
    serve = yaml.safe_load((output / "vllm-serve.yaml").read_text())
    assert serve["quantization"] == "ascend"
    assert serve["load-format"] == "safetensors"
    assert serve["safetensors-load-strategy"] == "lazy"
    assert serve["distributed-executor-backend"] == "mp"
    assert "compilation-config" not in serve
    worker = yaml.safe_load((output / "worker-pool1/workers/worker-00.yaml").read_text())
    assert worker["model"]["quantization"] == experiment["model"]["quantization"]
    assert "ascend_decode_graph" not in worker["worker"]
    compose = yaml.safe_load((output / "worker-pool1/compose.expert.yaml").read_text())
    assert compose["services"]["worker-00"]["entrypoint"] == ["ek-worker"]


def test_graph_generation_reports_missing_model_as_validation_error(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ASCEND))
    from schemas.experiment import ExperimentConfig

    _, experiment = _inputs(tmp_path)
    del experiment["model"]
    with pytest.raises(ValidationError) as error:
        ExperimentConfig.model_validate(experiment)
    assert any(
        row["loc"] == ("model",) and row["type"] == "missing" for row in error.value.errors()
    )


def test_graph_generation_connects_native_full_and_worker(tmp_path):
    cluster, experiment = _inputs(tmp_path)
    result = _generate(tmp_path, cluster, experiment)
    assert result.returncode == 0, result.stderr
    output = tmp_path / "generated"
    graph = json.loads((output / "ascend-decode-graph.json").read_text())
    worker = yaml.safe_load((output / "worker-pool1/workers/worker-00.yaml").read_text())
    assert worker["worker"]["ascend_decode_graph"] == graph
    assert worker["model"]["instance_id"] == graph["instance_id"] == 7
    assert worker["transport"]["listen"] == "0.0.0.0:53234"
    assert worker["weight_manager"]["peer"]["listen"] == "0.0.0.0:54234"
    attention = yaml.safe_load((output / "compose.attention.yaml").read_text())["services"][
        "attention"
    ]
    assert attention["network_mode"] == "host" and "networks" not in attention
    assert attention["environment"]["EK_INSTANCE_ID"] == "7"
    assert attention["command"][-2:] == ["--no-async-scheduling", "--no-enable-prefix-caching"]
    assert (
        attention["environment"]["EK_ASCEND_DECODE_GRAPH_CONFIG"]
        == "/configs/ascend-decode-graph.json"
    )
    assert "EK_TRANSPORT_CONFIG" not in attention["environment"]
    assert "./ascend-decode-graph.json:/configs/ascend-decode-graph.json:ro" in attention["volumes"]
    expert = yaml.safe_load((output / "worker-pool1/compose.expert.yaml").read_text())["services"][
        "worker-00"
    ]
    assert expert["network_mode"] == "host" and "networks" not in expert
    assert expert["entrypoint"] == ["ek-worker"]
    assert expert["image"] == attention["image"]
    builds = yaml.safe_load((output / "compose.build.yaml").read_text())["services"]
    for service in ("worker-image", "attention-image"):
        assert builds[service]["image"] == attention["image"]
        assert builds[service]["build"]["target"] == "ek-ascend-runtime"
        assert builds[service]["build"]["args"] == {
            "VLLM_ASCEND_IMAGE": cluster["images"]["vllm_ascend"]
        }
    benchmark = yaml.safe_load((output / "compose.attention.yaml").read_text())["services"][
        "benchmark"
    ]
    assert benchmark["network_mode"] == "host" and "networks" not in benchmark
    admin = yaml.safe_load((output / "compose.control.yaml").read_text())["services"]["admin"]
    assert (
        "./ascend-decode-graph.json:/etc/expert-kit/ascend-decode-graph.json:ro" in admin["volumes"]
    )
    bench = yaml.safe_load((output / "vllm-bench.yaml").read_text())
    assert "save-detailed" not in bench
    assert "metric-percentiles" not in bench
    serve = yaml.safe_load((output / "vllm-serve.yaml").read_text())
    assert serve["async-scheduling"] is False
    assert serve["max-num-seqs"] == 64
    assert serve["compilation-config"]["cudagraph_mode"] == "FULL_DECODE_ONLY"
    assert serve["compilation-config"]["cudagraph_capture_sizes"] == [1, 32, 64]
    assert "enforce-eager" not in serve
    assert yaml.safe_load((output / "compose.attention.dev.yaml").read_text())["services"][
        "attention"
    ] == {"network_mode": "host"}


def test_graph_benchmark_options_reach_native_cli(tmp_path):
    cluster, experiment = _inputs(tmp_path)
    experiment["run"].update(
        save_detailed=True,
        percentile_metrics="ttft,tpot,itl,e2el",
        metric_percentiles="50,99",
        ready_check_timeout_sec=0,
    )
    result = _generate(tmp_path, cluster, experiment)
    assert result.returncode == 0, result.stderr
    to_cmd = runpy.run_path(str(ASCEND / "bench_launcher.py"))["to_cmd"]
    command = to_cmd(tmp_path / "generated/vllm-bench.yaml")
    assert "--save-detailed" in command
    for flag, value in (
        ("--percentile-metrics", "ttft,tpot,itl,e2el"),
        ("--metric-percentiles", "50,99"),
        ("--ready-check-timeout-sec", "0"),
    ):
        assert command[command.index(flag) + 1] == value


@pytest.mark.parametrize(
    "case",
    [
        "two-frontends",
        "two-workers",
        "cross-host",
        "wrong-model",
        "dense-layer",
        "wrong-worker",
        "wrong-host",
        "port-collision",
        "eager",
        "active-two",
    ],
)
def test_graph_generation_rejects_unsupported_deployments(tmp_path, case):
    cluster, experiment = _inputs(tmp_path)
    graph = experiment["ascend_decode_graph"]
    if case == "two-frontends":
        cluster["attention"]["devices"] = [0, 2]
    elif case == "two-workers":
        cluster["pools"][0]["devices"] = [1, 2]
    elif case == "cross-host":
        cluster["pools"][0]["node"] = "node-b"
    elif case == "wrong-model":
        experiment["model"]["topk"] = 8
    elif case == "dense-layer":
        graph["layer_ids"] = list(range(27))
    elif case == "wrong-worker":
        graph["worker_id"] = "another-worker"
    elif case == "wrong-host":
        graph["rendezvous_host"] = "192.0.2.11"
    elif case == "port-collision":
        graph["rendezvous_port"] = 53234
    elif case == "eager":
        experiment["serve"]["enforce_eager"] = True
    else:
        cluster["expert"]["runtime"]["max_active_batches_per_device"] = 2
    result = _generate(tmp_path, cluster, experiment)
    assert result.returncode != 0
    assert not (tmp_path / "generated").exists()


def test_two_worker_generation_shares_rank_manifest_and_checks_all_ports(tmp_path):
    cluster, experiment = _inputs(tmp_path)
    cluster["pools"][0]["devices"] = [1, 2]
    graph = experiment["ascend_decode_graph"]
    del graph["worker_id"], graph["control_endpoint"]
    graph["workers"] = [
        {
            "worker_id": f"worker-pool1-worker-{index:02d}",
            "control_endpoint": f"192.0.2.10:{18079 + index}",
            "expert_start": index * 32,
            "expert_end": (index + 1) * 32,
        }
        for index in range(2)
    ]
    graph["rendezvous_port"] = 18081
    result = _generate(tmp_path, cluster, experiment)
    assert result.returncode == 0, result.stderr
    output = tmp_path / "generated"
    shared = json.loads((output / "ascend-decode-graph.json").read_text())
    for index in range(2):
        worker = yaml.safe_load(
            (output / f"worker-pool1/workers/worker-{index:02d}.yaml").read_text()
        )
        assert worker["worker"]["ascend_decode_graph"] == shared
        assert worker["worker"]["device"] == "npu:0"
        assert worker["worker"]["id"] == shared["workers"][index]["worker_id"]
    graph["workers"][1]["control_endpoint"] = "192.0.2.10:53235"
    # A fresh output directory makes the rejected render independent of the first.
    other = tmp_path / "collision"
    other.mkdir()
    rejected = _generate(other, cluster, experiment)
    assert rejected.returncode != 0
    assert not (other / "generated").exists()


@pytest.mark.parametrize("explicit", [False, True])
def test_qwen_generation_derives_model_before_expert_interval_validation(tmp_path, explicit):
    cluster, experiment = _inputs(tmp_path)
    experiment["model"].update(
        name="Qwen3-30B-A3B-Instruct-2507",
        num_layers=48,
        experts_per_layer=128,
        intermediate_dim=768,
        topk=8,
    )
    cluster["pools"][0]["devices"] = [1, 2]
    graph = experiment["ascend_decode_graph"]
    del graph["worker_id"], graph["control_endpoint"]
    graph["layer_ids"] = list(range(48))
    graph["rendezvous_port"] = 18081
    graph["workers"] = [
        {
            "worker_id": f"worker-pool1-worker-{index:02d}",
            "control_endpoint": f"192.0.2.10:{18079 + index}",
            "expert_start": index * 64,
            "expert_end": (index + 1) * 64,
        }
        for index in range(2)
    ]
    declaration = {
        "num_layers": 48,
        "experts_per_layer": 128,
        "hidden_dim": 2048,
        "expert_intermediate_dim": 768,
        "top_k": 8,
        "activation_dtype": "bf16",
        "weight_dtype": "bf16",
        "activation": "silu",
        "expert_compute": "swiglu",
        "swiglu_limit": 0.0,
        "quantization": None,
    }
    if explicit:
        graph["model"] = declaration.copy()
    result = _generate(tmp_path, cluster, experiment)
    assert result.returncode == 0, result.stderr
    shared = json.loads((tmp_path / "generated/ascend-decode-graph.json").read_text())
    assert shared["model"] == declaration
    serve = yaml.safe_load((tmp_path / "generated/vllm-serve.yaml").read_text())
    assert "distributed-executor-backend" not in serve
    for index in range(2):
        worker = yaml.safe_load(
            (tmp_path / f"generated/worker-pool1/workers/worker-{index:02d}.yaml").read_text()
        )
        assert worker["worker"]["ascend_decode_graph"] == shared
        assert worker["model"]["top_k"] == 8
    conflicting = tmp_path / "conflicting"
    conflicting.mkdir()
    graph["model"] = {**declaration, "hidden_dim": 4096}
    rejected = _generate(conflicting, cluster, experiment)
    assert rejected.returncode != 0
    assert "model declaration" in rejected.stderr


@pytest.mark.parametrize("case", ["duplicate-workers", "multiple-pools"])
def test_graph_generation_rejects_shared_worker_devices_and_multiple_pools(tmp_path, case):
    cluster, experiment = _inputs(tmp_path)
    cluster["pools"][0]["devices"] = [1, 1]
    graph = experiment["ascend_decode_graph"]
    del graph["worker_id"], graph["control_endpoint"]
    graph["workers"] = [
        {
            "worker_id": f"worker-pool1-worker-{index:02d}",
            "control_endpoint": f"192.0.2.10:{18079 + index}",
            "expert_start": index * 32,
            "expert_end": (index + 1) * 32,
        }
        for index in range(2)
    ]
    graph["rendezvous_port"] = 18081
    if case == "multiple-pools":
        cluster["pools"][0]["devices"] = [1]
        cluster["pools"].append({**cluster["pools"][0], "id": "worker-pool2", "devices": [2]})
        graph["workers"][1]["worker_id"] = "worker-pool2-worker-00"
    result = _generate(tmp_path, cluster, experiment)
    assert result.returncode != 0
    assert not (tmp_path / "generated").exists()
