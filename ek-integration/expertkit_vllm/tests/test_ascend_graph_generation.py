"""Validate the public fixed one/two-Worker graph deployment materialization."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

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
        env={**os.environ, "PYTHONPATH": str(ASCEND)},
        text=True,
        capture_output=True,
        timeout=30,
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
    benchmark = yaml.safe_load((output / "compose.attention.yaml").read_text())["services"][
        "benchmark"
    ]
    assert benchmark["network_mode"] == "host" and "networks" not in benchmark
    serve = yaml.safe_load((output / "vllm-serve.yaml").read_text())
    assert serve["async-scheduling"] is False
    assert serve["max-num-seqs"] == 64
    assert serve["compilation-config"]["cudagraph_mode"] == "FULL_DECODE_ONLY"
    assert serve["compilation-config"]["cudagraph_capture_sizes"] == [1, 32, 64]
    assert "enforce-eager" not in serve
    assert yaml.safe_load((output / "compose.attention.dev.yaml").read_text())["services"][
        "attention"
    ] == {"network_mode": "host"}


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
