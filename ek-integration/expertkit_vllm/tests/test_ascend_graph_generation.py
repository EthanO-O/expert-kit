"""Check generated graph choices, endpoint ownership and optional native settings."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
ASCEND = ROOT / "dev/ascend"
SHAPES = ("num_layers", "experts_per_layer", "hidden_dim", "intermediate_dim", "topk")


def _inputs(profile="qwen3-30b-a3b-w8a8"):
    example = ASCEND / "configs" / profile
    return tuple(
        yaml.safe_load((example / f"{kind}.example.yaml").read_text())
        for kind in ("cluster", "experiment")
    )


@pytest.fixture
def generator(monkeypatch):
    monkeypatch.syspath_prepend(str(ASCEND))
    from generate import TEMPLATE_DIR, generate_pools_config, generate_root_config
    from renderer import Renderer
    from schemas.cluster import ClusterConfig
    from schemas.context import TemplateContext
    from schemas.experiment import ExperimentConfig

    def render(output, cluster, experiment):
        context = TemplateContext.from_config(
            ClusterConfig.model_validate(cluster), ExperimentConfig.model_validate(experiment)
        )
        renderer = Renderer(TEMPLATE_DIR)
        generate_root_config(renderer, context, output)
        generate_pools_config(renderer, context, output)
        return context

    return render


def _read(output, name):
    return yaml.safe_load((output / name).read_text())


@pytest.mark.parametrize(
    "profile,count", (("qwen3-30b-a3b-w8a8", 2), ("deepseek-v4-flash-w8a8", 8))
)
def test_checked_in_examples_generate_frontend_options_and_local_worker_endpoints(
    tmp_path, profile, count
):
    from expertkit_transport.ascend_graph import AscendGraphOptions

    examples = ASCEND / "configs" / profile
    result = subprocess.run(
        [
            sys.executable,
            str(ASCEND / "cli.py"),
            "generate",
            "--cluster",
            str(examples / "cluster.example.yaml"),
            "--experiment",
            str(examples / "experiment.example.yaml"),
            "--output",
            str(tmp_path / "generated"),
        ],
        env={**os.environ, "PYTHONPATH": str(ASCEND)},
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    output = tmp_path / "generated"
    graph = json.loads((output / "ascend-decode-graph.json").read_text())
    AscendGraphOptions.model_validate(graph)
    assert set(graph) == {
        "mode",
        "capture_sizes",
        "timeout_seconds",
        "rendezvous_host",
        "rendezvous_port",
    }
    workers = sorted((output / "worker-pool1/workers").glob("*.yaml"))
    assert len(workers) == count
    for index, path in enumerate(workers):
        worker = yaml.safe_load(path.read_text())
        assert not set(SHAPES).intersection(worker["model"])
        assert "expert_intermediate_dim" not in worker["model"] and "top_k" not in worker["model"]
        assert "instance_id" not in worker["model"]
        assert worker["worker"]["device"] == "npu:0"
        assert worker["worker"]["ascend_decode_graph"] == {
            "control_listen": f"0.0.0.0:{18079 + index}",
            "control_advertise": f"192.0.2.10:{18079 + index}",
        }
    attention = _read(output, "compose.attention.yaml")["services"]["attention"]
    assert "EK_INSTANCE_ID" not in attention["environment"]
    assert (
        attention["environment"]["EK_ASCEND_DECODE_GRAPH_CONFIG"]
        == "/configs/ascend-decode-graph.json"
    )
    assert attention["network_mode"] == "host"
    assert attention["command"][-2:] == ["--no-async-scheduling", "--no-enable-prefix-caching"]
    assert "./ascend-decode-graph.json:/configs/ascend-decode-graph.json:ro" in attention["volumes"]
    admin = _read(output, "compose.control.yaml")["services"]["admin"]
    assert all("ascend-decode-graph" not in volume for volume in admin["volumes"])
    controller = _read(output, "controller.yaml")["inference"]
    assert "hidden_dim" not in controller and "intermediate_dim" not in controller
    serve = _read(output, "vllm-serve.yaml")
    assert serve["compilation-config"]["cudagraph_mode"] == "FULL_DECODE_ONLY"
    assert serve["quantization"] == "ascend"
    assert "max-num-seqs" not in serve and "max-num-batched-tokens" not in serve
    assert "block-size" not in serve and "num-gpu-blocks-override" not in serve
    if profile.startswith("deepseek"):
        assert serve["distributed-executor-backend"] == "mp"
        assert serve["load-format"] == "safetensors"


@pytest.mark.parametrize("mode", ("decode", "full"))
def test_exact_captures_and_optional_native_flags_are_not_derived_from_graphs(
    tmp_path, generator, mode
):
    cluster, experiment = _inputs()
    experiment["ascend_decode_graph"].update(mode=mode, capture_sizes=[7, 3, 7])
    values = dict(
        max_num_seqs=13,
        max_num_batched_tokens=96,
        max_num_scheduled_tokens=23,
        block_size=32,
        num_gpu_blocks_override=113,
        kv_cache_memory_bytes=123456789,
        enable_chunked_prefill=False,
    )
    experiment["serve"].update(values)
    generator(tmp_path, cluster, experiment)
    graph = json.loads((tmp_path / "ascend-decode-graph.json").read_text())
    assert graph["capture_sizes"] == [3, 7]
    serve = _read(tmp_path, "vllm-serve.yaml")
    assert serve["compilation-config"]["cudagraph_capture_sizes"] == [3, 7]
    assert serve["compilation-config"]["cudagraph_mode"] == (
        "FULL" if mode == "full" else "FULL_DECODE_ONLY"
    )
    for field, value in values.items():
        assert serve[field.replace("_", "-")] == value
    assert serve["enable-chunked-prefill"] is False


@pytest.mark.parametrize("explicit", (False, True))
def test_generated_model_facts_are_optional_assertions_resolved_by_weight_server(
    tmp_path, generator, monkeypatch, explicit
):
    from expertkit_worker.weights import metadata as module
    from expertkit_worker.weights.metadata import (
        ModelMetadata,
        QuantizationMetadata,
        resolve_worker_config,
    )

    cluster, experiment = _inputs()
    for field in ("expert_compute", "swiglu_limit", "quantization"):
        experiment["model"].pop(field, None)
    if explicit:
        experiment["model"].update(
            num_layers=48,
            experts_per_layer=128,
            hidden_dim=2048,
            intermediate_dim=768,
            topk=8,
            expert_compute="swiglu",
            swiglu_limit=0.0,
        )
    generator(tmp_path, cluster, experiment)
    document = _read(tmp_path, "worker-pool1/workers/worker-00.yaml")
    for field in ("expert_compute", "swiglu_limit"):
        assert (field in document["model"]) == explicit
    assert ("hidden_dim" in document["model"]) == explicit
    controller = _read(tmp_path, "controller.yaml")["inference"]
    assert ("hidden_dim" in controller) == explicit

    async def fetch(endpoint, name):
        assert (
            endpoint.rstrip("/") == "http://192.0.2.10:16543"
            and name == experiment["model"]["name"]
        )
        return ModelMetadata(
            schema_version=1,
            model_type="qwen3_moe",
            num_layers=48,
            moe_layer_start=0,
            moe_layer_end=48,
            experts_per_layer=128,
            hidden_dim=2048,
            expert_intermediate_dim=768,
            top_k=8,
            activation_dtype="bf16",
            quantization=QuantizationMetadata("modelslim-w8a8-dynamic", 8, None, True, None),
        )

    monkeypatch.setattr(module, "fetch_model_metadata", fetch)
    resolved = asyncio.run(resolve_worker_config(document))
    assert resolved.config.model.hidden_dim == 2048
    assert resolved.config.model.top_k == 8
    document["model"]["hidden_dim"] = 4096
    with pytest.raises(ValueError, match="hidden_dim conflicts"):
        asyncio.run(resolve_worker_config(document))


@pytest.mark.parametrize("cross_host", (False, True))
def test_multiple_user_selected_pools_do_not_declare_or_restrict_placement(
    tmp_path, generator, cross_host
):
    cluster, experiment = _inputs()
    first = cluster["pools"][0]
    first["devices"] = [1]
    other = copy.deepcopy(first)
    other.update(id="second-pool", devices=[1] if cross_host else [2])
    if cross_host:
        cluster["nodes"]["remote"] = {"address": "192.0.2.11"}
        other["node"] = "remote"
        experiment["ascend_decode_graph"]["rendezvous_host"] = "192.0.2.11"
    else:
        for field in ("start_worker_port", "start_peer_port", "start_graph_control_port"):
            other[field] += 10
    cluster["pools"].append(other)
    generator(tmp_path, cluster, experiment)
    worker = _read(tmp_path, "second-pool/workers/worker-00.yaml")
    address = "192.0.2.11" if cross_host else "192.0.2.10"
    assert (
        worker["worker"]["ascend_decode_graph"]["control_advertise"]
        == f"{address}:{other['start_graph_control_port']}"
    )
    assert set(worker["worker"]["ascend_decode_graph"]) == {"control_listen", "control_advertise"}


@pytest.mark.parametrize(
    "case",
    (
        "legacy",
        "missing-port",
        "graph-transport",
        "peer-transport",
        "rendezvous",
        "device",
        "two-frontends",
        "eager",
        "tracing",
        "active",
        "capacity",
        "port-overflow",
    ),
)
def test_known_deployment_conflicts_fail_before_rendering(tmp_path, generator, case):
    cluster, experiment = _inputs()
    pool = cluster["pools"][0]
    if case == "legacy":
        experiment["ascend_decode_graph"]["workers"] = []
    elif case == "missing-port":
        pool.pop("start_graph_control_port")
    elif case == "graph-transport":
        pool["start_graph_control_port"] = pool["start_worker_port"]
    elif case == "peer-transport":
        pool["start_peer_port"] = pool["start_worker_port"]
    elif case == "rendezvous":
        experiment["ascend_decode_graph"]["rendezvous_port"] = pool["start_graph_control_port"]
    elif case == "device":
        pool["devices"] = [1, 1]
    elif case == "two-frontends":
        cluster["attention"]["devices"] = [0, 3]
    elif case == "eager":
        experiment["serve"]["enforce_eager"] = True
    elif case == "tracing":
        experiment["tracing"] = {"enabled": True, "endpoint": "http://192.0.2.10:4317"}
    elif case == "active":
        cluster["expert"]["runtime"]["max_active_batches_per_device"] = 2
    elif case == "capacity":
        experiment["ascend_decode_graph"]["capture_sizes"] = [8192]
    else:
        pool["start_graph_control_port"] = 65535
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="Legacy Ascend" if case == "legacy" else None):
        generator(output, cluster, experiment)
    assert not output.exists()


def test_explicit_rendezvous_host_is_not_reassigned_or_rejected_offline(tmp_path, generator):
    cluster, experiment = _inputs()
    experiment["ascend_decode_graph"]["rendezvous_host"] = "192.0.2.99"
    generator(tmp_path, cluster, experiment)
    options = json.loads((tmp_path / "ascend-decode-graph.json").read_text())
    assert options["rendezvous_host"] == "192.0.2.99"


def test_same_host_pool_collisions_are_checked_beyond_the_first_pool(tmp_path, generator):
    cluster, experiment = _inputs()
    second = copy.deepcopy(cluster["pools"][0])
    second.update(id="second", devices=[3])
    cluster["pools"].append(second)
    with pytest.raises(ValueError, match="Ports"):
        generator(tmp_path / "output", cluster, experiment)


def test_ordinary_generation_has_no_graph_document_or_endpoint(tmp_path, generator):
    cluster, experiment = _inputs()
    experiment.pop("ascend_decode_graph")
    generator(tmp_path, cluster, experiment)
    assert not (tmp_path / "ascend-decode-graph.json").exists()
    worker = _read(tmp_path, "worker-pool1/workers/worker-00.yaml")
    assert "ascend_decode_graph" not in worker["worker"]
    assert "compilation-config" not in _read(tmp_path, "vllm-serve.yaml")


def test_graph_compose_preserves_native_image_network_and_separate_data_ports(tmp_path, generator):
    cluster, experiment = _inputs()
    generator(tmp_path, cluster, experiment)
    attention = _read(tmp_path, "compose.attention.yaml")["services"]
    expert = _read(tmp_path, "worker-pool1/compose.expert.yaml")["services"]["worker-00"]
    assert expert["image"] == attention["attention"]["image"]
    assert expert["entrypoint"] == ["ek-worker"] and expert["network_mode"] == "host"
    assert attention["benchmark"]["network_mode"] == "host"
    assert _read(tmp_path, "compose.attention.dev.yaml")["services"]["attention"] == {
        "network_mode": "host"
    }
    builds = _read(tmp_path, "compose.build.yaml")["services"]
    for name in ("worker-image", "attention-image"):
        assert builds[name]["build"]["target"] == "ek-ascend-runtime"
        assert builds[name]["build"]["args"] == {
            "VLLM_ASCEND_IMAGE": cluster["images"]["vllm_ascend"]
        }
    worker = _read(tmp_path, "worker-pool1/workers/worker-00.yaml")
    assert worker["transport"]["listen"] == "0.0.0.0:53234"
    assert worker["weight_manager"]["peer"]["listen"] == "0.0.0.0:54234"


def test_graph_benchmark_options_still_reach_native_cli(tmp_path, generator):
    import runpy

    cluster, experiment = _inputs()
    experiment["run"].update(
        save_detailed=True,
        percentile_metrics="ttft,tpot,itl,e2el",
        metric_percentiles="50,99",
        ready_check_timeout_sec=0,
    )
    generator(tmp_path, cluster, experiment)
    to_cmd = runpy.run_path(str(ASCEND / "bench_launcher.py"))["to_cmd"]
    command = to_cmd(tmp_path / "vllm-bench.yaml")
    assert "--save-detailed" in command
    for flag, value in (
        ("--percentile-metrics", "ttft,tpot,itl,e2el"),
        ("--metric-percentiles", "50,99"),
        ("--ready-check-timeout-sec", "0"),
    ):
        assert command[command.index(flag) + 1] == value


def test_v4_omitted_clipping_assertion_is_not_rendered_as_zero(tmp_path, generator):
    cluster, experiment = _inputs("deepseek-v4-flash-w8a8")
    experiment["model"].pop("swiglu_limit")
    generator(tmp_path, cluster, experiment)
    model = _read(tmp_path, "worker-pool1/workers/worker-00.yaml")["model"]
    assert model["expert_compute"] == "deepseek_v4"
    assert "swiglu_limit" not in model


@pytest.mark.parametrize(
    "field,value",
    (
        ("max_num_seqs", True),
        ("max_num_batched_tokens", 1.5),
        ("max_num_scheduled_tokens", "23"),
        ("block_size", False),
        ("num_gpu_blocks_override", 13.0),
        ("kv_cache_memory_bytes", "1G"),
        ("enable_chunked_prefill", 1),
    ),
)
def test_native_optional_settings_reject_types_without_coercion(tmp_path, generator, field, value):
    cluster, experiment = _inputs()
    experiment["serve"][field] = value
    with pytest.raises(ValueError):
        generator(tmp_path / "output", cluster, experiment)
    assert not (tmp_path / "output").exists()
