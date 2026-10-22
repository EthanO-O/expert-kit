import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def benchmark(monkeypatch):
    path = Path(__file__).parents[3] / "benchmarks" / "bench_backend.py"
    spec = importlib.util.spec_from_file_location("test_bench_backend_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("distribution", ["balanced", "uniform", "hot"])
def test_backend_benchmark_runs_torch_cpu_smoke(distribution: str) -> None:
    worker_root = Path(__file__).parents[3]
    completed = subprocess.run(
        [
            sys.executable,
            str(worker_root / "benchmarks" / "bench_backend.py"),
            "--backend",
            "torch",
            "--device",
            "cpu",
            "--dtype",
            "fp32",
            "--tokens",
            "4",
            "--distribution",
            distribution,
            "--hidden-dim",
            "8",
            "--intermediate-dim",
            "12",
            "--top-k",
            "2",
            "--experts",
            "4",
            "--concurrency",
            "2",
            "--routing-cases",
            "3",
            "--warmup",
            "0",
            "--repeats",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    result = json.loads(completed.stdout)

    assert result["correctness"]["checked"] is True
    assert result["correctness"]["criterion"] == "allclose"
    assert result["correctness"]["max_abs_error"] == 0
    assert result["correctness"]["max_relative_l2"] == 0
    assert result["correctness"]["min_cosine_similarity"] > 0.99999
    assert result["execution"]["concurrency"] == 2
    assert result["execution"]["profile"] is False
    assert result["routing"]["distribution"] == distribution
    assert result["routing"]["cases"] == 3
    assert 1 <= result["routing"]["unique_lease_keys"] <= 6
    assert result["latency_ms"]["mean"] > 0
    assert result["throughput"]["tokens_per_second"] > 0


@pytest.mark.parametrize("distribution", ["balanced", "uniform", "hot"])
def test_id_generators_are_seeded_and_preserve_distinct_topk(benchmark, distribution):
    args = benchmark.BenchmarkOptions(
        tokens=257,
        top_k=3,
        experts=19,
        distribution=benchmark.Distribution(distribution),
    )
    generate = benchmark.ID_GENERATORS[args.distribution]
    ids = generate(args, torch.Generator().manual_seed(47))
    repeated = generate(args, torch.Generator().manual_seed(47))
    assert torch.equal(ids, repeated)
    assert ids.shape == (257, 3)
    assert ids.dtype == torch.int32
    assert torch.all((ids >= 0) & (ids < 19))
    assert torch.all(ids.sort(dim=1).values.diff(dim=1) > 0)


def test_balanced_generator_has_even_assignment_counts(benchmark):
    args = benchmark.BenchmarkOptions(tokens=7, top_k=3, experts=5)
    ids = benchmark.ID_GENERATORS[args.distribution](args, torch.Generator())
    counts = torch.bincount(ids.flatten(), minlength=5)
    assert int(counts.max() - counts.min()) <= 1


def test_hot_generator_concentrates_on_configured_subset(benchmark):
    args = benchmark.BenchmarkOptions(
        tokens=2048,
        top_k=2,
        experts=32,
        distribution=benchmark.Distribution.HOT,
        hot_experts=2,
        hot_multiplier=50,
    )
    ids = benchmark.ID_GENERATORS[args.distribution](args, torch.Generator().manual_seed(47))
    counts = torch.bincount(ids.flatten(), minlength=32)
    assert int(counts.topk(2).values.sum()) > ids.numel() * 0.7


def test_routing_cases_are_seeded_and_vary_independently_from_concurrency(benchmark):
    args = benchmark.BenchmarkOptions(
        tokens=4,
        top_k=2,
        experts=32,
        concurrency=2,
        routing_cases=3,
        distribution=benchmark.Distribution.HOT,
    )
    batches = [
        benchmark._make_batch(
            args,
            dtype=torch.float32,
            device=torch.device("cpu"),
            slot=slot,
            routing_case=routing_case,
        )
        for routing_case in range(args.routing_cases)
        for slot in range(args.concurrency)
    ]
    repeated = benchmark._make_batch(
        args,
        dtype=torch.float32,
        device=torch.device("cpu"),
        slot=0,
        routing_case=0,
    )

    assert torch.equal(batches[0].expert_ids, repeated.expert_ids)
    assert len({tuple(batch.expert_ids.flatten().tolist()) for batch in batches}) == 6
    assert len({batch.distinct_expert_ids for batch in batches}) == 6


def test_routing_replay_loads_exact_routes_and_lease_keys(benchmark, tmp_path: Path):
    path = tmp_path / "routing-replay.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    {
                        "expert_ids": [[3, 1], [1, -1]],
                        "lease_expert_ids": [1, 3],
                    },
                    {
                        "expert_ids": [[2, 0], [2, 1]],
                        "lease_expert_ids": [0, 1, 2],
                    },
                ],
            }
        )
    )
    args = benchmark.BenchmarkOptions(
        tokens=2,
        top_k=2,
        experts=4,
        routing_replay=path,
    )

    cases = benchmark._load_routing_replay(args)

    assert cases is not None
    assert [case.expert_ids.tolist() for case in cases] == [
        [[3, 1], [1, -1]],
        [[2, 0], [2, 1]],
    ]
    assert [case.lease_expert_ids for case in cases] == [(1, 3), (0, 1, 2)]


def test_routing_replay_rejects_lease_that_does_not_match_routes(benchmark, tmp_path: Path):
    path = tmp_path / "routing-replay.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    {
                        "expert_ids": [[3, 1], [1, -1]],
                        "lease_expert_ids": [1],
                    }
                ],
            }
        )
    )
    args = benchmark.BenchmarkOptions(
        tokens=2,
        top_k=2,
        experts=4,
        routing_replay=path,
    )

    with pytest.raises(ValueError, match="must equal the sorted distinct"):
        benchmark._load_routing_replay(args)


def test_backend_benchmark_replays_normalized_routing(benchmark, tmp_path: Path):
    path = tmp_path / "routing-replay.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    {
                        "expert_ids": [[3, 1], [1, -1]],
                        "lease_expert_ids": [1, 3],
                    },
                    {
                        "expert_ids": [[2, 0], [2, 1]],
                        "lease_expert_ids": [0, 1, 2],
                    },
                ],
            }
        )
    )
    args = benchmark.BenchmarkOptions(
        backend=benchmark.BackendName.TORCH,
        device="cpu",
        dtype=benchmark.ActivationDType.FP32,
        tokens=2,
        hidden_dim=8,
        intermediate_dim=12,
        top_k=2,
        experts=4,
        routing_replay=path,
        warmup=1,
        repeats=1,
    )

    result = benchmark._run_benchmark(args)

    assert result["routing"]["source"] == "replay"
    assert result["routing"]["cases"] == 2
    assert result["routing"]["unique_lease_keys"] == 2


def test_submission_group_cycles_routing_cases_and_can_reset(benchmark):
    submitted = []

    class Completion:
        def wait_host(self):
            pass

        def close(self):
            pass

    class Backend:
        def submit(self, batch, output):
            submitted.append((batch, output))
            return Completion()

    setup = SimpleNamespace(
        concurrency=1,
        runtime=SimpleNamespace(device=torch.device("cpu")),
        backend=Backend(),
        batches=["case-0", "case-1", "case-2"],
        outputs=["output-0", "output-1", "output-2"],
        routing_cases=3,
    )
    runner = benchmark._SubmissionGroup(setup)
    try:
        for _ in range(4):
            runner.run()
        runner.reset_routing_case()
        runner.run()
    finally:
        runner.close()

    assert submitted == [
        ("case-0", "output-0"),
        ("case-1", "output-1"),
        ("case-2", "output-2"),
        ("case-0", "output-0"),
        ("case-0", "output-0"),
    ]


def test_masking_preserves_original_weights_and_distinct_metadata(benchmark):
    unmasked = benchmark._make_batch(
        benchmark.BenchmarkOptions(tokens=32, top_k=3, experts=19),
        dtype=torch.float32,
        device=torch.device("cpu"),
        slot=0,
    )
    masked = benchmark._make_batch(
        benchmark.BenchmarkOptions(tokens=32, top_k=3, experts=19, inactive_rate=0.8),
        dtype=torch.float32,
        device=torch.device("cpu"),
        slot=0,
    )
    active = masked.expert_ids >= 0
    assert torch.equal(masked.routing_weights[active], unmasked.routing_weights[active])
    assert torch.all(masked.routing_weights[~active] == 0)
    assert torch.any(~active.any(dim=1))
    assert masked.distinct_expert_ids == tuple(torch.unique(masked.expert_ids[active]).tolist())


def test_all_inactive_batch_is_finite_and_has_no_distinct_experts(benchmark):
    batch = benchmark._make_batch(
        benchmark.BenchmarkOptions(tokens=4, top_k=2, experts=8, inactive_rate=1),
        dtype=torch.float32,
        device=torch.device("cpu"),
        slot=0,
    )
    assert torch.all(batch.expert_ids == -1)
    assert torch.all(batch.routing_weights == 0)
    assert batch.distinct_expert_ids == ()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"top_k": 9, "experts": 8},
        {"inactive_rate": 1.1},
        {"hot_experts": 0},
        {"hot_experts": 9, "experts": 8},
        {"hot_multiplier": float("nan")},
        {"routing_cases": 0},
        {"repeats": 0},
    ],
)
def test_invalid_benchmark_options_are_rejected(benchmark, kwargs):
    with pytest.raises(ValueError):
        benchmark.BenchmarkOptions(**kwargs)


def test_typer_rejects_invalid_distribution_without_running_backend(benchmark):
    from typer.testing import CliRunner

    result = CliRunner().invoke(benchmark.app, ["--distribution", "invalid"])
    assert result.exit_code == 2
    assert "invalid" in result.output


@pytest.mark.parametrize(
    "updates",
    [
        {"device": "cpu"},
        {"dtype": "fp16"},
        {"concurrency": 2},
        {"backend": "fused"},
        {"inactive_rate": 0.1},
        {"experts": 129},
        {"expert_end": 9},
        {"expert_start": 8},
        {"expert_compute": "deepseek_v4"},
    ],
)
def test_graph_options_reject_unsupported_profiles(benchmark, updates):
    with pytest.raises(ValueError):
        benchmark.BenchmarkOptions(execution_mode=benchmark.ExecutionMode.GRAPH, **updates)


def test_graph_shard_and_v4_profile_options(benchmark):
    options = benchmark.BenchmarkOptions(
        execution_mode=benchmark.ExecutionMode.GRAPH,
        experts=256,
        expert_start=64,
        expert_end=96,
        expert_compute=benchmark.ExpertCompute.DEEPSEEK_V4,
        quantization=benchmark.BenchmarkQuantization.MODELSLIM_W8A8_DYNAMIC,
        swiglu_limit=10,
    )
    assert options.expert_end - options.expert_start == 32
    with pytest.raises(ValueError, match="require --execution-mode graph"):
        benchmark.BenchmarkOptions(expert_end=4)


def test_synthetic_modelslim_scales_match_across_modes_and_survive_bf16(benchmark):
    values = benchmark._make_source_values(
        expert_id=0,
        hidden_dim=4,
        intermediate_dim=8,
        dtype=torch.bfloat16,
        quantized=True,
        seed=1234,
    )
    for name, value in values.items():
        if name.endswith("weight_scale"):
            assert value.dtype is torch.float32
            assert torch.equal(value, value.bfloat16().float())


def _fake_graph_boundary(benchmark, monkeypatch, setup):
    from contextlib import contextmanager, nullcontext

    state = SimpleNamespace(captured=None, graphs=[], calls=[], events=[], elapsed=0)

    class Runtime:
        device = torch.device("cpu")

        def device_context(self):
            return nullcontext()

        def stream_context(self, stream):
            return nullcontext()

        def create_stream(self):
            return object()

        def capture_current_work(self):
            return SimpleNamespace(wait_host=lambda: state.events.append("producer_wait"))

        def synchronize_stream(self, stream):
            state.events.append("synchronize")

        def create_event(self, *, enable_timing):
            assert enable_timing
            return object()

        def record_event(self, event, stream):
            state.events.append("event")

        def elapsed_time_ms(self, start, end):
            state.elapsed += 1
            return float(state.elapsed)

        def memory_info(self):
            return (1 << 30, 2 << 30)

    class Graph:
        def __init__(self):
            state.graphs.append(self)
            self.callback = None

        def replay(self):
            state.events.append("replay")
            self.callback()

    @contextmanager
    def capture(graph, *, stream):
        state.captured = graph
        try:
            yield
        finally:
            state.captured = None

    class Resident:
        def __init__(self, **options):
            self._lease = setup.table.acquire_many(0, tuple(range(*options["expert_range"])))
            self._revision = setup.table.revision

        def require_current(self):
            assert self._revision == setup.table.revision

        def compute(self, layer, hidden, ids, routing, output):
            state.calls.append(
                (hidden.data_ptr(), ids.data_ptr(), routing.data_ptr(), output.data_ptr())
            )

            def work():
                output.copy_(hidden + ids[:, :1])

            if state.captured is not None:
                state.captured.callback = work
            work()

        def close(self, *, wait_for_completion):
            wait_for_completion()
            self._lease.close()

    monkeypatch.setattr(benchmark, "graph_resident_type", lambda config: Resident)
    setup.runtime = Runtime()
    return SimpleNamespace(NPUGraph=Graph, graph=capture), state


def _cpu_graph_setup(benchmark):
    return benchmark._build_setup(
        benchmark.BenchmarkOptions(
            backend=benchmark.BackendName.TORCH,
            device="cpu",
            dtype=benchmark.ActivationDType.BF16,
            hidden_dim=4,
            intermediate_dim=8,
            experts=4,
            tokens=2,
            top_k=2,
            routing_cases=3,
            distribution=benchmark.Distribution.UNIFORM,
        )
    )


def test_one_capture_replays_changing_inputs_without_mutating_source_cases(benchmark, monkeypatch):
    setup = _cpu_graph_setup(benchmark)
    npu, state = _fake_graph_boundary(benchmark, monkeypatch, setup)
    options = benchmark.BenchmarkOptions(
        execution_mode=benchmark.ExecutionMode.GRAPH,
        experts=4,
        hidden_dim=4,
        intermediate_dim=8,
        top_k=2,
        tokens=2,
        expert_start=1,
        expert_end=3,
    )
    original = [(batch.hidden_states.clone(), batch.expert_ids.clone()) for batch in setup.batches]
    runner = benchmark._GraphReplayGroup(setup, options, npu=npu)
    try:
        assert state.events[0] == "producer_wait"
        assert len(state.graphs) == 1 and len(state.calls) == 4
        assert len(set(state.calls)) == 1
        for index in (0, 1, 2, 0):
            runner.run()
            batch = setup.batches[index]
            torch.testing.assert_close(
                runner.last_outputs[0], batch.hidden_states + batch.expert_ids[:, :1]
            )
            assert state.events[-4:] == ["event", "replay", "event", "synchronize"]
        assert runner.device_samples_ms == [1.0, 2.0, 3.0, 4.0]
        assert len(state.calls) == 4  # Replay does not rerun Host compute or reacquire weights.
        assert setup.table.usage_count(0, 1) == 1
        for batch, (hidden, ids) in zip(setup.batches, original, strict=True):
            assert torch.equal(batch.hidden_states, hidden)
            assert torch.equal(batch.expert_ids, ids)
    finally:
        runner.close()
        assert setup.table.usage_count(0, 1) == 0
        setup.close()


def test_graph_json_separates_device_time_and_discards_warmup_samples(benchmark, monkeypatch):
    setup = _cpu_graph_setup(benchmark)
    npu, state = _fake_graph_boundary(benchmark, monkeypatch, setup)
    original_group = benchmark._GraphReplayGroup

    class Group(original_group):
        def __init__(self, setup, options):
            super().__init__(setup, options, npu=npu)

    monkeypatch.setattr(benchmark, "_GraphReplayGroup", Group)
    monkeypatch.setattr(benchmark, "_build_setup", lambda options: setup)

    def correctness(options, setup, runner):
        runner.run()
        return {"checked": True}

    monkeypatch.setattr(benchmark, "_check_correctness", correctness)
    options = benchmark.BenchmarkOptions(
        execution_mode=benchmark.ExecutionMode.GRAPH,
        experts=4,
        hidden_dim=4,
        intermediate_dim=8,
        tokens=2,
        warmup=2,
        repeats=3,
        expert_start=1,
        expert_end=3,
    )
    result = benchmark._run_benchmark(options)
    assert result["execution"]["mode"] == "graph"
    assert result["execution"]["captured_graphs"] == 1
    assert result["execution"]["operator_path"] == "ascend_fused.run_ascend_fused_moe"
    assert result["execution"]["resident_expert_range"] == [1, 3]
    assert result["execution"]["scope"] == "resident_compute_only"
    assert result["device_replay_latency_ms"]["mean"] == 5.0  # Samples 4,5,6 only.
    assert result["device_replay_latency_ms"]["min"] == 4.0
    assert result["memory"]["estimated_total_temporary_bytes"] is None
    assert len(state.graphs) == 1


def test_correctness_checks_later_routing_cases(benchmark):
    setup = benchmark._build_setup(
        benchmark.BenchmarkOptions(
            backend=benchmark.BackendName.TORCH,
            device="cpu",
            dtype=benchmark.ActivationDType.FP32,
            tokens=2,
            hidden_dim=4,
            intermediate_dim=8,
            experts=4,
            routing_cases=2,
        )
    )

    class BrokenLaterCase(benchmark._SubmissionGroup):
        def run(self):
            super().run()
            if self.last_batches[0] is setup.batches[1]:
                self.last_outputs[0].fill_(float("nan"))

    runner = BrokenLaterCase(setup)
    try:
        with pytest.raises(RuntimeError, match="batch=1"):
            benchmark._check_correctness(
                benchmark.BenchmarkOptions(dtype=benchmark.ActivationDType.FP32), setup, runner
            )
    finally:
        runner.close()
        setup.close()


def test_failed_graph_teardown_quarantines_inputs_and_weight_storage(benchmark, monkeypatch):
    setup = _cpu_graph_setup(benchmark)
    npu, _ = _fake_graph_boundary(benchmark, monkeypatch, setup)
    options = benchmark.BenchmarkOptions(
        execution_mode=benchmark.ExecutionMode.GRAPH,
        experts=4,
        hidden_dim=4,
        intermediate_dim=8,
        tokens=2,
    )
    runner = benchmark._GraphReplayGroup(setup, options, npu=npu)

    def failed(stream):
        raise RuntimeError("device completion unknown")

    monkeypatch.setattr(setup.runtime, "synchronize_stream", failed)
    with pytest.raises(RuntimeError, match="completion unknown"):
        runner.close()
    assert setup.table.quarantined
    setup.close()
    assert setup.table.usage_count(0, 0) == 1
    assert runner._graph is not None


def test_graph_setup_loads_only_resident_shard_and_reference_ignores_remote_experts(
    benchmark, monkeypatch
):
    from expertkit_worker.device import CpuWorkerRuntime

    monkeypatch.setattr(
        benchmark, "_make_runtime", lambda device: CpuWorkerRuntime(torch.device("cpu"))
    )
    options = benchmark.BenchmarkOptions(
        execution_mode=benchmark.ExecutionMode.GRAPH,
        backend=benchmark.BackendName.TORCH,
        tokens=2,
        hidden_dim=4,
        intermediate_dim=8,
        experts=4,
        top_k=2,
        expert_start=1,
        expert_end=3,
    )
    setup = benchmark._build_setup(options)
    try:
        assert set(setup.source_values) == {1, 2}
        assert len(setup.cpu_weights) == 2
        assert not setup.table.is_ready(0, 0)
        assert setup.table.is_ready(0, 1)
        assert not setup.table.is_ready(0, 3)
        batch = setup.batches[0]
        batch.expert_ids.fill_(0)  # Entire batch routes outside this Worker.
        assert torch.count_nonzero(benchmark._reference_output(setup, batch)) == 0
    finally:
        setup.close()
    assert not setup.table.is_ready(0, 1)


def test_graph_routing_file_rejects_eager_inactive_sentinel(benchmark, tmp_path):
    replay = tmp_path / "routing.json"
    replay.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    {"expert_ids": [[0, -1]], "lease_expert_ids": [0]},
                ],
            }
        )
    )
    options = benchmark.BenchmarkOptions(
        execution_mode=benchmark.ExecutionMode.GRAPH,
        tokens=1,
        experts=4,
        top_k=2,
        routing_replay=replay,
    )
    with pytest.raises(ValueError, match="valid global IDs"):
        benchmark._load_routing_replay(options)
