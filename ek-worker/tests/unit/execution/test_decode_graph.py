"""Validate graph command admission without touching an NPU."""

import asyncio
import socket
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import grpc
import pytest
import torch
from expertkit_transport.transports.ascend_graph.config import AscendGraphConfig
from expertkit_transport.transports.ascend_graph.contracts import (
    GraphCompletionUnproven,
    GraphControlRejected,
)
from expertkit_transport.transports.ascend_graph.control import (
    SERVICE,
    GraphControlServer,
    decode_message,
    encode_message,
)

from expertkit_worker.app import WorkerApplication
from expertkit_worker.execution.admission import DeviceAdmission
from expertkit_worker.execution.decode_graph import AscendDecodeGraphExecution, GraphWorkerExecution


@pytest.fixture
def execution():
    from expertkit_transport.ascend_graph import AscendGraphModel

    model = AscendGraphModel(
        num_layers=2,
        experts_per_layer=4,
        hidden_dim=4,
        expert_intermediate_dim=6,
        top_k=2,
        activation_dtype="bf16",
        weight_dtype="bf16",
        activation="silu",
        expert_compute="swiglu",
        swiglu_limit=0.0,
        quantization=None,
    )
    config = AscendGraphConfig(
        session_id="worker-cpu-session-001",
        instance_id=1,
        model_name="model",
        weight_version="revision",
        model=model,
        mode="decode",
        workers=[
            dict(
                worker_id="worker",
                expected_start_id="start-one",
                device="npu:0",
                control_endpoint="127.0.0.1:12345",
                experts_by_layer=[[0, 1, 2, 3]],
            )
        ],
        rendezvous_host="127.0.0.1",
        rendezvous_port=12346,
        layer_ids=(1,),
        capture_sizes=(1, 4),
        decode_capture_sizes=(1, 4),
        timeout_seconds=2.0,
        source_topology_version=1,
    )
    owner = AscendDecodeGraphExecution(
        instance_id=1,
        model_name="model",
        weight_version="revision",
        model=model,
        moe_layer_range=(1, 2),
        worker_id="worker",
        worker_start_id="start-one",
        device=torch.device("cpu"),
        device_name="npu:0",
        max_batch_tokens=2048,
        shutdown_grace_secs=2.0,
        admission=DeviceAdmission(),
        prepare_weights=lambda assignment, layer_ids: pytest.fail(
            "malformed commands must not prepare weights"
        ),
        guard_assignments=lambda assignment, layer_ids: Mock(require_current=Mock(), close=Mock()),
        terminate=lambda code: pytest.fail(f"unexpected process termination: {code}"),
    )
    owner._test_config = config
    owner._assignment_id = config.assignment_id
    try:
        yield owner
    finally:
        asyncio.run(owner.close())


def _reserve_execution(execution, config=None):
    from expertkit_transport.transports.ascend_graph.rendezvous import GraphRendezvousRequest

    facts = (config or execution._test_config).model_dump(mode="json")
    facts["rendezvous_port"] = 0
    reply = execution.rendezvous(
        GraphRendezvousRequest.from_facts(facts).model_dump(mode="json"), timeout=2
    )
    execution._test_config = AscendGraphConfig.model_validate(
        {**facts, "rendezvous_port": reply["rendezvous_port"]}
    )
    return execution._test_config


def request_for(execution, *, finishing=False, **updates):
    """Construct the full initial snapshot or a compact subsequent command."""
    config = execution.config or execution._test_config
    phase = updates.get("phase", "prepare")
    result = {
        "version": 2,
        "assignment_id": config.assignment_id,
        "sequence": 1,
        "phase": phase,
        "capacity": 0,
        "worker_start_id": "start-one",
        "ready_revision": 7 if finishing else None,
        **updates,
    }
    if phase == "prepare" and not finishing and "config" not in updates:
        result["config"] = config.model_dump(mode="json")
    return result


@pytest.mark.parametrize(
    "updates",
    [
        {"version": True},
        {"version": 1},
        {"phase": []},
        {"phase": {}},
        {"phase": True},
        {"phase": "unknown"},
        {"capacity": True},
        {"capacity": 1},
        {"sequence": True},
        {"sequence": 0},
        {"worker_start_id": "stale"},
        {"ready_revision": 0},
        {"unknown": 1},
        {"config": []},
    ],
)
def test_invalid_command_rejected_before_admission(execution, updates):
    with pytest.raises(GraphControlRejected):
        execution.begin(request_for(execution, **updates), timeout=2)
    assert execution._sequence == 0
    assert execution._pending is None


def test_strict_identity_rejects_bool_instance(execution):
    request = request_for(execution)
    request["config"]["instance_id"] = True
    with pytest.raises(ValueError, match="configuration"):
        execution._validate_request(request)


def test_finish_requires_exact_pending_command_and_clears_once(execution):
    request = request_for(execution, finishing=True)
    pending = Future()
    pending.set_result(None)
    execution._pending = pending
    execution._deadline = time.monotonic() + 2
    execution._pending_request = request
    with pytest.raises(GraphControlRejected):
        execution.finish({**request, "sequence": 2}, timeout=2)
    assert execution._pending is pending
    for name, value in (
        ("version", True),
        ("sequence", True),
        ("capacity", False),
        ("ready_revision", True),
    ):
        with pytest.raises(GraphControlRejected):
            execution.finish({**request, name: value}, timeout=2)
    assert execution._pending is pending
    assert execution.finish(request, timeout=2) == execution._reply(request, "complete", 7)
    with pytest.raises(GraphControlRejected):
        execution.finish(request, timeout=2)


def test_duplicate_finish_cannot_clear_the_next_step(execution):
    request = request_for(execution, finishing=True)
    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]

    class Pending:
        def __init__(self):
            self.calls = 0
            self.lock = threading.Lock()

        def result(self, timeout):
            with self.lock:
                index = self.calls
                self.calls += 1
            entered[index].set()
            assert release[index].wait(timeout)

    pending = Pending()
    execution._pending = pending
    execution._deadline = time.monotonic() + 2
    execution._pending_request = request
    with ThreadPoolExecutor(max_workers=2) as threads:
        first = threads.submit(execution.finish, request, timeout=2)
        assert entered[0].wait(2)
        second = threads.submit(execution.finish, request, timeout=2)
        assert entered[1].wait(2)
        release[0].set()
        assert first.result(timeout=2)["state"] == "complete"
        next_pending = Future()
        next_pending.set_result(None)
        next_request = {**request, "sequence": 2}
        with execution._lock:
            execution._pending = next_pending
            execution._pending_request = next_request
        release[1].set()
        with pytest.raises(GraphControlRejected, match="already completed"):
            second.result(timeout=2)
        assert execution._pending is next_pending
        assert execution._pending_request == next_request
        execution.finish(next_request, timeout=2)


def test_finish_timeout_retains_device_future_and_absolute_step_deadline(execution):
    request = request_for(execution, finishing=True)
    pending = Future()
    execution._pending = pending
    execution._deadline = time.monotonic() + 2
    execution._pending_request = request
    execution._deadline = 123.0
    with pytest.raises(GraphCompletionUnproven, match="could not be proven"):
        execution.finish(request, timeout=0)
    assert execution._poisoned
    assert execution._pending is pending and not pending.cancelled()
    assert execution._pending_request == request and execution._deadline == 123.0
    with pytest.raises(GraphControlRejected, match="cannot admit"):
        execution.begin(request_for(execution, sequence=2), timeout=2)
    pending.set_result(None)


def test_real_adapter_loopback_preserves_execution_validation_and_exact_finish(execution):
    server = GraphControlServer("127.0.0.1:0", begin=execution.begin, finish=execution.finish)
    port = server.start()
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            begin, finish = [
                channel.unary_unary(
                    f"/{SERVICE}/{name}",
                    request_serializer=encode_message,
                    response_deserializer=decode_message,
                )
                for name in ("Begin", "Finish")
            ]
            with pytest.raises(grpc.RpcError) as rejected:
                begin(request_for(execution, sequence=True), timeout=2)
            assert rejected.value.code() == grpc.StatusCode.FAILED_PRECONDITION
            assert execution._pending is None and execution._sequence == 0
            pending = Future()
            pending.set_result(None)
            execution._pending = pending
            execution._deadline = time.monotonic() + 2
            execution._pending_request = request_for(execution, finishing=True)
            assert finish(request_for(execution, finishing=True), timeout=2) == execution._reply(
                request_for(execution, finishing=True), "complete", 7
            )
            with pytest.raises(grpc.RpcError) as repeated:
                finish(request_for(execution, finishing=True), timeout=2)
            assert repeated.value.code() == grpc.StatusCode.FAILED_PRECONDITION
    finally:
        server.close()


def test_real_bind_failure_reaches_application_cleanup(execution):
    async def scenario(endpoint):
        ordinary = SimpleNamespace(start=AsyncMock(), close=AsyncMock())
        server = GraphControlServer(endpoint, begin=execution.begin, finish=execution.finish)
        graph_owner = GraphWorkerExecution(ordinary, execution, server)
        service = SimpleNamespace(start=AsyncMock(), close=AsyncMock())
        application = WorkerApplication(
            transfer=service,
            manager=SimpleNamespace(start=Mock(), close=AsyncMock()),
            peer_server=service,
            execution=graph_owner,
            control=SimpleNamespace(close=AsyncMock()),
            disk_cache=SimpleNamespace(close=Mock()),
            observability=service,
        )
        with pytest.raises(RuntimeError, match="bind"):
            await application.run()
        ordinary.close.assert_awaited_once()
        assert execution._closed and execution._monitor.done()
        for pool in (execution._executor, server._executor):
            with pytest.raises(RuntimeError, match="shutdown"):
                pool.submit(lambda: None)

    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        asyncio.run(scenario(f"127.0.0.1:{occupied.getsockname()[1]}"))


def test_idle_withdrawal_destroys_graphs_and_releases_real_leases(execution):
    from expertkit_worker.backends.torch.decode_graph import ResidentExpertGraphWeights
    from expertkit_worker.backends.torch.weights import TorchExpertWeights
    from expertkit_worker.weights import ReadyWeightTable

    table = ReadyWeightTable(2, 1)
    weight = TorchExpertWeights(
        *(torch.ones(shape, dtype=torch.bfloat16) for shape in ((6, 4), (6, 4), (4, 6)))
    )
    table.publish(1, 0, weight)
    execution._weights = ResidentExpertGraphWeights(
        layer_ids=(1,),
        experts_per_layer=1,
        hidden_dim=4,
        intermediate_dim=6,
        top_k=1,
        acquire_many=table.acquire_many,
        ready_revision=lambda: table.revision,
        operators=SimpleNamespace(),
    )
    execution._graphs[1] = object()
    closed = threading.Event()
    synchronized = threading.Event()

    def synchronize():
        if not synchronized.is_set():
            assert execution._graphs and table.usage_count(1, 0) == 1
            synchronized.set()

    def close_session():
        assert synchronized.is_set()
        assert not execution._graphs and table.usage_count(1, 0) == 0
        closed.set()

    execution._stream = SimpleNamespace(synchronize=synchronize)
    execution._session = SimpleNamespace(close=close_session)
    table.begin_withdrawal(1, 0)
    with pytest.raises(TimeoutError):
        table.finish_withdrawal(1, 0, timeout=0)

    async def observe():
        execution._monitor = asyncio.create_task(execution._watch())
        assert await asyncio.to_thread(closed.wait, 2)
        assert await asyncio.to_thread(table.finish_withdrawal, 1, 0, timeout=1) is weight
        execution._stopped.set()
        await asyncio.wait_for(execution._monitor, timeout=2)

    asyncio.run(observe())
    assert execution._poisoned and execution._weights is None


def test_ordinary_completion_does_not_cancel_graph_lifecycle_monitor(execution):
    async def scenario():
        waiting = asyncio.Event()
        finished = asyncio.Event()

        class Ordinary:
            async def wait(self):
                waiting.set()
                await finished.wait()

            async def close(self):
                finished.set()

        execution._graphs[1] = object()
        execution._monitor = asyncio.create_task(execution._watch())
        control = SimpleNamespace(start=lambda: None, stop=lambda: None, close=lambda: None)
        owner = GraphWorkerExecution(Ordinary(), execution, control)
        waiter = asyncio.create_task(owner.wait())
        await asyncio.wait_for(waiting.wait(), timeout=2)
        finished.set()
        await asyncio.wait_for(waiter, timeout=2)
        await asyncio.wait_for(owner.close(), timeout=2)

        assert not execution._monitor.cancelled()
        assert execution._graphs == {}

    asyncio.run(scenario())


def _v4_full_execution_config(execution):
    values = execution._test_config.model_dump()
    values.update(
        mode="full",
        layer_ids=tuple(range(43)),
        capture_sizes=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048),
        decode_capture_sizes=(1, 2, 4, 8, 16, 32, 64, 128, 256),
    )
    values["model"].update(
        num_layers=43,
        experts_per_layer=256,
        hidden_dim=4096,
        expert_intermediate_dim=2048,
        top_k=6,
        expert_compute="deepseek_v4",
        swiglu_limit=10.0,
        quantization=dict(type="modelslim-w8a8-dynamic", bits=8, group_size=None, symmetric=True),
    )
    values["workers"][0]["experts_by_layer"] = [list(range(256)) for _ in range(43)]
    config = AscendGraphConfig.model_validate(values)
    execution._assignment_id = config.assignment_id
    return config


def test_v4_worker_pairs_graph_keys_and_rejects_phase_contract_errors(execution):
    execution.config = _v4_full_execution_config(execution)
    execution._weights = SimpleNamespace(
        revision=1,
        require_current=lambda: None,
        close=lambda **kwargs: kwargs["wait_for_completion"](),
    )
    execution._session = SimpleNamespace(close=lambda: None)

    def command(phase="capture", **updates):
        return request_for(
            execution,
            phase=phase,
            capacity=4,
            graph_phase="decode",
            worker_start_id="start-one",
            ready_revision=1,
            **updates,
        )

    for phase in ("decode", "mixed"):
        request = command()
        request["graph_phase"] = phase
        assert execution._validate_request(request) == ("capture", 4, None)
        execution._graphs[(phase, 4)] = object()
        with pytest.raises(ValueError, match="already been captured"):
            execution._validate_request(request)
        assert execution._validate_request({**request, "phase": "replay"}) == ("replay", 4, None)
    for updates in (
        {"graph_phase": "bad"},
        {"graph_phase": []},
        {"phase": "warmup"},
        {"graph_phase": "decode", "capacity": 512},
        {"graph_phase": "mixed", "capacity": 1},
    ):
        request = {**command(), **updates}
        with pytest.raises(ValueError):
            execution._validate_request(request)
    missing = command()
    missing.pop("graph_phase")
    with pytest.raises(ValueError, match="fields"):
        execution._validate_request(missing)


def test_worker_capture_failure_retains_phase_graph_owner(execution, monkeypatch):
    from contextlib import nullcontext

    execution._weights = SimpleNamespace(
        require_current=lambda: None, close=lambda **kwargs: kwargs["wait_for_completion"]()
    )
    execution._session = SimpleNamespace(close=lambda: None)
    execution._stream = SimpleNamespace(synchronize=lambda: None)
    graph = object()
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            set_device=lambda _: None,
            stream=lambda _: nullcontext(),
            NPUGraph=lambda: graph,
            graph=lambda *_args, **_kwargs: nullcontext(),
        ),
        raising=False,
    )

    def fail(_capacity):
        assert execution._graphs[("mixed", 4)] is graph
        raise RuntimeError("capture failed")

    monkeypatch.setattr(execution, "_body", fail)
    with pytest.raises(RuntimeError, match="capture failed"):
        execution._run("capture", 4, threading.Event(), "mixed")
    assert execution._poisoned and execution._graphs[("mixed", 4)] is graph
    assert execution._admission._poisoned
    # A poisoned device cannot be drained by the normal fixture shutdown.
    execution._closed = True
    execution._executor.shutdown(wait=True)


def test_worker_phase_graph_replay_and_close_keep_both_captures(execution, monkeypatch):
    from contextlib import nullcontext

    execution.config = _v4_full_execution_config(execution)
    events = []

    class Graph:
        def __init__(self):
            self.number = len([item for item in events if isinstance(item, Graph)])
            events.append(self)

        def replay(self):
            events.append(("replay", self.number))

    def synchronized():
        events.append("sync")

    execution._weights = SimpleNamespace(
        require_current=lambda: None, close=lambda **kwargs: kwargs["wait_for_completion"]()
    )
    execution._session = SimpleNamespace(close=lambda: events.append("session-close"))
    execution._stream = SimpleNamespace(synchronize=synchronized)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            set_device=lambda _: None,
            stream=lambda _: nullcontext(),
            NPUGraph=Graph,
            graph=lambda *_args, **_kwargs: nullcontext(),
        ),
        raising=False,
    )
    monkeypatch.setattr(execution, "_body", lambda capacity: events.append(("body", capacity)))
    for phase in ("decode", "mixed"):
        execution._run("capture", 4, threading.Event(), phase)
    assert len(execution._graphs) == 2
    for phase in ("decode", "mixed", "decode"):
        execution._run("replay", 4, threading.Event(), phase)
    assert [item for item in events if isinstance(item, tuple) and item[0] == "replay"] == [
        ("replay", 0),
        ("replay", 1),
        ("replay", 0),
    ]
    execution._destroy()
    assert execution._graphs == {} and events[-2:] == ["sync", "session-close"]


def test_legacy_worker_rejects_phase_extension(execution):
    with pytest.raises(ValueError, match="fields"):
        execution._validate_request(request_for(execution, graph_phase="decode"))


def _fake_device(execution, monkeypatch):
    from contextlib import nullcontext

    import expertkit_worker.execution.decode_graph as module

    events = []
    stream = SimpleNamespace(synchronize=lambda: events.append("synchronize"))
    guard = Mock()
    guard.require_current.side_effect = lambda: events.append("guard-check")
    guard.close.side_effect = lambda: events.append("guard-close")
    weights = Mock(revision=7)
    weights.require_current.side_effect = lambda: events.append("ready-check")

    def release_weights(*, wait_for_completion):
        wait_for_completion()
        events.append("weights-close")

    weights.close.side_effect = release_weights

    class Session:
        def __init__(self, config, **options):
            events.append("hccl-create")
            self.config = config
            self.options = options

        def buffers(self, capacity):
            return SimpleNamespace(
                hidden=None, expert_ids=None, routing_weights=None, output=None, partial_output=None
            )

        def receive_inputs(self, buffers):
            events.append("receive")

        def combine(self, buffers):
            events.append("combine")

        def close(self):
            events.append("session-close")

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            set_device=lambda _: events.append("set-device"),
            Stream=lambda **kwargs: stream,
            stream=lambda _: nullcontext(),
        ),
        raising=False,
    )
    monkeypatch.setattr(module, "HcclGraphSession", Session)
    execution._prepare_weights = Mock(return_value=weights)
    execution._guard_assignments = Mock(return_value=guard)
    return events, guard, weights


@pytest.mark.parametrize("moe_layer_range", [(0, 7), (1, 6), (2, 7), (0, 5)])
def test_prepare_uses_actual_sparse_layers_within_metadata_range(
    execution, monkeypatch, moe_layer_range
):
    _events, _guard, _weights = _fake_device(execution, monkeypatch)
    data = execution._test_config.model_dump(mode="json")
    data["model"]["num_layers"] = 7
    data["layer_ids"] = [1, 5]
    data["workers"][0]["experts_by_layer"] *= 2
    config = AscendGraphConfig.model_validate(data)
    execution._model = config.model
    execution._moe_layer_range = moe_layer_range
    execution._test_config = config
    request = request_for(execution)
    if moe_layer_range in ((2, 7), (0, 5)):
        with pytest.raises(GraphControlRejected, match="facts do not match"):
            execution.begin(request, timeout=2)
        execution._guard_assignments.assert_not_called()
        execution._prepare_weights.assert_not_called()
        assert execution.config is None
        return

    config = _reserve_execution(execution, config)
    execution.begin(request_for(execution), timeout=2)
    execution.finish(dict(execution._pending_request), timeout=2)
    assert execution.config is not None
    layers = execution.config.layer_ids
    assert layers == (1, 5)
    execution._guard_assignments.assert_called_once_with(config.participants[0], layers)
    execution._prepare_weights.assert_called_once_with(config.participants[0], layers)
    assert execution._prepare_weights.call_args.args[1] is layers


def test_v2_real_control_binds_once_and_finishes_compactly(execution, monkeypatch):
    from expertkit_transport.ascend_graph import GraphStepControl

    events, _guard, weights = _fake_device(execution, monkeypatch)
    server = GraphControlServer(
        "127.0.0.1:0",
        begin=execution.begin,
        finish=execution.finish,
        rendezvous=execution.rendezvous,
    )
    port = server.start()
    data = execution._test_config.model_dump(mode="json")
    data["workers"][0]["control_endpoint"] = f"127.0.0.1:{port}"
    data["rendezvous_port"] = 0
    control = GraphStepControl.reserve(data, monotonic_deadline=time.monotonic() + 2)
    config = control.config
    try:
        assert execution.config is None and execution._rank is None
        control.begin("prepare")
        assert execution.config == config and execution._rank == 1
        assert "config" not in execution._pending_request
        assert execution._pending_request["ready_revision"] == 7
        assert "config" not in control._requests[0]
        control.finish()
        execution._prepare_weights.assert_called_once_with(config.participants[0], config.layer_ids)
        assert events.index("guard-check") < events.index("set-device")
        assert events.index("ready-check") < events.index("hccl-create")
        with pytest.raises(GraphControlRejected, match="already prepared"):
            execution.begin(request_for(execution, sequence=2), timeout=2)
        monkeypatch.setattr(
            AscendGraphConfig, "assignment_id", property(lambda _: pytest.fail("digest recomputed"))
        )
        monkeypatch.setattr(
            AscendGraphConfig,
            "model_validate",
            lambda *_args, **_kwargs: pytest.fail("config reparsed"),
        )
        control.begin("warmup", 4)
        control.finish()
        weights.compute.assert_called_once()
        control.begin("close")
        control.finish()
        assert events[-3:] == ["weights-close", "session-close", "guard-close"]
        assert execution._assignment_guard is None
    finally:
        control.close()
        server.close()


@pytest.mark.parametrize(
    "change",
    [
        "instance",
        "name",
        "version",
        "geometry",
        "quantization",
        "layers",
        "worker",
        "start",
        "device",
        "capacity",
    ],
)
def test_prepare_rejects_conflicting_local_facts_before_resources(execution, change):
    data = execution._test_config.model_dump(mode="json")
    if change == "instance":
        data["instance_id"] = 2
    elif change == "name":
        data["model_name"] = "other"
    elif change == "version":
        data["weight_version"] = "other"
    elif change == "geometry":
        data["model"]["hidden_dim"] = 8
    elif change == "quantization":
        data["model"]["quantization"] = dict(
            type="modelslim-w8a8-dynamic", bits=8, group_size=None, symmetric=True
        )
    elif change == "layers":
        data["layer_ids"] = [0]
    elif change == "worker":
        data["workers"][0]["worker_id"] = "other"
    elif change == "start":
        data["workers"][0]["expected_start_id"] = "stale"
    elif change == "device":
        data["workers"][0]["device"] = "npu:1"
    else:
        data.update(capture_sizes=[1, 4096], decode_capture_sizes=[1, 4096])
    changed = AscendGraphConfig.model_validate(data)
    execution._prepare_weights = Mock()
    execution._guard_assignments = Mock()
    request = request_for(execution, config=data, assignment_id=changed.assignment_id)
    with pytest.raises(GraphControlRejected):
        execution.begin(request, timeout=2)
    assert execution.config is None and execution._rank is None
    execution._prepare_weights.assert_not_called()
    execution._guard_assignments.assert_not_called()


def test_prepare_rejects_wrong_digest_and_initial_start_before_binding(execution):
    execution._prepare_weights = Mock()
    execution._guard_assignments = Mock()
    for updates in (
        {"assignment_id": "0" * 64},
        {"worker_start_id": None},
        {"worker_start_id": "stale"},
    ):
        with pytest.raises(GraphControlRejected):
            execution.begin(request_for(execution, **updates), timeout=2)
    assert execution.config is None
    execution._prepare_weights.assert_not_called()
    execution._guard_assignments.assert_not_called()


def test_guard_is_rechecked_after_acquisition_before_hccl(execution, monkeypatch):
    events, guard, weights = _fake_device(execution, monkeypatch)
    execution._guard_assignments.return_value = guard
    _reserve_execution(execution)
    guard.require_current.side_effect = [None, None, RuntimeError("selected target removed")]
    with pytest.raises(GraphCompletionUnproven):
        execution.begin(request_for(execution), timeout=2)
    assert execution._weights is weights
    assert "hccl-create" not in events
    guard.close.assert_not_called()
    execution._closed = True
    execution._executor.shutdown(wait=True)
    execution._rendezvous.close()


def test_expired_step_quarantines_without_detaching_assignment_guard(execution, monkeypatch):
    import expertkit_worker.execution.decode_graph as module

    retained = []
    monkeypatch.setattr(module, "retain_until_process_exit", retained.append)
    guard = Mock()
    weights, session = Mock(), Mock()
    execution._assignment_guard = guard
    execution._weights, execution._session = weights, session
    execution._pending = Future()
    execution._deadline = time.monotonic() - 1
    execution._terminate = Mock()
    asyncio.run(execution._watch())
    execution._terminate.assert_called_once_with(1)
    weights.quarantine.assert_called_once()
    session.quarantine.assert_called_once()
    guard.close.assert_not_called()
    assert execution in retained and execution._assignment_guard is guard
    execution._closed = True
    execution._executor.shutdown(wait=True)


def test_v4_decode_phase_uses_derived_membership_not_256(execution):
    data = _v4_full_execution_config(execution).model_dump(mode="json")
    data["decode_capture_sizes"] = list(data["capture_sizes"][:10])
    execution.config = AscendGraphConfig.model_validate(data)
    execution._assignment_id = execution.config.assignment_id
    execution._weights = SimpleNamespace(
        revision=7,
        require_current=lambda: None,
        close=lambda **kwargs: kwargs["wait_for_completion"](),
    )
    execution._session = SimpleNamespace(close=lambda: None)
    request = request_for(
        execution, phase="capture", graph_phase="decode", capacity=512, ready_revision=7
    )
    assert execution._validate_request(request) == ("capture", 512, None)
    with pytest.raises(ValueError, match="phase"):
        execution._validate_request({**request, "capacity": 1024})


def test_missing_selected_target_rejects_rendezvous_before_device_work(execution):
    execution._guard_assignments = Mock(side_effect=RuntimeError("not a Controller target"))
    execution._prepare_weights = Mock()
    with pytest.raises(GraphControlRejected, match="Controller target"):
        _reserve_execution(execution)
    assert execution.config is None and execution._pending is None
    execution._prepare_weights.assert_not_called()


def test_invalidation_teardown_keeps_the_original_active_step_deadline(execution, monkeypatch):
    import expertkit_worker.execution.decode_graph as module

    deadline_waits = []
    guard = Mock()
    guard.require_current.side_effect = RuntimeError("selected target removed")
    execution._assignment_guard = guard
    execution._weights = Mock()
    execution._pending = Future()
    execution._deadline = 100.0
    execution._terminate = Mock()
    work = Future()
    executor = execution._executor
    execution._executor = SimpleNamespace(submit=lambda callback: work)
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: 99.0))
    monkeypatch.setattr(module, "retain_until_process_exit", lambda owner: None)

    async def expired(awaitable, *, timeout):
        deadline_waits.append(timeout)
        raise TimeoutError

    monkeypatch.setattr(
        module,
        "asyncio",
        SimpleNamespace(wait_for=expired, shield=asyncio.shield, wrap_future=asyncio.wrap_future),
    )
    asyncio.run(execution._watch())
    assert deadline_waits == [1.0]
    assert execution._pending is not None and not work.cancelled()
    execution._terminate.assert_called_once_with(1)
    guard.close.assert_not_called()
    execution._closed = True
    executor.shutdown(wait=True)


def _assert_port_held(port):
    with socket.socket() as probe, pytest.raises(OSError):
        probe.bind(("127.0.0.1", port))


def _assert_port_released(port):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))


def test_reservation_duplicate_does_not_renew_and_foreign_commands_cannot_release(execution):
    from expertkit_transport.transports.ascend_graph.rendezvous import GraphRendezvousRequest

    config = _reserve_execution(execution)
    reservation = execution._rendezvous
    deadline = reservation.deadline
    request = reservation.request.model_dump(mode="json")
    assert execution.rendezvous(request, timeout=2)["rendezvous_port"] == config.rendezvous_port
    assert reservation.deadline == deadline
    for update in (
        {"session_id": "foreign-session-00001"},
        {"worker_start_id": "old-start"},
        {"reservation_id": "f" * 64},
        {"version": True},
    ):
        with pytest.raises(GraphControlRejected):
            execution.rendezvous({**reservation.request.release_request(), **update}, timeout=2)
        _assert_port_held(config.rendezvous_port)
    foreign = GraphRendezvousRequest.from_facts(
        {**config.model_dump(mode="json"), "session_id": "foreign-session-00001"}
    )
    with pytest.raises(GraphControlRejected):
        execution.rendezvous(foreign.model_dump(mode="json"), timeout=2)
    assert execution._rendezvous is reservation
    execution.rendezvous(reservation.request.release_request(), timeout=2)
    _assert_port_released(config.rendezvous_port)
    with pytest.raises(GraphControlRejected, match="already ended"):
        execution.rendezvous(request, timeout=2)


@pytest.mark.parametrize(
    "change", ["session_id", "rendezvous_host", "rendezvous_port", "capture_sizes", "rank"]
)
def test_foreign_prepare_cannot_adopt_or_displace_live_reservation(execution, change):
    config = _reserve_execution(execution)
    reservation, guard = execution._rendezvous, execution._assignment_guard
    facts = config.model_dump(mode="json")
    if change == "session_id":
        facts["session_id"] = "foreign-session-00001"
    elif change == "rendezvous_host":
        facts["rendezvous_host"] = "127.0.0.2"
    elif change == "rendezvous_port":
        facts["rendezvous_port"] = config.rendezvous_port % 65535 + 1
    elif change == "capture_sizes":
        facts["capture_sizes"] = facts["decode_capture_sizes"] = [1]
    else:
        facts["workers"][0]["experts_by_layer"] = [[2, 3]]
        facts["workers"].insert(
            0,
            {
                "worker_id": "before-worker",
                "expected_start_id": "other-start",
                "device": "npu:1",
                "control_endpoint": "127.0.0.1:12347",
                "experts_by_layer": [[0, 1]],
            },
        )
    changed = AscendGraphConfig.model_validate(facts)
    with pytest.raises(GraphControlRejected, match="reservation"):
        execution.begin(
            request_for(execution, config=facts, assignment_id=changed.assignment_id), timeout=2
        )
    assert execution._rendezvous is reservation and execution._assignment_guard is guard
    assert not reservation.adopted and execution.config is None and execution._pending is None
    guard.close.assert_not_called()
    _assert_port_held(config.rendezvous_port)


@pytest.mark.parametrize("reason", ["expiry", "invalidation"])
def test_unused_listener_released_without_waiting_for_ordinary_device(
    execution, monkeypatch, reason
):
    config = _reserve_execution(execution)
    reservation, guard = execution._rendezvous, execution._assignment_guard
    if reason == "expiry":
        reservation.deadline = time.monotonic() - 1
    else:
        guard.require_current.side_effect = RuntimeError("target removed")
    with execution._admission.use():
        execution._expire_rendezvous()
        assert execution._rendezvous is None and execution._assignment_guard is None
        _assert_port_released(config.rendezvous_port)
    guard.close.assert_called_once()


def test_shutdown_releases_unused_listener_even_while_native_bind_is_blocked(
    execution, monkeypatch
):
    from expertkit_transport.transports.ascend_graph.rendezvous import GraphRendezvousRequest

    import expertkit_worker.execution.decode_graph as module

    entered, release = threading.Event(), threading.Event()
    original = module.OwnedGraphRendezvous
    bound_ports = []

    def blocked(request, **kwargs):
        entered.set()
        assert release.wait(2)
        reservation = original(request, **kwargs)
        bound_ports.append(reservation.port)
        return reservation

    monkeypatch.setattr(module, "OwnedGraphRendezvous", blocked)
    facts = execution._test_config.model_dump(mode="json")
    facts["rendezvous_port"] = 0
    request = GraphRendezvousRequest.from_facts(facts).model_dump(mode="json")
    with ThreadPoolExecutor(1) as caller:
        pending = caller.submit(execution.rendezvous, request, timeout=2)
        try:
            assert entered.wait(2)
            guard = execution._assignment_guard
            # Host-only shutdown must not acquire ordinary device admission.
            with execution._admission.use():
                asyncio.run(asyncio.wait_for(execution.close(), timeout=1))
            guard.close.assert_called_once()
            assert execution._rendezvous_binding is None
        finally:
            release.set()
        with pytest.raises(GraphControlRejected, match="ended while binding"):
            pending.result(timeout=2)
    assert len(bound_ports) == 1
    _assert_port_released(bound_ports[0])


def test_prepare_failure_before_stream_creation_releases_adopted_host_resources(
    execution, monkeypatch
):
    _events, guard, _weights = _fake_device(execution, monkeypatch)
    config = _reserve_execution(execution)
    monkeypatch.setattr(
        torch.npu, "set_device", Mock(side_effect=RuntimeError("device unavailable"))
    )
    with pytest.raises(GraphCompletionUnproven):
        execution.begin(request_for(execution), timeout=2)
    assert execution._rendezvous.adopted and execution._stream is None
    asyncio.run(execution.close())
    guard.close.assert_called_once()
    assert execution._rendezvous is None
    _assert_port_released(config.rendezvous_port)


def test_real_grpc_bootstrap_shares_one_final_config_and_preserves_adopted_store(
    execution, monkeypatch
):
    from expertkit_transport.ascend_graph import GraphStepControl

    _fake_device(execution, monkeypatch)
    second = AscendDecodeGraphExecution(
        instance_id=execution._instance_id,
        model_name=execution._model_name,
        weight_version=execution._weight_version,
        model=execution._model,
        moe_layer_range=execution._moe_layer_range,
        worker_id="worker-z",
        worker_start_id="start-two",
        device=torch.device("cpu"),
        device_name="npu:1",
        max_batch_tokens=2048,
        shutdown_grace_secs=2.0,
        admission=DeviceAdmission(),
        prepare_weights=Mock(),
        guard_assignments=Mock(),
        terminate=Mock(),
    )
    _fake_device(second, monkeypatch)
    servers = [
        GraphControlServer(
            "127.0.0.1:0",
            begin=owner.begin,
            finish=owner.finish,
            rendezvous=owner.rendezvous,
        )
        for owner in (execution, second)
    ]
    ports = [server.start() for server in servers]
    facts = execution._test_config.model_dump(mode="json")
    facts["workers"] = [
        {
            "worker_id": worker,
            "expected_start_id": start,
            "device": f"npu:{rank}",
            "control_endpoint": f"127.0.0.1:{port}",
            "experts_by_layer": [experts],
        }
        for rank, (worker, start, port, experts) in enumerate(
            [
                ("worker", "start-one", ports[0], [0, 2]),
                ("worker-z", "start-two", ports[1], [1, 3]),
            ]
        )
    ]
    facts["rendezvous_port"] = 0
    control = None
    try:
        control = GraphStepControl.reserve(facts, monotonic_deadline=time.monotonic() + 2)
        config = control.config
        assert execution._session is None and second._session is None
        _assert_port_held(config.rendezvous_port)
        reservation = execution._rendezvous
        deadline = reservation.deadline
        control.begin("prepare")
        control.finish()
        assert execution.config == second.config == config
        assert execution._session.config.assignment_id == second._session.config.assignment_id
        assert execution._session.options["rendezvous_store"] is reservation.store
        assert second._session.options["rendezvous_store"] is None
        assert (
            execution.rendezvous(reservation.request.release_request(), timeout=2)["state"]
            == "attached"
        )
        assert reservation.deadline == deadline and reservation.adopted
        reservation.deadline = time.monotonic() - 1
        execution._expire_rendezvous()
        _assert_port_held(config.rendezvous_port)
        control.begin("warmup", 1)
        control.finish()
        control.begin("close")
        control.finish()
        assert execution._rendezvous is None and execution._assignment_guard is None
        _assert_port_released(config.rendezvous_port)
    finally:
        if control is not None:
            control.close()
        for server in servers:
            server.close()
        asyncio.run(second.close())


@pytest.mark.parametrize(
    "failure", ["wrong-start", "wrong-host", "bool-port", "extra", "lost-reply", "interrupt"]
)
def test_real_grpc_bad_or_lost_bootstrap_reply_releases_only_its_owner(execution, failure):
    from expertkit_transport.ascend_graph import GraphStepControl

    bound_ports = []

    def rendezvous(request, *, timeout):
        reply = execution.rendezvous(request, timeout=timeout)
        if request["action"] == "release":
            return reply
        bound_ports.append(reply["rendezvous_port"])
        if failure == "wrong-start":
            return {**reply, "worker_start_id": "restarted"}
        if failure == "wrong-host":
            return {**reply, "rendezvous_host": "127.0.0.2"}
        if failure == "bool-port":
            return {**reply, "rendezvous_port": True}
        if failure == "extra":
            return {**reply, "unrecognized": 1}
        if failure == "lost-reply":
            raise GraphCompletionUnproven("reply could not be delivered")
        return reply

    def interrupted(_config):
        raise KeyboardInterrupt("Frontend startup interrupted")

    server = GraphControlServer(
        "127.0.0.1:0", begin=execution.begin, finish=execution.finish, rendezvous=rendezvous
    )
    port = server.start()
    facts = execution._test_config.model_dump(mode="json")
    facts["workers"][0]["control_endpoint"] = f"127.0.0.1:{port}"
    facts["rendezvous_port"] = 0
    try:
        with pytest.raises((RuntimeError, grpc.RpcError, KeyboardInterrupt)):
            GraphStepControl.reserve(
                facts,
                monotonic_deadline=time.monotonic() + 2,
                require_current=interrupted if failure == "interrupt" else None,
            )
        assert len(bound_ports) == 1 and execution._rendezvous is None
        assert execution.config is None and execution._assignment_guard is None
        _assert_port_released(bound_ports[0])
    finally:
        server.close()


def test_cancelled_bootstrap_rpc_leaves_listener_owned_until_unused_expiry(execution):
    from expertkit_transport.transports.ascend_graph.rendezvous import GraphRendezvousRequest

    bound, release_reply = threading.Event(), threading.Event()

    def rendezvous(request, *, timeout):
        reply = execution.rendezvous(request, timeout=timeout)
        bound.set()
        assert release_reply.wait(2)
        return reply

    server = GraphControlServer(
        "127.0.0.1:0", begin=execution.begin, finish=execution.finish, rendezvous=rendezvous
    )
    port = server.start()
    facts = execution._test_config.model_dump(mode="json")
    facts["rendezvous_port"] = 0
    request = GraphRendezvousRequest.from_facts(facts)
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            method = channel.unary_unary(
                f"/{SERVICE}/Rendezvous",
                request_serializer=encode_message,
                response_deserializer=decode_message,
            )
            pending = method.future(request.model_dump(mode="json"), timeout=2)
            assert bound.wait(2)
            held_port = execution._rendezvous.port
            pending.cancel()
            _assert_port_held(held_port)
            execution._rendezvous.deadline = time.monotonic() - 1
            execution._expire_rendezvous()
            _assert_port_released(held_port)
            assert execution.config is None
    finally:
        release_reply.set()
        server.close()


def test_rank1_prepare_without_new_handshake_fails_before_admission(execution):
    with pytest.raises(GraphControlRejected, match="Rendezvous protocol 1"):
        execution.begin(request_for(execution), timeout=2)
    assert execution.config is None and execution._pending is None
    assert execution._assignment_guard is None


def test_reservation_uses_rpc_remaining_budget_without_renewing_it(execution):
    from expertkit_transport.transports.ascend_graph.rendezvous import GraphRendezvousRequest

    facts = execution._test_config.model_dump(mode="json")
    facts["rendezvous_port"] = 0
    request = GraphRendezvousRequest.from_facts(facts).model_dump(mode="json")
    before = time.monotonic()
    reply = execution.rendezvous(request, timeout=0.5)
    after = time.monotonic()
    deadline = execution._rendezvous.deadline
    assert before + 0.5 <= deadline <= after + 0.5
    assert execution.rendezvous(request, timeout=2) == reply
    assert execution._rendezvous.deadline == deadline


@pytest.mark.parametrize(
    "update",
    [
        {"version": True},
        {"version": 2},
        {"worker_start_id": "stale"},
        {"instance_id": 2},
        {"rendezvous_port": True},
        {"unknown": 1},
    ],
)
def test_invalid_reservation_rejected_before_listener_or_device_ownership(execution, update):
    from expertkit_transport.transports.ascend_graph.rendezvous import GraphRendezvousRequest

    facts = execution._test_config.model_dump(mode="json")
    facts["rendezvous_port"] = 0
    request = GraphRendezvousRequest.from_facts(facts).model_dump(mode="json")
    execution._guard_assignments = Mock()
    with pytest.raises(GraphControlRejected):
        execution.rendezvous({**request, **update}, timeout=2)
    assert execution._rendezvous is None and execution._rendezvous_binding is None
    assert execution._pending is None and execution.config is None
    execution._guard_assignments.assert_not_called()
