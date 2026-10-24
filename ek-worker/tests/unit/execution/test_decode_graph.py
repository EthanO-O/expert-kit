"""Validate graph command admission without touching an NPU."""

import asyncio
import socket
import threading
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
    config = AscendGraphConfig(
        session_id="worker-cpu-session-001",
        instance_id=1,
        worker_id="worker",
        weight_version="revision",
        control_endpoint="127.0.0.1:12345",
        rendezvous_host="127.0.0.1",
        rendezvous_port=12346,
        layer_ids=(1,),
        capture_sizes=(1, 4),
        timeout_seconds=2.0,
    )
    owner = AscendDecodeGraphExecution(
        config,
        worker_start_id="start-one",
        device=torch.device("cpu"),
        hidden_dim=4,
        top_k=2,
        admission=DeviceAdmission(),
        prepare_weights=lambda: pytest.fail("malformed commands must not prepare weights"),
        terminate=lambda code: pytest.fail(f"unexpected process termination: {code}"),
    )
    try:
        yield owner
    finally:
        asyncio.run(owner.close())


def request_for(execution, **updates):
    """Construct protocol version one preparation metadata."""
    return {
        "version": 1,
        "identity": execution.config.identity(),
        "sequence": 1,
        "phase": "prepare",
        "capacity": 0,
        "worker_start_id": None,
        "ready_revision": None,
        **updates,
    }


@pytest.mark.parametrize(
    "updates",
    [
        {"version": True},
        {"version": 2},
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
        {"identity": []},
    ],
)
def test_invalid_command_rejected_before_admission(execution, updates):
    with pytest.raises(GraphControlRejected):
        execution.begin(request_for(execution, **updates), timeout=2)
    assert execution._sequence == 0
    assert execution._pending is None


def test_strict_identity_rejects_bool_instance(execution):
    request = request_for(execution)
    request["identity"]["instance_id"] = True
    with pytest.raises(ValueError, match="identity"):
        execution._validate_request(request)


def test_finish_requires_exact_pending_command_and_clears_once(execution):
    request = request_for(execution)
    pending = Future()
    pending.set_result(None)
    execution._pending = pending
    execution._pending_request = request
    with pytest.raises(GraphControlRejected):
        execution.finish({**request, "sequence": 2}, timeout=2)
    assert execution._pending is pending
    assert execution.finish(request, timeout=2) == {"state": "complete", "sequence": 1}
    with pytest.raises(GraphControlRejected):
        execution.finish(request, timeout=2)


def test_duplicate_finish_cannot_clear_the_next_step(execution):
    request = request_for(execution)
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
    request = request_for(execution)
    pending = Future()
    execution._pending = pending
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
            execution._pending_request = request_for(execution)
            assert finish(request_for(execution), timeout=2) == {
                "state": "complete",
                "sequence": 1,
            }
            with pytest.raises(grpc.RpcError) as repeated:
                finish(request_for(execution), timeout=2)
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
    values = execution.config.model_dump(exclude={"worker_id", "control_endpoint"})
    values.update(
        mode="full",
        layer_ids=tuple(range(43)),
        capture_sizes=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048),
        workers=[
            dict(
                worker_id=f"worker-{i}",
                control_endpoint=f"127.0.0.1:{18000 + i}",
                expert_start=i * 32,
                expert_end=(i + 1) * 32,
            )
            for i in range(8)
        ],
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
    return AscendGraphConfig.model_validate(values)


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
        assert execution._validate_request(request) == ("capture", 4)
        execution._graphs[(phase, 4)] = object()
        with pytest.raises(ValueError, match="already been captured"):
            execution._validate_request(request)
        assert execution._validate_request({**request, "phase": "replay"}) == ("replay", 4)
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
