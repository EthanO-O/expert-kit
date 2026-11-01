"""Exercise metadata-only graph discovery with an explicitly controlled stream."""

import queue
import threading
import time
from types import SimpleNamespace

import pytest
from expertkit_proto.ek.control.v2 import lifecycle_pb2

from expertkit_transport.transports.ascend_graph import topology
from expertkit_transport.transports.ascend_graph.config import AscendGraphConfig


class Stream:
    def __init__(self):
        self.messages = queue.Queue()
        self.cancelled = threading.Event()

    def __iter__(self):
        while True:
            message, processed = self.messages.get(timeout=3)
            if isinstance(message, BaseException):
                processed.set()
                raise message
            if message is None:
                processed.set()
                return
            try:
                yield message
            finally:
                processed.set()

    def put(self, message):
        processed = threading.Event()
        self.messages.put((message, processed))
        return processed

    def cancel(self):
        self.cancelled.set()
        self.put(None)


def replica(worker="b", *, version="v1", start="start", capacity=12, device="npu:0"):
    return lifecycle_pb2.WorkerRoute(
        worker_id=worker,
        start_id=start,
        computation_endpoint=f"{worker}:5000",
        graph_control_endpoint=f"{worker}:5001",
        device=device,
        transport_type=lifecycle_pb2.WORKER_TRANSPORT_GRPC,
        max_active_batches=1,
        max_pending_batches=1,
        max_batch_tokens=capacity,
        weight_version=version,
    )


def message(version, entries, *, previous=None):
    value = lifecycle_pb2.TopologyMessage()
    part = value.snapshot if previous is None else value.update
    part.instance_id = 1
    part.topology_version = version
    part.part_count = 1
    if previous is not None:
        part.previous_version = previous
    target = part.routes if previous is None else part.changes
    for (layer, expert), replicas in entries.items():
        route = target.add(layer_id=layer, expert_id=expert)
        route.replicas.extend(replicas)
    return value


@pytest.fixture
def stream(monkeypatch):
    stream = Stream()
    channel = SimpleNamespace(close=lambda: stream.cancelled.set())
    monkeypatch.setattr(topology.grpc, "insecure_channel", lambda *_: channel)
    monkeypatch.setattr(
        topology.lifecycle_pb2_grpc,
        "TopologyServiceStub",
        lambda _: SimpleNamespace(WatchTopology=lambda *a, **k: stream),
    )
    return stream


def subscription():
    return topology.GraphTopologySubscription(
        "controller:5002", instance_id=1, num_layers=2, experts_per_layer=2
    )


def full_routes():
    return {
        (0, 0): [replica("b"), replica("a")],
        (0, 1): [replica("b")],
        (1, 0): [replica("b")],
        (1, 1): [replica("b")],
    }


def start(subscription):
    return subscription.start(
        layer_ids=(0, 1), max_capture_size=12, monotonic_deadline=time.monotonic() + 2
    )


def config(result):
    version, weight, workers = result
    return AscendGraphConfig(
        session_id="topology-test-session",
        instance_id=1,
        model_name="native-model",
        weight_version=weight,
        source_topology_version=version,
        model=dict(
            num_layers=2,
            experts_per_layer=2,
            hidden_dim=16,
            expert_intermediate_dim=32,
            top_k=1,
            activation_dtype="bf16",
            weight_dtype="bf16",
            activation="silu",
            expert_compute="swiglu",
            swiglu_limit=0.0,
            quantization=None,
        ),
        mode="full",
        capture_sizes=(3, 7, 12),
        decode_capture_sizes=(3, 7),
        timeout_seconds=2.0,
        rendezvous_host="frontend",
        rendezvous_port=5003,
        layer_ids=(0, 1),
        workers=workers,
    )


def test_freezes_one_replica_and_empty_layers_and_ignores_unselected_updates(stream):
    stream.put(message(1, full_routes()))
    sub = subscription()
    try:
        runtime = config(start(sub))
        assert [worker.worker_id for worker in runtime.workers] == ["a", "b"]
        assert runtime.workers[0].experts_by_layer == ((0,), ())
        assert runtime.workers[1].experts_by_layer == ((1,), (0, 1))
        digest = runtime.assignment_id
        changed = full_routes()
        changed[(0, 1)].insert(0, replica("0-earlier"))
        changed[(0, 0)].append(replica("z-unused", version="different"))
        assert stream.put(message(2, changed)).wait(2)
        sub.require_current(runtime)
        assert runtime.assignment_id == digest
    finally:
        sub.close()
    assert stream.cancelled.is_set()
    assert not sub._thread.is_alive()


def test_missing_coverage_waits_for_ready_update(stream):
    initial = full_routes()
    del initial[(1, 1)]
    stream.put(message(1, initial))
    stream.put(message(2, {(1, 1): [replica()]}, previous=1))
    sub = subscription()
    try:
        assert start(sub)[0] == 2
    finally:
        sub.close()


@pytest.mark.parametrize(
    "changed",
    [
        [],
        [replica("a", start="restarted")],
        [replica("a", version="v2")],
        [replica("a", device="npu:1")],
    ],
)
def test_selected_changes_are_terminal_even_if_route_returns(stream, changed):
    original = full_routes()
    stream.put(message(1, original))
    sub = subscription()
    runtime = config(start(sub))
    assert stream.put(message(2, {(0, 0): changed}, previous=1)).wait(2)
    sub._thread.join(timeout=2)
    stream.put(message(3, original))
    with pytest.raises(RuntimeError, match="restart required"):
        sub.require_current(runtime)
    sub.close()


@pytest.mark.parametrize("fault", [None, RuntimeError("disconnected"), "gap"])
def test_disconnect_and_gap_fail_closed(stream, fault):
    stream.put(message(1, full_routes()))
    sub = subscription()
    runtime = config(start(sub))
    if fault == "gap":
        fault = message(3, {(0, 0): [replica("a")]}, previous=2)
    assert stream.put(fault).wait(2)
    sub._thread.join(timeout=2)
    with pytest.raises(RuntimeError, match="restart required"):
        sub.require_current(runtime)
    sub.close()


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"capacity": 7}, "max_batch_tokens"),
        ({"version": ""}, "empty weight_version"),
        ({"version": "v2"}, "disagree"),
        ({"device": "cuda:0"}, "device"),
    ],
)
def test_initial_selection_checks_facts_before_prepare(stream, change, reason):
    routes = full_routes()
    routes[(0, 0)] = [replica("a", **change)]
    stream.put(message(1, routes))
    sub = subscription()
    with pytest.raises(ValueError, match=reason):
        start(sub)
    assert stream.cancelled.is_set()
    assert not sub._thread.is_alive()


def test_initial_missing_ready_deadline_closes_thread(stream):
    sub = subscription()
    with pytest.raises(TimeoutError, match="READY coverage"):
        sub.start(layer_ids=(0, 1), max_capture_size=12, monotonic_deadline=time.monotonic())
    assert not sub._thread.is_alive()
    assert stream.cancelled.is_set()


def test_coarse_admission_does_not_rescan_unchanged_selected_routes(stream, monkeypatch):
    stream.put(message(1, full_routes()))
    sub = subscription()
    runtime = config(start(sub))
    checks = []
    original = sub._check_selected

    def check():
        checks.append(sub._version)
        original()

    monkeypatch.setattr(sub, "_check_selected", check)
    try:
        sub.require_current(runtime)
        sub.require_current(runtime)
        assert checks == []
        assert stream.put(
            message(2, {(0, 0): [replica("a"), replica("earlier-unused")]}, previous=1)
        ).wait(2)
        assert checks == [2]
        sub.require_current(runtime)
        assert checks == [2]
    finally:
        sub.close()


def test_selected_update_latches_failure_before_releasing_admission_lock(stream, monkeypatch):
    stream.put(message(1, full_routes()))
    sub = subscription()
    runtime = config(start(sub))
    published = threading.Event()
    original_notify = sub._condition.notify_all
    observed = []

    def notify():
        if sub._version == 2:
            observed.append(sub._failure)
            published.set()
        original_notify()

    monkeypatch.setattr(sub._condition, "notify_all", notify)
    stream.put(message(2, {(0, 0): []}, previous=1))
    assert published.wait(2)
    assert observed and isinstance(observed[0], RuntimeError)
    with pytest.raises(RuntimeError, match="restart required"):
        sub.require_current(runtime)
    sub.close()
