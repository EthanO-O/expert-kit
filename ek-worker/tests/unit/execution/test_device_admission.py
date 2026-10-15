"""Queued cross-path work must never enter after unsafe device completion."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import expertkit_worker.execution.decode_graph as graph_module
from expertkit_worker.backends import BackendFatalError, BackendFatalReason
from expertkit_worker.execution.admission import DeviceAdmission
from expertkit_worker.execution.decode_graph import AscendDecodeGraphExecution
from expertkit_worker.execution.executor import WorkerExecutor


@pytest.mark.parametrize("graph_fails", [False, True])
def test_unsafe_owner_poisons_admission_before_waiting_opposite_path_enters(graph_fails):
    gate = DeviceAdmission()
    owner_entered = threading.Event()
    waiter_started = threading.Event()
    fail_owner = threading.Event()
    waiter_rejected = threading.Event()
    operations = []

    def owner():
        error = (
            RuntimeError("graph failed")
            if graph_fails
            else BackendFatalError(BackendFatalReason.ASYNC_EXECUTION, "ordinary device failed")
        )
        with pytest.raises(type(error)), gate.use(poison_on_error=graph_fails):
            owner_entered.set()
            assert fail_owner.wait(2)
            raise error

    def waiter():
        assert owner_entered.wait(2)
        waiter_started.set()
        with (
            pytest.raises(BackendFatalError, match="quarantined"),
            gate.use(poison_on_error=not graph_fails),
        ):
            operations.append("unsafe device operation")
        waiter_rejected.set()

    first = threading.Thread(target=owner)
    second = threading.Thread(target=waiter)
    first.start()
    second.start()
    assert waiter_started.wait(2)
    fail_owner.set()
    assert waiter_rejected.wait(2)
    first.join(timeout=2)
    second.join(timeout=2)
    assert not first.is_alive() and not second.is_alive()
    assert operations == []


class Result:
    """Fake external slot completion storage."""

    def release(self):
        pass


class Session:
    """Fake external HCCL group initialization."""

    def __init__(self, *args, **kwargs):
        pass

    def buffers(self, capacity):
        pass


class Stream:
    """Fake external NPU stream completion."""

    def synchronize(self):
        pass


async def _callsite_scenario(graph_first):
    admission = DeviceAdmission()
    entered = {name: threading.Event() for name in ("graph", "ordinary")}
    release = {name: threading.Event() for name in ("graph", "ordinary")}
    attempts = {name: threading.Event() for name in ("graph", "ordinary")}

    def prepare():
        entered["graph"].set()
        assert release["graph"].wait(3)
        return object()

    class Slot:
        def execute(self, *args, **kwargs):
            entered["ordinary"].set()
            assert release["ordinary"].wait(3)
            return Result()

    graph = AscendDecodeGraphExecution.__new__(AscendDecodeGraphExecution)
    graph.config = SimpleNamespace(capture_sizes=(1,))
    graph._admission = admission
    graph._device = torch.device("cpu")
    graph._stream = None
    graph._prepare_weights = prepare
    graph._hidden_dim = 4
    graph._top_k = 2
    graph._poisoned = False
    ordinary = WorkerExecutor.__new__(WorkerExecutor)
    ordinary._device_admission = admission
    ordinary._backend = object()
    ordinary._clock = None
    with ThreadPoolExecutor(max_workers=1) as pool:
        ordinary._executor = pool

        async def ordinary_call():
            attempts["ordinary"].set()
            await ordinary._run_slot(Slot(), object(), None)

        def graph_call():
            attempts["graph"].set()
            graph._run("prepare", 0, threading.Event())

        first = "graph" if graph_first else "ordinary"
        second = "ordinary" if graph_first else "graph"

        def start(name):
            return asyncio.create_task(
                asyncio.to_thread(graph_call) if name == "graph" else ordinary_call()
            )

        first_task = start(first)
        assert await asyncio.to_thread(entered[first].wait, 3)
        second_task = start(second)
        assert await asyncio.to_thread(attempts[second].wait, 3)
        assert admission._lock.locked()
        assert not entered[second].is_set()
        release[first].set()
        assert await asyncio.to_thread(entered[second].wait, 3)
        release[second].set()
        await asyncio.wait_for(asyncio.gather(first_task, second_task), 3)


@pytest.mark.parametrize("graph_first", [False, True])
def test_real_call_sites_hold_shared_admission_through_completion(monkeypatch, graph_first):
    fake_npu = SimpleNamespace(
        set_device=lambda device: None,
        Stream=lambda **kwargs: Stream(),
        stream=lambda stream: nullcontext(),
    )
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    monkeypatch.setattr(graph_module, "HcclGraphSession", Session)
    asyncio.run(_callsite_scenario(graph_first))
