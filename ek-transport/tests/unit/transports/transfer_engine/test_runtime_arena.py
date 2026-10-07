"""Registered arena bounds and generation reuse."""

import asyncio
import gc
import threading
import time
import weakref

import pytest
import torch

from expertkit_transport.transports.base import WorkerEndpointConfig
from expertkit_transport.transports.transfer_engine.arena import TransferArena
from expertkit_transport.transports.transfer_engine.session import TransferEngineSession


def endpoint_config() -> WorkerEndpointConfig:
    return WorkerEndpointConfig(7, 4, 8, 4, 3, 2, torch.float32)


def test_arena_uses_one_contiguous_allocation_with_typed_slot_views() -> None:
    arena = TransferArena(endpoint_config(), device="cpu", capacity=2)
    descriptor = arena.descriptor

    descriptor.validate(endpoint_config())
    assert arena.slab.data_ptr() == descriptor.base_address
    assert descriptor.addresses(0) == tuple(
        tensor.data_ptr()
        for tensor in (
            arena.slots[0].hidden_states,
            arena.slots[0].expert_ids,
            arena.slots[0].routing_weights,
            arena.slots[0].partial_output,
        )
    )
    assert descriptor.addresses(1) == tuple(
        tensor.data_ptr()
        for tensor in (
            arena.slots[1].hidden_states,
            arena.slots[1].expert_ids,
            arena.slots[1].routing_weights,
            arena.slots[1].partial_output,
        )
    )

    first = arena.take()
    assert first is arena.slots[0]
    assert first.generation == 1
    arena.put(first)
    again = arena.take()
    assert again is first
    assert again.generation == 2
    arena.put(again)
    arena.close()


def test_arena_rejects_capacity_above_the_wire_protocol_limit() -> None:
    with pytest.raises(ValueError, match="capacity cannot exceed 4096"):
        TransferArena(endpoint_config(), device="cpu", capacity=4097)


def test_session_rejects_two_active_sequences_for_the_same_client_slot() -> None:
    arena = TransferArena(endpoint_config(), device="cpu", capacity=2)
    session = TransferEngineSession(
        client_epoch="frontend-epoch",
        session_nonce="worker-nonce",
        target_session_id="frontend:19001",
        target_runtime_generation="frontend-generation",
        backend="ascend_direct",
        arena=arena.descriptor,
    )
    session.claim(sequence=1, slot_index=0, generation=1)
    with pytest.raises(Exception, match="slot is already active"):
        session.claim(sequence=2, slot_index=0, generation=2)
    session.release(1)
    session.claim(sequence=2, slot_index=0, generation=2)
    session.release(2)
    arena.close()


class _NativeEngine:
    def __init__(self) -> None:
        self.registered: dict[int, int] = {}
        self.unregister_status = 0

    def initialize(self, *args: object) -> int:
        assert args == ("localhost:19000", "P2PHANDSHAKE", "ascend_direct", "")
        return 0

    def get_configured_backend(self) -> str:
        return "ascend_direct"

    def get_rpc_port(self) -> int:
        return 19000

    def invalidate_drained_ascend_segment(self, name: str) -> int:
        return 0

    def register_memory(self, address: int, length: int) -> int:
        self.registered[address] = length
        return 0

    def unregister_memory(self, address: int) -> int:
        if not self.unregister_status:
            self.registered.pop(address)
        return self.unregister_status


def _cpu_runtime(engine: _NativeEngine, *, budget: int = 64):
    from expertkit_transport.transports.transfer_engine.runtime import (
        TransferEngineRuntime,
        TransferEngineRuntimeConfig,
    )

    # CPU storage substitutes only for the external native-memory boundary;
    # production configuration still requires an indexed NPU device.
    config = object.__new__(TransferEngineRuntimeConfig)
    for name, value in dict(
        segment_name="localhost:19000",
        device=torch.device("cpu"),
        max_workers=2,
        max_registered_bytes=budget,
        client_max_in_flight=None,
        ascend_receive_fence="stream",
    ).items():
        object.__setattr__(config, name, value)
    return TransferEngineRuntime(config, engine=engine)


def test_retired_arena_remains_owned_and_charged_until_runtime_shutdown() -> None:
    async def scenario() -> None:
        engine = _NativeEngine()
        runtime = _cpu_runtime(engine)
        tensor = torch.empty(16, dtype=torch.float32)
        owner = weakref.ref(tensor)
        await runtime.register_tensor(tensor)
        await runtime.retire_tensor(tensor)
        assert runtime.registered_bytes == 64
        with pytest.raises(RuntimeError, match="retired"):
            runtime._require_registered(tensor.data_ptr(), 4)
        with pytest.raises(ValueError, match="budget"):
            await runtime.register_tensor(torch.empty(1))
        del tensor
        gc.collect()
        assert owner() is not None
        assert len(engine.registered) == 1
        await runtime.close()
        assert runtime.registered_bytes == 0
        assert not engine.registered
        gc.collect()
        assert owner() is None

    asyncio.run(scenario())


def test_failed_final_deregistration_retains_native_allocation() -> None:
    async def scenario() -> None:
        engine = _NativeEngine()
        runtime = _cpu_runtime(engine)
        tensor = torch.empty(4)
        await runtime.register_tensor(tensor)
        engine.unregister_status = -1
        with pytest.raises(RuntimeError, match="deregistration"):
            await runtime.close()
        assert runtime.registered_bytes == 16
        assert runtime._quarantine_reason
        assert tensor.data_ptr() in engine.registered

    asyncio.run(scenario())


@pytest.mark.parametrize("interruption", ["cancel", "deadline"])
def test_native_registration_commits_before_interruption_escapes(interruption: str) -> None:
    from expertkit_transport.errors import TransportError

    async def scenario() -> None:
        engine = _NativeEngine()
        runtime = _cpu_runtime(engine)
        await runtime.start()
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        release = threading.Event()
        original = engine.register_memory

        def blocked(address: int, length: int) -> int:
            loop.call_soon_threadsafe(started.set)
            assert release.wait(3)
            return original(address, length)

        engine.register_memory = blocked
        tensor = torch.empty(4)
        deadline = time.monotonic() + 1 if interruption == "cancel" else time.monotonic() + 0.02
        task = asyncio.create_task(runtime.register_tensor(tensor, monotonic_deadline=deadline))
        try:
            await asyncio.wait_for(started.wait(), 2)
            if interruption == "cancel":
                task.cancel()
                await asyncio.sleep(0)
                task.cancel()
            else:
                # Advance the injected deadline clock after native submission;
                # no wall-clock sleep is needed to establish expiry.
                runtime._clock = lambda: deadline + 1
            assert not task.done()
            assert tensor.data_ptr() in runtime._pending_registrations
            release.set()
            with pytest.raises(
                asyncio.CancelledError if interruption == "cancel" else TransportError
            ):
                await task
            assert tensor.data_ptr() in engine.registered
            assert runtime.registered_bytes == 16
            assert not runtime._pending_registrations
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            await runtime.close()

    asyncio.run(scenario())


def test_failed_native_transfer_quarantines_registration_and_rejects_retry() -> None:
    async def scenario() -> None:
        engine = _NativeEngine()
        runtime = _cpu_runtime(engine)
        tensor = torch.empty(4)
        await runtime.register_tensor(tensor)

        def fail(*args: object) -> int:
            raise RuntimeError("submit failed after native ownership transfer")

        engine.batch_transfer_sync_read = fail
        with pytest.raises(RuntimeError, match="submit failed"):
            await runtime.batch_read(
                "peer:19001", (tensor,), (1234,), (16,), monotonic_deadline=time.monotonic() + 5
            )
        assert runtime._quarantine_reason
        await runtime.close()
        assert runtime.registered_bytes == 16
        assert tensor.data_ptr() in engine.registered

    asyncio.run(scenario())


def test_internally_cancelled_native_waiter_quarantines_without_spinning() -> None:
    async def scenario() -> None:
        runtime = _cpu_runtime(_NativeEngine())
        waiter = asyncio.get_running_loop().create_future()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(runtime._await_native_terminal(waiter), 1)
        assert runtime._quarantine_reason

    asyncio.run(scenario())
