"""Ascend control and memory ownership over a fake native boundary."""

from __future__ import annotations

import asyncio
import ctypes
import math
import time
from dataclasses import dataclass

import grpc
import pytest
import torch

from expertkit_transport.batches import WorkerBatch
from expertkit_transport.errors import TransportError, TransportErrorCode
from expertkit_transport.transports.base import (
    BatchBufferConfig,
    ReceivedBatch,
    WorkerBatchBuffers,
    WorkerEndpointConfig,
)
from expertkit_transport.transports.transfer_engine import (
    TransferEngineWorkerBatchReceiver,
    TransferEngineWorkerTransport,
)


def endpoint_config() -> WorkerEndpointConfig:
    return WorkerEndpointConfig(7, 4, 8, 4, 3, 2, torch.float32)


def worker_batch(*, offset: float = 0) -> WorkerBatch:
    return WorkerBatch(
        instance_id=7,
        layer_id=2,
        topology_version=11,
        hidden_states=torch.tensor(
            [[1 + offset, 2 + offset, 3 + offset], [4, 5, 6], [7, 8, 9]],
            dtype=torch.float32,
        ),
        token_indices=torch.tensor([2, 0], dtype=torch.int64),
        expert_ids=torch.tensor([[1, -1], [0, 3]], dtype=torch.int32),
        routing_weights=torch.tensor([[0.25, 0], [0.5, 0.5]], dtype=torch.float32),
        distinct_expert_ids=(0, 1, 3),
    )


class _FakeLink:
    def __init__(self) -> None:
        self.regions: dict[str, list[tuple[int, int, torch.Tensor]]] = {}
        self.operations: list[tuple[str, str, int]] = []
        self.write_started = asyncio.Event()
        self.allow_write = asyncio.Event()
        self.block_writes = False

    def register(self, session: str, tensor: torch.Tensor) -> None:
        self.regions.setdefault(session, []).append(
            (tensor.data_ptr(), tensor.numel() * tensor.element_size(), tensor)
        )

    def unregister(self, session: str, tensor: torch.Tensor) -> None:
        regions = self.regions[session]
        regions[:] = [entry for entry in regions if entry[2] is not tensor]

    def require_remote(self, session: str, address: int, length: int) -> None:
        assert any(
            base <= address and address + length <= base + capacity
            for base, capacity, _tensor in self.regions[session]
        )


class _FakeRuntime:
    def __init__(
        self,
        link: _FakeLink,
        session_id: str,
        *,
        backend: str = "ascend_direct",
        generation: str = "runtime-generation-a",
        client_max_in_flight: int | None = None,
    ) -> None:
        self._link = link
        self._session_id = session_id
        self._backend = backend
        self._generation = generation
        self.client_max_in_flight = client_max_in_flight
        self._closed = False
        self.quarantine_reason: str | None = None
        self.invalidated_sessions: list[str] = []
        self.disconnected_sessions: list[str] = []
        self.acquire_count = 0
        self.retired: list[torch.Tensor] = []

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def generation(self) -> str:
        return self._generation

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("fake Transfer Engine runtime is closed")

    async def register_tensor(
        self,
        tensor: torch.Tensor,
        *,
        monotonic_deadline: float = math.inf,
    ) -> None:
        del monotonic_deadline
        self._link.register(self._session_id, tensor)

    async def unregister_tensor(
        self,
        tensor: torch.Tensor,
        *,
        monotonic_deadline: float = math.inf,
    ) -> None:
        del monotonic_deadline
        self._link.unregister(self._session_id, tensor)

    async def retire_tensor(self, tensor: torch.Tensor) -> None:
        self.retired.append(tensor)

    async def wait_event(
        self,
        event: torch.Event,
        *,
        monotonic_deadline: float,
    ) -> None:
        del event, monotonic_deadline

    async def acquire_remote_writes(
        self,
        *,
        monotonic_deadline: float,
    ) -> None:
        assert monotonic_deadline > time.monotonic() or math.isinf(monotonic_deadline)
        self.acquire_count += 1

    async def batch_read(
        self,
        target_session: str,
        local_tensors: tuple[torch.Tensor, ...],
        remote_addresses: tuple[int, ...],
        lengths: tuple[int, ...],
        *,
        monotonic_deadline: float,
    ) -> None:
        assert monotonic_deadline > time.monotonic()
        self._link.operations.append(("read", target_session, len(lengths)))
        for local, remote, length in zip(local_tensors, remote_addresses, lengths, strict=True):
            self._link.require_remote(target_session, remote, length)
            ctypes.memmove(local.data_ptr(), remote, length)

    async def batch_write(
        self,
        target_session: str,
        local_tensors: tuple[torch.Tensor, ...],
        remote_addresses: tuple[int, ...],
        lengths: tuple[int, ...],
        *,
        monotonic_deadline: float,
    ) -> None:
        assert math.isinf(monotonic_deadline)
        self._link.operations.append(("write", target_session, len(lengths)))
        self._link.write_started.set()
        if self._link.block_writes:
            await self._link.allow_write.wait()
        for local, remote, length in zip(local_tensors, remote_addresses, lengths, strict=True):
            self._link.require_remote(target_session, remote, length)
            ctypes.memmove(remote, local.data_ptr(), length)

    async def disconnect_remote_session(self, target_session: str) -> None:
        self.disconnected_sessions.append(target_session)

    async def invalidate_remote_session(
        self,
        target_session: str,
        *,
        monotonic_deadline: float = math.inf,
    ) -> None:
        assert math.isinf(monotonic_deadline)
        self.invalidated_sessions.append(target_session)

    async def close(self) -> None:
        self._closed = True

    def ensure_healthy(self) -> None:
        if self.quarantine_reason is not None:
            raise TransportError(
                TransportErrorCode.UNAVAILABLE,
                retryable=False,
                diagnostic="fake Transfer Engine runtime is quarantined",
            )

    def quarantine(self, diagnostic: str) -> None:
        self.quarantine_reason = diagnostic


@dataclass(slots=True)
class _RunningPair:
    receiver: TransferEngineWorkerBatchReceiver
    transport: TransferEngineWorkerTransport
    buffers: WorkerBatchBuffers
    link: _FakeLink
    client_runtime: _FakeRuntime
    worker_runtime: _FakeRuntime

    async def close(self) -> None:
        try:
            await self.transport.close()
        finally:
            try:
                await self.receiver.close()
            finally:
                self.buffers.close()
                await self.client_runtime.close()
                await self.worker_runtime.close()


async def start_pair(
    *,
    max_active_batches: int = 1,
    max_pending_batches: int = 1,
    max_in_flight: int = 2,
    client_max_in_flight: int | None = None,
    session_close_grace_secs: float = 0.2,
    backend: str = "ascend_direct",
    tracer=None,
    interceptors=(),
) -> _RunningPair:
    link = _FakeLink()
    client_runtime = _FakeRuntime(
        link, "client:19001", backend=backend, client_max_in_flight=client_max_in_flight
    )
    worker_runtime = _FakeRuntime(link, "worker:19002", backend=backend)
    config = endpoint_config()
    receiver = TransferEngineWorkerBatchReceiver(
        "127.0.0.1:0",
        config,
        runtime=worker_runtime,
        worker_start_id="worker-start-a",
        max_active_batches=max_active_batches,
        max_pending_batches=max_pending_batches,
        session_close_grace_secs=session_close_grace_secs,
        owns_runtime=False,
        tracer=tracer,
        interceptors=interceptors,
    )
    buffers = receiver.create_batch_buffers(
        BatchBufferConfig(4, 3, 2, torch.float32, torch.device("cpu"))
    )
    await receiver.start()
    transport = TransferEngineWorkerTransport(
        f"127.0.0.1:{receiver.bound_port}",
        config,
        max_in_flight=max_in_flight,
        device="cpu",
        runtime=client_runtime,
        expected_worker_start_id="worker-start-a",
    )
    await transport.start()
    return _RunningPair(
        receiver,
        transport,
        buffers,
        link,
        client_runtime,
        worker_runtime,
    )


async def complete(received: ReceivedBatch, *, multiplier: float = 2) -> None:
    partial_output = received.batch.hidden_states * multiplier
    destination = received.output_destination
    assert destination is not None
    destination.copy_(partial_output)
    received.release_input()
    await received.complete(destination)


@pytest.mark.parametrize(("client_limit", "expected"), [(1, 1), (2, 2), (8, 2), (None, 2)])
def test_direct_client_limit_bounds_registered_slots_and_admission(
    client_limit: int | None, expected: int
) -> None:
    async def scenario() -> None:
        pair = await start_pair(client_max_in_flight=client_limit)
        try:
            session = next(iter(pair.receiver._sessions.values()))
            assert session.arena.slot_count == expected
            for _ in range(expected):
                await pair.transport._acquire(time.monotonic() + 1)
            assert pair.transport._semaphore.locked()
        finally:
            await pair.close()

    asyncio.run(scenario())


def test_worker_pulls_three_inputs_and_writes_one_output() -> None:
    async def scenario() -> None:
        pair = await start_pair()
        batch = worker_batch()
        output = torch.empty((2, 3), dtype=torch.float32)
        try:
            submission = asyncio.create_task(
                pair.transport.execute(
                    batch,
                    output,
                    monotonic_deadline=time.monotonic() + 5,
                )
            )
            received = await pair.receiver.receive()
            batch.hidden_states.fill_(99)
            assert received.batch.hidden_states.tolist() == [[7, 8, 9], [1, 2, 3]]
            await complete(received)
            await submission

            torch.testing.assert_close(
                output,
                torch.tensor([[14, 16, 18], [2, 4, 6]], dtype=torch.float32),
            )
            assert pair.link.operations == [
                ("read", "client:19001", 3),
                ("write", "client:19001", 1),
            ]
        finally:
            await pair.close()

    asyncio.run(scenario())


def test_unsafe_rejection_quarantines_worker_slot_and_registered_arena() -> None:
    async def scenario() -> None:
        pair = await start_pair(max_active_batches=1, max_pending_batches=1)
        submission = asyncio.create_task(
            pair.transport.execute(
                worker_batch(),
                torch.empty((2, 3), dtype=torch.float32),
                monotonic_deadline=time.monotonic() + 5,
            )
        )
        try:
            received = await pair.receiver.receive()
            worker_slot = received.slot  # type: ignore[attr-defined]
            await received.reject_unsafe(
                TransportError(
                    TransportErrorCode.UNAVAILABLE,
                    retryable=True,
                    diagnostic="device completion is ambiguous",
                )
            )
            with pytest.raises(TransportError) as captured:
                await submission

            assert captured.value.code is TransportErrorCode.UNAVAILABLE
            assert captured.value.retryable is False
            assert pair.worker_runtime.quarantine_reason is not None
            assert worker_slot.in_use
            assert worker_slot in pair.receiver._quarantined_slots  # type: ignore[attr-defined]

            assert pair.receiver._registered  # type: ignore[attr-defined]
            assert pair.link.regions[pair.worker_runtime.session_id]
        finally:
            if not submission.done():
                submission.cancel()
            await asyncio.gather(submission, return_exceptions=True)
            server = pair.receiver._server  # type: ignore[attr-defined]
            if server is not None:
                await server.stop(0)
            channel = pair.transport._channel  # type: ignore[attr-defined]
            if channel is not None:
                await channel.close()
            pair.buffers.close()
            await pair.client_runtime.close()
            await pair.worker_runtime.close()

    asyncio.run(scenario())


def test_slot_is_not_reused_until_output_write_is_terminal() -> None:
    async def scenario() -> None:
        pair = await start_pair(max_in_flight=2, client_max_in_flight=1)
        pair.link.block_writes = True
        first_output = torch.empty((2, 3), dtype=torch.float32)
        second_output = torch.empty((2, 3), dtype=torch.float32)
        first = asyncio.create_task(
            pair.transport.execute(
                worker_batch(),
                first_output,
                monotonic_deadline=time.monotonic() + 5,
            )
        )
        second: asyncio.Task[None] | None = None
        completion: asyncio.Task[None] | None = None
        try:
            received = await pair.receiver.receive()
            destination = received.output_destination
            assert destination is not None
            destination.copy_(received.batch.hidden_states * 2)
            received.release_input()
            completion = asyncio.create_task(received.complete(destination))
            await asyncio.wait_for(pair.link.write_started.wait(), 2)

            second = asyncio.create_task(
                pair.transport.execute(
                    worker_batch(offset=10),
                    second_output,
                    monotonic_deadline=time.monotonic() + 5,
                )
            )
            await asyncio.sleep(0)
            assert not first.done()
            assert not second.done()

            pair.link.block_writes = False
            pair.link.allow_write.set()
            await completion
            await first
            second_received = await pair.receiver.receive()
            await complete(second_received)
            await second
        finally:
            pair.link.block_writes = False
            pair.link.allow_write.set()
            for task in (first, second, completion):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(
                *(task for task in (first, second, completion) if task is not None),
                return_exceptions=True,
            )
            await pair.close()

    asyncio.run(scenario())


def test_open_session_rejects_a_stale_worker_start_id() -> None:
    async def scenario() -> None:
        link = _FakeLink()
        worker_runtime = _FakeRuntime(link, "worker:19002")
        client_runtime = _FakeRuntime(link, "client:19001")
        receiver = TransferEngineWorkerBatchReceiver(
            "127.0.0.1:0",
            endpoint_config(),
            runtime=worker_runtime,
            worker_start_id="current-start",
            max_active_batches=1,
            max_pending_batches=1,
            owns_runtime=False,
        )
        await receiver.start()
        transport = TransferEngineWorkerTransport(
            f"127.0.0.1:{receiver.bound_port}",
            endpoint_config(),
            max_in_flight=1,
            device="cpu",
            runtime=client_runtime,
            expected_worker_start_id="stale-start",
        )
        try:
            with pytest.raises(Exception, match=r"start ID|INVALID_ARGUMENT"):
                await transport.start()
        finally:
            await transport.close()
            await receiver.close()
            await client_runtime.close()
            await worker_runtime.close()

    asyncio.run(scenario())


def test_closed_endpoint_keeps_generation_tombstone() -> None:
    async def scenario() -> None:
        backend = "ascend_direct"
        link = _FakeLink()
        worker_runtime = _FakeRuntime(link, "worker:19002", backend=backend)
        original_runtime = _FakeRuntime(
            link,
            "client:19001",
            backend=backend,
            generation="frontend-generation-a",
        )
        same_generation_runtime = _FakeRuntime(
            link,
            "client:19001",
            backend=backend,
            generation="frontend-generation-a",
        )
        restarted_runtime = _FakeRuntime(
            link,
            "client:19001",
            backend=backend,
            generation="frontend-generation-b",
        )
        receiver = TransferEngineWorkerBatchReceiver(
            "127.0.0.1:0",
            endpoint_config(),
            runtime=worker_runtime,
            worker_start_id="worker-start-a",
            max_active_batches=1,
            max_pending_batches=1,
            owns_runtime=False,
        )
        await receiver.start()

        def transport(runtime: _FakeRuntime) -> TransferEngineWorkerTransport:
            return TransferEngineWorkerTransport(
                f"127.0.0.1:{receiver.bound_port}",
                endpoint_config(),
                max_in_flight=1,
                device="cpu",
                runtime=runtime,
                expected_worker_start_id="worker-start-a",
            )

        original = transport(original_runtime)
        same_generation = transport(same_generation_runtime)
        restarted = transport(restarted_runtime)
        try:
            await original.start()
            await original.close()
            assert receiver._target_generations == {  # type: ignore[attr-defined]
                "client:19001": "frontend-generation-a"
            }

            with pytest.raises(TransportError, match="INVALID_ARGUMENT"):
                await restarted.start()

            # Route removal/re-add within the same live Frontend process is
            # safe: its endpoint and runtime generation still identify the
            # descriptor owner retained by the Worker tombstone.
            await same_generation.start()
            await same_generation.close()
            assert receiver._target_generations == {  # type: ignore[attr-defined]
                "client:19001": "frontend-generation-a"
            }
        finally:
            await restarted.close()
            await same_generation.close()
            await original.close()
            await receiver.close()
            await original_runtime.close()
            await same_generation_runtime.close()
            await restarted_runtime.close()
            await worker_runtime.close()

    asyncio.run(scenario())


def test_retirement_failure_keeps_target_gated_and_registered_arena(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        pair = await start_pair(max_in_flight=1)
        pair.receiver._session_close_grace_secs = 0.01  # type: ignore[attr-defined]

        async def fail_retire(
            tensor: torch.Tensor,
            *,
            monotonic_deadline: float = math.inf,
        ) -> None:
            del tensor, monotonic_deadline
            raise RuntimeError("arena retirement failed")

        monkeypatch.setattr(pair.client_runtime, "retire_tensor", fail_retire)
        try:
            with pytest.raises(TransportError, match="arena is retained") as captured:
                await pair.transport.close()
            assert captured.value.retryable is False
            assert pair.client_runtime.quarantine_reason is not None
            assert pair.transport._registered  # type: ignore[attr-defined]
            assert pair.link.regions["client:19001"]
            session = next(iter(pair.receiver._sessions.values()))  # type: ignore[attr-defined]
            assert session.close_prepared
            assert pair.receiver._target_close_gates == {  # type: ignore[attr-defined]
                "client:19001": {session.client_epoch}
            }
        finally:
            await pair.receiver.close()
            pair.buffers.close()
            await pair.client_runtime.close()
            await pair.worker_runtime.close()

    asyncio.run(scenario())


def test_frontend_close_barrier_survives_caller_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        pair = await start_pair(max_in_flight=2, client_max_in_flight=1)
        invalidation_started = asyncio.Event()
        allow_invalidation = asyncio.Event()
        original_invalidate = pair.worker_runtime.invalidate_remote_session

        async def delayed_invalidation(
            target_session: str,
            *,
            monotonic_deadline: float = math.inf,
        ) -> None:
            invalidation_started.set()
            await allow_invalidation.wait()
            await original_invalidate(
                target_session,
                monotonic_deadline=monotonic_deadline,
            )

        monkeypatch.setattr(
            pair.worker_runtime,
            "invalidate_remote_session",
            delayed_invalidation,
        )
        caller = asyncio.create_task(pair.transport.close())
        try:
            await invalidation_started.wait()
            caller.cancel()
            with pytest.raises(asyncio.CancelledError):
                await caller
            assert pair.transport._close_task is not None  # type: ignore[attr-defined]
            assert not pair.transport._close_task.done()  # type: ignore[attr-defined]

            allow_invalidation.set()
            await pair.transport.close()
            assert pair.client_runtime.quarantine_reason is None
            assert pair.link.regions["client:19001"]
            assert pair.client_runtime.retired
        finally:
            allow_invalidation.set()
            await pair.receiver.close()
            pair.buffers.close()
            await pair.client_runtime.close()
            await pair.worker_runtime.close()

    asyncio.run(scenario())


def test_close_session_timeout_retains_arena_after_unresponsive_prepare(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from expertkit_transport.transports.transfer_engine import client

    monkeypatch.setattr(client, "_TERMINAL_DRAIN_GRACE_SECONDS", 0.05)

    async def scenario() -> None:
        pair = await start_pair(max_in_flight=1, session_close_grace_secs=0.01)
        prepare_started = asyncio.Event()
        allow_disconnect = asyncio.Event()
        original_disconnect = pair.worker_runtime.disconnect_remote_session

        async def delayed_disconnect(target: str) -> None:
            prepare_started.set()
            await allow_disconnect.wait()
            await original_disconnect(target)

        monkeypatch.setattr(pair.worker_runtime, "disconnect_remote_session", delayed_disconnect)
        closing = asyncio.create_task(pair.transport.close())
        try:
            await asyncio.wait_for(prepare_started.wait(), 2)
            with pytest.raises(TransportError, match="arena is retained") as captured:
                await asyncio.wait_for(closing, 0.5)
            assert not captured.value.retryable
            assert isinstance(captured.value.__cause__, grpc.aio.AioRpcError)
            assert captured.value.__cause__.code() is grpc.StatusCode.DEADLINE_EXCEEDED
            assert pair.client_runtime.quarantine_reason is not None
            assert pair.client_runtime.retired == []
            assert pair.transport._registered
            assert pair.link.regions["client:19001"]
            assert not pair.transport._arena._closed
            assert pair.link.operations == []
            with pytest.raises(RuntimeError, match="closing"):
                await pair.transport.start()
        finally:
            allow_disconnect.set()
            await pair.receiver.close()
            await asyncio.gather(pair.transport.close(), return_exceptions=True)
            pair.buffers.close()
            await pair.client_runtime.close()
            await pair.worker_runtime.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("disconnect_fails", [False, True])
def test_prepare_drains_only_its_target_before_disconnect_and_retirement(
    monkeypatch: pytest.MonkeyPatch, disconnect_fails: bool
) -> None:
    async def scenario() -> None:
        pair = await start_pair(max_active_batches=2, max_pending_batches=2)
        other_runtime = _FakeRuntime(pair.link, "other:19003")

        def transport(runtime: _FakeRuntime) -> TransferEngineWorkerTransport:
            return TransferEngineWorkerTransport(
                f"127.0.0.1:{pair.receiver.bound_port}",
                endpoint_config(),
                max_in_flight=1,
                device="cpu",
                runtime=runtime,
                expected_worker_start_id="worker-start-a",
            )

        sibling = transport(pair.client_runtime)
        other = transport(other_runtime)
        await sibling.start()
        await other.start()
        original_disconnect = pair.worker_runtime.disconnect_remote_session

        async def disconnect(target: str) -> None:
            assert target == "client:19001"
            if disconnect_fails:
                raise RuntimeError("ADXL disconnect failed")
            await original_disconnect(target)

        monkeypatch.setattr(pair.worker_runtime, "disconnect_remote_session", disconnect)
        active = asyncio.create_task(
            sibling.execute(
                worker_batch(), torch.empty((2, 3)), monotonic_deadline=time.monotonic() + 5
            )
        )
        received = await asyncio.wait_for(pair.receiver.receive(), 2)
        closing = asyncio.create_task(pair.transport.close())

        async def await_gate() -> None:
            while "client:19001" not in pair.receiver._target_close_gates:
                await asyncio.sleep(0)

        await asyncio.wait_for(await_gate(), 2)
        try:
            assert not closing.done()
            assert pair.worker_runtime.disconnected_sessions == []
            assert pair.client_runtime.retired == []
            independent = asyncio.create_task(
                other.execute(
                    worker_batch(), torch.empty((2, 3)), monotonic_deadline=time.monotonic() + 5
                )
            )
            unrelated = await asyncio.wait_for(pair.receiver.receive(), 2)
            await complete(unrelated)
            await asyncio.wait_for(independent, 2)
            assert not closing.done()
            await complete(received)
            await active
            if disconnect_fails:
                with pytest.raises(TransportError) as captured:
                    await closing
                assert not captured.value.retryable
                assert pair.receiver._target_close_gates
                assert pair.worker_runtime.quarantine_reason is not None
                assert pair.client_runtime.retired == []
            else:
                await closing
                assert pair.worker_runtime.disconnected_sessions == ["client:19001"]
                assert len(pair.client_runtime.retired) == 1
                assert pair.link.regions["client:19001"]
        finally:
            monkeypatch.setattr(
                pair.worker_runtime, "disconnect_remote_session", original_disconnect
            )
            if disconnect_fails:
                pair.receiver._session_close_grace_secs = 0.01
                await sibling._channel.close()
            else:
                await sibling.close()
            await other.close()
            await pair.receiver.close()
            pair.buffers.close()
            await pair.client_runtime.close()
            await other_runtime.close()
            await pair.worker_runtime.close()

    asyncio.run(scenario())


def test_unary_control_propagates_frontend_traceparent_to_worker() -> None:
    pytest.importorskip("opentelemetry.instrumentation.grpc")
    from opentelemetry import trace
    from opentelemetry.instrumentation.grpc import aio_server_interceptor
    from opentelemetry.sdk.trace import TracerProvider

    from expertkit_transport.observability import OpenTelemetryTracer
    from expertkit_transport.tracing import trace_span, use_tracer

    provider = TracerProvider()
    tracer = OpenTelemetryTracer(provider.get_tracer("test-ascend"))

    async def scenario() -> None:
        pair = await start_pair(
            tracer=tracer, interceptors=(aio_server_interceptor(tracer_provider=provider),)
        )
        try:
            with use_tracer(tracer), trace_span("frontend"):
                parent = trace.get_current_span().get_span_context()
                submission = asyncio.create_task(
                    pair.transport.execute(
                        worker_batch(), torch.empty((2, 3)), monotonic_deadline=time.monotonic() + 5
                    )
                )
                received = await asyncio.wait_for(pair.receiver.receive(), 2)
                assert received.trace_context is not None
                remote = trace.get_current_span(received.trace_context).get_span_context()
                assert remote.trace_id == parent.trace_id
                assert remote.span_id != parent.span_id
                await complete(received)
                await submission
        finally:
            await pair.close()

    try:
        asyncio.run(scenario())
    finally:
        provider.shutdown()
