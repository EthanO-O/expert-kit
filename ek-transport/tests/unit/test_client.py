"""Tests for high-level routed-layer client startup."""

import asyncio
import time
from contextlib import suppress
from typing import ClassVar

import pytest
import torch

import expertkit_transport.client as client_module
from expertkit_transport.client import BlockingRoutedMoEClient, RoutedMoEClient
from expertkit_transport.controller import ResolvedDefaultInstance
from expertkit_transport.errors import TransportError, TransportErrorCode


class FakeTopology:
    instances: ClassVar[list["FakeTopology"]] = []

    def __init__(self, endpoint: str, **configuration: object) -> None:
        self.endpoint = endpoint
        self.configuration = configuration
        self.pools: dict[object, object] = {}
        self.started = False
        self.closed = False
        self.__class__.instances.append(self)

    async def start(self, *, monotonic_deadline: float) -> None:
        assert monotonic_deadline > time.monotonic()
        self.started = True

    async def close(self) -> None:
        self.closed = True


class _FakeCudaTensor:
    def __init__(self, *, fail_record_stream: bool = False) -> None:
        self.device = torch.device("cuda:0")
        self.fail_record_stream = fail_record_stream
        self.recorded_streams: list[object] = []

    def record_stream(self, stream: object) -> None:
        self.recorded_streams.append(stream)
        if self.fail_record_stream:
            raise RuntimeError("injected tensor record_stream failure")


class _FakeCudaEvent:
    def __init__(self, *, fail_record: bool = False) -> None:
        self.fail_record = fail_record
        self.recorded_streams: list[object] = []

    def record(self, stream: object) -> None:
        self.recorded_streams.append(stream)
        if self.fail_record:
            raise RuntimeError("injected event record failure")


class _FakeCudaStream:
    def __init__(self, *, fail_wait: bool = False) -> None:
        self.fail_wait = fail_wait
        self.waited_events: list[object] = []

    def wait_event(self, event: object) -> None:
        self.waited_events.append(event)
        if self.fail_wait:
            raise RuntimeError("injected stream wait failure")


def _blocking_client(monkeypatch, async_client_type: type) -> BlockingRoutedMoEClient:
    monkeypatch.setattr(client_module, "RoutedMoEClient", async_client_type)
    client = BlockingRoutedMoEClient(
        "controller:5002",
        num_layers=1,
        experts_per_layer=2,
        hidden_dim=3,
        top_k=1,
        dtype=torch.float32,
        device="cpu",
    )
    client.start(timeout_seconds=1)
    return client


def _execute_fake_cuda(client: BlockingRoutedMoEClient) -> object:
    return client.execute(
        layer_id=0,
        hidden_states=_FakeCudaTensor(),  # type: ignore[arg-type]
        expert_ids=_FakeCudaTensor(),  # type: ignore[arg-type]
        routing_weights=_FakeCudaTensor(),  # type: ignore[arg-type]
        distinct_expert_ids=(0,),
        timeout_seconds=1,
    )


def test_client_resolves_default_before_constructing_instance_bound_topology(
    monkeypatch,
) -> None:
    async def scenario() -> None:
        requests: list[int | None] = []
        submitted: list[object] = []

        async def resolve(endpoint, *, requested_instance_id, timeout_seconds):
            assert endpoint == "controller:5002"
            assert timeout_seconds > 0
            requests.append(requested_instance_id)
            return ResolvedDefaultInstance(7, "model", "default")

        async def execute(batch, *args, **kwargs):
            submitted.append(batch)
            return batch.hidden_states

        FakeTopology.instances.clear()
        monkeypatch.setattr(client_module, "resolve_default_instance", resolve)
        monkeypatch.setattr(client_module, "ControllerTopologyWatcher", FakeTopology)
        monkeypatch.setattr(client_module, "execute_routed_layer", execute)
        client = RoutedMoEClient(
            "controller:5002",
            num_layers=2,
            experts_per_layer=4,
            hidden_dim=3,
            top_k=2,
            dtype=torch.float32,
            device="cpu",
        )

        await client.start(monotonic_deadline=time.monotonic() + 1)
        hidden = torch.ones((1, 3))
        result = await client.execute(
            layer_id=1,
            hidden_states=hidden,
            expert_ids=torch.tensor([[0, 1]], dtype=torch.int32),
            routing_weights=torch.tensor([[0.5, 0.5]], dtype=torch.float32),
            distinct_expert_ids=(0, 1),
            monotonic_deadline=time.monotonic() + 1,
        )

        assert result is hidden
        assert requests == [None]
        assert FakeTopology.instances[0].configuration["instance_id"] == 7
        assert submitted[0].instance_id == 7
        await client.close()
        assert FakeTopology.instances[0].closed is True

    asyncio.run(scenario())


def test_blocking_timeout_surfaces_a_late_unsafe_staging_fatal(monkeypatch) -> None:
    class LateUnsafeClient:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        async def start(self, *, monotonic_deadline: float) -> None:
            assert monotonic_deadline > time.monotonic()

        async def execute(self, **kwargs) -> torch.Tensor:
            del kwargs
            with suppress(asyncio.CancelledError):
                await asyncio.Future()
            # A terminal ownership fatal takes precedence over cancellation.
            raise TransportError(
                TransportErrorCode.UNAVAILABLE,
                retryable=False,
                unsafe_tensor_ownership=True,
                diagnostic="late CUDA input-copy completion could not be proven",
            )

        async def close(self) -> None:
            return None

    monkeypatch.setattr(client_module, "RoutedMoEClient", LateUnsafeClient)
    client = BlockingRoutedMoEClient(
        "controller:5002",
        num_layers=1,
        experts_per_layer=2,
        hidden_dim=3,
        top_k=1,
        dtype=torch.float32,
        device="cpu",
    )
    try:
        client.start(timeout_seconds=1)
        with pytest.raises(TransportError, match="could not be proven") as caught:
            client.execute(
                layer_id=0,
                hidden_states=torch.ones((1, 3)),
                expert_ids=torch.zeros((1, 1), dtype=torch.int32),
                routing_weights=torch.ones((1, 1), dtype=torch.float32),
                distinct_expert_ids=(0,),
                timeout_seconds=0.01,
            )
        assert not caught.value.retryable
        assert caught.value.unsafe_tensor_ownership
        assert caught.value.code is TransportErrorCode.UNAVAILABLE
        with pytest.raises(TransportError) as start_rejected:
            client.start(timeout_seconds=1)
        assert start_rejected.value is caught.value
        with pytest.raises(TransportError) as execute_rejected:
            client.execute(
                layer_id=0,
                hidden_states=torch.ones((1, 3)),
                expert_ids=torch.zeros((1, 1), dtype=torch.int32),
                routing_weights=torch.ones((1, 1), dtype=torch.float32),
                distinct_expert_ids=(0,),
                timeout_seconds=1,
            )
        assert execute_rejected.value is caught.value
    finally:
        client.close()


@pytest.mark.parametrize("phase", ["input", "output", "caller", "record_stream", "success"])
def test_blocking_fences_keep_caller_ownership(monkeypatch, phase: str) -> None:
    retained: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        client_module, "retain_until_process_exit", lambda *owners: retained.append(owners)
    )
    result = _FakeCudaTensor(fail_record_stream=phase == "record_stream")

    class ImmediateClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def start(self, **kwargs) -> None:
            pass

        async def execute(self, **kwargs):
            return result

        async def close(self) -> None:
            pass

    events = iter(
        [
            _FakeCudaEvent(fail_record=phase == "input"),
            _FakeCudaEvent(fail_record=phase == "output"),
        ]
    )
    caller = _FakeCudaStream(fail_wait=phase == "caller")
    streams = iter([_FakeCudaStream(), _FakeCudaStream(), _FakeCudaStream(), caller])
    monkeypatch.setattr(torch.cuda, "Event", lambda **kwargs: next(events))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: next(streams))
    client = _blocking_client(monkeypatch, ImmediateClient)
    try:
        if phase == "success":
            assert _execute_fake_cuda(client) is result
            assert result.recorded_streams == [caller]
            assert len(caller.waited_events) == 1
            assert not retained
        else:
            with pytest.raises(TransportError) as failed:
                _execute_fake_cuda(client)
            assert failed.value.unsafe_tensor_ownership
            assert not failed.value.retryable
            assert retained
            assert any(isinstance(owner, _FakeCudaTensor) for owner in retained[-1])
            with pytest.raises(TransportError) as poisoned:
                client.start(timeout_seconds=1)
            assert poisoned.value is failed.value
    finally:
        client.close()


def test_invalid_input_device_does_not_leave_blocking_client_active(monkeypatch) -> None:
    class Client:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def start(self, **kwargs) -> None:
            pass

        async def close(self) -> None:
            pass

    client = _blocking_client(monkeypatch, Client)
    try:
        with pytest.raises(ValueError, match="device must be"):
            client.execute(
                layer_id=0,
                hidden_states=torch.empty((1, 3), device="meta"),
                expert_ids=torch.zeros((1, 1), dtype=torch.int32),
                routing_weights=torch.ones((1, 1)),
                distinct_expert_ids=(0,),
                timeout_seconds=1,
            )
        assert not client._active
    finally:
        client.close()


@pytest.mark.parametrize("mode", ["resolve", "close", "retry"])
def test_startup_shutdown_keeps_runtime_owned_until_pending_peer_cleanup_finishes(
    monkeypatch, mode: str
) -> None:
    async def scenario() -> None:
        during_resolution = mode == "resolve"
        topologies = []
        started = asyncio.Event()
        allow_resolution = asyncio.Event()
        closing = asyncio.Event()
        drained = asyncio.Event()
        runtime_closed = False

        async def resolve(*args, **kwargs):
            if during_resolution:
                started.set()
                await allow_resolution.wait()
            return ResolvedDefaultInstance(7, "model", "default")

        class Runtime:
            async def close(self):
                nonlocal runtime_closed
                assert during_resolution or drained.is_set()
                runtime_closed = True

            def quarantine(self, diagnostic):
                pytest.fail(diagnostic)

        class Topology:
            def __init__(self, *args, **kwargs):
                self.close_task = None
                topologies.append(self)

            async def start(self, **kwargs):
                started.set()
                if len(topologies) == 1:
                    await asyncio.Event().wait()

            async def close(self):
                async def cleanup():
                    closing.set()
                    await drained.wait()

                if self.close_task is None:
                    self.close_task = asyncio.create_task(cleanup())
                await asyncio.shield(self.close_task)

        monkeypatch.setattr(client_module, "resolve_default_instance", resolve)
        monkeypatch.setattr(client_module, "ControllerTopologyWatcher", Topology)
        client = RoutedMoEClient(
            "controller:5002",
            num_layers=2,
            experts_per_layer=4,
            hidden_dim=3,
            top_k=2,
            dtype=torch.float32,
            device="cpu",
            transport_runtime=Runtime(),
        )
        startup = asyncio.create_task(client.start(monotonic_deadline=time.monotonic() + 5))
        await started.wait()
        if during_resolution:
            await client.close()
            allow_resolution.set()
            with pytest.raises(RuntimeError, match="closed"):
                await startup
            assert client._topology is None
        else:
            startup.cancel()
            await closing.wait()
            startup.cancel()
            with pytest.raises(asyncio.CancelledError):
                await startup
            if mode == "retry":
                retry = asyncio.create_task(client.start(monotonic_deadline=time.monotonic() + 5))
                await asyncio.sleep(0)
                assert len(topologies) == 1
                assert not runtime_closed
                drained.set()
                await asyncio.wait_for(retry, 2)
                assert len(topologies) == 2
                await client.close()
            else:
                shutdown = asyncio.create_task(client.close())
                await asyncio.sleep(0)
                assert not runtime_closed
                drained.set()
                await asyncio.wait_for(shutdown, 2)
        assert runtime_closed

    asyncio.run(scenario())
