"""Serialize whole-step Ascend graph execution with ordinary Worker admission."""

from __future__ import annotations

import asyncio
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from typing import TYPE_CHECKING, Any

import structlog
import torch
from expertkit_transport.quarantine import retain_until_process_exit
from expertkit_transport.transports.ascend_graph.config import (
    AscendGraphConfig,
    AscendGraphModel,
    AscendGraphWorker,
)
from expertkit_transport.transports.ascend_graph.contracts import (
    GraphAdmissionTimeout,
    GraphCompletionUnproven,
    GraphControlRejected,
)
from expertkit_transport.transports.ascend_graph.rendezvous import (
    GraphRendezvousRequest,
    OwnedGraphRendezvous,
)
from expertkit_transport.transports.ascend_graph.session import HcclGraphSession
from pydantic import ValidationError

from expertkit_worker.backends.base import ResidentGraphWeights
from expertkit_worker.execution.admission import DeviceAdmission
from expertkit_worker.weights.manager import GraphAssignmentGuard

if TYPE_CHECKING:
    from expertkit_transport.transports.ascend_graph.control import GraphControlServer

logger = structlog.get_logger(__name__)


class AscendDecodeGraphExecution:
    """Own a single graph thread, admission state and completion watchdog.

    A shared admission lock excludes ordinary slots until the graph stream has
    completed. Device faults terminate the process because a blocked HCCL stream
    cannot safely return storage to the Weight Manager.
    """

    def __init__(
        self,
        *,
        instance_id: int,
        model_name: str,
        weight_version: str,
        model: AscendGraphModel,
        moe_layer_range: tuple[int, int],
        worker_id: str,
        worker_start_id: str,
        device: torch.device,
        device_name: str,
        max_batch_tokens: int,
        shutdown_grace_secs: float,
        admission: DeviceAdmission,
        prepare_weights: Callable[[AscendGraphWorker, tuple[int, ...]], ResidentGraphWeights],
        guard_assignments: Callable[[AscendGraphWorker, tuple[int, ...]], GraphAssignmentGuard],
        terminate: Callable[[int], object] = os._exit,
    ) -> None:
        self.config: AscendGraphConfig | None = None
        self._instance_id = instance_id
        self._model_name = model_name
        self._weight_version = weight_version
        self._model = model
        self._moe_layer_range = moe_layer_range
        self._worker_id = worker_id
        self._worker_start_id = worker_start_id
        self._device_name = device_name
        self._max_batch_tokens = max_batch_tokens
        self._shutdown_grace_secs = shutdown_grace_secs
        self._rank: int | None = None
        self._assignment: AscendGraphWorker | None = None
        self._assignment_id: str | None = None
        self._guard_assignments = guard_assignments
        self._assignment_guard: GraphAssignmentGuard | None = None
        self._device = device
        self._hidden_dim = model.hidden_dim
        self._top_k = model.top_k
        self._admission = admission
        self._prepare_weights = prepare_weights
        self._terminate = terminate
        self._weights: ResidentGraphWeights | None = None
        self._session: HcclGraphSession | None = None
        self._rendezvous: OwnedGraphRendezvous | None = None
        self._rendezvous_binding: GraphRendezvousRequest | None = None
        self._rendezvous_deadline = 0.0
        self._retired_reservation_id: str | None = None
        self._stream: Any = None
        self._graphs: dict[int | tuple[str, int], Any] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ek-decode-graph")
        self._lock = threading.Lock()
        self._sequence = 0
        self._pending: Future[None] | None = None
        self._pending_request: dict[str, Any] | None = None
        self._deadline = 0.0
        self._poisoned = False
        self._closed = False
        self._monitor: asyncio.Task[None] | None = None
        self._stopped = asyncio.Event()

    async def start(self) -> None:
        """Supervise Host admission; NPU setup waits for an explicit PREPARE."""
        if self._monitor is not None:
            return
        self._monitor = asyncio.create_task(self._watch(), name="worker-graph-watchdog")

    async def wait(self) -> None:
        """Supervise graph admission and storage invalidation until shutdown."""
        if self._monitor is None:
            raise RuntimeError("Ascend graph execution has not started")
        # Supervising callers do not own cancellation of the lifecycle monitor.
        await asyncio.shield(self._monitor)

    async def close(self) -> None:
        """Stop new commands, drain graphs and release leases before weight teardown."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._release_unused_rendezvous()
        self._stopped.set()
        if self._monitor is not None:
            await self._monitor
        work = self._executor.submit(self._destroy)
        try:
            await asyncio.wait_for(
                asyncio.shield(asyncio.wrap_future(work)), timeout=self._timeout_seconds
            )
        except BaseException:
            self._fatal("graph_shutdown_completion_unproven")
            raise
        self._executor.shutdown(wait=True, cancel_futures=True)

    @property
    def _timeout_seconds(self) -> float:
        return self.config.timeout_seconds if self.config is not None else self._shutdown_grace_secs

    def _release_unused_rendezvous(self) -> None:
        reservation = self._rendezvous
        if reservation is not None and reservation.adopted:
            return
        request = reservation.request if reservation is not None else self._rendezvous_binding
        if request is None:
            return
        if reservation is not None:
            reservation.close()
        self._retired_reservation_id = request.reservation_id
        self._rendezvous = None
        self._rendezvous_binding = None
        if self._assignment_guard is not None:
            self._assignment_guard.close()
            self._assignment_guard = None

    def _expire_rendezvous(self) -> None:
        with self._lock:
            reservation = self._rendezvous
            if reservation is not None and reservation.adopted:
                return
            if reservation is None and self._rendezvous_binding is None:
                return
            deadline = (
                reservation.deadline if reservation is not None else self._rendezvous_deadline
            )
            try:
                assert self._assignment_guard is not None
                self._assignment_guard.require_current()
                if time.monotonic() < deadline:
                    return
            except RuntimeError:
                pass
            self._release_unused_rendezvous()

    def rendezvous(self, request: dict[str, Any], *, timeout: float | None) -> dict[str, object]:
        """Bind or release one Host-only listener without admitting device work.

        A lost reply leaves the listener owned until the original deadline. Only
        matching PREPARE can transfer it to the device session's lifetime.
        """
        try:
            return self._rendezvous_command(request, timeout=timeout)
        except (ValueError, RuntimeError) as error:
            raise GraphControlRejected(str(error)) from error
        except TimeoutError as error:
            raise GraphAdmissionTimeout(str(error)) from error

    def _rendezvous_command(
        self, request: dict[str, Any], *, timeout: float | None
    ) -> dict[str, object]:
        with self._lock:
            if not isinstance(request, dict):
                raise ValueError("graph rendezvous request must be an object")
            if request.get("action") == "release":
                reservation = self._rendezvous
                owner = reservation.request if reservation is not None else self._rendezvous_binding
                if (
                    type(request.get("version")) is not int
                    or owner is None
                    or request != owner.release_request()
                ):
                    raise ValueError("no matching graph rendezvous owner to release")
                reply = {key: value for key, value in request.items() if key != "action"}
                reply["state"] = (
                    "attached" if reservation is not None and reservation.adopted else "released"
                )
                self._release_unused_rendezvous()
                return reply
            parsed = GraphRendezvousRequest.model_validate(request)
            if (
                self._closed
                or self._poisoned
                or self._pending is not None
                or self.config is not None
            ):
                raise ValueError("graph session cannot reserve a listener")
            if (
                parsed.worker_start_id != self._worker_start_id
                or parsed.worker.worker_id != self._worker_id
                or parsed.worker.device != self._device_name
                or parsed.instance_id != self._instance_id
                or parsed.weight_version != self._weight_version
                or any(
                    not self._moe_layer_range[0] <= layer < self._moe_layer_range[1]
                    for layer in parsed.layer_ids
                )
            ):
                raise ValueError("graph rendezvous owner does not match the Worker")
            if self._rendezvous is not None:
                if parsed != self._rendezvous.request:
                    raise ValueError("another graph session owns the rendezvous listener")
                assert self._assignment_guard is not None
                self._assignment_guard.require_current()
                if time.monotonic() >= self._rendezvous.deadline:
                    self._release_unused_rendezvous()
                    raise ValueError("graph rendezvous reservation expired")
                return self._rendezvous.reply()
            if self._rendezvous_binding is not None:
                raise ValueError("a graph rendezvous listener is already binding")
            if parsed.reservation_id == self._retired_reservation_id:
                raise ValueError("graph rendezvous reservation already ended")
            budget = min(
                parsed.timeout_seconds, timeout if timeout is not None else parsed.timeout_seconds
            )
            deadline = time.monotonic() + budget
            guard = self._guard_assignments(parsed.worker, parsed.layer_ids)
            try:
                guard.require_current()
            except BaseException:
                guard.close()
                raise
            self._assignment_guard = guard
            self._rendezvous_binding = parsed
            self._rendezvous_deadline = deadline
        reservation = None
        try:
            # Native address resolution/connect may block; never hold admission's lock here.
            reservation = OwnedGraphRendezvous(
                parsed, monotonic_deadline=deadline, bind_timeout_seconds=self._shutdown_grace_secs
            )
            with self._lock:
                if (
                    self._closed
                    or self._poisoned
                    or self._rendezvous_binding is not parsed
                    or time.monotonic() >= deadline
                ):
                    raise ValueError("graph rendezvous reservation ended while binding")
                guard.require_current()
                self._rendezvous = reservation
                self._rendezvous_binding = None
                logger.info(
                    "worker_graph_rendezvous_listening",
                    worker_id=self._worker_id,
                    start_id=self._worker_start_id,
                    session_id=parsed.session_id,
                    rendezvous_host=parsed.rendezvous_host,
                    rendezvous_port=reservation.port,
                )
                return reservation.reply()
        except BaseException:
            if reservation is not None:
                reservation.close()
            with self._lock:
                if self._rendezvous_binding is parsed:
                    self._release_unused_rendezvous()
            raise

    def _validate_request(
        self, request: dict[str, Any]
    ) -> tuple[str, int, AscendGraphConfig | None]:
        if not isinstance(request, dict):
            raise ValueError("graph control request must be an object")
        phase = request.get("phase")
        if not isinstance(phase, str) or phase not in {
            "prepare",
            "warmup",
            "capture",
            "replay",
            "close",
        }:
            raise ValueError("unsupported graph control phase")
        paired = (
            self.config is not None
            and self.config.mode == "full"
            and self.config.model.expert_compute == "deepseek_v4"
            and phase in {"capture", "replay"}
        )
        fields = {
            "version",
            "assignment_id",
            "sequence",
            "phase",
            "capacity",
            "worker_start_id",
            "ready_revision",
        }
        if phase == "prepare":
            fields.add("config")
        if paired:
            fields.add("graph_phase")
        if set(request) != fields:
            raise ValueError("invalid graph control request fields")
        if type(request["version"]) is not int or request["version"] != 2:
            raise ValueError("graph control requires protocol version 2")
        if request["worker_start_id"] != self._worker_start_id:
            raise ValueError("graph Worker start identity changed")
        if type(request["sequence"]) is not int or request["sequence"] != self._sequence + 1:
            raise ValueError("graph control sequence is not the next step")
        capacity = request["capacity"]
        if type(capacity) is not int:
            raise ValueError("graph capacity must be an integer")
        config = None
        if phase == "prepare":
            if self.config is not None:
                raise ValueError("graph session is already prepared")
            if request["ready_revision"] is not None or capacity != 0:
                raise ValueError(
                    "new graph preparation requires a null ready token and zero capacity"
                )
            try:
                config = AscendGraphConfig.model_validate(request["config"])
            except ValidationError as error:
                raise ValueError("invalid graph session configuration") from error
            if config.assignment_id != request["assignment_id"]:
                raise ValueError("graph assignment digest does not match the snapshot")
            rank = config.worker_rank(self._worker_id)
            assignment = config.participants[rank - 1]
            if (
                config.instance_id != self._instance_id
                or config.model_name != self._model_name
                or config.weight_version != self._weight_version
                or config.model != self._model
                or any(
                    not self._moe_layer_range[0] <= layer_id < self._moe_layer_range[1]
                    for layer_id in config.layer_ids
                )
                or assignment.expected_start_id != self._worker_start_id
                or assignment.device != self._device_name
                or max(config.capture_sizes) > self._max_batch_tokens
            ):
                raise ValueError(
                    "graph session facts do not match the resolved Worker configuration"
                )
            if self._rendezvous_binding is not None or (rank != 1 and self._rendezvous is not None):
                raise ValueError("another graph session owns this Worker's rendezvous reservation")
            if rank == 1:
                if self._rendezvous is None:
                    raise ValueError("graph Rendezvous protocol 1 is required before rank1 PREPARE")
                self._rendezvous.validate_config(config)
        else:
            if self.config is None or request["assignment_id"] != self._assignment_id:
                raise ValueError("graph session is not prepared or its assignment digest changed")
            if type(request["ready_revision"]) is not int or request["ready_revision"] < 0:
                raise ValueError("graph ready-storage identity is invalid")
            if self._weights is None or request["ready_revision"] != self._weights.revision:
                raise ValueError("graph ready-storage revision changed")
            if phase != "close":
                if capacity not in self.config.capture_sizes:
                    raise ValueError("graph capacity is not a configured profile")
                if self._session is None:
                    raise ValueError("graph session has not been prepared")
                if self._assignment_guard is not None:
                    self._assignment_guard.require_current()
                self._weights.require_current()
            elif capacity != 0:
                raise ValueError("graph close capacity must be zero")
        if paired:
            graph_phase = request["graph_phase"]
            assert self.config is not None
            if (
                graph_phase not in ("decode", "mixed")
                or (graph_phase == "decode" and capacity not in self.config.decode_capture_sizes)
                or (graph_phase == "mixed" and capacity < 2)
            ):
                raise ValueError("capacity does not belong to the V4 full graph phase")
        key = (request["graph_phase"], capacity) if paired else capacity
        if phase == "capture" and key in self._graphs:
            raise ValueError("graph profile has already been captured")
        if phase == "replay" and key not in self._graphs:
            raise ValueError("graph profile has not been captured")
        return phase, capacity, config

    def _reply(self, request: dict[str, Any], state: str, revision: int) -> dict[str, object]:
        return {
            "version": 2,
            "assignment_id": self._assignment_id,
            "state": state,
            "sequence": request["sequence"],
            "worker_start_id": self._worker_start_id,
            "ready_revision": revision,
        }

    def begin(self, request: dict[str, Any], *, timeout: float | None) -> dict[str, object]:
        """Admit one step and acknowledge device ownership within a Host wait budget.

        A wait timeout poisons admission but leaves the device Future and weight
        leases owned until the watchdog or shutdown proves completion.
        """
        ready = threading.Event()
        with self._lock:
            if self._closed or self._poisoned or self._pending is not None:
                raise GraphControlRejected("graph session cannot admit work")
            try:
                phase, capacity, config = self._validate_request(request)
                if config is not None:
                    rank = config.worker_rank(self._worker_id)
                    assignment = config.participants[rank - 1]
                    guard = (
                        self._assignment_guard
                        if rank == 1
                        else self._guard_assignments(assignment, config.layer_ids)
                    )
                    assert guard is not None
                    try:
                        guard.require_current()
                    except BaseException:
                        guard.close()
                        raise
                    self.config = config
                    self._assignment_id = request["assignment_id"]
                    self._rank = rank
                    self._assignment = assignment
                    self._assignment_guard = guard
                    if rank == 1:
                        assert self._rendezvous is not None
                        self._rendezvous.adopted = True
            except (ValueError, RuntimeError) as error:
                raise GraphControlRejected(str(error)) from error
            self._sequence = request["sequence"]
            self._deadline = time.monotonic() + self._timeout_seconds
            self._pending_request = request
            self._pending = self._executor.submit(
                self._run, phase, capacity, ready, request.get("graph_phase")
            )
            pending = self._pending
        timeout = min(
            timeout if timeout is not None else self._timeout_seconds,
            max(0.0, self._deadline - time.monotonic()),
        )
        if not ready.wait(timeout):
            with self._lock:
                self._poisoned = True
            raise GraphAdmissionTimeout("graph admission timed out")
        if pending.done() and pending.exception() is not None:
            with self._lock:
                self._poisoned = True
            raise GraphCompletionUnproven("graph phase failed before admission")
        with self._lock:
            revision = (
                self._weights.revision if self._weights is not None else request["ready_revision"]
            )
            if type(revision) is not int:
                self._poisoned = True
                raise GraphCompletionUnproven("graph preparation did not retain a ready identity")
            normalized = {key: value for key, value in request.items() if key != "config"}
            normalized["ready_revision"] = revision
            self._pending_request = normalized
            return self._reply(normalized, "ready", revision)

    def finish(self, request: dict[str, Any], *, timeout: float | None) -> dict[str, object]:
        """Prove matching completion without extending the admitted step's deadline."""
        with self._lock:
            if (
                not isinstance(request, dict)
                or any(
                    type(request.get(name)) is not int
                    for name in ("version", "sequence", "capacity", "ready_revision")
                )
                or request != self._pending_request
                or self._pending is None
            ):
                raise GraphControlRejected("no matching graph step")
            pending = self._pending
        try:
            timeout = min(
                timeout if timeout is not None else self._timeout_seconds,
                max(0.0, self._deadline - time.monotonic()),
            )
            pending.result(timeout=timeout)
        except BaseException as error:
            with self._lock:
                self._poisoned = True
            raise GraphCompletionUnproven("graph completion could not be proven") from error
        with self._lock:
            if self._pending is not pending or self._pending_request != request:
                raise GraphControlRejected("graph step already completed")
            self._pending = None
            self._pending_request = None
        return self._reply(request, "complete", request["ready_revision"])

    def _run(
        self, phase: str, capacity: int, ready: threading.Event, graph_phase: str | None = None
    ) -> None:
        try:
            if phase == "close":
                ready.set()
                self._destroy()
                return
            with self._admission.use(poison_on_error=True):
                torch.npu.set_device(self._device)
                if self._stream is None:
                    self._stream = torch.npu.Stream(device=self._device)
                with torch.inference_mode(), torch.npu.stream(self._stream):
                    if phase == "prepare":
                        assert self._assignment is not None and self._assignment_guard is not None
                        assert self.config is not None and self._rank is not None
                        self._assignment_guard.require_current()
                        self._weights = self._prepare_weights(
                            self._assignment, self.config.layer_ids
                        )
                        self._assignment_guard.require_current()
                        self._weights.require_current()
                        self._session = HcclGraphSession(
                            self.config,
                            rank=self._rank,
                            device=self._device,
                            hidden_dim=self._hidden_dim,
                            top_k=self._top_k,
                            rendezvous_store=(
                                self._rendezvous.store
                                if self._rank == 1 and self._rendezvous is not None
                                else None
                            ),
                        )
                        for profile in self.config.capture_sizes:
                            self._session.buffers(profile)
                        self._stream.synchronize()
                        ready.set()
                        return
                    assert self._weights is not None and self._session is not None
                    if self._assignment_guard is not None:
                        self._assignment_guard.require_current()
                    self._weights.require_current()
                    ready.set()
                    key = (graph_phase, capacity) if graph_phase is not None else capacity
                    if phase == "capture":
                        graph = torch.npu.NPUGraph()
                        if graph_phase is not None:
                            # Paired graphs retain partial capture resources on poison.
                            self._graphs[key] = graph
                        with torch.npu.graph(graph, stream=self._stream):
                            self._body(capacity)
                        self._graphs[key] = graph
                        logger.info("worker_decode_graph_captured", token_capacity=capacity)
                    elif phase == "replay":
                        self._graphs[key].replay()
                    else:
                        self._body(capacity)
                    self._stream.synchronize()
        except BaseException:
            self._poisoned = True
            logger.error("worker_decode_graph_failed", phase=phase, exc_info=True)
            raise
        finally:
            ready.set()

    def _body(self, capacity: int) -> None:
        assert self._session is not None and self._weights is not None
        buffers = self._session.buffers(capacity)
        assert self.config is not None
        for layer_id in self.config.layer_ids:
            self._session.receive_inputs(buffers)
            self._weights.compute(
                layer_id,
                buffers.hidden,
                buffers.expert_ids,
                buffers.routing_weights,
                buffers.partial_output if buffers.partial_output is not None else buffers.output,
            )
            self._session.combine(buffers)

    def _destroy(self) -> None:
        if (
            self._weights is None
            and self._session is None
            and self._stream is None
            and not self._graphs
        ):
            if self._rendezvous is not None:
                self._rendezvous.close()
                self._rendezvous = None
            if self._assignment_guard is not None:
                self._assignment_guard.close()
                self._assignment_guard = None
            return

        def completed() -> None:
            if self._stream is not None:
                self._stream.synchronize()
            self._graphs.clear()

        with self._admission.use(poison_on_error=True):
            if self._weights is not None:
                self._weights.close(wait_for_completion=completed)
                self._weights = None
            else:
                completed()
            if self._session is not None:
                self._session.close()
                self._session = None
            if self._rendezvous is not None:
                self._rendezvous.close()
                self._rendezvous = None
            if self._assignment_guard is not None:
                self._assignment_guard.close()
                self._assignment_guard = None

    def _fatal(self, reason: str) -> None:
        self._poisoned = True
        retain_until_process_exit(self)
        if self._weights is not None:
            self._weights.quarantine()
        if self._session is not None:
            self._session.quarantine()
        logger.critical("worker_decode_graph_fatal", reason=reason)
        self._terminate(1)

    async def _watch(self) -> None:
        while not self._stopped.is_set():
            self._expire_rendezvous()
            pending = self._pending
            if pending is not None and time.monotonic() > self._deadline:
                self._fatal("graph_step_deadline_expired")
                return
            if pending is not None and pending.done() and pending.exception() is not None:
                self._fatal("graph_device_completion_failed")
                return
            if self._weights is not None and not self._poisoned:
                try:
                    if self._assignment_guard is not None:
                        self._assignment_guard.require_current()
                    self._weights.require_current()
                except RuntimeError:
                    self._poisoned = True
                    work = self._executor.submit(self._destroy)
                    timeout = self._timeout_seconds
                    if pending is not None:
                        timeout = min(timeout, max(0.0, self._deadline - time.monotonic()))
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(asyncio.wrap_future(work)), timeout=timeout
                        )
                    except BaseException:
                        self._fatal("graph_invalidation_completion_unproven")
                        return
            with suppress(TimeoutError):
                await asyncio.wait_for(self._stopped.wait(), timeout=0.1)


class GraphWorkerExecution:
    """Compose ordinary execution and optional graph execution at one lifecycle."""

    def __init__(
        self, ordinary: Any, graph: AscendDecodeGraphExecution, control: GraphControlServer
    ) -> None:
        self._ordinary = ordinary
        self._graph = graph
        self._control = control

    async def start(self) -> None:
        """Start both endpoints; neither creates resident graphs at startup."""
        await self._ordinary.start()
        await self._graph.start()
        self._control.start()

    async def wait(self) -> None:
        """Surface either execution failure to the Worker application supervisor."""
        tasks = [asyncio.create_task(part.wait()) for part in (self._ordinary, self._graph)]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                await task
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def close(self) -> None:
        """Close both admissions while draining before Weight Manager teardown."""
        # Protocol phase=close finishes before process shutdown. Shutdown rejects
        # new RPCs without cancelling admitted device work or releasing its leases.
        self._control.stop()
        try:
            await asyncio.gather(self._ordinary.close(), self._graph.close())
        finally:
            await asyncio.to_thread(self._control.close)
