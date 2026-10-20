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
from expertkit_transport.transports.ascend_graph.config import AscendGraphConfig
from expertkit_transport.transports.ascend_graph.contracts import (
    GraphAdmissionTimeout,
    GraphCompletionUnproven,
    GraphControlRejected,
)
from expertkit_transport.transports.ascend_graph.session import HcclGraphSession
from pydantic import ValidationError

from expertkit_worker.backends.torch.decode_graph import ResidentExpertGraphWeights
from expertkit_worker.execution.admission import DeviceAdmission

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
        config: AscendGraphConfig,
        *,
        worker_start_id: str,
        device: torch.device,
        hidden_dim: int,
        top_k: int,
        admission: DeviceAdmission,
        prepare_weights: Callable[[], ResidentExpertGraphWeights],
        terminate: Callable[[int], object] = os._exit,
        worker_id: str | None = None,
    ) -> None:
        self.config = config
        self._worker_start_id = worker_start_id
        self._rank = config.worker_rank(worker_id or config.participants[0].worker_id)
        self._device = device
        self._hidden_dim = hidden_dim
        self._top_k = top_k
        self._admission = admission
        self._prepare_weights = prepare_weights
        self._terminate = terminate
        self._weights: ResidentExpertGraphWeights | None = None
        self._session: HcclGraphSession | None = None
        self._stream: Any = None
        self._graphs: dict[int, Any] = {}
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
        if self._closed:
            return
        self._closed = True
        self._stopped.set()
        if self._monitor is not None:
            await self._monitor
        work = self._executor.submit(self._destroy)
        try:
            await asyncio.wait_for(
                asyncio.shield(asyncio.wrap_future(work)), timeout=self.config.timeout_seconds
            )
        except BaseException:
            self._fatal("graph_shutdown_completion_unproven")
            raise
        self._executor.shutdown(wait=True, cancel_futures=True)

    def _validate_request(self, request: dict[str, Any]) -> tuple[str, int]:
        if set(request) != {
            "version",
            "identity",
            "sequence",
            "phase",
            "capacity",
            "worker_start_id",
            "ready_revision",
        }:
            raise ValueError("invalid graph control request fields")
        try:
            identity = AscendGraphConfig.model_validate(request["identity"]).identity()
        except ValidationError as error:
            raise ValueError("invalid graph deployment identity") from error
        if (
            type(request["version"]) is not int
            or request["version"] != 1
            or identity != self.config.identity()
        ):
            raise ValueError("graph deployment identity does not match this Worker")
        phase = request["phase"]
        capacity = request["capacity"]
        if not isinstance(phase, str) or phase not in {
            "prepare",
            "warmup",
            "capture",
            "replay",
            "close",
        }:
            raise ValueError("unsupported graph control phase")
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise ValueError("graph capacity must be an integer")
        if phase in {"warmup", "capture", "replay"}:
            if capacity not in self.config.capture_sizes:
                raise ValueError("graph capacity is not a configured profile")
            if self._weights is None or self._session is None:
                raise ValueError("graph session has not been prepared")
            if request["worker_start_id"] != self._worker_start_id:
                raise ValueError("graph Worker start identity changed")
            if request["ready_revision"] != self._weights.revision:
                raise ValueError("graph ready-storage revision changed")
            self._weights.require_current()
        elif capacity != 0:
            raise ValueError("graph prepare/close capacity must be zero")
        if phase == "prepare" and self._weights is not None:
            raise ValueError("graph session is already prepared")
        if phase == "prepare" and (
            request["worker_start_id"] is not None or request["ready_revision"] is not None
        ):
            raise ValueError("new graph preparation must not reuse a previous Worker identity")
        if phase != "prepare" and (
            request["worker_start_id"] != self._worker_start_id
            or type(request["ready_revision"]) is not int
        ):
            raise ValueError("graph Worker identity is invalid")
        if phase == "close" and (
            self._weights is None or request["ready_revision"] != self._weights.revision
        ):
            raise ValueError("graph close does not match prepared ready storage")
        if phase == "capture" and capacity in self._graphs:
            raise ValueError("graph profile has already been captured")
        if phase == "replay" and capacity not in self._graphs:
            raise ValueError("graph profile has not been captured")
        sequence = request["sequence"]
        if (
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence != self._sequence + 1
        ):
            raise ValueError("graph control sequence is not the next step")
        return phase, capacity

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
                phase, capacity = self._validate_request(request)
            except (ValueError, RuntimeError) as error:
                raise GraphControlRejected(str(error)) from error
            self._sequence = request["sequence"]
            self._deadline = time.monotonic() + self.config.timeout_seconds
            self._pending_request = request
            self._pending = self._executor.submit(self._run, phase, capacity, ready)
            pending = self._pending
        timeout = min(
            timeout if timeout is not None else self.config.timeout_seconds,
            self.config.timeout_seconds,
        )
        if not ready.wait(timeout):
            with self._lock:
                self._poisoned = True
            raise GraphAdmissionTimeout("graph admission timed out")
        if pending.done() and pending.exception() is not None:
            with self._lock:
                self._poisoned = True
            raise GraphCompletionUnproven("graph phase failed before admission")
        return {
            "state": "ready",
            "sequence": self._sequence,
            "worker_start_id": self._worker_start_id,
            "ready_revision": (
                self._weights.revision
                if self._weights is not None
                else request["ready_revision"]
                if phase == "close"
                else 0
            ),
        }

    def finish(self, request: dict[str, Any], *, timeout: float | None) -> dict[str, object]:
        """Prove matching completion without extending the admitted step's deadline."""
        with self._lock:
            if request != self._pending_request or self._pending is None:
                raise GraphControlRejected("no matching graph step")
            pending = self._pending
        try:
            timeout = min(
                timeout if timeout is not None else self.config.timeout_seconds,
                self.config.timeout_seconds,
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
        return {"state": "complete", "sequence": request["sequence"]}

    def _run(self, phase: str, capacity: int, ready: threading.Event) -> None:
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
                        self._weights = self._prepare_weights()
                        self._session = HcclGraphSession(
                            self.config,
                            rank=self._rank,
                            device=self._device,
                            hidden_dim=self._hidden_dim,
                            top_k=self._top_k,
                        )
                        for profile in self.config.capture_sizes:
                            self._session.buffers(profile)
                        self._stream.synchronize()
                        ready.set()
                        return
                    assert self._weights is not None and self._session is not None
                    self._weights.require_current()
                    ready.set()
                    if phase == "capture":
                        graph = torch.npu.NPUGraph()
                        with torch.npu.graph(graph, stream=self._stream):
                            self._body(capacity)
                        self._graphs[capacity] = graph
                        logger.info("worker_decode_graph_captured", token_capacity=capacity)
                    elif phase == "replay":
                        self._graphs[capacity].replay()
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
            pending = self._pending
            if pending is not None and time.monotonic() > self._deadline:
                self._fatal("graph_step_deadline_expired")
                return
            if pending is not None and pending.done() and pending.exception() is not None:
                self._fatal("graph_device_completion_failed")
                return
            if self._weights is not None and not self._poisoned:
                try:
                    self._weights.require_current()
                except RuntimeError:
                    self._poisoned = True
                    work = self._executor.submit(self._destroy)
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(asyncio.wrap_future(work)),
                            timeout=self.config.timeout_seconds,
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
