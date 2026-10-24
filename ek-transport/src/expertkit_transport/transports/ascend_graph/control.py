"""Bounded gRPC control messages outside Ascend device graph capture."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import grpc

from expertkit_transport.transports.ascend_graph.config import AscendGraphConfig
from expertkit_transport.transports.ascend_graph.contracts import (
    GraphAdmissionTimeout,
    GraphCompletionUnproven,
    GraphControlRejected,
    GraphStepCommand,
)

SERVICE = "expertkit.internal.AscendGraphControl"
MAX_CONTROL_BYTES = 16 * 1024


def encode_message(message: dict[str, object]) -> bytes:
    """Encode bounded control metadata; tensors never use this channel."""
    encoded = json.dumps(message, separators=(",", ":"), allow_nan=False).encode()
    if len(encoded) > MAX_CONTROL_BYTES:
        raise ValueError("Ascend graph control message exceeds its limit")
    return encoded


def decode_message(payload: bytes) -> dict[str, Any]:
    """Decode one bounded metadata object carried by gRPC."""
    if len(payload) > MAX_CONTROL_BYTES:
        raise ValueError("Ascend graph control message exceeds its limit")

    def invalid_constant(value: str) -> None:
        raise ValueError(f"non-finite graph control value: {value}")

    message = json.loads(payload, parse_constant=invalid_constant)
    if not isinstance(message, dict):
        raise ValueError("Ascend graph control message must be an object")
    return message


def control_handler(
    begin: GraphStepCommand,
    finish: GraphStepCommand,
) -> grpc.GenericRpcHandler:
    """Bind the two coarse-step methods using the existing gRPC dependency."""

    def bind(method):
        def invoke(payload, context):
            try:
                message = decode_message(payload)
            except (ValueError, UnicodeError) as error:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
            try:
                return method(message, timeout=context.time_remaining())
            except GraphControlRejected as error:
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(error))
            except GraphAdmissionTimeout as error:
                context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, str(error))
            except GraphCompletionUnproven as error:
                context.abort(grpc.StatusCode.INTERNAL, str(error))

        return grpc.unary_unary_rpc_method_handler(invoke, response_serializer=encode_message)

    return grpc.method_handlers_generic_handler(
        SERVICE,
        {name: bind(method) for name, method in (("Begin", begin), ("Finish", finish))},
    )


class GraphControlServer:
    """Own the bounded gRPC endpoint and RPC threads, without owning device work."""

    def __init__(self, endpoint: str, *, begin: GraphStepCommand, finish: GraphStepCommand) -> None:
        self._endpoint = endpoint
        self._port: int | None = None
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ek-graph-control")
        self._server = grpc.server(
            self._executor,
            options=[
                ("grpc.max_receive_message_length", MAX_CONTROL_BYTES),
                ("grpc.max_send_message_length", MAX_CONTROL_BYTES),
            ],
        )
        self._server.add_generic_rpc_handlers((control_handler(begin, finish),))

    def start(self) -> int:
        """Bind and start Host control, returning the actual port for loopback clients."""
        if self._port is None:
            port = self._server.add_insecure_port(self._endpoint)
            if port == 0:
                raise RuntimeError("cannot bind Ascend graph control endpoint")
            self._server.start()
            self._port = port
        return self._port

    def stop(self) -> None:
        """Reject new RPCs; cancellation does not cancel an admitted device Future."""
        self._server.stop(grace=0)

    def close(self) -> None:
        """Join RPC callbacks after the owner has drained or quarantined device work."""
        self.stop()
        self._executor.shutdown(wait=True, cancel_futures=True)


@dataclass
class _PeerControl:
    channel: grpc.Channel
    begin: Any
    finish: Any
    worker_start_id: str | None = None
    ready_revision: int | None = None


class GraphStepControl:
    """Coordinate all fixed Workers under one absolute step deadline.

    Every peer RPC is submitted before waiting for any peer: graph preparation
    and collectives can require all ranks to join before any one finishes.
    """

    def __init__(self, config: AscendGraphConfig) -> None:
        self.config = config
        self._peers: list[_PeerControl] = []
        for worker in config.participants:
            channel = grpc.insecure_channel(
                worker.control_endpoint,
                options=[
                    ("grpc.max_send_message_length", MAX_CONTROL_BYTES),
                    ("grpc.max_receive_message_length", MAX_CONTROL_BYTES),
                ],
            )
            methods = [
                channel.unary_unary(
                    f"/{SERVICE}/{name}",
                    request_serializer=encode_message,
                    response_deserializer=decode_message,
                )
                for name in ("Begin", "Finish")
            ]
            self._peers.append(_PeerControl(channel, *methods))
        self._sequence = 0
        self._active: dict[str, object] | None = None
        self._requests: list[dict[str, object]] = []
        self._poisoned = False
        self._deadline = 0.0

    @property
    def worker_start_id(self) -> str | None:
        """Expose the original single-Worker identity for existing diagnostics."""
        return self._peers[0].worker_start_id

    @property
    def ready_revision(self) -> int | None:
        """Expose the original single-Worker revision for existing diagnostics."""
        return self._peers[0].ready_revision

    def _remaining(self) -> float:
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("graph group step deadline expired")
        return remaining

    def _fanout(self, method: str) -> list[dict[str, Any]]:
        futures = []
        try:
            for peer, request in zip(self._peers, self._requests, strict=True):
                futures.append(getattr(peer, method).future(request, timeout=self._remaining()))
            return [future.result(timeout=self._remaining()) for future in futures]
        except BaseException:
            self._poisoned = True
            for future in futures:
                future.cancel()
            raise

    def begin(self, phase: str, capacity: int = 0, *, graph_phase: str | None = None) -> None:
        """Wait until every Worker owns admission, preserving independent identities.

        Only V4 Full capture/replay carries a graph phase, pairing each Frontend
        graph with its independently captured Worker collective graph.
        """
        if self._poisoned or self._active is not None:
            raise RuntimeError("graph control is poisoned or already executing a step")
        paired = (
            self.config.mode == "full"
            and self.config.model.expert_compute == "deepseek_v4"
            and phase in {"capture", "replay"}
        )
        if paired:
            if graph_phase not in ("decode", "mixed"):
                raise ValueError("V4 full capture/replay requires a graph phase")
        elif graph_phase is not None:
            raise ValueError("graph phase is only valid for V4 full capture/replay")
        self._sequence += 1
        self._deadline = time.monotonic() + self.config.timeout_seconds
        self._requests = [
            {
                "version": 1,
                "identity": self.config.identity(),
                "sequence": self._sequence,
                "phase": phase,
                "capacity": capacity,
                "worker_start_id": peer.worker_start_id,
                "ready_revision": peer.ready_revision,
            }
            for peer in self._peers
        ]
        if paired:
            for request in self._requests:
                request["graph_phase"] = graph_phase
        self._active = self._requests[0]
        try:
            replies = self._fanout("begin")
            for peer, reply in zip(self._peers, replies, strict=True):
                if reply.get("sequence") != self._sequence or reply.get("state") != "ready":
                    raise RuntimeError("Worker acknowledged the wrong graph step")
                start_id = reply.get("worker_start_id")
                revision = reply.get("ready_revision")
                if not isinstance(start_id, str) or type(revision) is not int:
                    raise RuntimeError("Worker graph identity is invalid")
                if peer.worker_start_id is not None and start_id != peer.worker_start_id:
                    raise RuntimeError("Worker restarted during graph execution")
                if peer.ready_revision is not None and revision != peer.ready_revision:
                    raise RuntimeError("Worker weights changed during graph execution")
                peer.worker_start_id = start_id
                peer.ready_revision = revision
        except BaseException:
            self._poisoned = True
            raise

    def finish(self) -> None:
        """Prove every Worker's completion before admitting another step."""
        if self._poisoned or self._active is None:
            raise RuntimeError("graph control has no healthy active step")
        try:
            for reply in self._fanout("finish"):
                if reply.get("sequence") != self._sequence or reply.get("state") != "complete":
                    raise RuntimeError("Worker did not complete the matching graph step")
        except BaseException:
            self._poisoned = True
            raise
        self._active = None
        self._requests.clear()
        self._deadline = 0.0

    def poison(self) -> None:
        """Stop all admission after any partially submitted or unproven graph work."""
        self._poisoned = True

    def close(self) -> None:
        """Close Host channels without asserting any device completion."""
        self._poisoned = True
        for peer in self._peers:
            peer.channel.close()
