"""Bounded gRPC control messages outside Ascend device graph capture."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import grpc

from expertkit_transport.transports.ascend_graph.config import AscendGraphConfig

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
    begin: Callable[[dict[str, Any], grpc.ServicerContext], dict[str, object]],
    finish: Callable[[dict[str, Any], grpc.ServicerContext], dict[str, object]],
) -> grpc.GenericRpcHandler:
    """Bind the two coarse-step methods using the existing gRPC dependency."""

    def bind(method):
        def invoke(payload, context):
            try:
                message = decode_message(payload)
            except (ValueError, UnicodeError) as error:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
            return method(message, context)

        return grpc.unary_unary_rpc_method_handler(invoke, response_serializer=encode_message)

    return grpc.method_handlers_generic_handler(
        SERVICE,
        {name: bind(method) for name, method in (("Begin", begin), ("Finish", finish))},
    )


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

    def begin(self, phase: str, capacity: int = 0) -> None:
        """Wait until every Worker owns admission, preserving independent identities."""
        if self._poisoned or self._active is not None:
            raise RuntimeError("graph control is poisoned or already executing a step")
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
