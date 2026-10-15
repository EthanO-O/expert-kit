"""Bounded gRPC control messages outside Ascend device graph capture."""

from __future__ import annotations

import json
from collections.abc import Callable
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


class GraphStepControl:
    """Serialize one graph session with gRPC deadlines and explicit completion."""

    def __init__(self, config: AscendGraphConfig) -> None:
        self.config = config
        self._channel = grpc.insecure_channel(
            config.control_endpoint,
            options=[
                ("grpc.max_send_message_length", MAX_CONTROL_BYTES),
                ("grpc.max_receive_message_length", MAX_CONTROL_BYTES),
            ],
        )
        self._begin = self._channel.unary_unary(
            f"/{SERVICE}/Begin",
            request_serializer=encode_message,
            response_deserializer=decode_message,
        )
        self._finish = self._channel.unary_unary(
            f"/{SERVICE}/Finish",
            request_serializer=encode_message,
            response_deserializer=decode_message,
        )
        self._sequence = 0
        self._active: dict[str, object] | None = None
        self._poisoned = False
        self.worker_start_id: str | None = None
        self.ready_revision: int | None = None

    def begin(self, phase: str, capacity: int = 0) -> None:
        """Wait until the Worker owns device admission and can start this phase."""
        if self._poisoned or self._active is not None:
            raise RuntimeError("graph control is poisoned or already executing a step")
        self._sequence += 1
        request = {
            "version": 1,
            "identity": self.config.identity(),
            "sequence": self._sequence,
            "phase": phase,
            "capacity": capacity,
            "worker_start_id": self.worker_start_id,
            "ready_revision": self.ready_revision,
        }
        self._active = request
        try:
            reply = self._begin(request, timeout=self.config.timeout_seconds)
            if reply.get("sequence") != self._sequence or reply.get("state") != "ready":
                raise RuntimeError("Worker acknowledged the wrong graph step")
            start_id = reply.get("worker_start_id")
            revision = reply.get("ready_revision")
            if not isinstance(start_id, str) or not isinstance(revision, int):
                raise RuntimeError("Worker graph identity is invalid")
            if self.worker_start_id is not None and start_id != self.worker_start_id:
                raise RuntimeError("Worker restarted during graph execution")
            if self.ready_revision is not None and revision != self.ready_revision:
                raise RuntimeError("Worker weights changed during graph execution")
            self.worker_start_id = start_id
            self.ready_revision = revision
        except BaseException:
            self._poisoned = True
            raise

    def finish(self) -> None:
        """Prove Worker completion before admitting another step."""
        request = self._active
        if self._poisoned or request is None:
            raise RuntimeError("graph control has no healthy active step")
        try:
            reply = self._finish(request, timeout=self.config.timeout_seconds)
            if reply.get("sequence") != self._sequence or reply.get("state") != "complete":
                raise RuntimeError("Worker did not complete the matching graph step")
        except BaseException:
            self._poisoned = True
            raise
        self._active = None

    def poison(self) -> None:
        """Stop admission after any partially submitted or unproven graph work."""
        self._poisoned = True

    def close(self) -> None:
        """Close the Host channel without asserting device completion."""
        self._poisoned = True
        self._channel.close()
