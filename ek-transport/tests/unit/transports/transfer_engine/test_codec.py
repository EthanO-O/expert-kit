"""Bounded Ascend session metadata and terminal error validation."""

import json

import pytest
import torch

from expertkit_transport.errors import TransportError, TransportErrorCode, TransportProtocolError
from expertkit_transport.transports.base import WorkerEndpointConfig
from expertkit_transport.transports.transfer_engine.arena import TransferArena
from expertkit_transport.transports.transfer_engine.codec import (
    TransferOpenSession,
    decode_open_session_request,
    decode_terminal,
    encode_error,
    encode_open_session_request,
)


def spec() -> WorkerEndpointConfig:
    return WorkerEndpointConfig(7, 4, 8, 4, 3, 2, torch.float32)


@pytest.mark.parametrize("mutation", ["version", "capacity", "length", "start_id", "backend"])
def test_open_session_rejects_incompatible_or_invalid_arena_metadata(mutation: str) -> None:
    arena = TransferArena(spec(), device="cpu", capacity=2)
    request = TransferOpenSession(
        "epoch", "peer:19000", "generation", "start", "ascend_direct", arena.descriptor
    )
    payload = encode_open_session_request(request, spec())
    assert decode_open_session_request(payload, spec(), "start", "ascend_direct") == request
    fields = json.loads(payload)
    if mutation == "version":
        fields["version"] = 999
    elif mutation == "capacity":
        fields["arena"]["slot_count"] = 4097
    elif mutation == "length":
        fields["arena"]["slot_stride"] = 1
    elif mutation == "start_id":
        fields["expected_worker_start_id"] = "replaced"
    else:
        fields["backend"] = "rdma"
    with pytest.raises(TransportProtocolError):
        decode_open_session_request(json.dumps(fields).encode(), spec(), "start", "ascend_direct")
    arena.close()


@pytest.mark.parametrize("payload", [b"{", b"[]", b"x" * 16385])
def test_open_session_rejects_malformed_or_oversized_control_payload(payload: bytes) -> None:
    with pytest.raises(TransportProtocolError):
        decode_open_session_request(payload, spec(), "start", "ascend_direct")


def test_terminal_preserves_retry_and_topology_metadata_and_checks_sequence() -> None:
    error = TransportError(
        TransportErrorCode.EXPERT_NOT_READY,
        retryable=True,
        observed_topology_version=10,
        min_topology_version=11,
        unavailable_expert_ids=(2,),
        diagnostic="loading",
    )
    payload = encode_error(4, error)
    with pytest.raises(TransportError) as captured:
        decode_terminal(payload, 4, spec())
    result = captured.value
    assert result.code == error.code and result.retryable
    assert result.observed_topology_version == 10
    assert result.min_topology_version == 11
    assert result.unavailable_expert_ids == (2,)
    with pytest.raises(TransportProtocolError, match="sequence"):
        decode_terminal(payload, 5, spec())
