"""Check bounded graph control over its real gRPC boundary on CPU."""

from concurrent.futures import ThreadPoolExecutor

import grpc
import pytest
from pydantic import ValidationError

from expertkit_transport.transports.ascend_graph.config import AscendGraphConfig
from expertkit_transport.transports.ascend_graph.control import (
    MAX_CONTROL_BYTES,
    SERVICE,
    GraphStepControl,
    control_handler,
    decode_message,
    encode_message,
)


def config_for(port: int = 12345, **updates) -> AscendGraphConfig:
    """Create a small fixed deployment shared by both control peers."""
    return AscendGraphConfig.model_validate(
        {
            "session_id": "cpu-graph-session-0001",
            "instance_id": 1,
            "worker_id": "worker",
            "weight_version": "model-revision",
            "control_endpoint": f"127.0.0.1:{port}",
            "rendezvous_host": "127.0.0.1",
            "rendezvous_port": 12346,
            "layer_ids": [1, 2],
            "capture_sizes": [1, 4],
            "timeout_seconds": 2.0,
            **updates,
        }
    )


@pytest.mark.parametrize(
    "payload", [b"[]", b"{", b"\xff", b'{"x":NaN}', b" " * (MAX_CONTROL_BYTES + 1)]
)
def test_decode_rejects_malformed_or_oversize_messages(payload) -> None:
    with pytest.raises((ValueError, UnicodeError)):
        decode_message(payload)


def test_encode_round_trip_and_limits() -> None:
    message = {"identity": config_for().identity(), "sequence": 1}
    assert decode_message(encode_message(message)) == message
    with pytest.raises(ValueError):
        encode_message({"value": "x" * MAX_CONTROL_BYTES})
    with pytest.raises(ValueError):
        encode_message({"value": float("inf")})


@pytest.mark.parametrize(
    "updates",
    [
        {"capture_sizes": []},
        {"capture_sizes": [4, 1]},
        {"capture_sizes": [True]},
        {"layer_ids": [1, 1]},
        {"instance_id": True},
        {"control_endpoint": "http://host:3"},
        {"timeout_seconds": float("nan")},
        {"unknown": 1},
    ],
)
def test_fixed_deployment_rejects_ambiguous_values(updates) -> None:
    with pytest.raises(ValidationError):
        config_for(**updates)


def test_real_grpc_control_matches_steps_and_poison_after_restart() -> None:
    requests = []
    start_id = ["start-one"]

    def begin(request, context):
        requests.append(request)
        return {
            "state": "ready",
            "sequence": request["sequence"],
            "worker_start_id": start_id[0],
            "ready_revision": 7,
        }

    def finish(request, context):
        assert request == requests[-1]
        return {"state": "complete", "sequence": request["sequence"]}

    with ThreadPoolExecutor(max_workers=2) as executor:
        server = grpc.server(executor)
        server.add_generic_rpc_handlers((control_handler(begin, finish),))
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        control = GraphStepControl(config_for(port))
        try:
            control.begin("prepare")
            with pytest.raises(RuntimeError, match="already executing"):
                control.begin("prepare")
            control.finish()
            control.begin("capture", 4)
            assert requests[-1]["worker_start_id"] == "start-one"
            assert requests[-1]["ready_revision"] == 7
            control.finish()
            start_id[0] = "start-two"
            with pytest.raises(RuntimeError, match="restarted"):
                control.begin("replay", 4)
            with pytest.raises(RuntimeError, match="poisoned"):
                control.begin("replay", 4)
            with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
                raw_begin = channel.unary_unary(f"/{SERVICE}/Begin")
                with pytest.raises(grpc.RpcError) as malformed:
                    raw_begin(b"[]", timeout=2)
                assert malformed.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        finally:
            control.close()
            server.stop(0).wait(2)
