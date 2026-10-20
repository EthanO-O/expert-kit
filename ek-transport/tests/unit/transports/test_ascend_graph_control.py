"""Keep gRPC status and deadline handling outside graph execution."""

import threading

import grpc
import pytest

from expertkit_transport.transports.ascend_graph.contracts import (
    GraphAdmissionTimeout,
    GraphCompletionUnproven,
    GraphControlRejected,
)
from expertkit_transport.transports.ascend_graph.control import (
    SERVICE,
    GraphControlServer,
    decode_message,
    encode_message,
)


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (GraphControlRejected("not admitted"), grpc.StatusCode.FAILED_PRECONDITION),
        (GraphAdmissionTimeout("admission expired"), grpc.StatusCode.DEADLINE_EXCEEDED),
        (GraphCompletionUnproven("completion unknown"), grpc.StatusCode.INTERNAL),
    ],
)
def test_real_loopback_maps_domain_errors_without_passing_rpc_context(error, status):
    received = []

    def command(request, *, timeout):
        received.append((request, timeout))
        raise error

    server = GraphControlServer("127.0.0.1:0", begin=command, finish=command)
    port = server.start()
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            method = channel.unary_unary(f"/{SERVICE}/Begin", request_serializer=encode_message)
            with pytest.raises(grpc.RpcError) as result:
                method({"sequence": 1}, timeout=2)
            assert result.value.code() == status
            assert result.value.details() == str(error)
        assert len(received) == 1
        assert received[0][0] == {"sequence": 1}
        # gRPC rounds its wire deadline up to the next supported time unit.
        assert 0 < received[0][1] <= 3
    finally:
        server.close()


def test_stop_rejects_new_rpc_without_cancelling_admitted_domain_work():
    admitted = threading.Event()
    release = threading.Event()
    complete = threading.Event()

    def command(request, *, timeout):
        admitted.set()
        assert release.wait(2)
        complete.set()
        return {"sequence": request["sequence"], "state": "complete"}

    server = GraphControlServer("127.0.0.1:0", begin=command, finish=command)
    port = server.start()
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            method = channel.unary_unary(
                f"/{SERVICE}/Finish",
                request_serializer=encode_message,
                response_deserializer=decode_message,
            )
            pending = method.future({"sequence": 1}, timeout=2)
            assert admitted.wait(2)
            server.stop()
            with pytest.raises(grpc.RpcError):
                method({"sequence": 2}, timeout=1)
            assert not complete.is_set()
            release.set()
            assert complete.wait(2)
            with pytest.raises(grpc.RpcError):
                pending.result(timeout=2)
    finally:
        release.set()
        server.close()
