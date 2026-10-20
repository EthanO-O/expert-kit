"""Check bounded graph control over its real gRPC boundary on CPU."""

import threading
from concurrent.futures import ThreadPoolExecutor

import grpc
import pytest
import torch
from pydantic import ValidationError

from expertkit_transport.transports.ascend_graph.config import AscendGraphConfig
from expertkit_transport.transports.ascend_graph.contracts import GraphCompletionUnproven
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

    def begin(request, *, timeout):
        requests.append(request)
        return {
            "state": "ready",
            "sequence": request["sequence"],
            "worker_start_id": start_id[0],
            "ready_revision": 7,
        }

    def finish(request, *, timeout):
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


def two_worker_config(ports=(12344, 12345)):
    """Use canonical rank order with disjoint contiguous ownership."""
    data = config_for().model_dump(exclude={"worker_id", "control_endpoint"})
    data["workers"] = [
        {
            "worker_id": f"worker{rank}",
            "control_endpoint": f"127.0.0.1:{port}",
            "expert_start": rank * 32,
            "expert_end": (rank + 1) * 32,
        }
        for rank, port in enumerate(ports)
    ]
    return AscendGraphConfig.model_validate(data)


def test_canonical_participants_preserve_legacy_identity_and_reject_partial_ownership():
    legacy = config_for()
    explicit = legacy.model_dump(exclude={"worker_id", "control_endpoint"})
    explicit["workers"] = [legacy.participants[0].model_dump()]
    assert AscendGraphConfig.model_validate(explicit).identity() == legacy.identity()
    two = two_worker_config()
    assert two.worker_rank("worker1") == 2
    with pytest.raises(ValueError, match="absent"):
        two.worker_rank("missing")
    for change in (
        {"expert_start": 31},
        {"worker_id": "worker0"},
        {"control_endpoint": two.participants[0].control_endpoint},
    ):
        invalid = two.model_dump()
        invalid["workers"][1].update(change)
        with pytest.raises(ValidationError):
            AscendGraphConfig.model_validate(invalid)
    with pytest.raises(ValidationError):
        AscendGraphConfig.model_validate({**two.model_dump(), "worker_id": "legacy"})


def test_shared_model_normalizes_legacy_json_and_covers_qwen_intervals():
    legacy = config_for()
    assert legacy.model.model_dump() == {
        "num_layers": 27,
        "experts_per_layer": 64,
        "hidden_dim": 2048,
        "expert_intermediate_dim": 1408,
        "top_k": 6,
        "activation_dtype": "bf16",
        "weight_dtype": "bf16",
        "activation": "silu",
        "expert_compute": "swiglu",
        "swiglu_limit": 0.0,
        "quantization": None,
    }
    assert (
        AscendGraphConfig.model_validate_json(legacy.model_dump_json()).identity()
        == legacy.identity()
    )
    two = two_worker_config()
    old = two.model_dump(exclude={"model"})
    assert AscendGraphConfig.model_validate(old).identity() == two.identity()
    qwen = two.model_dump()
    qwen["model"].update(num_layers=48, experts_per_layer=128, expert_intermediate_dim=768, top_k=8)
    qwen["layer_ids"] = list(range(48))
    for index, worker in enumerate(qwen["workers"]):
        worker.update(expert_start=index * 64, expert_end=(index + 1) * 64)
    parsed = AscendGraphConfig.model_validate(qwen)
    assert parsed.participants[1].expert_end == 128
    assert parsed.identity()["model"] == parsed.model.model_dump(mode="json")
    qwen["workers"][1]["expert_end"] = 129
    with pytest.raises(ValidationError, match="partition"):
        AscendGraphConfig.model_validate(qwen)


@pytest.mark.parametrize(
    "update",
    [
        {"num_layers": 28},
        {"hidden_dim": 4096},
        {"expert_intermediate_dim": 768},
        {"top_k": 8},
        {"expert_compute": "deepseek_v4", "swiglu_limit": 7.0},
        {
            "quantization": {
                "type": "modelslim-w8a8-dynamic",
                "bits": 8,
                "group_size": None,
                "symmetric": True,
            }
        },
    ],
)
def test_model_computation_fields_change_group_identity(update):
    original = config_for()
    model = {**original.model.model_dump(), **update}
    changed = config_for(model=model)
    assert changed.identity() != original.identity()


@pytest.mark.parametrize(
    "update",
    [
        {"hidden_dim": True},
        {"top_k": 65},
        {"activation_dtype": "fp16"},
        {"weight_dtype": "int8"},
        {"activation": "gelu"},
        {"swiglu_limit": 1.0},
        {
            "quantization": {
                "type": "modelslim-w8a8-dynamic",
                "bits": 4,
                "group_size": None,
                "symmetric": True,
            }
        },
    ],
)
def test_shared_model_rejects_invalid_computation(update):
    with pytest.raises(ValidationError):
        config_for(model={**config_for().model.model_dump(), **update})


def test_explicit_model_is_complete_and_layers_are_bounded():
    with pytest.raises(ValidationError):
        config_for(model={"hidden_dim": 2048})
    with pytest.raises(ValidationError, match=r"model.num_layers"):
        config_for(layer_ids=[27])


def test_buffer_geometry_conflict_fails_before_process_group_creation():
    from types import SimpleNamespace

    from expertkit_transport.transports.ascend_graph.session import HcclGraphSession

    with pytest.raises(ValueError, match="buffer geometry"):
        HcclGraphSession(
            config_for(), rank=0, device=SimpleNamespace(type="npu"), hidden_dim=1, top_k=6
        )


def test_all_peer_begin_is_concurrent_and_partial_finish_preserves_poisoned_ownership():
    begin_barrier = threading.Barrier(2, timeout=2)
    finish_arrived = [threading.Event(), threading.Event()]
    release_finish = threading.Event()
    requests = [[], []]
    fail_finish = [False]
    servers = []
    ports = []

    def begin_for(rank):
        def begin(request, *, timeout):
            requests[rank].append(request)
            begin_barrier.wait()
            return {
                "state": "ready",
                "sequence": request["sequence"],
                "worker_start_id": f"start-{rank}",
                "ready_revision": 7 + rank,
            }

        return begin

    def finish_for(rank):
        def finish(request, *, timeout):
            finish_arrived[rank].set()
            if rank == 1:
                assert release_finish.wait(2)
                if fail_finish[0]:
                    raise GraphCompletionUnproven("peer completion failed")
            return {"state": "complete", "sequence": request["sequence"]}

        return finish

    with ThreadPoolExecutor(max_workers=4) as rpc_executor, ThreadPoolExecutor(1) as caller:
        for rank in range(2):
            server = grpc.server(rpc_executor)
            server.add_generic_rpc_handlers((control_handler(begin_for(rank), finish_for(rank)),))
            ports.append(server.add_insecure_port("127.0.0.1:0"))
            server.start()
            servers.append(server)
        control = GraphStepControl(two_worker_config(ports))
        try:
            control.begin("prepare")
            deadline = control._deadline
            pending = caller.submit(control.finish)
            assert all(event.wait(2) for event in finish_arrived)
            assert control._active is not None and control._deadline == deadline
            release_finish.set()
            pending.result(timeout=2)
            control.begin("capture", 4)
            for rank in range(2):
                assert requests[rank][-1]["worker_start_id"] == f"start-{rank}"
                assert requests[rank][-1]["ready_revision"] == 7 + rank
            fail_finish[0] = True
            with pytest.raises(grpc.RpcError):
                control.finish()
            assert control._poisoned and control._active is not None and control._deadline > 0
            with pytest.raises(RuntimeError, match="poisoned"):
                control.begin("replay", 4)
        finally:
            release_finish.set()
            control.close()
            for server in servers:
                server.stop(0).wait(2)


def test_fp32_reduce_zeroes_frontend_each_layer_and_preserves_worker_partial():
    from expertkit_transport.transports.ascend_graph.session import (
        GraphLayerBuffers,
        HcclGraphSession,
    )

    class Group:
        def __init__(self, rank):
            self.rank = rank
            self.seen = []

        def reduce(self, tensors, options):
            self.seen.append(tensors[0].clone())
            assert options.rootRank == 0
            if self.rank == 0:
                assert torch.count_nonzero(tensors[0]) == 0
                tensors[0].fill_(1.234567)
            return self

        def wait(self):
            return None

    for rank in (0, 1, 2):
        session = HcclGraphSession.__new__(HcclGraphSession)
        session.rank = rank
        session._group = Group(rank)
        buffers = GraphLayerBuffers(
            torch.empty(2, 4),
            torch.empty(2, 2),
            torch.empty(2, 2),
            torch.empty(2, 4, dtype=torch.bfloat16),
            torch.full((2, 4), 9.0),
        )
        for _ in range(2):
            session.combine(buffers)
        if rank == 0:
            assert torch.equal(buffers.output, torch.full_like(buffers.output, 1.234567))
        else:
            assert all(torch.equal(value, torch.full((2, 4), 9.0)) for value in session._group.seen)
