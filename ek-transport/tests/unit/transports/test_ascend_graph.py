"""Check bounded graph control over its real gRPC boundary on CPU."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import grpc
import pytest
import torch
from pydantic import ValidationError

from expertkit_transport.transports.ascend_graph.config import (
    AscendGraphConfig,
    AscendGraphOptions,
)
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
    """Create explicit resolved facts shared by the CPU control peers."""
    return AscendGraphConfig.model_validate(
        {
            "session_id": "cpu-graph-session-0001",
            "instance_id": 1,
            "model_name": "model",
            "weight_version": "model-revision",
            "source_topology_version": 7,
            "mode": "decode",
            "model": {
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
            },
            "workers": [
                {
                    "worker_id": "worker",
                    "expected_start_id": "start-one",
                    "device": "npu:0",
                    "control_endpoint": f"127.0.0.1:{port}",
                    "experts_by_layer": [list(range(64)), list(range(64))],
                }
            ],
            "rendezvous_host": "127.0.0.1",
            "rendezvous_port": 12346,
            "layer_ids": [1, 2],
            "capture_sizes": [1, 4],
            "decode_capture_sizes": updates.get("capture_sizes", [1, 4]),
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
    ("capture_sizes", "expected"),
    [
        ([256, 64, 128, 64], (64, 128, 256)),
        ([9, 3, 9], (3, 9)),
        ([17], (17,)),
    ],
)
def test_capture_sizes_normalize_without_adding_capacities(capture_sizes, expected) -> None:
    config = config_for(capture_sizes=capture_sizes)
    assert config.capture_sizes == expected
    assert config.identity()["capture_sizes"] == list(expected)


@pytest.mark.parametrize("capture_sizes", [[], [0], [-1], [1.5], [1.0], [True], ["4"]])
def test_capture_sizes_reject_empty_or_nonpositive_strict_integers(capture_sizes) -> None:
    with pytest.raises(ValidationError, match="capture_sizes"):
        config_for(capture_sizes=capture_sizes)


@pytest.mark.parametrize("capture_sizes", [[256, 64, 128, 64], (128, 256, 64)])
def test_equivalent_capture_sizes_preserve_group_identity(capture_sizes) -> None:
    canonical = config_for(capture_sizes=[64, 128, 256])
    equivalent = config_for(capture_sizes=capture_sizes)
    assert equivalent.identity() == canonical.identity()
    document = {**canonical.model_dump(mode="json"), "capture_sizes": capture_sizes}
    assert (
        AscendGraphConfig.model_validate_json(json.dumps(document)).identity()
        == canonical.identity()
    )
    assert config_for(capture_sizes=[64, 256]).identity() != canonical.identity()


@pytest.mark.parametrize(
    "updates",
    [
        {"layer_ids": [1, 1]},
        {"layer_ids": [2, 1]},
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
            "version": 2,
            "assignment_id": request["assignment_id"],
            "state": "ready",
            "sequence": request["sequence"],
            "worker_start_id": start_id[0],
            "ready_revision": 7,
        }

    def finish(request, *, timeout):
        expected = {key: value for key, value in requests[-1].items() if key != "config"}
        expected["ready_revision"] = 7
        assert request == expected
        return {
            "version": 2,
            "assignment_id": request["assignment_id"],
            "state": "complete",
            "sequence": request["sequence"],
            "worker_start_id": start_id[0],
            "ready_revision": 7,
        }

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
    """Use canonical rank order with two exact-once expert owners."""
    data = config_for().model_dump()
    data["workers"] = [
        {
            "worker_id": f"worker{rank}",
            "expected_start_id": f"start-{rank}",
            "device": f"npu:{rank}",
            "control_endpoint": f"127.0.0.1:{port}",
            "experts_by_layer": [
                list(range(rank * 32, (rank + 1) * 32)) for _ in data["layer_ids"]
            ],
        }
        for rank, port in enumerate(ports)
    ]
    return AscendGraphConfig.model_validate(data)


def test_runtime_rank_order_and_exact_once_coverage():
    config = two_worker_config()
    assert config.worker_rank("worker1") == 2
    with pytest.raises(ValueError, match="absent"):
        config.worker_rank("missing")
    for change in (
        {"experts_by_layer": [[31, *range(32, 64)], list(range(32, 64))]},
        {"worker_id": "worker0"},
        {"control_endpoint": config.participants[0].control_endpoint},
    ):
        invalid = config.model_dump()
        invalid["workers"][1].update(change)
        with pytest.raises(ValidationError):
            AscendGraphConfig.model_validate(invalid)


def test_runtime_model_is_required_and_json_round_trips():
    config = config_for()
    assert AscendGraphConfig.model_validate_json(config.model_dump_json()) == config
    with pytest.raises(ValidationError, match="model"):
        AscendGraphConfig.model_validate(config.model_dump(exclude={"model"}))


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


@pytest.mark.parametrize("worker_count", [2, 8])
def test_all_peer_begin_is_concurrent_and_partial_finish_preserves_poisoned_ownership(worker_count):
    begin_barrier = threading.Barrier(worker_count, timeout=2)
    finish_arrived = [threading.Event() for _ in range(worker_count)]
    release_finish = threading.Event()
    requests = [[] for _ in range(worker_count)]
    fail_finish = [False]
    servers = []
    ports = []

    def begin_for(rank):
        def begin(request, *, timeout):
            requests[rank].append(request)
            begin_barrier.wait()
            return {
                "version": 2,
                "assignment_id": request["assignment_id"],
                "state": "ready",
                "sequence": request["sequence"],
                "worker_start_id": f"start-{rank}",
                "ready_revision": 7 + rank,
            }

        return begin

    def finish_for(rank):
        def finish(request, *, timeout):
            finish_arrived[rank].set()
            if rank == worker_count - 1:
                assert release_finish.wait(2)
                if fail_finish[0]:
                    raise GraphCompletionUnproven("peer completion failed")
            return {
                "version": 2,
                "assignment_id": request["assignment_id"],
                "state": "complete",
                "sequence": request["sequence"],
                "worker_start_id": f"start-{rank}",
                "ready_revision": 7 + rank,
            }

        return finish

    with (
        ThreadPoolExecutor(max_workers=2 * worker_count) as rpc_executor,
        ThreadPoolExecutor(1) as caller,
    ):
        for rank in range(worker_count):
            server = grpc.server(rpc_executor)
            server.add_generic_rpc_handlers((control_handler(begin_for(rank), finish_for(rank)),))
            ports.append(server.add_insecure_port("127.0.0.1:0"))
            server.start()
            servers.append(server)
        if worker_count == 8:
            data = v4_config_data()
            for worker, port in zip(data["workers"], ports, strict=True):
                worker["control_endpoint"] = f"127.0.0.1:{port}"
            config = AscendGraphConfig.model_validate(data)
        else:
            config = two_worker_config(ports)
        control = GraphStepControl(config)
        try:
            control.begin("prepare")
            deadline = control._deadline
            pending = caller.submit(control.finish)
            assert all(event.wait(2) for event in finish_arrived)
            assert control._active is not None and control._deadline == deadline
            release_finish.set()
            pending.result(timeout=2)
            control.begin("capture", 4)
            for rank in range(worker_count):
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


def v4_config_data():
    """Declare explicit V4 geometry for existing buffer and control tests."""
    data = two_worker_config().model_dump()
    data["model"].update(
        num_layers=43,
        experts_per_layer=256,
        hidden_dim=4096,
        expert_intermediate_dim=2048,
        top_k=6,
        expert_compute="deepseek_v4",
        swiglu_limit=10.0,
        quantization={
            "type": "modelslim-w8a8-dynamic",
            "bits": 8,
            "group_size": None,
            "symmetric": True,
        },
    )
    data["workers"] = [
        {
            "worker_id": f"worker{rank}",
            "expected_start_id": f"start-{rank}",
            "control_endpoint": f"127.0.0.1:{12344 + rank}",
            "device": f"npu:{rank}",
            "experts_by_layer": [
                list(range(rank * 32, (rank + 1) * 32)) for _ in data["layer_ids"]
            ],
        }
        for rank in range(8)
    ]
    return data


@pytest.mark.parametrize("worker_count", [1, 3, 8])
def test_runtime_allows_worker_counts_without_experiment_whitelists(worker_count):
    data = v4_config_data()
    data["workers"] = [
        {
            "worker_id": f"worker{rank}",
            "expected_start_id": f"start-{rank}",
            "control_endpoint": f"127.0.0.1:{12344 + rank}",
            "device": f"npu:{rank}",
            "experts_by_layer": [list(range(rank, 256, worker_count)) for _ in data["layer_ids"]],
        }
        for rank in range(worker_count)
    ]
    config = AscendGraphConfig.model_validate(data)
    assert len(config.participants) == worker_count


@pytest.mark.parametrize(
    "update",
    [
        {"num_layers": 42},
        {"hidden_dim": 2048},
        {"expert_intermediate_dim": 2047},
        {"top_k": 5},
        {"swiglu_limit": 9.0},
        {"quantization": None},
    ],
)
def test_runtime_geometry_has_no_experiment_whitelist(update):
    data = v4_config_data()
    data["model"].update(update)
    assert AscendGraphConfig.model_validate(data).model.model_dump().items() >= update.items()


@pytest.mark.parametrize(
    "index,update",
    [
        (0, {"experts_by_layer": [list(range(31)), list(range(32))]}),
        (1, {"experts_by_layer": [list(range(31, 64)), list(range(32, 64))]}),
        (7, {"experts_by_layer": [list(range(224, 255)), list(range(224, 256))]}),
        (1, {"worker_id": "worker0"}),
        (1, {"control_endpoint": "127.0.0.1:12344"}),
    ],
)
def test_assignment_coverage_and_identity_are_checked(index, update):
    data = v4_config_data()
    data["workers"][index].update(update)
    with pytest.raises(ValidationError):
        AscendGraphConfig.model_validate(data)


@pytest.mark.parametrize("workers", [1, 2, 8])
def test_real_buffer_allocation_selects_multiworker_fp32_reduction(workers):
    from expertkit_transport.transports.ascend_graph.session import HcclGraphSession

    config = (
        config_for()
        if workers == 1
        else two_worker_config()
        if workers == 2
        else AscendGraphConfig.model_validate(v4_config_data())
    )

    class Group:
        def __init__(self, rank):
            self.rank = rank
            self.reductions = 0
            self.broadcasts = 0

        def reduce(self, tensors, options):
            self.reductions += 1
            assert tensors[0].dtype == torch.float32
            assert options.rootRank == 0
            if self.rank == 0:
                assert torch.count_nonzero(tensors[0]) == 0
                tensors[0].fill_(1.234567)
            else:
                assert torch.all(tensors[0] == 9)
            return self

        def broadcast(self, tensors, options):
            self.broadcasts += 1
            assert options.rootRank == 1
            assert tensors[0].dtype == torch.bfloat16
            return self

        def wait(self):
            return None

    for rank in (0, workers):
        session = HcclGraphSession.__new__(HcclGraphSession)
        session.config, session.rank = config, rank
        session.device = torch.device("cpu")
        session._hidden_dim, session._top_k = config.model.hidden_dim, config.model.top_k
        session._buffers = {}
        session._group = Group(rank)
        buffers = session.buffers(1)
        assert session.buffers(1) is buffers
        assert buffers.output.dtype == torch.bfloat16
        assert (buffers.partial_output is not None) == (workers > 1)
        if buffers.partial_output is not None:
            assert buffers.partial_output.dtype == torch.float32
            buffers.partial_output.fill_(9)
        for _ in range(2):
            session.combine(buffers)
        if workers > 1:
            assert session._group.reductions == 2 and session._group.broadcasts == 0
            if rank == 0:
                assert torch.equal(buffers.output, torch.full_like(buffers.output, 1.234567))
        else:
            assert session._group.broadcasts == 2 and session._group.reductions == 0


def qwen_config_data():
    """Declare explicit Qwen checkpoint facts for existing control tests."""
    data = v4_config_data()
    data["model"].update(
        num_layers=48,
        experts_per_layer=128,
        hidden_dim=2048,
        expert_intermediate_dim=768,
        top_k=8,
        expert_compute="swiglu",
        swiglu_limit=0.0,
    )
    data["layer_ids"] = list(range(48))
    for rank, worker in enumerate(data["workers"]):
        worker["experts_by_layer"] = [
            list(range(rank * 16, (rank + 1) * 16)) for _ in data["layer_ids"]
        ]
    return data


@pytest.mark.parametrize("mode", ["decode", "full"])
def test_exact_qwen_accepts_eight_ordered_16_expert_shards(mode):
    data = qwen_config_data()
    data["mode"] = mode
    config = AscendGraphConfig.model_validate(data)
    assert [config.worker_rank(f"worker{rank}") for rank in range(8)] == list(range(1, 9))
    assert (
        AscendGraphConfig.model_validate_json(config.model_dump_json()).identity()
        == config.identity()
    )


@pytest.mark.parametrize("mode", ["decode", "full"])
@pytest.mark.parametrize("change", ["missing", "reversed", "gap", "overlap", "layers"])
def test_assignments_reject_incomplete_or_ambiguous_declarations(mode, change):
    data = qwen_config_data()
    data["mode"] = mode
    if change == "missing":
        data["workers"].pop()
    elif change == "reversed":
        data["workers"].reverse()
    elif change == "gap":
        data["workers"][3]["experts_by_layer"][0].pop(0)
    elif change == "overlap":
        data["workers"][3]["experts_by_layer"][0].insert(0, 47)
    else:
        data["layer_ids"].pop()
    with pytest.raises(ValidationError):
        AscendGraphConfig.model_validate(data)


def test_runtime_capture_capacity_has_no_experiment_limit():
    data = qwen_config_data()
    data.update(mode="full", capture_sizes=[4096, 3], decode_capture_sizes=[3])
    assert AscendGraphConfig.model_validate(data).capture_sizes == (3, 4096)


def test_v4_full_capture_profile_uses_only_requested_capacities():
    values = v4_config_data()
    values.update(mode="full", capture_sizes=[257, 3, 257], decode_capture_sizes=[3])
    config = AscendGraphConfig.model_validate(values)
    assert config.capture_sizes == (3, 257)
    assert config.model.expert_compute == "deepseek_v4"


def test_v4_control_phase_field_is_narrow_and_legacy_payload_unchanged(monkeypatch):
    values = v4_config_data()
    values.update(
        mode="full",
        layer_ids=tuple(range(43)),
        capture_sizes=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048),
        decode_capture_sizes=(1, 2, 4, 8, 16, 32, 64, 128, 256),
    )
    for worker in values["workers"]:
        worker["experts_by_layer"] = [worker["experts_by_layer"][0] for _ in values["layer_ids"]]
    control = GraphStepControl(AscendGraphConfig.model_validate(values))
    requests = []

    def fanout(method):
        if method == "begin":
            requests.extend(dict(item) for item in control._requests)
            return [
                dict(
                    version=2,
                    assignment_id=control._assignment_id,
                    sequence=control._sequence,
                    state="ready",
                    worker_start_id=peer.worker_start_id,
                    ready_revision=1,
                )
                for peer in control._peers
            ]
        return [
            dict(
                version=2,
                assignment_id=control._assignment_id,
                sequence=control._sequence,
                state="complete",
                worker_start_id=peer.worker_start_id,
                ready_revision=peer.ready_revision,
            )
            for peer in control._peers
        ]

    monkeypatch.setattr(control, "_fanout", fanout)
    try:
        for graph_phase in (None, "bad", []):
            with pytest.raises(ValueError):
                control.begin("capture", 4, graph_phase=graph_phase)
        with pytest.raises(ValueError):
            control.begin("warmup", 4, graph_phase="decode")
        control.begin("prepare")
        control.finish()
        requests.clear()
        for phase in ("decode", "mixed"):
            control.begin("capture", 4, graph_phase=phase)
            control.finish()
        assert [item["graph_phase"] for item in requests] == ["decode"] * 8 + ["mixed"] * 8
        requests.clear()
        control.begin("warmup", 4)
        control.finish()
        assert all("graph_phase" not in item for item in requests)
    finally:
        control.close()


def test_legacy_control_rejects_graph_phase_before_any_rpc():
    control = GraphStepControl(config_for())
    try:
        with pytest.raises(ValueError, match="only valid"):
            control.begin("capture", 4, graph_phase="decode")
        assert control._sequence == 0 and control._active is None
    finally:
        control.close()


@pytest.mark.parametrize("capture_sizes", [[9, 3, 9], [17], [256, 64, 128, 64]])
def test_options_and_runtime_share_capacity_normalization(capture_sizes):
    options = AscendGraphOptions(
        rendezvous_host="127.0.0.1", rendezvous_port=12346, capture_sizes=capture_sizes
    )
    assert options.capture_sizes == config_for(capture_sizes=capture_sizes).capture_sizes


@pytest.mark.parametrize("capture_sizes", [[], [0], [-1], [True], [1.0], ["4"]])
def test_options_reject_invalid_capacities(capture_sizes):
    with pytest.raises(ValidationError, match="capture_sizes"):
        AscendGraphOptions(
            rendezvous_host="127.0.0.1", rendezvous_port=12346, capture_sizes=capture_sizes
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"rendezvous_host": ""},
        {"rendezvous_host": "bad host"},
        {"rendezvous_host": "http://host"},
        {"rendezvous_port": True},
        {"rendezvous_port": 0},
        {"rendezvous_port": 65536},
        {"timeout_seconds": float("inf")},
        {"timeout_seconds": 0.0},
        {"unknown": 1},
    ],
)
def test_options_and_runtime_reject_invalid_common_fields(updates):
    options = {"rendezvous_host": "127.0.0.1", "rendezvous_port": 12346, **updates}
    with pytest.raises(ValidationError):
        AscendGraphOptions.model_validate(options)
    with pytest.raises(ValidationError):
        config_for(**updates)


@pytest.mark.parametrize(
    "legacy_field",
    [
        "session_id",
        "workers",
        "worker_id",
        "control_endpoint",
        "model",
        "layer_ids",
        "instance_id",
        "weight_version",
    ],
)
def test_options_file_explains_legacy_deployment_migration(tmp_path, legacy_field):
    path = tmp_path / "graph.json"
    path.write_text(
        json.dumps(
            {
                "rendezvous_host": "127.0.0.1",
                "rendezvous_port": 12346,
                legacy_field: None,
            }
        )
    )
    with pytest.raises(ValueError, match=r"ordinary ek-cli schedule.*READY"):
        AscendGraphOptions.from_file(path)
    with pytest.raises(ValueError, match="control_listen/control_advertise"):
        AscendGraphConfig.from_file(path)


def test_options_only_file_round_trip_and_public_exports(tmp_path):
    from expertkit_transport.ascend_graph import AscendGraphOptions as PublicOptions
    from expertkit_transport.ascend_graph import AscendGraphWorker

    path = tmp_path / "options.json"
    path.write_text('{"rendezvous_host":"worker-host","rendezvous_port":12346}')
    options = PublicOptions.from_file(path)
    assert options.capture_sizes == (1, 2, 4, 8, 16, 32, 64)
    assert options.mode == "decode"
    assert isinstance(config_for().participants[0], AscendGraphWorker)
    assert set(type(options).model_fields) == {
        "mode",
        "capture_sizes",
        "timeout_seconds",
        "rendezvous_host",
        "rendezvous_port",
    }


def test_noncontiguous_layer_dependent_assignments_include_empty_layers():
    data = two_worker_config().model_dump(mode="json")
    data["model"]["experts_per_layer"] = 6
    data["workers"][0]["experts_by_layer"] = [[0, 2, 4], []]
    data["workers"][1]["experts_by_layer"] = [[1, 3, 5], [0, 1, 2, 3, 4, 5]]
    config = AscendGraphConfig.model_validate(data)
    worker = config.participants[0]
    mapped = worker.expert_ids_by_layer(config.layer_ids)
    assert mapped == {1: (0, 2, 4), 2: ()}
    mapped[1] = ()
    assert worker.experts_by_layer == ((0, 2, 4), ())
    with pytest.raises(ValidationError, match="frozen"):
        worker.device = "npu:1"
    with pytest.raises(ValidationError, match="frozen"):
        config.model.hidden_dim = 4
    with pytest.raises(ValidationError, match="frozen"):
        config.weight_version = "other"
    with pytest.raises(ValueError):
        worker.expert_ids_by_layer((1,))


@pytest.mark.parametrize(
    "change",
    [
        {"expected_start_id": None},
        {"expected_start_id": ""},
        {"expected_start_id": "x" * 129},
        {"device": "cuda:0"},
        {"device": "npu"},
        {"device": "npu:-1"},
        {"control_endpoint": "http://host:1234"},
        {"control_endpoint": "host:0"},
        {"expert_start": 0},
        {"experts_by_layer": [[], []]},
        {"experts_by_layer": [[0, 0], []]},
        {"experts_by_layer": [[2, 0], []]},
        {"experts_by_layer": [[-1], []]},
        {"experts_by_layer": [[True], []]},
        {"experts_by_layer": [[1.0], []]},
        {"experts_by_layer": [[64], []]},
        {"experts_by_layer": [list(range(64))]},
    ],
)
def test_runtime_rejects_invalid_worker_dependencies(change):
    data = config_for().model_dump(mode="json")
    data["workers"][0].update(change)
    with pytest.raises(ValidationError):
        AscendGraphConfig.model_validate(data)


def test_expected_start_is_required_in_runtime_snapshot():
    data = config_for().model_dump(mode="json")
    data["workers"][0].pop("expected_start_id")
    with pytest.raises(ValidationError, match="expected_start_id"):
        AscendGraphConfig.model_validate(data)


def test_assignment_digest_is_canonical_and_ignores_topology_provenance():
    import hashlib

    config = config_for()
    encoded = json.dumps(config.identity(), sort_keys=True, separators=(",", ":")).encode()
    assert config.assignment_id == hashlib.sha256(encoded).hexdigest()
    assert len(config.assignment_id) == 64
    assert config.identity()["version"] == 2
    assert "source_topology_version" not in config.identity()
    changed = config_for(source_topology_version=10000)
    assert config.assignment_id == changed.assignment_id
    reordered = dict(reversed(list(config.model_dump(mode="json").items())))
    assert AscendGraphConfig.model_validate(reordered).assignment_id == config.assignment_id
    assert (
        AscendGraphConfig.model_validate_json(config.model_dump_json()).assignment_id
        == config.assignment_id
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("session_id", "another-session-identity"),
        ("instance_id", 2),
        ("model_name", "other-model"),
        ("weight_version", "other-version"),
        ("layer_ids", [0, 2]),
        ("mode", "full"),
        ("capture_sizes", [1, 5]),
        ("timeout_seconds", 3.0),
        ("rendezvous_host", "other-host"),
        ("rendezvous_port", 12347),
    ],
)
def test_every_session_fact_changes_digest(field, value):
    original = config_for()
    changed = config_for(**{field: value})
    assert changed.assignment_id != original.assignment_id


@pytest.mark.parametrize(
    "field,value",
    [
        ("worker_id", "other-worker"),
        ("expected_start_id", "other-start"),
        ("device", "npu:3"),
        ("control_endpoint", "other-host:12345"),
    ],
)
def test_selected_worker_identity_changes_digest(field, value):
    original = config_for()
    data = original.model_dump(mode="json")
    data["workers"][0][field] = value
    assert AscendGraphConfig.model_validate(data).assignment_id != original.assignment_id


def test_selected_expert_owner_changes_digest():
    original = two_worker_config()
    data = original.model_dump(mode="json")
    data["workers"][0]["experts_by_layer"][0] = [0, *range(32, 63)]
    data["workers"][1]["experts_by_layer"][0] = [*range(1, 32), 63]
    assert AscendGraphConfig.model_validate(data).assignment_id != original.assignment_id


@pytest.mark.parametrize("version", ["", "x" * 257])
def test_runtime_requires_bounded_nonempty_weight_version(version):
    with pytest.raises(ValidationError, match="weight_version"):
        config_for(weight_version=version)


@pytest.mark.parametrize(
    "decode_sizes,expected",
    [
        ([3], (3,)),
        ([9, 3, 9], (3, 9)),
        ([17, 9, 3], (3, 9, 17)),
    ],
)
def test_derived_full_decode_capacities_are_a_normalized_prefix(decode_sizes, expected):
    config = config_for(mode="full", capture_sizes=[17, 3, 9], decode_capture_sizes=decode_sizes)
    assert config.decode_capture_sizes == expected
    assert config.identity()["decode_capture_sizes"] == list(expected)
    assert "decode_capture_sizes" not in AscendGraphOptions.model_fields


@pytest.mark.parametrize("decode_sizes", [[], [0], [True], [3.0], ["3"], [9], [3, 17], [3, 9, 18]])
def test_derived_decode_capacities_reject_empty_invalid_or_nonprefix_lists(decode_sizes):
    with pytest.raises(ValidationError, match="decode_capture_sizes"):
        config_for(mode="full", capture_sizes=[3, 9, 17], decode_capture_sizes=decode_sizes)


def test_decode_mode_requires_the_complete_capture_list_and_no_default():
    with pytest.raises(ValidationError, match="decode mode"):
        config_for(capture_sizes=[3, 9], decode_capture_sizes=[3])
    data = config_for().model_dump(mode="json")
    data.pop("decode_capture_sizes")
    with pytest.raises(ValidationError, match="decode_capture_sizes"):
        AscendGraphConfig.model_validate(data)


def test_derived_decode_capacities_change_assignment_digest():
    complete = config_for(mode="full")
    limited = config_for(mode="full", decode_capture_sizes=[1])
    assert complete.assignment_id != limited.assignment_id


@pytest.mark.parametrize("version", ["revision with spaces", " ", " revision "])
def test_runtime_preserves_graph_weight_labels_without_stripping(version):
    config = config_for(weight_version=version)
    assert config.weight_version == version
    assert config.identity()["weight_version"] == version
    assert AscendGraphConfig.model_validate_json(config.model_dump_json()).weight_version == version


def test_runtime_preserves_identifier_strings_without_invented_character_rules():
    data = config_for(model_name="model with spaces").model_dump(mode="json")
    data["workers"][0].update(worker_id="worker with spaces", expected_start_id="start with spaces")
    config = AscendGraphConfig.model_validate(data)
    assert config.model_name == "model with spaces"
    assert config.worker_rank("worker with spaces") == 1
    assert config.participants[0].expected_start_id == "start with spaces"


def test_prepare_preflights_every_snapshot_before_any_rpc():
    data = two_worker_config().model_dump(mode="json")
    data["model"]["experts_per_layer"] = 160000
    for rank, worker in enumerate(data["workers"]):
        worker["experts_by_layer"] = [list(range(rank, 160000, 2)) for _ in data["layer_ids"]]
    config = AscendGraphConfig.model_validate(data)
    control = GraphStepControl(config)
    try:
        with pytest.raises(ValueError, match="exceeds its limit"):
            control.begin("prepare")
        assert control._sequence == 0 and control._active is None
        assert control._requests == [] and not control._poisoned
    finally:
        control.close()


def test_first_prepare_reply_must_match_controller_observed_start():
    def begin(request, *, timeout):
        return {
            "version": 2,
            "assignment_id": request["assignment_id"],
            "sequence": request["sequence"],
            "state": "ready",
            "worker_start_id": "restarted-before-prepare",
            "ready_revision": 1,
        }

    with ThreadPoolExecutor(max_workers=1) as executor:
        server = grpc.server(executor)
        server.add_generic_rpc_handlers((control_handler(begin, begin),))
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        control = GraphStepControl(config_for(port))
        try:
            with pytest.raises(RuntimeError, match="restarted"):
                control.begin("prepare")
            assert control._poisoned
        finally:
            control.close()
            server.stop(0).wait(2)


def test_control_sends_snapshot_only_once_and_keeps_digest_cached(monkeypatch):
    seen = []

    def respond(request, state):
        seen.append(dict(request))
        return {
            "version": 2,
            "assignment_id": request["assignment_id"],
            "sequence": request["sequence"],
            "state": state,
            "worker_start_id": "start-one",
            "ready_revision": 7,
        }

    def begin(request, *, timeout):
        return respond(request, "ready")

    def finish(request, *, timeout):
        return respond(request, "complete")

    with ThreadPoolExecutor(max_workers=1) as executor:
        server = grpc.server(executor)
        server.add_generic_rpc_handlers((control_handler(begin, finish),))
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        control = GraphStepControl(config_for(port))
        try:
            digest = control._assignment_id
            control.begin("prepare")
            control.finish()
            monkeypatch.setattr(
                AscendGraphConfig,
                "assignment_id",
                property(lambda _: pytest.fail("digest recomputed")),
            )
            control.begin("warmup", 4)
            control.finish()
            assert "config" in seen[0]
            assert seen[0]["worker_start_id"] == "start-one"
            assert seen[0]["ready_revision"] is None
            assert all("config" not in request for request in seen[1:])
            assert all(request["ready_revision"] == 7 for request in seen[1:])
            assert all(request["assignment_id"] == digest for request in seen)
            assert len(encode_message(seen[1])) < 512
        finally:
            control.close()
            server.stop(0).wait(2)


@pytest.mark.parametrize(
    "mode,requests,tokens,scheduled,expected",
    [
        ("full", 8, 32, 24, ((3, 7), 7, 12, 24)),
        ("full", 16, 32, 24, ((3, 7, 12), 12, 12, 24)),
        ("decode", 16, 32, 24, ((3, 7, 12), 12, 12, 24)),
    ],
)
def test_limits_separate_native_allocations_from_scheduling(
    mode, requests, tokens, scheduled, expected
):
    from expertkit_transport.ascend_graph import resolve_graph_limits

    options = AscendGraphOptions(
        mode=mode, capture_sizes=(12, 3, 7), rendezvous_host="host", rendezvous_port=1234
    )
    limits = resolve_graph_limits(
        options, allocated_requests=requests, allocated_tokens=tokens, scheduled_tokens=scheduled
    )
    assert limits.capture_sizes == (3, 7, 12)
    assert (
        limits.decode_capture_sizes,
        limits.running_requests,
        limits.graph_stage_tokens,
        limits.ordinary_prefill_tokens,
    ) == expected


@pytest.mark.parametrize(
    "mode,requests,tokens,reason",
    [
        ("full", 2, 32, "no configured Decode graph"),
        ("full", 8, 8, "allocated token buffers"),
        ("decode", 8, 32, "allocated request buffers"),
        ("full", True, 32, "positive integer"),
    ],
)
def test_limits_report_actual_buffer_conflict(mode, requests, tokens, reason):
    from expertkit_transport.ascend_graph import resolve_graph_limits

    options = AscendGraphOptions(
        mode=mode, capture_sizes=(3, 7, 12), rendezvous_host="host", rendezvous_port=1234
    )
    with pytest.raises(ValueError, match=reason):
        resolve_graph_limits(
            options, allocated_requests=requests, allocated_tokens=tokens, scheduled_tokens=24
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"rendezvous_host": "host"},
        {"rendezvous_port": 1234},
        {"rendezvous_host": "host", "rendezvous_port": 1234},
    ],
)
def test_user_rendezvous_overrides_are_independently_optional(overrides):
    options = AscendGraphOptions(**overrides)
    assert options.rendezvous_host == overrides.get("rendezvous_host")
    assert options.rendezvous_port == overrides.get("rendezvous_port")
    for name in ("rendezvous_host", "rendezvous_port"):
        facts = config_for().model_dump(mode="json")
        facts[name] = None
        with pytest.raises(ValidationError):
            AscendGraphConfig.model_validate(facts)
    with pytest.raises(ValidationError):
        config_for(rendezvous_port=0)


@pytest.mark.parametrize("port", [0, -1, 65536, True, "1234"])
def test_optional_rendezvous_port_still_rejects_invalid_override(port):
    with pytest.raises(ValidationError):
        AscendGraphOptions(rendezvous_port=port)


def test_real_tcpstore_reservation_retains_ephemeral_listener_and_releases():
    import socket
    import time
    from datetime import timedelta

    from torch.distributed import TCPStore

    from expertkit_transport.transports.ascend_graph.rendezvous import (
        GraphRendezvousRequest,
        OwnedGraphRendezvous,
    )

    facts = config_for(rendezvous_port=12346).model_dump(mode="json")
    facts["rendezvous_port"] = 0
    request = GraphRendezvousRequest.from_facts(facts)
    first = OwnedGraphRendezvous(request, monotonic_deadline=time.monotonic() + 2)
    second_request = GraphRendezvousRequest.from_facts(
        {**facts, "session_id": "another-cpu-session-0002"}
    )
    second = OwnedGraphRendezvous(second_request, monotonic_deadline=time.monotonic() + 2)
    client = None
    try:
        assert 0 < first.port <= 65535 and first.port != second.port
        client = TCPStore("127.0.0.1", first.port, 2, False, timedelta(seconds=1))
        first.store.set("owned", "live")
        assert client.get("owned") == b"live"
        final = AscendGraphConfig.model_validate({**facts, "rendezvous_port": first.port})
        first.validate_config(final)
        with pytest.raises(ValueError, match="reservation"):
            first.validate_config(
                final.model_copy(update={"session_id": second_request.session_id})
            )
        with socket.socket() as probe, pytest.raises(OSError):
            probe.bind(("127.0.0.1", first.port))
        conflict = GraphRendezvousRequest.from_facts({**facts, "rendezvous_port": first.port})
        with pytest.raises(RuntimeError):
            OwnedGraphRendezvous(conflict, monotonic_deadline=time.monotonic() + 2)
        second.close()
        assert client.get("owned") == b"live"
        client = None
        first.close()
        with socket.socket() as released:
            released.bind(("127.0.0.1", first.port))
    finally:
        client = None
        first.close()
        second.close()


def test_rank1_hccl_reuses_retained_store_without_rebinding(monkeypatch):
    import sys
    import time
    from types import SimpleNamespace

    from expertkit_transport.transports.ascend_graph.rendezvous import (
        GraphRendezvousRequest,
        OwnedGraphRendezvous,
    )
    from expertkit_transport.transports.ascend_graph.session import HcclGraphSession

    facts = config_for().model_dump(mode="json")
    facts["rendezvous_port"] = 0
    reservation = OwnedGraphRendezvous(
        GraphRendezvousRequest.from_facts(facts),
        monotonic_deadline=time.monotonic() + 2,
        bind_timeout_seconds=0.1,
    )
    assert reservation.store.timeout.total_seconds() == 0.1
    config = AscendGraphConfig.model_validate({**facts, "rendezvous_port": reservation.port})

    class ProcessGroup:
        Options = SimpleNamespace

        def __init__(self, store, rank, world_size, options):
            store.set("group", "constructed")

    monkeypatch.setitem(
        sys.modules,
        "torch_npu._C._distributed_c10d",
        SimpleNamespace(ProcessGroupHCCL=ProcessGroup),
    )
    session = None
    try:
        session = HcclGraphSession(
            config,
            rank=1,
            device=SimpleNamespace(type="npu"),
            hidden_dim=config.model.hidden_dim,
            top_k=config.model.top_k,
            rendezvous_store=reservation.store,
        )
        assert session._store is reservation.store
        assert reservation.store.timeout.total_seconds() == config.timeout_seconds
        assert reservation.store.get(f"{config.session_id}/group") == b"constructed"
    finally:
        if session is not None:
            session.close()
        reservation.close()
