#!/usr/bin/env python3
"""Microbenchmark one EK computation backend with deterministic synthetic data."""

from __future__ import annotations

import json
import math
import statistics
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import torch
import torch.nn.functional as functional
import typer
from expertkit_transport.quarantine import retain_until_process_exit
from safetensors.torch import save

from expertkit_worker.backends import BackendBatch, ComputeBackend
from expertkit_worker.backends.factory import (
    create_compute_backend,
    create_weight_adapter,
    graph_resident_type,
    torch_dtype,
)
from expertkit_worker.config import ActivationDType, BackendName, WorkerConfig
from expertkit_worker.device import CpuWorkerRuntime, CudaWorkerRuntime, WorkerDeviceRuntime
from expertkit_worker.weights import ReadyWeightTable, WeightAdapter, parse_safetensors


class Distribution(StrEnum):
    BALANCED = "balanced"
    UNIFORM = "uniform"
    HOT = "hot"


class BenchmarkQuantization(StrEnum):
    NONE = "none"
    MODELSLIM_W8A8_DYNAMIC = "modelslim-w8a8-dynamic"


class ExecutionMode(StrEnum):
    EAGER = "eager"
    GRAPH = "graph"


class ExpertCompute(StrEnum):
    SWIGLU = "swiglu"
    DEEPSEEK_V4 = "deepseek_v4"


@dataclass(frozen=True, slots=True, kw_only=True)
class BenchmarkOptions:
    backend: BackendName = BackendName.ASCEND_FUSED
    device: str = "npu:0"
    execution_mode: ExecutionMode = ExecutionMode.EAGER
    expert_start: int = 0
    expert_end: int | None = None
    dtype: ActivationDType = ActivationDType.BF16
    quantization: BenchmarkQuantization = BenchmarkQuantization.NONE
    expert_compute: ExpertCompute = ExpertCompute.SWIGLU
    swiglu_limit: float = 0.0
    tokens: int = 32
    hidden_dim: int = 1024
    intermediate_dim: int = 512
    top_k: int = 2
    experts: int = 8
    distribution: Distribution = Distribution.BALANCED
    hot_experts: int | None = None
    hot_multiplier: float = 10.0
    inactive_rate: float = 0.0
    concurrency: int = 1
    routing_cases: int = 1
    routing_replay: Path | None = None
    warmup: int = 5
    repeats: int = 20
    seed: int = 1234
    atol: float | None = None
    rtol: float | None = None
    max_relative_l2: float = 0.06
    min_cosine: float = 0.995
    skip_correctness: bool = False
    profile: bool = False
    profile_dir: Path = Path("benchmark-traces")
    output: Path | None = None

    def __post_init__(self) -> None:
        for name in (
            "tokens",
            "hidden_dim",
            "intermediate_dim",
            "top_k",
            "experts",
            "concurrency",
            "routing_cases",
            "repeats",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"--{name.replace('_', '-')} must be positive")
        if self.warmup < 0:
            raise ValueError("--warmup must be nonnegative")
        if self.top_k > self.experts:
            raise ValueError("--top-k cannot exceed --experts")
        if not 0 <= self.inactive_rate <= 1:
            raise ValueError("--inactive-rate must be in [0, 1]")
        if self.hot_experts is not None and not 1 <= self.hot_experts <= self.experts:
            raise ValueError("--hot-experts must be in [1, --experts]")
        if not math.isfinite(self.hot_multiplier) or self.hot_multiplier < 1:
            raise ValueError("--hot-multiplier must be finite and at least 1")
        if not math.isfinite(self.swiglu_limit) or self.swiglu_limit < 0:
            raise ValueError("--swiglu-limit must be finite and nonnegative")
        for name in ("atol", "rtol", "max_relative_l2"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"--{name.replace('_', '-')} must be finite and nonnegative")
        if not -1 <= self.min_cosine <= 1:
            raise ValueError("--min-cosine must be in [-1, 1]")
        if self.expert_compute == ExpertCompute.SWIGLU and self.swiglu_limit != 0:
            raise ValueError("--swiglu-limit requires --expert-compute deepseek_v4")
        end = self.experts if self.expert_end is None else self.expert_end
        if not 0 <= self.expert_start < end <= self.experts:
            raise ValueError("resident expert interval must be within [0, --experts)")
        if self.execution_mode == ExecutionMode.EAGER:
            if self.expert_start != 0 or end != self.experts:
                raise ValueError("--expert-start/--expert-end require --execution-mode graph")
        else:
            if self.backend not in {BackendName.TORCH, BackendName.ASCEND_FUSED}:
                raise ValueError("graph replay supports only torch and ascend_fused")
            if not self.device.startswith("npu:") or self.dtype != ActivationDType.BF16:
                raise ValueError("graph replay requires an indexed NPU and BF16")
            if self.concurrency != 1:
                raise ValueError("graph replay requires --concurrency 1")
            if self.inactive_rate:
                raise ValueError(
                    "graph replay requires valid global IDs; use zero routing weights for padding"
                )
            maximum = 256 if self.expert_compute == ExpertCompute.DEEPSEEK_V4 else 128
            if self.experts > maximum or end - self.expert_start > 128:
                raise ValueError("graph expert counts exceed the resident GMM limits")
            if self.expert_compute == ExpertCompute.DEEPSEEK_V4 and (
                self.quantization != BenchmarkQuantization.MODELSLIM_W8A8_DYNAMIC
                or self.swiglu_limit != 10
                or end - self.expert_start < 2
            ):
                raise ValueError(
                    "V4 graphs require ModelSlim, --swiglu-limit 10 and at least two local experts"
                )


@dataclass(slots=True)
class _Setup:
    config: WorkerConfig
    runtime: WorkerDeviceRuntime
    adapter: WeightAdapter[Any, Any]
    table: ReadyWeightTable[Any]
    backend: ComputeBackend
    batches: list[BackendBatch]
    outputs: list[torch.Tensor]
    source_values: dict[int, dict[str, torch.Tensor]]
    source_buffers: list[bytearray]
    cpu_weights: list[Any]
    concurrency: int
    routing_cases: int
    routing_source: str

    def close(self) -> None:
        if self.table.quarantined:
            retain_until_process_exit(self)
            return
        for expert_id in self.source_values:
            self.table.begin_withdrawal(0, expert_id)
            ready = self.table.finish_withdrawal(0, expert_id)
            self.adapter.release_ready_weight(ready)


class _SubmissionGroup:
    def __init__(self, setup: _Setup) -> None:
        self._setup = setup
        concurrency = setup.concurrency
        self._case_index = 0
        self.last_batches = []
        self.last_outputs = []
        self._pool = (
            ThreadPoolExecutor(max_workers=concurrency)
            if concurrency > 1 and setup.runtime.device.type == "cpu"
            else None
        )
        if concurrency > 1 and self._pool is None:
            create_stream = getattr(setup.runtime, "create_stream", None)
            if create_stream is None:
                raise ValueError("concurrency > 1 requires an asynchronous device runtime")
            self._streams = [create_stream() for _ in range(concurrency)]
        else:
            self._streams = []

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()

    @torch.inference_mode()
    def run(self) -> None:
        start = self._case_index * self._setup.concurrency
        stop = start + self._setup.concurrency
        batches = self._setup.batches[start:stop]
        outputs = self._setup.outputs[start:stop]
        self.last_batches, self.last_outputs = batches, outputs
        self._case_index = (self._case_index + 1) % self._setup.routing_cases

        if self._pool is not None:
            futures = [
                self._pool.submit(self._submit_and_wait, batch, output)
                for batch, output in zip(
                    batches,
                    outputs,
                    strict=True,
                )
            ]
            for future in futures:
                future.result()
            return

        if not self._streams:
            self._submit_and_wait(batches[0], outputs[0])
            return

        completions = []
        try:
            for stream, batch, output in zip(
                self._streams,
                batches,
                outputs,
                strict=True,
            ):
                stream_context = self._setup.runtime.stream_context  # type: ignore[attr-defined]
                with stream_context(stream):
                    completions.append(self._setup.backend.submit(batch, output))
            for completion in completions:
                completion.wait_host()
        finally:
            for completion in completions:
                completion.close()

    def reset_routing_case(self) -> None:
        """Restart deterministic case selection without changing backend state."""
        self._case_index = 0

    def _submit_and_wait(self, batch: BackendBatch, output: torch.Tensor) -> None:
        completion = self._setup.backend.submit(batch, output)
        try:
            completion.wait_host()
        finally:
            completion.close()


class _GraphReplayGroup:
    """Capture resident computation once; stage changing cases into fixed buffers.

    Device events surround replay only. Host latency includes input copies, event
    recording, replay dispatch and completion wait. No transport is involved.
    """

    def __init__(self, setup: _Setup, args: BenchmarkOptions, *, npu=None) -> None:
        self._setup = setup
        self._runtime = setup.runtime
        # Input generation may still be queued on the caller's stream.
        try:
            self._runtime.capture_current_work().wait_host()
        except BaseException:
            setup.table.quarantine()
            retain_until_process_exit(setup)
            raise
        self._stream = self._runtime.create_stream()
        self._resident = None
        self._graph = None
        self._case_index = 0
        self.last_batches = []
        self.last_outputs = []
        self.device_samples_ms = []
        npu = torch.npu if npu is None else npu
        try:
            with (
                self._runtime.device_context(),
                self._runtime.stream_context(self._stream),
                torch.inference_mode(),
            ):
                self._resident = graph_resident_type(setup.config)(
                    layer_ids=(0,),
                    experts_per_layer=args.experts,
                    hidden_dim=args.hidden_dim,
                    intermediate_dim=args.intermediate_dim,
                    top_k=args.top_k,
                    acquire_many=setup.table.acquire_many,
                    ready_revision=lambda: setup.table.revision,
                    expert_range=(
                        args.expert_start,
                        args.experts if args.expert_end is None else args.expert_end,
                    ),
                    expert_compute=args.expert_compute,
                    swiglu_limit=args.swiglu_limit,
                )
                first = setup.batches[0]
                self._hidden = first.hidden_states.clone()
                self._ids = first.expert_ids.clone()
                self._routing = first.routing_weights.clone()
                self._output = setup.outputs[0]
                # Compile and warm device kernels before capture, independently
                # of the user's post-capture replay warmup count.
                for _ in range(3):
                    self._body()
                self._runtime.synchronize_stream(self._stream)
                self._graph = npu.NPUGraph()
                with npu.graph(self._graph, stream=self._stream):
                    self._body()
                self._runtime.synchronize_stream(self._stream)
                self._start = self._runtime.create_event(enable_timing=True)
                self._end = self._runtime.create_event(enable_timing=True)
        except BaseException:
            self.close()
            raise

    def _body(self) -> None:
        self._resident.compute(0, self._hidden, self._ids, self._routing, self._output)

    @torch.inference_mode()
    def run(self) -> None:
        self._resident.require_current()
        batch = self._setup.batches[self._case_index]
        self._case_index = (self._case_index + 1) % self._setup.routing_cases
        self.last_batches = [batch]
        self.last_outputs = [self._output]
        with self._runtime.device_context(), self._runtime.stream_context(self._stream):
            self._hidden.copy_(batch.hidden_states)
            self._ids.copy_(batch.expert_ids)
            self._routing.copy_(batch.routing_weights)
            self._runtime.record_event(self._start, self._stream)
            self._graph.replay()
            self._runtime.record_event(self._end, self._stream)
            self._runtime.synchronize_stream(self._stream)
            self.device_samples_ms.append(self._runtime.elapsed_time_ms(self._start, self._end))

    def reset_routing_case(self) -> None:
        self._case_index = 0

    def close(self) -> None:
        def completed():
            self._runtime.synchronize_stream(self._stream)
            self._graph = None

        try:
            if self._resident is not None:
                self._resident.close(wait_for_completion=completed)
            else:
                completed()
        except BaseException:
            self._setup.table.quarantine()
            retain_until_process_exit(self)
            raise


def _make_config(args: BenchmarkOptions) -> WorkerConfig:
    model: dict[str, Any] = {
        "instance_id": 1,
        "name": "benchmark/model",
        "weight_version": f"synthetic-seed-{args.seed}",
        "num_layers": 1,
        "experts_per_layer": args.experts,
        "hidden_dim": args.hidden_dim,
        "expert_intermediate_dim": args.intermediate_dim,
        "top_k": args.top_k,
        "activation_dtype": args.dtype,
        "weight_dtype": args.dtype,
        "expert_compute": args.expert_compute,
        "swiglu_limit": args.swiglu_limit,
    }
    if args.quantization != "none":
        model["quantization"] = {
            "type": args.quantization,
            "bits": 8,
            "group_size": None,
        }
    worker: dict[str, Any] = {
        "id": "backend-benchmark",
        "backend": args.backend,
        "device": args.device,
        "max_batch_tokens": args.tokens,
        "max_active_batches_per_device": args.concurrency,
        "device_memory_limit": "64GiB",
    }
    if args.backend == "ggml":
        worker["ggml"] = {"cpu_threads": max(1, torch.get_num_threads())}
    return WorkerConfig.model_validate(
        {
            "model": model,
            "worker": worker,
            "transport": {
                "type": "grpc",
                "max_pending_batches_per_device": args.concurrency,
                "listen": "127.0.0.1:50051",
                "advertise": "benchmark:50051",
            },
            "controller": {"endpoint": "127.0.0.1:50050"},
            "weight_manager": {
                "max_concurrent_loads": 1,
                "disk_cache": {"path": "/tmp/ek-backend-benchmark-cache"},
                "peer": {
                    "listen": "127.0.0.1:50052",
                    "advertise": "http://benchmark:50052",
                },
                "weight_server_endpoint": "http://127.0.0.1:50053",
            },
        }
    )


def _make_runtime(device_name: str) -> WorkerDeviceRuntime:
    if device_name == "cpu":
        return CpuWorkerRuntime(torch.device("cpu"))
    if device_name.startswith("cuda:"):
        runtime = CudaWorkerRuntime(torch.device(device_name))
        runtime.set_current_device()
        return runtime
    if device_name.startswith("npu:"):
        from expertkit_worker.device.ascend import AscendWorkerRuntime

        runtime = AscendWorkerRuntime(torch.device(device_name))
        runtime.set_current_device()
        return runtime
    raise ValueError(f"unsupported benchmark device: {device_name}")


def _make_source_values(
    *,
    expert_id: int,
    hidden_dim: int,
    intermediate_dim: int,
    dtype: torch.dtype,
    quantized: bool,
    seed: int,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed + expert_id * 17)
    prefix = f"layers.0.ffn.experts.{expert_id}"
    shapes = (
        ("w1", (intermediate_dim, hidden_dim)),
        ("w3", (intermediate_dim, hidden_dim)),
        ("w2", (hidden_dim, intermediate_dim)),
    )
    values: dict[str, torch.Tensor] = {}
    for name, shape in shapes:
        if quantized:
            values[f"{prefix}.{name}.weight"] = torch.randint(
                -16,
                17,
                shape,
                generator=generator,
                dtype=torch.int8,
            )
            scale = torch.rand((shape[0], 1), generator=generator, dtype=torch.float32)
            # The graph provider requires lossless BF16 scales. Use the same
            # synthetic source recipe in eager and graph modes for matched runs.
            values[f"{prefix}.{name}.weight_scale"] = (
                scale.mul_(0.015).add_(0.005).bfloat16().float()
            )
            values[f"{prefix}.{name}.weight_offset"] = torch.zeros(
                (shape[0], 1),
                dtype=torch.float32,
            )
        else:
            scale = 0.03 / math.sqrt(shape[1])
            values[f"{prefix}.{name}.weight"] = (
                torch.randn(shape, generator=generator, dtype=torch.float32) * scale
            ).to(dtype)
    return values


def _balanced_ids(args: BenchmarkOptions, generator: torch.Generator) -> torch.Tensor:
    """Distribute assignments round-robin; K <= G keeps each token's IDs distinct."""
    return (
        torch.arange(args.tokens * args.top_k).reshape(args.tokens, args.top_k) % args.experts
    ).to(torch.int32)


def _uniform_ids(args: BenchmarkOptions, generator: torch.Generator) -> torch.Tensor:
    """Sample distinct experts independently for each token."""
    probabilities = torch.ones((args.tokens, args.experts), dtype=torch.float32)
    return torch.multinomial(probabilities, args.top_k, replacement=False, generator=generator).to(
        torch.int32
    )


def _hot_ids(args: BenchmarkOptions, generator: torch.Generator) -> torch.Tensor:
    """Favor a seeded hot set while preserving distinct top-k IDs within each token."""
    hot_count = args.top_k if args.hot_experts is None else args.hot_experts
    hot_ids = torch.randperm(args.experts, generator=generator)[:hot_count]
    probabilities = torch.ones(args.experts, dtype=torch.float32)
    probabilities[hot_ids] *= args.hot_multiplier
    return torch.multinomial(
        probabilities.expand(args.tokens, -1),
        args.top_k,
        replacement=False,
        generator=generator,
    ).to(torch.int32)


type GenerateIds = Callable[[BenchmarkOptions, torch.Generator], torch.Tensor]

ID_GENERATORS: dict[Distribution, GenerateIds] = {
    Distribution.BALANCED: _balanced_ids,
    Distribution.UNIFORM: _uniform_ids,
    Distribution.HOT: _hot_ids,
}


@dataclass(frozen=True, slots=True)
class _RoutingReplayCase:
    expert_ids: torch.Tensor
    lease_expert_ids: tuple[int, ...]


def _require_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must contain integers")
    return value


def _load_routing_replay(args: BenchmarkOptions) -> list[_RoutingReplayCase] | None:
    path = args.routing_replay
    if path is None:
        return None
    if args.inactive_rate:
        raise ValueError("--inactive-rate cannot be combined with --routing-replay")
    try:
        document = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read --routing-replay: {error}") from error
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("--routing-replay must use schema_version 1")
    raw_cases = document.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("--routing-replay cases must be a non-empty list")
    if len(raw_cases) % args.concurrency:
        raise ValueError("--routing-replay case count must be divisible by --concurrency")

    cases: list[_RoutingReplayCase] = []
    expected_shape = (args.tokens, args.top_k)
    for case_index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict):
            raise ValueError(f"routing replay case {case_index} must be an object")
        raw_ids = raw_case.get("expert_ids")
        if not isinstance(raw_ids, list) or len(raw_ids) != args.tokens:
            raise ValueError(
                f"routing replay case {case_index} expert_ids must have shape {expected_shape}"
            )
        rows: list[list[int]] = []
        for row in raw_ids:
            if not isinstance(row, list) or len(row) != args.top_k:
                raise ValueError(
                    f"routing replay case {case_index} expert_ids must have shape {expected_shape}"
                )
            values = [
                _require_int(value, field=f"routing replay case {case_index} expert_ids")
                for value in row
            ]
            if args.execution_mode == ExecutionMode.GRAPH and any(value < 0 for value in values):
                raise ValueError(
                    "graph routing replay requires valid global IDs; -1 is unsupported"
                )
            if any(value < -1 or value >= args.experts for value in values):
                raise ValueError(
                    f"routing replay case {case_index} expert IDs must be -1 or in [0, experts)"
                )
            active = [value for value in values if value >= 0]
            if len(active) != len(set(active)):
                raise ValueError(
                    f"routing replay case {case_index} has duplicate active IDs in one token"
                )
            rows.append(values)

        raw_lease = raw_case.get("lease_expert_ids")
        if not isinstance(raw_lease, list):
            raise ValueError(f"routing replay case {case_index} lease_expert_ids must be a list")
        lease = tuple(
            _require_int(
                value,
                field=f"routing replay case {case_index} lease_expert_ids",
            )
            for value in raw_lease
        )
        if not lease:
            raise ValueError(f"routing replay case {case_index} lease_expert_ids must not be empty")
        expected_lease = tuple(sorted({value for row in rows for value in row if value >= 0}))
        if lease != expected_lease:
            raise ValueError(
                f"routing replay case {case_index} lease_expert_ids must equal the sorted "
                "distinct active routing IDs"
            )
        cases.append(
            _RoutingReplayCase(
                expert_ids=torch.tensor(rows, dtype=torch.int32),
                lease_expert_ids=lease,
            )
        )
    return cases


def _make_batch(
    args: BenchmarkOptions,
    *,
    dtype: torch.dtype,
    device: torch.device,
    slot: int,
    routing_case: int = 0,
    replay: _RoutingReplayCase | None = None,
) -> BackendBatch:
    batch_index = routing_case * args.concurrency + slot
    generator = torch.Generator().manual_seed(args.seed + 10_000 + batch_index)
    hidden = torch.randn(
        (args.tokens, args.hidden_dim),
        generator=generator,
        dtype=torch.float32,
    ).mul_(0.1)
    ids = (
        ID_GENERATORS[args.distribution](args, generator)
        if replay is None
        else replay.expert_ids.clone()
    )
    routing = torch.rand(ids.shape, generator=generator, dtype=torch.float32)
    routing.div_(routing.sum(dim=1, keepdim=True))
    if replay is None and args.inactive_rate:
        mask = torch.rand(ids.shape, generator=generator) < args.inactive_rate
        ids.masked_fill_(mask, -1)
    routing.masked_fill_(ids.eq(-1), 0)
    distinct = (
        tuple(torch.unique(ids[ids.ge(0)], sorted=True).tolist())
        if replay is None
        else replay.lease_expert_ids
    )
    return BackendBatch(
        layer_id=0,
        hidden_states=hidden.to(device=device, dtype=dtype),
        expert_ids=ids.to(device=device),
        routing_weights=routing.to(device=device),
        distinct_expert_ids=distinct,
    )


def _build_setup(args: BenchmarkOptions) -> _Setup:
    replay_cases = _load_routing_replay(args)
    config = _make_config(args)
    runtime = _make_runtime(args.device)
    dtype = torch_dtype(config.model.activation_dtype)
    adapter = create_weight_adapter(
        config,
        source_dtype=torch_dtype(config.model.weight_dtype),
        compute_dtype=dtype,
        runtime=runtime,
    )
    adapter.initialize_ready_storage(args.experts)
    table: ReadyWeightTable[Any] = ReadyWeightTable(1, args.experts)
    source_values: dict[int, dict[str, torch.Tensor]] = {}
    source_buffers: list[bytearray] = []
    cpu_weights: list[Any] = []
    quantized = args.quantization != "none"
    with runtime.device_context():
        resident_ids = range(args.experts)
        if args.execution_mode == ExecutionMode.GRAPH:
            resident_ids = range(
                args.expert_start, args.experts if args.expert_end is None else args.expert_end
            )
        for expert_id in resident_ids:
            values = _make_source_values(
                expert_id=expert_id,
                hidden_dim=args.hidden_dim,
                intermediate_dim=args.intermediate_dim,
                dtype=dtype,
                quantized=quantized,
                seed=args.seed,
            )
            buffer = bytearray(save(values))
            cpu_weight = adapter.make_cpu_weight(parse_safetensors(buffer))
            ready = adapter.make_ready_weight(cpu_weight, layer_id=0, expert_id=expert_id)
            table.publish(0, expert_id, ready)
            source_values[expert_id] = values
            source_buffers.append(buffer)
            cpu_weights.append(cpu_weight)
        backend = create_compute_backend(
            config,
            dtype=dtype,
            runtime=runtime,
            acquire_many=table.acquire_many,
        )
        if replay_cases is None:
            routing_cases = args.routing_cases
            batches = [
                _make_batch(
                    args,
                    dtype=dtype,
                    device=runtime.device,
                    slot=slot,
                    routing_case=routing_case,
                )
                for routing_case in range(routing_cases)
                for slot in range(args.concurrency)
            ]
            routing_source = "synthetic"
        else:
            routing_cases = len(replay_cases) // args.concurrency
            batches = [
                _make_batch(
                    args,
                    dtype=dtype,
                    device=runtime.device,
                    slot=index % args.concurrency,
                    routing_case=index // args.concurrency,
                    replay=replay,
                )
                for index, replay in enumerate(replay_cases)
            ]
            routing_source = "replay"
        outputs = [torch.empty_like(batch.hidden_states) for batch in batches]
    return _Setup(
        config=config,
        runtime=runtime,
        adapter=adapter,
        table=table,
        backend=backend,
        batches=batches,
        outputs=outputs,
        source_values=source_values,
        source_buffers=source_buffers,
        cpu_weights=cpu_weights,
        concurrency=args.concurrency,
        routing_cases=routing_cases,
        routing_source=routing_source,
    )


def _projection(
    source: dict[str, torch.Tensor],
    expert_id: int,
    name: str,
    *,
    quantized: bool,
) -> torch.Tensor:
    prefix = f"layers.0.ffn.experts.{expert_id}.{name}"
    weight = source[f"{prefix}.weight"].float()
    if quantized:
        weight = weight * source[f"{prefix}.weight_scale"].float()
    return weight


def _reference_output(setup: _Setup, batch: BackendBatch) -> torch.Tensor:
    hidden = batch.hidden_states.float().cpu()
    expert_ids = batch.expert_ids.cpu()
    routing = batch.routing_weights.cpu()
    output = torch.zeros_like(hidden)
    quantized = setup.config.model.quantization is not None
    for expert_id in batch.distinct_expert_ids:
        if expert_id not in setup.source_values:
            continue  # The graph benchmark measures this resident shard's contribution.
        coordinates = torch.nonzero(expert_ids == expert_id, as_tuple=False)
        token_indices = coordinates[:, 0]
        route_indices = coordinates[:, 1]
        x = hidden.index_select(0, token_indices)
        source = setup.source_values[expert_id]
        gate = functional.linear(
            x,
            _projection(source, expert_id, "w1", quantized=quantized),
        )
        up = functional.linear(
            x,
            _projection(source, expert_id, "w3", quantized=quantized),
        )
        if setup.config.model.expert_compute == "deepseek_v4":
            limit = setup.config.model.swiglu_limit
            if limit > 0:
                gate.clamp_(max=limit)
                up.clamp_(min=-limit, max=limit)
        activated = functional.silu(gate) * up
        route = routing[token_indices, route_indices].unsqueeze(1)
        expert_output = functional.linear(
            activated,
            _projection(source, expert_id, "w2", quantized=quantized),
        )
        expert_output.mul_(route)
        output.index_add_(0, token_indices, expert_output)
    return output


def _check_correctness(
    args: BenchmarkOptions,
    setup: _Setup,
    runner: _SubmissionGroup | _GraphReplayGroup,
) -> dict[str, Any]:
    quantized = setup.config.model.quantization is not None
    default_tolerance = 0.08 if quantized else (0.02 if args.dtype != "fp32" else 1e-5)
    atol = default_tolerance if args.atol is None else args.atol
    rtol = default_tolerance if args.rtol is None else args.rtol
    maximum = 0.0
    mean = 0.0
    maximum_relative_l2 = 0.0
    minimum_cosine = 1.0
    checked_batches = 0
    runner.reset_routing_case()
    for case_index in range(setup.routing_cases):
        runner.run()
        for slot, (batch, actual) in enumerate(
            zip(runner.last_batches, runner.last_outputs, strict=True)
        ):
            checked_batches += 1
            batch_index = case_index * setup.concurrency + slot
            expected = _reference_output(setup, batch)
            actual_cpu = actual.float().cpu()
            difference = (actual_cpu - expected).abs()
            max_abs_error = difference.max().item()
            relative_l2 = float(difference.norm() / expected.norm().clamp_min(1e-10))
            if torch.count_nonzero(expected) == 0:
                cosine = 1.0 if torch.count_nonzero(actual_cpu) == 0 else 0.0
            else:
                cosine = float(
                    functional.cosine_similarity(
                        actual_cpu.flatten(),
                        expected.flatten(),
                        dim=0,
                    )
                )
            maximum = max(maximum, max_abs_error)
            mean += difference.mean().item()
            maximum_relative_l2 = max(maximum_relative_l2, relative_l2)
            minimum_cosine = min(minimum_cosine, cosine)
            if quantized:
                valid = (
                    torch.isfinite(actual_cpu).all()
                    and relative_l2 < args.max_relative_l2
                    and cosine > args.min_cosine
                )
            else:
                valid = torch.allclose(actual_cpu, expected, atol=atol, rtol=rtol)
            if not valid:
                raise RuntimeError(
                    "backend correctness check failed: "
                    f"batch={batch_index}, max_abs_error={max_abs_error:.6g}, "
                    f"relative_l2={relative_l2:.6g}, cosine={cosine:.6g}, "
                    f"atol={atol}, rtol={rtol}, "
                    f"max_relative_l2={args.max_relative_l2}, min_cosine={args.min_cosine}"
                )
    return {
        "checked": True,
        "reference": "independent_fp32_dequantized" if quantized else "independent_fp32",
        "criterion": "relative_l2_and_cosine" if quantized else "allclose",
        "atol": atol,
        "rtol": rtol,
        "max_abs_error": maximum,
        "mean_abs_error": mean / checked_batches,
        "checked_batches": checked_batches,
        "max_relative_l2": maximum_relative_l2,
        "max_relative_l2_limit": args.max_relative_l2,
        "min_cosine_similarity": minimum_cosine,
        "min_cosine_similarity_limit": args.min_cosine,
    }


def _profile_context(args: BenchmarkOptions, device: torch.device):
    if not args.profile:
        return nullcontext(None)
    args.profile_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "npu":
        from torch_npu import profiler

        return profiler.profile(
            activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
            on_trace_ready=profiler.tensorboard_trace_handler(str(args.profile_dir)),
            record_shapes=True,
        )
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return torch.profiler.profile(
        activities=activities,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(args.profile_dir)),
        record_shapes=True,
    )


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "npu":
        import torch_npu

        torch_npu.npu.reset_peak_memory_stats(device)


def _peak_memory(device: torch.device) -> int | None:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device)
    if device.type == "npu":
        import torch_npu

        return torch_npu.npu.max_memory_allocated(device)
    return None


def _memory_info(runtime: WorkerDeviceRuntime) -> tuple[int, int] | None:
    try:
        return runtime.memory_info()
    except RuntimeError:
        return None


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "min": min(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": max(values),
    }


def _run_benchmark(args: BenchmarkOptions) -> dict[str, Any]:
    setup = _build_setup(args)
    runner = None
    try:
        runner = (
            _GraphReplayGroup(setup, args)
            if args.execution_mode == ExecutionMode.GRAPH
            else _SubmissionGroup(setup)
        )
        correctness = (
            {"checked": False} if args.skip_correctness else _check_correctness(args, setup, runner)
        )
        runner.reset_routing_case()
        for _ in range(args.warmup):
            runner.run()

        if isinstance(runner, _GraphReplayGroup):
            runner.device_samples_ms.clear()
        memory_before = _memory_info(setup.runtime)
        _reset_peak_memory(setup.runtime.device)
        durations_ms: list[float] = []
        with _profile_context(args, setup.runtime.device) as active_profiler:
            for _ in range(args.repeats):
                started = time.perf_counter_ns()
                with (
                    torch.profiler.record_function("microbench.iteration")
                    if active_profiler is not None
                    else nullcontext()
                ):
                    runner.run()
                durations_ms.append((time.perf_counter_ns() - started) / 1_000_000)
                if active_profiler is not None:
                    active_profiler.step()
        memory_after = _memory_info(setup.runtime)
        peak_allocated = _peak_memory(setup.runtime.device)
        mean_ms = statistics.fmean(durations_ms)
        total_tokens = args.tokens * args.concurrency
        estimate = (
            setup.backend.estimate_resources(args.tokens)
            if args.execution_mode == ExecutionMode.EAGER
            else None
        )
        return {
            "schema_version": 1,
            "backend": args.backend,
            "device": args.device,
            "dtype": args.dtype,
            "quantization": args.quantization,
            "expert_compute": args.expert_compute,
            "swiglu_limit": args.swiglu_limit,
            "synthetic_scale_policy": "bf16_exact_fp32"
            if args.quantization != BenchmarkQuantization.NONE
            else None,
            "shape": {
                "tokens": args.tokens,
                "hidden_dim": args.hidden_dim,
                "intermediate_dim": args.intermediate_dim,
                "top_k": args.top_k,
                "experts": args.experts,
            },
            "routing": {
                "source": setup.routing_source,
                "distribution": args.distribution,
                "hot_experts": args.top_k if args.hot_experts is None else args.hot_experts,
                "hot_multiplier": args.hot_multiplier,
                "inactive_rate": args.inactive_rate,
                "cases": setup.routing_cases,
                "unique_lease_keys": len({batch.distinct_expert_ids for batch in setup.batches}),
                "seed": args.seed,
            },
            "execution": {
                "mode": args.execution_mode,
                "operator_path": (
                    "ascend_fused.run_ascend_fused_moe"
                    if args.backend == BackendName.ASCEND_FUSED
                    else "torch.resident"
                    if args.execution_mode == ExecutionMode.GRAPH
                    else "backend.submit"
                ),
                "captured_graphs": 1 if args.execution_mode == ExecutionMode.GRAPH else 0,
                "scope": "resident_compute_only"
                if args.execution_mode == ExecutionMode.GRAPH
                else "eager_backend_lifecycle",
                "host_timing_includes": "input_staging,replay,device_events,completion_wait"
                if args.execution_mode == ExecutionMode.GRAPH
                else "submit,wait_host,close",
                "resident_expert_range": [
                    args.expert_start,
                    args.experts if args.expert_end is None else args.expert_end,
                ],
                "concurrency": args.concurrency,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "profile": args.profile,
            },
            "correctness": correctness,
            "latency_ms": {
                "mean": mean_ms,
                "min": min(durations_ms),
                "p50": _percentile(durations_ms, 0.50),
                "p95": _percentile(durations_ms, 0.95),
                "max": max(durations_ms),
            },
            "device_replay_latency_ms": (
                _latency_summary(runner.device_samples_ms)
                if isinstance(runner, _GraphReplayGroup)
                else None
            ),
            "throughput": {
                "tokens_per_second": total_tokens * 1000 / mean_ms,
                "batches_per_second": args.concurrency * 1000 / mean_ms,
            },
            "memory": {
                "free_bytes_before": None if memory_before is None else memory_before[0],
                "free_bytes_after": None if memory_after is None else memory_after[0],
                "total_bytes": None if memory_after is None else memory_after[1],
                "peak_allocated_bytes": peak_allocated,
                "estimated_temporary_bytes_per_active_batch": (
                    None if estimate is None else estimate.temporary_bytes_per_active_batch
                ),
                "estimated_total_temporary_bytes": None
                if estimate is None
                else estimate.total_bytes(args.concurrency),
            },
        }
    finally:
        try:
            if runner is not None:
                runner.close()
        finally:
            setup.close()


app = typer.Typer(help=__doc__, pretty_exceptions_show_locals=False, add_completion=False)


@app.command()
def main(
    backend: BackendName = BackendName.ASCEND_FUSED,
    device: str = "npu:0",
    execution_mode: ExecutionMode = ExecutionMode.EAGER,
    expert_start: int = 0,
    expert_end: int | None = None,
    dtype: ActivationDType = ActivationDType.BF16,
    quantization: BenchmarkQuantization = BenchmarkQuantization.NONE,
    expert_compute: ExpertCompute = ExpertCompute.SWIGLU,
    swiglu_limit: float = 0.0,
    tokens: int = 32,
    hidden_dim: int = 1024,
    intermediate_dim: int = 512,
    top_k: int = 2,
    experts: int = 8,
    distribution: Distribution = Distribution.BALANCED,
    hot_experts: Annotated[
        int | None, typer.Option(help="Hot-set size; defaults to top-k.")
    ] = None,
    hot_multiplier: Annotated[
        float, typer.Option(help="Hot-expert sampling weight multiplier.")
    ] = 10.0,
    inactive_rate: float = 0.0,
    concurrency: int = 1,
    routing_cases: Annotated[
        int,
        typer.Option(help="Pre-generated routing cases cycled across iterations."),
    ] = 1,
    routing_replay: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Schema-v1 routing replay JSON; cases are grouped consecutively by "
                "concurrency and must match --tokens/--top-k."
            )
        ),
    ] = None,
    warmup: int = 5,
    repeats: int = 20,
    seed: int = 1234,
    atol: float | None = None,
    rtol: float | None = None,
    max_relative_l2: float = 0.06,
    min_cosine: float = 0.995,
    skip_correctness: Annotated[bool, typer.Option("--skip-correctness")] = False,
    profile: Annotated[bool, typer.Option("--profile")] = False,
    profile_dir: Path = Path("benchmark-traces"),
    output: Path | None = None,
) -> None:
    """Benchmark one backend; profiling is opt-in and excludes setup/warmup."""
    try:
        args = BenchmarkOptions(
            backend=backend,
            device=device,
            execution_mode=execution_mode,
            expert_start=expert_start,
            expert_end=expert_end,
            dtype=dtype,
            quantization=quantization,
            expert_compute=expert_compute,
            swiglu_limit=swiglu_limit,
            tokens=tokens,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            top_k=top_k,
            experts=experts,
            distribution=distribution,
            hot_experts=hot_experts,
            hot_multiplier=hot_multiplier,
            inactive_rate=inactive_rate,
            concurrency=concurrency,
            routing_cases=routing_cases,
            routing_replay=routing_replay,
            warmup=warmup,
            repeats=repeats,
            seed=seed,
            atol=atol,
            rtol=rtol,
            max_relative_l2=max_relative_l2,
            min_cosine=min_cosine,
            skip_correctness=skip_correctness,
            profile=profile,
            profile_dir=profile_dir,
            output=output,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    result = _run_benchmark(args)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    app()
