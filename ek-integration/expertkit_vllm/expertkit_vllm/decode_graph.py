"""Coordinate native Ascend FULL capture and whole-step Worker replay."""

from __future__ import annotations

import atexit
import os
import threading
import time
from contextvars import ContextVar
from functools import wraps
from importlib.metadata import version
from inspect import signature
from pathlib import Path
from typing import Any
from uuid import uuid4

import structlog
import torch
from expertkit_transport.ascend_graph import (
    AscendGraphConfig,
    AscendGraphModel,
    AscendGraphOptions,
    AscendGraphQuantization,
    GraphStepControl,
    GraphTopologySubscription,
    HcclGraphSession,
    resolve_default_instance_sync,
    resolve_graph_limits,
)
from expertkit_transport.quarantine import retain_until_process_exit

from expertkit_vllm.utils.config import collect_ek_client_config

logger = structlog.get_logger(__name__)
_ACTIVE: ContextVar[DecodeGraphSession | None] = ContextVar("ek_decode_graph_session", default=None)
_CAPTURE: ContextVar[bool] = ContextVar("ek_graph_capture_scope", default=False)
_SYNTHETIC: ContextVar[bool] = ContextVar("ek_graph_synthetic", default=False)
_REAL_STEP: ContextVar[tuple[bool, int] | None] = ContextVar("ek_graph_real_step", default=None)
_PROFILE: ContextVar[bool] = ContextVar("ek_decode_graph_profile", default=False)
_FORWARD_SESSIONS: ContextVar[list[DecodeGraphSession] | None] = ContextVar(
    "ek_decode_graph_forward_sessions", default=None
)


def is_graph_profile() -> bool:
    """Identify startup memory profiling, which must not admit remote work."""
    return _PROFILE.get()


def graph_config_from_environment() -> AscendGraphOptions | None:
    """Read user graph choices without inventing resolved model or placement facts."""
    path = os.environ.get("EK_ASCEND_DECODE_GRAPH_CONFIG")
    return None if not path else AscendGraphOptions.from_file(Path(path))


def active_graph_session() -> DecodeGraphSession | None:
    """Return the whole-step session active during native graph capture."""
    return _ACTIVE.get()


def _native_graph_quantization(
    vllm_config: Any, layer_ids: tuple[int, ...]
) -> AscendGraphQuantization | None:
    """Resolve the qualified expert recipe from the actual native configuration."""
    model = vllm_config.model_config
    native = getattr(vllm_config, "quant_config", None)
    is_v4 = model.hf_text_config.model_type == "deepseek_v4"
    if model.quantization is None and native is None and not is_v4:
        return None
    if (
        model.hf_text_config.model_type not in ("qwen3_moe", "deepseek_v4")
        or model.quantization != "ascend"
    ):
        raise ValueError("Ascend graph native quantization recipe is not qualified")

    from vllm_ascend.quantization.modelslim_config import AscendModelSlimConfig

    if type(native) is not AscendModelSlimConfig:
        raise ValueError("Ascend graph requires the native ModelSlim configuration")
    description = native.quant_description
    for layer_id in layer_ids:
        prefix = f"model.layers.{layer_id}.mlp"
        if description.get(f"{prefix}.gate.weight") != "FLOAT":
            raise ValueError("Ascend graph requires unquantized native router gates")
        if is_v4:
            routing_parameter = (
                "tid2eid" if layer_id < model.hf_text_config.num_hash_layers else "bias"
            )
            if description.get(f"{prefix}.gate.{routing_parameter}") != "FLOAT":
                raise ValueError("Ascend graph requires native V4 routing metadata descriptors")
            for projection in ("gate_proj", "up_proj", "down_proj"):
                name = f"{prefix}.shared_experts.{projection}.weight"
                if description.get(name) != "W8A8_DYNAMIC":
                    raise ValueError("Ascend graph requires native V4 ModelSlim shared experts")
        expert_count = (
            model.hf_text_config.n_routed_experts if is_v4 else model.hf_text_config.num_experts
        )
        for expert_id in range(expert_count):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                name = f"{prefix}.experts.{expert_id}.{projection}.weight"
                if description.get(name) != "W8A8_DYNAMIC":
                    raise ValueError(
                        "Ascend graph requires every expert to use ModelSlim W8A8 dynamic"
                    )
    return AscendGraphQuantization(
        type="modelslim-w8a8-dynamic", bits=8, group_size=None, symmetric=True
    )


def resolve_graph_model(vllm_config: Any) -> tuple[AscendGraphModel, tuple[int, ...]]:
    """Read computation facts and actual routed-layer order from native HF readers."""
    hf = vllm_config.model_config.hf_text_config
    if hf.model_type == "deepseek_v2":
        frequency = getattr(hf, "moe_layer_freq", 1)
        routed_layers = tuple(
            layer
            for layer in range(hf.num_hidden_layers)
            if layer >= hf.first_k_dense_replace and layer % frequency == 0
        )
        experts = hf.n_routed_experts
    elif hf.model_type == "qwen3_moe":
        routed_layers = tuple(
            layer
            for layer in range(hf.num_hidden_layers)
            if layer not in getattr(hf, "mlp_only_layers", [])
            and (layer + 1) % hf.decoder_sparse_step == 0
        )
        experts = hf.num_experts
    elif hf.model_type == "deepseek_v4":
        from vllm_ascend.ascend_config import get_ascend_config

        if getattr(hf, "vision_n_layers", 0) != 0 or get_ascend_config().mix_placement:
            raise ValueError("Ascend graph V4 requires the text-only fixed placement")
        if hf.scoring_func != "sqrtsoftplus" or not hf.norm_topk_prob or hf.n_shared_experts != 1:
            raise ValueError(
                "Ascend graph V4 requires native sqrtsoftplus routing and shared expert"
            )
        routed_layers = tuple(range(hf.num_hidden_layers))
        experts = hf.n_routed_experts
    else:
        raise ValueError("Ascend graph model computation is not supported")
    if not routed_layers:
        raise ValueError("Ascend graph requires at least one native routed layer")
    return AscendGraphModel(
        num_layers=hf.num_hidden_layers,
        experts_per_layer=experts,
        hidden_dim=hf.hidden_size,
        expert_intermediate_dim=hf.moe_intermediate_size,
        top_k=hf.num_experts_per_tok,
        activation_dtype="bf16",
        weight_dtype="bf16",
        activation=hf.hidden_act,
        expert_compute="deepseek_v4" if hf.model_type == "deepseek_v4" else "swiglu",
        swiglu_limit=float(hf.swiglu_limit) if hf.model_type == "deepseek_v4" else 0.0,
        quantization=_native_graph_quantization(vllm_config, routed_layers),
    ), routed_layers


def validate_graph_model(vllm_config: Any, config: AscendGraphConfig) -> None:
    """Cross-check a resolved runtime against the actual native model and runtime."""
    _validate_graph_runtime(vllm_config, config)
    native_model, routed_layers = resolve_graph_model(vllm_config)
    if config.model != native_model:
        raise ValueError("Ascend graph model declaration does not match the native Frontend model")
    if config.layer_ids != routed_layers:
        raise ValueError("Ascend graph requires every routed model layer in native order")
    options = AscendGraphOptions.model_validate(
        config.model_dump(include=set(AscendGraphOptions.model_fields))
    )
    scheduler = vllm_config.scheduler_config
    limits = _native_graph_limits(
        vllm_config,
        options,
        allocated_requests=scheduler.max_num_seqs,
        allocated_tokens=scheduler.max_num_batched_tokens,
    )
    if config.decode_capture_sizes != limits.decode_capture_sizes:
        raise ValueError("resolved Decode captures do not match native request buffers")


def _validate_graph_runtime(
    vllm_config: Any, config: AscendGraphConfig | AscendGraphOptions
) -> None:
    from vllm.config import CUDAGraphMode

    model = vllm_config.model_config
    parallel = vllm_config.parallel_config
    if (
        version("vllm").split("+", 1)[0] != "0.26.0"
        or torch.__version__.split("+", 1)[0] != "2.10.0"
        or version("vllm-ascend") != "0.1.dev4525+g155f68974"
        or version("torch-npu") != "2.10.0.post4"
    ):
        raise ValueError(
            "Ascend Decode graphs require vLLM 0.26.0, Ascend 155f68974 and Torch-NPU 2.10.0.post4"
        )
    if vllm_config.cache_config.enable_prefix_caching or os.environ.get("EK_TRACE_ENDPOINT"):
        raise ValueError("Ascend Decode graph requires prefix caching and tracing disabled")
    if (
        parallel.tensor_parallel_size != 1
        or parallel.data_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
        or parallel.enable_dbo
        or parallel.enable_eplb
        or parallel.enable_expert_parallel
        or model.enforce_eager
        or model.dtype != torch.bfloat16
        or vllm_config.speculative_config is not None
        or vllm_config.lora_config is not None
        or vllm_config.scheduler_config.async_scheduling
        or vllm_config.compilation_config.cudagraph_mode
        != (CUDAGraphMode.FULL if config.mode == "full" else CUDAGraphMode.FULL_DECODE_ONLY)
    ):
        raise ValueError(
            "Ascend graph requires single-rank synchronous BF16 "
            "and its configured native graph mode"
        )
    if set(vllm_config.compilation_config.cudagraph_capture_sizes) != set(config.capture_sizes):
        raise ValueError("vLLM and Worker graph capture sizes must match")
    if config.mode == "full" and not vllm_config.scheduler_config.enable_chunked_prefill:
        raise ValueError("Full graph requires native chunked Prefill scheduling")


def _native_graph_limits(
    vllm_config: Any,
    options: AscendGraphOptions,
    *,
    allocated_requests: int,
    allocated_tokens: int,
):
    scheduler = vllm_config.scheduler_config
    if scheduler.scheduler_cls is not None:
        from vllm.v1.core.sched.scheduler import Scheduler

        if scheduler.get_scheduler_cls() is not Scheduler:
            raise ValueError("Ascend graphs require the default native Scheduler")
    native_sizes = vllm_config.compilation_config.cudagraph_capture_sizes
    if tuple(sorted(set(native_sizes))) != options.capture_sizes:
        raise ValueError("vLLM and Worker graph capture sizes must match")
    scheduled = scheduler.max_num_scheduled_tokens
    if scheduled is None:
        scheduled = scheduler.max_num_batched_tokens
    limits = resolve_graph_limits(
        options,
        allocated_requests=allocated_requests,
        allocated_tokens=allocated_tokens,
        scheduled_tokens=scheduled,
    )
    logger.info(
        "ascend_graph_execution_limits",
        mode=options.mode,
        requested_capture_sizes=options.capture_sizes,
        allocated_requests=allocated_requests,
        allocated_tokens=allocated_tokens,
        requested_scheduled_tokens=scheduled,
        effective_running_requests=limits.running_requests,
        effective_scheduled_tokens=(
            limits.graph_stage_tokens if options.mode == "full" else scheduled
        ),
        decode_capture_sizes=limits.decode_capture_sizes,
        reason="configured graph capacities bounded by native request and token buffers",
    )
    return limits


def install_graph_scheduler(options: AscendGraphOptions) -> None:
    """Limit native scheduling fields while preserving original allocation settings."""
    from vllm.config import SchedulerConfig
    from vllm.v1.core.sched.scheduler import Scheduler

    select = SchedulerConfig.get_scheduler_cls
    if not getattr(select, "_expertkit_graph_scheduler", False):

        @wraps(select)
        def selected(config):
            scheduler_class = select(config)
            if scheduler_class is not Scheduler:
                raise ValueError("Ascend graphs require the default native Scheduler")
            return scheduler_class

        selected._expertkit_graph_scheduler = True
        SchedulerConfig.get_scheduler_cls = selected
    original = Scheduler.__init__
    if getattr(original, "_expertkit_graph_scheduler", False):
        return
    arguments = signature(original)

    @wraps(original)
    def initialize(scheduler, *args, **kwargs):
        bound = arguments.bind(scheduler, *args, **kwargs)
        native = bound.arguments["vllm_config"]
        limits = _native_graph_limits(
            native,
            options,
            allocated_requests=native.scheduler_config.max_num_seqs,
            allocated_tokens=native.scheduler_config.max_num_batched_tokens,
        )
        original(scheduler, *args, **kwargs)
        scheduler.max_num_running_reqs = min(
            scheduler.max_num_running_reqs, limits.running_requests
        )
        if options.mode == "full":
            scheduler.max_num_scheduled_tokens = min(
                scheduler.max_num_scheduled_tokens, limits.graph_stage_tokens
            )

    initialize._expertkit_graph_scheduler = True
    Scheduler.__init__ = initialize


def _resolve_graph_config(runner: Any, options: AscendGraphOptions):
    """Resolve one startup snapshot without creating ordinary Worker connections."""
    deadline = time.monotonic() + options.timeout_seconds
    vllm_config = runner.vllm_config
    _validate_graph_runtime(vllm_config, options)
    model, layers = resolve_graph_model(vllm_config)
    limits = _native_graph_limits(
        vllm_config,
        options,
        allocated_requests=runner.max_num_reqs,
        allocated_tokens=runner.max_num_tokens,
    )
    client = collect_ek_client_config()
    instance = resolve_default_instance_sync(
        client.controller_endpoint,
        requested_instance_id=client.instance_id,
        monotonic_deadline=deadline,
    )
    subscription = GraphTopologySubscription(
        client.controller_endpoint,
        instance_id=instance.instance_id,
        num_layers=model.num_layers,
        experts_per_layer=model.experts_per_layer,
    )
    try:
        source_version, weight_version, workers = subscription.start(
            layer_ids=layers,
            max_capture_size=max(options.capture_sizes),
            monotonic_deadline=deadline,
        )
        host = options.rendezvous_host
        if host is None:
            host = workers[0].control_endpoint.rpartition(":")[0]
            if host.startswith("[") and host.endswith("]"):
                host = host[1:-1]
        facts = {
            **options.model_dump(mode="json"),
            "session_id": uuid4().hex,
            "instance_id": instance.instance_id,
            "model_name": instance.model_name,
            "weight_version": weight_version,
            "model": model.model_dump(mode="json"),
            "layer_ids": list(layers),
            "decode_capture_sizes": list(limits.decode_capture_sizes),
            "workers": [worker.model_dump(mode="json") for worker in workers],
            "source_topology_version": source_version,
            "rendezvous_host": host,
            "rendezvous_port": options.rendezvous_port or 0,
        }
        control = GraphStepControl.reserve(
            facts, monotonic_deadline=deadline, require_current=subscription.require_current
        )
        config = control.config
        logger.info(
            "frontend_graph_rendezvous_resolved",
            session_id=config.session_id,
            worker_id=config.workers[0].worker_id,
            rendezvous_host=config.rendezvous_host,
            rendezvous_port=config.rendezvous_port,
        )
        return config, subscription, deadline, control
    except BaseException:
        subscription.close()
        raise


def _is_v4_full(config: AscendGraphConfig) -> bool:
    return config.mode == "full" and config.model.expert_compute == "deepseek_v4"


class DecodeGraphSession:
    """Own Frontend data buffers and one bounded watchdog for a native FULL wrapper."""

    def __init__(
        self,
        config: AscendGraphConfig,
        device: torch.device,
        *,
        topology: GraphTopologySubscription | None = None,
        attention: Any = None,
        startup_deadline: float | None = None,
        control: GraphStepControl | None = None,
    ) -> None:
        self.config = config
        self._topology = topology
        self._startup_deadline = startup_deadline
        self._control = (
            control
            if control is not None
            else GraphStepControl(
                config,
                require_current=(
                    None if topology is None else lambda: topology.require_current(config)
                ),
            )
        )
        self._data: HcclGraphSession | None = None
        self._deadline: float | None = None
        self._stop = threading.Event()
        self._poisoned = False
        self._closed = False
        self._native_entries: dict | None = None
        self._v4_phase_entries: dict[str, dict] | None = None
        self._v4_attention: Any = attention
        self._native_capture_failure: BaseException | None = None
        self._startup_failure: BaseException | None = None
        self._layer_index = 0
        self._capacity = 0
        self._watchdog = threading.Thread(
            target=self._watch, name="ek-decode-frontend-watchdog", daemon=True
        )
        self._watchdog.start()
        try:
            self._begin("prepare")
            self._data = HcclGraphSession(
                config,
                rank=0,
                device=device,
                hidden_dim=config.model.hidden_dim,
                top_k=config.model.top_k,
            )
            for capacity in config.capture_sizes:
                self._data.buffers(capacity)
            torch.npu.current_stream().synchronize()
            self._finish()
            self._startup_deadline = None
        except BaseException as error:
            # A failed native constructor can own resources before _data is assigned.
            self._startup_failure = error
            self.poison()
            self.close()
            raise
        atexit.register(self.close)

    def _begin(self, phase: str, capacity: int = 0) -> None:
        if self._poisoned or (self._closed and phase != "close"):
            raise RuntimeError("Frontend Decode graph session is poisoned or closed")
        if (
            phase == "prepare"
            and self._startup_deadline is not None
            and time.monotonic() >= self._startup_deadline
        ):
            raise TimeoutError("graph startup deadline expired before PREPARE")
        self._deadline = time.monotonic() + self.config.timeout_seconds
        if _is_v4_full(self.config) and phase in {"capture", "replay"}:
            if self._v4_attention is None:
                raise RuntimeError("V4 full attention phase owner is missing")
            self._control.begin(phase, capacity, graph_phase=self._v4_attention.phase)
        else:
            self._control.begin(phase, capacity)
        self._layer_index = 0
        self._capacity = capacity

    def _finish(self) -> None:
        self._control.finish()
        self._deadline = None

    def warmup(self, capacity: int) -> None:
        """Warm all communication and GMM operators outside native model capture."""
        assert self._data is not None
        self._begin("warmup", capacity)
        buffers = self._data.buffers(capacity)
        for _ in self.config.layer_ids:
            self._data.receive_inputs(buffers)
            self._data.combine(buffers)
        torch.npu.current_stream().synchronize()
        self._finish()

    def exchange(
        self, layer_id: int, hidden: torch.Tensor, ids: torch.Tensor, routing: torch.Tensor
    ) -> torch.Tensor:
        """Record one layer's device operations while preserving Frontend routing."""
        if self._poisoned or self._data is None:
            raise RuntimeError("Frontend Decode graph session is not ready")
        if (
            self._layer_index >= len(self.config.layer_ids)
            or layer_id != self.config.layer_ids[self._layer_index]
            or hidden.shape != (self._capacity, self.config.model.hidden_dim)
            or ids.shape != (self._capacity, self.config.model.top_k)
            or routing.shape != ids.shape
            or hidden.dtype != torch.bfloat16
            or ids.dtype != torch.int32
            or routing.dtype != torch.float32
        ):
            raise ValueError(
                "graph layer order or routed Tensor contract does not match the profile"
            )
        self._layer_index += 1
        return self._data.exchange(hidden, ids, routing)

    def run(self, wrapper: Any, original: Any, args: tuple, kwargs: dict, capacity: int) -> Any:
        """Coordinate the Worker before both native capture and native replay."""
        from vllm.forward_context import get_forward_context

        descriptor = get_forward_context().batch_descriptor
        self._native_entries = wrapper.concrete_aclgraph_entries
        entry = wrapper.concrete_aclgraph_entries.get(descriptor)
        capturing = entry is None or entry.aclgraph is None
        try:
            if capturing:
                self.warmup(capacity)
            self._begin("capture" if capturing else "replay", capacity)
            pending = _FORWARD_SESSIONS.get()
            if pending is not None:
                if pending:
                    raise RuntimeError("one native forward must own exactly one FULL graph")
                pending.append(self)
            token = _ACTIVE.set(self)
            try:
                result = original(wrapper, *args, **kwargs)
            finally:
                _ACTIVE.reset(token)
            expected_calls = len(self.config.layer_ids) if capturing else 0
            if self._layer_index != expected_calls:
                raise RuntimeError(
                    "native FULL graph executed an unexpected number of Python MoE calls"
                )
            if pending is None:
                self._complete_forward()
            return result
        except BaseException as error:
            # Native capture publishes its graph only after successful exit.
            # Its traceback owns any graph still local to the failed call.
            self._native_capture_failure = error
            self.poison()
            raise

    def _complete_forward(self) -> None:
        """Fence after native attention updates, then acknowledge Worker completion."""
        torch.npu.current_stream().synchronize()
        self._finish()

    def poison(self) -> None:
        """Retain buffers and reject fallback after partially submitted graph work."""
        self._poisoned = True
        self._control.poison()
        if self._data is not None:
            self._data.quarantine()
        retain_until_process_exit(self)

    def close(self) -> None:
        """Drain both peers and destroy graphs before releasing fixed buffers."""
        if self._closed:
            return
        self._closed = True
        try:
            if not self._poisoned:
                self._begin("close")
                torch.npu.current_stream().synchronize()
                if self._native_entries is not None:
                    self._native_entries.clear()
                if self._v4_phase_entries is not None:
                    for entries in self._v4_phase_entries.values():
                        entries.clear()
                if self._v4_attention is not None:
                    self._v4_attention.close()
                self._finish()
                if self._data is not None:
                    self._data.close()
        except BaseException:
            self.poison()
            logger.error("frontend_decode_graph_shutdown_failed", exc_info=True)
        finally:
            self._stop.set()
            self._watchdog.join(timeout=1)
            try:
                self._control.close()
            finally:
                if self._topology is not None:
                    self._topology.close()

    def _watch(self) -> None:
        while not self._stop.wait(0.1):
            deadline = self._deadline
            if deadline is not None and time.monotonic() > deadline:
                self.poison()
                logger.critical("frontend_decode_graph_deadline_exceeded")
                os._exit(1)


def install_decode_graph_wrapper(options: AscendGraphOptions) -> None:
    """Hook the exact Ascend FULL wrapper while leaving its attention/KV logic intact."""
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context
    from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    install_graph_scheduler(options)
    from expertkit_vllm.v4_full_attention import install_v4_memory_reservation

    install_v4_memory_reservation()
    dummy_run = NPUModelRunner._dummy_run
    if not getattr(dummy_run, "_expertkit_graph_synthetic", False):
        dummy_arguments = signature(dummy_run)

        @wraps(dummy_run)
        def dummy(*args, **kwargs):
            # Bind the native signature so positional profile flags retain meaning.
            bound = dummy_arguments.bind(*args, **kwargs)
            token = _SYNTHETIC.set(True)
            profile_token = _PROFILE.set(
                is_graph_profile() or bound.arguments.get("is_profile", False)
            )
            try:
                return dummy_run(*args, **kwargs)
            finally:
                _PROFILE.reset(profile_token)
                _SYNTHETIC.reset(token)

        dummy._expertkit_graph_synthetic = True
        NPUModelRunner._dummy_run = dummy
    execute_model = NPUModelRunner.execute_model
    if not getattr(execute_model, "_expertkit_graph_request", False):
        execute_arguments = signature(execute_model)

        @wraps(execute_model)
        def execute(*args, **kwargs):
            bound = execute_arguments.bind(*args, **kwargs)
            output = bound.arguments["scheduler_output"]
            counts = output.num_scheduled_tokens
            if any(type(count) is not int or count <= 0 for count in counts.values()):
                raise ValueError("graph scheduling requires positive Host query counts")
            token = _REAL_STEP.set(
                (bool(counts) and all(c == 1 for c in counts.values()), sum(counts.values()))
            )
            try:
                return execute_model(*args, **kwargs)
            finally:
                _REAL_STEP.reset(token)

        execute._expertkit_graph_request = True
        NPUModelRunner.execute_model = execute

    model_forward = NPUModelRunner._model_forward
    if not getattr(model_forward, "_expertkit_graph_completion", False):

        @wraps(model_forward)
        def forward(*args, **kwargs):
            if _FORWARD_SESSIONS.get() is not None:
                raise RuntimeError("nested native model forwards are unsupported")
            pending: list[DecodeGraphSession] = []
            token = _FORWARD_SESSIONS.set(pending)
            try:
                result = model_forward(*args, **kwargs)
                # MLA records captured attention events after the model wrapper
                # returns. Waiting inside that wrapper would block their record.
                for session in pending:
                    session._complete_forward()
                return result
            except BaseException:
                for session in pending:
                    session.poison()
                raise
            finally:
                _FORWARD_SESSIONS.reset(token)

        forward._expertkit_graph_completion = True
        NPUModelRunner._model_forward = forward

    capture_graphs = NPUModelRunner._capture_cudagraphs
    if not getattr(capture_graphs, "_expertkit_v4_phases", False):

        @wraps(capture_graphs)
        def capture(runner, batch_descriptors, cudagraph_runtime_mode, profiler=None):
            token = _CAPTURE.set(True)
            adapter = getattr(runner, "_expertkit_v4_full_attention", None)
            try:
                if adapter is None or cudagraph_runtime_mode != CUDAGraphMode.FULL:
                    return capture_graphs(
                        runner, batch_descriptors, cudagraph_runtime_mode, profiler
                    )
                config = runner.model._expertkit_decode_session.config
                for phase in ("decode", "mixed"):
                    descriptors = [
                        descriptor
                        for descriptor in batch_descriptors
                        if (
                            descriptor.num_tokens in config.decode_capture_sizes
                            if phase == "decode"
                            else descriptor.num_tokens >= 2
                            and descriptor.num_tokens in config.capture_sizes
                        )
                    ]
                    with adapter.capture_phase(phase):
                        capture_graphs(runner, descriptors, cudagraph_runtime_mode, profiler)
            except BaseException:
                session = runner.model.__dict__.get("_expertkit_decode_session")
                if session is not None:
                    session.poison()
                raise
            finally:
                _CAPTURE.reset(token)

        capture._expertkit_v4_phases = True
        NPUModelRunner._capture_cudagraphs = capture

    profile_run = NPUModelRunner.profile_run
    if not getattr(profile_run, "_expertkit_graph_profile", False):

        @wraps(profile_run)
        def profile(*args, **kwargs):
            runner = args[0]
            wrapper = runner.model
            if (
                not isinstance(wrapper, ACLGraphWrapper)
                or wrapper.runtime_mode != CUDAGraphMode.FULL
            ):
                raise ValueError(
                    "graph memory profiling requires the native main-model FULL wrapper"
                )
            session = wrapper.__dict__.get("_expertkit_decode_session")
            if session is None:
                config, topology, deadline, control = _resolve_graph_config(runner, options)
                adapter = None
                try:
                    if _is_v4_full(config):
                        from expertkit_vllm.v4_full_attention import install_v4_full_attention

                        adapter = install_v4_full_attention(runner, config)
                        wrapper.__dict__["_expertkit_v4_full_attention"] = adapter
                    session = DecodeGraphSession(
                        config,
                        runner.device,
                        topology=topology,
                        attention=adapter,
                        startup_deadline=deadline,
                        control=control,
                    )
                    wrapper.__dict__["_expertkit_decode_session"] = session
                except BaseException as error:
                    # Retain failed constructors' unpublished native resources too.
                    retain_until_process_exit(error)
                    try:
                        if control is not None:
                            control.close()
                    finally:
                        topology.close()
                    raise
            config = session.config
            token = _PROFILE.set(True)
            try:
                session.warmup(max(config.capture_sizes))
                if session._v4_attention is not None:
                    session._v4_attention.prepare_profile_metadata()
                return profile_run(*args, **kwargs)
            except BaseException:
                session.poison()
                raise
            finally:
                _PROFILE.reset(token)

        profile._expertkit_graph_profile = True
        NPUModelRunner.profile_run = profile

    original = ACLGraphWrapper.__call__
    if getattr(original, "_expertkit_decode_graph", False):
        return

    @wraps(original)
    def call(wrapper, *args, **kwargs):
        context = get_forward_context()
        if active_graph_session() is not None:
            return original(wrapper, *args, **kwargs)
        descriptor = context.batch_descriptor
        session = wrapper.__dict__.get("_expertkit_decode_session")
        choices = options if session is None else session.config
        real = _REAL_STEP.get()
        synthetic = is_graph_profile() or _SYNTHETIC.get()
        pure_decode = (
            real[0]
            if real is not None
            else descriptor is not None
            and descriptor.uniform
            and descriptor.num_tokens > 0
            and descriptor.num_reqs == descriptor.num_tokens
        )
        required = (
            not synthetic
            and (choices.mode == "full" or pure_decode)
            and (real is None or real[1] > 0)
        ) or (_CAPTURE.get() and context.cudagraph_runtime_mode == CUDAGraphMode.FULL)
        if real is not None and descriptor is not None and descriptor.num_tokens < real[1]:
            raise ValueError("native padded tokens are smaller than the Host scheduled token total")
        sizes = choices.capture_sizes
        qualified = (
            not is_graph_profile()
            and wrapper.runtime_mode == CUDAGraphMode.FULL
            and context.cudagraph_runtime_mode == CUDAGraphMode.FULL
            and descriptor is not None
            and (
                choices.mode == "full"
                or (
                    pure_decode
                    and descriptor.uniform
                    and descriptor.num_reqs == descriptor.num_tokens
                )
            )
            and descriptor.num_tokens in sizes
            and not descriptor.has_lora
        )
        if not qualified:
            if required:
                raise ValueError("graph-required request has no eligible native FULL graph")
            previous = context.cudagraph_runtime_mode
            context.cudagraph_runtime_mode = CUDAGraphMode.NONE
            try:
                return original(wrapper, *args, **kwargs)
            finally:
                context.cudagraph_runtime_mode = previous
        session = wrapper.__dict__.get("_expertkit_decode_session")
        if session is None:
            raise RuntimeError("graph session must be resolved during native profile_run")
        config = session.config
        if pure_decode and descriptor.num_tokens not in config.decode_capture_sizes:
            raise ValueError("Decode graph capacity exceeds native allocated request buffers")
        adapter = wrapper.__dict__.get("_expertkit_v4_full_attention")
        if adapter is None and _is_v4_full(config):
            raise RuntimeError("V4 full attention must be installed before profiling")
        if adapter is None:
            return session.run(wrapper, original, args, kwargs, descriptor.num_tokens)
        previous = wrapper.concrete_aclgraph_entries
        maps = wrapper.__dict__.setdefault(
            "_expertkit_v4_phase_entries", {"decode": previous, "mixed": {}}
        )
        session._v4_attention = adapter
        session._v4_phase_entries = maps
        phase = adapter.phase
        if phase not in maps:
            raise ValueError("V4 full attention phase must be decode or mixed")
        wrapper.concrete_aclgraph_entries = maps[phase]
        try:
            return session.run(wrapper, original, args, kwargs, descriptor.num_tokens)
        finally:
            wrapper.concrete_aclgraph_entries = previous

    call._expertkit_decode_graph = True
    ACLGraphWrapper.__call__ = call
