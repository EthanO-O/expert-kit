"""Coordinate native Ascend FULL capture and whole-step Worker replay."""

from __future__ import annotations

import atexit
import os
import threading
import time
from contextvars import ContextVar
from functools import wraps
from importlib.metadata import version
from pathlib import Path
from typing import Any

import structlog
import torch
from expertkit_transport.ascend_graph import AscendGraphConfig, GraphStepControl, HcclGraphSession
from expertkit_transport.quarantine import retain_until_process_exit
from expertkit_transport.transports.ascend_graph.config import (
    AscendGraphModel,
    AscendGraphQuantization,
)

logger = structlog.get_logger(__name__)
_ACTIVE: ContextVar[DecodeGraphSession | None] = ContextVar("ek_decode_graph_session", default=None)
_PROFILE: ContextVar[bool] = ContextVar("ek_decode_graph_profile", default=False)
_FORWARD_SESSIONS: ContextVar[list[DecodeGraphSession] | None] = ContextVar(
    "ek_decode_graph_forward_sessions", default=None
)


def is_graph_profile() -> bool:
    """Identify startup memory profiling, which must not admit remote work."""
    return _PROFILE.get()


def graph_config_from_environment() -> AscendGraphConfig | None:
    """Read the optional shared graph deployment document; default is disabled."""
    path = os.environ.get("EK_ASCEND_DECODE_GRAPH_CONFIG")
    return None if not path else AscendGraphConfig.from_file(Path(path))


def active_graph_session() -> DecodeGraphSession | None:
    """Return the whole-step session active during native graph capture."""
    return _ACTIVE.get()


def _native_graph_quantization(vllm_config: Any) -> AscendGraphQuantization | None:
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
    for layer_id in range(model.hf_text_config.num_hidden_layers):
        prefix = f"model.layers.{layer_id}.mlp"
        if description.get(f"{prefix}.gate.weight") != "FLOAT":
            raise ValueError("Ascend graph requires unquantized native router gates")
        if is_v4:
            routing_parameter = "tid2eid" if layer_id < 3 else "bias"
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


def validate_graph_model(vllm_config: Any, config: AscendGraphConfig) -> None:
    """Reject unsupported model/runtime/scheduling combinations before capture."""
    from vllm.config import CUDAGraphMode

    model = vllm_config.model_config
    parallel = vllm_config.parallel_config
    hf = model.hf_text_config
    if (
        version("vllm").split("+", 1)[0] != "0.26.0"
        or torch.__version__.split("+", 1)[0] != "2.10.0"
        or version("vllm-ascend") != "0.1.dev4525+g155f68974"
        or version("torch-npu") != "2.10.0.post4"
    ):
        raise ValueError(
            "Ascend Decode graphs require vLLM 0.26.0, Ascend 155f68974 and Torch-NPU 2.10.0.post4"
        )
    if hf.model_type == "deepseek_v2":
        expected = {
            "num_hidden_layers": 27,
            "first_k_dense_replace": 1,
            "hidden_size": 2048,
            "moe_intermediate_size": 1408,
            "n_routed_experts": 64,
            "num_experts_per_tok": 6,
            "n_shared_experts": 2,
            "scoring_func": "softmax",
            "norm_topk_prob": False,
            "routed_scaling_factor": 1.0,
        }
        experts_per_layer = hf.n_routed_experts
        routed_layers = tuple(range(1, hf.num_hidden_layers))
    elif hf.model_type == "qwen3_moe":
        expected = {
            "num_hidden_layers": 48,
            "hidden_size": 2048,
            "moe_intermediate_size": 768,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "decoder_sparse_step": 1,
            "norm_topk_prob": True,
            "mlp_only_layers": [],
        }
        if getattr(hf, "shared_expert_intermediate_size", 0) != 0:
            raise ValueError("Qwen3 graph shared experts do not match the qualified model")
        experts_per_layer = hf.num_experts
        routed_layers = tuple(range(hf.num_hidden_layers))
    elif hf.model_type == "deepseek_v4":
        from vllm_ascend.ascend_config import get_ascend_config

        expected = {
            "num_hidden_layers": 43,
            "hidden_size": 4096,
            "moe_intermediate_size": 2048,
            "n_routed_experts": 256,
            "n_shared_experts": 1,
            "num_experts_per_tok": 6,
            "num_hash_layers": 3,
            "scoring_func": "sqrtsoftplus",
            "norm_topk_prob": True,
            "routed_scaling_factor": 1.5,
            "swiglu_limit": 10.0,
            "hidden_act": "silu",
            "vocab_size": 129280,
        }
        if getattr(hf, "vision_n_layers", 0) != 0 or get_ascend_config().mix_placement:
            raise ValueError("Ascend Decode graph V4 requires the text-only fixed placement")
        experts_per_layer = hf.n_routed_experts
        routed_layers = tuple(range(hf.num_hidden_layers))
    else:
        raise ValueError("Ascend Decode graph model is not qualified")
    if any(getattr(hf, name, None) != value for name, value in expected.items()):
        raise ValueError("Ascend Decode graph model does not match qualified semantics")
    native_model = AscendGraphModel(
        num_layers=hf.num_hidden_layers,
        experts_per_layer=experts_per_layer,
        hidden_dim=hf.hidden_size,
        expert_intermediate_dim=hf.moe_intermediate_size,
        top_k=hf.num_experts_per_tok,
        activation_dtype="bf16",
        weight_dtype="bf16",
        activation=hf.hidden_act,
        expert_compute="deepseek_v4" if hf.model_type == "deepseek_v4" else "swiglu",
        swiglu_limit=10.0 if hf.model_type == "deepseek_v4" else 0.0,
        quantization=_native_graph_quantization(vllm_config),
    )
    if config.model != native_model:
        raise ValueError("Ascend graph model declaration does not match the native Frontend model")
    if config.layer_ids != routed_layers:
        raise ValueError("Ascend graph requires every routed model layer in native order")
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
    if config.mode == "decode":
        if vllm_config.scheduler_config.max_num_seqs < max(config.capture_sizes):
            raise ValueError("vLLM max_num_seqs must cover every graph capture size")
    elif vllm_config.scheduler_config.max_num_batched_tokens < max(config.capture_sizes):
        raise ValueError("vLLM max_num_batched_tokens must cover every graph capture size")
    if os.environ.get("EK_INSTANCE_ID") != str(config.instance_id):
        raise ValueError("EK_INSTANCE_ID must match the fixed graph deployment")


class DecodeGraphSession:
    """Own Frontend data buffers and one bounded watchdog for a native FULL wrapper."""

    def __init__(self, config: AscendGraphConfig, device: torch.device) -> None:
        self.config = config
        self._control = GraphStepControl(config)
        self._data: HcclGraphSession | None = None
        self._deadline: float | None = None
        self._stop = threading.Event()
        self._poisoned = False
        self._closed = False
        self._native_entries: dict | None = None
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
        except BaseException:
            self.poison()
            raise
        atexit.register(self.close)

    def _begin(self, phase: str, capacity: int = 0) -> None:
        if self._poisoned:
            raise RuntimeError("Frontend Decode graph session is poisoned")
        self._deadline = time.monotonic() + self.config.timeout_seconds
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
        except BaseException:
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
                self._finish()
                if self._data is not None:
                    self._data.close()
        except BaseException:
            self.poison()
            logger.error("frontend_decode_graph_shutdown_failed", exc_info=True)
        finally:
            self._stop.set()
            self._watchdog.join(timeout=1)
            self._control.close()

    def _watch(self) -> None:
        while not self._stop.wait(0.1):
            deadline = self._deadline
            if deadline is not None and time.monotonic() > deadline:
                self.poison()
                logger.critical("frontend_decode_graph_deadline_exceeded")
                os._exit(1)


def install_decode_graph_wrapper(config: AscendGraphConfig) -> None:
    """Hook the exact Ascend FULL wrapper while leaving its attention/KV logic intact."""
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context
    from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

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
                validate_graph_model(wrapper.vllm_config, config)
                session = DecodeGraphSession(config, runner.device)
                wrapper.__dict__["_expertkit_decode_session"] = session
            # Allocate persistent communication storage and initialize HCCL
            # before vLLM observes free memory and budgets its KV cache.
            session.warmup(max(config.capture_sizes))
            token = _PROFILE.set(True)
            try:
                return profile_run(*args, **kwargs)
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
        if (
            wrapper.runtime_mode != CUDAGraphMode.FULL
            or context.cudagraph_runtime_mode != CUDAGraphMode.FULL
            or active_graph_session() is not None
        ):
            return original(wrapper, *args, **kwargs)
        descriptor = context.batch_descriptor
        qualified = (
            not is_graph_profile()
            and descriptor is not None
            and (
                config.mode == "full"
                or (descriptor.uniform and descriptor.num_reqs == descriptor.num_tokens)
            )
            and descriptor.num_tokens in config.capture_sizes
            and not descriptor.has_lora
        )
        if not qualified:
            previous = context.cudagraph_runtime_mode
            context.cudagraph_runtime_mode = CUDAGraphMode.NONE
            try:
                return original(wrapper, *args, **kwargs)
            finally:
                context.cudagraph_runtime_mode = previous
        session = wrapper.__dict__.get("_expertkit_decode_session")
        if session is None:
            validate_graph_model(wrapper.vllm_config, config)
            session = DecodeGraphSession(config, torch.device(f"npu:{torch.npu.current_device()}"))
            wrapper.__dict__["_expertkit_decode_session"] = session
        return session.run(wrapper, original, args, kwargs, descriptor.num_tokens)

    call._expertkit_decode_graph = True
    ACLGraphWrapper.__call__ = call
