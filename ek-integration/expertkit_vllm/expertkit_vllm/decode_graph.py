"""Coordinate native Ascend FULL Decode capture and whole-step Worker replay."""

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


def validate_graph_model(vllm_config: Any, config: AscendGraphConfig) -> None:
    """Reject unsupported model/runtime/scheduling combinations before capture."""
    from vllm.config import CUDAGraphMode

    model = vllm_config.model_config
    parallel = vllm_config.parallel_config
    hf = model.hf_text_config
    if (
        version("vllm") != "0.30.0"
        or torch.__version__.split("+", 1)[0] != "2.10.0"
        or not version("vllm-ascend").startswith("0.1.dev1+g8d4409d62.")
        or version("torch-npu") != "2.10.0.post4"
    ):
        raise ValueError(
            "Ascend Decode graphs require the qualified vLLM 0.30/Torch-NPU 2.10 stack"
        )
    expected = {
        "model_type": "deepseek_v2",
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
    if any(getattr(hf, name, None) != value for name, value in expected.items()):
        raise ValueError("Ascend Decode graph model does not match DeepSeek-V2-Lite BF16")
    if config.layer_ids != tuple(range(1, 27)):
        raise ValueError("DeepSeek-V2-Lite graph requires all routed layers 1..26")
    if (
        parallel.tensor_parallel_size != 1
        or parallel.data_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
        or parallel.enable_dbo
        or parallel.enable_eplb
        or parallel.enable_expert_parallel
        or model.enforce_eager
        or model.dtype != torch.bfloat16
        or model.quantization is not None
        or vllm_config.speculative_config is not None
        or vllm_config.lora_config is not None
        or vllm_config.scheduler_config.async_scheduling
        or vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY
    ):
        raise ValueError(
            "Ascend Decode graph requires single-rank synchronous FULL_DECODE_ONLY BF16"
        )
    if set(vllm_config.compilation_config.cudagraph_capture_sizes) != set(config.capture_sizes):
        raise ValueError("vLLM and Worker graph capture sizes must match")
    if vllm_config.scheduler_config.max_num_seqs < max(config.capture_sizes):
        raise ValueError("vLLM max_num_seqs must cover every graph capture size")
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
            self._data = HcclGraphSession(config, rank=0, device=device, hidden_dim=2048, top_k=6)
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
            self._data.broadcast(buffers.output, 1)
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
            or hidden.shape != (self._capacity, 2048)
            or ids.shape != (self._capacity, 6)
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
            and descriptor.uniform
            and descriptor.num_reqs == descriptor.num_tokens
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
