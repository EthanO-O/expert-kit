"""Identify idle data-parallel forwards at the pinned platform Worker boundary."""

from __future__ import annotations

from contextvars import ContextVar
from functools import wraps
from typing import Any

_DUMMY_FORWARD: ContextVar[bool] = ContextVar("expertkit_dummy_forward", default=False)


def is_dummy_forward() -> bool:
    """Return whether the current execution belongs to an idle DP dummy batch."""
    return _DUMMY_FORWARD.get()


def install_dummy_batch_guard(device_type: str) -> None:
    """Mark idle DP batches without changing model execution or collectives.

    Startup profiling, warmup, and capture call the model runner directly and
    must retain remote execution to account for Transport buffer allocations.
    """
    if device_type == "npu":
        from vllm_ascend.worker.worker import NPUWorker as Worker
    else:
        from vllm.v1.worker.gpu_worker import Worker

    original = Worker.execute_dummy_batch
    if getattr(original, "_expertkit_dummy_batch", False):
        return

    @wraps(original)
    def dummy_batch(*args: Any, **kwargs: Any) -> Any:
        token = _DUMMY_FORWARD.set(True)
        try:
            return original(*args, **kwargs)
        finally:
            _DUMMY_FORWARD.reset(token)

    dummy_batch._expertkit_dummy_batch = True
    Worker.execute_dummy_batch = dummy_batch
