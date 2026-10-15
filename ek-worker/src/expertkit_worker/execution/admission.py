"""Exclude ordinary and graph device work after an unproven completion."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

from expertkit_worker.backends import BackendFatalError, BackendFatalReason


class DeviceAdmission:
    """Serialize device owners and reject waiters before unsafe ownership unlocks."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._poisoned = False

    @contextmanager
    def use(self, *, poison_on_error: bool = False) -> Iterator[None]:
        """Hold admission through completion; poison while still owning the lock.

        Ordinary slots classify safe request failures themselves. Graph errors
        can follow partially submitted HCCL work and therefore poison on every
        exception. Waiters never execute device work after either unsafe failure.
        """
        with self._lock:
            if self._poisoned:
                raise BackendFatalError(
                    BackendFatalReason.ASYNC_EXECUTION,
                    "device admission is quarantined after unproven completion",
                )
            try:
                yield
            except BaseException as error:
                if poison_on_error or isinstance(error, BackendFatalError):
                    self._poisoned = True
                raise
