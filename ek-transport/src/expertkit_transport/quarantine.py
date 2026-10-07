"""Retain unsafe device ownership until the process exits."""

import atexit


def _release_at_process_exit(owners: tuple[object, ...]) -> None:
    del owners


def retain_until_process_exit(*owners: object) -> None:
    """Prevent garbage collection after device access cannot be proven terminal.

    The exit callback holds strong references independently of caller lifetime.
    Register only on fatal failures; normal close cannot make unsafe DMA safe.
    """
    atexit.register(_release_at_process_exit, owners)
