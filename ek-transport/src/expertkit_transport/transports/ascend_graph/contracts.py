"""Graph-step command failures independent of their control transport."""

from typing import Any, Protocol


class GraphControlRejected(RuntimeError):
    """The command does not match the current graph admission state."""


class GraphAdmissionTimeout(TimeoutError):
    """Graph admission was not acknowledged within the caller's wait budget."""


class GraphCompletionUnproven(RuntimeError):
    """Submitted graph work failed or its completion could not be proven."""


class GraphStepCommand(Protocol):
    """Accept metadata and a Host wait budget without transferring work ownership."""

    def __call__(self, request: dict[str, Any], *, timeout: float | None) -> dict[str, object]:
        """Execute a command; timeout or cancellation never releases device storage."""
        ...
