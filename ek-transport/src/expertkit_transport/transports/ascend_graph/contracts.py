"""Graph-step command failures independent of their control transport."""

from typing import Any, Protocol


def validate_graph_control_endpoint(endpoint: str) -> None:
    """Require a graph-control host and nonzero port without URL components."""
    if not isinstance(endpoint, str):
        raise ValueError("graph_control_endpoint must be a string")
    host, separator, port = endpoint.rpartition(":")
    if (
        not separator
        or not host
        or any(character in host for character in "/?#@")
        or (":" in host and not (host.startswith("[") and host.endswith("]")))
        or any(character.isspace() for character in host)
        or not port.isdecimal()
        or not 1 <= int(port) <= 65535
    ):
        raise ValueError("graph_control_endpoint must use host:port without a scheme")


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
