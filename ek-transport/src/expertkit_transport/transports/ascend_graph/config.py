"""Validate one fixed Frontend/Worker Ascend graph deployment."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator


def _validate_endpoint(endpoint: str) -> None:
    host, separator, port = endpoint.rpartition(":")
    if not separator or not host or not port.isdecimal() or not 1 <= int(port) <= 65535:
        raise ValueError("control_endpoint must use host:port")
    if "://" in host or any(c.isspace() for c in host):
        raise ValueError("graph addresses must be hostnames without a URL scheme or whitespace")


class AscendGraphWorker(BaseModel):
    """Identify one fixed Worker and its contiguous global expert interval."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    worker_id: str = Field(min_length=1, max_length=128)
    control_endpoint: str = Field(min_length=3, max_length=256)
    expert_start: int = Field(ge=0, lt=64)
    expert_end: int = Field(gt=0, le=64)

    @model_validator(mode="after")
    def validate_worker(self) -> AscendGraphWorker:
        """Reject an empty interval or an ambiguous control address."""
        if self.expert_start >= self.expert_end:
            raise ValueError("expert interval must be nonempty")
        _validate_endpoint(self.control_endpoint)
        return self


class AscendGraphConfig(BaseModel):
    """Declare the identity, graph capacities and fixed one/two-Worker group.

    All participants consume the same document. Changing any field requires a new
    session and process restart; graph mode does not perform online migration.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    session_id: str = Field(min_length=16, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    instance_id: int = Field(gt=0)
    worker_id: str | None = Field(default=None, min_length=1, max_length=128)
    weight_version: str = Field(min_length=1, max_length=256)
    control_endpoint: str | None = Field(default=None, min_length=3, max_length=256)
    workers: tuple[AscendGraphWorker, ...] = Field(default=(), strict=False)
    rendezvous_host: str = Field(min_length=1, max_length=253)
    rendezvous_port: int = Field(gt=0, le=65535)
    layer_ids: tuple[StrictInt, ...] = Field(strict=False)
    capture_sizes: tuple[StrictInt, ...] = Field(default=(1, 2, 4, 8, 16, 32, 64), strict=False)
    timeout_seconds: float = Field(default=120.0, gt=0, le=600, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_fixed_profiles(self) -> AscendGraphConfig:
        """Reject ambiguous layer order, empty capacities and invalid endpoints."""
        for name, values, minimum in (
            ("layer_ids", self.layer_ids, 0),
            ("capture_sizes", self.capture_sizes, 1),
        ):
            if (
                not values
                or any(isinstance(value, bool) or value < minimum for value in values)
                or tuple(sorted(set(values))) != values
            ):
                raise ValueError(f"{name} must be nonempty, sorted and unique")
        if self.workers:
            if self.worker_id is not None or self.control_endpoint is not None:
                raise ValueError("workers cannot be combined with legacy Worker fields")
        elif self.worker_id is None or self.control_endpoint is None:
            raise ValueError("provide workers or both legacy Worker fields")
        participants = self.participants
        expected = ((0, 64),) if len(participants) == 1 else ((0, 32), (32, 64))
        if (
            len(participants) not in (1, 2)
            or tuple((worker.expert_start, worker.expert_end) for worker in participants)
            != expected
        ):
            raise ValueError(
                "graph Workers must partition all 64 experts into one or two fixed intervals"
            )
        if len({worker.worker_id for worker in participants}) != len(participants):
            raise ValueError("graph Worker IDs must be distinct")
        if len({worker.control_endpoint for worker in participants}) != len(participants):
            raise ValueError("graph Worker control endpoints must be distinct")
        if any(c.isspace() for c in self.rendezvous_host):
            raise ValueError("graph rendezvous host must not contain whitespace")
        return self

    @classmethod
    def from_file(cls, path: Path) -> AscendGraphConfig:
        """Read the shared JSON deployment document with strict unknown-field checks."""
        return cls.model_validate_json(path.read_bytes())

    @property
    def participants(self) -> tuple[AscendGraphWorker, ...]:
        """Return canonical rank order, accepting the original single-Worker JSON."""
        if self.workers:
            return self.workers
        assert self.worker_id is not None and self.control_endpoint is not None
        return (
            AscendGraphWorker(
                worker_id=self.worker_id,
                control_endpoint=self.control_endpoint,
                expert_start=0,
                expert_end=64,
            ),
        )

    def worker_rank(self, worker_id: str) -> int:
        """Resolve exactly one Worker to its stable private HCCL rank."""
        for rank, worker in enumerate(self.participants, 1):
            if worker.worker_id == worker_id:
                return rank
        raise ValueError("Worker is absent from the graph deployment")

    def identity(self) -> dict[str, object]:
        """Return one canonical group identity regardless of legacy input syntax."""
        identity = json.loads(
            self.model_dump_json(exclude={"worker_id", "control_endpoint", "workers"})
        )
        identity["workers"] = [worker.model_dump(mode="json") for worker in self.participants]
        return identity
