"""Validate graph choices and one immutable, Controller-resolved graph session."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
    model_validator,
)

from expertkit_transport.transports.ascend_graph.contracts import validate_graph_control_endpoint

_MIGRATION_ERROR = (
    "Legacy Ascend graph deployment documents are no longer supported. Use ordinary "
    "ek-cli schedule and wait for READY experts; keep only mode, capture_sizes, "
    "timeout_seconds, rendezvous_host and rendezvous_port in the graph options file. "
    "Configure each Worker's control_listen/control_advertise locally; model facts, "
    "instance identity, weight version and expert assignments are resolved at startup."
)
_LEGACY_FIELDS = {
    "session_id",
    "workers",
    "worker_id",
    "control_endpoint",
    "model",
    "layer_ids",
    "instance_id",
    "weight_version",
}


def normalize_capture_sizes(values: tuple[int, ...]) -> tuple[int, ...]:
    """Sort and deduplicate positive capacities without adding unrequested sizes."""
    if not values or any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("capture_sizes must be nonempty and contain positive integers")
    return tuple(sorted(set(values)))


def _validate_rendezvous_host(value: str) -> str:
    if any(character.isspace() for character in value) or any(c in value for c in "/?#@"):
        raise ValueError("rendezvous_host must be a host without whitespace or URL components")
    return value


_CaptureSizes = Annotated[
    tuple[StrictInt, ...], Field(strict=False), AfterValidator(normalize_capture_sizes)
]
_RendezvousHost = Annotated[
    str, Field(min_length=1, max_length=253), AfterValidator(_validate_rendezvous_host)
]
_IdentityText = Annotated[str, Field(min_length=1, max_length=128)]
_WeightVersion = Annotated[str, Field(min_length=1, max_length=256)]
_ExpertIds = Annotated[tuple[StrictInt, ...], Field(strict=False)]


class AscendGraphQuantization(BaseModel):
    """Declare the physical ModelSlim format independently of computation dtype."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    type: Literal["modelslim-w8a8-dynamic"]
    bits: Literal[8]
    group_size: None
    symmetric: Literal[True]


class AscendGraphModel(BaseModel):
    """Fix geometry and expert computation semantics shared by every rank."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    num_layers: int = Field(gt=0)
    experts_per_layer: int = Field(gt=0)
    hidden_dim: int = Field(gt=0)
    expert_intermediate_dim: int = Field(gt=0)
    top_k: int = Field(gt=0)
    activation_dtype: Literal["bf16"]
    weight_dtype: Literal["bf16"]
    activation: Literal["silu"]
    expert_compute: Literal["swiglu", "deepseek_v4"]
    swiglu_limit: float = Field(ge=0, allow_inf_nan=False)
    quantization: AscendGraphQuantization | None

    @model_validator(mode="after")
    def validate_computation(self) -> AscendGraphModel:
        """Reject impossible routing or inconsistent activation declarations."""
        if self.top_k > self.experts_per_layer:
            raise ValueError("graph top_k cannot exceed experts_per_layer")
        if self.expert_compute == "swiglu" and self.swiglu_limit != 0:
            raise ValueError("graph swiglu_limit requires deepseek_v4 computation")
        return self


class AscendGraphOptions(BaseModel):
    """Hold user graph choices without duplicating model or placement authority."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    mode: Literal["decode", "full"] = "decode"
    capture_sizes: _CaptureSizes = (1, 2, 4, 8, 16, 32, 64)
    timeout_seconds: float = Field(default=120.0, gt=0, le=600, allow_inf_nan=False)
    rendezvous_host: _RendezvousHost | None = None
    rendezvous_port: StrictInt | None = Field(default=None, gt=0, le=65535)

    @model_validator(mode="before")
    @classmethod
    def reject_legacy_document(cls, value: object) -> object:
        """Explain how to migrate documents that formerly controlled placement."""
        if isinstance(value, dict) and _LEGACY_FIELDS.intersection(value):
            raise ValueError(_MIGRATION_ERROR)
        return value

    @classmethod
    def from_file(cls, path: Path) -> AscendGraphOptions:
        """Read graph options with strict unknown-field and legacy-document checks."""
        return cls.model_validate_json(path.read_bytes())


class AscendGraphWorker(BaseModel):
    """Freeze one selected Worker process and its IDs aligned with session layers."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    worker_id: _IdentityText
    expected_start_id: _IdentityText
    control_endpoint: str = Field(min_length=3, max_length=256)
    device: str = Field(pattern=r"^npu:[0-9]+$")
    experts_by_layer: tuple[_ExpertIds, ...] = Field(strict=False)

    @field_validator("control_endpoint")
    @classmethod
    def validate_endpoint(cls, value: str) -> str:
        """Require an independently reachable graph-control host and port."""
        validate_graph_control_endpoint(value)
        return value

    @field_validator("experts_by_layer")
    @classmethod
    def validate_expert_order(
        cls, values: tuple[tuple[int, ...], ...]
    ) -> tuple[tuple[int, ...], ...]:
        """Preserve layer order and reject duplicate, negative or unsorted local IDs."""
        if any(ids != tuple(sorted(set(ids))) or any(i < 0 for i in ids) for ids in values):
            raise ValueError("expert IDs must be nonnegative, sorted and unique per layer")
        if not any(values):
            raise ValueError("a graph participant must own at least one expert")
        return values

    def expert_ids_by_layer(self, layer_ids: tuple[int, ...]) -> dict[int, tuple[int, ...]]:
        """Return a new backend mapping, retaining empty layers and immutable ID tuples."""
        return dict(zip(layer_ids, self.experts_by_layer, strict=True))


class AscendGraphConfig(BaseModel):
    """Freeze resolved facts consumed by every rank for exactly one graph session.

    Construct this object at startup from native model facts and Controller READY
    routes. A dependency change requires process restart and a new session.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    session_id: str = Field(min_length=16, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    instance_id: StrictInt = Field(gt=0, le=(1 << 64) - 1)
    model_name: str = Field(min_length=1, max_length=256)
    weight_version: _WeightVersion
    model: AscendGraphModel
    layer_ids: tuple[StrictInt, ...] = Field(strict=False)
    mode: Literal["decode", "full"]
    capture_sizes: _CaptureSizes
    decode_capture_sizes: _CaptureSizes
    timeout_seconds: float = Field(gt=0, le=600, allow_inf_nan=False)
    rendezvous_host: _RendezvousHost
    rendezvous_port: StrictInt = Field(gt=0, le=65535)
    workers: tuple[AscendGraphWorker, ...] = Field(min_length=1, strict=False)
    source_topology_version: StrictInt = Field(ge=0, le=(1 << 64) - 1)

    @model_validator(mode="after")
    def validate_assignments(self) -> AscendGraphConfig:
        """Require ordered ranks and exact-once coverage of every routed-layer expert."""
        if (
            not self.layer_ids
            or self.layer_ids != tuple(sorted(set(self.layer_ids)))
            or self.layer_ids[0] < 0
            or self.layer_ids[-1] >= self.model.num_layers
        ):
            raise ValueError(
                "layer_ids must be nonempty, sorted, unique and within model.num_layers"
            )
        if self.decode_capture_sizes != self.capture_sizes[: len(self.decode_capture_sizes)]:
            raise ValueError("decode_capture_sizes must be a nonempty prefix of capture_sizes")
        if self.mode == "decode" and self.decode_capture_sizes != self.capture_sizes:
            raise ValueError("decode mode requires decode_capture_sizes to equal capture_sizes")
        worker_ids = tuple(worker.worker_id for worker in self.workers)
        if worker_ids != tuple(sorted(set(worker_ids))):
            raise ValueError("graph Worker IDs must be distinct and sorted in rank order")
        if len({worker.control_endpoint for worker in self.workers}) != len(self.workers):
            raise ValueError("graph Worker control endpoints must be distinct")
        if any(len(worker.experts_by_layer) != len(self.layer_ids) for worker in self.workers):
            raise ValueError("each Worker expert list must align with every session layer")
        for index in range(len(self.layer_ids)):
            seen: set[int] = set()
            for worker in self.workers:
                for expert_id in worker.experts_by_layer[index]:
                    if expert_id >= self.model.experts_per_layer or expert_id in seen:
                        raise ValueError(
                            "graph assignments must cover each expert exactly once per layer"
                        )
                    seen.add(expert_id)
            if len(seen) != self.model.experts_per_layer:
                raise ValueError("graph assignments must cover each expert exactly once per layer")
        return self

    @classmethod
    def from_file(cls, path: Path) -> AscendGraphConfig:
        """Reject the former public deployment-file API with migration instructions."""
        raise ValueError(_MIGRATION_ERROR)

    @property
    def participants(self) -> tuple[AscendGraphWorker, ...]:
        """Return the immutable Controller-selected rank order."""
        return self.workers

    def worker_rank(self, worker_id: str) -> int:
        """Resolve one selected Worker to its stable private HCCL rank."""
        for rank, worker in enumerate(self.workers, 1):
            if worker.worker_id == worker_id:
                return rank
        raise ValueError("Worker is absent from the graph session")

    def identity(self) -> dict[str, object]:
        """Return protocol-v2 session dependencies, excluding topology provenance."""
        return {
            "version": 2,
            **self.model_dump(mode="json", exclude={"source_topology_version"}),
        }

    @property
    def assignment_id(self) -> str:
        """Hash canonical executable/session facts without unrelated topology changes."""
        encoded = json.dumps(
            self.identity(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class GraphExecutionLimits:
    """Keep native allocation budgets distinct from effective scheduling limits."""

    capture_sizes: tuple[int, ...]
    decode_capture_sizes: tuple[int, ...]
    running_requests: int
    graph_stage_tokens: int
    ordinary_prefill_tokens: int


def resolve_graph_limits(
    options: AscendGraphOptions,
    *,
    allocated_requests: int,
    allocated_tokens: int,
    scheduled_tokens: int,
) -> GraphExecutionLimits:
    """Validate actual padded-buffer bounds without changing native allocations."""
    for name, value in (
        ("allocated_requests", allocated_requests),
        ("allocated_tokens", allocated_tokens),
        ("scheduled_tokens", scheduled_tokens),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    captures = options.capture_sizes
    if captures[-1] > allocated_tokens:
        raise ValueError(
            f"graph capture {captures[-1]} exceeds native allocated token buffers "
            f"{allocated_tokens}"
        )
    decode = tuple(capacity for capacity in captures if capacity <= allocated_requests)
    if not decode:
        raise ValueError(
            f"no configured Decode graph fits native allocated request buffers {allocated_requests}"
        )
    if options.mode == "decode" and decode != captures:
        raise ValueError(
            f"Decode-only captures {captures} exceed native allocated request buffers "
            f"{allocated_requests}"
        )
    return GraphExecutionLimits(
        capture_sizes=captures,
        decode_capture_sizes=decode,
        running_requests=min(allocated_requests, decode[-1]),
        graph_stage_tokens=min(scheduled_tokens, captures[-1]),
        ordinary_prefill_tokens=scheduled_tokens,
    )
