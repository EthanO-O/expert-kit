"""Validate one fixed Frontend/Worker Ascend graph deployment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator


def _validate_endpoint(endpoint: str) -> None:
    host, separator, port = endpoint.rpartition(":")
    if not separator or not host or not port.isdecimal() or not 1 <= int(port) <= 65535:
        raise ValueError("control_endpoint must use host:port")
    if "://" in host or any(c.isspace() for c in host):
        raise ValueError("graph addresses must be hostnames without a URL scheme or whitespace")


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


def _legacy_v2_model() -> AscendGraphModel:
    return AscendGraphModel(
        num_layers=27,
        experts_per_layer=64,
        hidden_dim=2048,
        expert_intermediate_dim=1408,
        top_k=6,
        activation_dtype="bf16",
        weight_dtype="bf16",
        activation="silu",
        expert_compute="swiglu",
        swiglu_limit=0.0,
        quantization=None,
    )


class AscendGraphWorker(BaseModel):
    """Identify one fixed Worker and its contiguous global expert interval."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    worker_id: str = Field(min_length=1, max_length=128)
    control_endpoint: str = Field(min_length=3, max_length=256)
    expert_start: int = Field(ge=0)
    expert_end: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_worker(self) -> AscendGraphWorker:
        """Reject an empty interval or an ambiguous control address."""
        if self.expert_start >= self.expert_end:
            raise ValueError("expert interval must be nonempty")
        _validate_endpoint(self.control_endpoint)
        return self


class AscendGraphConfig(BaseModel):
    """Declare identity, capacities and the model's fixed Worker partition.

    All participants consume the same document. Changing any field requires a new
    session and process restart; graph mode does not perform online migration.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    session_id: str = Field(min_length=16, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    mode: Literal["decode", "full"] = "decode"
    instance_id: int = Field(gt=0)
    worker_id: str | None = Field(default=None, min_length=1, max_length=128)
    weight_version: str = Field(min_length=1, max_length=256)
    model: AscendGraphModel = Field(default_factory=_legacy_v2_model)
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
        boundaries = [0, *(worker.expert_end for worker in participants)]
        qwen = (
            self.model.num_layers,
            self.model.experts_per_layer,
            self.model.hidden_dim,
            self.model.expert_intermediate_dim,
            self.model.top_k,
            self.model.expert_compute,
        ) == (48, 128, 2048, 768, 8, "swiglu") and self.model.quantization is not None
        qwen_eight = qwen and len(participants) == 8
        v4 = (
            self.model.num_layers,
            self.model.experts_per_layer,
            self.model.hidden_dim,
            self.model.expert_intermediate_dim,
            self.model.top_k,
            self.model.expert_compute,
            self.model.swiglu_limit,
        ) == (43, 256, 4096, 2048, 6, "deepseek_v4", 10.0) and self.model.quantization is not None
        if (
            (len(participants) != 8 if v4 else len(participants) not in (1, 2) and not qwen_eight)
            or (
                v4 and any(worker.expert_end - worker.expert_start != 32 for worker in participants)
            )
            or (
                qwen_eight
                and (
                    self.layer_ids != tuple(range(48))
                    or any(worker.expert_end - worker.expert_start != 16 for worker in participants)
                )
            )
            or boundaries[-1] != self.model.experts_per_layer
            or any(
                worker.expert_start != start
                for worker, start in zip(participants, boundaries[:-1], strict=True)
            )
        ):
            raise ValueError(
                "graph Workers must partition all model experts into one or two fixed intervals, "
                "or eight 16-expert Qwen / 32-expert V4 intervals for exact ModelSlim declarations"
            )
        if self.layer_ids[-1] >= self.model.num_layers:
            raise ValueError("graph layer IDs exceed model.num_layers")
        if self.mode == "full":
            if (
                not qwen
                or self.layer_ids != tuple(range(48))
                or tuple((worker.expert_start, worker.expert_end) for worker in participants)
                not in (
                    ((0, 64), (64, 128)),
                    tuple((start, start + 16) for start in range(0, 128, 16)),
                )
            ):
                raise ValueError(
                    "full graph mode requires the complete two- or eight-Worker Qwen3 W8A8 model"
                )
            if max(self.capture_sizes) > 2048:
                raise ValueError("full graph token capacity cannot exceed 2048")
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
                expert_end=self.model.experts_per_layer,
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
