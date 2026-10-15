"""Validate one fixed Frontend/Worker Ascend graph deployment."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator


class AscendGraphConfig(BaseModel):
    """Declare the identity, graph capacities and rendezvous of one 1A1F group.

    Both processes consume the same document. Changing any field requires a new
    session and process restart; graph mode does not perform online migration.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    session_id: str = Field(min_length=16, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    instance_id: int = Field(gt=0)
    worker_id: str = Field(min_length=1, max_length=128)
    weight_version: str = Field(min_length=1, max_length=256)
    control_endpoint: str = Field(min_length=3, max_length=256)
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
        host, separator, port = self.control_endpoint.rpartition(":")
        if not separator or not host or not port.isdecimal() or not 1 <= int(port) <= 65535:
            raise ValueError("control_endpoint must use host:port")
        if "://" in host or any(c.isspace() for c in host + self.rendezvous_host):
            raise ValueError("graph addresses must be hostnames without a URL scheme or whitespace")
        return self

    @classmethod
    def from_file(cls, path: Path) -> AscendGraphConfig:
        """Read the shared JSON deployment document with strict unknown-field checks."""
        return cls.model_validate_json(path.read_bytes())

    def identity(self) -> dict[str, object]:
        """Return the shared deployment identity for coarse-step control messages."""
        return json.loads(self.model_dump_json())
