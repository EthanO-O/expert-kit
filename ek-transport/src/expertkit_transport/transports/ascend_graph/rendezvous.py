"""Retain one session-owned TCPStore listener before device-group preparation."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import timedelta
from typing import Annotated, Any, Literal

import torch.distributed as distributed
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from expertkit_transport.transports.ascend_graph.config import (
    AscendGraphConfig,
    AscendGraphWorker,
    _IdentityText,
    _RendezvousHost,
    _WeightVersion,
)


def reservation_digest(facts: dict[str, Any]) -> str:
    """Bind final session dependencies to the requested, possibly zero, port."""
    identity = {"version": 2, **{k: v for k, v in facts.items() if k != "source_topology_version"}}
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


class GraphRendezvousRequest(BaseModel):
    """Describe one selected owner's bounded reservation without a draft config."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: StrictInt = Field(ge=1, le=1)
    action: Literal["reserve"]
    reservation_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    session_id: str = Field(min_length=16, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    worker_start_id: _IdentityText
    instance_id: StrictInt = Field(gt=0, le=(1 << 64) - 1)
    weight_version: _WeightVersion
    worker: AscendGraphWorker
    layer_ids: Annotated[tuple[StrictInt, ...], Field(strict=False)]
    world_size: StrictInt = Field(ge=2)
    timeout_seconds: float = Field(gt=0, le=600, allow_inf_nan=False)
    rendezvous_host: _RendezvousHost
    rendezvous_port: StrictInt = Field(ge=0, le=65535)

    @model_validator(mode="after")
    def validate_owner(self) -> GraphRendezvousRequest:
        """Keep the selected assignment aligned with a nonempty sorted layer list."""
        if (
            not self.layer_ids
            or self.layer_ids != tuple(sorted(set(self.layer_ids)))
            or self.layer_ids[0] < 0
            or len(self.worker.experts_by_layer) != len(self.layer_ids)
            or self.worker.expected_start_id != self.worker_start_id
        ):
            raise ValueError("invalid graph rendezvous owner or layers")
        return self

    @classmethod
    def from_facts(cls, facts: dict[str, Any]) -> GraphRendezvousRequest:
        """Build a reservation from resolved facts before a final port exists."""
        worker = facts["workers"][0]
        return cls.model_validate(
            {
                "version": 1,
                "action": "reserve",
                "reservation_id": reservation_digest(facts),
                "session_id": facts["session_id"],
                "worker_start_id": worker["expected_start_id"],
                "instance_id": facts["instance_id"],
                "weight_version": facts["weight_version"],
                "worker": worker,
                "layer_ids": facts["layer_ids"],
                "world_size": len(facts["workers"]) + 1,
                "timeout_seconds": facts["timeout_seconds"],
                "rendezvous_host": facts["rendezvous_host"],
                "rendezvous_port": facts["rendezvous_port"],
            }
        )

    def release_request(self) -> dict[str, object]:
        """Return ownership metadata known before the reserve reply arrives."""
        return {
            "version": 1,
            "action": "release",
            "reservation_id": self.reservation_id,
            "session_id": self.session_id,
            "worker_start_id": self.worker_start_id,
        }


class OwnedGraphRendezvous:
    """Hold an exclusive live listener until unused release or proven group teardown.

    The Worker serializes access and owns assignment guards and expiry supervision.
    An adopted store must remain retained until all referencing device work ends.
    """

    def __init__(
        self,
        request: GraphRendezvousRequest,
        *,
        monotonic_deadline: float,
        bind_timeout_seconds: float | None = None,
    ) -> None:
        self.request = request
        self.deadline = monotonic_deadline
        self.adopted = False
        remaining = monotonic_deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("graph rendezvous deadline expired before binding")
        self.store = distributed.TCPStore(
            request.rendezvous_host,
            request.rendezvous_port,
            request.world_size,
            is_master=True,
            timeout=timedelta(seconds=min(remaining, bind_timeout_seconds or remaining)),
            wait_for_workers=False,
        )
        self.port = self.store.port
        if time.monotonic() >= monotonic_deadline:
            self.close()
            raise TimeoutError("graph rendezvous deadline expired while binding")

    def reply(self) -> dict[str, object]:
        """Return the endpoint of the retained store, never an unbound candidate."""
        return {
            "version": 1,
            "state": "listening",
            "reservation_id": self.request.reservation_id,
            "session_id": self.request.session_id,
            "worker_start_id": self.request.worker_start_id,
            "rendezvous_host": self.request.rendezvous_host,
            "rendezvous_port": self.port,
        }

    def validate_config(self, config: AscendGraphConfig) -> None:
        """Reject a final snapshot that cannot adopt this exact owner's listener."""
        facts = config.model_dump(mode="json")
        facts["rendezvous_port"] = self.request.rendezvous_port
        if (
            self.store is None
            or self.adopted
            or time.monotonic() >= self.deadline
            or config.session_id != self.request.session_id
            or config.participants[0] != self.request.worker
            or len(config.participants) + 1 != self.request.world_size
            or config.rendezvous_host != self.request.rendezvous_host
            or config.rendezvous_port != self.port
            or reservation_digest(facts) != self.request.reservation_id
        ):
            raise ValueError("graph PREPARE does not match its live rendezvous reservation")

    def close(self) -> None:
        """Drop the listener only when the owner proves it is no longer in use."""
        self.store = None
