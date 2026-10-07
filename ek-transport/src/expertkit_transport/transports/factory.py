"""Create Frontend connections for Worker Transport types published by Controller."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from expertkit_proto.ek.control.v2 import lifecycle_pb2

from expertkit_transport.transports.base import WorkerEndpointConfig, WorkerTransport
from expertkit_transport.transports.grpc.client import GrpcWorkerTransport
from expertkit_transport.transports.shm.client import ShmWorkerTransport

if TYPE_CHECKING:
    from expertkit_transport.transports.transfer_engine import TransferEngineRuntime


def create_worker_transport(
    *,
    transport_type: int,
    endpoint: str,
    instance_id: int,
    num_layers: int,
    experts_per_layer: int,
    max_batch_tokens: int,
    hidden_dim: int,
    top_k: int,
    dtype: torch.dtype,
    device: torch.device,
    max_in_flight: int,
    worker_start_id: str = "",
    transport_runtime: TransferEngineRuntime | None = None,
) -> WorkerTransport:
    """Create the data connection declared by one validated Worker route."""

    endpoint_config = WorkerEndpointConfig(
        instance_id=instance_id,
        num_layers=num_layers,
        experts_per_layer=experts_per_layer,
        max_batch_tokens=max_batch_tokens,
        hidden_dim=hidden_dim,
        top_k=top_k,
        dtype=dtype,
    )
    if transport_type == lifecycle_pb2.WORKER_TRANSPORT_GRPC:
        return GrpcWorkerTransport(
            endpoint,
            endpoint_config,
            max_in_flight=max_in_flight,
            device=device,
        )
    if transport_type == lifecycle_pb2.WORKER_TRANSPORT_SHM:
        return ShmWorkerTransport(
            endpoint,
            endpoint_config,
            max_in_flight=max_in_flight,
            device=device,
        )
    if transport_type == lifecycle_pb2.WORKER_TRANSPORT_TRANSFER_ENGINE:
        if transport_runtime is None:
            raise ValueError("Ascend transport requires EK_TRANSPORT_CONFIG or an explicit runtime")
        from expertkit_transport.transports.transfer_engine import TransferEngineWorkerTransport

        return TransferEngineWorkerTransport(
            endpoint,
            endpoint_config,
            device=device,
            max_in_flight=max_in_flight,
            runtime=transport_runtime,
            expected_worker_start_id=worker_start_id,
        )
    raise ValueError(f"unsupported Worker transport type: {transport_type}")
