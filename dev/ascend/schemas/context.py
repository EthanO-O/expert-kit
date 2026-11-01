"""Resolved template context built from cluster and experiment inputs."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Self

from expertkit_transport.ascend_graph import AscendGraphOptions
from expertkit_worker.config import BackendName
from pydantic import BaseModel, computed_field, model_serializer, model_validator

from .cluster import (
    AscendTransportConfig,
    AttentionConfig,
    ClusterConfig,
    ControlConfig,
    ExpertRuntimeConfig,
    ImageConfig,
    Inference,
    NodeConfig,
    PoolConfig,
)
from .experiment import (
    DatasetConfig,
    ExperimentConfig,
    ModelConfig,
    RunConfig,
    ServeConfig,
    TracingConfig,
)


class BackendExecutionMode(StrEnum):
    EAGER = "eager"
    GRAPH = "graph"


class DuplicateDeviceError(ValueError):
    """Raised when frontend and expert workers share a device."""


class DuplicatePortError(ValueError):
    """Raised when two services claim the same host port."""


class TemplateContext(BaseModel):
    """Fully resolved values consumed by deployment templates."""

    project_name: str
    inference: Inference
    images: ImageConfig
    transport: AscendTransportConfig
    attention: NodeBindingContext[AttentionConfig]
    control: NodeBindingContext[ControlConfig]
    expert: ExpertContext
    model: ArtifactContext[ModelConfig]
    dataset: ArtifactContext[DatasetConfig]
    serve: ServeConfig
    run: RunConfig
    tracing: TracingConfig
    ascend_decode_graph: AscendGraphOptions | None = None
    results_path: Path

    @classmethod
    def from_config(cls, cluster: ClusterConfig, experiment: ExperimentConfig) -> Self:
        nodes = cluster.nodes

        attention = NodeBindingContext[AttentionConfig].resolve(
            config=cluster.attention,
            nodes=nodes,
        )
        control = NodeBindingContext[ControlConfig].resolve(
            config=cluster.control,
            nodes=nodes,
        )
        expert = ExpertContext.resolve(
            pools=cluster.pools,
            nodes=nodes,
            runtime=cluster.expert.runtime,
            backend=cluster.expert.backend,
        )
        paths = cluster.paths

        model = ArtifactContext[ModelConfig].resolve(
            experiment.model,
            paths.models,
            kind="model",
        )
        dataset = ArtifactContext[DatasetConfig].resolve(
            experiment.dataset,
            paths.datasets,
            kind="dataset",
        )
        run = experiment.run

        return cls(
            project_name=cluster.project_name,
            inference=cluster.inference,
            images=cluster.images,
            transport=cluster.transport,
            attention=attention,
            control=control,
            expert=expert,
            model=model,
            dataset=dataset,
            serve=experiment.serve,
            run=run,
            tracing=experiment.tracing,
            ascend_decode_graph=experiment.ascend_decode_graph,
            results_path=cluster.paths.results,
        )

    @model_validator(mode="after")
    def validate_device_placement(self) -> Self:
        """Require distinct physical devices on each actual deployment host."""
        occupied: dict[str, set[int]] = {}
        roles = [(self.attention.node, self.attention.config.devices)] + [
            (pool.node, pool.config.devices) for pool in self.expert.pools
        ]
        for node, devices in roles:
            address = str(node.address)
            used = occupied.setdefault(address, set())
            if len(devices) != len(set(devices)) or used.intersection(devices):
                raise DuplicateDeviceError(
                    f"Devices {devices} are already used on {address}"
                )
            used.update(devices)
        return self

    @model_validator(mode="after")
    def validate_port_placement(self) -> Self:
        """Check actual service ports per host without inferring graph placement."""
        occupied: dict[str, set[int]] = {}

        def claim(address: str, ports: list[int]) -> None:
            used = occupied.setdefault(address, set())
            if (
                any(not 1 <= port <= 65535 for port in ports)
                or len(ports) != len(set(ports))
                or used.intersection(ports)
            ):
                raise DuplicatePortError(
                    f"Ports {ports} conflict or are out of range on {address}"
                )
            used.update(ports)

        claim(str(self.attention.node.address), [self.attention.config.vllm_port])
        control = self.control.config
        claim(
            str(self.control.node.address),
            [
                control.postgres_port,
                control.worker_control_port,
                control.frontend_control_port,
                control.weight_server_port,
            ],
        )
        for pool in self.expert.pools:
            ports = []
            for worker in pool.config.workers:
                ports.extend([worker.port, worker.peer_port])
                if self.ascend_decode_graph is not None:
                    if worker.graph_control_port is None:
                        raise ValueError(
                            "graph generation requires pools.start_graph_control_port"
                        )
                    ports.append(worker.graph_control_port)
            claim(str(pool.node.address), ports)
        if self.ascend_decode_graph is not None:
            graph = self.ascend_decode_graph
            claim(graph.rendezvous_host, [graph.rendezvous_port])
        return self

    @model_validator(mode="after")
    def validate_decode_graph(self) -> Self:
        """Check known deployment conflicts while runtime resolves model/READY facts."""
        graph = self.ascend_decode_graph
        if graph is None:
            return self
        if len(self.attention.config.devices) != 1 or self.expert.total_workers == 0:
            raise ValueError(
                "ascend_decode_graph requires one Frontend and at least one Worker"
            )
        if self.serve.enforce_eager or self.tracing.enabled:
            raise ValueError(
                "ascend_decode_graph requires graph execution with tracing disabled"
            )
        if self.expert.runtime.max_active_batches_per_device != 1:
            raise ValueError(
                "ascend_decode_graph requires max_active_batches_per_device=1"
            )
        if max(graph.capture_sizes) > self.expert.runtime.max_batch_tokens:
            raise ValueError(
                "ascend_decode_graph capacity exceeds Worker max_batch_tokens"
            )
        return self

    @computed_field
    @property
    def label(self) -> str:
        backend = self.expert.backend
        backend_execution_mode = (
            BackendExecutionMode.EAGER
            if self.ascend_decode_graph is None
            else BackendExecutionMode.GRAPH
        )

        num_attention = len(self.attention.config.devices)
        num_workers = self.expert.total_workers
        placement = f"{num_attention}A{num_workers}E"
        dataset_name = self.dataset.config.name

        max_concurrency = self.run.max_concurrency
        output_len = self.run.output_len
        label = (
            f"{backend}-{backend_execution_mode}"
            f"-{placement}-c{max_concurrency}"
            f"-{dataset_name}-o{output_len}"
        )

        return label


class ExpertContext(BaseModel):
    """Resolved expert pools and their shared runtime limits."""

    pools: list[NodeBindingContext[PoolConfig]]
    runtime: ExpertRuntimeConfig
    backend: BackendName

    @property
    def total_workers(self) -> int:
        return sum(pool.config.worker_count for pool in self.pools)

    @classmethod
    def resolve(
        cls,
        pools: list[PoolConfig],
        nodes: dict[str, NodeConfig],
        runtime: ExpertRuntimeConfig,
        backend: BackendName,
    ) -> Self:
        resolved_pools = [
            NodeBindingContext[PoolConfig].resolve(
                config=pool,
                nodes=nodes,
            )
            for pool in pools
        ]

        return cls(
            pools=resolved_pools,
            runtime=runtime,
            backend=backend,
        )

    def get_pool_by_node(self, node: str) -> PoolConfig | None:
        for pool in self.pools:
            if pool.config.node != node:
                continue
            return pool.config
        return None


class NodeBindingContext[T: AttentionConfig | ControlConfig | PoolConfig](BaseModel):
    """Configuration object bound to a named deployment node."""

    node: NodeConfig
    config: T

    @classmethod
    def resolve(
        cls,
        config: T,
        nodes: dict[str, NodeConfig],
    ) -> Self:
        node = resolve_ref(config.node, nodes, kind="node")
        return cls(node=node, config=config)

    @model_serializer
    def serialize(self) -> dict[str, Any]:
        return {
            **self.config.model_dump(mode="json"),
            **self.node.model_dump(mode="json"),
        }


class ArtifactContext[T: ModelConfig | DatasetConfig](BaseModel):
    """Experiment artifact resolved through the cluster path registry."""

    config: T
    path: Path | None = None

    @classmethod
    def resolve(
        cls,
        config: T,
        paths: dict[str, Path],
        *,
        kind: str,
    ) -> Self:
        path_ref = config.path_ref
        if path_ref is None:
            return cls(config=config)

        path = resolve_ref(path_ref, paths, kind=kind)
        return cls(path=path, config=config)

    @model_serializer
    def serialize(self) -> dict[str, Any]:
        return {
            **self.config.model_dump(mode="json"),
            "path": self.path,
        }


def resolve_ref[T](ref: str, registry: dict[str, T], *, kind: str) -> T:
    if ref not in registry:
        raise KeyError(f"{kind} ref {ref} not found in registered list.")

    return registry[ref]
