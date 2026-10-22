"""Resolved template context built from cluster and experiment inputs."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Self

from expertkit_transport.transports.ascend_graph.config import AscendGraphConfig
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
    ascend_decode_graph: AscendGraphConfig | None = None
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
        node = self.attention.config.node
        pool = self.expert.get_pool_by_node(node)

        if pool is None:
            return self

        # Expert and attention are on a same node
        attention_devices = set(self.attention.config.devices)
        pool_devices = set(pool.devices)
        duplicates = attention_devices & pool_devices
        if duplicates:
            raise DuplicateDeviceError(f"Devices {duplicates} have been used.")
        return self

    @model_validator(mode="after")
    def validate_port_placement(self) -> Self:
        def check_duplicate_ports(p1: set[int], p2: set[int]) -> None:
            duplicates = p1 & p2
            if duplicates:
                raise DuplicatePortError(f"Ports {duplicates} have been used.")

        def check_duplicate_ports_with_expert(
            src_node: str,
            src_ports: set[int],
        ) -> None:
            pool = self.expert.get_pool_by_node(src_node)
            if pool is None:
                return

            pool_ports = pool.allocated_ports
            check_duplicate_ports(src_ports, pool_ports)

        attention_node = self.attention.config.node
        attention_ports = self.attention.config.allocated_ports
        check_duplicate_ports_with_expert(attention_node, attention_ports)

        control_node = self.control.config.node
        control_ports = self.control.config.allocated_ports
        check_duplicate_ports_with_expert(control_node, control_ports)

        if attention_node == control_node:
            # Same node, check if attention and control shares any port
            check_duplicate_ports(attention_ports, control_ports)

        return self

    @model_validator(mode="after")
    def validate_decode_graph(self) -> Self:
        """Restrict graph deployment to qualified resident experts on one host."""
        graph = self.ascend_decode_graph
        if graph is None:
            return self
        if len(self.attention.config.devices) != 1 or self.expert.total_workers != len(
            graph.participants
        ):
            raise ValueError(
                "ascend_decode_graph requires one Frontend and matching Workers"
            )
        if len(self.expert.pools) != 1:
            raise ValueError("ascend_decode_graph currently requires one expert pool")
        pool = self.expert.pools[0].config
        if pool.node != self.attention.config.node:
            raise ValueError("ascend_decode_graph currently requires one host")
        devices = [*self.attention.config.devices, *pool.devices]
        if len(set(devices)) != len(devices):
            raise DuplicateDeviceError(
                "ascend_decode_graph requires a distinct physical device for every role"
            )
        model = self.model.config
        if graph.model != model.graph_model():
            raise ValueError(
                "ascend_decode_graph model declaration must match model configuration"
            )
        geometry = (
            model.num_layers,
            model.experts_per_layer,
            model.hidden_dim,
            model.intermediate_dim,
            model.topk,
        )
        if geometry == (27, 64, 2048, 1408, 6) and model.expert_compute == "swiglu":
            routed_layers = tuple(range(1, 27))
        elif geometry == (48, 128, 2048, 768, 8) and model.expert_compute == "swiglu":
            routed_layers = tuple(range(48))
        elif (
            geometry == (43, 256, 4096, 2048, 6)
            and model.expert_compute == "deepseek_v4"
            and model.swiglu_limit == 10.0
            and model.quantization is not None
        ):
            routed_layers = tuple(range(43))
        else:
            raise ValueError(
                "ascend_decode_graph requires qualified V2-Lite, Qwen3 or ModelSlim V4 computation"
            )
        if graph.layer_ids != routed_layers:
            raise ValueError(
                "ascend_decode_graph layer order must match all routed model layers"
            )
        generated_workers = {
            f"{pool.config.id}-{worker.id}": pool
            for pool in self.expert.pools
            for worker in pool.config.workers
        }
        if {
            worker.worker_id for worker in graph.participants
        } != generated_workers.keys() or graph.weight_version != model.weight_version:
            raise ValueError(
                "ascend_decode_graph identity must match Workers and model weights"
            )
        graph_ports = {graph.rendezvous_port}
        occupied = self.attention.config.allocated_ports.copy()
        for pool in self.expert.pools:
            occupied |= pool.config.allocated_ports
        address = str(self.attention.node.address)
        if graph.rendezvous_host != address:
            raise ValueError(
                "ascend_decode_graph endpoints must use the Worker host address"
            )
        for worker in graph.participants:
            host, _, port_text = worker.control_endpoint.rpartition(":")
            if host != address:
                raise ValueError(
                    "ascend_decode_graph endpoints must use the Worker host address"
                )
            graph_ports.add(int(port_text))
        if self.control.config.node == self.attention.config.node:
            occupied |= self.control.config.allocated_ports
        if len(graph_ports) != len(graph.participants) + 1 or graph_ports & occupied:
            raise DuplicatePortError(
                "ascend_decode_graph ports must be distinct and unallocated"
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
