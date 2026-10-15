"""Resolved template context built from cluster and experiment inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, computed_field, model_serializer, model_validator

from expertkit_transport.transports.ascend_graph.config import AscendGraphConfig

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
        """Restrict graph deployment to fixed resident DeepSeek BF16 on one host."""
        graph = self.ascend_decode_graph
        if graph is None:
            return self
        if len(self.attention.config.devices) != 1 or self.expert.total_workers != 1:
            raise ValueError("ascend_decode_graph requires one Frontend and one Worker")
        pool = self.expert.pools[0]
        worker = pool.config.workers[0]
        if pool.config.node != self.attention.config.node:
            raise ValueError("ascend_decode_graph currently requires one host")
        model = self.model.config
        if (
            model.num_layers,
            model.experts_per_layer,
            model.hidden_dim,
            model.intermediate_dim,
            model.topk,
            str(model.weight_dtype),
            str(model.activation_dtype),
        ) != (27, 64, 2048, 1408, 6, "bf16", "bf16"):
            raise ValueError(
                "ascend_decode_graph requires DeepSeek-V2-Lite BF16 geometry"
            )
        if graph.layer_ids != tuple(range(1, 27)):
            raise ValueError(
                "ascend_decode_graph requires routed layers 1..26; layer 0 is dense"
            )
        if (
            graph.worker_id != f"{pool.config.id}-{worker.id}"
            or graph.weight_version != model.weight_version
        ):
            raise ValueError(
                "ascend_decode_graph identity must match Worker and model weights"
            )
        host, _, port_text = graph.control_endpoint.rpartition(":")
        address = str(pool.node.address)
        if host != address or graph.rendezvous_host != address:
            raise ValueError(
                "ascend_decode_graph endpoints must use the Worker host address"
            )
        graph_ports = {int(port_text), graph.rendezvous_port}
        occupied = pool.config.allocated_ports | self.attention.config.allocated_ports
        if self.control.config.node == pool.config.node:
            occupied |= self.control.config.allocated_ports
        if len(graph_ports) != 2 or graph_ports & occupied:
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
        num_attention = len(self.attention.config.devices)
        num_workers = self.expert.total_workers
        placement = f"{num_attention}A{num_workers}E"
        dataset_name = self.dataset.config.name

        max_concurrency = self.run.max_concurrency
        output_len = self.run.output_len
        label = f"{placement}-c{max_concurrency}-{dataset_name}-o{output_len}"

        return label


class ExpertContext(BaseModel):
    """Resolved expert pools and their shared runtime limits."""

    pools: list[NodeBindingContext[PoolConfig]]
    runtime: ExpertRuntimeConfig

    @property
    def total_workers(self) -> int:
        return sum(pool.config.worker_count for pool in self.pools)

    @classmethod
    def resolve(
        cls,
        pools: list[PoolConfig],
        nodes: dict[str, NodeConfig],
        runtime: ExpertRuntimeConfig,
    ) -> Self:
        resolved_pools = [
            NodeBindingContext[PoolConfig].resolve(
                config=pool,
                nodes=nodes,
            )
            for pool in pools
        ]

        return cls(pools=resolved_pools, runtime=runtime)

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
