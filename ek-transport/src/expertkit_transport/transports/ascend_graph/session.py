"""Address-stable HCCL buffers used inside a fixed one/two-Worker device graph."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as distributed

from expertkit_transport.quarantine import retain_until_process_exit
from expertkit_transport.transports.ascend_graph.config import AscendGraphConfig


@dataclass(frozen=True, slots=True)
class GraphLayerBuffers:
    """Own BF16 activations/output, int32 IDs and FP32 routing at fixed addresses."""

    hidden: torch.Tensor
    expert_ids: torch.Tensor
    routing_weights: torch.Tensor
    output: torch.Tensor
    partial_output: torch.Tensor | None = None


class HcclGraphSession:
    """Own an isolated HCCL process group without changing vLLM's default group."""

    def __init__(
        self,
        config: AscendGraphConfig,
        *,
        rank: int,
        device: torch.device,
        hidden_dim: int,
        top_k: int,
    ) -> None:
        world_size = len(config.participants) + 1
        if not 0 <= rank < world_size or device.type != "npu":
            raise ValueError("Ascend graph session requires a configured rank and an indexed NPU")
        self.config = config
        self.rank = rank
        self.device = device
        self._hidden_dim = hidden_dim
        self._top_k = top_k
        timeout = timedelta(seconds=config.timeout_seconds)
        # Worker owns the store listener; it can acknowledge preparation before
        # the Frontend connects without touching the default vLLM rendezvous.
        self._store = distributed.TCPStore(
            config.rendezvous_host,
            config.rendezvous_port,
            world_size,
            is_master=rank == 1,
            timeout=timeout,
            wait_for_workers=False,
        )
        self._prefix_store = distributed.PrefixStore(config.session_id, self._store)
        from torch_npu._C._distributed_c10d import ProcessGroupHCCL

        options = ProcessGroupHCCL.Options()
        options._timeout = timeout
        self._group = ProcessGroupHCCL(self._prefix_store, rank, world_size, options)
        self._buffers: dict[int, GraphLayerBuffers] = {}

    def buffers(self, capacity: int) -> GraphLayerBuffers:
        """Allocate once per qualified profile, before its graph is captured."""
        if capacity not in self.config.capture_sizes:
            raise ValueError("capacity is not a configured graph profile")
        if capacity not in self._buffers:
            self._buffers[capacity] = GraphLayerBuffers(
                torch.zeros(capacity, self._hidden_dim, dtype=torch.bfloat16, device=self.device),
                torch.zeros(capacity, self._top_k, dtype=torch.int32, device=self.device),
                torch.zeros(capacity, self._top_k, dtype=torch.float32, device=self.device),
                torch.empty(capacity, self._hidden_dim, dtype=torch.bfloat16, device=self.device),
                torch.empty(capacity, self._hidden_dim, dtype=torch.float32, device=self.device)
                if len(self.config.participants) == 2
                else None,
            )
        return self._buffers[capacity]

    def broadcast(self, tensor: torch.Tensor, source: int) -> None:
        """Record one ordered HCCL broadcast on the current NPU stream."""
        options = distributed.BroadcastOptions()
        options.rootRank = source
        options.rootTensor = 0
        self._group.broadcast([tensor], options).wait()

    def receive_inputs(self, buffers: GraphLayerBuffers) -> None:
        """Match every Frontend layer's hidden/ID/weight communication order."""
        self.broadcast(buffers.hidden, 0)
        self.broadcast(buffers.expert_ids, 0)
        self.broadcast(buffers.routing_weights, 0)

    def exchange(
        self, hidden: torch.Tensor, expert_ids: torch.Tensor, routing_weights: torch.Tensor
    ) -> torch.Tensor:
        """Copy dynamic routed data and return caller-consumed prepared output."""
        buffers = self.buffers(hidden.shape[0])
        buffers.hidden.copy_(hidden)
        buffers.expert_ids.copy_(expert_ids)
        buffers.routing_weights.copy_(routing_weights)
        self.receive_inputs(buffers)
        self.combine(buffers)
        return buffers.output

    def combine(self, buffers: GraphLayerBuffers) -> None:
        """Reduce FP32 shard sums and cast once at the Frontend, inside capture."""
        partial = buffers.partial_output
        if partial is None:
            self.broadcast(buffers.output, 1)
            return
        if self.rank == 0:
            # This write must replay for every layer, not just initialize storage.
            partial.zero_()
        options = distributed.ReduceOptions()
        options.rootRank = 0
        options.rootTensor = 0
        options.reduceOp = distributed.ReduceOp.SUM
        self._group.reduce([partial], options).wait()
        if self.rank == 0:
            buffers.output.copy_(partial)

    def quarantine(self) -> None:
        """Keep every process-group and buffer reference until process termination."""
        retain_until_process_exit(self)

    def close(self) -> None:
        """Release the isolated group only after both peers prove completion.

        Owners must destroy every graph referencing these buffers first. A peer
        failure instead quarantines the session and terminates its process.
        """
        self._buffers.clear()
        self._group = None
        self._prefix_store = None
        self._store = None
