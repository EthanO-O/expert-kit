"""Own a metadata-only Controller stream for one fixed graph assignment."""

from __future__ import annotations

import threading
import time

import grpc
from expertkit_proto.ek.control.v2 import lifecycle_pb2, lifecycle_pb2_grpc

from expertkit_transport.controller.messages import (
    RouteDescriptions,
    TopologyMessageAssembler,
    WorkerRoute,
)
from expertkit_transport.transports.ascend_graph.config import AscendGraphConfig, AscendGraphWorker


class GraphTopologySubscription:
    """Freeze READY owners once and stop admission after a selected dependency changes.

    A disconnected or discontinuous stream is terminal until process restart.
    This subscription never opens ordinary Worker transports or changes placement.
    """

    def __init__(
        self,
        controller_endpoint: str,
        *,
        instance_id: int,
        num_layers: int,
        experts_per_layer: int,
    ) -> None:
        self._endpoint = controller_endpoint
        self._instance_id = instance_id
        self._num_layers = num_layers
        self._experts = experts_per_layer
        self._condition = threading.Condition()
        self._routes: RouteDescriptions = {}
        self._version = 0
        self._received = False
        self._selected: dict[tuple[int, int], WorkerRoute] | None = None
        self._workers: tuple[AscendGraphWorker, ...] | None = None
        self._layers: tuple[int, ...] = ()
        self._weight_version: str | None = None
        self._failure: BaseException | None = None
        self._closed = False
        self._channel = None
        self._call = None
        self._thread: threading.Thread | None = None

    def start(
        self,
        *,
        layer_ids: tuple[int, ...],
        max_capture_size: int,
        monotonic_deadline: float,
    ) -> tuple[int, str, tuple[AscendGraphWorker, ...]]:
        """Wait for full READY coverage and validate capacity before any PREPARE.

        The caller owns one absolute discovery deadline, including instance resolution.
        Failure closes unused host resources; no replacement owners are selected.
        """
        if (
            not layer_ids
            or layer_ids != tuple(sorted(set(layer_ids)))
            or layer_ids[0] < 0
            or layer_ids[-1] >= self._num_layers
            or type(max_capture_size) is not int
            or max_capture_size <= 0
        ):
            raise ValueError("graph layers and maximum capture capacity are invalid")
        try:
            with self._condition:
                if self._thread is not None or self._closed:
                    raise RuntimeError("graph topology subscription can only start once")
                self._layers = layer_ids
                self._thread = threading.Thread(
                    target=self._watch, name="ek-graph-topology", daemon=True
                )
                self._thread.start()
                while True:
                    self._require_healthy()
                    remaining = monotonic_deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("deadline waiting for complete graph READY coverage")
                    selected = self._select()
                    if selected is not None:
                        workers, weight_version = self._freeze(selected, max_capture_size)
                        self._selected = selected
                        self._workers = workers
                        self._weight_version = weight_version
                        return self._version, weight_version, workers
                    self._condition.wait(remaining)
        except BaseException:
            self.close()
            raise

    def _select(self) -> dict[tuple[int, int], WorkerRoute] | None:
        if not self._received:
            return None
        selected = {}
        for layer in self._layers:
            for expert in range(self._experts):
                candidates = [
                    route
                    for route in self._routes.get((layer, expert), ())
                    if route.graph_control_endpoint
                ]
                if not candidates:
                    return None
                selected[(layer, expert)] = min(
                    candidates, key=lambda route: route.identity.worker_id
                )
        return selected

    def _freeze(
        self, selected: dict[tuple[int, int], WorkerRoute], max_capture_size: int
    ) -> tuple[tuple[AscendGraphWorker, ...], str]:
        by_worker: dict[str, WorkerRoute] = {}
        versions = set()
        for route in selected.values():
            existing = by_worker.setdefault(route.identity.worker_id, route)
            if existing != route:
                raise ValueError("selected graph Worker has inconsistent process or route metadata")
            if route.max_batch_tokens < max_capture_size:
                raise ValueError(
                    f"selected Worker {route.identity.worker_id} max_batch_tokens "
                    f"{route.max_batch_tokens} cannot hold graph capture {max_capture_size}"
                )
            if not route.weight_version:
                raise ValueError("selected graph READY route has an empty weight_version")
            versions.add(route.weight_version)
        if len(versions) != 1:
            raise ValueError("selected graph READY routes disagree on weight_version")
        workers = tuple(
            AscendGraphWorker(
                worker_id=worker_id,
                expected_start_id=route.identity.start_id,
                control_endpoint=route.graph_control_endpoint,
                device=route.device,
                experts_by_layer=tuple(
                    tuple(
                        expert
                        for expert in range(self._experts)
                        if selected[(layer, expert)].identity.worker_id == worker_id
                    )
                    for layer in self._layers
                ),
            )
            for worker_id, route in sorted(by_worker.items())
        )
        if len({worker.control_endpoint for worker in workers}) != len(workers):
            raise ValueError("selected graph Workers repeat a control endpoint")
        return workers, versions.pop()

    @staticmethod
    def _dependency(route: WorkerRoute) -> tuple:
        return (
            route.identity,
            route.device,
            route.graph_control_endpoint,
            route.weight_version,
            route.max_batch_tokens,
        )

    def _check_selected(self) -> None:
        if self._selected is None:
            return
        for key, selected in self._selected.items():
            if not any(
                self._dependency(route) == self._dependency(selected)
                for route in self._routes.get(key, ())
            ):
                raise RuntimeError("selected graph READY dependency changed; restart required")

    def _require_healthy(self) -> None:
        if self._closed:
            raise RuntimeError("graph topology subscription is closed")
        if self._failure is not None:
            raise RuntimeError(
                "graph topology stream is no longer current; restart required"
            ) from (self._failure)

    def require_current(self, config: AscendGraphConfig) -> None:
        """Check frozen selected dependencies locally before coarse control admission."""
        with self._condition:
            self._require_healthy()
            if (
                config.instance_id != self._instance_id
                or config.layer_ids != self._layers
                or config.workers != self._workers
                or config.weight_version != self._weight_version
            ):
                raise ValueError(
                    "runtime graph configuration does not match frozen READY selection"
                )

    def _watch(self) -> None:
        assembler = TopologyMessageAssembler(
            instance_id=self._instance_id,
            num_layers=self._num_layers,
            experts_per_layer=self._experts,
        )
        try:
            channel = grpc.insecure_channel(self._endpoint)
            with self._condition:
                self._channel = channel
                if self._closed:
                    channel.close()
                    return
                self._call = lifecycle_pb2_grpc.TopologyServiceStub(channel).WatchTopology(
                    lifecycle_pb2.WatchTopologyRequest(instance_id=self._instance_id),
                    wait_for_ready=True,
                )
                call = self._call
            for message in call:
                with self._condition:
                    if self._closed:
                        return
                    try:
                        complete = assembler.consume(
                            message,
                            installed_version=self._version,
                            installed_routes=self._routes,
                        )
                        if complete is not None:
                            self._version, self._routes = complete
                            self._received = True
                            self._check_selected()
                    except BaseException as error:
                        # Publish invalidation before releasing the admission lock.
                        self._failure = error
                        raise
                    finally:
                        self._condition.notify_all()
            raise RuntimeError("graph topology stream disconnected; restart required")
        except BaseException as error:
            with self._condition:
                if not self._closed and self._failure is None:
                    self._failure = error
                self._condition.notify_all()
        finally:
            assembler.discard_incomplete()

    def close(self) -> None:
        """Cancel the stream, close its channel, and join the owned thread boundedly."""
        with self._condition:
            self._closed = True
            call, channel, thread = self._call, self._channel, self._thread
            self._condition.notify_all()
        if call is not None:
            call.cancel()
        if channel is not None:
            channel.close()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
            if thread.is_alive():
                raise RuntimeError("graph topology stream did not stop after cancellation")
