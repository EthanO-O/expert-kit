"""vLLM general plugin registration for remote Routed-MoE execution."""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)


def register() -> None:
    """Wrap the factory of an explicitly supported vLLM runtime when enabled."""

    if os.getenv("EK_ENABLE") != "1":
        return

    from vllm.platforms import current_platform

    if current_platform.device_type == "npu":
        # General-plugin ordering is not a platform contract. Force the public
        # Ascend pre-registration hook to install its FusedMoE wrapper before
        # Expert Kit captures the active factory.
        current_platform.pre_register_and_update()

    from expertkit_vllm.tracing import install_forward_tracing

    install_forward_tracing()

    import vllm.model_executor.layers.fused_moe as fused_moe_package
    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    from expertkit_vllm.experts.remote_moe import (
        is_expertkit_fused_moe_factory,
        wrap_fused_moe_factory,
    )

    factory_name = "FusedMoEFactory" if hasattr(fused_moe_layer, "FusedMoEFactory") else "FusedMoE"
    platform_factory = getattr(fused_moe_layer, factory_name)
    if not is_expertkit_fused_moe_factory(platform_factory):
        remote_factory = wrap_fused_moe_factory(platform_factory)
        # These are deliberately mutable module exports: model modules import the
        # package binding, while the platform wrapper owns the layer binding.
        fused_moe_layer.__dict__[factory_name] = remote_factory
        fused_moe_package.__dict__[factory_name] = remote_factory
        for name, module in tuple(sys.modules.items()):
            if (
                name.startswith(("vllm.model_executor.models.", "vllm.models."))
                and module is not None
                and module.__dict__.get(factory_name) is platform_factory
            ):
                module.__dict__[factory_name] = remote_factory
        logger.info(
            f"enabled Expert Kit remote MoE using {factory_name} on {current_platform.device_type}"
        )

    from expertkit_vllm.decode_graph import (
        graph_config_from_environment,
        install_decode_graph_wrapper,
    )

    graph_config = graph_config_from_environment()
    if graph_config is not None:
        if current_platform.device_type != "npu":
            raise ValueError("EK_ASCEND_DECODE_GRAPH_CONFIG requires the Ascend platform")
        install_decode_graph_wrapper(graph_config)

    from expertkit_vllm.dummy_batch import install_dummy_batch_guard

    # Worker imports must see the remote factory exports above.
    install_dummy_batch_guard(current_platform.device_type)
