# Expert Kit integration for vLLM

This package replaces vLLM's routed MoE factory with an implementation that
keeps vLLM's router and shared experts local while sending one complete routed
layer to Expert Kit Transport. The PyPI CUDA engine extra targets vLLM `0.25.1`.
The Ascend Decode graph lane uses the official prebuilt vLLM `0.26.0` /
vLLM-Ascend `v0.26.0rc2-a3` image described below.

The ordinary vLLM 0.25.1 lane is runtime-qualified with the official
DeepSeek-V2-Lite-Chat BF16 checkpoint. Ordinary CI keeps the full model run
disabled because it requires a Controller, Weight Server, CUDA Frontend, and a
Worker with enough memory for every routed expert.

## Install

Install the vLLM workspace extra from the repository root:

```bash
uv sync --locked --extra vllm
```

This installs `expertkit-vllm`, its workspace Transport dependency, and vLLM
0.25.1 in the root `.venv`. Ordinary workspace development omits this extra.

For a standalone wheel installation, request `expertkit-vllm[engine]` to install
vLLM `0.25.1`; the base package leaves the engine choice to the deployment.

The ordinary Ascend deployment selects the prebuilt vLLM-Ascend
`v0.26.0rc2-a3` image with vLLM `0.26.0`, Torch `2.10.0`, Torch-NPU
`2.10.0.post4`, and CANN `9.1.0`. Install the base plugin in that image without
the CUDA `engine` extra. The workspace marks CUDA engine and NPU extras as
mutually exclusive.

The Frontend and Worker share an EK image derived from that fixed prebuilt
image. Its four EK Python wheels install with the image's exact dependency
constraints; the `ascend-graph` extra supplies configuration dependencies and
leaves the engine choice to the image. Follow the [Ascend graph guide](../../doc/tutorial/ascend/decode-graphs.md)
for Qwen3/V4 examples, the shared image, configuration and startup order. The ordinary root
workspace environment remains separate.

Tests importing vLLM interfaces require the corresponding engine environment.
Run Ascend integration tests inside the prebuilt Ascend image with this
checkout available and pytest installed. The default workspace does not
install vLLM, so the complete suite cannot run there:

```bash
PYTHONPATH=ek-integration/expertkit_vllm \
  python3 -m pytest ek-integration/expertkit_vllm/tests
```

Building the plugin wheel from the host checkout does not require vLLM:

```bash
uv build --package expertkit-vllm --wheel
```

## Configure

The plugin is disabled unless `EK_ENABLE=1` is set. When enabled, it reads:

- `EK_ADDR`: Controller gRPC endpoint. The default is `localhost:5002`.
- `EK_INSTANCE_ID`: optional positive numeric override. When omitted, Transport
  resolves the Controller's configured default instance.
- `EK_CLIENT_TIMEOUT`: positive timeout in seconds. The default is `6`.

For example:

```bash
export EK_ENABLE=1
export EK_ADDR=controller.internal:5002
export EK_CLIENT_TIMEOUT=6
```

The Controller and Workers must already be running, and the instance's expert
weights must be ready before inference starts. The MVP assumes a trusted,
isolated cluster network and does not provide TLS or application authentication.
Each Worker registers its Transport type and the Controller publishes it in
topology; the plugin does not select a Transport through an environment
variable. Shared memory requires vLLM and the Worker to use the same Host,
`/dev/shm` namespace, and Unix user.

## Current limits

- Routed experts support the unquantized SiLU FFN and declared ModelSlim dynamic W8A8 paths.
- Tensor, expert, sequence, prefill-context parallelism and EPLB are rejected.
- Pipeline and ordinary data-parallel model processes are allowed.
- On the ordinary path, full graph capture is changed to piecewise capture
  because remote MoE calls perform network I/O between graph segments.
- The optional Ascend path preserves native `FULL_DECODE_ONLY`, keeps all routed
  experts resident, and coordinates one control exchange around each complete
  Decode step. Prefill and mixed batches use the ordinary path. Qwen3 uses one
  Frontend and two expert Workers; V4 uses one Frontend and eight Workers. Each
  role requires a distinct device and fixed placement. The V2-Lite profile is
  also accepted; its historical qualification does not qualify the new image.
  See the graph guide for model declarations and target-server validation.
- The integration depends on vLLM internal interfaces and therefore stays
  pinned until a later version is reviewed and adapted.

The plugin preserves vLLM's model router, including grouped or custom routing,
and sends final `int32` expert assignments and FP32 routing weights through
`expertkit-transport`. Transport returns the weighted, aggregated activation
Tensor to the vLLM layer.

## Inference tracing

See [the tracing tutorial](../../doc/tutorial/inference-tracing.md) for the
Ascend generator setting and direct Jaeger inspection. Manual deployments use
`EK_TRACE_ENDPOINT`, `EK_TRACE_SAMPLE_RATIO`, and `--enforce-eager` for the
Frontend, plus Worker `observability.tracing` YAML for the complete trace.
