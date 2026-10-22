# Expert Kit Python Worker

`expertkit-worker` runs routed expert FFNs for one model instance on one compute
device. It receives bounded layer batches through gRPC or the experimental
same-host shared-memory data path, executes only experts that the Controller has
made ready, and returns one weighted partial output per batch.

## Install

Create the default locked Torch environment from the repository root:

```bash
uv sync --locked
```

This creates the repository root `.venv` and installs Proto, Transport, Worker,
and the Torch integration. Run all commands below from the repository root.

The optional CPU-only GGML environment is installed with:

```bash
uv sync --locked --extra ggml
```

The experimental NVIDIA fused environment is installed with:

```bash
uv sync --locked --extra fused
```

GGML remains experimental and CPU-only. The fused Backend supports the current
unquantized FP16 and BF16 SiLU path. Torch is the default Backend.

## Configuration

Start from
[`examples/qwen3-30b-a3b.torch.yaml`](./examples/qwen3-30b-a3b.torch.yaml).
The file is loaded once at startup and unknown fields are rejected. Worker
fields do not have environment-variable overrides.

Important settings:

- `model.instance_id` is optional. When omitted, startup resolves the instance
  named by the Controller configuration. An explicit ID must match that default.
- `model.name` must match the model name served by the Weight Server.
- `worker.id` must match the node name assigned by the Controller.
- `worker.device` is one explicit `cuda:<id>` for Torch. Start another process
  for another device.
- `worker.device_memory_limit` is the complete device budget for this process.
  Startup rejects a budget larger than the device or current available memory.
- `worker.max_batch_tokens` defaults to `4096`.
- `worker.max_active_batches_per_device` defaults to `1`.
- `worker.ggml.cpu_threads` is required when `worker.backend` is `ggml`.
- `transport.max_pending_batches_per_device` is the number of decoded requests
  allowed to wait outside the fixed execution slots. It defaults to
  `worker.max_active_batches_per_device` when omitted.
- `weight_manager.max_concurrent_loads` defaults to `64`.
- The Host weight budget defaults to enough bytes for the model's complete expert
  cache plus concurrent conversion capacity. An explicit
  `weight_manager.dram_cache.max_bytes` includes both; startup subtracts
  `max_concurrent_loads * adapter.host_conversion_temporary_bytes()` before sizing
  the LRU cache. See [quantization implementation](./QUANTIZATION.md) for details.
- Disk-cache writeback defaults to enabled. A remote weight is validated before
  it is published in the cache.
- Heartbeats default to every 3 seconds with a 10-second Controller timeout.
- Expert state changes are sent in groups of at most 64 or after 50 ms. Heartbeat
  and expert state reporting use separate control streams.

For a gRPC Tensor Worker, configure:

```yaml
transport:
  type: grpc
  max_pending_batches_per_device: 1
  listen: 0.0.0.0:51051
  advertise: worker-a100:51051
```

For a same-host shared-memory Worker, configure:

```yaml
transport:
  type: shm
  max_pending_batches_per_device: 1
  rpc_listen: 0.0.0.0:51051
  rpc_advertise: worker-a100:51051
  shared_memory_dir: /dev/shm
```

`advertise` or `rpc_advertise`, together with
`weight_manager.peer.advertise`, must be reachable from the relevant
processes. Their listen counterparts select local bind addresses. In SHM mode,
the RPC endpoint handles only session setup and small notifications; Tensor
payloads use `/dev/shm`. The Frontend and Worker must see the same shared-memory
namespace and run as the same Unix user.

The Weight Manager looks for a requested assigned expert in this order:

```text
DRAM cache -> disk cache -> Controller-provided peers -> Weight Server
```

A computation request never starts a weight load. The Controller sends placement
commands, and the Worker reports an expert as ready only after its final Backend
weight is usable on the configured device.

## Start

Run the Worker directly:

```bash
uv run --package expertkit-worker ek-worker --config /absolute/path/to/worker.yaml
```

The configuration path may instead be selected with `EK_CONFIG`. An explicit
`--config` takes precedence:

```bash
EK_CONFIG=/absolute/path/to/worker.yaml \
  uv run --package expertkit-worker ek-worker
```

Or use the unified launcher, which replaces itself with the same Python process:

```bash
uv run --package expertkit-worker \
  target/release/ek-cli --config /absolute/path/to/worker.yaml worker
```

The unified launcher accepts the same environment-based selection:

```bash
EK_CONFIG=/absolute/path/to/worker.yaml \
  uv run --package expertkit-worker target/release/ek-cli worker
```

Sending `SIGTERM` starts the Controller-coordinated shutdown. The Worker keeps
serving the published topology until replacements are ready, then stops new
admission, finishes already accepted work, flushes state, and exits within
`worker.shutdown_grace_secs`.

## Logging and observability

The default console output is human-readable and matches the Rust services'
`<LEVEL>(timestamp) message` layout. Set `logging.format: json` when a log
collector requires structured JSON.

Prometheus and OpenTelemetry require the optional dependencies:

```bash
uv sync --locked --extra observability
```

Both are disabled by default. Enable them in the Worker YAML:

```yaml
observability:
  prometheus:
    enabled: true
    listen: 127.0.0.1:9091
  tracing:
    enabled: true
    endpoint: http://127.0.0.1:4317
    sample_ratio: 0.01
```

Prometheus serves `/metrics`. The OpenTelemetry exporter uses asynchronous,
sampled plaintext OTLP over gRPC.

For each sampled computation call, the automatic gRPC server span contains
Worker child spans for request decoding, waiting for an execution slot, active
batch execution, input preparation, Backend submission and completion, output
preparation, device completion waiting, and response encoding. CUDA execution
adds the following attributes to `worker.batch.execute`:

- `expertkit.cuda.input_stage_ms`
- `expertkit.cuda.backend_stage_ms`
- `expertkit.cuda.output_stage_ms`
- `expertkit.cuda.total_stage_ms`

These values use CUDA Events on the Worker's existing stream and are read only
after the response path's existing completion wait. Tracing does not add a CUDA
synchronization. The stage values include any stream idle time between their
recorded boundaries, so they describe Worker stream stages rather than pure
copy-engine or kernel-only time. Unsampled requests skip custom spans and CUDA
timing. An incoming standard `traceparent` remains on the Host and parents the
Worker spans; it is not part of the computation payload or any Tensor.

## Transport and security limits

The gRPC path serializes Tensor bytes through Host memory. The SHM path avoids
protobuf Tensor payloads and loopback Tensor copies, but CUDA inputs and outputs
still pass through pinned Host memory. Neither path is GPU Direct. RDMA, NCCL,
NVSHMEM, Arrow Flight, and Mooncake are not implemented by this Worker.

There is no TLS, mTLS, authentication, or authorization. Run all Controller,
Worker, Weight Manager, Weight Server, metrics, and tracing endpoints only on a
trusted isolated network protected by network-level rules.

## Tests

```bash
uv run --package expertkit-worker ruff check ek-worker/src ek-worker/tests
uv run --package expertkit-worker ruff format --check ek-worker/src ek-worker/tests
uv run --package expertkit-worker pytest ek-worker/tests
```

CUDA, direct-I/O, and real multi-process checks require the matching hardware or
filesystem and are marked separately.

## Quantized routed experts

### Ascend ModelSlim grouped Prefill

The Torch backend groups ordinary multi-token Qwen3 ModelSlim W8A8 batches on
`Ascend910_9362` with BF16 activations, hidden size 2048, intermediate size 768,
and top-8 routing. It packs routes once and executes separate gate, up, and down
INT32 grouped projections using the existing ready weights. Rescaling and ordered
routing accumulation use FP32, with BF16 projection and output boundaries.
Native per-expert projections can use a different FP32 multiplication order, so
generated tokens need not match bitwise between these implementations.

This path uses the shared Ascend image's Triton-Ascend installation and warms its
helpers before Worker registration. If Triton is absent, ordinary execution stays
available; other initialization errors propagate. Single-token execution and
other model profiles retain their existing computation paths.

### Ascend ModelSlim resident Decode graphs

Both `worker.backend: torch` and `worker.backend: ascend_fused` implement the
same resident graph interface. Their computation implementations are independent.
The Ascend fused provider retains packed W13/W2 weights, maps contiguous global
expert intervals by subtraction and masking, and captures the same
`run_ascend_fused_moe()` pipeline as eager execution. Nonlocal routes use the
inactive expert sentinel and zero routing weight. ModelSlim scales are converted
to the native scaled-GMM dtype once before capture; they need not round-trip
losslessly through BF16. FP32 transport carries the BF16 native partial result.
The fused provider does not use the Torch graph's INT32 rescaling or optional
Triton reduction. Both providers share profile validation and safe lease ownership.
Torch's original V4 computation is retained: it applies routing to the FP32
activation before down projection. Ascend fused instead applies routing during
native output combination. Compare numerical accuracy against the reference,
not bitwise equality between the two pipelines.
Graph execution requires one active BF16 NPU slot and fixed placement.
CPU operator tests compare resident fused output with the eager fused pipeline.
Native capture/replay and checkpoint output accuracy still require qualification
on the target NPU runtime. To check one capture against eager execution with
changing routes, duplicates and zero local assignments, run on an NPU host:

```bash
uv run --no-sync pytest -q ek-worker/tests/unit/backends/ascend_fused/test_decode_graph.py \
  -k native_capture
```

The Torch ModelSlim Qwen resident path is enabled explicitly through
`worker.ascend_decode_graph` with the `modelslim-w8a8-dynamic` recipe. Its fused
ordered reduction requires the optional Triton-Ascend package with an Ascend
backend. Install a build compatible with the existing torch-npu and CANN stack
before enabling this path; the Worker does not install or upgrade it at startup.
The qualified stack used torch 2.10.0, torch-npu 2.10.0.post4, CANN 9.1.0, and
triton-ascend 3.2.2 (Triton module version 3.5.0).

Missing Triton or its Ascend backend causes an explicit startup error for this
path. Kernel compilation and warmup happen before graph capture; failures retain
the resident weights in quarantine until process exit. Other ordinary ModelSlim
profiles, BF16 resident graphs, and CPU/CUDA imports do not load these kernels.

The Torch Qwen graph computation supports 128 experts, hidden size 2048, expert
intermediate size 768, top-8 routing, SwiGLU, and BF16 activations. Each
Worker owns one contiguous 64-expert shard, either `[0, 64)` or `[64, 128)`. Each gate, up, and
down projection produces BF16 values. Routing weights multiply the
down-projection results in FP32, and the local sum follows expert ID order and
route-column order before
rounding to BF16. An FP32 transport destination carries that BF16-rounded local
result promoted to FP32.

Full-model resident Decode graphs also require the matching Frontend and
Transport graph-session integration. Enable the same session and capture
capacities across the Frontend and both Workers; enabling the Worker option
alone does not establish a full-model graph session.

Torch resident ModelSlim weight scales must be finite and positive, and an FP32 to
BF16 to FP32 round trip must preserve each scale exactly. Construction rejects
scales that fail this requirement.

### Checkpoint recipes

Weight Server reads checkpoint `config.json` and publishes the **routed expert**
recipe through `/meta/model/{model}`. The Worker selects an adapter at startup and
validates the tensors against that recipe. Original V4 declares global FP8 but
stores routed experts in FP4; selecting from the global `quant_method` alone is
incorrect. Quantization also needs its activation scheme and scale granularity.

Use [`examples/deepseek-v4-flash.torch.yaml`](./examples/deepseek-v4-flash.torch.yaml)
for either V4 checkpoint, changing `model.name` to the Weight Server directory
name. Keep `auto_model_metadata: true` and `metadata_required: true`; no manual
quantization entry is needed. Dimensions and routing settings still belong in
Worker configuration and are checked against the server. Change `weight_version`
and the disk-cache directory when replacing a checkpoint.

| Checkpoint format | Torch execution | Ready GPU weights |
| --- | --- | --- |
| Original DeepSeek-V4-Flash, packed E2M1 FP4 + E8M0 scales | A100 compatibility path: decode at placement, emulate block FP8 activation rounding, use floating GEMMs | BF16 with the example configuration |
| `sgl-npu/DeepSeek-V4-Flash-W8A8`, compressed-tensors `int-quantized` | Dynamic per-token symmetric INT8 activation quantization, INT8 GEMM with INT32 accumulation, per-channel rescaling | INT8 matrices + FP32 scales |
| Symmetric canonical AutoGPTQ v1 INT4 | Decode at placement, floating GEMMs | FP16/BF16 |

The W8A8 adapter accepts per-channel `weight_scale` shaped `[output, 1]` in
FP16/BF16/FP32, and optional INT8 zero-valued `weight_zero_point`. It rejects
static or asymmetric activations, grouped weights, transforms, regex targets,
partially ignored routed experts, and unexpected auxiliary tensors. The name
“W8A8” alone does not establish checkpoint compatibility. Older `blockwise_int8`
exports, QuaRot transforms, and Ascend execution require their own validated
format or backend support.

Both V4 paths apply the configured SwiGLU clipping, calculate the intermediate
activation in FP32, and apply routing weights **before** the down projection.
The FP4 compatibility path uses different GEMM accumulation from the native V4
kernels; it is not a claim of bitwise equality or native FP4 acceleration. Its
expanded weights do not save GPU weight memory. W8A8 keeps weights compressed,
but the eager quantization and scaling operations can cost more time for small
batches; no end-to-end speedup has been established.

These changes support **routed expert computation**. The current Frontend
loader does not implement the complete V4 model, including its attention,
routing, shared experts, and hyper-connections. Full-model generation requires
that integration and separate accuracy and performance validation. A100 tests
use synthetic expert tensors with the official 4096/2048 dimensions; no V4
checkpoint weights are downloaded for these tests.

Run CPU tests and the CUDA qualification from the repository root:

```bash
uv run pytest ek-worker/tests/unit
uv run pytest ek-worker/tests/integration/weights/test_v4_quantized_cuda.py -v
cargo test -p ek-db --lib
```

CUDA qualification must report two passed tests, not skips. The metadata and
layout fixtures were taken from the [original V4 configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json)
and [the selected W8A8 export](https://modelscope.cn/models/sgl-npu/DeepSeek-V4-Flash-W8A8).
Expert arithmetic follows the [official inference implementation](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/inference/model.py).


### Compare resident graph computation

`benchmarks/bench_backend.py --execution-mode graph` compares `torch` and
`ascend_fused` through their actual resident providers. It captures one graph
for the selected token capacity and copies each routing case into fixed buffers
before replay. `execution.operator_path` records the pipeline used by each backend.
It measures one layer's local contribution, without HCCL, RPC or
vLLM. Use the same seed, geometry and expert interval for both backends, in
separate processes. All local experts are loaded and leased before capture.

For the Qwen3 ModelSlim 64-expert shard, run from the repository root on the NPU:

```bash
for backend in torch ascend_fused; do
  uv run --no-sync python ek-worker/benchmarks/bench_backend.py \
    --backend "$backend" --execution-mode graph --device npu:0 --dtype bf16 \
    --quantization modelslim-w8a8-dynamic \
    --tokens 1 --hidden-dim 2048 --intermediate-dim 768 --top-k 8 --experts 128 \
    --expert-start 0 --expert-end 64 --distribution uniform --routing-cases 8 \
    --warmup 20 --repeats 200 --output "results/${backend}-graph.json"
done
```

Repeat with token capacities 1/4/16/64. `device_replay_latency_ms` uses NPU
stream events around graph replay and excludes input staging. `latency_ms`
measures Host elapsed time including input copies, replay dispatch, events and
completion wait; throughput uses that Host interval. Capture and compilation
are outside both intervals. Every routing case is checked before timing unless
`--skip-correctness` is explicit. Graph input IDs must be valid global expert
IDs; eager replay files containing `-1` cannot be used in graph mode.

`--expert-start` and `--expert-end` are graph-only benchmark options. They select
a resident shard while routing still ranges over `--experts` global experts.
The eager backend supports sharded deployment, but this microbenchmark does not
currently expose eager shard filtering. Removing these flags measures all selected
experts; changing `--experts` to 32 changes the routing workload and cannot be
compared directly with a 32-expert shard of a 256-expert model.

For the V4 ModelSlim profile, compare both backends at capacities 1/4/16/64 with
the same global expert count, shard and synthetic routing cases:

```bash
mkdir -p results
for backend in torch ascend_fused; do
  for tokens in 1 4 16 64; do
    uv run --no-sync python ek-worker/benchmarks/bench_backend.py \
      --backend "$backend" --execution-mode graph --device npu:0 --dtype bf16 \
      --quantization modelslim-w8a8-dynamic \
      --expert-compute deepseek_v4 --swiglu-limit 10 \
      --tokens "$tokens" --hidden-dim 4096 --intermediate-dim 2048 \
      --top-k 6 --experts 256 --expert-start 0 --expert-end 32 \
      --concurrency 1 --distribution uniform --routing-cases 35 --seed 1234 \
      --warmup 35 --repeats 35 \
      --output "results/random-graph-${backend}-v4-t${tokens}.json" || exit 1
  done
done
```

Synthetic ModelSlim sources use positive FP32 scales exactly representable in
BF16 in both modes, recorded as `synthetic_scale_policy: bf16_exact_fp32`.
This satisfies the Torch resident provider's scale guard and keeps seeded inputs
identical across backends and modes; it is not checkpoint accuracy validation.
Graph runs require BF16, an indexed NPU and concurrency one. Graph temporary
estimates are reported as null rather than reusing eager estimates; peak device
allocation is still reported. Add `--profile` for NPU kernel traces. Full vLLM
benchmarks remain necessary to establish an end-to-end serving improvement.
