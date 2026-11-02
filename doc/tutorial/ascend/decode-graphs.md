# Ascend Decode graphs

By default, the graph path captures the complete model forward for Decode,
including routed experts on separate Workers. Prefill and mixed batches use
ordinary EK Transport. Full mode also covers Prefill and mixed steps with the
supported Torch or `ascend_fused` backend and native model/operator checks.
Sampling stays outside the captured forward. Every role uses the same Docker
runtime; models are mounted read-only.

| Component | Version |
| --- | --- |
| Python / CANN | 3.12 / 9.1.0 |
| Torch / Torch-NPU | 2.10.0+cpu / 2.10.0.post4 |
| Transformers | 5.14.1 |
| Triton / Triton-Ascend / NumPy | 3.5.0 / 3.2.2 / 1.26.4 |
| vLLM EMPTY | 0.26.0+empty |
| vLLM-Ascend | 0.1.dev4525+g155f68974 |

On ARM64, Torch-NPU adds NPU execution to the CPU Torch wheel.
The shared image derives from the pinned prebuilt `v0.26.0rc2-a3` runtime.
The Docker build installs EK wheels without rebuilding or patching the engine.
Both graph modes reuse this image; normal runtime kernel JIT and graph warmup
still run.
The PyPI `expertkit-vllm[engine]` extra selects a different CUDA engine; use the
Docker build below for Ascend.

## Configure the existing examples

The checked-in `dev/ascend/configs/qwen3-30b-a3b-w8a8` and
`deepseek-v4-flash-w8a8` pairs select two and eight Workers respectively.
These are example placements, not required Worker counts or expert partitions.
Choose the pools, hosts, unused devices and ports for your deployment. Each
Worker process serves one device. `backend: torch` and `ascend_fused` use their
existing computation paths; the Torch path retains original FP32 scales, while
the fused path performs its required BF16 scale conversion.

From the repository root:

```bash
cp dev/ascend/configs/qwen3-30b-a3b-w8a8/cluster.example.yaml \
  dev/ascend/configs/cluster.yaml
cp dev/ascend/configs/qwen3-30b-a3b-w8a8/experiment.example.yaml \
  dev/ascend/configs/experiment.yaml
```

For V4 substitute `deepseek-v4-flash-w8a8`. Replace example addresses and absolute
checkpoint/result paths. Reserve checkpoint, image and expert-cache disk space.
Each pool's `start_graph_control_port` reserves one graph-control port per Worker,
separate from `start_worker_port` and `start_peer_port`. Generated Workers bind
`0.0.0.0` and advertise the pool host address; they receive no model/placement
manifest from the Frontend.

The experiment's graph mapping is only:

```yaml
ascend_decode_graph:
  mode: decode
  capture_sizes: [1, 2, 4, 8, 16, 32, 64]
  timeout_seconds: 600.0
  rendezvous_host: 192.0.2.10
  rendezvous_port: 18090
```

Do not add `session_id`, `instance_id`, `model`, `layer_ids`, expert ranges or
Workers here. Frontend bootstrap creates a fresh session from native vLLM/Hugging
Face model facts and the Controller's READY topology. Workers resolve geometry,
expert computation, clipping and quantization from Weight Server metadata,
validate optional explicit model assertions, and check PREPARE against the
resolved facts. The examples omit these repeated model facts. Neither the
generator nor the user needs a second HF parser or placement list.

Native Frontend loading options are separate `serve` fields. Both W8A8 examples
set `quantization: ascend`. V4 additionally sets `load_format: safetensors`,
`safetensors_load_strategy: lazy`, `distributed_executor_backend: mp`,
`trust_remote_code: false` and `compilation_mode: VLLM_COMPILE`. Compilation modes
may be native strings or integers. Older explicit model quantization/V4 hints
still supply the previous Frontend defaults for omitted `serve` options; explicit
values take precedence, and `null` leaves the corresponding native key absent.

The graph's TCPStore listener belongs to the first selected Worker rank, ordered
by Worker ID after READY replica selection. `rendezvous_host` must address that
Worker's host. On multiple hosts, check the actual selected participants; the
Frontend is not automatically the listener. The generator preserves this explicit
address and checks known collisions, but cannot establish runtime rank ownership.
Multi-host configuration acceptance does not establish hardware qualification.

### Full graphs and native serving capacity

`mode: decode` selects `FULL_DECODE_ONLY`; Prefill and mixed steps use ordinary
execution. `mode: full` also captures Prefill and mixed steps. For example:

```yaml
serve:
  gpu_memory_utilization: 0.8
  max_model_len: 4096
  max_num_seqs: 128
  max_num_batched_tokens: 2048
  enable_chunked_prefill: true
ascend_decode_graph:
  mode: full
  capture_sizes: [64, 128, 256]
  timeout_seconds: 600.0
  rendezvous_host: 192.0.2.10
  rendezvous_port: 18090
```

Only these graph sizes are captured; smaller steps use the smallest compatible
bucket with padding. Capture sizes need not be powers of two or start at 1.
Long Prefill is chunked by native scheduling rather than rejected as a request
larger than one graph. Request count, token allocation/scheduling budget and
context length remain distinct. In Decode-only mode, ordinary Prefill is not
capped by the Decode graph size.

Optional native fields also include `max_num_scheduled_tokens`, `block_size`,
`num_gpu_blocks_override` and `kv_cache_memory_bytes`. Unspecified fields stay
absent; the generator does not supply hidden 64/2048/128 defaults. Normally omit
KV overrides and use native automatic planning. Actual padded-buffer, native
operator and KV-scratch compatibility is checked at startup. Keep
`serve.enforce_eager` false and tracing disabled; both modes use the same pinned
prebuilt runtime without a new native build.

Real provider limits still apply. The current V4 attention adapter uses the
pinned A3 native backend, 64 query heads and 512 sparse selections for ratio-4
layers; virtual kernel-block splitting is not supported. A context auto-fit
change after private metadata preparation requires rebuilding and budgeting that
metadata. These are startup compatibility checks, not presets for service
capacity. Arbitrary Worker layouts, changed graph sizes and singleton V4 serving
still require target-device qualification; generated YAML does not establish it.

## Generate, build and start

```bash
uv run --locked python dev/ascend/cli.py generate
dev/ascend/run-compose.sh image config -q
dev/ascend/run-compose.sh control config -q
dev/ascend/run-compose.sh attention config -q
dev/ascend/run-compose.sh expert worker-pool1 config -q

mkdir -p dist/attention-wheels
for package in expertkit-proto expertkit-transport expertkit-worker expertkit-vllm; do
  uv build --wheel --package "$package" --out-dir dist/attention-wheels
done
dev/ascend/run-compose.sh image build control-image attention-image
dev/ascend/run-compose.sh control up -d controller
dev/ascend/run-compose.sh expert worker-pool1 up -d
```

For additional pools start each bundle on its configured host. After Workers
register, schedule through the existing Controller path:

```bash
dev/ascend/run-compose.sh control run --rm admin \
  ek-cli --config /etc/expert-kit/config.yaml schedule rebalance
```

The existing manual/static/duplicate commands remain available. Wait for complete
READY expert coverage of the configured instance using Controller/Worker state
and logs. Do not run `schedule graph`: it is retired and will reject the request
without reading the old JSON or touching services. The generated graph JSON is
Frontend options only and cannot be used as scheduling input.

Start the Frontend once the selected experts are READY. Bootstrap waits within
its configured timeout and verifies process/weight identity before PREPARE:

```bash
dev/ascend/run-compose.sh attention up -d attention
dev/ascend/run-compose.sh attention logs -f attention
```

## Verify and stop

```bash
curl -fsS http://<HOST_IP>:18000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<MODEL_NAME>","messages":[{"role":"user","content":"你好，请自我介绍一下自己"}],"temperature":0,"max_tokens":64}'
```

Retain input/output and actual capture/replay evidence. A successful HTTP request
or generated YAML alone does not qualify graph execution. Full mode additionally
needs changing prompt lengths, chunked Prefill/mixed steps and subsequent Decode
checks. Changed graph configuration still requires target-device qualification.

```bash
dev/ascend/run-compose.sh attention stop
dev/ascend/run-compose.sh expert worker-pool1 stop
dev/ascend/run-compose.sh control stop
```

Known issue: when multiple Workers stop together, the Controller may reassign
experts to a shutting-down Worker, causing a nonzero exit. Retain containers and
logs and verify that all owned processes have stopped. Online graph rebuilding
remains unsupported.

Keep containers and caches. Before planned related placement/weight changes,
stop incoming requests and drain the old session. Restart affected graph
execution processes after the new assignment becomes READY; bootstrap creates
new identities and captures. Unrelated topology changes do not require rebuild.
A participating process failure or invalid weights stop the session; a partially
submitted step cannot retry through eager execution. Online recapture/migration
is not provided.

For ordinary EK with the same Controller placement, omit `ascend_decode_graph`
and select the ordinary execution settings; no saved graph-placement JSON is
needed. Compare paths only after independently verifying their actual placement
and runtime settings.
