# Ascend Decode graphs

By default, the graph path captures the complete model forward for Decode,
including routed experts on separate Workers. Prefill and mixed batches use
ordinary EK Transport. Qwen3 W8A8 with the Torch backend can also enable full
graphs for Prefill, mixed and Decode steps.
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

## Select a model

Use one host with a separate physical NPU for each role:

| Example directory under `dev/ascend/configs/` | Frontend | Workers | Experts per Worker, per layer | Total routed experts |
| --- | ---: | ---: | ---: | ---: |
| `qwen3-30b-a3b-w8a8` | 1 | 2 | 64 | 6,144 |
| `deepseek-v4-flash-w8a8` | 1 | 8 | 32 | 11,008 |

Both examples select `backend: ascend_fused` in the expert pool and use
ModelSlim dynamic W8A8 weights and BF16 activations. Set the pool's backend to
`torch` to select the existing three-projection implementation instead. The fused
resident provider captures the same `run_ascend_fused_moe()` operator pipeline
used by eager `ascend_fused`: packed W13 projection, clipped SwiGLU, W2 projection,
and native routing-weighted token unpermute. It maps global expert IDs to the
local interval with subtraction and masking; nonlocal routes use an inactive
sentinel and zero routing weight. All experts in the local interval are loaded
and leased before capture and remain resident until the session closes.

The fused provider implements the shared resident interface without inheriting
Torch's computation. It uses native scaled GMM and scale conversion, rather than
Torch's explicit INT32-dot rescaling or optional Triton ordered reduction. The
lossless FP32/BF16 scale round-trip guard belongs to the Torch graph provider;
`ascend_fused` converts scales just as its eager operator path does.

Worker capture records communication and computation across the configured routed
layers for each token capacity. Later decode steps replay those graphs with new
hidden states and routing. A backend-only graph benchmark measures one layer's
local shard; full-model HCCL capture/replay and generation accuracy require
separate qualification on the target runtime. See the
[resident graph benchmark](../../../ek-worker/README.md#compare-resident-graph-computation)
for a matched backend comparison.

Reserve space for the checkpoint, images and expert cache before deployment; the V4
expert cache alone needs approximately 277 GB. The examples reserve Worker
memory separately from Frontend KV cache memory.

From the repository root, select one example pair:

```bash
cp dev/ascend/configs/qwen3-30b-a3b-w8a8/cluster.example.yaml \
  dev/ascend/configs/cluster.yaml
cp dev/ascend/configs/qwen3-30b-a3b-w8a8/experiment.example.yaml \
  dev/ascend/configs/experiment.yaml
```

For V4, substitute `deepseek-v4-flash-w8a8` in both source paths. In the copied
files, replace every `192.0.2.10` with the host's routable address and fill in
absolute checkpoint and result paths. Choose unused devices and ports. The
example `instance_id: 1` assumes a fresh deployment database; an existing
installation must use its actual configured instance ID. All roles must agree
on `weight_version`, model geometry and the same graph document.

### Enable Qwen3 Prefill graphs

In the copied Qwen3 `cluster.yaml`, change the existing `expert.backend` from
`ascend_fused` to `torch`. In the copied `experiment.yaml`, update the existing
graph mapping while retaining its identity, endpoints, layers and Workers:

```yaml
ascend_decode_graph:
  mode: full
  capture_sizes: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
  # Retain the other fields from the Qwen3 example.
```

Keep all 48 routed layers and the two Worker intervals `[0, 64)` and `[64, 128)`.
The generator selects synchronous native `FULL`, a 2,048-token step budget and
`max-num-seqs: 64`. Capture sizes count tokens, including padding; they do not
raise the concurrent request limit to 2,048. Small Decode buckets remain present.
Full mode also selects native FULL attention for Decode, so a comparison with
decode-only mode includes that attention-path change.

Omitting `mode` keeps `decode` and `FULL_DECODE_ONLY`. Use decode mode for V4,
V2-Lite and the `ascend_fused` backend. The checked-in examples are unchanged.

Generate and check the Compose files:

```bash
uv run --locked python dev/ascend/cli.py generate
dev/ascend/run-compose.sh image config -q
dev/ascend/run-compose.sh control config -q
dev/ascend/run-compose.sh attention config -q
dev/ascend/run-compose.sh expert worker-pool1 config -q
```

## Build and start

Build the four EK wheels from the checked-out source, then build the shared
Frontend/Worker runtime image:

```bash
mkdir -p dist/attention-wheels
for package in expertkit-proto expertkit-transport expertkit-worker expertkit-vllm; do
  uv build --wheel --package "$package" --out-dir dist/attention-wheels
done
dev/ascend/run-compose.sh image build control-image attention-image
dev/ascend/run-compose.sh control up -d controller
dev/ascend/run-compose.sh expert worker-pool1 up -d
```

After all Workers register, apply the exact expert intervals from the generated
graph document and wait for loading:

```bash
dev/ascend/run-compose.sh control run --rm admin \
  ek-cli --config /etc/expert-kit/config.yaml schedule graph \
  --graph-config /etc/expert-kit/ascend-decode-graph.json \
  --wait-ready-seconds 1800
```

This command is for startup with automatic assignment disabled. Keep the
Frontend stopped and do not run another scheduling command concurrently. It
validates the instance, model geometry, active Workers, capacity and complete
expert partition, then atomically installs placement. The readiness wait checks
the exact assignments. Frontend PREPARE additionally checks live Worker identity
and acquires resident weights.

Start the Frontend after the placement command succeeds:

```bash
dev/ascend/run-compose.sh attention up -d attention
dev/ascend/run-compose.sh attention logs -f attention
```

In default decode mode, the generator selects synchronous `FULL_DECODE_ONLY`,
disables prefix caching and passes the same graph configuration to all
participants. The examples capture batch sizes 1, 2, 4, 8, 16, 32 and 64;
intermediate sizes use padding.
Keep `serve.enforce_eager` false and tracing disabled for this graph path.
Prefill and mixed batches use ordinary EK execution in decode mode. Full mode
captures them as well.
## Verify and stop

Use the selected `model.name` and host address:

```bash
curl -fsS http://<HOST_IP>:18000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<MODEL_NAME>","messages":[{"role":"user","content":"你好，请自我介绍一下自己"}],"temperature":0,"max_tokens":64}'
```

Retain both input and output. Confirm every declared Worker prepared its full
layer range and participated in Decode replay. A successful HTTP response alone
does not establish graph execution. In full mode, also verify Prefill and mixed
replay with changing prompt lengths and subsequent Decode steps.

Stop the deployment while retaining containers and caches:

```bash
dev/ascend/run-compose.sh attention stop
dev/ascend/run-compose.sh expert worker-pool1 stop
dev/ascend/run-compose.sh control stop
```

Placement and weights stay fixed for a graph session. After a peer failure or
session timeout, stop all participants and use a new `session_id` before
restarting. A failed partially submitted graph step cannot fall back to ordinary
RPC.

To compare ordinary EK with the same expert placement, retain a copy of the
generated graph JSON before removing `ascend_decode_graph` from the local
experiment. Set `serve.enforce_eager: true` for the ordinary comparison.
Regenerate, start the control bundle and Workers, then apply that JSON only as
placement input:

```bash
dev/ascend/run-compose.sh control run --rm \
  --volume <ABSOLUTE_PLACEMENT_JSON>:/configs/placement.json:ro admin \
  ek-cli --config /etc/expert-kit/config.yaml schedule graph \
  --graph-config /configs/placement.json --wait-ready-seconds 1800
```

The ordinary Workers do not enable graph execution. Start the Frontend after
placement succeeds. For ordinary deployments without fixed intervals, use
`schedule rebalance` from the common deployment workflow.
