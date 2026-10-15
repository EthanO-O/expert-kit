# DeepSeek-V2-Lite Ascend Decode graphs

This optional deployment keeps vLLM's complete Decode graph, including attention,
KV updates, shared experts and routed expert layers. Sampling remains in the
native runner outside the captured model forward. One Frontend NPU sends final int32 expert
IDs and FP32 routing weights to one Worker NPU. The Worker retains all routed
weights and captures its complete layer loop with production grouped matmul.
Prefill and mixed batches use the existing Transport path.

The pinned runtime has passed trained-weight GMM graph checks and complete
native model generation, including Chinese and arithmetic prompts with natural
EOS, padded batch 3, and batches 32/64. Device diagnostics confirm replay uses
no per-layer Python or ordinary RPC calls, and normal graph close releases its
resources. Matched ordinary/graph EK measurements and the native Transformers reference
have completed. Both peer-exit directions have passed bounded device failure
containment: the surviving role exits through its watchdog before external
cleanup. The ordinary
vLLM 0.25.1 workspace installation is independent.

The two execution paths produced different greedy token choices in one Chinese
response, starting at the fifteenth generated token; arithmetic and all fixed
benchmark outputs matched. The independent expert-compute tolerance was
unchanged. Differences already appeared in the Prefill output log probabilities, so the result
does not identify Decode graph replay as their cause.

## Install an isolated runtime

Use Python 3.12, CANN 9.1.0 and HDK 25.5.1. The graph engine sources are pinned:

| Component | Source revision |
| --- | --- |
| vLLM 0.30.0, built for EMPTY/NPU | `ced6857afa0ea7b2e3f0846a62e1394e90f15607` |
| vLLM-Ascend | `8d4409d6256d8a6729140ddcc0d1889e3f96cdd6` |
| Official `deepseek-ai/DeepSeek-V2-Lite-Chat` checkpoint | `85864749cd611b4353ce1decdb286193298f64c7` |

The tested preparation starts from an independent copy of a source-built runtime
at these exact revisions. Keep its source trees, venv and compiled modules
together at their recorded paths, outside the generator's `generated/` output.
The generator replaces that output directory. Verify source HEADs, tracked diffs,
package metadata and Torch/Torch-NPU compiled-module import paths before use.
Never mutate the original frozen runtime.

The engine build uses `VLLM_TARGET_DEVICE=empty`; a PyPI CPU vLLM wheel has a
different Torch dependency and is unsuitable. The recorded source metadata pins
`opencv-python-headless==4.11.0.86` in vLLM's `requirements/common.txt` and changes
Ascend's FastAPI requirement to `>=0.133.0,<0.137.0`. Preserve these two source
diffs and the exact commit IDs. A fresh engine-wheel deployment is an optional
later packaging route and has not been qualified by the current preparation.

Build the four EK leaf wheels from the repository:

```bash
uv build --package expertkit-proto --wheel
uv build --package expertkit-transport --wheel
uv build --package expertkit-worker --wheel
uv build --package expertkit-vllm --wheel
```

Install all four EK wheels into that independently owned runtime with normal
dependency resolution and
[graph-runtime-constraints.txt](../../../dev/ascend/graph-runtime-constraints.txt).
From the repository root, set `GRAPH_PYTHON` to that runtime's Python:

```bash
GRAPH_PYTHON=/absolute/path/to/owned/controlled/.venv/bin/python
# Replace previously installed same-version EK leaves before resolving new wheels.
"$GRAPH_PYTHON" -m pip uninstall -y \
  expertkit-proto expertkit-transport expertkit-worker expertkit-vllm
"$GRAPH_PYTHON" -m pip install \
  --constraint dev/ascend/graph-runtime-constraints.txt \
  numpy==1.26.4 grpcio==1.71.0 protobuf==5.29.6 \
  structlog==25.4.0 googleapis-common-protos==1.75.0 \
  dist/expertkit_proto-*.whl dist/expertkit_transport-*.whl \
  dist/expertkit_worker-*.whl 'dist/expertkit_vllm-0.1.0-py3-none-any.whl[ascend-graph]'
"$GRAPH_PYTHON" -m pip check
```

Record the engine metadata before and after installation and verify it did not
change. For offline preparation, provide the pinned dependency wheels through
`--no-index --find-links`; keep dependency resolution enabled. The current
experiment verifies installed EK Python and metadata bytes against each supplied
wheel and audits the recursive engine/EK/compiler dependency closure. Its global
`pip check` remains failed for three unrelated CANN tools (`affinity-sched`,
`mindstudio-kpp`, `ms-service-profiler`), which are explicitly outside that
closure; this exception does not excuse a core runtime conflict.

The `ascend-graph` extra supplies optional configuration dependencies; it does
not install an engine. Do not combine it with the `engine` extra, which selects
vLLM 0.25.1. Keep the image's complete CANN/TBE compiler `PATH`, `PYTHONPATH` and
`LD_LIBRARY_PATH`. NumPy 1.26.4 matches Triton-Ascend 3.2.2's requirement. Record
any unrelated vendor profiling-tool conflicts separately; EK, engine, Torch,
Torch-NPU and compiler dependencies must be consistent.

## Generate configuration

Use the [standard generator](../../../dev/ascend/README.md) and set the model
geometry to 27 layers, 64 experts, hidden size 2048, intermediate size 1408,
top-k 6 and BF16 activation/weight dtypes. Layer 0 is dense; routed layers are
1 through 26. Existing model metadata and Controller placement honor
`first_k_dense_replace=1` from the checkpoint.

The first deployment supports one host with one Frontend device and one distinct
Worker device, one pool and one Worker. Set
`expert.runtime.max_active_batches_per_device: 1` and reserve at least the
26.81 GiB routed BF16 weights plus measured Worker graph/runtime memory. Use
images containing the isolated runtime above; the default older image does not
supply this graph stack. In both images, prepend that exact venv’s `bin/` to
`PATH` and verify `vllm` and `ek-worker` resolve there. The graph Worker Compose
uses the installed `ek-worker` entrypoint directly, so it does not select a
workspace `uv run` environment.

Add this optional section to `experiment.yaml`, replacing the address, session,
instance ID and weight version with the actual deployment values:

```yaml
ascend_decode_graph:
  session_id: deepseek-resident-session-v1
  instance_id: 1
  worker_id: worker-pool1-worker-00
  weight_version: main
  control_endpoint: 192.0.2.10:18079
  rendezvous_host: 192.0.2.10
  rendezvous_port: 18080
  layer_ids: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
  capture_sizes: [1, 2, 4, 8, 16, 32, 64]
  timeout_seconds: 120.0
```

Both endpoints bind the Worker host. Their ports must be distinct and unused.
The generator writes `ascend-decode-graph.json`, embeds exactly the same document
in Worker YAML, mounts it in the Frontend, and sets `EK_INSTANCE_ID` and
`EK_ASCEND_DECODE_GRAPH_CONFIG`. Both roles use host networking. Generated vLLM
configuration selects `FULL_DECODE_ONLY`, matching capture sizes, sufficient
`max-num-seqs`, BF16 and synchronous scheduling. Eager execution and tracing are
rejected for this lane.

## Startup and qualification

Start the existing database/migrations, Weight Server, model registration,
Controller and actual `ek-worker` service first. Confirm that the configured
instance exists and **all 1,664 routed experts in layers 1..26 are READY on the
one Worker** before starting `vllm serve`. A running Worker or allocated placement
alone does not establish readiness. PREPARE acquires the resident weights once
and fails if readiness or identity disagrees.

Keep placement and weights fixed for the session. Withdrawal invalidates its
captured graphs, drains device work and releases leases before weight removal.
A failed or timed-out graph session requires both roles to restart with a fresh
session ID; it does not fall back after a partially submitted step.

Qualify trained-weight GMM with changing expert counts and empty experts, then
full greedy generation and peer exit. Verify native FULL replay executes no
per-layer Python or RPC. Performance acceptance requires repeated matched full
inference runs against ordinary EK and native Transformers using the same
checkpoint, devices, dtype, prompt/output lengths and warmup/run counts. A
standalone GMM timing does not establish an end-to-end benefit.
