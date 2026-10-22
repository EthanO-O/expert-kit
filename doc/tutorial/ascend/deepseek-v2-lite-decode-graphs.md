# DeepSeek-V2-Lite Ascend Decode graphs

This optional deployment keeps vLLM's complete Decode graph, including attention,
KV updates, shared experts and routed expert layers. Sampling remains in the
native runner outside the captured model forward. One Frontend NPU sends final
int32 expert IDs and FP32 routing weights to one or two expert Worker NPUs. Each
Worker retains its fixed local routed weights and captures its complete layer
loop with production grouped matmul.
Prefill and mixed batches use the existing Transport path.

The pinned runtime has passed trained-weight GMM and complete native model
qualification with one Worker and with two disjoint expert Workers. Tests cover
Chinese and arithmetic prompts with natural EOS, padded batch 3, and batches
32/64. Device diagnostics confirm replay uses no per-layer Python or ordinary
RPC calls, both expert Workers acknowledge each step, and healthy graph close
releases its resources. The original single-Worker JSON also completes actual
generation and close on the extended runtime.

Repeated ordinary/graph EK measurements and native Transformers references use
the same checkpoint, saved 128-token prompts, 128 generated tokens, batch sizes
1/32/64, two warmups and three measured runs. The two-Worker comparison uses
three distinct NPUs for each path; Transformers uses a native layer partition,
while EK partitions the routed experts. The graph path shows a repeatable
end-to-end benefit over matched ordinary EK. Fixed benchmark outputs match
across all three paths. Adding a Worker is not itself
a measured scaling benefit: each Worker executes padded fixed-capacity FFNs.

Failure tests kill each distinct role after all Begin acknowledgements and
before the Frontend invokes native replay. Both survivors exit through their
own deadline mechanisms within 150 seconds, before external cleanup. This
qualifies that admitted-step boundary; it does not establish arbitrary active
kernel or partial-Finish fault behavior. The ordinary vLLM 0.25.1 workspace
installation remains independent.

Ordinary and graph execution produced different greedy token choices in a Chinese
response: the single-Worker comparison first differed at the fifteenth generated
token, and the two-Worker comparison at the eighteenth. Arithmetic and all fixed
benchmark outputs matched in each comparison. The independent expert-compute
tolerance was unchanged. Differences already appeared in the Prefill output log
probabilities; these measurements do not identify Decode graph replay or its
reduction order as the cause. They establish the tested output and numerical
scope, without a general quality-equivalence claim.

## Build the shared runtime image

Use the [common deployment workflow](README.md#3-build-the-images) to build the
four EK Python wheels and the shared `ek-ascend-runtime` image. Both
`images.attention` and `images.expert` select this same image. The example
configuration selects the official `v0.26.0rc2-a3` tag. The Dockerfile default
also freezes the inspected base by digest:

```text
quay.io/ascend/vllm-ascend:v0.26.0rc2-a3@sha256:8cbb72814ec80978ab15c0164c43b4e60541cecd1d9f6f4b143ba6ecb4ff63f6
```

This base already provides Python 3.12, CANN 9.1.0, Torch 2.10.0, Torch-NPU
2.10.0.post4, Triton-Ascend 3.2.2 and the EMPTY/NPU vLLM 0.26.0 engine. Install
EK with normal dependency resolution and the checked-in
[image constraints](../../../dev/ascend/graph-runtime-constraints.txt). The
Dockerfile uses binary packages and leaves the image's engine and device stack
in place. There are no engine patches, source builds or separate native
Mooncake build steps. EK Triton kernels compile on first use during warmup.

The `ascend-graph` extra supplies configuration dependencies; the CUDA `engine`
extra is a separate installation profile. Keep the base image's `PATH`,
`PYTHONPATH` and `LD_LIBRARY_PATH` so its CANN compiler and Python modules remain
available. Frontend runs `vllm` and Worker runs the installed `ek-worker`
entrypoint directly.

The validation results earlier in this guide were recorded with the previous
runtime. They do not establish qualification of this prebuilt image for
DeepSeek-V2-Lite.

## Generate configuration

Use the [standard generator](../../../dev/ascend/README.md) and set the model
geometry to 27 layers, 64 experts, hidden size 2048, intermediate size 1408,
top-k 6 and BF16 activation/weight dtypes. Layer 0 is dense; routed layers are
1 through 26. Existing model metadata and Controller placement honor
`first_k_dense_replace=1` from the checkpoint.

The deployment uses one host and one expert pool, with a distinct physical device
for the Frontend and every Worker. The qualified single-Worker deployment has
one Frontend device and one distinct Worker device. The optional two-Worker
configuration below uses a third distinct device and has passed the fixed
placement qualification above. Set
`expert.runtime.max_active_batches_per_device: 1` and reserve at least the
routed BF16 weights: 26.81 GiB on a single Worker or 13.41 GiB on each of
two Workers, plus each role's measured graph/runtime memory. Use
the same shared runtime image above for Frontend and Workers. Verify that
`vllm` and `ek-worker` resolve in that image; no workspace environment is needed.

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

For two Workers, replace the legacy `worker_id` and `control_endpoint` fields
with the following ordered list. Keep the other shared fields above. Each Worker
has one physical device exposed as logical `npu:0`; the Frontend remains native
TP/DP 1. Control ports and the rendezvous port must all be distinct.

```yaml
  workers:
    - worker_id: worker-pool1-worker-00
      control_endpoint: 192.0.2.10:18079
      expert_start: 0
      expert_end: 32
    - worker_id: worker-pool1-worker-01
      control_endpoint: 192.0.2.10:18081
      expert_start: 32
      expert_end: 64
```

Configure the Controller's desired assignments explicitly: every layer 1..26
must assign experts 0..31 only to the first Worker and 32..63 only to the second.
The generator emits runtime configuration; it does not establish this placement.
The ordinary manual placement command replicates its selected experts across
Workers and must not be assumed to produce this split. Verify actual READY
routes and each registered process start ID before starting the Frontend.

Two-Worker graphs reduce FP32 partial outputs on the Frontend and cast the final
sum once to BF16. Ordinary multi-Worker execution rounds each partial to BF16
before combining, so exact greedy token equality is not promised. The fixed GMM
input shape currently pads unused rows onto the last local expert and masks their
contribution before weighting. Each Worker executes `capacity * top_k` FFN rows,
for twice as many executed rows as useful assignments across the two Workers;
this implementation establishes cooperation rather than compute scaling.

Control and rendezvous endpoints bind the shared host. Their ports must be
distinct and unused.
The generator writes `ascend-decode-graph.json`, embeds exactly the same document
in each Worker YAML, mounts it in the Frontend, and sets `EK_INSTANCE_ID` and
`EK_ASCEND_DECODE_GRAPH_CONFIG`. All participants use host networking. Generated vLLM
configuration selects `FULL_DECODE_ONLY`, matching capture sizes, sufficient
`max-num-seqs`, BF16 and synchronous scheduling. Eager execution and tracing are
rejected for this lane.

## Startup and qualification

Start the existing database/migrations, Weight Server, model registration,
Controller and actual `ek-worker` service first. Confirm that the configured
instance exists and **all 1,664 routed experts in layers 1..26 are READY on the
declared Worker or disjoint pair of Workers** before starting `vllm serve`. A running Worker or allocated placement
alone does not establish readiness. PREPARE acquires the resident weights once
and fails if readiness or identity disagrees.

Keep placement and weights fixed for the session. Withdrawal invalidates its
captured graphs, drains device work and releases leases before weight removal.
A failed or timed-out graph session requires all participants to restart with a fresh
session ID; it does not fall back after a partially submitted step. A peer with
unconfirmed device completion retains storage until its watchdog terminates the
process. A peer whose Finish has already confirmed completion may remain idle
with its weights pinned until owned shutdown; partial Finish failure still
poisons the Frontend and prevents another Begin.

Qualify trained-weight GMM with changing expert counts and empty experts, then
full greedy generation and peer exit. Verify native FULL replay executes no
per-layer Python or RPC. Performance acceptance requires repeated matched full
inference runs against ordinary EK and native Transformers using the same
checkpoint, devices, dtype, prompt/output lengths and warmup/run counts. A
standalone GMM timing does not establish an end-to-end benefit.
