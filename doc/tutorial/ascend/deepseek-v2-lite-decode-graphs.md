# Historical DeepSeek-V2-Lite Ascend Decode graphs

This page preserves V2-Lite results from an earlier source-built runtime.
For current Docker configuration and startup, use the
[Qwen3/V4 Full and Decode graph guide](decode-graphs.md), including its
[hardware qualification scope](decode-graphs.md#hardware-qualification).
Those examples use different model geometry and expert counts; their results do
not qualify V2-Lite on the current prebuilt runtime. For the CUDA/Transformers
V2-Lite deployment, use the [standalone tutorial](../standalone/deepseek-v2-lite.md).

## V2-Lite model declaration

The official `deepseek-ai/DeepSeek-V2-Lite-Chat` checkpoint revision
`85864749cd611b4353ce1decdb286193298f64c7` has 27 layers, 64 routed experts per
MoE layer, hidden size 2048, expert intermediate size 1408 and top-k 6.
Layer 0 is dense. Current startup obtains routed layers and geometry from native
Frontend and Weight Server facts; do not put layer IDs or shapes in the graph
options document. Explicit Worker model values are optional assertions.

Choose Worker count/devices in the cluster pools and use ordinary Controller
scheduling. Graph startup consumes READY assignments, including noncontinuous
IDs and different per-layer ownership; no ordered interval manifest is required.
The model has 1,664 routed experts in total. Reserve routed weights plus graph
buffers/runtime memory according to actual ownership; the historical one-Worker
weight estimate was 26.81 GiB and two equal shards 13.41 GiB each.

## Historical qualification

The earlier CANN 9.1 / Torch 2.10.0 / Torch-NPU 2.10.0.post4 runtime used vLLM
source `ced6857afa0ea7b2e3f0846a62e1394e90f15607` and vLLM-Ascend source
`8d4409d6256d8a6729140ddcc0d1889e3f96cdd6`. It passed changing-input trained-weight
GMM, full-model generation, padded B3, B32/B64, healthy graph close and the
original single-Worker JSON regression. Failure tests covered peer exit after
Begin acknowledgements and before Frontend replay.

Matched ordinary/graph and native Transformers measurements used identical
128-token inputs, 128 generated tokens, two warmups and three measured runs.
Graph execution improved end-to-end time over ordinary EK. Chinese greedy
responses sometimes diverged between paths; arithmetic and fixed benchmark
outputs matched. Expert-compute tolerances were unchanged. These results belong
to that historical runtime and do not qualify V2-Lite on a different stack.

Multiple Workers cooperate on each layer, with FP32 reduction before the final
BF16 cast. Fixed graph capacity can execute padded rows, so adding Workers does
not by itself establish a throughput improvement.
