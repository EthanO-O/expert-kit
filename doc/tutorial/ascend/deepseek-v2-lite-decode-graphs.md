# DeepSeek-V2-Lite Ascend Decode graphs

For the current Docker runtime and startup commands, use the
[Ascend Decode graph guide](decode-graphs.md). The checked-in Qwen3 and V4 examples
use different model geometry and expert counts from DeepSeek-V2-Lite.

## V2-Lite model declaration

The official `deepseek-ai/DeepSeek-V2-Lite-Chat` checkpoint revision
`85864749cd611b4353ce1decdb286193298f64c7` has 27 layers, 64 routed experts per
MoE layer, hidden size 2048, expert intermediate size 1408 and top-k 6.
Layer 0 is dense; `layer_ids` must contain exactly 1 through 26. Use BF16
weights and activations without a quantization declaration.

A single Worker owns all 64 experts on each routed layer. For two Workers,
declare ordered intervals `[0,32)` and `[32,64)`. Both arrangements require
1,664 routed experts READY before Frontend PREPARE. Reserve at least 26.81 GiB
of routed weights for one Worker, or 13.41 GiB per Worker for two, plus graph
buffers and runtime memory.

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
