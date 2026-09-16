# V4 configuration fixtures

These are configuration-only snapshots read on 2026-09-16, with no weight data.

- `deepseek_v4_fp4_config.json`: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/resolve/main/config.json
- `deepseek_v4_w8a8_config.json`: https://modelscope.cn/models/sgl-npu/DeepSeek-V4-Flash-W8A8/resolve/master/config.json

Tests use the complete configurations to distinguish global quantization from
routed expert quantization, and to validate the selected compressed-tensors
recipe. Synthetic extraction fixtures replace dimensions with 128/256 and write
constant, explicitly synthetic tensors into temporary directories.
