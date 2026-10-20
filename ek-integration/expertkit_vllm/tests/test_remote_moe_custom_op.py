"""Keep the compiler-visible mutation schema consistent with the real MoE op."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from expertkit_vllm.experts import remote_moe


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("strided", [False, True])
def test_mutation_only_op_preserves_public_output_and_fake_contract(monkeypatch, dtype, strided):
    calls = []
    layer = remote_moe.RemoteMoERunner.__new__(remote_moe.RemoteMoERunner)
    nn.Module.__init__(layer)
    layer.layer_name = "model.layers.0.mlp"

    def external_compute(hidden_states, router_logits, input_ids):
        calls.append((hidden_states, router_logits, input_ids))
        return hidden_states + router_logits

    monkeypatch.setattr(layer, "_forward_impl", external_compute)
    monkeypatch.setattr(
        remote_moe,
        "get_forward_context",
        lambda: SimpleNamespace(no_compile_layers={layer.layer_name: layer}),
    )
    hidden = torch.arange(12, dtype=dtype).reshape(2, 6)
    hidden = hidden[:, ::2] if strided else hidden[:, :3].contiguous()
    router_logits = torch.full_like(hidden, 2)
    input_ids = torch.tensor([1, 2], dtype=torch.int64)
    expected = hidden + router_logits
    pointer, stride = hidden.data_ptr(), hidden.stride()

    output = layer(hidden, router_logits, input_ids)

    assert output is hidden
    assert hidden.data_ptr() == pointer
    assert output.shape == expected.shape and output.dtype == dtype
    assert output.stride() == stride
    assert len(calls) == 1
    assert calls[0][0] is hidden and calls[0][1] is router_logits
    assert calls[0][2] is input_ids
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    result = torch.library.opcheck(
        torch.ops.vllm.expertkit_remote_moe.default,
        (hidden, router_logits, input_ids, remote_moe._encode_layer_name(layer.layer_name)),
        test_utils=("test_schema", "test_faketensor"),
    )
    assert result == {"test_schema": "SUCCESS", "test_faketensor": "SUCCESS"}
