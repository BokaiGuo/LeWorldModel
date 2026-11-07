import pytest
import torch

from experiments.pc1_layer_audit.dino_source_compatible_v2 import (
    SourceMatchedSelfAttention, install_source_interpolation,
)
from test_pc1_dino_source_compatible import small_model


def test_fused_projection_matches_reference_arithmetic_and_preserves_weights():
    model = small_model()
    weights = {k: (p, p.clone()) for k, p in model.named_parameters()}
    install_source_interpolation(model)
    attn = model.encoder.layer[0].attention.attention
    assert isinstance(attn, SourceMatchedSelfAttention)
    x = torch.randn(2, 13, 12, requires_grad=True)
    modules = (attn.query, attn.key, attn.value)
    fused = torch.nn.Linear(12, 36)
    with torch.no_grad():
        fused.weight.copy_(torch.cat([m.weight for m in modules]))
        fused.bias.copy_(torch.cat([m.bias for m in modules]))
    qkv = fused(x).reshape(2, 13, 3, 3, 4)
    q, k, v = (t.transpose(1, 2) for t in torch.unbind(qkv, 2))
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v).transpose(1, 2).contiguous().view(2, 13, 12)
    actual, _ = attn(x)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.sum().backward()
    assert torch.isfinite(x.grad).all()
    for key, p in model.named_parameters():
        assert p is weights[key][0] and p.grad is None
        torch.testing.assert_close(p, weights[key][1], atol=0, rtol=0)
    with pytest.raises(ValueError, match="uncertified"):
        attn(x, oracle=True)


def test_source_matched_model_stays_frozen_and_differentiable():
    model = install_source_interpolation(small_model())
    assert all(not m.training for m in model.modules())
    assert all(not p.requires_grad for p in model.parameters())
    image = torch.randn(1, 3, 14, 22, requires_grad=True)
    model(image).last_hidden_state[..., 0].sum().backward()
    assert torch.isfinite(image.grad).all()
