"""Source positional grid AND fused QKV arithmetic, without changing weights."""
import torch
from torch import nn
from transformers.models.dinov2.modeling_dinov2 import Dinov2SelfAttention

from experiments.pc1_layer_audit import dino_source_compatible as v1


class SourceMatchedSelfAttention(Dinov2SelfAttention):
    def __init__(self, original):
        nn.Module.__init__(self)
        if not isinstance(original, Dinov2SelfAttention):
            raise TypeError("expected HF DINOv2 self attention")
        if isinstance(original, SourceMatchedSelfAttention):
            raise ValueError("source attention already installed")
        if original.dropout_prob != 0:
            raise ValueError("only zero-dropout frozen encoder is supported")
        self.config = original.config
        self.num_attention_heads = original.num_attention_heads
        self.attention_head_size = original.attention_head_size
        self.all_head_size = original.all_head_size
        self.dropout_prob = original.dropout_prob
        self.scaling = original.scaling
        self.is_causal = original.is_causal
        self.query, self.key, self.value = original.query, original.key, original.value
        self.train(original.training)

    def forward(self, hidden_states, **kwargs):
        if kwargs:
            raise ValueError(f"uncertified attention options: {sorted(kwargs)}")
        layers = (self.query, self.key, self.value)
        weights = torch.cat([layer.weight for layer in layers])
        if any(layer.bias is None for layer in layers):
            if not all(layer.bias is None for layer in layers):
                raise ValueError("inconsistent QKV bias configuration")
            bias = None
        else:
            bias = torch.cat([layer.bias for layer in layers])
        b, n, c = hidden_states.shape
        qkv = nn.functional.linear(hidden_states, weights, bias).reshape(
            b, n, 3, self.num_attention_heads, self.attention_head_size)
        q, k, v = (part.transpose(1, 2) for part in torch.unbind(qkv, 2))
        output = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0., is_causal=False)
        return output.transpose(1, 2).contiguous().view(b, n, c), None


def install_source_interpolation(backbone):
    """Historical entry name retained for runner injection; applies both fixes."""
    before = dict(backbone.named_parameters())
    v1.install_source_interpolation(backbone)
    for layer in backbone.encoder.layer:
        layer.attention.attention = SourceMatchedSelfAttention(layer.attention.attention)
    after = dict(backbone.named_parameters())
    if before.keys() != after.keys() or any(after[k] is not p for k, p in before.items()):
        raise AssertionError("source repair changed parameter identity")
    return backbone
