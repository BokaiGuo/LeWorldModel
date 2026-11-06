import pytest
import torch

from experiments.pc_dinov2_adapter_equivalence_dev_20261106_codex import convert, stats


def test_dino_conversion_covers_sources_and_qkv_order():
    source = {key: torch.tensor([0.0]) for key in (
        "embeddings.cls_token", "embeddings.mask_token", "embeddings.patch_embeddings.projection.bias",
        "embeddings.patch_embeddings.projection.weight", "embeddings.position_embeddings",
        "layernorm.bias", "layernorm.weight")}
    for i in range(12):
        prefix = f"encoder.layer.{i}."
        for suffix in ("weight", "bias"):
            for j, name in enumerate(("query", "key", "value")):
                source[prefix + f"attention.attention.{name}.{suffix}"] = torch.tensor([float(j)])
            for name in ("attention.output.dense.", "norm1.", "norm2.", "mlp.fc1.", "mlp.fc2."):
                source[prefix + name + suffix] = torch.tensor([0.0])
        for name in ("layer_scale1.lambda1", "layer_scale2.lambda1"):
            source[prefix + name] = torch.tensor([0.0])
    converted = convert(source)
    assert converted["blocks.0.attn.qkv.weight"].tolist() == [0.0, 1.0, 2.0]
    with pytest.raises(ValueError):
        convert({**source, "unexpected": torch.zeros(1)})


def test_difference_statistics():
    assert stats(torch.ones(2), torch.ones(2))["max_abs"] == 0
