from __future__ import annotations

import torch

from experiments.checkpoint_compat import remap_legacy_vit_keys


def test_legacy_vit_key_remap_is_explicit_and_collision_free():
    source = {
        "encoder.encoder.layer.0.attention.attention.query.weight": torch.ones(2, 2),
        "encoder.encoder.layer.0.attention.attention.key.bias": torch.zeros(2),
        "encoder.encoder.layer.0.attention.output.dense.weight": torch.ones(2, 2),
        "encoder.encoder.layer.0.intermediate.dense.weight": torch.ones(2, 2),
        "encoder.encoder.layer.0.output.dense.bias": torch.zeros(2),
        "predictor.layers.0.weight": torch.ones(2, 2),
    }
    remapped, changed = remap_legacy_vit_keys(source)
    assert changed == 5
    assert "encoder.layers.0.attention.q_proj.weight" in remapped
    assert "encoder.layers.0.attention.k_proj.bias" in remapped
    assert "encoder.layers.0.attention.o_proj.weight" in remapped
    assert "encoder.layers.0.mlp.fc1.weight" in remapped
    assert "encoder.layers.0.mlp.fc2.bias" in remapped
    assert "predictor.layers.0.weight" in remapped
