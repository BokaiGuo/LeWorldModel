from __future__ import annotations

import pytest
import torch

from experiments.theory_spectral_transfer.run_jepa_wm_pointmaze_temporal_eval import (
    build_pointmaze_datasets,
    canonical_pooled_latent,
)


def test_canonical_pooled_latent_preserves_feature_axis_for_token_layouts() -> None:
    token_first = torch.arange(16 * 384, dtype=torch.float32).reshape(16, 384)
    channel_first = token_first.transpose(0, 1)
    assert canonical_pooled_latent(token_first).shape == (384,)
    assert torch.allclose(canonical_pooled_latent(token_first), token_first.mean(dim=0))
    assert torch.allclose(canonical_pooled_latent(channel_first), channel_first.mean(dim=1))


def test_canonical_pooled_latent_fails_closed_on_ambiguous_or_wrong_shape() -> None:
    with pytest.raises(ValueError, match="unique feature axis"):
        canonical_pooled_latent(torch.zeros(16, 32))
    with pytest.raises(ValueError, match="unique feature axis"):
        canonical_pooled_latent(torch.zeros(384, 384))


def test_pointmaze_export_leaves_visual_transform_to_model_encode(tmp_path) -> None:
    calls = []

    class FakeDataset:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.transform = kwargs.get("transform")

    stats, observations = build_pointmaze_datasets(FakeDataset, tmp_path)

    assert stats is not observations
    assert calls == [
        {"data_path": str(tmp_path), "normalize_action": True},
        {
            "data_path": str(tmp_path),
            "transform": None,
            "normalize_action": False,
        },
    ]
    assert observations.transform is None
