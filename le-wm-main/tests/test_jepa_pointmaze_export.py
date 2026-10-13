from __future__ import annotations

import pytest
import torch

from experiments.theory_spectral_transfer.run_jepa_wm_pointmaze_temporal_eval import canonical_pooled_latent


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
