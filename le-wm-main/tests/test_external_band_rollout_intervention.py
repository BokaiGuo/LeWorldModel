import numpy as np
import torch

from experiments.empirical_spectral_audit.run_external_band_rollout_intervention import (
    _apply_band_dropout,
)


def test_band_dropout_removes_only_selected_coordinates_in_identity_basis():
    prediction = torch.arange(192, dtype=torch.float32).reshape(1, 192)
    mean = torch.zeros(192)
    basis = torch.eye(192)
    result = _apply_band_dropout(prediction, mean, basis, (0, 8))
    assert torch.allclose(result[:, :8], torch.zeros(1, 8))
    assert torch.allclose(result[:, 8:], prediction[:, 8:])


def test_none_intervention_is_identity():
    prediction = torch.randn(2, 192)
    result = _apply_band_dropout(prediction, torch.zeros(192), torch.eye(192), None)
    assert np.allclose(result.numpy(), prediction.numpy())
