import pytest
import torch

from experiments.spectral_reachability.p1_official_rollout_parity_20261031 import compare


def test_comparison_tracks_both_time_boundaries():
    native = torch.zeros((1, 1, 5, 192))
    pred5 = torch.zeros((1, 1, 192))
    pred10 = torch.zeros((1, 1, 192))
    assert compare(pred5, pred10, native) == (0.0, 0.0)
    native[0, 0, 3, 8] = 2
    native[0, 0, 4, 9] = -3
    assert compare(pred5, pred10, native) == (2.0, 3.0)


def test_comparison_rejects_wrong_shape_and_nonfinite():
    prediction = torch.zeros((1, 1, 192))
    with pytest.raises(ValueError, match="shape"):
        compare(prediction, prediction, torch.zeros((1, 1, 4, 192)))
    native = torch.zeros((1, 1, 5, 192))
    native[0, 0, 4, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        compare(prediction, prediction, native)
