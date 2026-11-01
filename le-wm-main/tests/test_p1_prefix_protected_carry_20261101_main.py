import torch
import pytest
from experiments.spectral_reachability.p1_prefix_protected_carry_20261101_main import suffix_only_gradient


def test_only_suffix_updated_without_mutating_original():
    gradient = torch.ones((1, 3, 5, 2))
    masked = suffix_only_gradient(gradient)
    assert not masked[:, 0].any()
    assert torch.equal(masked[:, 1:], gradient[:, 1:])
    assert gradient[:, 0].all()


def test_wrong_horizon_rejected():
    with pytest.raises(ValueError):
        suffix_only_gradient(torch.ones((1, 2, 5, 2)))
