import pytest
import torch

from experiments.pc_pointmaze_causal_residual_dev_20261106_codex import correct


def test_correction_uses_only_prior_residual():
    assert torch.equal(correct(torch.tensor([3., 4.]), torch.tensor([1., 2.]),
                               torch.tensor([2., 0.])), torch.tensor([4., 2.]))


def test_rejects_broadcast_and_nonfinite():
    with pytest.raises(ValueError):
        correct(torch.ones(2), torch.ones(1), torch.ones(2))
    with pytest.raises(ValueError):
        correct(torch.tensor([float("nan")]), torch.ones(1), torch.ones(1))
