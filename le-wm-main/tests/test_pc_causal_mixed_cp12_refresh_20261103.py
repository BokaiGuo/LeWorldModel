import pytest
import torch

from experiments.pc_causal_mixed_cp12_refresh_20261103 import mse


def test_mse_requires_same_finite_tensor_shape():
    assert mse(torch.tensor([1., 3.]), torch.tensor([0., 1.])) == 2.5
    with pytest.raises(ValueError):
        mse(torch.ones(2), torch.ones(3))
    with pytest.raises(ValueError):
        mse(torch.tensor([float("nan")]), torch.zeros(1))
