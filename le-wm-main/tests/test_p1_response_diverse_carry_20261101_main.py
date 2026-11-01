import numpy as np
import pytest
from experiments.spectral_reachability.p1_response_diverse_carry_20261101_main import response_bank


def test_multiple_response_directions_and_fixed_roster():
    jac = np.zeros((192, 30))
    jac[:30] = np.diag(np.arange(30, 0, -1))
    bank, singular = response_bank(jac, np.eye(192), np.ones(192), np.zeros(192))
    assert bank.shape == (8, 3, 5, 2)
    assert bank.dtype == np.float32
    assert abs(bank).max() <= 1
    assert np.linalg.matrix_rank(bank.reshape(8, -1)) >= 3
    for i in (2, 4, 6):
        np.testing.assert_array_equal(bank[i], -bank[i + 1])
    assert singular[0] == 30


def test_zero_response_and_invalid_inputs():
    bank, _ = response_bank(np.zeros((192, 30)), np.eye(192), np.ones(192), np.zeros(192))
    assert not bank.any()
    with pytest.raises(ValueError):
        response_bank(np.full((192, 30), np.nan), np.eye(192), np.ones(192), np.zeros(192))
