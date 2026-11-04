import numpy as np
import pytest

from experiments.pc_pointmaze_five_action_response_bank_20261103 import make_bank


def test_bank_has_bounded_paired_full_action_directions():
    jac = np.zeros((2, 10))
    jac[0, 0], jac[1, 1] = 2, 1
    bank, singular, directions = make_bank(jac)
    assert bank.shape == (9, 5, 2)
    np.testing.assert_allclose(singular, [2, 1])
    np.testing.assert_allclose(bank[1::2], -bank[2::2])
    np.testing.assert_allclose(jac @ directions[2:].T, 0, atol=1e-12)
    assert np.max(np.abs(bank)) == .5


def test_rank_and_finiteness_fail_closed():
    with pytest.raises(ValueError, match="rank below two"):
        make_bank(np.zeros((2, 10)))
    with pytest.raises(ValueError, match="finite"):
        make_bank(np.full((2, 10), np.nan))
