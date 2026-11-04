import numpy as np
import pytest

from experiments.pc_causal_mixed_prefix_search_20261103 import candidates, select


def test_mixed_candidates_are_deterministic_bounded_and_distinct():
    jac = np.zeros((2, 10))
    jac[0, 0], jac[1, 1] = 2, 1
    bank, singular = candidates(jac, 123)
    assert bank.shape == (2, 4, 5, 2)
    assert np.max(np.abs(bank)) <= .5
    np.testing.assert_array_equal(bank, candidates(jac, 123)[0])
    assert not np.allclose(bank[0], bank[1])
    assert np.linalg.norm(bank[1, :, :, :].reshape(4, 10)[:, :2]) < np.linalg.norm(bank[0].reshape(4, 10)[:, :2])
    np.testing.assert_allclose(singular, [2, 1])
    with pytest.raises(ValueError):
        candidates(np.zeros((2, 10)), 123)


def test_nonprogress_does_not_force_a_selection():
    first = [1., .7, 1.1, 1.2, .8, .9]
    assert select(first, [.5, .4, .45, .49, .3, .2], 1.)[:2] == (5, None)
    assert select(first, [.5, .4, .3, .6, .8, .9], 1.)[:2] == (2, 2)
    with pytest.raises(ValueError):
        select(first, [1.] * 5, 1.)
