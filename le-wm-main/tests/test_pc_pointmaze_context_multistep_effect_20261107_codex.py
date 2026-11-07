import numpy as np
import pytest

from experiments.pc_pointmaze_context_multistep_effect_20261107_codex import action_bank, agreement


def test_fixed_bank_and_zero_action():
    bank = action_bank()
    assert bank.shape == (8, 25, 2) and np.max(np.abs(bank)) <= .4
    np.testing.assert_array_equal(bank[0], 0)
    np.testing.assert_array_equal(bank, action_bank())
    for start in range(0, 25, 5):
        np.testing.assert_array_equal(bank[:, start:start + 5], np.repeat(bank[:, start:start + 1], 5, axis=1))


def test_pairwise_rank_and_ties():
    a = np.arange(8.)
    assert agreement(a, a) == (28, 28)
    assert agreement(a, -a) == (0, 28)
    assert agreement(np.zeros(8), a) == (0, 0)
    with pytest.raises(ValueError):
        agreement([1], [1])
