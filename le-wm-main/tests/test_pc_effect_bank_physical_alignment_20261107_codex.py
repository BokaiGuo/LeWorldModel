import numpy as np
import pytest

from experiments.pc_effect_bank_physical_alignment_20261107_codex import sign_stats


def test_sign_stats_excludes_ties_and_unsuccessful_stratum():
    predicted = np.ones((8, 8))
    actual = np.ones((8, 8))
    actual[0, 0] = 0
    predicted[0, 1] = -1
    eligible = np.zeros(8, dtype=bool)
    eligible[0] = True
    assert sign_stats(predicted, actual, eligible) == (6, 7)


def test_sign_stats_rejects_wrong_dimensions():
    with pytest.raises(ValueError):
        sign_stats(np.zeros((8, 7)), np.zeros((8, 8)), np.ones(8, bool))
