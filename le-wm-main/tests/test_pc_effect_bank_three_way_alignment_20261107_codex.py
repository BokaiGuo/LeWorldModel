import numpy as np
import pytest

from experiments.pc_effect_bank_three_way_alignment_20261107_codex import counts


def test_counts_ties_and_prefix_mask():
    a = np.ones((8, 8))
    b = a.copy()
    b[0, 0] = -1
    b[0, 1] = 0
    eligible = np.ones(8, dtype=bool)
    eligible[1] = False
    assert counts(a, b, eligible) == [54, 55]


def test_counts_fail_closed():
    with pytest.raises(ValueError):
        counts(np.ones((8, 7)), np.ones((8, 8)), np.ones(8))
    with pytest.raises(ValueError):
        counts(np.full((8, 8), np.nan), np.ones((8, 8)), np.ones(8))
    assert counts(np.zeros((8, 8)), np.ones((8, 8)), np.ones(8, dtype=bool)) == [0, 0]
