import numpy as np
import pytest

from experiments.pc_tworoom_pairwise_tolerance_audit_20261107_codex import counts
from experiments.pc_tworoom_within_prefix_pairwise_posthoc_20261107_codex import pair_counts


def test_zero_tolerance_reproduces_existing_code():
    rng = np.random.default_rng(7)
    x, y = rng.normal(size=(2, 8, 8))
    x[0, 1] = x[0, 0]
    mask = np.array([True, False] * 4)
    assert counts(x, y, mask, 0, 0) == pair_counts(x, y, mask)


def test_near_ties_do_not_claim_order():
    x = np.tile(np.arange(8, dtype=float), (8, 1))
    y = x.copy()
    x[0, 1] = x[0, 0] + 0.001
    y[0, 1] = y[0, 0] + 0.00001
    mask = np.array([True] + [False] * 7)
    no_tol = counts(x, y, mask, 0, 0)
    tol = counts(x, y, mask, 0.002, 0.00002)
    assert no_tol["agree"] == 28
    assert tol["agree"] == 27 and tol["both_tie"] == 1


def test_invalid_tolerance_rejected():
    with pytest.raises(ValueError):
        counts(np.zeros((8, 8)), np.zeros((8, 8)), np.ones(8, dtype=bool), -1, 0)
