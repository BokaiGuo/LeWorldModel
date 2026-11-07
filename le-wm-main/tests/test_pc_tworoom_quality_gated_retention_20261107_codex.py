import numpy as np
import pytest

from experiments.pc_tworoom_quality_gated_retention_20261107_codex import choose_retained, select


def test_gated_second_is_in_top_four_and_both_rules_match_first():
    costs = np.arange(8, dtype=float)[:, None] * np.ones((1, 8))
    costs[7, 7] += 100
    costs[3, 7] += 10
    selected = choose_retained(costs, 3)
    assert selected["lookahead"] == [0, 1]
    assert selected["vector"] == [0, 7]
    assert selected["gated"] == [0, 3]
    assert selected["shuffled"][0] == 0
    assert selected["shuffled"][1] in (1, 2, 3)


def test_selection_cannot_use_unqueried_b_and_keeps_incumbent():
    a = np.ones((8, 8)) * 4
    ids = np.arange(16)
    b = np.ones(16) * 3
    assert select(a, b, ids, [0, 1], [2, 6]) == (2, "incumbent", 0)
    b[3] = 1
    assert select(a, b, ids, [0, 1], [2, 6]) == (1, "b", 3)
    with pytest.raises(ValueError, match="missing"):
        select(a, b, ids, [0, 2], [2, 6])


def test_invalid_score():
    a = np.ones((8, 8))
    a[0, 0] = np.nan
    with pytest.raises(ValueError):
        choose_retained(a, 1)
