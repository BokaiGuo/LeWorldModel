import numpy as np
import pytest

from experiments.pc_scored_candidate_selection_20261107_codex import ScoredCandidate, select_scored
from experiments.pc_tworoom_parent_competition_replay_20261107_codex import select_all


def test_all_scored_parent_can_win_without_unqueried_b():
    a = np.full((8, 8), 10.0)
    parents_incumbents = np.array([1.0] + [9.0] * 7 + [8.0, 7.0])
    ids = np.array(list(range(8)) + list(range(8, 16)))
    result = select_all(a, parents_incumbents, ids, np.full(16, 4.0), [0, 1])
    assert (result.source, result.index) == ("parent", 0)
    with pytest.raises(ValueError, match="missing"):
        select_all(a, parents_incumbents, ids, np.full(16, 4.0), [0, 2])


def test_ignores_other_arm_b_and_preserves_tie_order():
    a = np.full((8, 8), 10.0)
    a[0, 0] = 1
    ids = np.arange(24)
    b = np.full(24, 5.0)
    b[16] = -10
    scores = np.full(10, 1.0)
    result = select_all(a, scores, ids, b, [0, 1])
    assert (result.source, result.index) == ("a", 0)


def test_rejects_duplicate_identity_and_nonfinite():
    with pytest.raises(ValueError):
        select_scored([ScoredCandidate("a", 0, 1), ScoredCandidate("a", 0, 2)])
    with pytest.raises(ValueError):
        select_scored([ScoredCandidate("a", 0, np.nan)])
