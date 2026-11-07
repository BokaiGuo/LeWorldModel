import numpy as np
import pytest

from experiments import pc_tworoom_common_tail_retention_20261107_codex as trial


def test_retention_uses_only_a_and_preserves_anchor():
    costs = np.tile([3., 4., 5., 6., 7.], (13, 1))
    costs[3] = [0., 2., 4., 6., 8.]
    costs[9] = [2., 1., 1., 1., 30.]
    kept = trial.retain(costs, np.arange(13))
    assert kept["lookahead"][0] == 3
    assert kept["descriptor"][0] == 3
    assert kept["descriptor"][1] == 9
    assert kept["descriptor"] == kept["shuffled"]
    assert all(len(set(v)) == 2 for v in kept.values())


def test_b_scores_outside_retained_do_not_affect_selection():
    scores = np.full((13, 4), 10.)
    scores[2, 1] = 1.
    assert trial.final_choice(scores, [2, 5]) == [2, 1]
    scores[12, 0] = -100.
    assert trial.final_choice(scores, [2, 5]) == [2, 1]
    with pytest.raises(ValueError):
        trial.final_choice(scores, [2, 2])


def test_common_tails_are_action_identical_and_legal(monkeypatch):
    geometry = (np.array([80., 80.]), np.array([145., 80.]), 112., 50.)
    monkeypatch.setattr(wide_geometry := trial.wide.repair.v2.v1, "geometry",
                        lambda current, goal: geometry)
    prefixes, tails = trial.libraries(None, None)
    matrix = trial.chains(prefixes, tails)
    assert matrix.shape == (13, 9, 25, 2)
    assert np.max(np.abs(matrix)) <= 1.
    np.testing.assert_array_equal(matrix[0, 5, 10:], matrix[12, 5, 10:])
    np.testing.assert_array_equal(tails[4], np.zeros((15, 2)))
