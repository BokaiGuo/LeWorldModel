import numpy as np

from experiments.pc_exact_action_identity_retention_dev_20261106_codex import heldout_predictions, select


def test_selection_shares_lookahead_anchor_and_does_not_mutate_data():
    scores = np.arange(64, dtype=float).reshape(8, 8)
    saved = scores.copy()
    result = select(scores, np.arange(8)[::-1], np.arange(8), 22)
    np.testing.assert_array_equal(scores, saved)
    assert result["parent"] == [7, 6]
    assert result["immediate"] == [0, 1]
    assert result["lookahead"] == [0, 1]
    assert result["descriptor"][0] == result["shuffled"][0] == 0
    assert all(len(set(indices)) == 2 for indices in result.values())


def test_heldout_uses_action_identity_and_not_heldout_scores():
    a = np.zeros((8, 10, 2))
    a[:, 0, 0] = np.arange(8)
    b = a[[2, 5]]
    costs = np.vstack([np.arange(8), -np.arange(8)]).astype(float)
    predictions = heldout_predictions(a, b, costs, np.array([10., 20.]))
    assert predictions["descriptor"].shape == (2, 2)
    assert predictions["descriptor"][0, 0] < predictions["descriptor"][0, 1]
    assert predictions["descriptor"][1, 0] > predictions["descriptor"][1, 1]
    np.testing.assert_array_equal(predictions["parent"][:, 0], [10., 20.])
