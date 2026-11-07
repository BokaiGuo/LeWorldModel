import numpy as np

from experiments.pc_tworoom_rgb_action_effect_alignment_20261107_codex import ACTIONS, pair_agreement, ranking_agreement


def test_action_bank_and_sign_ties():
    assert ACTIONS.shape == (5, 2)
    assert np.all(np.abs(ACTIONS) <= 1)
    assert np.array_equal(ACTIONS[0], [0, 0])
    assert pair_agreement([1, -1, 0], [2, 1, 0]) == {"n": 3, "matching_sign": 2, "ties_either": 1}


def test_ranking_counts_and_reversal():
    scores = np.tile(np.arange(5, dtype=float), (5, 1))
    assert ranking_agreement(scores, scores) == {"n": 50, "matching_sign": 50, "ties_either": 0}
    assert ranking_agreement(scores, -scores) == {"n": 50, "matching_sign": 0, "ties_either": 0}
