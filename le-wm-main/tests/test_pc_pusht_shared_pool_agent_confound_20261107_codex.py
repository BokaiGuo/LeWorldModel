import numpy as np
import pytest

from experiments.pc_pusht_shared_pool_agent_confound_20261107_codex import matched_pairs, pair_result


def test_matched_pairs_respect_object_wrap_and_agent_gap():
    states = np.array([[0, 0, 2, 2, np.pi - .02, 0, 0],
                       [30, 0, 2, 2, -np.pi + .02, 0, 0],
                       [60, 0, 30, 30, 0, 0, 0]], dtype=float)
    closer, farther, _ = matched_pairs(states, np.zeros(7))
    assert closer.tolist() == [0]
    assert farther.tolist() == [1]
    assert pair_result(np.array([2., 3., 0.]), closer, farther)["closer_lower"] == 1


def test_invalid_state_and_score_ties():
    with pytest.raises(ValueError, match="state contract"):
        matched_pairs(np.zeros((3, 6)), np.zeros(7))
    result = pair_result(np.array([1., 1.]), np.array([0]), np.array([1]))
    assert result == {"closer_lower": 0, "farther_lower": 0, "ties": 1}
