import numpy as np
import pytest

from experiments.spectral_reachability.p1_uphill_adam_candidate_replay_20261031 import comparison


def test_comparison_records_action_and_score_differences():
    actions = [np.full((2, 5, 2), i / 4, dtype="<f4") for i in range(4)]
    saved = {"candidate_scores": [1, 2, 3, 4], "selected_index": 0,
             "selected_actions": actions[0].tolist()}
    exact = comparison([1, 2, 3, 4], 0, actions, saved)
    assert exact["same_winner"] and exact["same_winner_action_bytes"]
    changed = comparison([1.1, 2, 3, 4], 1, actions, saved)
    assert not changed["same_winner"] and not changed["same_winner_action_bytes"]
    assert changed["max_abs_score_delta"] == pytest.approx(0.1)


def test_comparison_rejects_invalid_scores():
    actions = [np.zeros((2, 5, 2), dtype="<f4")] * 4
    saved = {"candidate_scores": [1, 2, 3, 4], "selected_index": 0,
             "selected_actions": actions[0].tolist()}
    with pytest.raises(ValueError, match="finite"):
        comparison([1, np.nan, 3, 4], 0, actions, saved)
