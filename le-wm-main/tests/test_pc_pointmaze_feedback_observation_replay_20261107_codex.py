import numpy as np
import pytest

from experiments.pc_pointmaze_feedback_observation_replay_20261107_codex import score_comparison


def test_score_comparison_detects_winner_and_elite_changes():
    old = np.tile(np.arange(16, dtype=float), (3, 1))
    equal = score_comparison(old, old.copy())
    assert equal["winner_matches"] and all(equal["elite_set_matches"])
    changed = old.copy()
    changed[0, 0] = 100
    result = score_comparison(old, changed)
    assert not result["winner_matches"]
    assert result["elite_set_matches"] == [False, True, True]
    with pytest.raises(ValueError):
        score_comparison(old, np.zeros((3, 15)))
