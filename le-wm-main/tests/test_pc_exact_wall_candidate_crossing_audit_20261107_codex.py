import numpy as np
import pytest

from experiments.pc_exact_wall_candidate_crossing_audit_20261107_codex import summarize


def test_crossing_and_blocked_are_distinguished():
    task = {"start": [-.1, 0., 0.], "goal": [.5, 0.]}
    actions = np.zeros((2, 15, 2))
    actions[0, 0, 0] = 1
    result = summarize(task, actions, np.array([.6, .6]))
    assert result["crossed_count"] == 0
    assert result["any_blocked_count"] == 1


def test_score_mismatch_is_rejected():
    with pytest.raises(AssertionError):
        summarize({"start": [-.1, 0., 0.], "goal": [.5, 0.]}, np.zeros((1, 15, 2)),
                  np.array([0.]))
