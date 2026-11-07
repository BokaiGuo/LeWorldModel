import numpy as np
import pytest

from experiments import pc_conditional_cem_batch_replay_20261107_codex as audit
from experiments import pc_rgb_conditional_temporal_cem_20261107_codex as src


def test_elites_and_choice_detect_real_changes():
    inc = np.zeros((2, 25, 2), dtype=np.float32)
    actions = np.zeros((src.ROUNDS * src.N, 25, 2), dtype=np.float32)
    actions[0, -1, 0] = 0.5
    inc_scores = np.full((2, 5), 20.0)
    scores = np.full((src.ROUNDS * src.N, 5), 10.0)
    first = audit.decisions(inc, actions, inc_scores, scores, "endpoint")
    scores[10, -1] = 1.0
    second = audit.decisions(inc, actions, inc_scores, scores, "endpoint")
    difference = audit.compare(first, second)
    assert difference["elite_set_changed"] == [True, False, False]
    assert difference["choice_index_changed"]
    assert difference["choice_action_changed"]


def test_ties_can_change_index_without_action_change():
    inc = np.zeros((2, 25, 2), dtype=np.float32)
    actions = np.zeros((src.ROUNDS * src.N, 25, 2), dtype=np.float32)
    inc_scores = np.full((2, 5), 10.0)
    scores = np.full((src.ROUNDS * src.N, 5), 20.0)
    first = audit.decisions(inc, actions, inc_scores, scores, "temporal")
    inc_scores[1, -1] = 1.0
    second = audit.decisions(inc, actions, inc_scores, scores, "temporal")
    diff = audit.compare(first, second)
    assert diff["choice_index_changed"] and not diff["choice_action_changed"]


def test_reject_nonfinite_scores():
    inc = np.zeros((2, 25, 2))
    actions = np.zeros((src.ROUNDS * src.N, 25, 2))
    scores = np.zeros((src.ROUNDS * src.N, 5))
    scores[0, 0] = np.nan
    with pytest.raises(ValueError):
        audit.decisions(inc, actions, np.zeros((2, 5)), scores, "endpoint")
