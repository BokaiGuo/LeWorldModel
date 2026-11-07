import copy

import numpy as np
import pytest

from experiments.pc_rgb_cache_feedback_replay_20261107_codex import compare_decisions


def example():
    return {"rounds": [{"costs": [1, 2, 3, 4, 5]} for _ in range(3)],
            "chosen": 1, "actions": np.zeros((5, 2)), "incumbent_cost": 8.0,
            "chosen_cost": 1.0}


def test_decision_comparison_detects_elite_and_witness_changes():
    baseline = example()
    assert compare_decisions(baseline, copy.deepcopy(baseline), baseline)["best_gap"] == 0
    changed = copy.deepcopy(baseline)
    changed["actions"][0, 0] = 0.1
    with pytest.raises(ValueError, match="chosen"):
        compare_decisions(baseline, changed, baseline)


def test_decision_comparison_rejects_score_drift():
    baseline = example()
    changed = copy.deepcopy(baseline)
    changed["rounds"][0]["costs"][0] = 9
    with pytest.raises(ValueError, match="scores"):
        compare_decisions(baseline, changed, baseline)
