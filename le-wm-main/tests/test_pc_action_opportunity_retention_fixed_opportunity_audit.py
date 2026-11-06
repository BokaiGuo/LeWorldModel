from experiments.pc_action_opportunity_retention_fixed_opportunity_audit_20261104_codex import (
    _select,
)


def test_retention_preserves_greedy_and_selects_requested_policy():
    candidates = [
        {"immediate_score": 0.0},
        {"immediate_score": 1.0},
        {"immediate_score": 2.0},
    ]
    descriptors = [
        {"a_max_improvement": 0.0, "a_improvements": [0.0, 0.0]},
        {"a_max_improvement": 2.0, "a_improvements": [1.0, 0.0]},
        {"a_max_improvement": 1.0, "a_improvements": [1.0, 3.0]},
    ]
    assert _select(candidates, descriptors, "probe_min", 2) == [0, 1]
    assert _select(candidates, descriptors, "effect_complement", 2) == [0, 2]


def test_farthest_point_supports_more_than_two_slots():
    candidates = [{"immediate_score": i} for i in range(4)]
    descriptors = [
        {"a_improvements": [0.0, 0.0]},
        {"a_improvements": [1.0, 0.0]},
        {"a_improvements": [0.0, 2.0]},
        {"a_improvements": [3.0, 3.0]},
    ]
    result = _select(candidates, descriptors, "effect_complement", 3)
    assert result[0] == 0
    assert len(set(result)) == 3
