from experiments.spectral_reachability.pc_recursive_action_effect_dev import action_effect_choice


def test_effect_choice_respects_absolute_cost_window_and_protected_slot():
    records = [
        {"candidate": 0, "probe_min": 0.4, "improvements": [0.0, 0.0]},
        {"candidate": 1, "probe_min": 0.43, "improvements": [0.2, 0.2]},
        {"candidate": 2, "probe_min": 0.8, "improvements": [9.0, 9.0]},
    ]
    assert action_effect_choice(records, [0], "effect_min") == (1, "min")
    assert action_effect_choice(records, [0], "effect_complement") == (1, "effect")


def test_effect_choice_falls_back_when_protected_not_measured():
    records = [{"candidate": 1, "probe_min": 0.4, "improvements": [0.0, 0.0]},
               {"candidate": 2, "probe_min": 0.6, "improvements": [3.0, 3.0]}]
    assert action_effect_choice(records, [0], "effect_complement") == (1, "fallback_greedy_not_probed")


def test_effect_can_choose_different_qualified_candidate():
    records = [
        {"candidate": 0, "probe_min": 0.4, "improvements": [0.0, 0.0]},
        {"candidate": 1, "probe_min": 0.41, "improvements": [0.01, 0.0]},
        {"candidate": 2, "probe_min": 0.44, "improvements": [0.2, 0.1]},
    ]
    assert action_effect_choice(records, [0], "effect_min") == (1, "min")
    assert action_effect_choice(records, [0], "effect_complement") == (2, "effect")
