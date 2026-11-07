from experiments.pc_scored_incumbent_real_policy_parity_20261107_codex import differences


def test_comparison_respects_state_tolerance_and_exact_actions():
    old = {"before": [1.0], "action": [0.2], "terminated": False}
    new = {"before": [1.0 + 1e-6], "action": [0.2], "terminated": False}
    found = []
    differences(old, new, "trace", found)
    assert not found
    new["action"] = [0.2 + 1e-6]
    differences(old, new, "trace", found)
    assert len(found) == 1 and "action" in found[0]


def test_comparison_detects_missing_keys_and_changed_length():
    found = []
    differences({"selected": [1, 2], "mode": "a"}, {"selected": [1]}, "audit", found)
    assert len(found) == 2
