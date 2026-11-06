from experiments.pc_action_opportunity_retention_absolute_probe_audit_20261104_codex import (
    select_absolute,
)


def test_absolute_probe_accounts_for_candidate_specific_baseline():
    candidates = [{"immediate_score": 0.0}, {"immediate_score": 1.0}, {"immediate_score": 2.0}]
    descriptors = [
        {"baseline_score": 1.0, "a_improvements": [0.0]},
        {"baseline_score": 10.0, "a_improvements": [5.0]},
        {"baseline_score": 1.0, "a_improvements": [0.5]},
    ]
    assert select_absolute(candidates, descriptors, 2) == [0, 2]
