from experiments.spectral_reachability import (
    p1_dev302_v11_audited_prefix_physical_20261101_codex as subject,
)


def test_v11_binds_v8_and_outcome_free_greedy():
    value = subject.build()
    rows = {row["role"]: row for row in value["roles"]}
    assert value["planner"]["name"] == subject.planner.NAME
    assert value["planner"]["prep_sha256"] == subject.PLAN_PREP_SHA
    assert value["planner"]["report_sha256"] == subject.PLAN_REPORT_SHA
    assert rows["selected"]["candidate_index"] == 16
    assert rows["same_bank_greedy"]["candidate_index"] == 4
    assert value["prefix_source"]["physical_outcomes_used_for_selection"] is False
    assert value["P1_admission"] is False
