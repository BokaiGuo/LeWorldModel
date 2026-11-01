from experiments.spectral_reachability import (
    p1_dev302_v9_task_greedy_prefix_physical_20261101_codex as subject,
)


def test_v9_selected_is_distinct_from_rgb_task_greedy():
    value = subject.build()
    rows = {row["role"]: row for row in value["roles"]}
    assert rows["selected"]["candidate_index"] == 16
    assert rows["same_bank_greedy"]["candidate_index"] == 4
    assert rows["selected"]["candidate_index"] != rows["same_bank_greedy"]["candidate_index"]
    assert value["planner"]["prep_sha256"] == subject.PLAN_PREP_SHA
    assert value["planner"]["report_sha256"] == subject.PLAN_REPORT_SHA
    assert value["prefix_source"]["mode"] == "rgb_only"
    assert value["prefix_source"]["simulator_state_used_for_prefix_generation"] is False
    assert value["prefix_source"]["physical_outcomes_used_for_selection"] is False
    assert value["P1_admission"] is False
