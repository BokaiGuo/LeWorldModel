from experiments.spectral_reachability import (
    p1_dev302_v4_four_prefix_tail_physical_20261101_codex as subject,
)


def test_contract_binds_four_equal_budget_replanned_tails():
    value = subject.build()
    assert [row["role"] for row in value["roles"]] == list(subject.ROLES)
    assert [row["candidate_index"] for row in value["roles"]] == [5, 0, 4, 7]
    assert all(len(row["actions"]) == 15 for row in value["roles"])
    assert value["feedback_boundary"] == 5
    assert value["physical_horizon"] == 15
    assert value["same_replanning_budget"] is True
    assert value["same_feedback_count"] is True
    assert value["post_cp5_physical_outcomes_used_for_selection"] is False
    assert value["P1_admission"] is False
