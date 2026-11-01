from experiments.spectral_reachability import (
    p1_dev302_mixed_prefix_physical_20261101_codex as subject,
)


def test_contract_has_selected_zero_greedy_random_and_equal_horizon():
    value = subject.build()
    assert [row["role"] for row in value["roles"]] == list(subject.ROLES)
    assert all(len(row["actions"]) == 5 for row in value["roles"])
    assert value["feedback_boundary"] == 5
    assert value["same_physical_horizon"] is True
    assert value["post_prefix_outcomes_used_for_planning"] is False
    assert value["P1_admission"] is False
