from experiments.spectral_reachability import (
    p1_dev302_v5_contact_tail_physical_20261101_codex as subject,
)


def test_v5_physical_contract_binds_four_full_programs():
    value = subject.build()
    assert [row["role"] for row in value["roles"]] == list(subject.ROLES)
    assert all(len(row["actions"]) == 15 for row in value["roles"])
    assert value["same_replanning_budget"] is True
    assert value["post_cp5_physical_outcomes_used_for_selection"] is False
    assert value["P1_admission"] is False
