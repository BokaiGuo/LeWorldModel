from experiments.spectral_reachability import (
    p1_dev302_v13_audited_tail_physical_20261101_codex as subject,
)


def test_v13_physical_contract_is_equal_budget_and_audited():
    value = subject.build()
    assert [row["candidate_index"] for row in value["roles"]] == [16, 0, 4, 7]
    assert value["same_replanning_budget"] is True
    assert value["same_feedback_count"] is True
    assert value["P1_admission"] is False
