from experiments.spectral_reachability import (
    p1_dev302_v13_audited_tail_replan_20261101_codex as subject,
)


def test_v13_uses_audited_v11_prefixes_and_equal_budget():
    value = subject.protocol()
    assert value["schema"] == subject.NAME + ".protocol.v1"
    assert [row["candidate_index"] for row in value["roles"]] == [16, 0, 4, 7]
    assert value["equal_cem_and_robust_budget"] is True
    assert value["post_cp5_physical_outcomes_used"] is False
    assert value["P1_admission"] is False
