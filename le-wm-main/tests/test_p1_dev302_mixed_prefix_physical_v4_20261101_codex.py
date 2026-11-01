from experiments.spectral_reachability import (
    p1_dev302_mixed_prefix_physical_v4_20261101_codex as subject,
)


def test_v4_physical_contract_binds_new_planner_and_four_roles():
    value = subject.build()
    assert value["schema"] == subject.NAME + ".contract.v1"
    assert value["planner"]["name"] == subject.planner.NAME
    assert value["planner"]["prep_sha256"] == subject.PLAN_PREP_SHA
    assert value["planner"]["report_sha256"] == subject.PLAN_REPORT_SHA
    assert [row["candidate_index"] for row in value["roles"]] == [5, 0, 4, 7]
    assert [row["candidate_label"] for row in value["roles"]] == [
        "rgb_goal_support_top_route",
        "zero",
        "rgb_goal_support_direct",
        "random_control",
    ]
    assert value["prefix_source"]["simulator_state_used_for_prefix_generation"] is False
    assert value["P1_admission"] is False
