from experiments.spectral_reachability import p1_dev305_triprefix_midfeedback_saved_20261102 as saved


def test_sealed_feedback_execution_roster_and_budget():
    result = saved.analyze()
    assert [row["arm"] for row in result["rows"]] == ["candidate_02", "candidate_00", "zero"]
    assert all(row["status"] == "fixed_horizon_completed" for row in result["rows"])
    assert result["p1_admission"] is False
    assert result["full_cost_certified"] is False
