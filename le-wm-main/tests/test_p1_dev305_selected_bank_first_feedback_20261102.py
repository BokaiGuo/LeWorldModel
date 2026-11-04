from experiments.spectral_reachability import p1_dev305_selected_bank_first_feedback_saved_20261102 as saved


def test_feedback_saved_chain_and_visibility():
    result = saved.analyze()
    assert [r["arm"] for r in result["rows"]] == ["candidate_02", "candidate_00", "zero"]
    assert all(all(frame["eligible"] for frame in row["feedback_visibility"]) for row in result["rows"])
    assert not result["p1_admission"]
    assert not result["full_cost_certified"]
