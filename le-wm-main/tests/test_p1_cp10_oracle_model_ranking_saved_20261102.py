from experiments.spectral_reachability import p1_cp10_oracle_model_ranking_saved_20261102 as saved


def test_fixed_candidate_model_physics_ranking():
    result = saved.analyze()
    assert len(result["rows"]) == 2
    assert all(r["candidate_count"] == 64 for r in result["rows"])
    assert all(1 <= r["oracle_rank_by_model_competition"] <= 64 for r in result["rows"])
    assert not result["P1_admission"] and not result["independent_confirmation"]
