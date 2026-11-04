from experiments.spectral_reachability import p1_cp10_physics_oracle_saved_20261102 as saved


def test_oracle_winners_and_model_tails_replay():
    result = saved.analyze()
    assert len(result["rows"]) == 2
    assert all(r["candidate_successes"] == 0 for r in result["rows"])
    assert all(r["oracle"]["terminal"]["score"] < r["old"]["terminal"]["score"]
               and r["oracle"]["terminal"]["score"] < r["new"]["terminal"]["score"]
               for r in result["rows"])
    assert not result["P1_admission"] and not result["independent_confirmation"]
