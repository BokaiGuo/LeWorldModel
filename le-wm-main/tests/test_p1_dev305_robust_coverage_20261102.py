from experiments.spectral_reachability import p1_dev305_robust_coverage_saved_20261102 as saved


def test_rescore_and_saved_candidate_coverage():
    result = saved.analyze()
    assert len(result["rows"]) == 54
    assert all(row["exact_sampled"] for row in result["rows"] if row["name"] == "cem_selected")
    assert not result["independent_confirmation"] and not result["p1_admission"]
