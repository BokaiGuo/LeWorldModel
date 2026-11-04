from experiments.spectral_reachability.p1_dev305_component_vectors_saved_20261102 import analyze


def test_saved_actual_vectors_recompute():
    result = analyze()
    assert len(result["rows"]) == 3
    assert not result["independent_confirmation"]
    assert all(max(row["score_max_abs_difference_from_previous"].values()) <= 1e-12
               for row in result["rows"])
