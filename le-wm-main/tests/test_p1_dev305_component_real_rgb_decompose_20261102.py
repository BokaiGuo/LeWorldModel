from experiments.spectral_reachability.p1_dev305_component_real_rgb_decompose_20261102 import analyze


def test_sealed_actual_vectors_and_pair_roster():
    result = analyze("3de74b0aea62b665cb0e0963bad34a5ba7cd4ba654aae99fea199a3ef9bac45c")
    assert len(result["rows"]) == 3
    assert not result["independent_confirmation"]
    for row in result["rows"]:
        for metric in row["metrics"].values():
            counts = metric["predicted_vs_real_score"]
            assert sum(counts.values()) == 289 * 288 // 2
        assert row["actual_mse_max_abs_difference"] < 1e-6
