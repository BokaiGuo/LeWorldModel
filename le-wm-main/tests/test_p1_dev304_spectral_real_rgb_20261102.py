from experiments.spectral_reachability import p1_dev304_spectral_real_rgb_saved_20261102 as saved


def test_spectral_endpoint_alignment_panel():
    result = saved.analyze()
    assert len(result["rows"]) == 18 and len(result["pairs"]) == 72
    assert all(row["linear_relative_error"] is None for row in result["rows"] if row["kind"] == "anchor")
    assert result["pred_vs_real_latent"]["agreement"] + result["pred_vs_real_latent"]["disagreement"] + result["pred_vs_real_latent"]["ties"] == 72
    assert not result["independent_confirmation"] and not result["p1_admission"]
