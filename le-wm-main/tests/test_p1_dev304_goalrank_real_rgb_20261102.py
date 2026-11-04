from experiments.spectral_reachability import p1_dev304_goalrank_real_rgb_saved_20261102 as saved


def test_saved_panel_integrity():
    panel = saved.analyze()
    assert len(panel["rows"]) == 18
    assert len(panel["pairs"]) == 72
    assert sum(row["model_selected"] for row in panel["rows"]) == 2
    assert all(row["linear_relative_error"] is None
               for row in panel["rows"] if row["kind"] == "anchor")
    for key in ("pred_vs_real_latent", "real_latent_vs_position",
                "real_latent_vs_angle", "real_latent_vs_reward"):
        assert sum(panel[key].values()) == 72
    assert not panel["independent_confirmation"]
    assert not panel["p1_admission"]
