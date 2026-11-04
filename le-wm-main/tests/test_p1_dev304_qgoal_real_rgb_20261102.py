from experiments.spectral_reachability import p1_dev304_qgoal_real_rgb_saved_20261102 as saved


def test_qgoal_rgb_panel_bindings():
    result = saved.analyze()
    assert len(result["rows"]) == 18
    assert len(result["pairs"]) == 72
    assert sum(row["model_selected"] for row in result["rows"]) == 2
    for key in ("pred_q_vs_real_q", "real_q_vs_position", "real_q_vs_angle", "real_q_vs_reward"):
        assert sum(result[key].values()) == 72
    assert all(row["linear_relative_error"] is None
               for row in result["rows"] if row["kind"] == "anchor")
    assert not result["p1_admission"] and not result["independent_confirmation"]
