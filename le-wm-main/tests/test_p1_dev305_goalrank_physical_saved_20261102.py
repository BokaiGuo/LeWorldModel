"""Regression checks on the sealed development physical panel."""

from experiments.spectral_reachability import p1_dev305_goalrank_physical_saved_20261102 as saved
from experiments.spectral_reachability import p1_dev305_goalrank_real_rgb_saved_20261102 as rgb_saved


def test_all_fixed_candidates_and_stop_states_are_verified():
    result = saved.analyze()
    assert len(result["rows"]) == 18
    assert result["frames_verified"] == 252
    assert {row["prefix"] for row in result["rows"]} == {"original", "coherent"}
    assert sum(row["selected"] for row in result["rows"]) == 2
    assert all(row["status"] == "fixed_horizon_completed" and row["tail_steps"] == 10
               and not row["terminated"] and not row["truncated"] for row in result["rows"])


def test_real_rgb_pair_analysis_covers_all_within_prefix_pairs():
    result = rgb_saved.analyze()
    assert len(result["rows"]) == 18
    assert len(result["pairs"]) == 72
    assert sum(result["pred_vs_real_latent"].values()) == 72
    assert sum(row["model_selected"] for row in result["rows"]) == 2
