from experiments.empirical_spectral_audit.summarize_forward_dynamics_deep import _bootstrap


def test_deep_summary_bootstrap_is_episode_level_and_finite():
    result = _bootstrap([0.1, 0.2, 0.3, 0.4], seed=3, repeats=100)
    assert result["bootstrap_unit"] == "held-out episode"
    assert result["episodes"] == 4
    assert result["ci95"][0] <= result["mean"] <= result["ci95"][1]
