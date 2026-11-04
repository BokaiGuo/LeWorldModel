from experiments.spectral_reachability import p1_dev305_real_history_contrast_saved_20261102 as saved


def test_real_history_contrast_is_paired_and_replays_prior_scores():
    report = saved.analyze()
    assert len(report["roles"]) == 3
    for role in report["roles"]:
        assert role["real_history_lower_count"] + role["real_history_higher_count"] <= 18
        for counts in role["ordering"].values():
            assert sum(counts.values()) == 153
    assert not report["p1_admission"]
