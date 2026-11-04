from experiments.spectral_reachability import p1_dev304_coherent_holdout_20261102 as subject


def test_development_protocol_is_not_independent():
    spec = subject.specification()
    assert spec["seed"] == 301000004
    assert spec["methods"] == ["original", "coherent"]
    assert spec["same_budget"]["predict_samples_per_arm"] == 3200
    assert not spec["p1_admission"] and not spec["independent_confirmation"]
    assert "unknown" in spec["classification"]
