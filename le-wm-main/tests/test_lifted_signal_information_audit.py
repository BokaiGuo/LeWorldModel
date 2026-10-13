import numpy as np

from experiments.empirical_spectral_audit.run_lifted_signal_information_audit import (
    causal_delay_time_frequency_features,
    gaussian_conditional_mutual_information,
    transfer_sufficiency_report,
)


def test_causal_signal_features_do_not_cross_episode_boundaries():
    signal = np.arange(24, dtype=float).reshape(12, 2)
    action = np.ones((12, 1))
    episode = np.asarray([0] * 6 + [1] * 6)
    frame = np.asarray(list(range(6)) * 2)
    features = causal_delay_time_frequency_features(signal, action, episode, frame, history_length=3)
    assert features["causal_window"] is True
    assert features["indices"].tolist() == [2, 3, 4, 5, 8, 9, 10, 11]
    assert features["base"].shape[0] == 8
    assert np.isfinite(features["time_frequency"]).all()
    # The spectral lift must retain the action interface of the base model.
    assert features["time_frequency"].shape[1] > features["base"].shape[1]
    assert np.allclose(features["time_frequency"][:, :3], features["base"][:, :3])
    assert features["delay_spectral"].shape[1] == (
        features["delay"].shape[1] + features["time_frequency"].shape[1]
    )


def test_conditional_information_and_heldout_sufficiency_are_finite():
    rng = np.random.default_rng(31)
    n = 160
    base = rng.normal(size=(n, 2))
    extra = rng.normal(size=(n, 2))
    target = (base[:, :1] + 2.0 * extra[:, :1] + 0.03 * rng.normal(size=(n, 1)))
    cmi = gaussian_conditional_mutual_information(target, extra, base)
    lifted = np.concatenate((base, extra), axis=1)
    fit = np.zeros(n, dtype=bool)
    fit[:100] = True
    outcome = ~fit
    report = transfer_sufficiency_report(target, base, lifted, fit, outcome)
    assert cmi["estimate_bits"] > 0.0
    assert report["lifted_beats_base"] is True
    assert np.isfinite(report["relative_rmse_reduction"])


def test_rank_matched_lift_is_fit_only_and_finite():
    rng = np.random.default_rng(7)
    n = 40
    base = rng.normal(size=(n, 3))
    lifted = rng.normal(size=(n, 19))
    target = rng.normal(size=(n, 2))
    fit = np.zeros(n, dtype=bool)
    fit[:20] = True
    outcome = ~fit
    report = transfer_sufficiency_report(
        target, base, lifted, fit, outcome, feature_rank=3
    )
    matched = report["fit_only_rank_matched"]
    assert matched["feature_rank"] == 3
    assert matched["base_projection"]["retained_rank"] == 3
    assert matched["lifted_projection"]["retained_rank"] == 3
    assert np.isfinite(matched["lifted_outcome_rmse"])
