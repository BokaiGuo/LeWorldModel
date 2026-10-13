import numpy as np

from experiments.empirical_spectral_audit.run_horizon_conditioned_transfer_sufficiency import (
    build_report,
    declared_horizon_policy,
)


def _arrays() -> dict[str, object]:
    rng = np.random.default_rng(404)
    episode_id = np.repeat(np.arange(8), 6)
    source_frame = np.tile(np.arange(6), 8)
    latent = rng.normal(size=(48, 5))
    action = rng.normal(size=(48, 2))
    source_state = rng.normal(size=(48, 3))
    target_state = source_state + 0.1 * rng.normal(size=(48, 3))
    return {
        "source_latent": latent,
        "action": action,
        "source_state": source_state,
        "target_state": target_state,
        "episode_id": episode_id,
        "source_frame": source_frame,
        "metadata": {"environment": "fixture"},
    }


def test_horizon_conditioned_report_never_deploys_selector():
    report = build_report(
        _arrays(),
        split_seed=9,
        history_length=4,
        horizon=4,
        feature_rank=4,
        instability_trials=4,
        minimum_fit_episodes=3,
        stability_threshold=1.1,
    )
    assert report["status"] == "available"
    assert report["protocol"]["carrier"] == "fixed_fourier"
    assert report["gates"]["selector_uses_outcome"] is False
    assert report["deployment_policy"]["fail_closed"] is True
    assert report["deployment_policy"]["selector_used"] is False
    assert report["deployment_policy"]["variant"] == "fourier_plus_phase"


def test_horizon_conditioned_report_has_all_declared_arms():
    report = build_report(
        _arrays(),
        split_seed=10,
        history_length=4,
        horizon=1,
        feature_rank=4,
        instability_trials=3,
        minimum_fit_episodes=3,
    )
    assert set(report["candidate_reports"]) == {
        "fixed_fourier",
        "spectral_phase",
        "fourier_plus_phase",
        "fourier_gated_event_residual",
    }
    assert report["frozen_horizon_policy"]["variant"] == "fixed_fourier"


def test_multistep_candidate_policy_is_declared_not_selector_tuned():
    assert declared_horizon_policy(1, "carrier_plus_phase_for_multistep_v2") == "fixed_fourier"
    assert declared_horizon_policy(2, "carrier_plus_phase_for_multistep_v2") == "fourier_plus_phase"
    assert declared_horizon_policy(4, "carrier_plus_phase_for_multistep_v2") == "fourier_plus_phase"
