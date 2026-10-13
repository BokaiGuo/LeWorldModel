import numpy as np

from experiments.empirical_spectral_audit.run_horizon_conditioned_transfer_sufficiency import (
    build_report,
)


def _arrays() -> dict[str, object]:
    rng = np.random.default_rng(505)
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


def test_summary_contract_reports_non_deployed_selector():
    report = build_report(
        _arrays(),
        split_seed=9,
        history_length=4,
        horizon=4,
        feature_rank=4,
        instability_trials=2,
        minimum_fit_episodes=3,
        stability_threshold=1.1,
    )
    assert report["deployment_policy"]["variant"] == "fourier_plus_phase"
    assert report["selector_diagnostic"]["selection_is_not_deployed"] is True
