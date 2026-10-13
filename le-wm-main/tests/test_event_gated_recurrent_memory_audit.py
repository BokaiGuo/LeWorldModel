import numpy as np

from experiments.empirical_spectral_audit.run_event_gated_recurrent_memory_audit import (
    _windows,
    build_report,
)


def _arrays() -> dict[str, object]:
    rng = np.random.default_rng(101)
    episode_id = np.repeat(np.arange(6), 5)
    source_frame = np.tile(np.arange(5), 6)
    latent = rng.normal(size=(30, 12))
    action = rng.normal(size=(30, 2))
    source_state = rng.normal(size=(30, 3))
    target_state = source_state + 0.1 * rng.normal(size=(30, 3))
    return {
        "source_latent": latent,
        "action": action,
        "source_state": source_state,
        "target_state": target_state,
        "episode_id": episode_id,
        "source_frame": source_frame,
        "metadata": {"environment": "fixture"},
    }


def test_recurrent_windows_use_declared_fit_only_state_interface():
    arrays = _arrays()
    projected = np.asarray(arrays["source_latent"])[:, :3]
    windows = _windows(arrays, 3, state_signal=projected)
    assert windows["sequence"].shape[-1] == 5
    assert windows["event_sequence"].shape[-1] == 7
    assert np.isfinite(windows["sequence"]).all()


def test_recurrent_report_is_available_with_equal_state_interface():
    report = build_report(
        _arrays(),
        split_seed=7,
        history_length=3,
        state_rank=3,
        hidden_dim=4,
        epochs=4,
        minimum_fit_episodes=3,
        minimum_fit_sequences=4,
    )
    assert report["status"] == "available"
    assert report["projection"]["retained_rank"] == 3
    assert report["recurrent"]["gru"]["parameter_count"] < 1000
    assert report["gates"]["mode_conditioned_branch"].startswith("unavailable")
