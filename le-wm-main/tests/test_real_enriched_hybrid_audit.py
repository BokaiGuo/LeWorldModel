import json

import numpy as np

from experiments.empirical_spectral_audit.enriched_record_schema import make_payload
from experiments.empirical_spectral_audit.run_real_enriched_hybrid_audit import (
    build_real_report,
    _contiguous_current_indices,
    _fit_memory_windows,
    _evaluate_memory_windows,
    _memory_action_parts,
    episode_disjoint_split,
    load_enriched_arrays,
)


def _payload(tmp_path):
    rng = np.random.default_rng(22)
    rows = []
    for episode in range(6):
        latent = rng.normal(size=3)
        for frame in range(5):
            action = rng.normal(size=(2, 2))
            next_latent = 0.8 * latent + np.asarray([action[0, 0], action[0, 1], 0.5 * action[0, 0]])
            state = np.asarray([*latent[:2], 0.1 * latent[2]])
            next_state = np.asarray([*next_latent[:2], 0.1 * next_latent[2]])
            rows.append({
                "sample_id": f"{episode}:{frame}",
                "episode_id": episode,
                "source_frame": frame,
                "source_latent": latent.tolist(),
                "source_state_raw": state.tolist(),
                "source_proprio_raw": state.tolist(),
                "action_blocks_raw": action.tolist(),
                "action_blocks_normalized": action.tolist(),
                "latent_by_horizon": {
                    "1": {
                        "predicted": next_latent.tolist(),
                        "target": next_latent.tolist(),
                        "target_state_raw": next_state.tolist(),
                    }
                },
            })
            latent = next_latent
    payload = make_payload(
        architecture="fixture",
        checkpoint="fixture",
        environment="Fixture",
        seed="0",
        training_stage=1,
        latent_dimension=3,
        action_distribution="fixture",
        split={"name": "all", "episode_ids": list(range(6))},
        horizons=[1],
        enriched_records=rows,
        runtime="test",
    )
    source = tmp_path / "fixture.json"
    source.write_text(json.dumps(payload))
    return source


def test_real_loader_and_report_are_episode_disjoint_and_finite(tmp_path):
    source = _payload(tmp_path)
    arrays = load_enriched_arrays(source)
    fit, outcome, split = episode_disjoint_split(arrays["episode_id"], fit_fraction=0.5)
    assert len(fit) == 15
    assert len(outcome) == 15
    assert not set(arrays["episode_id"][fit]).intersection(set(arrays["episode_id"][outcome]))
    report = build_real_report(arrays, fit_fraction=0.5, horizons=(1,), memory_lengths=(1, 2))
    assert report["protocol"]["fit_only_estimation"] is True
    assert report["protocol"]["mode_labels"] == "unavailable; no contact mode inferred"
    assert np.isfinite(report["outcome"]["latent_next_rmse_projected"])
    assert report["memory_closure"]["1"]["status"] == "fit"


def test_memory_pairing_uses_same_current_transitions():
    episode = np.repeat(np.arange(2), 4)
    frame = np.tile(np.arange(4), 2)
    indices = np.arange(8)
    paired = _contiguous_current_indices(episode, frame, indices, memory_length=3)
    # The first two rows of each episode cannot be current rows for a
    # three-step history; both episodes must contribute the same support.
    assert paired.tolist() == [2, 3, 6, 7]


def test_seeded_episode_split_is_reproducible_and_disjoint():
    episode = np.repeat(np.arange(10), 2)
    first = episode_disjoint_split(episode, fit_fraction=0.6, split_seed=17)
    second = episode_disjoint_split(episode, fit_fraction=0.6, split_seed=17)
    assert first[2] == second[2]
    assert not set(episode[first[0]]).intersection(set(episode[first[1]]))
    assert first[2]["split_order"] == "seeded_episode_permutation"


def test_memory_action_history_is_causal_and_matches_fit_features():
    rng = np.random.default_rng(7)
    episode = np.repeat(np.arange(2), 5)
    frame = np.tile(np.arange(5), 2)
    eta = rng.normal(size=(10, 3))
    action = rng.normal(size=(10, 2))
    target = rng.normal(size=(10, 3))
    indices = np.arange(10)
    fit = _fit_memory_windows(
        eta, target, action, episode, frame, indices,
        memory_length=3, ridge=1e-4, include_action_history=True,
    )
    assert fit["include_action_history"] is True
    assert fit["feature_dimension"] == 3 * 3 + 2 * 3 + 1
    value = _evaluate_memory_windows(
        fit, eta, target, action, episode, frame, indices,
        include_action_history=True,
    )
    assert np.isfinite(value)


def test_summary_action_history_is_low_dimensional_and_causal():
    action = np.asarray([[1.0, 2.0], [3.0, 5.0], [8.0, 13.0]])
    parts = _memory_action_parts(action, [0, 1, 2], mode="summary")
    assert len(parts) == 3
    assert sum(part.size for part in parts) == 6
    np.testing.assert_allclose(parts[0], action[2])
    np.testing.assert_allclose(parts[1], np.mean(action, axis=0))
    np.testing.assert_allclose(parts[2], action[2] - action[1])


def test_fourier_residual_state_history_is_causal_and_finite():
    from experiments.empirical_spectral_audit.run_real_enriched_hybrid_audit import _memory_state_parts

    eta = np.asarray([[0.0, 1.0], [1.0, 3.0], [2.0, 2.0], [4.0, 6.0]])
    parts = _memory_state_parts(eta, [0, 1, 2, 3], mode="fourier_residual")
    assert len(parts) == 3
    assert parts[0].shape == (2,)
    assert all(np.isfinite(part).all() for part in parts)
    np.testing.assert_allclose(parts[0], eta[3])


def test_fourier_memory_fit_and_evaluation_use_same_feature_interface():
    rng = np.random.default_rng(23)
    episode = np.repeat(np.arange(2), 6)
    frame = np.tile(np.arange(6), 2)
    eta = rng.normal(size=(12, 3))
    action = rng.normal(size=(12, 2))
    target = rng.normal(size=(12, 3))
    fit = _fit_memory_windows(
        eta, target, action, episode, frame, np.arange(12), memory_length=4,
        ridge=1e-4, action_history_mode="summary", state_history_mode="fourier_residual",
    )
    value = _evaluate_memory_windows(
        fit, eta, target, action, episode, frame, np.arange(12),
        action_history_mode="summary", state_history_mode="fourier_residual",
    )
    assert fit["state_history_mode"] == "fourier_residual"
    assert fit["action_history_mode"] == "summary"
    assert np.isfinite(value)


def test_adaptive_time_frequency_lift_exposes_legacy_and_event_contracts():
    from experiments.empirical_spectral_audit.run_lifted_signal_information_audit import (
        adaptive_time_frequency_features,
    )

    rng = np.random.default_rng(31)
    episode = np.repeat(np.arange(2), 5)
    frame = np.tile(np.arange(5), 2)
    signal = rng.normal(size=(10, 3))
    action = rng.normal(size=(10, 2))
    features = adaptive_time_frequency_features(signal, action, episode, frame, history_length=3)
    assert features["legacy_first_harmonic"].shape[0] == 6
    assert features["event_aware"].shape[0] == 6
    assert features["feature_contract"]["threshold_source"] == "none"
    assert np.isfinite(features["event_score"]).all()


def test_fit_only_standardized_projection_is_finite_and_fit_only():
    from experiments.empirical_spectral_audit.run_lifted_signal_information_audit import (
        fit_only_standardized_feature_projection,
    )

    rng = np.random.default_rng(32)
    features = rng.normal(size=(8, 12))
    projected, metadata = fit_only_standardized_feature_projection(
        features, np.asarray([True] * 5 + [False] * 3), rank=3
    )
    assert projected.shape == (8, 3)
    assert metadata["method"] == "fit_only_standardized_feature_pca"
    assert np.isfinite(projected).all()


def test_fit_only_memory_selector_is_inner_fit_only_and_fail_closed():
    from experiments.empirical_spectral_audit.run_lifted_signal_information_audit import (
        fit_only_select_transfer_variant,
    )

    rng = np.random.default_rng(33)
    episode = np.repeat(np.arange(8), 2)
    target = rng.normal(size=(16, 2))
    candidates = {
        "event": rng.normal(size=(16, 7)),
        "fourier": rng.normal(size=(16, 7)),
    }
    fit = np.isin(episode, np.arange(6))
    outcome = ~fit
    result = fit_only_select_transfer_variant(
        target, candidates, episode, fit, outcome, feature_rank=3, split_seed=9
    )
    assert result["status"] == "pass"
    assert result["selected_variant"] in candidates
    assert result["selection_outcome_leakage"] is False
    assert set(result["inner_fit_episode_ids"]).isdisjoint(result["inner_validation_episode_ids"])

    insufficient = fit_only_select_transfer_variant(
        target, candidates, episode, np.isin(episode, [0, 1, 2]), outcome, feature_rank=3
    )
    assert insufficient["status"] == "insufficient_support"
    assert insufficient["selected_variant"] is None


def test_adaptive_summary_keeps_selector_and_fourier_gates_separate(tmp_path):
    from experiments.empirical_spectral_audit.summarize_adaptive_time_frequency_memory_audit import (
        VARIANTS,
        summarize,
    )

    paths = []
    for split_seed, selected, compact_reduction, legacy_reduction in (
        (1, "legacy_first_harmonic", 0.20, 0.30),
        (2, "compact_event_aware", 0.20, 0.30),
    ):
        variants = {}
        for name in VARIANTS:
            reduction = compact_reduction if name == "compact_event_aware" else legacy_reduction if name == "legacy_first_harmonic" else 0.01
            variants[name] = {
                "fit_only_rank_matched": {
                    "relative_rmse_reduction": reduction,
                    "lifted_beats_base": True,
                    "base_outcome_rmse": 10.0,
                    "lifted_outcome_rmse": 10.0 * (1.0 - reduction),
                }
            }
        report = {
            "split": {"split_seed": split_seed},
            "support": {"fit_rows": 4, "outcome_rows": 2},
            "variants": variants,
            "fit_only_selection": {
                "status": "pass",
                "selected_variant": selected,
                "outer_selected_report": variants[selected]["fit_only_rank_matched"],
            },
        }
        path = tmp_path / f"split_{split_seed}.json"
        path.write_text(json.dumps(report))
        paths.append(path)
    summary = summarize(paths)
    assert summary["fit_only_selected"]["status"] == "pass"
    assert summary["fit_only_selected"]["wins_against_base"] == 2
    assert summary["gates"]["fit_only_selector_beats_base"]["status"] == "pass"
    assert summary["gates"]["fit_only_selector_beats_historical_first_harmonic"]["status"] == "fail_or_partial"
