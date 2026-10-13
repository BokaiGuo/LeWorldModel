import numpy as np

from experiments.theory_spectral_transfer.action_interventions import (
    derange_action_blocks,
    derangement_indices,
    intervene_future_action_blocks,
)

from experiments.theory_spectral_transfer import (
    TransferConfig,
    controllability_alignment_sweep_report,
    band_energy_summary,
    fit_loglog_slope,
    heat_filter,
    heat_filter_recovery_report,
    identifiability_controllability_report,
    operator_alignment_report,
    operator_alignment_theorem_report,
    oblique_frame_stress_report,
    residual_band_penalty,
    sigreg_local_anisotropy_report,
    spectral_transfer_report,
)
from experiments.theory_spectral_transfer.run_jepa_operator_anchor import (
    _commutator_norm,
    _fit_operator,
    _random_rotation,
)
from experiments.theory_spectral_transfer.analyze_action_intervention_statistics import (
    holm_adjust,
    paired_stats,
)


def test_oracle_transfer_slope_matches_expected_alpha():
    config = TransferConfig(
        dim=256,
        intrinsic_dim=4.0,
        input_exponent=2.0,
        encoder_exponent=1.0,
    )

    report = spectral_transfer_report(config)
    alpha = report["slopes"]["latent_eigenvalues"]["1:end"]["alpha"]

    assert abs(alpha - report["expected_oracle_alpha"]) < 1e-10


def test_default_theory_report_has_nontrivial_residual_spectrum():
    report = spectral_transfer_report(TransferConfig())
    residual_stats = report["spectrum_stats"]["residual_eigenvalues"]

    assert residual_stats["trace"] > 0.0
    assert residual_stats["lambda_max"] > 0.0
    assert np.isfinite(report["slopes"]["residual_eigenvalues"]["1:end"]["alpha"])
    assert "operator_alignment" in report
    assert "operator_alignment_theorem" in report


def test_operator_alignment_report_reveals_noncommutativity_peak():
    report = operator_alignment_report(
        TransferConfig(dim=64, input_exponent=1.0, encoder_exponent=1.0, predictor_exponent=1.0),
        angles=[0, 15, 30, 45, 60, 75, 90],
        horizons=[1, 2, 4, 8],
        ks=[4, 8, 16, 32],
        random_trials=32,
        seed=0,
    )

    contrast = report["contrast"]
    peak = max(report["records"], key=lambda row: row["commutator_norm"])

    assert contrast["aligned_angle_degrees"] == 0.0
    assert contrast["peak_oblique_angle_degrees"] == 45.0
    assert contrast["aligned_commutator_norm"] < contrast["peak_oblique_commutator_norm"]
    assert contrast["endpoint_commutator_norm"] < contrast["peak_oblique_commutator_norm"]
    assert report["random_null"]["commutator_norm"]["mean"] > 0.0
    assert peak["angle_degrees"] == 45.0


def test_action_derangement_is_deterministic_bijective_and_has_no_fixed_points():
    for count in (2, 3, 21, 200):
        first = derangement_indices(count, seed=17)
        second = derangement_indices(count, seed=17)
        assert first == second
        assert sorted(first) == list(range(count))
        assert all(index != mapped for index, mapped in enumerate(first))


def test_within_episode_block_derangement_preserves_prefix_and_multiset():
    blocks = np.arange(24, dtype=np.float32).reshape(6, 4)
    import torch

    result, permutation = derange_action_blocks(torch.tensor(blocks), seed=4, episode_index=3, preserve_prefix=2)
    np.testing.assert_array_equal(result[:2].numpy(), blocks[:2])
    np.testing.assert_array_equal(np.sort(result[2:].numpy(), axis=0), np.sort(blocks[2:], axis=0))
    assert sorted(permutation) == list(range(4))
    assert all(i != j for i, j in enumerate(permutation))


def test_future_action_interventions_preserve_history_prefix():
    import torch

    blocks = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    zeroed = intervene_future_action_blocks(blocks, "future_zero", preserve_prefix=2)
    reversed_future = intervene_future_action_blocks(blocks, "future_reverse", preserve_prefix=2)
    np.testing.assert_array_equal(zeroed[:2].numpy(), blocks[:2].numpy())
    np.testing.assert_array_equal(reversed_future[:2].numpy(), blocks[:2].numpy())
    np.testing.assert_array_equal(zeroed[2:].numpy(), np.zeros((3, 4), dtype=np.float32))
    np.testing.assert_array_equal(reversed_future[2:].numpy(), blocks[2:].flip(0).numpy())


def test_holm_adjustment_is_monotone_and_controls_familywise_error():
    adjusted = holm_adjust([0.001, 0.02, 0.03, 0.8])
    assert adjusted == [0.004, 0.06, 0.06, 0.8]


def test_small_sample_sign_flip_uses_exact_enumeration():
    import torch

    stats = paired_stats(torch.tensor([1.0, 2.0, 3.0]), 200, 50, seed=0)
    assert stats["sign_flip_method"] == "exact_enumeration"
    assert stats["sign_flip_permutations_evaluated"] == 8
    assert stats["sign_flip_p_two_sided"] == 0.25


def test_operator_alignment_theorem_matches_numeric_and_flat_nulls():
    theorem = operator_alignment_theorem_report(
        TransferConfig(dim=64, input_exponent=1.0, encoder_exponent=1.0, predictor_exponent=1.0),
        angles=[0, 15, 30, 45, 60, 75, 90],
    )
    sweep = operator_alignment_report(
        TransferConfig(dim=64, input_exponent=1.0, encoder_exponent=1.0, predictor_exponent=1.0),
        angles=[0, 15, 30, 45, 60, 75, 90],
        horizons=[1, 2, 4, 8],
        ks=[4, 8, 16, 32],
        random_trials=16,
        seed=0,
    )

    numeric = {str(row["angle_degrees"]): row["commutator_norm"] for row in sweep["records"]}
    exact = theorem["theorem"]["exact_by_angle"]

    for angle, value in exact.items():
        assert np.isclose(value, numeric[angle], rtol=1e-10, atol=1e-12)
    assert theorem["theorem"]["peak_angle_degrees"] == 45.0
    assert all(value == 0.0 for value in theorem["falsifiers"]["flat_latent_spectrum"].values())
    assert all(value == 0.0 for value in theorem["falsifiers"]["flat_predictor_spectrum"].values())


def test_jepa_operator_anchor_fit_preserves_spectrum_under_basis_rotation():
    rng = np.random.default_rng(3)
    source = rng.normal(size=(24, 10))
    target = source @ np.diag(np.linspace(0.2, 0.9, 10)).T + 0.01 * rng.normal(size=(24, 10))
    _, basis, _, covariance, operator, x, y = _fit_operator(source[:18], target[:18], rank=6, ridge=1e-2)

    assert basis.shape == (10, 6)
    assert covariance.shape == (6, 6)
    assert operator.shape == (6, 6)
    assert x.shape == y.shape == (18, 6)
    rotation = _random_rotation(6, rng)
    rotated = rotation @ operator @ rotation.T
    np.testing.assert_allclose(
        np.sort_complex(np.linalg.eigvals(operator)),
        np.sort_complex(np.linalg.eigvals(rotated)),
    )
    assert _commutator_norm(covariance, operator) >= 0.0


def test_multi_horizon_report_shows_residual_growth():
    report = spectral_transfer_report(TransferConfig(), horizons=[1, 2, 4, 8])
    growth = report["multi_horizon"]["growth"]

    assert growth["predicted_monotone_nonincreasing"]
    assert growth["residual_monotone_nondecreasing"]
    assert growth["residual_trace"]["8"] > growth["residual_trace"]["1"]
    assert growth["predicted_trace"]["8"] < growth["predicted_trace"]["1"]


def test_broken_transfer_reports_different_head_and_tail_slopes():
    config = TransferConfig(
        dim=128,
        intrinsic_dim=2.0,
        input_exponent=1.0,
        encoder_exponent=0.5,
        tail_encoder_exponent=3.0,
        breakpoint=32,
    )

    report = spectral_transfer_report(config, slope_windows=[(1, 24), (64, 128)])
    head = report["slopes"]["latent_eigenvalues"]["1:24"]["alpha"]
    tail = report["slopes"]["latent_eigenvalues"]["64:128"]["alpha"]

    assert tail > head


def test_heat_filter_suppresses_high_rate_modes_more():
    values = np.ones(4)
    rates = np.array([1.0, 2.0, 4.0, 8.0])

    filtered = heat_filter(values, rates, time=0.2)

    assert filtered[0] > filtered[1] > filtered[2] > filtered[3]


def test_heat_filter_recovery_reports_high_r2_for_known_generator():
    config = TransferConfig(dim=64, intrinsic_dim=2.0, input_exponent=1.0, encoder_exponent=1.0)

    report = heat_filter_recovery_report(config, times=[0.01, 0.02])

    assert report["median_r2"] > 0.999


def test_oblique_frame_stress_reports_finite_window_bias():
    config = TransferConfig(dim=96, intrinsic_dim=2.0, input_exponent=1.0, encoder_exponent=1.0)

    report = oblique_frame_stress_report(config, obliqueness=0.5, seed=7, slope_windows=[(1, 16), (48, 96)])

    assert report["frame_condition_number"] > 1.0
    assert set(report["finite_window_alpha_bias"]) == {"1:16", "48:96"}


def test_sigreg_toy_hides_local_anisotropy_in_global_covariance():
    report = sigreg_local_anisotropy_report(dim=24, cells=12, anisotropy=20.0, seed=0)

    assert report["global_over_local_stable_rank"] > 1.0


def test_identifiability_controllability_toy_separates_alignment():
    report = identifiability_controllability_report(dim=32)

    assert report["aligned_controllability_lift"] > report["mismatched_controllability_lift"]


def test_controllability_alignment_sweep_separates_angle_and_null():
    report = controllability_alignment_sweep_report(
        TransferConfig(dim=32, input_exponent=1.5, predictor_exponent=1.0),
        controllable_dim=4,
        angles=[0, 45, 90],
        ks=[4, 8],
        samples=256,
        random_trials=16,
        seed=0,
    )

    assert report["contrast"]["aligned_top_k_lift"] > report["contrast"]["orthogonal_top_k_lift"]
    assert report["random_rotation_null"]["top_band_lift"]["8"]["mean"] > 0.0
    assert report["records"][0]["prediction_residual_energy_share"] < report["records"][-1]["prediction_residual_energy_share"]


def test_band_energy_summary_normalizes_mass():
    profile = band_energy_summary(np.array([4.0, 2.0, 1.0, 1.0]), bands=[(0, 1), (1, 4)])

    assert profile["0:1"] == 0.5
    assert abs(sum(profile.values()) - 1.0) < 1e-12


def test_residual_band_penalty_uses_weights_and_bands():
    residuals = np.zeros((5, 4))
    residuals[:, 0] = 2.0
    residuals[:, 3] = 1.0
    basis = np.eye(4)

    penalty = residual_band_penalty(
        residuals,
        basis,
        bands=[(0, 1), (3, 4)],
        weights=[0.5, 2.0],
    )

    assert penalty == 4.0


def test_fit_loglog_slope_handles_zero_tail_values():
    values = np.array([4.0, 1.0, 0.0, 0.0])

    slope = fit_loglog_slope(values)

    assert np.isfinite(slope["alpha"])
