import numpy as np
import torch

from experiments.empirical_spectral_audit import (
    EmpiricalAuditConfig,
    build_condition_labels,
    build_tangent_families,
    empirical_spectral_report,
    empirical_operator_alignment_bridge_report,
    empirical_operator_alignment_paper_outline,
    empirical_operator_alignment_paper_spine,
    load_latent_arrays,
    multi_horizon_report,
    norm_matched_random_tangents,
    nuisance_contamination_ratio,
    observability_probe_report,
    planning_projected_tangent_energy,
    prediction_spectral_error,
    projected_covariance_lift,
    rank_fraction_bands,
    run_empirical_audit,
    run_intervention_audit,
    spectral_exploration_report,
    spectral_planner_distance,
    spectral_planner_metric_report,
    spectral_planner_weights,
    spectral_shift_report,
    uncertainty_residual_report,
)
from experiments.empirical_spectral_audit.regularizers import directional_regularizer_losses
from experiments.empirical_spectral_audit.metrics import controllability_spectrum_report
from experiments.directional_metrics import compute_eigenspace
from experiments.directional_metrics import matched_eigenvalue_random_rotation_like
from experiments.io_utils import to_numpy


def test_prediction_spectral_error_is_zero_for_matching_tangents():
    rng = np.random.default_rng(0)
    latents = rng.normal(size=(128, 8)).astype("float32")
    tangents = rng.normal(size=(128, 8)).astype("float32")
    eig = compute_eigenspace(latents)

    pse = prediction_spectral_error(
        to_numpy(eig.eigenvectors),
        tangents,
        tangents.copy(),
        rank_fraction_bands(8),
    )

    assert pse["mse"] == 0.0
    assert pse["profile_l1"] == 0.0
    assert pse["mean_cosine"] > 0.999


def test_ppte_lift_is_high_for_aligned_direction():
    rng = np.random.default_rng(1)
    tangents = np.zeros((256, 6), dtype="float32")
    tangents[:, 0] = rng.normal(size=256)
    direction = np.array([1, 0, 0, 0, 0, 0], dtype="float32")

    ppte = planning_projected_tangent_energy(tangents, direction)

    assert ppte["share"] > 0.99
    assert ppte["lift"] > 5.9


def test_projected_covariance_lift_detects_amplified_latent_axis():
    rng = np.random.default_rng(2)
    latents = rng.normal(size=(512, 6)).astype("float32")
    latents[:, 0] *= 5.0

    ppte = projected_covariance_lift(latents, np.array([1, 0, 0, 0, 0, 0], dtype="float32"))

    assert ppte["lift"] > 4.0


def test_norm_matched_random_tangents_preserve_row_norms():
    rng = np.random.default_rng(3)
    tangents = rng.normal(size=(64, 5)).astype("float32")

    control = norm_matched_random_tangents(tangents, seed=4)

    assert np.allclose(np.linalg.norm(control, axis=1), np.linalg.norm(tangents, axis=1))


def test_matched_eigenvalue_random_rotation_like_preserves_spectrum_shape():
    rng = np.random.default_rng(13)
    tangents = rng.normal(size=(4096, 6)).astype("float32")
    tangents *= np.array([6.0, 4.0, 2.0, 1.0, 0.5, 0.25], dtype="float32")

    control = matched_eigenvalue_random_rotation_like(tangents, seed=14)

    source = compute_eigenspace(tangents).eigenvalues.cpu().numpy()
    target = compute_eigenspace(control).eigenvalues.cpu().numpy()
    source = source / source.sum()
    target = target / target.sum()

    assert np.allclose(source, target, atol=0.08)


def test_nuisance_contamination_reports_planning_purity():
    basis = np.eye(4, dtype="float32")
    planning = np.zeros((64, 4), dtype="float32")
    nuisance = np.zeros((64, 4), dtype="float32")
    planning[:, 0] = 1.0
    nuisance[:, 3] = 1.0

    ncr = nuisance_contamination_ratio(basis, planning, nuisance, k=1)

    assert ncr["planning_share"] > 0.99
    assert ncr["nuisance_share"] < 0.01
    assert ncr["planning_purity"] > 1e6


def test_empirical_report_global_detects_dynamics_alignment_over_controls():
    rng = np.random.default_rng(5)
    n = 768
    d = 12
    latents = rng.normal(size=(n, d)).astype("float32")
    latents[:, :2] *= 5.0
    dynamics = np.zeros((n, d), dtype="float32")
    dynamics[:, :2] = rng.normal(size=(n, 2))
    planning = dynamics.copy()
    nuisance = np.zeros((n, d), dtype="float32")
    nuisance[:, -2:] = rng.normal(size=(n, 2))
    predicted = dynamics + 0.01 * rng.normal(size=(n, d)).astype("float32")

    report = empirical_spectral_report(
        {
            "latents": latents,
            "dynamics": dynamics,
            "planning": planning,
            "nuisance": nuisance,
            "predicted_dynamics": predicted,
        },
        condition="global",
        ks=[2],
        min_cell_size=32,
        seed=0,
    )

    dyn_lift = report["weighted_mean"]["lifts"]["dynamics"]["2"]
    random_lift = report["weighted_mean"]["controls"]["norm_matched_random_dynamics_lift"]["2"]
    pse = report["weighted_mean"]["prediction_spectral_error"]
    ncr = report["weighted_mean"]["ncr"]["2_planning_purity"]

    assert dyn_lift > 5.0
    assert random_lift < 2.0
    assert pse["mean_cosine"] > 0.99
    assert ncr > 10.0


def test_conditioned_controllability_reports_variance_controllability_gap():
    rng = np.random.default_rng(51)
    n, d = 256, 6
    latents = rng.normal(size=(n, d)).astype("float32")
    latents[:128, 0] *= 5.0
    latents[128:, 1] *= 5.0
    actions = rng.normal(size=(n, 2)).astype("float32")
    dynamics = np.zeros((n, d), dtype="float32")
    dynamics[:, 0] = actions[:, 0]
    dynamics[:, 1] = actions[:, 1]
    report = controllability_spectrum_report(
        latents,
        {"dynamics": dynamics},
        actions,
        ks=(k for k in (1, 2)),
        labels=np.repeat(np.arange(2), 128),
        min_cell_size=64,
    )

    assert report["conditioning"] == "provided_labels"
    assert report["num_cells"] == 2
    assert report["cell_summary"]["r2_min"] > 0.99
    assert all("dynamics_minus_controllable_lift" in cell for cell in report["cells"])


def test_load_arrays_and_config_api_support_old_latents_schema(tmp_path):
    rng = np.random.default_rng(6)
    latents = rng.normal(size=(64, 6)).astype("float32")
    dynamics = rng.normal(size=(64, 6)).astype("float32")
    path = tmp_path / "latents.npz"
    np.savez_compressed(path, latents=latents, dynamics=dynamics)

    arrays = load_latent_arrays(path)
    report = run_empirical_audit(
        arrays,
        EmpiricalAuditConfig(condition="global", ks=(2,), min_cell_size=16, include_interventions=False),
    )

    assert report["schema"]["required"] == ["latents"]
    assert "dynamics" in report["schema"]["available_tangents"]


def test_load_arrays_supports_deterministic_row_sampling(tmp_path):
    rng = np.random.default_rng(61)
    latents = rng.normal(size=(100, 6)).astype("float32")
    dynamics = rng.normal(size=(100, 6)).astype("float32")
    path = tmp_path / "latents.npz"
    np.savez_compressed(path, latents=latents, dynamics=dynamics)

    arrays = load_latent_arrays(path, max_samples=17, seed=5)
    again = load_latent_arrays(path, max_samples=17, seed=5)

    assert arrays["latents"].shape == (17, 6)
    assert np.array_equal(arrays["_sample_index"], again["_sample_index"])


def test_condition_builders_support_goal_and_phase_labels():
    rng = np.random.default_rng(7)
    arrays = {
        "latents": rng.normal(size=(80, 4)).astype("float32"),
        "dynamics": rng.normal(size=(80, 4)).astype("float32"),
        "goals": np.repeat(np.arange(4), 20)[:, None].astype("float32"),
        "episode_id": np.repeat(np.arange(8), 10).astype("int64"),
        "timestep": np.arange(80).astype("float32"),
    }

    goal_labels = build_condition_labels(arrays, EmpiricalAuditConfig(condition="goals", clusters=4))
    episode_labels = build_condition_labels(arrays, EmpiricalAuditConfig(condition="episode", clusters=8))
    phase_labels = build_condition_labels(arrays, EmpiricalAuditConfig(condition="phase", clusters=4))

    assert np.unique(goal_labels).size == 4
    assert np.unique(episode_labels).size == 8
    assert np.unique(phase_labels).size == 4


def test_multi_horizon_report_detects_noisy_horizon_collapse():
    rng = np.random.default_rng(8)
    latents = rng.normal(size=(512, 8)).astype("float32")
    latents[:, :2] *= 4
    h1 = np.zeros((512, 8), dtype="float32")
    h1[:, :2] = rng.normal(size=(512, 2))
    h3 = rng.normal(size=(512, 8)).astype("float32")

    report = multi_horizon_report(latents, {"dynamics": h1, "dynamics_h3": h3}, ks=[2])

    assert report["horizon_collapse"]["dynamics_h3"]["2"] < 1.0


def test_observability_probe_reports_physical_variable_r2():
    rng = np.random.default_rng(9)
    latents = rng.normal(size=(256, 6)).astype("float32")
    latents[:, 0] *= 5.0
    physical_x = latents[:, :1] * 2.0

    report = observability_probe_report(latents, {"physical_x": physical_x}, ks=[1, 3])

    assert report["physical_x"]["3"] > 0.95


def test_run_empirical_audit_reports_label_controls_and_interventions():
    rng = np.random.default_rng(10)
    n = 256
    d = 8
    latents = rng.normal(size=(n, d)).astype("float32")
    latents[:, :2] *= 3
    dynamics = np.zeros((n, d), dtype="float32")
    dynamics[:, :2] = rng.normal(size=(n, 2))
    dynamics_h3 = np.zeros((n, d), dtype="float32")
    dynamics_h3[:, :2] = 0.5 * dynamics[:, :2] + 0.1 * rng.normal(size=(n, 2))
    nuisance = np.zeros((n, d), dtype="float32")
    nuisance[:, -2:] = rng.normal(size=(n, 2))
    arrays = {
        "latents": latents,
        "dynamics": dynamics,
        "dynamics_h3": dynamics_h3,
        "predicted_dynamics_h3": dynamics_h3.copy(),
        "planning": dynamics.copy(),
        "nuisance": nuisance,
        "actions": rng.normal(size=(n, 2)).astype("float32"),
        "physical_state": latents[:, :1] * 2.0,
        "counterfactual_wrong": rng.normal(size=(n, d)).astype("float32"),
    }

    report = run_empirical_audit(
        arrays,
        EmpiricalAuditConfig(condition="global", ks=(2,), horizons=(1, 3), min_cell_size=32, include_interventions=True),
    )

    assert "state_cluster_shuffle" in report["label_controls"]
    assert "action_shuffle" in report["label_controls"]
    assert "counterfactual_impossible_transition" in report["label_controls"]
    assert "dynamics_h3" in report["multi_horizon"]
    assert report["observability"]["physical_state"]["2"] > 0.9
    assert "band_dropout" in report["planner_interventions"]
    assert "warnings" in report
    assert not any("multi-horizon" in warning for warning in report["warnings"])
    assert not any("observability" in warning for warning in report["warnings"])
    assert not any("counterfactual" in warning for warning in report["warnings"])


def test_uncertainty_residual_report_summarizes_residual_bands():
    rng = np.random.default_rng(11)
    latents = rng.normal(size=(128, 6)).astype("float32")
    residual = rng.normal(size=(128, 6)).astype("float32")

    report = uncertainty_residual_report(latents, {"prediction_residual": residual}, ks=[2])

    assert report["prediction_residual"]["norm_p90"] > 0
    assert "residual_band_fingerprint" in report["prediction_residual"]


def test_spectral_planner_metric_weights_downweight_nuisance_tail():
    basis = np.eye(6, dtype="float32")
    planning = np.zeros((128, 6), dtype="float32")
    dynamics = np.zeros((128, 6), dtype="float32")
    nuisance = np.zeros((128, 6), dtype="float32")
    planning[:, 0] = 1.0
    dynamics[:, 0] = 1.0
    nuisance[:, -1] = 1.0

    weights = spectral_planner_weights(basis, planning, dynamics, nuisance)
    dist = spectral_planner_distance(np.eye(6, dtype="float32"), basis, weights)

    assert weights[0] > weights[-1]
    assert dist[0] > dist[-1]


def test_spectral_exploration_report_returns_score_components():
    rng = np.random.default_rng(12)
    latents = rng.normal(size=(128, 6)).astype("float32")
    dynamics = rng.normal(size=(128, 6)).astype("float32")
    residual = rng.normal(size=(128, 6)).astype("float32")

    report = spectral_exploration_report(
        latents,
        {"dynamics": dynamics, "prediction_residual": residual, "planning": dynamics},
        k=2,
    )

    assert report["score_p90"] >= report["score_mean"]
    assert "sci_mean" in report["components"]


def test_spectral_shift_report_detects_subspace_change():
    rng = np.random.default_rng(13)
    source = rng.normal(size=(256, 6)).astype("float32")
    target = rng.normal(size=(256, 6)).astype("float32")
    source[:, 0] *= 5.0
    target[:, -1] *= 5.0

    report = spectral_shift_report(source, target, k=1)

    assert report["spectrum_l1"] >= 0.0
    assert report["topk_subspace_overlap"] < 0.5


def test_empirical_operator_alignment_bridge_summarizes_rivals_and_transport():
    audit = {
        "condition": "states",
        "config": {"num_samples": 1000, "num_cells": 8, "ks": [4, 8, 16, 32]},
        "warnings": [],
        "weighted_mean": {
            "lifts": {
                "dynamics": {"32": 4.0},
                "planning": {"32": 3.5},
                "predicted_dynamics": {"32": 4.1},
                "prediction_residual": {"32": 2.8},
            },
            "controls": {
                "norm_matched_random_dynamics_lift": {"32": 1.0},
                "random_basis_dynamics_lift": {"32": 0.95},
                "matched_eigenvalue_random_rotation_dynamics_lift": {"32": 0.98},
                "isotropic_gaussian_dynamics_lift": {"32": 0.99},
            },
        },
        "label_controls": {
            "state_cluster_shuffle": {"weighted_mean": {"lifts": {"dynamics": {"32": 1.2}}}},
            "time_shuffle": {"weighted_mean": {"lifts": {"dynamics": {"32": 1.1}}}},
            "action_shuffle": {"weighted_mean": {"lifts": {"dynamics": {"32": 1.05}}}},
        },
        "eigenspace_transport": {"k": 4, "transitions": {"0->1": 0.1, "1->2": 0.05}},
        "spectral_planner_metric": {
            "weights": [1.0, 1.0, 1.0, 1.0],
            "weight_summary": {"top_weight_mean": 1.2, "tail_weight_mean": 0.8, "mean": 1.0, "min": 0.8, "max": 1.2},
        },
        "controllability": {"action_to_dynamics_r2": 0.7},
    }
    theorem = {"theorem": {"peak_angle_degrees": 45.0, "statement": "same-spectrum law"}, "claim_boundary": "toy"}

    report = empirical_operator_alignment_bridge_report(audit, theorem=theorem, k=32, null_trials=64, seed=0)

    assert report["bridge"]["primary_lifts"]["dynamics"] == 4.0
    assert report["bridge"]["primary_to_random_ratio"]["dynamics"] == 4.0
    assert report["bridge"]["label_shuffle_lifts"]["time_shuffle"]["dynamics"] == 1.1
    assert np.isclose(report["bridge"]["transport"]["observed"]["mean"], 0.075)
    assert report["bridge"]["theory_anchor"]["peak_angle_degrees"] == 45.0


def test_empirical_operator_alignment_paper_outline_is_standalone():
    audit = {
        "condition": "states",
        "config": {"num_samples": 1000, "num_cells": 8, "ks": [32]},
        "warnings": [],
        "weighted_mean": {
            "lifts": {
                "dynamics": {"32": 4.0},
                "planning": {"32": 3.5},
                "predicted_dynamics": {"32": 4.1},
                "prediction_residual": {"32": 2.8},
            },
            "controls": {
                "norm_matched_random_dynamics_lift": {"32": 1.0},
            },
        },
        "label_controls": {
            "state_cluster_shuffle": {"weighted_mean": {"lifts": {"dynamics": {"32": 1.2}}}},
            "time_shuffle": {"weighted_mean": {"lifts": {"dynamics": {"32": 1.1}}}},
            "action_shuffle": {"weighted_mean": {"lifts": {"dynamics": {"32": 1.05}}}},
        },
        "eigenspace_transport": {"k": 4, "transitions": {"0->1": 0.1, "1->2": 0.05}},
        "spectral_planner_metric": {
            "weights": [1.0, 1.0, 1.0, 1.0],
            "weight_summary": {"top_weight_mean": 1.2, "tail_weight_mean": 0.8, "mean": 1.0, "min": 0.8, "max": 1.2},
        },
    }
    report = empirical_operator_alignment_bridge_report(audit, theorem=None, k=32, null_trials=16, seed=0)

    outline = empirical_operator_alignment_paper_outline(report)

    assert "Frozen PushT Latent Geometry Exhibits Basis-Sensitive Spectral Transfer" in outline
    assert "standalone empirical geometry claim" in outline
    assert "It does not prove the synthetic operator-alignment theorem." in outline


def test_empirical_operator_alignment_paper_spine_is_manuscript_form():
    audit = {
        "condition": "states",
        "config": {"num_samples": 1000, "num_cells": 8, "ks": [32]},
        "warnings": [],
        "weighted_mean": {
            "lifts": {
                "dynamics": {"32": 4.0},
                "planning": {"32": 3.5},
                "predicted_dynamics": {"32": 4.1},
                "prediction_residual": {"32": 2.8},
            },
            "controls": {
                "norm_matched_random_dynamics_lift": {"32": 1.0},
            },
        },
        "label_controls": {
            "state_cluster_shuffle": {"weighted_mean": {"lifts": {"dynamics": {"32": 1.2}}}},
            "time_shuffle": {"weighted_mean": {"lifts": {"dynamics": {"32": 1.1}}}},
            "action_shuffle": {"weighted_mean": {"lifts": {"dynamics": {"32": 1.05}}}},
        },
        "eigenspace_transport": {"k": 4, "transitions": {"0->1": 0.1, "1->2": 0.05}},
        "spectral_planner_metric": {
            "weights": [1.0, 1.0, 1.0, 1.0],
            "weight_summary": {"top_weight_mean": 1.2, "tail_weight_mean": 0.8, "mean": 1.0, "min": 0.8, "max": 1.2},
        },
    }
    report = empirical_operator_alignment_bridge_report(audit, theorem=None, k=32, null_trials=16, seed=0)

    spine = empirical_operator_alignment_paper_spine(report)

    assert spine.startswith("# Frozen PushT Latent Geometry Exhibits Basis-Sensitive Spectral Transfer")
    assert "## Abstract" in spine
    assert "A completed frozen PushT latent audit as a standalone empirical geometry paper" in spine
    assert "It does not collapse into the theory paper" in spine


def test_full_report_includes_research_proposal_outputs():
    rng = np.random.default_rng(14)
    n = 160
    d = 6
    latents = rng.normal(size=(n, d)).astype("float32")
    dynamics = rng.normal(size=(n, d)).astype("float32")
    report = run_empirical_audit(
        {
            "latents": latents,
            "dynamics": dynamics,
            "planning": dynamics,
            "nuisance": rng.normal(size=(n, d)).astype("float32"),
            "prediction_residual": rng.normal(size=(n, d)).astype("float32"),
            "ood_latents": rng.normal(size=(n, d)).astype("float32"),
        },
        EmpiricalAuditConfig(condition="global", ks=(2,), min_cell_size=16, include_interventions=False),
    )

    assert "spectral_planner_metric" in report
    assert "spectral_exploration" in report
    assert "ood_spectral_shift" in report


def test_regularizer_helpers_return_all_directional_losses():
    torch.manual_seed(0)
    emb = torch.randn(4, 5, 6)
    target = emb[:, 2:]
    pred = target + 0.01 * torch.randn_like(target)
    batch = {"pixels": torch.randn(4, 5, 3, 8, 8)}

    losses = directional_regularizer_losses(
        emb=emb,
        pred_emb=pred,
        target_emb=target,
        batch=batch,
        top_k=2,
        encode_pixels=lambda _: emb,
        nuisance_enabled=True,
    )

    assert "spectral_prediction_loss" in losses
    assert "residual_band_regularization_loss" in losses
    assert "contact_event_spectral_auxiliary_loss" in losses
    assert losses["nuisance_deemphasis_loss"].ndim == 0
