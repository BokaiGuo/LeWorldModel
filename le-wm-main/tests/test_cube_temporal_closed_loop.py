import numpy as np
import pytest
from argparse import Namespace

from experiments.run_cube_temporal_closed_loop import (
    POST_CONTACT_PROBE_STEPS,
    TASK_ALIGNED_PROBE_SCALE,
    ADAPTIVE_TASK_ALIGNED_PROBE_SCALE,
    TASK_PROBE_RECENCY_DECAY,
    _prelude_action,
    _readout_family_name,
    _fit_goal_local_gate,
    _fit_local_residual_gate,
    _fit_task_probe_gate,
    _local_prediction_residual,
    _post_contact_holdout_residual,
    _task_aligned_probe_score,
    _has_informative_task_probe_action,
    _paired_root_summary,
    parse_args,
    _rank_consensus_score,
    _robust_rank_score,
    _rgb_block_goal_error,
    _support_exploration_action,
    _post_contact_study_plan,
    _validate_post_contact_protocol,
    _uses_post_contact_probe,
    _verified_rgb_action,
)


def test_optional_post_contact_studies_require_disjoint_calibration_carrier_and_gate():
    """Each enabled repair study must compare a gated local arm to its matched carrier."""
    plan = _post_contact_study_plan(Namespace(
        goal_gated_local=False,
        residual_gated_local=False,
        holdout_residual_gated_local=False,
        task_probe_gated_local=True,
        adaptive_task_probe_local=True,
    ))
    assert plan == [
        {
            "name": "task_probe",
            "carrier_method": "rgb_geometry_action_post_contact_task_probe",
            "calibration_method": "rgb_verified_effect_post_contact_task_probe_calibration",
            "gated_method": "rgb_verified_effect_post_contact_task_probe_gated",
            "gate_key": "task_probe_gate",
        },
        {
            "name": "adaptive_task_probe",
            "carrier_method": "rgb_geometry_action_post_contact_adaptive_task_probe",
            "calibration_method": "rgb_verified_effect_post_contact_adaptive_task_probe_calibration",
            "gated_method": "rgb_verified_effect_post_contact_adaptive_task_probe_gated",
            "gate_key": "adaptive_task_probe_gate",
        },
    ]


def test_post_contact_repair_refuses_to_run_without_an_explicit_rgb_approach():
    """Zero-response prelude rows are not a valid post-contact identification state."""
    plan = _post_contact_study_plan(Namespace(
        goal_gated_local=False,
        residual_gated_local=False,
        holdout_residual_gated_local=False,
        task_probe_gated_local=False,
        adaptive_task_probe_local=True,
    ))
    with pytest.raises(ValueError, match="positive --approach-steps"):
        _validate_post_contact_protocol(
            plan, approach_steps=0, prelude_steps=20,
            phase_order="descent_then_align",
        )
    _validate_post_contact_protocol(
        plan, approach_steps=20, prelude_steps=40,
        phase_order="align_then_descend",
    )


def test_closed_loop_prelude_is_fixed_and_does_not_read_state():
    assert np.allclose(_prelude_action(0), [0, 0, -1, 0, 0])
    assert np.allclose(_prelude_action(14), [0, 0, -1, 0, 0])
    assert np.allclose(_prelude_action(15), [0, 0, 0, 0, 1])
    assert np.allclose(_prelude_action(19), [0, 0, 0, 0, 1])
    assert np.allclose(_prelude_action(20), [0, 0, 0, 0, 0])


def test_support_exploration_is_fixed_four_direction_cycle():
    candidates = np.asarray([
        [0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [-1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0], [0, -1, 0, 0, 0], [0, 0, -1, 0, 0],
        [0, 0, 0, 1, 0], [0, 0, 0, 0, -1],
    ], dtype=np.float32)
    np.testing.assert_allclose(_support_exploration_action(0, candidates), candidates[1])
    np.testing.assert_allclose(_support_exploration_action(4, candidates), candidates[1])


def test_align_then_descend_is_a_reachable_declared_protocol(monkeypatch):
    """The alternate hybrid phase order must reach every runtime arm via CLI."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_cube_temporal_closed_loop.py",
            "--output", "result.json",
            "--phase-order", "align_then_descend",
        ],
    )
    assert parse_args().phase_order == "align_then_descend"

    np.testing.assert_allclose(
        _prelude_action(
            0, np.zeros((8, 8, 3), dtype=np.uint8), approach_steps=2,
            phase_order="align_then_descend",
        )[2:],
        [0.0, 0.0, 0.0],
    )
    assert _prelude_action(
        2, approach_steps=2, phase_order="align_then_descend",
    )[2] == -1.0
    assert _prelude_action(
        17, approach_steps=2, phase_order="align_then_descend",
    )[4] == 1.0


def test_post_contact_identification_is_limited_and_matched_between_arms():
    assert POST_CONTACT_PROBE_STEPS == 4
    assert _uses_post_contact_probe("rgb_verified_effect_post_contact")
    assert _uses_post_contact_probe("rgb_verified_effect_post_contact_residual_gated")
    assert _uses_post_contact_probe("rgb_verified_effect_post_contact_holdout_residual_gated")
    assert _uses_post_contact_probe("rgb_geometry_action_post_contact_task_probe")
    assert _uses_post_contact_probe("rgb_verified_effect_post_contact_task_probe_gated")
    assert _uses_post_contact_probe("rgb_geometry_action_post_contact_adaptive_task_probe")
    assert _uses_post_contact_probe("rgb_verified_effect_post_contact_adaptive_task_probe_gated")
    assert _uses_post_contact_probe("rgb_geometry_action_post_contact_probe")
    assert not _uses_post_contact_probe("rgb_verified_effect")
    assert _readout_family_name("rgb_geometry_action_post_contact_probe") == "rgb_geometry_action"
    assert _readout_family_name("rgb_verified_effect_post_contact_residual_gated") == "rgb_geometry_action"
    assert _readout_family_name("rgb_verified_effect_post_contact_holdout_residual_gated") == "rgb_geometry_action"
    assert _readout_family_name("rgb_geometry_action_post_contact_task_probe") == "rgb_geometry_action"
    assert _readout_family_name(
        "rgb_verified_effect_post_contact_adaptive_task_probe_gated"
    ) == "rgb_geometry_action"


def test_goal_local_gate_uses_only_calibration_pairs_and_fails_closed_on_losses():
    initials = [
        {"seed": 1, "task_id": 1, "goal_rgb_centroid": [0.1, 0.2]},
        {"seed": 2, "task_id": 1, "goal_rgb_centroid": [0.1, 0.2]},
        {"seed": 3, "task_id": 2, "goal_rgb_centroid": [0.7, 0.8]},
        {"seed": 4, "task_id": 2, "goal_rgb_centroid": [0.7, 0.8]},
    ]
    carrier = [
        {"seed": 1, "final_goal_distance_m": 1.0},
        {"seed": 2, "final_goal_distance_m": 1.0},
        {"seed": 3, "final_goal_distance_m": 1.0},
        {"seed": 4, "final_goal_distance_m": 1.0},
    ]
    local = [
        {"seed": 1, "final_goal_distance_m": 0.8},
        {"seed": 2, "final_goal_distance_m": 0.9},
        {"seed": 3, "final_goal_distance_m": 0.8},
        {"seed": 4, "final_goal_distance_m": 1.2},
    ]
    gate = _fit_goal_local_gate(initials, local, carrier)
    charts = {row["calibration_task_id"]: row for row in gate["charts"]}
    assert charts[1]["permit_local"] is True
    assert charts[2]["permit_local"] is False
    assert charts[2]["losses"] == 1


def test_rgb_feedback_error_uses_physical_red_component():
    current = np.zeros((32, 32, 3), dtype=np.uint8)
    goal = np.zeros((32, 32, 3), dtype=np.uint8)
    current[10:14, 8:12, 0] = 220
    goal[10:14, 20:24, 0] = 220
    error, valid = _rgb_block_goal_error(current, goal)
    assert valid
    assert error > 0.5


def test_rank_consensus_is_scale_invariant_and_finite():
    latent = np.asarray([1.0, 2.0, 4.0, 8.0], dtype=np.float32)
    rgb = np.asarray([100.0, 200.0, 400.0, 800.0], dtype=np.float32)
    import torch

    score = _rank_consensus_score(torch.from_numpy(latent), torch.from_numpy(rgb))
    np.testing.assert_allclose(score.numpy(), 2.0 * _robust_rank_score(torch.from_numpy(latent)).numpy())
    assert np.isfinite(score.numpy()).all()


def test_verified_rgb_effect_fails_closed_until_rank_two_redundant_support():
    candidates = np.asarray([
        [0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [-1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0], [0, -1, 0, 0, 0],
    ], dtype=np.float32)
    action, diagnostics = _verified_rgb_action(
        [np.asarray([1.0, 0.0])] * 3,
        [np.asarray([1.0, 0.0])] * 3,
        np.asarray([1.0, 0.0]), candidates,
    )
    assert action is None
    assert diagnostics["status"] == "insufficient_support"


def test_verified_rgb_effect_selects_observed_goal_improving_direction():
    candidates = np.asarray([
        [0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [-1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0], [0, -1, 0, 0, 0],
    ], dtype=np.float32)
    actions = [
        np.asarray([1.0, 0.0]), np.asarray([-1.0, 0.0]),
        np.asarray([0.0, 1.0]), np.asarray([0.0, -1.0]),
    ]
    deltas = [
        np.asarray([0.2, 0.0]), np.asarray([-0.2, 0.0]),
        np.asarray([0.0, 0.1]), np.asarray([0.0, -0.1]),
    ]
    action, diagnostics = _verified_rgb_action(
        actions, deltas, np.asarray([1.0, 0.0]), candidates,
    )
    np.testing.assert_allclose(action, candidates[1])
    assert diagnostics["status"] == "ok"
    assert diagnostics["rank"] == 2
    assert diagnostics["response_rms"] > 0.0
    assert diagnostics["opposite_action_antisymmetry"] == 0.0


def test_adaptive_local_chart_separates_drift_and_reweights_recent_effects():
    candidates = np.asarray([
        [0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [-1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0], [0, -1, 0, 0, 0],
    ], dtype=np.float32)
    actions = [
        np.asarray([1.0, 0.0]), np.asarray([-1.0, 0.0]),
        np.asarray([0.0, 1.0]), np.asarray([0.0, -1.0]),
    ]
    # delta = action @ diag(0.2, 0.1) + [0.03, -0.02].
    deltas = [
        np.asarray([0.23, -0.02]), np.asarray([-0.17, -0.02]),
        np.asarray([0.03, 0.08]), np.asarray([0.03, -0.12]),
    ]
    action, diagnostics = _verified_rgb_action(
        actions, deltas, np.asarray([1.0, 0.0]), candidates,
        recency_decay=TASK_PROBE_RECENCY_DECAY, include_drift=True,
    )
    np.testing.assert_allclose(action, candidates[1])
    np.testing.assert_allclose(diagnostics["estimated_drift_delta"], [0.03, -0.02])
    assert diagnostics["recency_decay"] == TASK_PROBE_RECENCY_DECAY
    residual = _local_prediction_residual(
        np.asarray(diagnostics["mapping"]), np.asarray([1.0, 0.0]),
        np.asarray([0.23, -0.02]), response_rms=diagnostics["response_rms"],
        drift_delta=np.asarray(diagnostics["estimated_drift_delta"]),
    )
    assert residual["normalized_residual"] < 1.0e-10


def test_residual_gate_uses_valid_root_level_conformal_rank_and_fails_closed_when_too_small():
    episodes = [
        {"verified_residual_scores": [float(index + 1), 0.1]}
        for index in range(20)
    ]
    gate = _fit_local_residual_gate(episodes, alpha=0.1)
    # ceil((20 + 1) * .9) = 19.  The prior clipped n*(1-alpha)
    # construction would choose a different index and under-cover.
    assert gate["conformal_rank"] == 19
    assert gate["certified_finite_threshold"] is True
    assert gate["threshold"] == 19.0

    too_small = _fit_local_residual_gate(episodes[:8], alpha=0.1)
    assert too_small["conformal_rank"] == 9
    assert too_small["certified_finite_threshold"] is False
    assert np.isinf(too_small["threshold"])


def test_local_prediction_residual_is_scale_normalized_and_observation_only():
    residual = _local_prediction_residual(
        np.asarray([[0.2, 0.0], [0.0, 0.1]]),
        np.asarray([1.0, 0.0]), np.asarray([0.3, 0.0]),
        response_rms=0.1,
    )
    assert residual["status"] == "ok"
    assert residual["predicted_delta"] == [0.2, 0.0]
    assert residual["raw_residual"] == 0.09999999999999998
    assert residual["normalized_residual"] < 1.0


def test_task_aligned_probe_requires_positive_goal_direction_progress():
    positive = _task_aligned_probe_score(
        np.asarray([[0.2, 0.0], [0.0, 0.1]]),
        np.asarray([1.0, 0.0]), np.asarray([TASK_ALIGNED_PROBE_SCALE, 0.0]),
        np.asarray([1.0, 0.0]), np.asarray([0.06, 0.0]), response_rms=0.1,
    )
    assert positive["status"] == "ok"
    assert positive["task_progress_positive"] is True
    assert positive["normalized_task_mismatch"] < 1.0

    wrong_direction = _task_aligned_probe_score(
        np.asarray([[0.2, 0.0], [0.0, 0.1]]),
        np.asarray([1.0, 0.0]), np.asarray([TASK_ALIGNED_PROBE_SCALE, 0.0]),
        np.asarray([1.0, 0.0]), np.asarray([-0.01, 0.0]), response_rms=0.1,
    )
    assert wrong_direction["status"] == "nonpositive_task_progress"
    assert np.isinf(wrong_direction["normalized_task_mismatch"])


def test_task_probe_score_records_and_uses_declared_adaptive_scale():
    score = _task_aligned_probe_score(
        np.asarray([[0.02, 0.0], [0.0, 0.01]]),
        np.asarray([1.0, 0.0]), np.asarray([ADAPTIVE_TASK_ALIGNED_PROBE_SCALE, 0.0]),
        np.asarray([1.0, 0.0]), np.asarray([0.01, 0.0]), response_rms=0.02,
        probe_scale=ADAPTIVE_TASK_ALIGNED_PROBE_SCALE,
    )
    assert score["status"] == "ok"
    assert score["probe_scale"] == ADAPTIVE_TASK_ALIGNED_PROBE_SCALE
    # q=0.50 makes the response-RMS denominator 0.01, matching the observed
    # task effect exactly; a legacy q=0.25 denominator would be inconsistent.
    assert score["normalization_scale"] == 0.01
    assert score["normalized_task_mismatch"] == 0.0


def test_task_probe_requires_a_nonzero_xy_intervention():
    """A no-op cannot be treated as an observed action-effect experiment."""
    assert not _has_informative_task_probe_action(np.asarray([0.0, 0.0, 0.0, 0.0, 0.0]))
    assert not _has_informative_task_probe_action(np.asarray([np.nan, 0.0, 0.0, 0.0, 0.0]))
    assert _has_informative_task_probe_action(np.asarray([1.0, 0.0, 0.0, 0.0, 0.0]))


def test_task_probe_gate_uses_one_pre_release_score_per_root_and_fails_closed():
    episodes = [{"task_probe_scores": [float(index + 1)]} for index in range(20)]
    gate = _fit_task_probe_gate(episodes, alpha=0.1)
    assert gate["conformal_rank"] == 19
    assert gate["threshold"] == 19.0
    assert gate["certified_finite_threshold"] is True

    incomplete = _fit_task_probe_gate(episodes[:8], alpha=0.1)
    assert incomplete["certified_finite_threshold"] is False
    assert np.isinf(incomplete["threshold"])

    conditional = _fit_task_probe_gate(
        [{"task_probe_scores": [0.2]}] * 20 + [{"task_probe_scores": [float("inf")]}] * 5,
        alpha=0.1,
    )
    assert conditional["all_calibration_roots"] == 25
    assert conditional["calibration_roots"] == 20
    assert conditional["roots_rejected_before_conditional_calibration"] == 5
    assert conditional["certified_finite_threshold"] is True


def test_post_contact_holdout_residual_excludes_the_fourth_probe_from_the_map():
    actions = [
        np.asarray([1.0, 0.0]), np.asarray([-1.0, 0.0]),
        np.asarray([0.0, 1.0]), np.asarray([0.0, -1.0]),
    ]
    deltas = [0.1 * action for action in actions]
    residual = _post_contact_holdout_residual(actions, deltas)
    assert residual["status"] == "ok"
    assert residual["withheld_probe_index"] == 3
    assert residual["normalized_residual"] == 0.0


def test_paired_root_summary_keeps_roots_as_statistical_unit():
    summary = _paired_root_summary(
        np.asarray([-0.1, -0.1, -0.1, -0.1, -0.1, -0.1]),
        seed=7, draws=200,
    )
    assert summary["unit"] == "paired outcome root"
    assert summary["n_roots"] == 6
    assert summary["wins"] == 6
    assert summary["losses"] == 0
    assert summary["exact_two_sided_sign_p"] == 0.03125
    np.testing.assert_allclose(summary["paired_bootstrap_ci95"], [-0.1, -0.1])
