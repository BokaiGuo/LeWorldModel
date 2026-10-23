from __future__ import annotations

import numpy as np
import pytest

from experiments.run_live_pointmaze_planning import (
    PRIVILEGED_METHODS,
    GOAL_STATE,
    OnlineLocalEffectResidual,
    PriorAnchoredResidualJacobianModel,
    PASSIVE_PRIOR_ANCHORED_RESIDUAL_METHOD,
    VerifiedLocalActionEffectModel,
    VisualCalibration,
    VisualDynamicsCalibration,
    RGBMazeMap,
    action_bank,
    active_probe_actions,
    active_probe_trigger_decision,
    probe_conditioned_probe_decision,
    probe_conditioned_release_decision,
    task_safe_candidate_anchored_probe_actions,
    task_safe_candidate_anchored_residual_probe_actions,
    dynamics_safe_balanced_residual_probe_actions,
    task_safe_probe_action_metrics,
    task_safe_probe_actions,
    task_aligned_probe_actions,
    authorize_gdd_teacher,
    choose_action,
    deployment_context_windows,
    deployment_warmup_blocks,
    ground_truth_candidate_scores,
    green_center,
    load_episode_states,
    _make_gdd_source_contract_context,
    paired_gdd_action_sequences,
    nearest_supported_action,
    parse_episode_list,
    perfect_dynamics_mpc_action,
    probe_residual_trusted,
    prior_anchored_release_gate,
    project_action_to_dynamics_support,
    select_gate_tolerance,
    select_effect_visual_tolerance,
    validate_effect_policy_execution_contract,
    validate_output_path,
    uses_frozen_effect_residual,
    uses_candidate_anchored_affine_probe,
    uses_local_phase_reset,
    uses_local_visual_residual,
    uses_prior_anchored_residual_chart,
    visual_dynamics_task_costs,
    fixed_candidate_bank_digest,
    validate_fixed_candidate_bank,
    validate_runtime_candidate_bank,
)
from experiments.run_pointmaze_rgb_closed_loop_strict import (
    DEPLOYABLE_METHODS as STRICT_DEPLOYABLE_METHODS,
    PRIVILEGED_METHODS as STRICT_PRIVILEGED_METHODS,
    STRICT_METHODS,
    validate_root_partition,
    validate_strict_report,
)


def test_effect_policy_rejects_legacy_twenty_step_teacher_at_one_block_deployment() -> None:
    class Policy:
        action_steps = 20
        candidate_family = "runtime_repeated_pulse_v1"

    with pytest.raises(ValueError, match="action horizon"):
        validate_effect_policy_execution_contract(Policy())


def test_effect_policy_accepts_runtime_matched_one_block_teacher() -> None:
    class Policy:
        action_steps = 5
        candidate_family = "runtime_one_block_pulse_v1"

    validate_effect_policy_execution_contract(Policy())


def test_decision_ambiguity_probe_trigger_is_margin_bounded_and_does_not_require_nonrelease() -> None:
    probe, reason = active_probe_trigger_decision(
        "conditional_decision_ambiguity",
        effect_policy_released=True,
        state_support_passed=True,
        decision_margin=0.02,
        ambiguity_margin=0.03,
    )
    assert probe is True
    assert reason == "episode_start_decision_ambiguity"

    probe, reason = active_probe_trigger_decision(
        "conditional_decision_ambiguity",
        effect_policy_released=False,
        state_support_passed=False,
        decision_margin=0.04,
        ambiguity_margin=0.03,
    )
    assert probe is False
    assert reason == "episode_start_decision_margin_confident"


def test_probe_conditioned_information_and_release_gates_are_prospective_and_fail_closed() -> None:
    costs = np.asarray([0.10, 0.11, 0.40], dtype=np.float64)
    assert probe_conditioned_probe_decision(costs, ambiguity_margin=None) == (True, None)
    probe, margin = probe_conditioned_probe_decision(costs, ambiguity_margin=0.02)
    assert probe is True and margin == pytest.approx(0.01)
    probe, margin = probe_conditioned_probe_decision(costs, ambiguity_margin=0.005)
    assert probe is False and margin == pytest.approx(0.01)
    assert probe_conditioned_release_decision(
        local_best_cost=0.20, visual_best_cost=0.24, minimum_advantage=0.03,
    ) == (True, pytest.approx(0.04))
    assert probe_conditioned_release_decision(
        local_best_cost=0.20, visual_best_cost=0.22, minimum_advantage=0.03,
    ) == (False, pytest.approx(0.02))
    with pytest.raises(ValueError, match="at least two"):
        probe_conditioned_probe_decision(np.asarray([0.1]), ambiguity_margin=0.0)


def test_gdd_teacher_authorization_fails_closed_on_calibration_regression() -> None:
    teacher = [
        {"success": True, "final_distance": 0.40},
        {"success": True, "final_distance": 0.60},
    ]
    visual = [
        {"success": True, "final_distance": 0.30},
        {"success": True, "final_distance": 0.50},
    ]
    decision = authorize_gdd_teacher(teacher, visual)
    assert decision["authorized"] is False
    assert decision["final_distance_delta_teacher_minus_visual"] > 0.0


def test_gdd_teacher_authorization_requires_no_binary_loss() -> None:
    teacher = [{"success": False, "final_distance": 0.1}]
    visual = [{"success": True, "final_distance": 0.2}]
    assert authorize_gdd_teacher(teacher, visual)["authorized"] is False


def test_gdd_teacher_release_rate_gate_rejects_fallback_only_authorization() -> None:
    teacher = [{"success": True, "final_distance": 0.1,
                "aggregate": {"gate_fallback_rate": 0.95}}]
    visual = [{"success": True, "final_distance": 0.1,
               "aggregate": {"gate_fallback_rate": 0.0}}]
    decision = authorize_gdd_teacher(teacher, visual, minimum_release_rate=0.20)
    assert decision["authorized"] is False
    assert decision["release_rate_gate_passed"] is False
    assert np.isclose(decision["teacher_release_rate"], 0.05)


def test_gdd_source_contract_matches_frozen_export_round_trip() -> None:
    import torch

    visual = torch.tensor([[[[[0.0, 127.5], [255.0, 64.0]], [[10.0, 20.0], [30.0, 40.0]], [[50.0, 60.0], [70.0, 80.0]]]]])
    proprio = torch.zeros((1, 1, 4), dtype=torch.float32)
    context = {"visual": visual, "proprio": proprio}

    def transform(value: torch.Tensor) -> torch.Tensor:
        return value + 0.125

    transformed = _make_gdd_source_contract_context(context, transform)
    assert torch.allclose(transformed["visual"], visual + 31.875)
    assert transformed["proprio"] is proprio


def test_paired_gdd_action_sequences_match_frozen_candidate_shape() -> None:
    sequences = paired_gdd_action_sequences(np.asarray([0.0, 0.0]), np.asarray([1.0, 1.0]))
    assert sequences.shape == (16, 20, 2)
    # The first two candidates are an ordered pair and its reversal.
    assert np.allclose(sequences[0, :10], sequences[1, 10:])
    assert np.allclose(sequences[0, 10:], sequences[1, :10])


def test_support_preserving_mapping_releases_only_declared_bank_action() -> None:
    bank = np.asarray([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]])
    index, action, distance = nearest_supported_action(
        np.asarray([0.46, 0.08]), bank,
    )
    assert index == 1
    assert np.allclose(action, bank[index])
    assert np.isclose(distance, np.linalg.norm(np.asarray([0.46, 0.08]) - bank[index]))


def test_support_preserving_mapping_rejects_invalid_bank() -> None:
    with pytest.raises(ValueError, match="support bank"):
        nearest_supported_action(np.zeros(2), np.zeros((0, 2)))


def test_exact_dynamics_planners_are_reported_as_privileged() -> None:
    assert "ground_truth_candidate_oracle" in PRIVILEGED_METHODS
    assert "perfect_dynamics_mpc" in PRIVILEGED_METHODS


def test_history_aware_effect_policy_controls_live_context_window() -> None:
    class Policy:
        history_windows = 4

    class HistoryOnePolicy:
        history_windows = 1

    assert deployment_context_windows(None) == 2
    assert deployment_context_windows(HistoryOnePolicy()) == 1
    assert deployment_context_windows(Policy()) == 4


def test_deployment_warmup_uses_source_frame_clock() -> None:
    assert deployment_warmup_blocks(1, 1) == 1
    assert deployment_warmup_blocks(1, 4) == 4
    assert deployment_warmup_blocks(5, 4) == 5
    with pytest.raises(ValueError):
        deployment_warmup_blocks(-1, 4)


def test_green_center_extracts_only_the_marker() -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[12:20, 21:29] = [20, 180, 20]
    assert green_center(image) == (24.5, 15.5)


def test_green_center_fails_closed_without_marker() -> None:
    with pytest.raises(ValueError, match="no detectable green marker"):
        green_center(np.zeros((16, 16, 3), dtype=np.uint8))


def test_action_bank_is_goal_relative_and_finite() -> None:
    bank = action_bank(np.asarray([0.5, 0.5]), GOAL_STATE, 8, [0.25, 0.5])
    assert bank.shape == (16, 2)
    assert np.isfinite(bank).all()
    assert np.allclose(np.linalg.norm(bank, axis=1)[:8], 0.25)
    assert np.allclose(np.linalg.norm(bank, axis=1)[8:], 0.50)


def test_action_bank_rejects_too_few_directions() -> None:
    with pytest.raises(ValueError, match="at least four"):
        action_bank(np.zeros(2), GOAL_STATE, 3, [0.25])


def test_strict_fixed_candidate_bank_is_exact_and_requires_full_declared_bank() -> None:
    bank = action_bank(np.asarray([0.0, 0.0]), GOAL_STATE, 8, [0.25, 0.5, 0.75, 1.0])
    validated = validate_fixed_candidate_bank(bank)
    assert validated.shape == (32, 2)
    assert fixed_candidate_bank_digest(validated) == fixed_candidate_bank_digest(bank)
    with pytest.raises(ValueError, match=r"shape \[32, 2\]"):
        validate_fixed_candidate_bank(bank[:16])


def test_runtime_candidate_bank_allows_a_non_strict_local_bank_without_weakening_strict_contract() -> None:
    local = np.asarray([
        [0.25, 0.0], [0.0, 0.25], [-0.25, 0.0], [0.0, -0.25],
    ])
    assert validate_runtime_candidate_bank(local).shape == (4, 2)
    with pytest.raises(ValueError, match=r"shape \[32, 2\]"):
        validate_fixed_candidate_bank(local)


def test_strict_root_partition_rejects_adapter_htsc_or_test_reuse() -> None:
    provenance = {
        "adapter_train_roots": [1, 2],
        "htsc_calibration_roots": [3, 4],
    }
    validate_root_partition(provenance, [5], [6], [7])
    with pytest.raises(ValueError, match="split overlap"):
        validate_root_partition(provenance, [5], [6], [4])


def test_strict_report_requires_rgb_only_deployable_arms_and_exact_root_bank() -> None:
    actions = [[0.25, 0.0]] * 32
    digest = fixed_candidate_bank_digest(np.asarray(actions, dtype=np.float64))
    rows = []
    for method in STRICT_METHODS:
        privileged = method in STRICT_PRIVILEGED_METHODS
        rows.append({
            "episode_id": 99,
            "method": method,
            "blocks_executed": 2,
            "final_distance": 1.0,
            "success": False,
            "policy_contract": {
                "simulator_state_in_policy_input": privileged,
                "strict_shared_root_candidate_bank": True,
                "candidate_bank_digest": digest,
                "candidate_actions": actions,
                "target_chart": "particle_relative_qpos_v1",
                "planner_goal_qpos": [1.17, 1.19],
                "evaluator_goal_qpos": [1.2, 1.2],
                "planner_evaluator_goal_gap_qpos": float(np.linalg.norm(np.asarray([1.17, 1.19]) - np.asarray([1.2, 1.2]))),
                "evaluation_objective_id": "terminal_global_goal_distance_v1",
                "replanning_semantics": "receding_horizon_one_block_reobserve_v1",
            },
            "trace": [{
                "candidate_bank_digest": digest,
                "action_transfer_gate": (
                    {
                        "selection_objective_id": (
                            "one_step_active_waypoint_distance_v1"
                            if method == "ground_truth_candidate_oracle"
                            else "bounded_beam_active_waypoint_distance_v1"
                        ),
                        "search_completeness": (
                            "exact_one_step_fixed_bank_only"
                            if method == "ground_truth_candidate_oracle"
                            else "bounded_beam_not_terminal_complete"
                        ),
                    }
                    if privileged else {}
                ),
            }],
        })
    report = {
        "protocol": {
            "methods": list(STRICT_METHODS),
            "test_episodes": [99],
            "privileged_methods": list(STRICT_PRIVILEGED_METHODS),
            "action_bank": {"strict_shared_root_candidate_bank": True, "max_execution_blocks": 3},
        },
        "rows": rows,
    }
    grouped = validate_strict_report(report, expected_test_roots=[99], max_blocks=3)
    assert set(grouped[99][index]["method"] for index in range(len(STRICT_METHODS))) == set(STRICT_METHODS)
    assert STRICT_DEPLOYABLE_METHODS.isdisjoint(STRICT_PRIVILEGED_METHODS)
    report["rows"][0]["policy_contract"]["simulator_state_in_policy_input"] = True
    with pytest.raises(ValueError, match="received simulator state"):
        validate_strict_report(report, expected_test_roots=[99], max_blocks=3)


def test_strict_report_rejects_missing_target_contract_and_out_of_bank_action() -> None:
    actions = [[0.25, 0.0]] * 32
    digest = fixed_candidate_bank_digest(np.asarray(actions, dtype=np.float64))

    def row(method: str) -> dict:
        privileged = method in STRICT_PRIVILEGED_METHODS
        gate = {}
        if method == "ground_truth_candidate_oracle":
            gate = {
                "selection_objective_id": "one_step_active_waypoint_distance_v1",
                "search_completeness": "exact_one_step_fixed_bank_only",
            }
        elif method == "perfect_dynamics_mpc":
            gate = {
                "selection_objective_id": "bounded_beam_active_waypoint_distance_v1",
                "search_completeness": "bounded_beam_not_terminal_complete",
            }
        return {
            "episode_id": 99, "method": method, "blocks_executed": 1,
            "final_distance": 1.0, "success": False,
            "policy_contract": {
                "simulator_state_in_policy_input": privileged,
                "strict_shared_root_candidate_bank": True,
                "candidate_bank_digest": digest, "candidate_actions": actions,
                "target_chart": "particle_relative_qpos_v1",
                "planner_goal_qpos": [1.17, 1.19], "evaluator_goal_qpos": [1.2, 1.2],
                "planner_evaluator_goal_gap_qpos": float(np.linalg.norm(np.asarray([1.17, 1.19]) - np.asarray([1.2, 1.2]))),
                "evaluation_objective_id": "terminal_global_goal_distance_v1",
                "replanning_semantics": "receding_horizon_one_block_reobserve_v1",
            },
            "trace": [{"candidate_bank_digest": digest, "selected_action": actions[0], "action_transfer_gate": gate}],
        }

    report = {
        "protocol": {
            "methods": list(STRICT_METHODS), "test_episodes": [99],
            "privileged_methods": list(STRICT_PRIVILEGED_METHODS),
            "action_bank": {"strict_shared_root_candidate_bank": True, "max_execution_blocks": 2},
        },
        "rows": [row(method) for method in STRICT_METHODS],
    }
    report["rows"][0]["policy_contract"].pop("target_chart")
    with pytest.raises(ValueError, match="target chart"):
        validate_strict_report(report, expected_test_roots=[99], max_blocks=2)
    report["rows"][0]["policy_contract"]["target_chart"] = "particle_relative_qpos_v1"
    report["rows"][0]["trace"][0]["selected_action"] = [99.0, 99.0]
    with pytest.raises(ValueError, match="outside its pinned candidate bank"):
        validate_strict_report(report, expected_test_roots=[99], max_blocks=2)


def test_rgb_maze_segment_gate_rejects_inferred_wall_crossing() -> None:
    maze = RGBMazeMap(
        free_cells=((0, 0), (1, 0), (2, 0), (2, 1)),
        cell_positions={
            (0, 0): np.asarray([0.0, 0.0]),
            (1, 0): np.asarray([1.0, 0.0]),
            (2, 0): np.asarray([2.0, 0.0]),
            (2, 1): np.asarray([2.0, 1.0]),
        },
    )
    assert maze.segment_is_free(np.asarray([0.0, 0.0]), np.asarray([2.0, 0.0]))
    assert maze.segment_is_free(np.asarray([2.0, 0.0]), np.asarray([2.0, 1.0]))
    assert not maze.segment_is_free(np.asarray([0.0, 0.0]), np.asarray([0.0, 1.0]))


def test_active_probe_actions_match_goal_relative_runtime_chart() -> None:
    probes = active_probe_actions(np.asarray([0.0, 0.0]), np.asarray([1.0, 1.0]))
    assert probes.shape == (4, 2)
    assert np.allclose(np.linalg.norm(probes, axis=1), 0.5)
    assert np.allclose(probes[0], -probes[1])
    assert np.allclose(probes[2], -probes[3])
    assert np.isclose(np.dot(probes[0], probes[2]), 0.0)


def test_active_probe_actions_supports_fixed_eight_direction_coverage() -> None:
    probes = active_probe_actions(
        np.asarray([0.0, 0.0]), np.asarray([1.0, 1.0]), directions=8,
    )
    assert probes.shape == (8, 2)
    assert np.allclose(np.linalg.norm(probes, axis=1), 0.5)
    assert np.allclose(probes[0], -probes[4])
    with pytest.raises(ValueError, match="even number"):
        active_probe_actions(
            np.asarray([0.0, 0.0]), np.asarray([1.0, 1.0]), directions=5,
        )


def test_task_aligned_probes_keep_affine_excitation_without_reverse_goal_action() -> None:
    probes = task_aligned_probe_actions(
        np.asarray([0.0, 0.0]), np.asarray([1.0, 0.0]), directions=4,
    )
    assert probes.shape == (4, 2)
    assert np.allclose(np.linalg.norm(probes, axis=1), 0.5)
    assert np.all(probes[:, 0] > 0.0)
    assert np.linalg.matrix_rank(np.c_[np.ones(4), probes]) == 3


def test_task_safe_probes_reject_inertially_harmful_positive_actions() -> None:
    class _Dynamics:
        @staticmethod
        def predict_position_delta(_velocity: np.ndarray, action: np.ndarray) -> np.ndarray:
            # The geometric probe still points right, but incoming drift makes
            # it predictably move left in the local RGB chart.
            return np.asarray(action, dtype=np.float64) + np.asarray([-1.0, 0.0])

    probes, progress = task_safe_probe_actions(
        np.asarray([0.0, 0.0]), np.asarray([1.0, 0.0]), np.zeros(2), _Dynamics(),
        directions=4, magnitude=0.5,
    )
    assert probes.shape == (0, 2)
    assert progress.shape == (0,)


def test_candidate_anchored_task_safe_probes_are_full_rank_members_of_planning_bank() -> None:
    class _Dynamics:
        @staticmethod
        def predict_position_delta(_velocity: np.ndarray, action: np.ndarray) -> np.ndarray:
            return 0.2 * np.asarray(action, dtype=np.float64)

    bank = np.asarray([
        [0.25, 0.00], [0.25, 0.18], [0.25, -0.18], [0.50, 0.00],
        [-0.25, 0.00], [0.00, 0.25],
    ])
    probes, progress, indices = task_safe_candidate_anchored_probe_actions(
        np.asarray([0.0, 0.0]), np.asarray([1.0, 0.0]), np.zeros(2), _Dynamics(),
        bank, probe_count=4, max_magnitude=0.5,
    )
    assert probes.shape == (4, 2)
    assert np.all(progress > 0.0)
    assert np.array_equal(probes, bank[indices])
    assert np.linalg.matrix_rank(np.c_[np.ones(len(probes)), probes]) == 3


def test_runtime_task_safety_rejects_a_stale_candidate_anchored_pulse() -> None:
    class _Dynamics:
        @staticmethod
        def predict_position_delta(_velocity: np.ndarray, action: np.ndarray) -> np.ndarray:
            return 0.2 * np.asarray(action, dtype=np.float64)

    action = np.asarray([0.25, 0.18])
    initial = task_safe_probe_action_metrics(
        np.asarray([0.0, 0.0]), np.asarray([1.0, 0.0]), np.zeros(2), _Dynamics(), action,
    )
    stale = task_safe_probe_action_metrics(
        np.asarray([1.2, 0.0]), np.asarray([1.0, 0.0]), np.zeros(2), _Dynamics(), action,
    )
    assert initial[0] > 0.0 and initial[1] > 0.0
    assert stale[0] < 0.0 and stale[1] < 0.0


def test_candidate_anchored_probes_keep_an_identifiable_three_action_safe_subset() -> None:
    class _Dynamics:
        @staticmethod
        def predict_position_delta(_velocity: np.ndarray, action: np.ndarray) -> np.ndarray:
            return 0.2 * np.asarray(action, dtype=np.float64)

    # The frozen magnitude cap intentionally admits exactly three positive,
    # non-collinear bank actions.  V5 must not add a lateral/backward pulse
    # simply to reach the nominal four-probe cap.
    bank = np.asarray([
        [0.25, 0.00], [0.20, 0.12], [0.20, -0.12], [0.00, 0.25],
        [-0.25, 0.00],
    ])
    probes, progress, indices = task_safe_candidate_anchored_probe_actions(
        np.asarray([0.0, 0.0]), np.asarray([1.0, 0.0]), np.zeros(2), _Dynamics(),
        bank, probe_count=4, max_magnitude=0.25,
    )
    assert probes.shape == (3, 2)
    assert np.all(progress > 0.0)
    assert np.array_equal(probes, bank[indices])
    assert np.all(probes[:, 0] > 0.0)
    assert np.linalg.matrix_rank(np.c_[np.ones(len(probes)), probes]) == 3


def test_prior_anchored_residual_probes_need_only_two_safe_independent_bank_actions() -> None:
    class _Dynamics:
        @staticmethod
        def predict_position_delta(_velocity: np.ndarray, action: np.ndarray) -> np.ndarray:
            return 0.2 * np.asarray(action, dtype=np.float64)

    bank = np.asarray([
        [0.25, 0.00], [0.20, 0.12], [0.20, -0.12], [-0.25, 0.00],
    ])
    probes, progress, indices = task_safe_candidate_anchored_residual_probe_actions(
        np.asarray([0.0, 0.0]), np.asarray([1.0, 0.0]), np.zeros(2), _Dynamics(),
        bank, probe_count=2, max_magnitude=0.25,
    )
    assert probes.shape == (2, 2)
    assert np.all(progress > 0.0)
    assert np.array_equal(probes, bank[indices])
    assert np.linalg.matrix_rank(probes) == 2


def test_balanced_residual_probes_admit_safe_lateral_excitation_but_reject_unsafe_paths() -> None:
    class _Dynamics:
        @staticmethod
        def predict_position_delta(_velocity: np.ndarray, action: np.ndarray) -> np.ndarray:
            return 0.2 * np.asarray(action, dtype=np.float64)

    class _Map:
        @staticmethod
        def segment_is_free(_start: np.ndarray, endpoint: np.ndarray) -> bool:
            # The negative-x pulse is deliberately blocked; lateral motion is
            # safe and must remain available to identify inertial response.
            return bool(float(endpoint[0]) >= -1.0e-12)

    bank = np.asarray([
        [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0],
    ])
    probes, progress, indices = dynamics_safe_balanced_residual_probe_actions(
        np.asarray([0.0, 0.0]), np.asarray([1.0, 0.0]), np.zeros(2), _Dynamics(), _Map(),
        bank, max_predicted_regression=0.05,
    )
    assert probes.shape == (2, 2)
    assert np.array_equal(probes, bank[indices])
    assert np.linalg.matrix_rank(probes) == 2
    assert not np.any(np.all(probes == np.asarray([-1.0, 0.0]), axis=1))
    assert np.any(np.isclose(probes[:, 0], 0.0))
    assert np.all(progress >= -0.05)


def test_prior_anchored_residual_chart_releases_only_anchor_hull_interpolation() -> None:
    chart = PriorAnchoredResidualJacobianModel()
    chart.update(np.asarray([0.25, 0.0]), np.asarray([0.05, 0.0]))
    chart.update(np.asarray([0.0, 0.25]), np.asarray([0.0, 0.05]))
    assert chart.trusted is True
    queries = np.asarray([
        [0.125, 0.125],  # conv(0, first, second)
        [0.25, 0.25],    # outside the declared triangular support
        [-0.10, 0.10],   # requires a negative coefficient
    ])
    assert chart.action_support_mask(queries).tolist() == [True, False, False]
    assert np.allclose(chart.predict(queries[:1]), np.asarray([[0.025, 0.025]]), atol=1.0e-6)
    assert np.isinf(chart.leave_one_out_error())


def test_passive_prior_anchored_method_uses_residual_chart_without_probe_family() -> None:
    # V9 learns only from actions HTSC already executes.  It still uses the
    # calibration-prior residual basis, but must not silently enter the
    # candidate-anchored information-action schedule.
    assert uses_prior_anchored_residual_chart(PASSIVE_PRIOR_ANCHORED_RESIDUAL_METHOD)
    assert not uses_candidate_anchored_affine_probe(PASSIVE_PRIOR_ANCHORED_RESIDUAL_METHOD)


def test_prior_anchored_contrast_gate_requires_calibration_frozen_advantage() -> None:
    # V8 may overturn a visual ranking only for a finite predicted terminal
    # gain.  V7 keeps its agreement gate, so it cannot silently acquire this
    # new behavior.
    assert prior_anchored_release_gate(
        contrast_release=False,
        visual_consistent=False,
        local_advantage=0.20,
        min_advantage=0.05,
    ) is False
    assert prior_anchored_release_gate(
        contrast_release=True,
        visual_consistent=False,
        local_advantage=0.06,
        min_advantage=0.05,
    ) is True
    assert prior_anchored_release_gate(
        contrast_release=True,
        visual_consistent=False,
        local_advantage=0.04,
        min_advantage=0.05,
    ) is False
    assert prior_anchored_release_gate(
        contrast_release=True,
        visual_consistent=False,
        local_advantage=0.20,
        min_advantage=float("inf"),
    ) is False


def test_online_local_effect_residual_uses_only_observed_pairs() -> None:
    residual = OnlineLocalEffectResidual()
    actions = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    assert np.allclose(residual.predict(actions), 0.0)
    residual.update(actions[0], np.asarray([0.2, 0.0]), velocity=np.asarray([0.1, 0.0]))
    residual.update(actions[1], np.asarray([0.0, 0.3]), velocity=np.asarray([0.0, 0.1]))
    residual.update(actions[2], np.asarray([-0.2, 0.0]), velocity=np.asarray([-0.1, 0.0]))
    residual.update(actions[0], np.asarray([0.2, 0.0]), velocity=np.asarray([0.0, 0.0]))
    prediction = residual.predict(actions, velocity=np.zeros(2))
    assert residual.sample_count == 4
    assert residual.trusted is False
    residual.record_verification(np.asarray([0.2, 0.0]), np.asarray([0.2, 0.0]))
    residual.record_verification(np.asarray([0.0, 0.3]), np.asarray([0.0, 0.3]))
    assert residual.trusted is True
    prediction = residual.predict(actions, velocity=np.zeros(2))
    assert prediction.shape == actions.shape
    assert np.isfinite(prediction).all()
    assert np.linalg.norm(prediction, axis=1).max() <= residual.max_correction_norm + 1.0e-12


def test_online_local_effect_residual_stores_observed_minus_frozen_prediction() -> None:
    residual = OnlineLocalEffectResidual(min_samples=1)
    residual.update(
        np.asarray([1.0, 0.0]),
        np.asarray([0.7, 0.0]),
        predicted_effect=np.asarray([0.5, 0.0]),
    )
    assert np.allclose(residual.effects[0], np.asarray([0.2, 0.0]))


def test_online_local_effect_residual_uses_probe_hull_for_local_support() -> None:
    residual = OnlineLocalEffectResidual()
    probes = np.asarray([
        [0.5, 0.0], [0.0, 0.5], [-0.5, 0.0], [0.0, -0.5],
    ])
    for action in probes:
        residual.update(action, 0.1 * action)
    mask = residual.action_support_mask(np.asarray([
        [0.0, 0.0], [0.25, 0.25], [0.5, 0.5],
    ]))
    assert mask.tolist() == [True, True, False]


def test_local_chart_uncertainty_increases_outside_observed_action_coverage() -> None:
    chart = VerifiedLocalActionEffectModel(ridge=1.0e-2)
    # A rank-three, slightly non-affine set gives a finite LOO scale.
    for action, effect in (
        (np.asarray([0.5, 0.0]), np.asarray([0.10, 0.00])),
        (np.asarray([0.0, 0.5]), np.asarray([0.00, 0.12])),
        (np.asarray([-0.5, 0.0]), np.asarray([-0.11, 0.01])),
        (np.asarray([0.0, -0.5]), np.asarray([0.01, -0.10])),
    ):
        chart.update(action, effect)
    uncertainty = chart.prediction_uncertainty(np.asarray([
        [0.25, 0.25], [1.0, -1.0],
    ]))
    assert np.isfinite(uncertainty).all()
    assert uncertainty[1] > uncertainty[0]


def test_velocity_conditioned_local_chart_requires_and_uses_velocity() -> None:
    chart = VerifiedLocalActionEffectModel(
        condition_on_velocity=True, min_samples=8, ridge=1.0e-2,
    )
    with pytest.raises(ValueError, match="requires velocity"):
        chart.update(np.asarray([0.5, 0.0]), np.asarray([0.1, 0.0]))
    actions = np.asarray([
        [0.5, 0.0], [0.0, 0.5], [-0.5, 0.0], [0.0, -0.5],
        [0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5],
        [0.25, 0.10], [-0.15, 0.35], [0.40, -0.20], [-0.35, -0.10],
    ])
    velocities = np.asarray([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0],
        [0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5],
        [0.30, -0.70], [-0.80, 0.20], [0.60, 0.40], [-0.40, 0.80],
    ])
    for action, velocity in zip(actions, velocities, strict=True):
        chart.update(action, 0.1 * action + 0.01 * action * velocity, velocity)
    assert chart.trusted
    with pytest.raises(ValueError, match="requires RGB-derived velocity"):
        chart.predict(actions[:1])
    prediction = chart.predict(actions[:1], velocity=velocities[:1])
    assert prediction.shape == (1, 2)
    assert np.isfinite(chart.prediction_uncertainty(actions[:1], velocities[:1])).all()


def test_phase_reset_is_enabled_for_residual_phase_chart_only() -> None:
    assert uses_local_phase_reset("effect_adapter_rgb_online_local_phase")
    assert uses_local_phase_reset("effect_adapter_rgb_online_local_residual_phase_task_aligned")
    assert not uses_local_phase_reset("effect_adapter_rgb_online_local_residual_task_aligned")


def test_online_local_effect_residual_rejects_invalid_predicted_effect() -> None:
    residual = OnlineLocalEffectResidual()
    with pytest.raises(ValueError, match="predicted effect"):
        residual.update(
            np.asarray([1.0, 0.0]), np.asarray([0.2, 0.0]),
            predicted_effect=np.asarray([0.1]),
        )


def test_verified_local_action_effect_model_is_low_dimensional_and_observed_only() -> None:
    model = VerifiedLocalActionEffectModel()
    actions = np.asarray([[0.5, 0.0], [-0.5, 0.0], [0.0, 0.5], [0.0, -0.5]])
    for action in actions:
        model.update(action, np.asarray([0.2 * action[0] + 0.01, 0.4 * action[1] - 0.02]))
    assert model.trusted is True
    prediction = model.predict(np.asarray([[0.25, 0.25], [-0.25, 0.25]]))
    assert prediction.shape == (2, 2)
    assert np.isfinite(prediction).all()
    assert np.allclose(model.predict(np.asarray([[0.25, 0.25]])), prediction[:1])


def test_verified_local_execution_failure_quarantines_without_chart_contamination() -> None:
    model = VerifiedLocalActionEffectModel()
    actions = np.asarray([[0.5, 0.0], [-0.5, 0.0], [0.0, 0.5], [0.0, -0.5]])
    for action in actions:
        model.update(action, action.copy())
    before = model.sample_count
    assert model.trusted is True
    assert model.verify_execution(
        np.asarray([0.2, 0.0]), np.asarray([0.8, 0.0]), tolerance=0.25
    ) is False
    assert model.quarantined is True
    assert model.sample_count == before
    assert model.trusted is False


def test_online_local_effect_residual_ignores_fallback_execution() -> None:
    residual = OnlineLocalEffectResidual()
    accepted = residual.update_verified_execution(
        np.asarray([1.0, 0.0]), np.asarray([0.2, 0.0]), predicted_effect=None,
    )
    assert accepted is False
    assert residual.sample_count == 0
    assert residual.verification_errors == []


def test_online_verification_removes_no_action_inertia_before_direction_check() -> None:
    residual = OnlineLocalEffectResidual()
    accepted = residual.update_verified_execution(
        np.asarray([1.0, 0.0]),
        np.asarray([1.0, 0.0]),
        predicted_effect=np.asarray([-1.0, 0.0]),
        verification_baseline_effect=np.asarray([-2.0, 0.0]),
    )
    assert accepted is True
    assert np.allclose(residual.effects[0], np.asarray([2.0, 0.0]))
    assert np.allclose(residual.directional_scores[0], 1.0)


def test_online_local_effect_residual_rejects_reversed_direction() -> None:
    residual = OnlineLocalEffectResidual(
        verification_threshold=3.0,
        verification_direction_threshold=0.2,
    )
    action = np.asarray([1.0, 0.0])
    for _ in range(4):
        residual.update(action, np.asarray([0.1, 0.0]))
    residual.record_verification(np.asarray([1.0, 0.0]), np.asarray([-1.0, 0.0]))
    residual.record_verification(np.asarray([1.0, 0.0]), np.asarray([-1.0, 0.0]))
    assert residual.trusted is False
    assert np.allclose(residual.predict(np.asarray([[1.0, 0.0]])), 0.0)


def test_ground_truth_candidate_scores_restore_runtime_state() -> None:
    class FakeRuntime:
        def __init__(self) -> None:
            self._state = np.asarray([0.0, 0.0, 0.0, 0.0])

        @property
        def state(self) -> np.ndarray:
            return self._state.copy()

        def reset(self, state: np.ndarray) -> None:
            self._state = np.asarray(state, dtype=np.float64).copy()

        def step_block(self, action: np.ndarray) -> np.ndarray:
            self._state[:2] += np.asarray(action, dtype=np.float64)
            return self.state

    runtime = FakeRuntime()
    scores = ground_truth_candidate_scores(
        runtime,
        np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
        np.asarray([1.0, 0.0]),
    )
    assert np.argmin(scores) == 0
    assert np.allclose(runtime.state, np.zeros(4))


def test_planner_goal_from_rgb_is_the_canonical_calibrated_qpos_target(monkeypatch) -> None:
    import experiments.run_live_pointmaze_planning as live

    class Calibration:
        def position(self, center):
            assert center == (11.0, 13.0)
            return np.asarray([1.17, 1.19], dtype=np.float64)

    monkeypatch.setattr(live, "green_center", lambda _image: (11.0, 13.0))
    target = live.planner_goal_from_rgb(np.zeros((2, 2, 3), dtype=np.uint8), Calibration())
    assert np.allclose(target, [1.17, 1.19])
    assert not np.array_equal(target, live.GOAL_STATE)


def test_privileged_controls_declare_local_waypoint_objectives(monkeypatch) -> None:
    """Local privileged controls must never masquerade as terminal-task ceilings."""

    import experiments.run_live_pointmaze_planning as live

    class Calibration:
        def position(self, center):
            return np.asarray(center, dtype=np.float64)

    class Maze:
        def path_waypoints(self, current, goal):
            del current, goal
            return [np.asarray([9.0, 9.0], dtype=np.float64)]

    class Runtime:
        pass

    bank = np.zeros((32, 2), dtype=np.float64)
    captured: dict[str, np.ndarray] = {}

    def score(runtime, actions, target):
        del runtime, actions
        captured["one_block"] = np.asarray(target, dtype=np.float64)
        return np.zeros(len(bank), dtype=np.float64)

    def mpc(runtime, target, directions, magnitudes, **kwargs):
        del runtime, directions, magnitudes, kwargs
        captured["mpc"] = np.asarray(target, dtype=np.float64)
        return bank[0], {"privileged_state_used": True}

    monkeypatch.setattr(live, "green_center", lambda _image: (0.0, 0.0))
    monkeypatch.setattr(live, "ground_truth_candidate_scores", score)
    monkeypatch.setattr(live, "perfect_dynamics_mpc_action", mpc)
    common = dict(
        model=None, readout=None, current_rgb=np.zeros((2, 2, 3)), goal_rgb=np.zeros((2, 2, 3)),
        context_images=[], context_proprios=[np.zeros(4)], calibration=Calibration(),
        action_mean=np.zeros(2), action_std=np.ones(2), directions=8, magnitudes=[0.25],
        dynamics=None, maze_map=Maze(), planner_map="rgb_maze", kp=1.0, kv=0.0,
        wm_dynamics_tolerance=0.0, rng=np.random.default_rng(0), runtime=Runtime(),
        fixed_candidate_bank=bank,
    )
    _, one_block = live.choose_action("ground_truth_candidate_oracle", **common)
    _, mpc_result = live.choose_action("perfect_dynamics_mpc", **common)
    assert np.array_equal(captured["one_block"], np.asarray([9.0, 9.0]))
    assert np.array_equal(captured["mpc"], np.asarray([9.0, 9.0]))
    assert one_block["action_transfer_gate"]["selection_objective_id"] == "one_step_active_waypoint_distance_v1"
    assert one_block["action_transfer_gate"]["search_completeness"] == "exact_one_step_fixed_bank_only"
    assert mpc_result["action_transfer_gate"]["selection_objective_id"] == "bounded_beam_active_waypoint_distance_v1"
    assert mpc_result["action_transfer_gate"]["search_completeness"] == "bounded_beam_not_terminal_complete"


def test_depth_one_perfect_mpc_uses_observed_root_candidate_bank() -> None:
    class FakeRuntime:
        def __init__(self) -> None:
            self._state = np.asarray([0.10, 0.20, 0.0, 0.0])

        @property
        def state(self) -> np.ndarray:
            return self._state.copy()

        def reset(self, state: np.ndarray) -> None:
            self._state = np.asarray(state, dtype=np.float64).copy()

        def step_block(self, action: np.ndarray) -> np.ndarray:
            self._state[:2] += np.asarray(action, dtype=np.float64)
            return self.state

    runtime = FakeRuntime()
    target = np.asarray([1.0, 1.0])
    observed_current = np.asarray([0.0, 0.5])
    candidates = action_bank(observed_current, target, 8, [0.25, 0.50])
    expected = int(np.argmin(ground_truth_candidate_scores(runtime, candidates, target)))
    selected, info = perfect_dynamics_mpc_action(
        runtime, target, 8, [0.25, 0.50], root_current=observed_current,
        depth=1, beam_width=len(candidates),
    )
    assert np.allclose(selected, candidates[expected])
    assert info["privileged_state_used"] is True
    assert np.allclose(runtime.state, np.asarray([0.10, 0.20, 0.0, 0.0]))


def test_parse_episode_list_accepts_explicit_root_lists() -> None:
    assert parse_episode_list("1800, 1812,1899", "episodes") == [1800, 1812, 1899]
    assert parse_episode_list("", "episodes") is None


def test_load_episode_states_does_not_require_test_rgb(tmp_path) -> None:
    import torch

    states = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
    torch.save(torch.from_numpy(states), tmp_path / "states.pth")

    loaded = load_episode_states(tmp_path, 2)

    assert np.array_equal(loaded, states[2])
    with pytest.raises(ValueError, match="outside"):
        load_episode_states(tmp_path, 3)


@pytest.mark.parametrize("value", ["1800,1800", "-1,1800", "abc,1800"])
def test_parse_episode_list_rejects_unsafe_root_lists(value: str) -> None:
    with pytest.raises(ValueError):
        parse_episode_list(value, "episodes")


def test_output_path_rejects_existing_directory(tmp_path) -> None:
    with pytest.raises(ValueError, match="JSON file path"):
        validate_output_path(tmp_path)
    assert validate_output_path(tmp_path / "report.json") == tmp_path / "report.json"


def test_select_gate_tolerance_uses_calibration_performance_before_fallback_rate() -> None:
    candidates = [
        {"tolerance": 0.50, "aggregate": {"success_rate": 1.0, "mean_final_distance": 0.20, "gate_fallback_rate": 0.10}},
        {"tolerance": 0.10, "aggregate": {"success_rate": 1.0, "mean_final_distance": 0.20, "gate_fallback_rate": 0.40}},
        {"tolerance": 0.05, "aggregate": {"success_rate": 0.0, "mean_final_distance": 0.01, "gate_fallback_rate": 0.00}},
    ]
    assert select_gate_tolerance(candidates) == 0.50


def test_guarded_world_model_falls_back_when_dynamics_regret_exceeds_tolerance(monkeypatch) -> None:
    """The runtime gate must reject an unsafe WM ranking, not silently use it."""

    def marker(x: int, y: int) -> np.ndarray:
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        image[y - 1 : y + 2, x - 1 : x + 2] = [20, 180, 20]
        return image

    current_rgb = marker(2, 2)
    goal_rgb = marker(10, 10)
    identity = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    calibration = VisualCalibration(
        pixel_to_position=identity,
        state_to_pixel=identity,
        pixel_delta_to_velocity=identity,
        position_rmse=np.zeros(2),
        velocity_rmse=np.zeros(2),
        calibration_episodes=(0,),
    )
    dynamics = VisualDynamicsCalibration(
        position_delta_weights=np.asarray(
            [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
        ),
        velocity_weights=np.zeros((5, 2)),
        position_delta_rmse=np.zeros(2),
        velocity_rmse=np.zeros(2),
        calibration_episodes=(0,),
    )
    monkeypatch.setattr(
        "experiments.run_live_pointmaze_planning.model_candidate_scores",
        lambda *args, **kwargs: np.asarray([0.0, 0.0, 0.0, -1.0] + [1.0] * 4),
    )

    _action, decision = choose_action(
        "wm_rgb_guarded",
        None,
        None,
        current_rgb,
        goal_rgb,
        [current_rgb, current_rgb],
        [np.zeros(4), np.zeros(4)],
        calibration,
        np.zeros(2),
        np.ones(2),
        8,
        [0.25],
        dynamics,
        None,
        "direct",
        0.8,
        0.2,
        0.0,
        np.random.default_rng(0),
    )

    gate = decision["action_transfer_gate"]
    assert gate["fallback_used"] is True
    assert gate["fallback_reason"] == "wm_action_outside_calibration_dynamics_tolerance"


def test_effect_consistency_gate_falls_back_on_visual_disagreement(monkeypatch) -> None:
    """A released effect rank must still abstain when visual dynamics rejects it."""

    def marker(x: int, y: int) -> np.ndarray:
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        image[y - 1 : y + 2, x - 1 : x + 2] = [20, 180, 20]
        return image

    current_rgb = marker(2, 2)
    goal_rgb = marker(10, 10)
    identity = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    calibration = VisualCalibration(
        pixel_to_position=identity,
        state_to_pixel=identity,
        pixel_delta_to_velocity=identity,
        position_rmse=np.zeros(2),
        velocity_rmse=np.zeros(2),
        calibration_episodes=(0,),
    )
    dynamics = VisualDynamicsCalibration(
        position_delta_weights=np.zeros((5, 2)),
        velocity_weights=np.zeros((5, 2)),
        position_delta_rmse=np.zeros(2),
        velocity_rmse=np.zeros(2),
        calibration_episodes=(0,),
    )

    class Policy:
        latent_support = {"threshold": 1.0}
        action_support = {"threshold": 1.0}
        regret_calibration = {"status": "ok", "bound": 0.01}

    monkeypatch.setattr(
        "experiments.run_live_pointmaze_planning.effect_policy_candidate_scores",
        lambda *args, **kwargs: {
            "costs": np.asarray([0.0] + [1.0] * 7),
            "effects": np.zeros((8, 2)),
            "terminals": np.zeros((8, 2)),
            "action_support_distances": np.zeros(8),
            "latent_support_distance": 0.0,
        },
    )
    monkeypatch.setattr(
        "experiments.run_live_pointmaze_planning.visual_dynamics_scores",
        lambda *args, **kwargs: np.asarray([1.0, 0.0] + [2.0] * 6),
    )

    _action, decision = choose_action(
        "effect_adapter_rgb_htsc_consistency",
        None,
        None,
        current_rgb,
        goal_rgb,
        [current_rgb, current_rgb],
        [np.zeros(4), np.zeros(4)],
        calibration,
        np.zeros(2),
        np.ones(2),
        8,
        [0.25],
        dynamics,
        None,
        "direct",
        0.8,
        0.2,
        0.0,
        np.random.default_rng(0),
        effect_policy=Policy(),
        effect_visual_tolerance=0.10,
    )

    gate = decision["action_transfer_gate"]
    assert gate["effect_policy_released"] is True
    assert gate["effect_visual_consistency_passed"] is False
    assert gate["effect_visual_action_disagreement"] is True
    assert gate["effect_visual_release"] is False
    assert gate["fallback_used"] is True
    assert gate["fallback_reason"] == "effect_visual_consistency_failed"
    assert decision["selected_index"] == 1


def test_effect_visual_tolerance_selection_keeps_visual_only_control_when_tied() -> None:
    """A release-heavy threshold cannot win a calibration tie by itself."""

    candidates = [
        {
            "tolerance": -1.0,
            "aggregate": {
                "success_rate": 1.0,
                "mean_final_distance": 0.10,
                "gate_fallback_rate": 1.0,
            },
        },
        {
            "tolerance": 0.20,
            "aggregate": {
                "success_rate": 1.0,
                "mean_final_distance": 0.10,
                "gate_fallback_rate": 0.0,
            },
        },
    ]
    assert select_effect_visual_tolerance(candidates) == pytest.approx(-1.0)


def test_effect_visual_tolerance_selection_releases_only_for_calibration_gain() -> None:
    candidates = [
        {
            "tolerance": -1.0,
            "aggregate": {
                "success_rate": 1.0,
                "mean_final_distance": 0.20,
                "gate_fallback_rate": 1.0,
            },
        },
        {
            "tolerance": 0.05,
            "aggregate": {
                "success_rate": 1.0,
                "mean_final_distance": 0.15,
                "gate_fallback_rate": 0.50,
            },
        },
    ]
    assert select_effect_visual_tolerance(candidates) == pytest.approx(0.05)


def test_local_jacobian_probe_status_does_not_require_frozen_residual() -> None:
    """Observed-only probe arms must not abort while recording their chart."""

    chart = VerifiedLocalActionEffectModel()
    assert probe_residual_trusted(None, chart) is False
    for action in (
        np.asarray([0.5, 0.0]), np.asarray([-0.5, 0.0]),
        np.asarray([0.0, 0.5]), np.asarray([0.0, -0.5]),
    ):
        chart.update(action, action * 0.1)
    assert probe_residual_trusted(None, chart) is True


def test_local_jacobian_uses_observed_chart_not_absent_frozen_teacher() -> None:
    assert uses_frozen_effect_residual("effect_adapter_rgb_online_local_jacobian") is False
    assert uses_frozen_effect_residual("effect_adapter_rgb_online_local_direct") is False
    assert uses_frozen_effect_residual("effect_adapter_rgb_online") is True
    assert uses_local_visual_residual("effect_adapter_rgb_online_local_residual_task_aligned")
    assert not uses_local_visual_residual("effect_adapter_rgb_online_local_task_aligned")


def test_local_action_effect_chart_reports_crossfitted_probe_error() -> None:
    chart = VerifiedLocalActionEffectModel(ridge=1.0e-6)
    for action in (
        np.asarray([0.5, 0.0]), np.asarray([-0.5, 0.0]),
        np.asarray([0.0, 0.5]), np.asarray([0.0, -0.5]),
    ):
        chart.update(action, np.asarray([0.2 * action[0], -0.1 * action[1]]))
    assert chart.trusted is True
    assert chart.leave_one_out_error() < 1.0e-4

    noisy = VerifiedLocalActionEffectModel(ridge=1.0e-6)
    for index, action in enumerate((
        np.asarray([0.5, 0.0]), np.asarray([-0.5, 0.0]),
        np.asarray([0.0, 0.5]), np.asarray([0.0, -0.5]),
    )):
        noisy.update(action, np.asarray([0.2 * action[0], -0.1 * action[1] + (1.0 if index == 3 else 0.0)]))
    assert noisy.leave_one_out_error() > 0.1


def test_visual_task_cost_excludes_controller_regularizers_for_local_comparison() -> None:
    dynamics = VisualDynamicsCalibration(
        position_delta_weights=np.asarray(
            [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
        ),
        velocity_weights=np.zeros((5, 2)),
        position_delta_rmse=np.zeros(2),
        velocity_rmse=np.zeros(2),
        calibration_episodes=(0,),
    )
    costs = visual_dynamics_task_costs(
        np.zeros(2), np.zeros(2), np.asarray([1.0, 0.0]),
        np.asarray([[0.5, 0.0], [0.0, 0.5]]), dynamics,
    )
    assert np.allclose(costs, np.asarray([0.5, np.sqrt(1.25)]))


def test_action_projection_keeps_identical_supported_endpoint() -> None:
    dynamics = VisualDynamicsCalibration(
        position_delta_weights=np.asarray(
            [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
        ),
        velocity_weights=np.zeros((5, 2)),
        position_delta_rmse=np.zeros(2),
        velocity_rmse=np.zeros(2),
        calibration_episodes=(0,),
    )
    action = np.asarray([0.25, -0.50])
    projected, alpha, score, support = project_action_to_dynamics_support(
        action, action, np.zeros(2), np.zeros(2), np.asarray([1.0, 1.0]),
        dynamics, 0.8, 0.2, 0.0,
    )
    assert np.allclose(projected, action)
    assert alpha == pytest.approx(1.0)
    assert score == pytest.approx(support)


def test_action_projection_moves_out_of_support_wm_action_toward_dynamics() -> None:
    dynamics = VisualDynamicsCalibration(
        position_delta_weights=np.asarray(
            [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
        ),
        velocity_weights=np.zeros((5, 2)),
        position_delta_rmse=np.zeros(2),
        velocity_rmse=np.zeros(2),
        calibration_episodes=(0,),
    )
    wm = np.asarray([1.0, 0.0])
    safe = np.asarray([0.0, 1.0])
    projected, alpha, score, support = project_action_to_dynamics_support(
        wm, safe, np.zeros(2), np.zeros(2), np.asarray([0.0, 1.0]),
        dynamics, 0.8, 0.2, 0.01,
    )
    assert 0.0 <= alpha < 1.0
    assert np.all(projected >= -1.0) and np.all(projected <= 1.0)
    assert score <= support + 0.01 + 1.0e-8
    assert not np.allclose(projected, wm)
