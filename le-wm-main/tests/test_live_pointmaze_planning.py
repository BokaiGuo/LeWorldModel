from __future__ import annotations

import numpy as np
import pytest

from experiments.run_live_pointmaze_planning import (
    PRIVILEGED_METHODS,
    GOAL_STATE,
    OnlineLocalEffectResidual,
    VerifiedLocalActionEffectModel,
    VisualCalibration,
    VisualDynamicsCalibration,
    RGBMazeMap,
    action_bank,
    active_probe_actions,
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
    project_action_to_dynamics_support,
    select_gate_tolerance,
    validate_effect_policy_execution_contract,
    validate_output_path,
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

    assert deployment_context_windows(None) == 2
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
    assert gate["fallback_used"] is True
    assert gate["fallback_reason"] == "effect_visual_consistency_failed"
    assert decision["selected_index"] == 1


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
