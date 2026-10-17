from __future__ import annotations

import numpy as np
import pytest

from experiments.run_live_pointmaze_planning import (
    PRIVILEGED_METHODS,
    GOAL_STATE,
    VisualCalibration,
    VisualDynamicsCalibration,
    action_bank,
    choose_action,
    ground_truth_candidate_scores,
    green_center,
    load_episode_states,
    parse_episode_list,
    perfect_dynamics_mpc_action,
    project_action_to_dynamics_support,
    select_gate_tolerance,
)


def test_exact_dynamics_planners_are_reported_as_privileged() -> None:
    assert "ground_truth_candidate_oracle" in PRIVILEGED_METHODS
    assert "perfect_dynamics_mpc" in PRIVILEGED_METHODS


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
