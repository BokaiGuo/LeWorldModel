from __future__ import annotations

import numpy as np
import pytest

from experiments.run_live_pointmaze_planning import (
    GOAL_STATE,
    VisualCalibration,
    VisualDynamicsCalibration,
    action_bank,
    choose_action,
    green_center,
)


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
