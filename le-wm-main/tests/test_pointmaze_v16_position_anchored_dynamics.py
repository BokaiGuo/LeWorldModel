from __future__ import annotations

import numpy as np

from experiments.run_pointmaze_v16_position_anchored_dynamics import (
    _oracle_headroom_selection,
    _position_anchored_state,
    _select,
)


class _Calibration:
    def position(self, center: tuple[float, float]) -> np.ndarray:
        return np.asarray(center, dtype=np.float64) / 10.0


class _Dynamics:
    def predict_position_delta(self, velocity: np.ndarray, action: np.ndarray) -> np.ndarray:
        return np.asarray(velocity, dtype=np.float64) * 0.1 + np.asarray(action, dtype=np.float64)


def _green_marker() -> np.ndarray:
    image = np.zeros((3, 3, 3), dtype=np.uint8)
    image[1, 1] = [0, 255, 0]
    return image


def test_position_anchor_keeps_direct_rgb_position_and_v14_velocity() -> None:
    value = _position_anchored_state(np.asarray([9.0, 8.0, 1.5, -2.5]), _green_marker(), _Calibration())
    assert np.allclose(value, [0.1, 0.1, 1.5, -2.5])


def test_anchored_selector_uses_direct_position_but_estimated_velocity() -> None:
    image = _green_marker()
    estimated = np.asarray([4.0, 4.0, 0.0, 1.0])
    bank = np.asarray([[1.0, 0.0], [0.0, -1.0]])
    chosen = _select(
        "rgb_position_anchored_velocity", estimated=estimated, image=image, calibration=_Calibration(),
        current=np.asarray([8.0, 8.0, 0.0, 0.0]), target=np.asarray([1.0, 0.1]), bank=bank,
        dynamics=_Dynamics(), model=None,
    )
    # Direct RGB position is [.1, .1]; velocity makes the first candidate's
    # endpoint [1.1, .2], while the second moves away from the target.
    assert chosen == 0


def test_oracle_headroom_gate_rejects_noisy_near_zero_planner_gap() -> None:
    candidates = [
        {"goal_qpos": [0.8, 1.8], "oracle_minus_greedy_success_rate": 0.0,
         "oracle_minus_greedy_terminal_distance_ci": {"lower": -0.01, "mean_gain": 0.04, "upper": 0.09}},
        {"goal_qpos": [2.8, 1.8], "oracle_minus_greedy_success_rate": 0.0,
         "oracle_minus_greedy_terminal_distance_ci": {"lower": -0.02, "mean_gain": 0.05, "upper": 0.10}},
    ]
    selected, gate = _oracle_headroom_selection(candidates, min_lower_bound=0.0)
    assert selected is None
    assert gate["status"] == "rejected_insufficient_oracle_headroom"


def test_oracle_headroom_gate_selects_by_lower_bound_not_mean_only() -> None:
    candidates = [
        {"goal_qpos": [0.8, 1.8], "oracle_minus_greedy_success_rate": 0.0,
         "oracle_minus_greedy_terminal_distance_ci": {"lower": 0.01, "mean_gain": 0.30, "upper": 0.60}},
        {"goal_qpos": [2.8, 1.8], "oracle_minus_greedy_success_rate": 0.0,
         "oracle_minus_greedy_terminal_distance_ci": {"lower": 0.02, "mean_gain": 0.04, "upper": 0.06}},
    ]
    selected, gate = _oracle_headroom_selection(candidates, min_lower_bound=0.0)
    assert selected is not None
    assert selected["goal_qpos"] == [2.8, 1.8]
    assert gate["status"] == "passed_oracle_headroom"
