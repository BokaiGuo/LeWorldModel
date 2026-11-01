import numpy as np
import pytest

from experiments.spectral_reachability.p1_dev303_dynamics_saved_analysis_20261102_main import measures


def test_metrics_keep_pusher_velocity_and_object_separate():
    goal = np.zeros(7)
    state = np.array([3., 4., 0., 0., 0., 6., 8.])
    row = measures(state, goal)
    assert row["pusher_position_error"] == 5
    assert row["velocity_squared_error"] == 100
    assert row["object_position_error"] == row["object_angle_error"] == 0


def test_angle_is_two_pi_periodic():
    state = np.zeros(7)
    state[4] = 2 * np.pi + .2
    assert measures(state, np.zeros(7))["object_angle_error"] == pytest.approx(.2)


def test_invalid_state_rejected():
    with pytest.raises(ValueError):
        measures(np.zeros(6), np.zeros(7))
