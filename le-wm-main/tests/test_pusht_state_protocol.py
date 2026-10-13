import numpy as np

from experiments.pusht_state import raw_angle_error, raw_block_angle, raw_state_to_pose_chart


def test_raw_state_uses_scalar_angle_not_agent_velocity():
    state = np.asarray([100.0, 120.0, 200.0, 220.0, 1.25, -93.0, 344.0])
    assert raw_block_angle(state) == 1.25
    assert np.isclose(raw_angle_error(state[4], 1.25), 0.0)


def test_raw_state_pose_chart_is_explicit_sincos_conversion():
    state = np.asarray([100.0, 120.0, 200.0, 220.0, 1.25, -93.0, 344.0])
    chart = raw_state_to_pose_chart(state)
    np.testing.assert_allclose(chart, [100.0, 120.0, 200.0, 220.0, np.sin(1.25), np.cos(1.25)])


def test_wrapped_scalar_angle_error_handles_branch_cut():
    assert np.isclose(raw_angle_error(-np.pi + 0.01, np.pi - 0.01), 0.02)
