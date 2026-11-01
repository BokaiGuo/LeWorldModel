import numpy as np
import pytest
from experiments.spectral_reachability import main_midfeedback_physical_effect_20261028_v1 as m


def test_native_success_is_strict_and_not_velocity_distance():
    g = np.zeros(7)
    s = np.array([19., 0., 0., 0., 0., 1000., 0.])
    assert m.metrics(s, g)['native_success']
    assert m.metrics(s, g)['native_reward'] < -1000
    s[0] = 20.
    assert not m.metrics(s, g)['native_success']
    s[0] = 0.; s[4] = np.pi/9
    assert not m.metrics(s, g)['native_success']


def test_all_native_axes_and_wrapping():
    s = np.array([3., 4., 0., 12., 2*np.pi-.1, 6., 8.])
    x = m.metrics(s, np.zeros(7))
    assert x['agent_xy_error'] == 5 and x['object_xy_error'] == 12
    assert x['joint_xy_error'] == 13 and x['agent_velocity_error'] == 10
    assert x['wrapped_angle_error'] == abs(float(np.arctan2(np.sin(s[4]), np.cos(s[4]))))


@pytest.mark.parametrize('state', [[0.]*6, [float('nan')]*7])
def test_unknown_or_invalid_state_not_synthetic_success(state):
    with pytest.raises(ValueError):
        m.metrics(state, [0.]*7)
