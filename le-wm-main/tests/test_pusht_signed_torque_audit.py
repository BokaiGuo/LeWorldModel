import numpy as np

from experiments.run_pusht_signed_torque_audit import (
    angle_error,
    contact_features,
    pulse_actions,
    raw_signed_delta,
    wrap_angle,
)


def test_wrap_and_raw_delta_are_circular():
    assert abs(float(wrap_angle(3.5)) + 2.7831853071795862) < 1e-8
    before = np.array([0.0, 0.0, 1.0, 1.0, 6.20, 0.0, 0.0])
    after = before.copy()
    after[4] = 0.05
    assert abs(raw_signed_delta(before, after) - 0.13318530717958623) < 1e-8
    assert abs(angle_error(after[4], before[4]) - 0.13318530717958623) < 1e-8


def test_pulse_actions_are_paired_and_contact_inward():
    state = np.array([100.0, 100.0, 130.0, 100.0, 0.4, 0.0, 0.0])
    actions, signs = pulse_actions(state, (0.06, 0.12), 0.08)
    assert actions.shape == (4, 2)
    assert np.array_equal(signs, np.array([1.0, -1.0, 1.0, -1.0]))
    normal = np.array([1.0, 0.0])
    assert np.all(actions @ normal > 0.0)
    assert np.isfinite(actions).all()


def test_contact_features_are_finite_and_include_signed_moment():
    state = np.array([100.0, 100.0, 130.0, 100.0, 0.4, 0.0, 0.0])
    positive = contact_features(state, np.array([0.08, 0.12]))
    negative = contact_features(state, np.array([0.08, -0.12]))
    assert positive.shape == (20,)
    assert np.isfinite(positive).all()
    assert positive[13] != negative[13]
