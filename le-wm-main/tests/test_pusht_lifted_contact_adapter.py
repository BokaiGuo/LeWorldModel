import numpy as np

from experiments.run_pusht_lifted_contact_adapter import (
    contact_aware_pulse_actions,
    contact_recovery_action,
    lifted_contact_features,
)


def test_lifted_features_are_finite_and_expanded():
    state = np.asarray([100.0, 100.0, 130.0, 100.0, 0.4, 1.0, -2.0])
    value = lifted_contact_features(state, np.asarray([0.08, 0.12]))
    assert value.ndim == 1
    assert value.shape[0] > 50
    assert np.isfinite(value).all()


def test_lifted_features_use_previous_action_history():
    state = np.asarray([100.0, 100.0, 130.0, 100.0, 0.4, 1.0, -2.0])
    action = np.asarray([0.08, 0.12])
    first = lifted_contact_features(state, action, np.zeros(2))
    second = lifted_contact_features(state, action, np.asarray([-0.2, 0.1]))
    assert not np.allclose(first, second)


def test_lifted_features_are_periodic_in_object_angle():
    state = np.asarray([100.0, 100.0, 130.0, 100.0, 0.4, 1.0, -2.0])
    action = np.asarray([0.08, 0.12])
    shifted = state.copy()
    shifted[4] += 2.0 * np.pi
    np.testing.assert_allclose(
        lifted_contact_features(state, action),
        lifted_contact_features(shifted, action),
        atol=1e-8,
    )


def test_contact_mode_gate_changes_only_the_action_bank_after_contact_loss():
    near = np.asarray([100.0, 100.0, 130.0, 100.0, 0.4, 0.0, 0.0])
    far = near.copy()
    far[0] = 180.0
    near_actions, _ = contact_aware_pulse_actions(near, (0.06, 0.10), 0.08)
    far_actions, _ = contact_aware_pulse_actions(far, (0.06, 0.10), 0.08)
    near_tangent = np.max(np.abs(near_actions @ np.asarray([0.0, 1.0])))
    far_tangent = np.max(np.abs(far_actions @ np.asarray([0.0, 1.0])))
    assert far_tangent < near_tangent
    assert not np.allclose(far_actions, near_actions)


def test_contact_recovery_action_uses_declared_support_normal():
    normal = np.asarray([3.0, 4.0])
    action = contact_recovery_action(normal, 0.06)
    np.testing.assert_allclose(action, np.asarray([0.036, 0.048], dtype=np.float32))
    assert np.isclose(np.linalg.norm(action), 0.06)
