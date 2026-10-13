import numpy as np

from experiments.run_pusht_contact_aware_closure import (
    ContactAwareModel,
    block_motion,
    contact_aware_candidates,
    normalize_pose_batch,
    pose_vector,
    predict_pose_batch,
    transition_features,
)


def test_transition_features_encode_relative_contact_geometry():
    pose = np.array([100.0, 120.0, 140.0, 150.0, 1.0, 0.0])
    action = np.array([0.2, -0.3])
    features = transition_features(pose, action)
    assert features.shape == (17,)
    assert np.isfinite(features).all()
    np.testing.assert_allclose(features[8:10], [40.0, 30.0])


def test_pose_vector_can_retain_agent_velocity_for_contact_dynamics():
    state = np.array([100.0, 120.0, 140.0, 150.0, 1.0, 0.25, 3.0])
    pose_only = pose_vector(state)
    augmented = pose_vector(state, include_agent_velocity=True)
    assert pose_only.shape == (6,)
    assert augmented.shape == (8,)
    np.testing.assert_allclose(augmented[-2:], [0.25, 3.0])
    assert transition_features(augmented, np.array([0.2, -0.3])).shape == (19,)


def test_block_motion_wraps_angle_and_ignores_agent_only_motion():
    before = np.array([100.0, 100.0, 200.0, 200.0, 2.0 * np.pi - 1e-6, 0.0, 0.0])
    after_agent_only = before.copy(); after_agent_only[:2] += 20.0
    after_angle_wrap = before.copy(); after_angle_wrap[4] = 1e-6
    assert not block_motion(before, after_agent_only)
    assert not block_motion(before, after_angle_wrap)
    after_block = before.copy(); after_block[2] += 1e-2
    assert block_motion(before, after_block)


def test_normalize_pose_batch_preserves_unit_orientation_and_bounds_position():
    values = np.array([[600.0, -4.0, 700.0, -8.0, 3.0, 4.0]])
    normalized = normalize_pose_batch(values)
    np.testing.assert_allclose(np.linalg.norm(normalized[:, 4:6], axis=1), [1.0])
    assert (normalized[:, :4] >= 5.0).all()
    assert (normalized[:, :4] <= 507.0).all()


def test_contact_aware_candidates_include_multistage_and_angular_variants():
    state = np.array([100.0, 100.0, 200.0, 200.0, 0.0, 0.0, 0.0])
    goal = np.array([100.0, 100.0, 300.0, 300.0, 0.8, 0.0, 0.0])
    candidates = contact_aware_candidates(np.random.default_rng(3), state, goal, 12, 2)
    assert candidates.shape == (12, 2, 2)
    assert np.isfinite(candidates).all()
    assert np.unique(candidates.reshape(12, -1), axis=0).shape[0] > 4


class _Regressor:
    def predict(self, features):
        del features
        return np.tile(np.array([[1.0, 2.0, 10.0, 11.0, 0.0, 0.0]]), (1, 1))


class _Classifier:
    def __init__(self, probability):
        self.probability = probability

    def predict_proba(self, features):
        del features
        return np.tile(np.array([[1.0 - self.probability, self.probability]]), (1, 1))


def test_no_contact_gate_freezes_block_pose():
    current = np.array([[100.0, 100.0, 200.0, 200.0, 1.0, 0.0]])
    features = np.zeros((1, 13))
    model = ContactAwareModel(_Regressor(), _Classifier(0.1), _Regressor(), _Classifier(0.1), 0.85)
    predicted = predict_pose_batch(model, current, features)
    np.testing.assert_allclose(predicted[:, 2:6], current[:, 2:6])
