import numpy as np
import gymnasium as gym
import stable_worldmodel  # noqa: F401

from experiments.run_pusht_mode_aware_shared_candidate import (
    action_moment_sign,
    boundary_inward_normal,
    candidate_actions,
    contact_chart_features,
    current_mode_for_env,
    make_boundary_contact_roots,
    online_effect_key,
    physical_contact_geometry,
    update_contact_sign_consensus,
    current_mode,
    planning_pose_cost,
    predicted_state,
    smallest_action_for_moment_sign,
    successor_mode,
    transition_target,
    update_mode_sign_consensus,
    update_observed_sign_calibration,
)


def _state(distance: float) -> np.ndarray:
    return np.asarray([100.0, 100.0, 100.0 + distance, 100.0, 0.2, 0.0, 0.0])


def test_current_mode_thresholds_are_explicit():
    assert current_mode(_state(100.0)) == 0
    assert current_mode(_state(60.0)) == 1
    assert current_mode(_state(30.0)) == 2


def test_successor_modes_distinguish_stall_slide_and_separation():
    before = _state(35.0)
    stalled = before.copy()
    sliding = before.copy()
    sliding[2] += 3.0
    separated = before.copy()
    separated[:2] = [0.0, 0.0]
    assert successor_mode(before, stalled) == 1
    assert successor_mode(before, sliding) == 2
    assert successor_mode(before, separated) == 3


def test_transition_target_is_finite_and_reconstructs_pose():
    before = _state(35.0)
    after = before.copy()
    after[2] += 1.5
    after[4] += 0.25
    target = transition_target({"before": before, "after": after})
    assert target.shape == (6,)
    assert np.isfinite(target).all()
    reconstructed = predicted_state(before, target)
    np.testing.assert_allclose(reconstructed[2:5], after[2:5], atol=1e-8)


def test_observed_sign_calibration_uses_action_and_angle_sign():
    assert update_observed_sign_calibration(None, 1.0, -0.2, 1.0e-3) == (-1.0, True)
    assert update_observed_sign_calibration(-1.0, 1.0, 0.0, 1.0e-3) == (-1.0, False)


def test_smallest_signed_candidate_is_deterministic():
    state = _state(30.0)
    actions = np.asarray([[0.1, 0.0], [0.0, 0.05], [0.0, -0.1]], dtype=np.float64)
    signs = [action_moment_sign(state, action) for action in actions]
    assert set(signs) == {-1.0, 0.0, 1.0}
    index = smallest_action_for_moment_sign(state, actions, signs[1])
    assert index is not None
    assert signs[index] == signs[1]
    assert smallest_action_for_moment_sign(state, actions, 0.0) is None


def test_phase_aware_cost_prioritizes_translation_when_far():
    current = _state(30.0)
    goal = current.copy()
    goal[2:4] = current[2:4] + np.asarray([100.0, 0.0])
    imagined = current.copy()
    imagined[2:4] += np.asarray([10.0, 0.0])
    rotated = current.copy()
    rotated[4] += 0.8
    assert planning_pose_cost(current, imagined, goal, True) < planning_pose_cost(current, rotated, goal, True)


def test_mode_sign_consensus_accumulates_informative_pulses():
    votes = {}
    assert update_mode_sign_consensus(votes, 2, 1.0, -0.2, 1.0e-3) == (-1.0, True)
    assert update_mode_sign_consensus(votes, 2, 1.0, 0.0, 1.0e-3) == (None, False)
    assert update_mode_sign_consensus(votes, 2, 1.0, 0.2, 1.0e-3) == (None, True)
    assert votes[2] == 0


def test_contact_chart_is_finite_and_region_consensus_is_separate():
    state = _state(30.0)
    features = contact_chart_features(state, np.asarray([0.1, -0.05]), np.zeros(2))
    assert features.ndim == 1
    assert np.isfinite(features).all()
    votes = {}
    assert update_contact_sign_consensus(votes, "shape0:edge2:bin1", 1.0, -0.2, 1.0e-3) == (-1.0, True)
    assert update_contact_sign_consensus(votes, "shape0:edge2:bin1", 1.0, 0.2, 1.0e-3) == (None, True)
    assert votes["shape0:edge2:bin1"] == 0


def test_boundary_contact_protocol_generates_real_polygon_contacts():
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    try:
        roots = make_boundary_contact_roots(env, 4, 20261016)
        gaps = [float(physical_contact_geometry(env, root["state"])["distance"]) for root in roots]
        assert all(12.0 <= gap <= 18.0 for gap in gaps)
        assert all(current_mode_for_env(env, root["state"]) == 2 for root in roots)
    finally:
        env.close()


def test_boundary_action_frame_points_into_nearest_polygon_edge():
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    try:
        root = make_boundary_contact_roots(env, 1, 20261017)[0]
        state = root["state"]
        geometry = physical_contact_geometry(env, state)
        normal = boundary_inward_normal(env, state)
        toward_edge = np.asarray(geometry["contact_point_world"]) - state[:2]
        assert float(np.dot(normal, toward_edge)) > 0.99 * float(np.linalg.norm(toward_edge))
        actions = candidate_actions(
            env, state, (0.06, 0.10), 0.08, boundary_action_frame=True,
        )
        assert actions.shape[1] == 2
        assert float(np.max(actions @ normal)) > 0.0
        assert np.isfinite(actions).all()
    finally:
        env.close()


def test_online_effect_key_does_not_alias_local_action_directions():
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    try:
        state = make_boundary_contact_roots(env, 1, 20261018)[0]["state"]
        normal = boundary_inward_normal(env, state)
        tangent = np.asarray([-normal[1], normal[0]], dtype=np.float64)
        # Equal-norm candidates with the same tangent sign can still have
        # different normal/tangent composition and therefore different
        # sticking/sliding effects.  The runtime cache must keep them apart.
        first = 0.10 * (np.cos(np.pi / 6.0) * normal + np.sin(np.pi / 6.0) * tangent)
        second = 0.10 * (np.cos(np.pi / 3.0) * normal + np.sin(np.pi / 3.0) * tangent)
        assert np.isclose(np.linalg.norm(first), np.linalg.norm(second))
        assert online_effect_key(env, state, first) != online_effect_key(env, state, second)
    finally:
        env.close()
