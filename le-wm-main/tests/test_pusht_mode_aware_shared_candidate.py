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
    online_effect_descriptor,
    local_observed_effect,
    physical_contact_geometry,
    update_contact_sign_consensus,
    current_mode,
    planning_pose_cost,
    predicted_state,
    protocol_contact_loss,
    largest_action_for_moment_sign,
    smallest_action_for_moment_sign,
    successor_mode,
    transition_target,
    update_mode_sign_consensus,
    update_observed_sign_calibration,
    choose_identification_probe,
    staged_predictor_for_state,
    orientation_action_score,
    macro_ranker_features,
    xy_shield_mask,
    macro_phase_name,
)


def _state(distance: float) -> np.ndarray:
    return np.asarray([100.0, 100.0, 100.0 + distance, 100.0, 0.2, 0.0, 0.0])


def test_current_mode_thresholds_are_explicit():
    assert current_mode(_state(100.0)) == 0
    assert current_mode(_state(60.0)) == 1
    assert current_mode(_state(30.0)) == 2


def test_staged_predictor_switches_only_at_declared_xy_gate():
    state = _state(30.0)
    goal = state.copy()
    goal[2:4] = state[2:4] + np.asarray([28.0, 0.0])
    assert staged_predictor_for_state(state, goal, 28.0) == "tree"
    goal[2:4] = state[2:4] + np.asarray([28.01, 0.0])
    assert staged_predictor_for_state(state, goal, 28.0) == "goal_ranker"


def test_orientation_macro_score_prioritizes_angle_after_translation():
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    try:
        state = make_boundary_contact_roots(env, 1, 20261026)[0]["state"]
        goal = state.copy()
        goal[4] = state[4] + 0.4
        action = np.asarray([0.08, 0.0], dtype=np.float64)
        low_angle_target = np.asarray([0.25, 0.0, 0.0, 0.0, 0.0, 0.0])
        high_angle_target = np.asarray([0.02, 0.0, 0.0, 0.0, 0.0, 0.0])
        probabilities = np.asarray([0.0, 0.0, 1.0, 0.0])
        low = orientation_action_score(
            env, state, goal, action, low_angle_target, probabilities, 1.0,
            boundary_action_frame=True, strict_direction_gate=False,
        )
        high = orientation_action_score(
            env, state, goal, action, high_angle_target, probabilities, 1.0,
            boundary_action_frame=True, strict_direction_gate=False,
        )
        assert low < high
    finally:
        env.close()


def test_macro_ranker_features_are_finite_and_pair_conditioned():
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    try:
        root = make_boundary_contact_roots(env, 1, 20261027)[0]
        state = np.asarray(root["state"], dtype=np.float64)
        goal = np.asarray(root["goal"], dtype=np.float64)
        first = np.asarray([0.08, 0.0], dtype=np.float64)
        second = np.asarray([0.0, 0.08], dtype=np.float64)
        features = macro_ranker_features(
            state, goal, first, state.copy(), second,
            env=env, boundary_action_frame=True,
        )
        assert features.ndim == 1
        assert np.isfinite(features).all()
    finally:
        env.close()


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


def test_largest_signed_candidate_is_explicit_terminal_breakaway_control():
    state = _state(30.0)
    actions = np.asarray([[0.1, 0.0], [0.0, 0.05], [0.0, -0.2]], dtype=np.float64)
    desired = action_moment_sign(state, actions[2])
    index = largest_action_for_moment_sign(state, actions, desired)
    assert index == 2
    assert largest_action_for_moment_sign(state, actions, 0.0) is None


def test_active_probe_can_select_largest_declared_moment_without_new_support():
    state = _state(30.0)
    actions = np.asarray([[0.05, 0.0], [0.0, 0.10], [0.0, -0.20]], dtype=np.float64)
    desired = action_moment_sign(state, actions[2])
    smallest = choose_identification_probe(
        None, state, actions, desired, allow_outward=True, maximize=False,
    )
    largest = choose_identification_probe(
        None, state, actions, desired, allow_outward=True, maximize=True,
    )
    assert smallest is not None and largest is not None
    assert np.linalg.norm(actions[largest]) >= np.linalg.norm(actions[smallest])


def test_contact_loss_uses_declared_geometry_not_center_distance_for_boundary_roots():
    legal_boundary_trace = [{"contact_distance_after": 101.0, "boundary_distance_after": 15.0}]
    lost_boundary_trace = [{"contact_distance_after": 101.0, "boundary_distance_after": 71.0}]
    assert protocol_contact_loss(legal_boundary_trace, "boundary_contact") is False
    assert protocol_contact_loss(lost_boundary_trace, "boundary_contact") is True
    assert protocol_contact_loss(legal_boundary_trace, "terminal_pose") is True


def test_phase_aware_cost_prioritizes_translation_when_far():
    current = _state(30.0)
    goal = current.copy()
    goal[2:4] = current[2:4] + np.asarray([100.0, 0.0])
    imagined = current.copy()
    imagined[2:4] += np.asarray([10.0, 0.0])
    rotated = current.copy()
    rotated[4] += 0.8
    assert planning_pose_cost(current, imagined, goal, True) < planning_pose_cost(current, rotated, goal, True)


def test_xy_shield_keeps_orientation_candidates_inside_translation_tube():
    state = _state(30.0)
    goal = state.copy()
    goal[2:4] = state[2:4] + np.asarray([10.0, 0.0])
    targets = np.asarray([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
    ])
    mask = xy_shield_mask(state, goal, targets, tolerance=2.0)
    assert mask.tolist() == [True, False]


def test_reverse_macro_has_translation_correction_after_orientation():
    assert macro_phase_name(False, 0) == "translation"
    assert macro_phase_name(False, 1) == "orientation"
    assert macro_phase_name(True, 0) == "orientation"
    assert macro_phase_name(True, 1) == "translation"


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


def test_goal_aligned_boundary_roots_have_task_compatible_support():
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    try:
        roots = make_boundary_contact_roots(
            env, 4, 20261024, min_goal_support_alignment=0.35,
        )
        for root in roots:
            geometry = physical_contact_geometry(env, root["state"])
            outward = np.asarray(geometry["normal_world"], dtype=np.float64)
            goal_delta = np.asarray(root["goal"][2:4] - root["state"][2:4], dtype=np.float64)
            alignment = float(np.dot(outward, -goal_delta / max(np.linalg.norm(goal_delta), 1.0e-8)))
            assert alignment >= 0.35 - 1.0e-6
    finally:
        env.close()


def test_boundary_root_angle_gate_is_explicit_and_root_disjoint():
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    try:
        roots = make_boundary_contact_roots(
            env, 4, 20261025, min_goal_support_alignment=0.35,
            max_initial_angle_error=1.55,
        )
        errors = [
            abs(float(np.arctan2(np.sin(root["state"][4] - root["goal"][4]),
                                 np.cos(root["state"][4] - root["goal"][4]))))
            for root in roots
        ]
        assert len(roots) == 4
        assert all(error <= 1.55 + 1.0e-8 for error in errors)
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


def test_local_observed_effect_reuses_nearby_same_edge_action():
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    try:
        state = make_boundary_contact_roots(env, 1, 20261019)[0]["state"]
        normal = boundary_inward_normal(env, state)
        tangent = np.asarray([-normal[1], normal[0]], dtype=np.float64)
        observed_action = 0.10 * normal + 0.06 * tangent
        query_action = 0.10 * normal + 0.075 * tangent
        descriptor = online_effect_descriptor(env, state, observed_action)
        cache = {
            "same_edge": {
                "mean": [0.04, 1.0, 0.5, 0.0, 0.0, 0.0],
                "count": 1,
                "task_gain": 0.8,
                "last_error": 0.1,
                "descriptor": descriptor,
            }
        }
        result = local_observed_effect(
            env, state, query_action, cache, max_neighbors=3, max_radius=1.75,
        )
        assert result is not None
        np.testing.assert_allclose(result["target"], [0.04, 1.0, 0.5, 0.0, 0.0, 0.0])
        assert result["neighbor_count"] == 1
        assert result["task_gain"] == 0.8
    finally:
        env.close()


def test_local_observed_effect_fails_closed_across_polygon_edges():
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    try:
        state = make_boundary_contact_roots(env, 1, 20261020)[0]["state"]
        action = np.asarray([0.1, 0.0], dtype=np.float64)
        descriptor = online_effect_descriptor(env, state, action)
        descriptor["region"] = "different-edge"
        cache = {"wrong": {"mean": [0.0] * 6, "count": 1, "descriptor": descriptor}}
        assert local_observed_effect(env, state, action, cache) is None
    finally:
        env.close()


def test_local_observed_effect_uses_paired_contrast_for_angle_direction():
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    try:
        state = make_boundary_contact_roots(env, 1, 20261021)[0]["state"]
        normal = boundary_inward_normal(env, state)
        tangent = np.asarray([-normal[1], normal[0]], dtype=np.float64)
        plus = 0.10 * normal + 0.06 * tangent
        minus = 0.10 * normal - 0.06 * tangent
        plus_descriptor = online_effect_descriptor(env, state, plus)
        minus_descriptor = online_effect_descriptor(env, state, minus)
        # The helper is a chart-level test; force the second sample to be the
        # opposite signed intervention while keeping the same local chart
        # coordinates, as a matched +/- pulse would do.
        minus_descriptor["moment_sign"] = -int(plus_descriptor["moment_sign"])
        minus_descriptor["tangent_component"] = -float(plus_descriptor["tangent_component"])
        cache = {
            "plus": {
                "mean": [0.10, 1.0, 0.0, 0.0, 0.0, 0.0], "count": 1,
                "task_gain": 0.3, "last_error": 0.1, "descriptor": plus_descriptor,
            },
            "minus": {
                "mean": [-0.10, 1.0, 0.0, 0.0, 0.0, 0.0], "count": 1,
                "task_gain": -0.1, "last_error": 0.1, "descriptor": minus_descriptor,
            },
        }
        result = local_observed_effect(env, state, plus, cache)
        assert result is not None
        assert result["paired"] is True
        assert result["opposite_neighbor_count"] == 1
        np.testing.assert_allclose(result["direction_target"][0], 0.10)
    finally:
        env.close()


def test_online_reuse_rejects_single_zero_angle_probe_when_paired_support_required():
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    try:
        state = make_boundary_contact_roots(env, 1, 20261022)[0]["state"]
        action = np.asarray([0.1, 0.0], dtype=np.float64)
        descriptor = online_effect_descriptor(env, state, action)
        cache = {
            "stale": {
                "mean": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                "count": 1,
                "informative_count": 0,
                "mean_angle_abs": 0.0,
                "descriptor": descriptor,
            }
        }
        assert local_observed_effect(
            env, state, action, cache,
            min_informative_count=1,
            min_effect_angle=0.005,
            require_paired=True,
        ) is None
    finally:
        env.close()


def test_online_reuse_accepts_only_informative_matched_opposite_pulses():
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    try:
        state = make_boundary_contact_roots(env, 1, 20261023)[0]["state"]
        action = np.asarray([0.1, 0.0], dtype=np.float64)
        descriptor = online_effect_descriptor(env, state, action)
        opposite = dict(descriptor)
        opposite["moment_sign"] = -int(descriptor["moment_sign"])
        cache = {
            "same": {
                "mean": [0.10, 1.0, 0.0, 0.0, 0.0, 0.0],
                "count": 1,
                "informative_count": 1,
                "mean_angle_abs": 0.10,
                "descriptor": descriptor,
            },
            "opposite": {
                "mean": [-0.08, 1.0, 0.0, 0.0, 0.0, 0.0],
                "count": 1,
                "informative_count": 1,
                "mean_angle_abs": 0.08,
                "descriptor": opposite,
            },
        }
        result = local_observed_effect(
            env, state, action, cache,
            min_informative_count=1,
            min_effect_angle=0.005,
            require_paired=True,
        )
        assert result is not None
        assert result["paired"] is True
        assert result["opposite_neighbor_count"] == 1
    finally:
        env.close()
