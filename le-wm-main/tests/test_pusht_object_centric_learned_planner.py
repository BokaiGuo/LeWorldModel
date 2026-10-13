import numpy as np

from experiments.run_pusht_object_centric_learned_planner import (
    block_pose_cost,
    candidate_bank,
    feedback_contact_action,
    fit_contact_gate,
    local_effect_action_bank,
    local_effect_score,
    planner_feature_vector,
    staged_contact_action,
    torque_contact_frame_features,
    torque_action_bank,
    update_translation_closure_lock,
    choose_verified_torque_effect_action,
    choose_verified_contact_sequence_action,
    choose_verified_cem_action,
    choose_verified_path_cem_action,
    select_torque_effect_model,
    choose_direct_torque_effect_action,
    choose_goal_conditioned_feasibility_action,
    choose_goal_conditioned_phase_score_action,
    choose_goal_conditioned_phase_policy_action,
    goal_conditioned_ranker_features,
    strict_task_success,
)


def test_candidate_bank_has_shared_finite_phase_and_random_candidates():
    state = np.array([100.0, 120.0, 145.0, 160.0, 0.4, 12.0, -8.0])
    goal = np.array([50.0, 60.0, 280.0, 320.0, 1.2, 0.0, 0.0])
    bank = candidate_bank(np.random.default_rng(5), state, goal, 32, 6, 0.65)
    assert bank.shape == (32, 6, 2)
    assert np.isfinite(bank).all()
    assert np.allclose(bank[0], 0.0)
    assert np.any(np.linalg.norm(bank[1:], axis=-1) > 0.0)


def test_block_pose_cost_uses_block_and_orientation_not_agent_position():
    goal = np.array([10.0, 20.0, 100.0, 100.0, 0.0, 0.0, 0.0])
    first = np.array([0.0, 0.0, 110.0, 100.0, 0.1, 0.0, 0.0])
    moved_agent = first.copy()
    moved_agent[:2] = [450.0, 450.0]
    assert np.isclose(block_pose_cost(first, goal), block_pose_cost(moved_agent, goal))


def test_staged_contact_action_acquires_before_learned_push_and_returns_after_closure():
    goal = np.array([80.0, 90.0, 400.0, 300.0, 0.0, 0.0, 0.0])
    far = np.array([420.0, 100.0, 180.0, 220.0, 0.0, 0.0, 0.0])
    action, phase = staged_contact_action(far, goal)
    assert phase == "acquire"
    assert np.isfinite(action).all()
    assert np.all(np.abs(action) <= 1.0)

    contact = far.copy()
    contact[:2] = contact[2:4] + np.array([-30.0, 0.0])
    assert staged_contact_action(contact, goal) is None

    settled_contact = far.copy()
    settled_contact[:2] = settled_contact[2:4] + np.array([-55.0, 0.0])
    assert staged_contact_action(settled_contact, goal) is None

    closed = goal.copy()
    closed[:2] = goal[2:4]
    action, phase = staged_contact_action(closed, goal)
    assert phase == "return"
    assert np.isfinite(action).all()
    assert np.all(np.abs(action) <= 1.0)


def test_feedback_contact_action_is_bounded_and_sweeps_directions():
    goal = np.array([80.0, 90.0, 400.0, 300.0, 0.0, 0.0, 0.0])
    state = np.array([245.0, 300.0, 300.0, 300.0, 0.0, 0.0, 0.0])
    first, first_phase, next_index = feedback_contact_action(state, goal, 0)
    second, second_phase, final_index = feedback_contact_action(state, goal, next_index)
    assert first_phase.startswith("feedback_")
    assert second_phase.startswith("feedback_")
    assert np.isfinite(first).all() and np.isfinite(second).all()
    assert np.all(np.abs(first) <= 1.0) and np.all(np.abs(second) <= 1.0)
    assert not np.allclose(first, second)
    assert final_index == next_index + 1


def test_local_effect_phase_uses_finite_force_fan_and_hard_translation_tube():
    goal = np.array([80.0, 90.0, 400.0, 300.0, 0.0, 0.0, 0.0])
    state = np.array([245.0, 300.0, 300.0, 300.0, 0.8, 0.0, 0.0])
    bank = local_effect_action_bank(state, goal, directions=12, scale=0.55)
    assert bank.shape == (13, 2)
    assert np.isfinite(bank).all()
    assert np.all(np.abs(bank) <= 1.0)

    safe = state.copy()
    safe[2:4] = goal[2:4] + np.array([5.0, 0.0])
    unsafe = safe.copy()
    unsafe[2:4] = goal[2:4] + np.array([46.0, 0.0])
    safe_value, safe_metrics = local_effect_score(state, safe, goal)
    unsafe_value, unsafe_metrics = local_effect_score(state, unsafe, goal)
    assert unsafe_metrics["leaves_translation_tube"]
    assert unsafe_value > safe_value
    assert not safe_metrics["leaves_translation_tube"]

    monotone_value, monotone_metrics = local_effect_score(
        state, safe, goal, xy_margin=1.0
    )
    assert monotone_metrics["translation_limit"] <= monotone_metrics["before_xy"] + 1.0 + 1e-8
    assert np.isfinite(monotone_value)


def test_contact_gate_learns_block_motion_label_without_future_state_access():
    states = []
    actions = []
    previous = []
    next_states = []
    for index in range(64):
        state = np.array([100.0, 100.0, 180.0, 180.0, 0.0, 0.0, 0.0])
        action = np.array([0.8, 0.0]) if index % 2 else np.array([0.0, 0.8])
        target = state.copy()
        if index % 2:
            target[2] += 5.0
        states.append(state)
        actions.append(action)
        previous.append(np.zeros(2))
        next_states.append(target)
    data = {
        "states": np.asarray(states),
        "actions": np.asarray(actions),
        "previous_actions": np.asarray(previous),
        "next_states": np.asarray(next_states),
    }
    gate = fit_contact_gate(data, "object_centric", trees=32, seed=7)
    assert len(gate.classes_) == 2
    assert np.isfinite(gate.predict_proba(
        np.asarray([
            planner_feature_vector(states[0], actions[0], previous[0], "object_centric")
        ])
    )).all()


def test_torque_action_bank_is_complete_and_bounded():
    bank = torque_action_bank(directions=12, magnitudes=(0.35, 0.55))
    assert bank.shape == (25, 2)
    assert np.isfinite(bank).all()
    assert np.all(np.linalg.norm(bank, axis=1) <= 0.55 + 1e-6)


def test_torque_contact_frame_features_are_finite_and_rotation_local():
    state = np.array([100.0, 100.0, 150.0, 100.0, 0.4, 2.0, -1.0])
    features = torque_contact_frame_features(state, np.array([0.5, -0.2]), np.zeros(2))
    assert features.shape == (12,)
    assert np.isfinite(features).all()


def test_goal_conditioned_features_encode_current_contact_chart():
    state = np.array([100.0, 100.0, 145.0, 100.0, 0.4, 2.0, -1.0])
    goal = state.copy()
    features = goal_conditioned_ranker_features(
        state, np.array([0.5, -0.2]), np.zeros(2), goal, "contact_frame"
    )
    assert features.shape == (21,)
    assert np.isfinite(features).all()
    # The final two entries are the current-mode indicators.
    assert np.allclose(features[-2:], [1.0, 0.0])


def test_torque_effect_specialist_routes_only_on_current_contact_mode():
    fallback = object()
    approach = object()
    contact = object()
    model = {
        "torque_effect": fallback,
        "torque_effect_specialists": {
            "fallback": fallback,
            "approach": approach,
            "contact": contact,
            "free": fallback,
        },
    }
    contact_state = np.array([100.0, 100.0, 130.0, 100.0, 0.0, 0.0, 0.0])
    approach_state = np.array([100.0, 100.0, 170.0, 100.0, 0.0, 0.0, 0.0])
    assert select_torque_effect_model(model, contact_state) is contact
    assert select_torque_effect_model(model, approach_state) is approach


def test_direct_knn_calibration_is_candidate_gated_and_finite():
    class ZeroModel:
        def predict(self, features):
            return np.zeros((len(features), 3), dtype=np.float64)

    state = np.array([100.0, 100.0, 140.0, 100.0, 0.0, 0.0, 0.0])
    goal = state.copy()
    history = [
        {
            "mode": "contact",
            "action_local": np.array([0.15, 0.0]),
            "effect_local": np.array([0.0, 0.0, 0.0]),
        }
        for _ in range(4)
    ]
    action, info = choose_direct_torque_effect_action(
        ZeroModel(),
        state,
        goal,
        directions=8,
        magnitudes=(0.15,),
        online_local_history=history,
        online_local_strategy="knn",
        online_local_radius=0.01,
        online_local_neighbors=3,
    )
    assert action.shape == (2,)
    assert np.isfinite(action).all()
    assert 0 <= info["online_local_supported_candidates"] <= 9
    assert info["online_local_neighbor_count"] == 3


def test_direct_state_knn_calibration_uses_observed_state_chart():
    class ZeroModel:
        def predict(self, features):
            return np.zeros((len(features), 3), dtype=np.float64)

    state = np.array([100.0, 100.0, 140.0, 100.0, 0.0, 0.0, 0.0])
    goal = state.copy()
    history = [
        {
            "mode": "contact",
            "action_local": np.array([0.15, 0.0]),
            "effect_local": np.array([0.0, 0.0, 0.0]),
            "state_before": state.copy(),
            "action_world": np.array([0.15, 0.0]),
            "previous_action": np.zeros(2),
        }
        for _ in range(4)
    ]
    action, info = choose_direct_torque_effect_action(
        ZeroModel(),
        state,
        goal,
        directions=8,
        magnitudes=(0.15,),
        online_local_history=history,
        online_local_strategy="state_knn",
        online_local_radius=3.0,
        online_local_neighbors=3,
    )
    assert action.shape == (2,)
    assert np.isfinite(action).all()
    assert info["online_local_fit_used"]
    assert info["online_local_strategy"] == "state_knn"
    assert info["online_local_supported_candidates"] == 9


def test_phase_feasibility_uses_translation_and_closure_heads_separately():
    class FakeHead:
        def __init__(self, preferred):
            self.classes_ = np.asarray([0, 1])
            self.preferred = int(preferred)

        def predict_proba(self, features):
            scores = np.full(len(features), 0.01, dtype=np.float64)
            scores[self.preferred] = 0.99
            return np.column_stack([1.0 - scores, scores])

    state = np.array([100.0, 100.0, 140.0, 100.0, 0.0, 0.0, 0.0])
    goal = state.copy()
    goal[2:4] = [180.0, 100.0]
    actions = torque_action_bank(8, magnitudes=(0.15,))
    models = {
        "translation": FakeHead(1),
        "closure": FakeHead(2),
    }
    translation_action, translation_info = choose_goal_conditioned_feasibility_action(
        models, state, goal, directions=8, magnitudes=(0.15,), translation_first=True
    )
    closure_action, closure_info = choose_goal_conditioned_feasibility_action(
        models, state, goal, directions=8, magnitudes=(0.15,), translation_first=False
    )
    assert translation_info["phase_specific_feasibility_head"]
    assert closure_info["phase_specific_feasibility_head"]
    assert translation_info["feasibility_phase"] == "translation"
    assert closure_info["feasibility_phase"] == "closure"
    assert np.array_equal(translation_action, actions[1].astype(np.float32))
    assert np.array_equal(closure_action, actions[2].astype(np.float32))


def test_phase_score_uses_matching_head_and_maximizes_utility():
    class FakeHead:
        def __init__(self, preferred):
            self.preferred = int(preferred)

        def predict(self, features):
            values = np.full(len(features), -1.0, dtype=np.float64)
            values[self.preferred] = 2.0
            return values

    state = np.array([100.0, 100.0, 140.0, 100.0, 0.0, 0.0, 0.0])
    goal = state.copy()
    goal[2:4] = [180.0, 100.0]
    models = {"translation": FakeHead(1), "closure": FakeHead(2)}
    translation_action, translation_info = choose_goal_conditioned_phase_score_action(
        models, state, goal, directions=8, magnitudes=(0.15,), translation_first=True
    )
    closure_action, closure_info = choose_goal_conditioned_phase_score_action(
        models, state, goal, directions=8, magnitudes=(0.15,), translation_first=False
    )
    assert translation_info["phase_specific_score_head"]
    assert closure_info["phase_specific_score_head"]
    assert np.array_equal(
        translation_action,
        torque_action_bank(8, magnitudes=(0.15,))[1].astype(np.float32),
    )
    assert np.array_equal(
        closure_action,
        torque_action_bank(8, magnitudes=(0.15,))[2].astype(np.float32),
    )


def test_phase_policy_emits_an_action_from_the_fixed_bank():
    class FakeHead:
        classes_ = np.asarray([1, 3])

        def predict_proba(self, features):
            return np.asarray([[0.2, 0.8]])

    models = {"translation": FakeHead(), "closure": FakeHead()}
    state = np.array([100.0, 100.0, 140.0, 100.0, 0.0, 0.0, 0.0])
    goal = state.copy()
    action, info = choose_goal_conditioned_phase_policy_action(
        models, state, goal, directions=8, magnitudes=(0.15,), translation_first=True
    )
    assert info["phase_specific_policy_head"]
    assert np.array_equal(action, torque_action_bank(8, magnitudes=(0.15,))[3].astype(np.float32))


def test_translation_closure_lock_has_hysteresis():
    goal = np.array([0.0, 0.0, 200.0, 200.0, 0.0, 0.0, 0.0])
    state = goal.copy()
    state[2:4] += np.array([19.5, 0.0])
    assert update_translation_closure_lock(False, state, goal)

    small_excursion = goal.copy()
    small_excursion[2:4] += np.array([23.0, 0.0])
    assert update_translation_closure_lock(True, small_excursion, goal)

    large_excursion = goal.copy()
    large_excursion[2:4] += np.array([27.0, 0.0])
    assert not update_translation_closure_lock(True, large_excursion, goal)


def test_strict_task_success_requires_orientation_and_joint_position():
    goal = np.array([100.0, 100.0, 200.0, 200.0, 0.0, 0.0, 0.0])
    wrong_angle = goal.copy()
    wrong_angle[4] = 1.0
    assert not strict_task_success(wrong_angle, goal)
    near_but_outside_joint_margin = goal.copy()
    near_but_outside_joint_margin[:2] = [104.0, 100.0]
    near_but_outside_joint_margin[2:4] = [220.5, 200.0]
    assert not strict_task_success(near_but_outside_joint_margin, goal)
    assert strict_task_success(goal, goal)


def test_verified_probe_restores_commit_state():
    class ZeroModel:
        def predict(self, features):
            return np.zeros((len(features), 3), dtype=np.float64)

    class SpyEnv:
        def __init__(self):
            self.state = None
            self.last_reset_state = None

        def reset(self, seed=None, options=None):
            self.state = np.asarray(options["state"], dtype=np.float64).copy()
            self.last_reset_state = self.state.copy()
            return {"state": self.state.copy()}, {"goal_state": options["goal_state"]}

        def step(self, action):
            self.state = self.state.copy()
            self.state[:2] += np.asarray(action, dtype=np.float64)
            return {"state": self.state.copy()}, 0.0, False, False, {}

    env = SpyEnv()
    state = np.array([100.0, 100.0, 180.0, 180.0, 0.0, 0.0, 0.0])
    goal = np.array([100.0, 100.0, 200.0, 200.0, 0.0, 0.0, 0.0])
    choose_verified_torque_effect_action(
        env, ZeroModel(), state, goal, np.zeros(2), directions=8, topk=3
    )
    assert np.allclose(env.last_reset_state, state)
def test_sequence_and_cem_fallbacks_are_bounded_and_restore_state():
    class SpyEnv:
        def __init__(self):
            self.state = None
            self.last_reset_state = None

        def reset(self, seed=None, options=None):
            self.state = np.asarray(options["state"], dtype=np.float64).copy()
            self.last_reset_state = self.state.copy()
            return {"state": self.state.copy()}, {"goal_state": options["goal_state"]}

        def step(self, action):
            self.state = self.state.copy()
            self.state[:2] += np.asarray(action, dtype=np.float64)
            return {"state": self.state.copy()}, 0.0, False, False, {}

    env = SpyEnv()
    state = np.array([100.0, 100.0, 180.0, 180.0, 0.0, 0.0, 0.0])
    goal = np.array([100.0, 100.0, 200.0, 200.0, 0.0, 0.0, 0.0])
    bank = np.asarray([[0.0, 0.0], [0.4, 0.0], [-0.4, 0.0]], dtype=np.float32)
    action, info = choose_verified_contact_sequence_action(
        env, state, goal, bank, horizon=2, translation_first=True
    )
    assert action.shape == (2,)
    assert info["sequence_candidate_effects_finite"]
    assert np.allclose(env.last_reset_state, state)
    action, info = choose_verified_cem_action(
        env, state, goal, bank, translation_first=True, hard_xy_limit=None,
        population=16, iterations=1, seed=3,
    )
    assert action.shape == (2,)
    assert info["cem_candidate_effects_finite"]
    assert np.allclose(env.last_reset_state, state)
    action, info = choose_verified_path_cem_action(
        env, state, goal, bank, horizon=2, population=16, iterations=1, seed=4
    )
    assert action.shape == (2,)
    assert info["path_cem_candidate_effects_finite"]
    assert np.allclose(env.last_reset_state, state)


def test_orientation_path_cem_is_explicit_and_restores_state():
    class AngleSpyEnv:
        def __init__(self):
            self.state = None
            self.last_reset_state = None

        def reset(self, seed=None, options=None):
            self.state = np.asarray(options["state"], dtype=np.float64).copy()
            self.last_reset_state = self.state.copy()
            return {"state": self.state.copy()}, {"goal_state": options["goal_state"]}

        def step(self, action):
            self.state = self.state.copy()
            self.state[4] += 0.08 * float(np.asarray(action)[0])
            self.state[2:4] += 0.05 * np.asarray(action, dtype=np.float64)
            return {"state": self.state.copy()}, 0.0, False, False, {}

    env = AngleSpyEnv()
    state = np.array([100.0, 100.0, 180.0, 180.0, -0.8, 0.0, 0.0])
    goal = np.array([100.0, 100.0, 180.0, 180.0, 0.8, 0.0, 0.0])
    bank = np.asarray([[0.0, 0.0], [0.7, 0.0], [-0.7, 0.0]], dtype=np.float32)
    action, info = choose_verified_path_cem_action(
        env,
        state,
        goal,
        bank,
        horizon=3,
        population=16,
        iterations=1,
        path_xy_limit=2.0,
        objective="orientation",
        seed=9,
    )
    assert action.shape == (2,)
    assert info["path_cem_objective"] == "orientation"
    assert info["path_cem_candidate_effects_finite"]
    assert np.isfinite(info["path_cem_final_angle_error"])
    assert np.allclose(env.last_reset_state, state)
