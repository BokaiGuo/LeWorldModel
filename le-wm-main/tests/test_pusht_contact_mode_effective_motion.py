import numpy as np

from experiments.audit_pusht_contact_mode_effective_motion import (
    candidate_utility_cost,
    contact_mode_features,
    fit_direct_utility_ranker,
    fit_stratified_utility_rankers,
    goal_for_state,
    row_from_transition,
    select_direct_utility,
    select_stratified_utility,
    select_balanced_states,
)


def test_contact_mode_features_are_current_state_only_and_finite():
    state = np.array([100.0, 120.0, 145.0, 160.0, 0.4, 12.0, -8.0])
    action = np.array([0.35, -0.25])
    previous = np.array([-0.1, 0.2])
    goal = state.copy()
    goal[2:4] += np.array([30.0, -20.0])
    features = contact_mode_features(state, action, previous, goal)
    assert features.shape == (31,)
    assert np.isfinite(features).all()


def test_goal_generation_is_bounded_and_reproducible():
    state = np.array([100.0, 120.0, 145.0, 160.0, 0.4, 12.0, -8.0])
    first = goal_for_state(state, 3, np.random.default_rng(7))
    second = goal_for_state(state, 3, np.random.default_rng(7))
    np.testing.assert_allclose(first, second)
    assert (first[2:4] >= 20.0).all() and (first[2:4] <= 492.0).all()


def test_feasibility_label_requires_motion_and_contact_for_contact_source():
    state = np.array([100.0, 100.0, 130.0, 100.0, 0.0, 0.0, 0.0])
    goal = state.copy()
    goal[2:4] += np.array([20.0, 0.0])
    stationary = state.copy()
    stationary[0] += 1.0
    row = row_from_transition(state, np.array([0.2, 0.0]), np.zeros(2), goal, stationary)
    assert row["useful_motion"] == 0
    assert row["feasible"] == 0


def test_state_sampling_respects_requested_mode_subset():
    states = np.arange(18, dtype=np.float64).reshape(6, 3)
    pool = {"states": states, "modes": np.asarray(["free", "contact", "approach", "contact", "free", "approach"])}
    selected = select_balanced_states(pool, 2, 11, ("contact",))
    assert len(selected) == 2
    np.testing.assert_allclose(
        np.sort(np.asarray(selected), axis=0), np.sort(states[[1, 3]], axis=0)
    )


def test_direct_utility_ranker_uses_current_candidate_features_and_is_finite():
    rng = np.random.default_rng(13)
    rows = []
    for index in range(12):
        rows.append({
            "features": rng.normal(size=31),
            "phase": "closure" if index % 2 else "translation",
            "mode": "contact" if index % 3 == 0 else "approach",
            "xy_progress": float(rng.normal()),
            "angle_progress": float(rng.normal()),
            "useful_motion": int(index % 2 == 0),
            "preserves_contact": int(index % 3 != 1),
        })
    model = fit_direct_utility_ranker(rows, trees=8, seed=4)
    selected, info = select_direct_utility(rows, model)
    assert 0 <= selected < len(rows)
    assert info["predicted_utility_cost_finite"] is True
    assert len(info["predicted_order"]) == len(rows)
    assert np.isfinite([candidate_utility_cost(row) for row in rows]).all()


def test_stratified_utility_ranker_routes_by_mode_and_phase():
    rng = np.random.default_rng(17)
    rows = []
    for mode in ("free", "approach", "contact"):
        for phase in ("translation", "closure"):
            for _ in range(8):
                rows.append({
                    "features": rng.normal(size=31),
                    "phase": phase,
                    "mode": mode,
                    "xy_progress": float(rng.normal()),
                    "angle_progress": float(rng.normal()),
                    "useful_motion": 1,
                    "preserves_contact": 1,
                })
    models = fit_stratified_utility_rankers(rows, trees=4, seed=5)
    selected, info = select_stratified_utility(rows[:8], models)
    assert 0 <= selected < 8
    assert info["utility_route"] == "free/translation"
