import numpy as np

from experiments.run_pusht_contact_closure_learned_selector import candidate_rows


def test_candidate_rows_use_same_goal_and_finite_current_features():
    state = np.array([100.0, 100.0, 130.0, 100.0, 0.2, 0.0, 0.0])
    goal = state.copy()
    goal[2:4] += np.array([5.0, 3.0])
    actions = np.asarray([[0.0, 0.0], [0.55, 0.0], [-0.55, 0.0]])
    rows = candidate_rows(state, goal, np.zeros(2), actions)
    assert len(rows) == len(actions)
    assert all(np.array_equal(row["goal"], goal) for row in rows)
    assert all(np.isfinite(row["features"]).all() for row in rows)
