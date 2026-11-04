import numpy as np

from experiments.spectral_reachability import pc_exact_wall_kill as wall


def test_blocked_crossing_and_opening():
    state, blocked = wall.step_batch(np.array([-0.1, 0.0]), np.array([0.2, 0.0]))
    np.testing.assert_array_equal(state, [-0.1, 0.0])
    assert blocked
    state, blocked = wall.step_batch(np.array([-0.1, 1.0]), np.array([0.2, 0.0]))
    np.testing.assert_allclose(state, [0.1, 1.0])
    assert not blocked


def test_endpoint_contact_cannot_leak_through_wall():
    state, blocked = wall.step_batch(np.array([-0.2, 0.0]), np.array([0.2, 0.0]))
    np.testing.assert_array_equal(state, [-0.2, 0.0])
    assert blocked


def test_witness_is_bounded_crosses_and_reaches_goal():
    program, _ = wall.witness()
    costs, blocked = wall.rollout_batch(program[None])
    assert costs[0] < 1e-12 and blocked[0] == 0
    state = wall.START.copy()
    distances = [np.linalg.norm(state - wall.GOAL)]
    for action in program:
        state, _ = wall.step_batch(state, action)
        distances.append(np.linalg.norm(state - wall.GOAL))
    assert max(distances) > distances[0]


def test_all_planners_produce_valid_complete_programs():
    for method in (wall.greedy, wall.cem, wall.beam, wall.witness):
        program, ledger = method()
        assert program.shape == (wall.HORIZON, 2)
        assert np.isfinite(wall.rollout_batch(program[None])[0]).all()
        assert ledger["transitions"] > 0
