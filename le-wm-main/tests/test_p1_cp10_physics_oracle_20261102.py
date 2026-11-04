import numpy as np
import pytest

from experiments.spectral_reachability import p1_cp10_physics_oracle_20261102 as oracle


def test_bank_has_fixed_candidates_and_common_random_part():
    old = np.full((5, 2), .3, np.float32)
    new = np.full((5, 2), -.2, np.float32)
    first, second = oracle.bank(old, new), oracle.bank(new, old)
    assert first.shape == (64, 5, 2)
    np.testing.assert_array_equal(first[0], old)
    np.testing.assert_array_equal(first[1], new)
    np.testing.assert_array_equal(first[2:], second[2:])


def test_physical_objective_wraps_angle_and_ignores_pusher():
    goal = np.zeros(7)
    state = np.array([100., 50., 3., 4., 2*np.pi-.1, 7., 9.])
    out = oracle.objective(state, goal)
    assert out["position"] == pytest.approx(5)
    assert out["angle"] == pytest.approx(.1)
    assert out["score"] == pytest.approx(9)


def test_oracle_declares_privileged_and_non_confirmation():
    spec = oracle.protocol()
    assert spec["uses_privileged_state"] and not spec["planner_uses_oracle_outputs"]
    assert len(spec["arms"]) == 2 and not spec["P1_admission"]
