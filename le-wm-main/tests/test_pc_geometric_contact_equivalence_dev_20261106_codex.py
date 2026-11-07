import gymnasium as gym
import numpy as np
import pytest
import stable_worldmodel  # noqa: F401

from experiments.pc_geometric_contact_equivalence_dev_20261106_codex import geometry, pose_cost, state_for_side


@pytest.mark.parametrize("theta", [0., np.pi / 4, np.pi / 2, 3 * np.pi / 4])
def test_boundary_positions_are_nonpenetrating_and_zero_stable(theta):
    env = gym.make("swm/PushT-v1", resolution=224)
    base = np.array([80., 80., 256., 256., theta, 0., 0.])
    try:
        env.reset(seed=12, options={"state": base.copy(), "goal_state": base.copy()})
        states = [state_for_side(env.unwrapped, side) for side in (-1, 1)]
        for state in states:
            env.reset(seed=12, options={"state": state.copy(), "goal_state": base.copy()})
            g = geometry(env.unwrapped)
            assert not g["overlaps"]
            assert g["clearance"] == pytest.approx(1., abs=1e-6)
            before = env.unwrapped._get_obs()
            for _ in range(5):
                env.unwrapped.step(np.zeros(2))
            np.testing.assert_allclose(env.unwrapped._get_obs(), before, atol=1e-6, rtol=0)
    finally:
        env.close()


def test_pose_metric_wraps_angle_and_excludes_agent():
    a = np.array([0., 0., 2., 3., .1, 0., 0.])
    b = a.copy()
    b[[0, 1, 5, 6]] = 100
    b[4] += 2 * np.pi
    assert pose_cost(a, b) == pytest.approx(0., abs=1e-20)
