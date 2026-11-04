import gymnasium as gym
import numpy as np

from experiments.spectral_reachability.run_pc_ap_reacher_rgb_physical_dev import ObserveSteps


def test_observation_wrapper_preserves_snapshot_goal():
    class GoalEnv(gym.Env):
        def __init__(self):
            self._goal = np.zeros((2, 2, 3), dtype=np.uint8)

    underlying = GoalEnv()
    wrapped = ObserveSteps(underlying)
    assert wrapped._goal is underlying._goal
