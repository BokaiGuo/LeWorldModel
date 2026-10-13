import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401

from experiments.run_causal_visual_pusht_world_model import terminal_pose_cases


def test_terminal_pose_cases_skip_empty_support_candidate_sets():
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    try:
        cases = terminal_pose_cases(
            env, roots=4, seed=10181, position_jitter=16.0,
            support_gap_max=12.0, support_angle_samples=32,
        )
    finally:
        env.close()
    assert len(cases) == 4
    assert all(np.isfinite(case[0]).all() for case in cases)
