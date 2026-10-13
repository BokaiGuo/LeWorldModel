import gymnasium as gym
import numpy as np

from experiments.run_causal_visual_pusht_world_model import terminal_pose_cases
from experiments.run_rgb_local_effect_adapter import (
    contact_boundary_chart_from_images,
    resize_image,
)


def test_support_gap_terminal_roots_match_runtime_support_side():
    env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=80)
    try:
        roots = terminal_pose_cases(
            env,
            roots=2,
            seed=718181,
            position_jitter=16.0,
            support_gap_max=16.0,
            support_angle_samples=32,
        )
        for state, goal, image, *_ in roots:
            chart = contact_boundary_chart_from_images(
                resize_image(image, 64)[None],
                np.zeros((1, 2), dtype=np.float32),
            )[0]
            support_normal = -chart[2:4] / max(float(np.linalg.norm(chart[2:4])), 1.0e-6)
            goal_direction = goal[2:4] - state[2:4]
            goal_direction /= max(float(np.linalg.norm(goal_direction)), 1.0e-6)
            assert float(chart[8] * 256.0) <= 16.0 + 1.0e-6
            assert float(np.dot(support_normal, goal_direction)) >= 0.35 - 1.0e-6
    finally:
        env.close()
