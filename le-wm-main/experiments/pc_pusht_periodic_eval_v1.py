"""Versioned PushT evaluation adapter; the upstream wrapper is unchanged.

Use ``make_periodic_env`` in a newly frozen experiment instead of upstream
``make_env``.  This changes the success evaluator only, not observations,
actions, environment dynamics, planning input, or the reported state distance.
"""
import numpy as np

from evals.simu_env_planning.envs.pusht_gym_wrap import PushTWrapper
from evals.simu_env_planning.envs.pusht_env.pusht_env import PushTEnv
from evals.simu_env_planning.envs.wrappers.time_limit import TimeLimit


class PeriodicPushTWrapper(PushTWrapper):
    def eval_state(self, goal_state, cur_state):
        goal = np.asarray(goal_state)
        current = np.asarray(cur_state)
        if (goal.ndim != 1 or current.shape != goal.shape or len(goal) < 5
                or not np.isfinite(goal).all() or not np.isfinite(current).all()):
            raise ValueError("invalid PushT evaluation state")
        position = np.linalg.norm(goal[:4] - current[:4])
        delta = goal[4] - current[4]
        angle = np.abs(np.arctan2(np.sin(delta), np.cos(delta)))
        return {
            "success": bool(position < 20 and angle < np.pi / 9),
            # Preserve the upstream distance contract; it is not periodic.
            "state_dist": np.linalg.norm(goal - current),
        }


def make_periodic_env(cfg, env_cls=None):
    """Mirror upstream factory but install the explicitly versioned evaluator."""
    if not cfg.task_specification.task.startswith("pusht-"):
        raise ValueError("Unknown task:", cfg.task_specification.task)
    env = PushTEnv(
        with_velocity=cfg.task_specification.env.with_velocity,
        with_target=cfg.task_specification.env.with_target,
    )
    env = PeriodicPushTWrapper(env, cfg)
    env = TimeLimit(env, max_episode_steps=cfg.task_specification.max_episode_steps)
    env.max_episode_steps = env._max_episode_steps
    return env
