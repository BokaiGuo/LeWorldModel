"""Opt-in PushT planning environment with the periodic success evaluator.

Mirrors the single-task PushT branch of upstream ``envs.init.make_env``.
It deliberately does not monkey-patch the upstream registry.
"""
import gym

from evals.simu_env_planning.envs.wrappers.pixels import PixelWrapper
from evals.simu_env_planning.envs.wrappers.tensor import TensorWrapper
from experiments.pc_pusht_periodic_eval_v1 import make_periodic_env


class PreparedTensorWrapper(TensorWrapper):
    """Reset the time limit after explicit state preparation, as PixelWrapper does."""

    def prepare(self, *args, **kwargs):
        observation, info = self.env.prepare(*args, **kwargs)
        self.env.set_elapsed_steps(0)
        return observation, info


def make_periodic_planning_env(cfg):
    spec = cfg.task_specification
    if not spec.task.startswith("pusht-") or spec.multitask:
        raise ValueError("Only single-task PushT is supported")
    gym.logger.set_level(40)
    if spec.goal_source in ("dset", "random_action"):
        spec.max_episode_steps = cfg.frameskip * spec.goal_H
        spec.goal_max_episode_steps = cfg.frameskip * spec.goal_H
    elif spec.goal_source == "random_state":
        spec.max_episode_steps = cfg.frameskip * spec.goal_H
    elif spec.get("max_episode_steps", None) is None:
        spec.max_episode_steps = 100
    env = PreparedTensorWrapper(make_periodic_env(cfg))
    obs = spec.get("obs", "state")
    if obs in ("rgb", "rgb_state"):
        env = PixelWrapper(cfg, env)
    try:
        cfg.obs_shape = {key: value.shape for key, value in env.observation_space.spaces.items()}
    except AttributeError:
        cfg.obs_shape = {obs: env.observation_space.shape}
    if obs == "rgb_state":
        cfg.obs_shape = {"state": [4], "rgb": cfg.obs_shape["rgb_state"]}
    cfg.action_dim = env.action_space.shape[0]
    cfg.episode_length = env.max_episode_steps
    cfg.meta.seed_steps = max(1000, 5 * cfg.episode_length)
    return env
