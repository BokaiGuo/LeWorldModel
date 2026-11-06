import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from evals.simu_env_planning.envs.init import make_env
from experiments.pc_pusht_periodic_planning_env_v1 import make_periodic_planning_env


def config(obs):
    return OmegaConf.create({
        "frameskip": 2, "meta": {},
        "task_specification": {
            "task": "pusht-test", "multitask": False, "goal_source": "random_state",
            "goal_H": 5, "obs": obs, "env": {"with_velocity": True, "with_target": False},
            "num_frames": 1, "num_proprios": 1, "obs_concat_channels": False, "img_size": 96,
        },
    })


@pytest.mark.parametrize("obs", ["state", "rgb"])
def test_real_planning_factory_parity_and_periodic_routing(obs):
    original_cfg, periodic_cfg = config(obs), config(obs)
    original, periodic = make_env(original_cfg), make_periodic_planning_env(periodic_cfg)
    try:
        assert OmegaConf.to_container(original_cfg) == OmegaConf.to_container(periodic_cfg)
        assert original.observation_space.shape == periodic.observation_space.shape
        assert original.action_space.shape == periodic.action_space.shape
        goal, state = np.zeros(7), np.zeros(7)
        state[4] = 2 * np.pi + 1
        assert original.eval_state(goal, state)["success"]
        assert not periodic.eval_state(goal, state)["success"]
        initial = np.array([100, 120, 250, 250, 0, 0, 0], dtype=float)
        old_obs, old_info = original.prepare(123, initial)
        new_obs, new_info = periodic.prepare(123, initial)
        np.testing.assert_array_equal(np.asarray(old_obs), np.asarray(new_obs))
        np.testing.assert_array_equal(old_info["state"], new_info["state"])
        if obs == "state":
            # Upstream TensorWrapper.prepare bypasses TimeLimit.reset.
            original.set_elapsed_steps(0)
        action = torch.zeros(2)
        old_out, new_out = original.step(action), periodic.step(action)
        assert torch.equal(old_out[0], new_out[0])
        np.testing.assert_array_equal(old_out[-1]["state"], new_out[-1]["state"])
        assert old_out[1:4] == new_out[1:4]
    finally:
        original.close()
        periodic.close()


def test_other_tasks_rejected():
    cfg = config("state")
    cfg.task_specification.task = "maze-test"
    with pytest.raises(ValueError):
        make_periodic_planning_env(cfg)


def test_prepare_resets_state_mode_clock_on_initial_and_repeated_use():
    original, periodic = make_env(config("state")), make_periodic_planning_env(config("state"))
    initial = np.array([100, 120, 250, 250, 0, 0, 0], dtype=float)
    try:
        original.prepare(123, initial)
        with pytest.raises(TypeError):
            original.step(torch.zeros(2))
        periodic.prepare(123, initial)
        periodic.step(torch.zeros(2))
        assert periodic.elapsed_steps() == 1
        periodic.prepare(123, initial)
        assert periodic.elapsed_steps() == 0
        periodic.step(torch.zeros(2))
        assert periodic.elapsed_steps() == 1
    finally:
        original.close()
        periodic.close()
