from types import SimpleNamespace

import numpy as np
import pytest

from evals.simu_env_planning.envs.pusht_gym_wrap import PushTWrapper
from experiments.pc_pusht_periodic_eval_v1 import PeriodicPushTWrapper, make_periodic_env


def test_actual_upstream_counterexample_and_periodic_fix():
    goal = np.zeros(7)
    state = np.zeros(7)
    state[4] = 2 * np.pi + 1
    assert PushTWrapper.eval_state(None, goal, state)["success"]
    assert not PeriodicPushTWrapper.eval_state(None, goal, state)["success"]


@pytest.mark.parametrize("delta", [0, .2, 1, np.pi, 2 * np.pi - .2])
def test_in_range_parity_and_integer_rotation_invariance(delta):
    goal = np.zeros(7)
    state = np.zeros(7)
    state[4] = delta
    expected = PushTWrapper.eval_state(None, goal, state)["success"]
    for turns in (-5, -2, 0, 2, 5):
        shifted = state.copy()
        shifted[4] += turns * 2 * np.pi
        assert PeriodicPushTWrapper.eval_state(None, goal, shifted)["success"] == expected
    assert PeriodicPushTWrapper.eval_state(None, goal, state)["state_dist"] == \
        PushTWrapper.eval_state(None, goal, state)["state_dist"]


def test_invalid_state_fails_closed():
    with pytest.raises(ValueError):
        PeriodicPushTWrapper.eval_state(None, np.zeros(7), np.full(7, np.nan))


def test_real_factory_routes_evaluator_through_time_limit():
    spec = SimpleNamespace(task="pusht-test", env=SimpleNamespace(with_velocity=True, with_target=False),
                           max_episode_steps=10)
    env = make_periodic_env(SimpleNamespace(task_specification=spec))
    try:
        goal, state = np.zeros(7), np.zeros(7)
        state[4] = 2 * np.pi + 1
        assert isinstance(env.env, PeriodicPushTWrapper)
        assert not env.eval_state(goal, state)["success"]
    finally:
        env.close()
