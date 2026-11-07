import pytest
import numpy as np
from types import SimpleNamespace

from experiments.pc_pusht_goal_render_alignment_20261107_codex import compare, set_render_pose


def test_pair_success_and_order():
    result = compare([1, 3], [True, False])
    assert result["choice"] == 0 and result["correct"]
    assert not compare([1, 3], [False, True])["correct"]
    assert compare([1, 3], [True, True])["correct"] is None


def test_pair_tolerance_and_invalid():
    assert compare([1, 1.0005], [True, False])["choice"] is None
    with pytest.raises(ValueError):
        compare([1, float("nan")], [True, False])


def test_render_pose_does_not_step_physics():
    actor = SimpleNamespace(agent=SimpleNamespace(), block=SimpleNamespace())
    set_render_pose(actor, [1, 2, 3, 4, 5, 6, 7])
    assert actor.agent.position == [1, 2]
    assert actor.block.position == [3, 4]
    assert actor.block.angle == 5
    assert actor.agent.velocity == [6, 7]
    with pytest.raises(ValueError):
        set_render_pose(actor, [1, 2])


def test_real_environment_render_pose_round_trip():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    try:
        env.reset(seed=19)
        expected = np.array([150., 200., 240., 250., 1.2, 2., -3.])
        set_render_pose(env.unwrapped, expected)
        np.testing.assert_allclose(env.unwrapped._get_obs(), expected, atol=1e-8)
    finally:
        env.close()
