import numpy as np

from experiments.pc_rgb_cem_reward_decomposition_20261107_codex import decompose


def test_reward_includes_agent_velocity_and_object_state():
    snapshot = [3, 4, 6, 8, 0, 0, 12, 0, 0, 0, 0]
    goal = [0, 0, 0, 0, 0, 0, 0]
    result = decompose(snapshot, goal)
    assert result["agent_position"] == 5
    assert result["object_position"] == 12
    assert result["agent_velocity"] == 10
    assert np.isclose(result["reward"], -np.sqrt(25 + 144 + 100))
    assert result["success"]


def test_reward_uses_raw_angle_but_success_wraps():
    snapshot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 2 * np.pi - .1, 0]
    result = decompose(snapshot, [0] * 7)
    assert result["success"]
    assert result["raw_angle"] > 6
    assert np.isclose(result["wrapped_angle"], .1)
