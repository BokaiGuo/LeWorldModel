import numpy as np
import pytest

from experiments import pc_exact_blind_detour_20261107_codex as blind


def test_prefix_bounds_and_replay():
    start = np.array([0.0, 0.0, 0.0])
    goal = np.array([1.0, 1.0])
    offsets = np.array([-.3, 0., .3, .6])
    actions, states = blind.generate(start, goal, offsets)
    plain, _ = blind.control.no_detour(start, goal, offsets)
    np.testing.assert_allclose(actions[:, :5], plain[:, :5], atol=1e-12, rtol=0)
    assert np.max(np.abs(actions)) <= 1.
    replay, _ = blind.trial.base.rollout_n(start, actions)
    np.testing.assert_allclose(states, replay, atol=1e-12, rtol=0)


def test_blind_trigger_ignores_collision_flag(monkeypatch):
    original = blind.trial.base.wall.physics.step_batch

    def changed_flag(*args):
        position, flag = original(*args)
        return position, np.logical_not(flag)

    start, goal, offsets = [0., 0., 0.], [1., 1.], [0., .2]
    expected = blind.generate(start, goal, offsets)
    monkeypatch.setattr(blind.trial.base.wall.physics, "step_batch", changed_flag)
    actual = blind.generate(start, goal, offsets)
    for a, b in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(a, b)


def test_invalid_input():
    with pytest.raises(ValueError):
        blind.generate([0., 0., 0.], [0., 0.], [0.])
