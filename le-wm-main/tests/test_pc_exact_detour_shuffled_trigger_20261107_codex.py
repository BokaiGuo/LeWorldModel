import numpy as np
import pytest

from experiments import pc_exact_detour_shuffled_trigger_20261107_codex as shuffled


def test_identity_schedule_reconstructs_triggered():
    start, goal = [0., 0., 0.], [1., 1.]
    offsets = np.linspace(-.5, .5, 8)
    expected, states, schedule, _ = shuffled.trial.generate(start, goal, offsets)
    actual, actual_states = shuffled.generate(start, goal, offsets, schedule)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(actual_states, states, atol=1e-12, rtol=0)


def test_shuffle_preserves_classes_and_all_schedules():
    n = 384
    order = shuffled.permutation(n, 9)
    np.testing.assert_array_equal(np.sort(order), np.arange(n))
    np.testing.assert_array_equal(order % 4, np.arange(n) % 4)
    np.testing.assert_array_equal(order, shuffled.permutation(n, 9))
    assert np.any(order != np.arange(n))


def test_schedule_validation_and_replay():
    start, goal, offsets = [0., 0., 0.], [1., 1.], [0., .2, -.2, .3]
    schedule = np.zeros((4, 15), dtype=bool)
    schedule[0, 4] = True
    schedule[1, [2, 7]] = True
    actions, trace = shuffled.generate(start, goal, offsets, schedule)
    assert np.max(np.abs(actions)) <= 1
    replay, _ = shuffled.trial.base.rollout_n(start, actions)
    np.testing.assert_allclose(trace, replay, atol=1e-12, rtol=0)
    with pytest.raises(ValueError):
        shuffled.generate(start, goal, offsets, schedule.astype(int))
