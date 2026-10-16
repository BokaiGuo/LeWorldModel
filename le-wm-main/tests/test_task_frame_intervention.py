import numpy as np

from experiments.decision_benchmark.run_task_frame_intervention import (
    _from_task_frame,
    _to_task_frame,
)


def test_goal_relative_task_frame_round_trips_to_raw_xy():
    source = np.asarray([[0.0, 0.0], [1.0, 1.0]])
    goals = np.asarray([[1.0, 0.0], [1.0, 2.0]])
    effect = np.asarray([[0.5, 0.2], [-0.3, 0.4]])
    local = _to_task_frame(effect, source, goals, "goal_relative")
    recovered = _from_task_frame(local, source, goals, "goal_relative")
    np.testing.assert_allclose(recovered, effect, atol=1.0e-8)


def test_raw_task_frame_is_identity():
    effect = np.asarray([[0.5, -0.2]])
    source = np.asarray([[0.0, 0.0]])
    goals = np.asarray([[1.0, 1.0]])
    np.testing.assert_allclose(_to_task_frame(effect, source, goals, "raw_xy"), effect)
    np.testing.assert_allclose(_from_task_frame(effect, source, goals, "raw_xy"), effect)
