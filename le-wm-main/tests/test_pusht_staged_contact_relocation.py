import numpy as np

from experiments.run_pusht_staged_contact_relocation import (
    block_pose_success,
    strict_success,
)


def _state(xy=(100.0, 100.0), block=(120.0, 120.0), angle=0.0):
    return np.asarray([*xy, *block, angle, 0.0, 0.0], dtype=np.float64)


def test_block_pose_success_is_separate_from_agent_return():
    goal = _state(xy=(150.0, 150.0), block=(120.0, 120.0), angle=0.0)
    solved_block_far_agent = _state(xy=(40.0, 40.0), block=(120.0, 120.0), angle=0.01)
    assert block_pose_success(solved_block_far_agent, goal)
    assert not strict_success(solved_block_far_agent, goal)


def test_strict_success_requires_both_agent_and_block_pose():
    goal = _state(xy=(150.0, 150.0), block=(120.0, 120.0), angle=0.0)
    solved = _state(xy=(150.0, 150.0), block=(120.0, 120.0), angle=0.01)
    assert block_pose_success(solved, goal)
    assert strict_success(solved, goal)


def test_pose_failures_are_finite_and_angle_wrapped():
    goal = _state(angle=np.pi - 0.01)
    near_wrap = _state(angle=-np.pi + 0.01)
    assert block_pose_success(near_wrap, goal)
