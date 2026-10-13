import numpy as np

from experiments.run_pusht_obstacle_aware_local_effect import block_pose_success
from experiments.run_pusht_receding_block_pose_planner import (
    closure_return_trajectory_cost,
    terminal_stability_certificate,
)


def test_block_pose_success_uses_conjunctive_push_t_tolerance():
    goal = np.array([100.0, 100.0, 200.0, 200.0, 0.0, 0.0, 0.0])
    closed = goal.copy()
    outside_xy = goal.copy()
    outside_xy[2] += 20.0
    outside_angle = goal.copy()
    outside_angle[4] = np.pi / 9.0
    assert block_pose_success(closed, goal)
    assert not block_pose_success(outside_xy, goal)
    assert not block_pose_success(outside_angle, goal)


def test_closure_return_rejects_intermediate_block_pose_regression():
    goal = np.array([100.0, 100.0, 200.0, 200.0, 0.0, 0.0, 0.0])
    start = goal.copy()
    start[:2] = [150.0, 100.0]
    regressed = start.copy()
    regressed[2] += 25.0
    score, metrics = closure_return_trajectory_cost(
        np.stack([start, regressed, start]), goal
    )
    assert score > 100000.0
    assert not metrics["closure_preserved"]
    assert not metrics["task_success"]


def test_closure_return_uses_full_joint_success_not_agent_only_distance():
    goal = np.array([100.0, 100.0, 200.0, 200.0, 0.0, 0.0, 0.0])
    block_closed_agent_far = goal.copy()
    block_closed_agent_far[:2] = [150.0, 100.0]
    score, metrics = closure_return_trajectory_cost(
        np.stack([block_closed_agent_far, block_closed_agent_far]), goal
    )
    assert metrics["closure_preserved"]
    assert not metrics["task_success"]
    assert score == 50.0


def test_terminal_stability_certificate_rejects_pose_reopening():
    class ReopeningEnv:
        def reset(self, seed=None, options=None):
            self.state = np.asarray(options["state"], dtype=np.float64).copy()
            return {"state": self.state.copy()}, {"goal_state": options["goal_state"]}

        def step(self, action):
            self.state = self.state.copy()
            self.state[2] += 25.0
            return {"state": self.state.copy()}, 0.0, False, False, {}

    goal = np.array([100.0, 100.0, 200.0, 200.0, 0.0, 0.0, 0.0])
    final, metrics = terminal_stability_certificate(
        ReopeningEnv(), goal, goal, steps=1
    )
    assert not metrics["stable"]
    assert metrics["failure"] == "strict_criterion_lost_during_hold"
    assert final[2] == goal[2] + 25.0
