import numpy as np

from experiments.run_rgb_local_effect_adapter import constrained_trajectory_pose_costs


def _goal_rows(count: int) -> np.ndarray:
    return np.repeat(np.asarray([[10.0, 10.0, 0.0, 1.0]], dtype=np.float32), count, axis=0)


def test_intermediate_contact_loss_is_penalized_even_when_endpoint_matches():
    # Two candidates have the same final pose.  Candidate 1 leaves the
    # contact band midway through the committed horizon and returns, so an
    # endpoint-only target cannot distinguish the unsafe trajectory.
    trajectory = np.asarray([
        [[0.0, 0.0, 10.0, 10.0, 0.0, 1.0],
         [0.0, 0.0, 10.0, 10.0, 0.0, 1.0],
         [0.0, 100.0, 10.0, 10.0, 0.0, 1.0],
         [0.0, 0.0, 10.0, 10.0, 0.0, 1.0]],
        [[0.0, 0.0, 10.0, 10.0, 0.0, 1.0],
         [0.0, 0.0, 10.0, 10.0, 0.0, 1.0],
         [0.0, 0.0, 10.0, 10.0, 0.0, 1.0],
         [0.0, 0.0, 10.0, 10.0, 0.0, 1.0]],
    ], dtype=np.float32)
    data = {"trajectory_pose": trajectory, "root_id": np.asarray([0, 0])}
    # The root rows are intentionally treated as two separate candidates by
    # this synthetic direct call only through distinct root IDs below.
    data["root_id"] = np.asarray([0, 1])
    costs = constrained_trajectory_pose_costs(
        data, _goal_rows(2), contact_penalty=3.0, contact_distance_max=20.0,
    )
    assert costs[0] > costs[1]


def test_zero_penalties_reduce_to_endpoint_cost():
    trajectory = np.asarray([
        [[0.0, 0.0, 10.0, 10.0, 0.0, 1.0],
         [0.0, 0.0, 12.0, 10.0, 0.0, 1.0]],
    ], dtype=np.float32)
    data = {"trajectory_pose": trajectory, "root_id": np.asarray([0])}
    costs = constrained_trajectory_pose_costs(data, _goal_rows(1))
    expected = 2.0 / 20.0
    assert np.isclose(costs[0], expected)
