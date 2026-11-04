import numpy as np
import pytest

from experiments.pc_e0_response_waypoint_scale_sweep_posthoc_20261104_codex import (
    fixed_branch_jacobian,
)
from experiments.pc_e0_wall_independent_branch_replay_dev_20261104_codex import independent


@pytest.mark.parametrize("start", [[-2.0, 2.0, 0.3], [-0.05, 0.0, 0.0]])
def test_fixed_branch_jacobian_matches_float64_finite_difference(start):
    actions = np.array([[0.25, 0.03], [0.2, -0.04], [0.15, 0.02],
                        [0.1, 0.01], [0.2, -0.02]], dtype=np.float64)
    path, blocked = independent(start, actions[None])
    jac = fixed_branch_jacobian(actions, path[0], blocked[0])
    epsilon = 1e-6
    columns = []
    for index in range(actions.size):
        delta = np.zeros_like(actions)
        delta.flat[index] = epsilon
        candidate_paths, candidate_blocked = independent(
            start, np.stack((actions + delta, actions - delta))
        )
        np.testing.assert_array_equal(candidate_blocked, np.repeat(blocked, 2, axis=0))
        columns.append((candidate_paths[0, -1, :2] - candidate_paths[1, -1, :2])
                       / (2 * epsilon))
    np.testing.assert_allclose(jac, np.stack(columns, axis=1), atol=2e-10, rtol=1e-8)
    assert bool(np.any(blocked)) == (start[0] == -0.05)
