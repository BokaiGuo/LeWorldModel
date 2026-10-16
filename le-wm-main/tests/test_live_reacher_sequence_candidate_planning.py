import numpy as np
import pytest
import torch

from experiments.run_live_reacher_sequence_candidate_planning import _qpos_distance
from experiments.run_live_tworoom_planning import CandidateBankCEMSolver


def test_qpos_distance_wraps_joint_angles() -> None:
    current = np.asarray([np.pi - 0.1, -np.pi + 0.2])
    target = np.asarray([-np.pi + 0.1, np.pi - 0.2])
    assert _qpos_distance(current, target) == pytest.approx(
        np.linalg.norm([-0.2, 0.4])
    )


def test_qpos_distance_requires_two_joints() -> None:
    with pytest.raises(ValueError, match="at least two joints"):
        _qpos_distance(np.asarray([0.0]), np.asarray([0.0]))


def test_candidate_solver_validates_final_action_selection() -> None:
    with pytest.raises(ValueError, match="final_action_selection"):
        CandidateBankCEMSolver(
            model=torch.nn.Linear(1, 1),
            candidate_bank=torch.zeros(1, 1, 1),
            final_action_selection="median",
        )
