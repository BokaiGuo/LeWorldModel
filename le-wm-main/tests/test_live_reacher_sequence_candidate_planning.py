import numpy as np
import pytest
import torch

from experiments.run_live_reacher_sequence_candidate_planning import (
    _qpos_distance,
    validate_reacher_capacity_contract,
)
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


def test_reacher_capacity_contract_rejects_tworoom_chart() -> None:
    with pytest.raises(ValueError, match="absolute-qpos contract"):
        validate_reacher_capacity_contract(
            {
                "target_mode": "root_centered",
                "goal_conditioning": "latent_relative",
                "phase_conditioning": "tworoom_rgb_phase_waypoint_action",
                "action_frame": "raw",
            }
        )


def test_reacher_capacity_contract_accepts_absolute_raw_chart() -> None:
    validate_reacher_capacity_contract(
        {
            "target_mode": "absolute",
            "goal_conditioning": "none",
            "phase_conditioning": "none",
            "action_frame": "raw",
        }
    )


def test_reacher_capacity_contract_accepts_proposal_relative_chart() -> None:
    validate_reacher_capacity_contract(
        {
            "target_mode": "absolute",
            "goal_conditioning": "none",
            "phase_conditioning": "none",
            "action_frame": "proposal_relative",
        }
    )


def test_candidate_solver_validates_final_action_selection() -> None:
    with pytest.raises(ValueError, match="final_action_selection"):
        CandidateBankCEMSolver(
            model=torch.nn.Linear(1, 1),
            candidate_bank=torch.zeros(1, 1, 1),
            final_action_selection="median",
        )
