import json

import numpy as np

from experiments.empirical_spectral_audit.run_pusht_spectral_failure_boundary_audit import run


def _spectral(path):
    A = np.eye(2).tolist()
    B = np.eye(2).tolist()
    D = np.eye(2).tolist()
    path.write_text(json.dumps({
        "fit": {
            "action_excitation": {"condition_number": 2000.0, "passes": True},
            "model": {"A": A, "B": B, "D": D, "task_rmse": 2.0, "dynamics_rmse": 0.1},
        }
    }))


def test_failure_boundary_bridge_preserves_mode_vs_pose_gap(tmp_path):
    spectral = tmp_path / "spectral.json"
    contact = tmp_path / "contact.json"
    _spectral(spectral)
    contact.write_text(json.dumps({
        "held_out": {"mode_accuracy": 0.95},
        "closed_loop": {"contact_chart_lookahead": {"metrics": {
            "pose_success": {"estimate": 0.125},
            "final_angle": {"estimate": 0.8},
            "contact_loss_rate": {"estimate": 0.0},
        }}},
    }))
    report = run([spectral], contact)
    assert report["bridge_tests"]["action_excitation_is_formally_full_rank_but_ill_conditioned"]
    assert report["bridge_tests"]["mode_accuracy_does_not_imply_pose_closure"]
    assert report["claim_boundary"].startswith("This artifact connects")
    assert np.isfinite(report["aggregate"]["action_condition_number_mean"])
