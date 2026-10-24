from experiments.spectral_reachability.select_reacher_d2p_damping import select


def _report(damping: float, values):
    rows = [{"seed": seed, "split": "calibration", "methods": {"d2p_goal_progress": {"physical": {"final_qpos_distance": value}, "receipt": {"damping": damping}}}} for seed, value in enumerate(values)]
    return {
        "schema": "d2p_reacher_p0_goal_progress.v1", "status": "calibration_only",
        "gates": {"cuda_only": True}, "inputs": {"weights_sha256": "w", "adapter_sha256": "a"},
        "protocol": {"calibration_only": True, "action_parameterization": "box", "environment_action_box_raw": [[-1], [1]], "model_action_box_normalized": [[-2], [2]], "future_horizon": 2, "action_block": 5, "checkpoint_action_block": 5, "action_layout": "flattened time-major [action_block, action_dim]", "relinearization_steps": 1, "d2p_restarts": 1, "execution_root_reconstruction": "snapshot"},
        "rows": rows,
    }


def test_damping_selection_uses_only_calibration_d2p_outcome_and_stable_tie_break():
    result = select([_report(1.0, [0.4, 0.6]), _report(0.1, [0.5, 0.5])])
    assert result["status"] == "pass"
    assert result["selected"]["damping"] == 0.1


def test_damping_selection_rejects_non_calibration_receipt():
    bad = _report(1.0, [0.4, 0.6]); bad["status"] = "method_negative_against_cem"
    result = select([_report(0.1, [0.5, 0.5]), bad])
    assert result["status"] == "invalid"


def test_damping_selection_rejects_mixed_action_layout_contracts():
    incompatible = _report(1.0, [0.4, 0.6])
    incompatible["protocol"]["action_layout"] = "channel-major"
    result = select([_report(0.1, [0.5, 0.5]), incompatible])
    assert result["status"] == "invalid"


def test_damping_selection_binds_execution_root_reconstruction():
    incompatible = _report(1.0, [0.4, 0.6])
    incompatible["protocol"]["execution_root_reconstruction"] = "reset_and_replay"
    result = select([_report(0.1, [0.5, 0.5]), incompatible])
    assert result["status"] == "invalid"
