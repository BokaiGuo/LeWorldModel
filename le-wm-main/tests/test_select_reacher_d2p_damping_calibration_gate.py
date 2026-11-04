from experiments.spectral_reachability.select_reacher_d2p_damping import select


def _report(damping: float, value: float) -> dict:
    return {
        "schema": "d2p_reacher_p0_goal_progress.v1",
        "status": "calibration_only",
        "protocol": {
            "calibration_only": True,
            "action_parameterization": "tanh mapped to environment raw action box after checkpoint normalization",
            "environment_action_box_raw": [[-1.0], [1.0]],
            "model_action_box_normalized": [[-1.0], [1.0]],
            "future_horizon": 2,
            "history_len": 3,
            "action_block": 5,
            "checkpoint_action_block": 5,
            "action_layout": "flattened time-major [action_block, action_dim]",
            "relinearization_steps": 4,
            "d2p_restarts": 1,
            "d2p_initialization": "screened_gaussian_proposals",
            "d2p_proposal_samples": 32,
            "d2p_proposal_elites": 4,
            "context_mode": "executed_warmup",
            "context_support": "uniform",
            "rollout_root_conditioning": "observed_post_warmup_rgb_anchor",
            "execution_root_reconstruction": "evaluator_only_observed_root_snapshot_restore",
            "warmup_policy": "seeded_bounded_excitation_v1",
            "qpos_readout_binding": None,
            "warmup_steps": 15,
        },
        "inputs": {
            "weights_sha256": "a" * 64,
            "adapter_sha256": "b" * 64,
            "runner_sha256": "c" * 64,
        },
        "gates": {
            "cuda_only": True,
            "frozen_parameter_hash_unchanged": True,
            "d2p_equals_gauss_newton_each_root": True,
            "all_physical_outcomes_finite": True,
            "planner_and_execution_root_rgb_identity_verified": True,
            "all_executed_raw_actions_within_environment_box": True,
            "actual_planning_predict_samples_matched_to_cem": False,
        },
        "rows": [{
            "split": "calibration",
            # Every damping candidate must be evaluated on the same
            # calibration roots; only the damping changes.
            "seed": 100,
            "methods": {
                "d2p_goal_progress": {
                    "physical": {"final_qpos_distance": value},
                    "receipt": {"damping": damping},
                }
            },
        }],
    }


def test_calibration_selector_exempts_outcome_only_matching_gate():
    result = select([_report(0.01, 0.2), _report(10.0, 0.3)])
    assert result["status"] == "pass"
    assert result["selected"]["damping"] == 0.01


def test_calibration_selector_still_rejects_other_failed_gate():
    report = _report(0.01, 0.2)
    report["gates"]["cuda_only"] = False
    result = select([report, _report(10.0, 0.3)])
    assert result["status"] == "invalid"
