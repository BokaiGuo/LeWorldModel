from experiments.spectral_reachability.select_reacher_d2p_p1_margin_guard import select


def _report():
    return {
        "schema": "d2p_reacher_p1_reconfiguration.v1", "status": "calibration_only",
        "inputs": {"weights_sha256": "weights", "adapter_sha256": "adapter", "runner_sha256": "runner"},
        "protocol": {"mechanism_observability": "observed_encoded_goal_cost", "proposal_distribution": "physical_uniform", "selection_metric": "bounded_dls_suffix_goal_cost", "damping": 10.0, "action_block": 5},
        "rows": [
            {"seed": 1, "methods": {"reconfiguration": {"first_receipt": {"non_greedy_witness_found": True, "non_greedy_model_margin": 0.01}}}},
            {"seed": 2, "methods": {"reconfiguration": {"first_receipt": {"non_greedy_witness_found": True, "non_greedy_model_margin": 0.03}}}},
            {"seed": 3, "methods": {"reconfiguration": {"first_receipt": {"non_greedy_witness_found": True, "non_greedy_model_margin": 0.05}}}},
            {"seed": 4, "methods": {"reconfiguration": {"first_receipt": {"non_greedy_witness_found": True, "non_greedy_model_margin": 0.08}}}},
        ],
    }


def _fidelity():
    return {
        "schema": "d2p_reacher_p1_first_chunk_fidelity_audit.v1", "status": "completed",
        "rows": [
            {"seed": 1, "model_candidate_minus_zero_goal_cost": 0.01, "observed_candidate_minus_zero_goal_cost": -0.01, "model_to_observation_goal_order_agrees": False},
            {"seed": 2, "model_candidate_minus_zero_goal_cost": 0.03, "observed_candidate_minus_zero_goal_cost": 0.01, "model_to_observation_goal_order_agrees": True},
            {"seed": 3, "model_candidate_minus_zero_goal_cost": 0.05, "observed_candidate_minus_zero_goal_cost": 0.02, "model_to_observation_goal_order_agrees": True},
            {"seed": 4, "model_candidate_minus_zero_goal_cost": 0.08, "observed_candidate_minus_zero_goal_cost": 0.03, "model_to_observation_goal_order_agrees": True},
        ],
    }


def test_margin_guard_selects_smallest_clean_threshold_without_using_outcomes():
    result = select(report=_report(), fidelity=_fidelity(), minimum_passing_roots=3)
    assert result["status"] == "pass"
    assert result["selected"]["min_non_greedy_margin"] == 0.03
    assert result["selected"]["calibration_admitted_root_seeds"] == [2, 3, 4]
