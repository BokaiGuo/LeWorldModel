import pytest

from experiments.spectral_reachability.select_reacher_d2p_p1_margin_band_guard import select


def _report():
    return {
        "schema": "d2p_reacher_p1_reconfiguration.v1", "status": "calibration_only",
        "inputs": {"weights_sha256": "weights", "adapter_sha256": "adapter", "runner_sha256": "runner"},
        "protocol": {"mechanism_observability": "observed_encoded_goal_cost", "proposal_distribution": "physical_uniform", "selection_metric": "bounded_dls_suffix_goal_cost", "damping": 10.0, "action_block": 5, "target_delta": [0.65, -0.65]},
        "rows": [
            {"seed": 1, "methods": {"reconfiguration": {"first_receipt": {"non_greedy_witness_found": True, "non_greedy_model_margin": 0.01}}}},
            {"seed": 2, "methods": {"reconfiguration": {"first_receipt": {"non_greedy_witness_found": True, "non_greedy_model_margin": 0.02}}}},
            {"seed": 3, "methods": {"reconfiguration": {"first_receipt": {"non_greedy_witness_found": True, "non_greedy_model_margin": 0.03}}}},
            {"seed": 4, "methods": {"reconfiguration": {"first_receipt": {"non_greedy_witness_found": True, "non_greedy_model_margin": 0.04}}}},
            {"seed": 5, "methods": {"reconfiguration": {"first_receipt": {"non_greedy_witness_found": True, "non_greedy_model_margin": 0.05}}}},
            {"seed": 6, "methods": {"reconfiguration": {"first_receipt": {"non_greedy_witness_found": True, "non_greedy_model_margin": 0.06}}}},
        ],
    }


def _fidelity():
    def row(seed, margin, observed, agrees):
        return {"seed": seed, "model_candidate_minus_zero_goal_cost": margin, "observed_candidate_minus_zero_goal_cost": observed, "model_to_observation_goal_order_agrees": agrees}
    return {
        "schema": "d2p_reacher_p1_first_chunk_fidelity_audit.v1", "status": "completed",
        "source_binding": {"checkpoint_weights_sha256": "weights", "adapter_sha256": "adapter", "target_delta": [0.65, -0.65]},
        "rows": [
            row(1, 0.01, 0.02, True), row(2, 0.02, 0.03, True), row(3, 0.03, -0.01, False),
            row(4, 0.04, 0.02, True), row(5, 0.05, 0.03, True), row(6, 0.06, 0.04, True),
        ],
    }


def test_margin_band_selects_largest_contiguous_clean_calibration_subset():
    result = select(report=_report(), fidelity=_fidelity(), minimum_passing_roots=3)
    assert result["status"] == "pass"
    assert result["selected"]["min_non_greedy_margin"] == 0.04
    assert result["selected"]["max_non_greedy_margin"] == 0.06
    assert result["selected"]["calibration_admitted_root_seeds"] == [4, 5, 6]


def test_margin_band_rejects_model_replay_inconsistency():
    fidelity = _fidelity()
    fidelity["rows"][0]["model_candidate_minus_zero_goal_cost"] = 0.2
    try:
        select(report=_report(), fidelity=fidelity, minimum_passing_roots=3)
    except ValueError as error:
        assert "inconsistent" in str(error)
    else:
        raise AssertionError("expected model replay inconsistency to fail closed")


def test_margin_band_rejects_fidelity_from_a_different_checkpoint_or_target_geometry():
    fidelity = _fidelity()
    fidelity["source_binding"]["checkpoint_weights_sha256"] = "other-weights"
    with pytest.raises(ValueError, match="checkpoint_weights_sha256"):
        select(report=_report(), fidelity=fidelity, minimum_passing_roots=3)
