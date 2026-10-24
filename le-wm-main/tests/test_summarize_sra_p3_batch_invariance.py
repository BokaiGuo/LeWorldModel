from experiments.spectral_reachability.summarize_sra_p3_batch_invariance import compare_reports


def _report(batch: int, model_cost: float = 1.0, choice: int = 0):
    horizon = {
        "tree_sequences": 30,
        "model_first_costs": [model_cost, 2.0],
        "immediate_first_costs": [3.0, 4.0],
        "physical_first_costs": [0.1, 0.2],
        "spectral_index": choice,
        "immediate_index": 1,
        "physical_index": 0,
        "spectral_regret": 0.0,
        "immediate_regret": 0.1,
    }
    return {
        "schema": "sra_p3_fixed_tree_horizon_expansion.v1",
        "status": "mechanism_ready",
        "inputs": {"roots": [1], "horizons": [2], "model_batch_size": batch, "checkpoint_sha256": "x", "physical_position_backend": {"x": 1}},
        "gates": {"cuda_only": True},
        "aggregate_spectral_minus_immediate_regret": {"2": -0.1},
        "rows": [{"root": 1, "horizons": {"2": horizon}}],
    }


def test_batch_reducer_accepts_roundoff_without_action_or_outcome_change():
    result = compare_reports(_report(64), _report(8, model_cost=1.0 + 1.0e-6), atol=1.0e-4)
    assert result["status"] == "pass"
    assert result["gates"]["same_selection_indices"]


def test_batch_reducer_rejects_selection_change():
    result = compare_reports(_report(64), _report(8, choice=1), atol=1.0e-4)
    assert result["status"] == "invalid"
    assert not result["gates"]["same_selection_indices"]
