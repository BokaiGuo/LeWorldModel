import numpy as np
import pytest

from experiments.spectral_reachability import pc_spectral_bank_planner_20261103_codex as planner


def test_spectral_bank_recovers_model_response_and_counts_calls():
    meter = planner.Meter({"prediction_samples": 20, "svd": 1})
    matrix = np.diag(np.arange(1, 11))
    bank, evidence = planner.spectral_bank(lambda x: matrix @ x.reshape(-1), meter, 7)
    np.testing.assert_allclose(evidence["jacobian"], matrix, atol=1e-12)
    assert len(bank) == 15
    assert meter.counts == {"prediction_samples": 20, "svd": 1}
    for item in bank[11:]:
        assert item["source"] == "rgb_model"
        assert np.max(np.abs(item["prefix"])) <= 0.5


def test_shared_suffix_greedy_and_selected_nonprogress_binding():
    meter = planner.Meter({"prediction_samples": 100, "suffix_optimizations": 20})
    bank = [
        {"name": "zero", "prefix": np.zeros((5, 2)).tolist()},
        {"name": "greedy", "prefix": np.full((5, 2), 0.2).tolist()},
        {"name": "delayed", "prefix": np.full((5, 2), -0.2).tolist()},
    ]
    def score(x):
        return 1 - x.mean()

    def optimize(prefix, budget, ledger):
        terminal = ledger.call("prediction_samples", 1, lambda x: 0.1 if x.mean() < 0 else 0.5, prefix)
        return np.zeros((10, 2)), terminal

    result = planner.plan_bank(bank, 1.0, score, optimize, meter, 4, retained_per_group=2)
    assert result["same_bank_greedy_name"] == "greedy"
    assert result["selected"]["name"] == "delayed"
    assert result["qualified_against_zero_only"]
    assert meter.counts == {"prediction_samples": 6, "suffix_optimizations": 3}


def test_budget_exhaustion_and_unmetered_optimizer_fail_closed():
    meter = planner.Meter({"prediction_samples": 2, "svd": 1})
    with pytest.raises(RuntimeError, match="budget exhausted"):
        planner.spectral_bank(lambda x: x, meter, 0)
    bank = [{"name": "zero", "prefix": np.zeros((5, 2)).tolist()}]
    meter = planner.Meter({"prediction_samples": 5, "suffix_optimizations": 1})
    with pytest.raises(RuntimeError, match="sample cap"):
        planner.plan_bank(bank, 1, lambda x: 1, lambda p, b, m: (np.zeros((10, 2)), 0), meter, 4)


def test_zero_response_does_not_create_arbitrary_spectral_candidates():
    meter = planner.Meter({"prediction_samples": 20, "svd": 1})
    bank, evidence = planner.spectral_bank(lambda x: np.zeros(2), meter, 0)
    assert len(bank) == 11
    assert evidence["singular_values"] == [0, 0]
