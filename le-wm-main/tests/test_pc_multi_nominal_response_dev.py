import numpy as np
import pytest

from experiments.spectral_reachability.pc_multi_nominal_response_dev import generate
from experiments.spectral_reachability.pc_spectral_bank_planner_20261103_codex import Meter


def test_multinominal_ridge_svd_equivalence_and_call_counts():
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(4, 10))
    target = rng.normal(size=4)
    nominals = rng.uniform(-0.3, 0.3, size=(3, 5, 2))
    def response(action):
        return matrix @ action.reshape(-1)
    outputs = []
    for mode in ("ridge", "svd"):
        meter = Meter({"prediction_samples": 63, "linear_solve": 3, "svd": 3})
        bank, evidence = generate(response, target, nominals, meter, mode=mode)
        assert meter.counts["prediction_samples"] == 63
        assert meter.counts["linear_solve" if mode == "ridge" else "svd"] == 3
        assert len(bank) == 4
        for record in evidence:
            np.testing.assert_allclose(record["jacobian"], matrix, atol=1e-12)
            assert np.max(np.abs(record["proposal"])) <= 1
        outputs.append(bank)
    np.testing.assert_allclose([x["prefix"] for x in outputs[0]],
                               [x["prefix"] for x in outputs[1]], atol=1e-10)


def test_rejects_nominal_outside_finite_difference_domain_and_budget():
    meter = Meter({"prediction_samples": 20, "linear_solve": 1})
    with pytest.raises(ValueError):
        generate(lambda a: np.zeros(1), np.zeros(1), np.ones((1, 5, 2)), meter)
    with pytest.raises(RuntimeError, match="budget exhausted"):
        generate(lambda a: np.zeros(1), np.zeros(1), np.zeros((1, 5, 2)), meter)
