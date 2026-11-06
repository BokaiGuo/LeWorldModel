import numpy as np

from experiments import pc_v25_retained_suffix_budget_curve_dev_20261104_codex as experiment


def test_loads_exact_response_retention_from_sealed_plan():
    indices, prefixes = experiment.retained(0, "original")
    assert len(indices) == 4
    assert len(set(indices)) == 4
    assert prefixes.shape == (4, 5, 2)
    assert np.isfinite(prefixes).all()
    assert np.abs(prefixes).max() <= 1
