import numpy as np
import pytest

from experiments import pc_v25_witness_suffix_budget_curve_dev_20261104_codex as experiment


@pytest.mark.parametrize("arm", experiment.ARMS)
def test_budget_checkpoints_and_reference_match(arm):
    prefixes = np.zeros((1, 5, 2), np.float32)
    counts = []
    def predict(actions):
        actions = np.asarray(actions)
        counts.append(len(actions))
        return np.zeros((len(actions), 2)), np.sum(actions**2, axis=(1, 2))
    records = experiment.optimize(predict, prefixes, 0, arm)
    old = experiment.previous.optimize(predict, prefixes, 0, arm)
    assert sum(counts) == (8 + 2) * 121
    assert records[0]["curve"]["2"] == old[0]["best_score"]
    assert records[0]["curve"]["8"] <= records[0]["curve"]["4"] <= records[0]["curve"]["2"]
