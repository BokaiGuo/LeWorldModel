import numpy as np
import pytest

from experiments import pc_v25_witness_suffix_expression_dev_20261104_codex as experiment


@pytest.mark.parametrize("arm", experiment.ARMS)
def test_each_arm_charges_all_samples_and_preserves_prefix(arm):
    prefixes = np.array([np.full((5, 2), .2), np.full((5, 2), -.2)], np.float32)
    batches = []
    def predict(actions):
        actions = np.asarray(actions, np.float32)
        batches.append(actions.copy())
        return np.zeros((len(actions), 2)), np.sum(actions**2, axis=(1, 2))
    output = experiment.optimize(predict, prefixes, 0, arm)
    assert sum(len(batch) for batch in batches) == 2 * 2 * 121
    assert len(output) == 2
    for index, item in enumerate(output):
        for run in item["runs"]:
            np.testing.assert_array_equal(np.asarray(run["actions"], np.float32)[:5], prefixes[index])
            assert run["samples"] == 121
