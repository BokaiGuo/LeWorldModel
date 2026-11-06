import numpy as np

from experiments.pc_cheap_direction_edit_dev_20261105_codex import cheap_arm


def test_cheap_budget_bounds_and_incumbent():
    calls = []

    def predict(chains):
        chains = np.asarray(chains)
        assert np.all(np.abs(chains) <= 1)
        calls.extend(chains.copy())
        features = chains.sum(axis=1)
        return features, np.linalg.norm(features - np.array([0.5, -0.4]), axis=1)

    parent = np.full((15, 2), 0.98, np.float32)
    result = cheap_arm(predict, parent, [0.5, -0.4], 7, False)
    assert len(calls) == 103
    scored = np.linalg.norm(np.array(calls).sum(axis=1) - np.array([0.5, -0.4]), axis=1)
    assert np.isclose(result["final_score"], scored.min())
    assert len(result["direction_records"]) == 3


def test_cheap_zero_response_and_equal_budget():
    calls = []

    def predict(chains):
        calls.extend(np.asarray(chains).copy())
        return np.zeros((len(chains), 2)), np.ones(len(chains))

    result = cheap_arm(predict, np.zeros((15, 2), np.float32), [1, 1], 11, True)
    assert len(calls) == 157
    assert all(x["zero_direction"] for x in result["direction_records"])
    assert result["final_score"] == 1
