import numpy as np
import pytest

from experiments.spectral_reachability.pc_fixed_bank_suffix_diagnostic_v1 import evaluate_bank


def test_equal_budget_and_fixed_prefixes():
    calls = []
    def predict(actions):
        calls.append(actions.copy())
        return np.zeros((len(actions), 1)), (actions ** 2).sum(axis=(1, 2))
    bank = np.array([[[0.]], [[.5]], [[.8]]], np.float32)
    result = evaluate_bank(predict, bank, {"a": [0, 1], "b": [1, 2]},
                           tail_steps=2, samples=7, restarts=2, seed=13, batch_size=3, elites=2)
    assert sum(len(batch) for batch in calls) == 3 * 2 * 7
    assert result["samples_per_candidate"] == 14
    assert result["rules"]["a"]["best_score"] == 0
    assert result["rules"]["b"]["gap_to_bank_best"] >= .25
    for index in range(3):
        for run in result["candidates"][index]["runs"]:
            np.testing.assert_array_equal(np.asarray(run["actions"][:1]), bank[index])


def test_rejects_unequal_capacity_and_bad_indices():
    bank = np.zeros((3, 1, 1), np.float32)
    with pytest.raises(ValueError, match="capacity"):
        evaluate_bank(None, bank, {"a": [0], "b": [0, 1]}, tail_steps=1,
                      samples=7, restarts=1, seed=1, batch_size=3)
    with pytest.raises(ValueError, match="indices"):
        evaluate_bank(None, bank, {"a": [3]}, tail_steps=1,
                      samples=7, restarts=1, seed=1, batch_size=3)
