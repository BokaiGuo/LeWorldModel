import numpy as np
import pytest

from experiments.pc_v23_baseline_prefix_injection_posthoc import measure
from experiments.pc_v23_cem_generation_reconstruction import reconstruct
from experiments.spectral_reachability import pc_shared_multistart_cem as cem


def test_reconstruct_cem_detects_altered_action_and_accounts_short_batch():
    calls = []

    def evaluate(actions):
        scores = np.square(actions - .2).sum(axis=(1, 2))
        calls.append((actions.copy(), scores.copy()))
        return scores

    tails = np.zeros((2, 3, 2), np.float32)
    result = cem.optimize(evaluate, np.empty((0, 2), np.float32), tails, 23,
                          seed=11, batch_size=4, elites=2)
    rebuilt = reconstruct(*calls[0], calls[1:], result["starts"], 11, 4, 2)
    assert sum(start["samples"] for start in rebuilt) == 23
    assert any(g["samples"] < 4 for start in rebuilt for g in start["generations"])
    changed = [(actions.copy(), scores.copy()) for actions, scores in calls[1:]]
    changed[0][0][0, 0, 0] += .01
    with pytest.raises(AssertionError):
        reconstruct(*calls[0], changed, result["starts"], 11, 4, 2)


def test_prefix_injection_distinguishes_all_and_retained():
    reference = np.zeros((15, 2), np.float32)
    reference[:, 0] = 0.1
    layer = {"depth": 0, "candidates": [
        {"chain": reference[:5].tolist()},
        {"chain": np.zeros((5, 2), np.float32).tolist()}],
        "retained": [1]}
    result = measure([-0.5, 2, 0], [-0.05, 2], reference, layer)
    assert result["all"]["injected_success_count"] == 1
    assert result["retained"]["injected_success_count"] == 0
    assert result["all"]["nearest_action_rms"] == 0
    assert result["retained"]["nearest_action_rms"] > 0


def test_prefix_injection_rejects_out_of_range_retention():
    layer = {"depth": 0, "candidates": [{"chain": np.zeros((5, 2)).tolist()}], "retained": [1]}
    with pytest.raises(ValueError):
        measure([-0.5, 2, 0], [0, 2], np.zeros((15, 2)), layer)
