import numpy as np

from experiments.spectral_reachability.pc_exact_unicycle_development import evaluate
from experiments.spectral_reachability.pc_exact_unicycle_random_bridge import random_bridge


def test_fixed_seed_and_matching_budget():
    goal = np.array([0.4, 0.7])
    actions, ledger = random_bridge(goal, 91200)
    again, _ = random_bridge(goal, 91200)
    np.testing.assert_array_equal(actions, again)
    assert ledger["transitions"] == 10
    assert ledger["objective_evaluations"] == ledger["response_evaluations"] == 5
    assert np.isfinite(evaluate(actions, goal)[0])
