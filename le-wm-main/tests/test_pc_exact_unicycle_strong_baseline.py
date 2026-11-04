import numpy as np

from experiments.spectral_reachability.pc_exact_unicycle_development import evaluate
from experiments.spectral_reachability.pc_exact_unicycle_strong_baseline import (
    analytical_witness, multistart_gradient, objective_gradient,
)


def test_analytic_gradient_matches_central_difference():
    point = np.array([0.2, -0.6, 0.7])
    goal = np.array([0.4, -0.3])
    _, gradient = objective_gradient(point, goal)
    epsilon = 1e-6
    finite = []
    for i in range(3):
        direction = np.eye(3)[i] * epsilon
        finite.append((objective_gradient(point + direction, goal)[0] -
                       objective_gradient(point - direction, goal)[0]) / (2 * epsilon))
    np.testing.assert_allclose(gradient, finite, rtol=1e-8, atol=1e-8)


def test_oracle_witness_reaches_goals_in_all_quadrants():
    for goal in ([0.4, 0.7], [-0.4, 0.7], [-0.4, -0.7], [0.4, -0.7]):
        cost, _ = evaluate(analytical_witness(goal), goal)
        assert cost < 1e-12


def test_gradient_budget_and_feasibility():
    actions, ledger = multistart_gradient(np.array([0.2, 0.9]))
    assert ledger["objective_evaluations"] <= 32
    assert ledger["analytic_gradient_evaluations"] == ledger["objective_evaluations"]
    assert evaluate(actions, np.array([0.2, 0.9]))[0] < 0.1
