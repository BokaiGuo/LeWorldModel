import numpy as np

from experiments.spectral_reachability import pc_spectral_start_diagnostic_20261103_codex as d
from experiments.spectral_reachability import pc_exact_unicycle_strong_baseline as strong


def test_one_dimensional_svd_and_response_projection_agree():
    goal = np.array([0.2, 0.8])
    spectral = d.response_bank(goal, True)
    ordinary = d.response_bank(goal, False)
    np.testing.assert_allclose([x["point"] for x in spectral], [x["point"] for x in ordinary])
    np.testing.assert_allclose([x["error"] for x in spectral], [x["error"] for x in ordinary])


def test_analytical_score_agrees_with_replay():
    point, goal = np.array([-.3, .8, .7]), np.array([.2, .6])
    score, _ = strong.objective_gradient(point, goal)
    distance, _ = d.base.evaluate(d.chain(point), goal)
    np.testing.assert_allclose(score, distance**2, atol=1e-14)


def test_optimizer_counts_generation_and_bounds():
    goal = np.array([.2, .7])
    starts = [x["point"] for x in d.response_bank(goal, True)[:4]]
    result = d.optimize(goal, starts, 16, 9)
    assert result["total_objective_evaluations"] <= 16
    assert sum(result["allocations"]) == 7
    assert result["analytic_gradient_evaluations"] == result["optimizer_objective_evaluations"]
    assert all(lo <= x <= hi for x, (lo, hi) in zip(result["point"], d.BOUNDS, strict=True))
