import numpy as np

from experiments.spectral_reachability import pc_full_action_composition_20261103_codex as exp
from experiments.spectral_reachability import pc_exact_unicycle_strong_baseline as strong


def test_response_generators_agree_and_allow_both_prefix_coordinates():
    goal = np.array([.7, .6])
    spectral, info = exp.generate_response(goal, True)
    ordinary, ablation = exp.generate_response(goal, False)
    np.testing.assert_allclose(spectral, ordinary, atol=1e-12)
    assert np.any((np.abs(spectral[:, 0]) > 1e-6) & (np.abs(spectral[:, 1]) > 1e-6))
    assert info["svd_calls"] == 1 and ablation["svd_calls"] == 0
    assert info["generation_evaluations"] == 5


def test_objective_replay_and_budgets():
    goal = np.array([.4, .7])
    prefixes, _ = exp.generate_response(goal, True)
    for staged, generation in ((True, 5), (False, 0)):
        outcome = exp.optimize(goal, prefixes, 24, generation, staged)
        point = np.asarray(outcome["point"])
        value, _ = strong.objective_gradient(point, goal)
        replay, _ = exp.base.evaluate(exp.make_chain(point), goal)
        np.testing.assert_allclose(np.sqrt(value), replay, atol=1e-12)
        assert outcome["total_evaluations"] <= 24
        assert sum(outcome["allocations"]) == 24 - generation
        assert outcome["analytic_gradient_evaluations"] == outcome["optimizer_evaluations"]
        assert all(lo <= x <= hi for x, (lo, hi) in zip(point, exp.BOUNDS, strict=True))


def test_mechanism_uses_own_zero_suffix():
    result = exp.mechanism(np.array([0., .8]), [0., np.pi / 2, .8])
    assert result["selected_active_change"] < result["zero_active_change"]
    assert result["selected_prefix_optimized_distance"] < result["zero_prefix_optimized_distance"]
