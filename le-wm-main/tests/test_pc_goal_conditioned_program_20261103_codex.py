import numpy as np

from experiments.spectral_reachability.pc_goal_conditioned_program_20261103_codex import (
    compile_program, generate,
)
from experiments.spectral_reachability.pc_spectral_bank_planner_20261103_codex import Meter


def meter():
    return Meter({"prediction_samples": 200, "svd": 3, "linear_solve": 1,
                  "suffix_optimizations": 10})


def response(action):
    x = np.asarray(action).reshape(-1)
    return np.array([10 * x[0], x[1]])


def test_goal_conditioned_mode_differs_from_largest_singular():
    goal, m = np.array([0., 1.]), meter()
    _, goal_info = generate(response, goal, m, mode="goal", seed=4)
    _, sigma_info = generate(response, goal, meter(), mode="singular", seed=4)
    assert goal_info["selected_mode"] == 1
    assert sigma_info["selected_mode"] == 0
    assert m.counts == {"prediction_samples": 21, "svd": 1}


def test_full_program_selection_and_same_chain_binding():
    m = meter()
    goal = np.array([0., 1.])
    def prefix_score(prefix):
        return float(np.linalg.norm(response(prefix) - goal))
    def suffix(prefix, samples, current_meter):
        score = current_meter.call("prediction_samples", 1, prefix_score, prefix)
        return np.zeros((10, 2)), score
    result = compile_program(response, goal, prefix_score, suffix, m, mode="goal",
                             seed=4, initial_score=1., suffix_samples=3)
    assert result["selected"]["terminal_score"] == min(
        item["terminal_score"] for item in result["programs"])
    assert np.asarray(result["selected"]["prefix"]).shape == (5, 2)
    assert np.asarray(result["selected"]["suffix"]).shape == (10, 2)
    assert m.counts["prediction_samples"] == 27


def test_random_and_ridge_are_metered_as_distinct_controls():
    for mode in ("random", "ridge"):
        m = meter()
        bank, _ = generate(response, np.array([0., 1.]), m, mode=mode, seed=1)
        assert len(bank) == 3
        assert m.counts["prediction_samples"] == 21
        assert m.counts.get("svd", 0) == 0
        assert m.counts.get("linear_solve", 0) == (mode == "ridge")
