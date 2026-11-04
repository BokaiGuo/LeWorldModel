import numpy as np

from experiments import pc_rgb_successor_exact_same_policy_dev_20261103_codex as base
from experiments.spectral_reachability import pc_response_diverse_compiler_v1 as compiler
from experiments.spectral_reachability.pc_spectral_bank_planner_20261103_codex import Meter


def test_response_modes_recover_linear_subspace_and_count_evaluations():
    matrix = np.zeros((2, 10))
    matrix[0, 0] = 2
    matrix[1, 1] = 1
    meter = Meter({"prediction_samples": 21, "svd": 1})

    def predict(actions):
        return np.asarray(actions).reshape(len(actions), 10) @ matrix.T, np.zeros(len(actions))

    directions, evidence = compiler.directions_from_response(np.zeros(2), predict, meter, np.zeros((5, 2)))
    np.testing.assert_allclose(evidence["jacobian"], matrix, atol=1e-12)
    np.testing.assert_allclose(directions.T @ directions, np.diag([1, 1] + [0] * 8), atol=1e-12)
    assert meter.counts["prediction_samples"] == 21
    assert meter.counts["svd"] == 1


def test_all_arms_use_legal_inputs_and_matched_cap():
    task = base.panel.tasks()[0]
    for arm in compiler.ARMS:
        meter = base.Budget()
        result = compiler.decide(arm, 0, np.asarray(task["goal"]), lambda x: meter.predict(task, x), meter)
        assert meter.token_samples <= 480
        assert np.max(np.abs(result["selected"]["chain"])) <= 1
        if arm == "full_cem":
            assert not result["bank"] and result["structure_counts"].get("svd", 0) == 0
            continue
        assert len(result["first_bank"]) == 5 and len(result["bank"]) == 6
        assert len(set(result["retained"])) == 2
        assert all(program["samples"] >= 13 for program in result["programs"])
        assert all(len(p["prefix"]) == 10 for p in result["bank"])
        if arm.startswith("response"):
            assert result["structure_counts"]["svd"] == 3
        else:
            assert result["structure_counts"].get("svd", 0) == 0
