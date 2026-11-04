import json
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import pc_rank2_nominal_side_20261103 as exp


def test_task_mode_and_full_ridge():
    matrix, residual = np.diag([8., 1.]), np.array([0., 1.])
    task = exp.increment(matrix, residual, "task")
    np.testing.assert_allclose(task, exp.increment(matrix, residual, "ridge"))
    np.testing.assert_array_equal(exp.increment(matrix, residual, "gain"), [0., 0.])
    assert task[1] > .99


@pytest.mark.parametrize("method", exp.METHODS)
def test_bank_rank_cost_bounds_and_selected_replay(method):
    goal = np.array([.2, .8])
    result = exp.plan(goal, 0, 64, method)
    counts = result["generation_counts"]
    assert len(result["bank"]) == 8
    assert result["generation_calls"] == (8 if method == "random" else
                                            56 if method == "task_greedy" else 48)
    assert result["total_calls"] == result["generation_calls"] + result["optimizer_objectives"] <= 64
    assert sum(result["allocations"]) + result["generation_calls"] == 64
    assert counts["svd"] == (8 if method in ("task", "gain", "task_greedy") else 0)
    for item in result["bank"]:
        if "response" in item:
            assert np.linalg.matrix_rank(item["response"]) == 2
    point = np.asarray(result["point"])
    assert np.all(point >= [-1., -exp.joint.dynamics.TURN, -1.])
    assert np.all(point <= [1., exp.joint.dynamics.TURN, 1.])
    distance, _ = exp.joint.dynamics.evaluate(exp.joint.actions(point), goal)
    assert distance == pytest.approx(result["terminal_distance"], abs=1e-10)
    assert distance <= result["pre_refinement_incumbent"]["terminal_distance"] + 1e-10


def test_same_bank_greedy_and_task_candidates_identical():
    goal = np.array([.4, -.6])
    task, _ = exp.generate(goal, 2, "task")
    greedy, _ = exp.generate(goal, 2, "task_greedy")
    for a, b in zip(task, greedy, strict=True):
        np.testing.assert_array_equal(a["point"], b["point"])
        assert a["terminal_distance"] == b["terminal_distance"]


def test_saved_complete_panel_replay_binding_and_interventions():
    report = json.loads(exp.OUTPUT.read_text())
    for path, digest in report["dependencies"].items():
        assert exp.joint.digest(Path(path)) == digest
    assert len(report["rows"]) == 16
    assert exp.summarize(report["rows"]) == report["summary"]
    replayed = 0
    for row in report["rows"]:
        goal = np.asarray(row["goal"])
        for cap in exp.CAPS:
            arms = {x["method"]: x for x in row["results"] if x["cap"] == cap}
            assert set(arms) == set(exp.METHODS)
            for arm in arms.values():
                error, _ = exp.joint.dynamics.evaluate(exp.joint.actions(arm["point"]), goal)
                assert abs(error - arm["terminal_distance"]) < 1e-10
                assert arm["generation_calls"] + arm["optimizer_objectives"] == arm["total_calls"] <= cap
                assert arm["intervention"] == exp.joint.intervention(
                    goal, arm["point"], arms["task_greedy"]["point"])
                for singular in arm["diagnostic_response_singular_values"]:
                    assert singular[1] > singular[0] * 1e-10
                replayed += 1
    assert replayed == 240
