import json
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import pc_svd2_side_20261103 as d


def test_two_dimensional_response_and_actual_rank():
    for turn in np.linspace(-np.pi/2, np.pi/2, 9):
        matrix = d.response_matrix(turn)
        np.testing.assert_allclose(matrix, [[1., np.cos(turn)], [0., np.sin(turn)]], atol=1e-12)
        assert np.linalg.matrix_rank(matrix) == (1 if turn == 0 else 2)


def test_full_svd_matches_no_svd_including_rank_deficiency():
    for turn in np.linspace(-np.pi/2, np.pi/2, 9):
        matrix = d.response_matrix(turn)
        for goal in ([.3, .8], [-.4, .2]):
            goal = np.asarray(goal)
            np.testing.assert_allclose(d.project(matrix, goal, "svd2"),
                                       d.project(matrix, goal, "ridge_no_svd"), atol=1e-11)


def test_second_mode_and_first_velocity_are_not_artificially_removed():
    goal = np.array([.3, .8])
    matrix = d.response_matrix(np.pi/4)
    full = d.project(matrix, goal, "svd2")
    truncated = d.project(matrix, goal, "svd1_same_matrix")
    assert np.linalg.norm(matrix @ full - goal) < np.linalg.norm(matrix @ truncated - goal)
    assert any(abs(x["point"][0]) > .01 for x in d.bank(goal, "svd2"))


@pytest.mark.parametrize("method", d.METHODS)
def test_budget_bounds_replay_and_retained_generation_best(method):
    result = d.run_one(np.array([.3, .8]), 0, 16, method)
    assert result["total_objective_evaluations"] <= 16
    assert result["total_objective_evaluations"] == (
        result["generation_objective_evaluations"] + result["optimizer_objective_evaluations"])
    for value, (lo, hi) in zip(result["point"], d.prior.BOUNDS, strict=True):
        assert lo <= value <= hi
    np.testing.assert_allclose(result["error"], result["diagnostic"]["replay_error"], atol=1e-12)
    if result["candidates"]:
        assert result["error"] <= result["generation_best_error"]


def test_saved_panel_replays_and_binds_sources():
    if not d.OUTPUT.exists():
        pytest.skip("run development experiment before saved-panel audit")
    report = json.loads(d.OUTPUT.read_text())
    for path, digest in report["sha256"].items():
        assert d.prior.sha(Path(path)) == digest
    assert len(report["rows"]) == 16
    assert report["summary"] == d.summarize(report["rows"])
    for index, row in enumerate(report["rows"]):
        assert row["index"] == index
        angle = 2 * np.pi * (index + .37) / 16
        np.testing.assert_allclose(row["goal"], .9 * np.array([np.cos(angle), np.sin(angle)]))
        assert {(x["method"], x["cap"]) for x in row["results"]} == {
            (method, cap) for method in d.METHODS for cap in (16, 32, 64)}
        for result in row["results"]:
            assert result["total_objective_evaluations"] <= result["cap"]
            assert result["svd_calls"] == (9 if result["method"] in
                                          ("svd2", "svd1_same_matrix", "old_turn_only") else 0)
            observed = d.diagnose(result["point"], np.asarray(row["goal"]))
            np.testing.assert_allclose(observed["replay_error"], result["error"], atol=1e-12)
            assert observed == result["diagnostic"]
        for cap in (16, 32, 64):
            paired = {x["method"]: x for x in row["results"] if x["cap"] == cap}
            # Near-tied symmetric candidates can swap score order due to
            # roundoff. Compare each fixed-angle construction, not tie order.
            np.testing.assert_allclose(
                [x["point"] for x in paired["svd2"]["candidates"]],
                [x["point"] for x in paired["ridge_no_svd"]["candidates"]],
                atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(paired["svd2"]["error"], paired["ridge_no_svd"]["error"],
                                       atol=1e-10, rtol=1e-10)
