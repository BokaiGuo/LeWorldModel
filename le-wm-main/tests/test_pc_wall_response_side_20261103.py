import json
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import pc_wall_response_side_20261103 as exp


def test_task_roster_and_privileged_feasibility_not_planner_input():
    assert len(exp.tasks()) == 8
    for task in exp.tasks():
        records = exp.feasibility(task)
        assert any(x["within_horizon"] and x["terminal_distance"] < 1e-10
                   and not any(x["blocked"]) for x in records)


def test_batch_matches_scalar_continuous_history_and_wall_blocking():
    actions = np.random.default_rng(19).uniform(-1, 1, (3, 16, 2)) * exp.HIGH
    trace, blocked = exp.rollout([-.5, .25, .3], actions)
    for i in range(3):
        scalar, events = exp.rollout([-.5, .25, .3], actions[i:i+1])
        np.testing.assert_array_equal(trace[i], scalar[0])
        np.testing.assert_array_equal(blocked[i], events[0])
    direct = np.zeros((1, 16, 2))
    direct[:, :, 0] = .3
    path, events = exp.rollout([-.5, 0., 0.], direct)
    assert events.any()
    assert np.all(path[0, :, 0] < 0)
    with pytest.raises(ValueError):
        exp.rollout([-.5, 0., 0.], direct * 2)


@pytest.mark.parametrize("arm", exp.ARMS)
def test_planner_budget_binding_and_incumbent(arm):
    task = exp.tasks()[0]
    result = exp.plan(task, 0, 512, arm)
    assert result["total_calls"] == 512
    assert sum(result["allocations"]) + result["generation_calls"] == 512
    assert result["generation_calls"] == (264 if arm in ("task", "gain", "ridge") else 4)
    assert result["terminal_distance"] <= min(result["start_costs"])
    trace, events = exp.rollout(task["start"], [result["actions"]])
    np.testing.assert_array_equal(trace[0], result["trajectory"])
    np.testing.assert_array_equal(events[0], result["blocked"])
    if arm == "ridge":
        assert result["svd_calls"] == 0 and result["solve_calls"] == 4


def test_saved_panel_sources_and_every_chain():
    if not exp.OUTPUT.exists():
        pytest.skip("full development panel not yet run")
    report = json.loads(exp.OUTPUT.read_text())
    for path, expected in report["dependencies"].items():
        assert exp.digest(Path(path)) == expected
    assert len(report["rows"]) == 8
    count = 0
    for row in report["rows"]:
        assert {(r["arm"], r["cap"]) for r in row["results"]} == {
            (arm, cap) for arm in exp.ARMS for cap in exp.CAPS}
        for result in row["results"]:
            trace, events = exp.rollout(row["task"]["start"], [result["actions"]])
            np.testing.assert_array_equal(trace[0], result["trajectory"])
            np.testing.assert_array_equal(events[0], result["blocked"])
            assert abs(np.linalg.norm(trace[0, -1, :2] - row["task"]["goal"])
                       - result["terminal_distance"]) < 1e-12
            assert result["total_calls"] == result["cap"]
            assert sum(result["allocations"]) + result["generation_calls"] == result["cap"]
            count += 1
    assert count == 96
    for cap in exp.CAPS:
        for arm in exp.ARMS:
            values = [r["terminal_distance"] for row in report["rows"]
                      for r in row["results"] if r["cap"] == cap and r["arm"] == arm]
            saved = report["summary"][str(cap)][arm]
            assert saved["mean_distance"] == float(np.mean(values))
            assert saved["successes_below_005"] == sum(x < .05 for x in values)


def test_seeded_plan_reproducible_without_feasibility_injection(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("planner must not use feasibility witness")
    monkeypatch.setattr(exp, "feasibility", forbidden)
    task = exp.tasks()[1]
    first = exp.plan(task, 1, 512, "ridge")
    second = exp.plan(task, 1, 512, "ridge")
    for key in ("actions", "starts", "responses", "optimization_trace", "total_calls"):
        assert first[key] == second[key]
