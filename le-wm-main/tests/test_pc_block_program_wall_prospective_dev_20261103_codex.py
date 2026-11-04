import json

import numpy as np
import pytest

from experiments import pc_block_program_wall_prospective_dev_20261103_codex as experiment


def test_task_panel_and_block_action_mapping():
    tasks = experiment.tasks()
    assert len(tasks) == 24
    assert len({tuple(t["start"]) + tuple(t["goal"]) for t in tasks}) == 24
    point = np.array([[.1, .2, .3, -.2, -.4, .1]])
    chain = experiment.decode(point)[0]
    np.testing.assert_array_equal(chain[:5], np.tile(point[0, :2], (5, 1)))
    np.testing.assert_array_equal(chain[5:10], np.tile(point[0, 2:4], (5, 1)))
    np.testing.assert_array_equal(chain[10:], np.tile(point[0, 4:], (5, 1)))
    with pytest.raises(ValueError):
        experiment.decode(np.full((1, 6), 1.1))


@pytest.mark.parametrize("arm", experiment.ARMS)
@pytest.mark.parametrize("budget", experiment.BUDGETS)
def test_budget_and_selected_chain_are_traced(arm, budget):
    result = experiment.run_one(experiment.tasks()[0], 0, budget, arm)
    assert result["used"] == budget
    assert sum(len(item["actions"]) for item in result["trace"]) == budget
    selected = np.asarray(result["selected"]["chain"])
    assert selected.shape == (15, 2)
    assert any(np.any(np.all(np.asarray(item["actions"]) == selected, axis=(1, 2)))
               for item in result["trace"])
    if result["incumbent"]:
        assert result["selected"]["score"] <= result["incumbent"]["score"]


def test_ridge_and_svd_are_equivalent():
    task = experiment.tasks()[0]
    svd = experiment.run_one(task, 0, 384, "svd_response")
    ridge = experiment.run_one(task, 0, 384, "ridge_response")
    np.testing.assert_allclose([x["parameters"] for x in svd["bank"]],
                               [x["parameters"] for x in ridge["bank"]], atol=1e-11)


def test_sealed_outputs_if_available():
    if not experiment.INTERVENTION.exists():
        pytest.skip("development panel not yet completed")
    plans = json.loads(experiment.PLANS.read_text())
    audit = json.loads(experiment.AUDIT.read_text())
    interventions = json.loads(experiment.INTERVENTION.read_text())
    assert len(plans["rows"]) == len(audit["rows"]) == 24 * 2 * 5
    assert interventions["plans_sha256"] == audit["plans_sha256"] == experiment.sha(experiment.PLANS)
    assert interventions["audit_sha256"] == experiment.sha(experiment.AUDIT)
    original = {(x["root"], x["budget"]): x for x in plans["rows"] if x["arm"] == "svd_response"}
    assert len(interventions["rows"]) == 48
    for row in interventions["rows"]:
        source = original[row["root"], row["budget"]]
        chain = np.asarray(row["chain"])
        task = source["task"]
        goal = np.asarray(task["goal"])
        np.testing.assert_array_equal(chain, source["selected"]["chain"])
        greedy = int(np.argmin(source["prefix_scores"]))
        prefixes = {"selected": chain[:5], "zero": np.zeros((5, 2)),
                    "greedy": np.asarray(source["bank"][greedy]["chain"][:5])}
        suffixes = [np.tile(x, (10, 1)) for x in
                    ((0., 0.), (.5, 0.), (-.5, 0.), (.5, .5), (.5, -.5))]
        suffixes.append(chain[5:])
        controls = row["controls"]
        for name, prefix in prefixes.items():
            control = controls[name]
            np.testing.assert_array_equal(control["prefix"], prefix)
            distance = np.linalg.norm(experiment.exact.endpoints(task, prefix[None])[0] - goal)
            np.testing.assert_allclose(control["prefix_distance"], distance, atol=1e-12)
            actions = np.stack([np.concatenate((prefix, suffix)) for suffix in suffixes])
            scores = np.linalg.norm(experiment.exact.endpoints(task, actions) - goal, axis=1)
            np.testing.assert_allclose(control["common_scores"], scores, atol=1e-12)
            np.testing.assert_allclose(control["active_changes"], scores - scores[0], atol=1e-12)
            best, score, values = experiment.optimize(
                prefix, 15, 120, np.random.default_rng(2026120700 + row["root"]),
                lambda actions: np.linalg.norm(experiment.exact.endpoints(task, actions) - goal, axis=1))
            np.testing.assert_allclose(control["own_scores"], values, atol=1e-12)
            np.testing.assert_allclose(control["own_chain"], best, atol=1e-12)
            np.testing.assert_allclose(control["own_score"], score, atol=1e-12)
            np.testing.assert_allclose(score, min(values), atol=1e-12)
        selected = controls["selected"]
        initial = np.linalg.norm(np.asarray(task["start"])[:2] - goal)
        assert row["nonprogress"] == (selected["prefix_distance"] >= initial - 1e-10)
        assert row["active_response"] == all(
            selected["active_changes"][-1] < controls[name]["active_changes"][-1] - 1e-10
            for name in ("zero", "greedy"))
        assert row["own_plan_advantage"] == all(
            selected["own_score"] < controls[name]["own_score"] - 1e-10
            for name in ("zero", "greedy"))
        assert row["limited_witness"] == (
            row["nonprogress"] and row["active_response"] and row["own_plan_advantage"])
