import json

import numpy as np
import pytest

from experiments import pc_rgb_successor_exact_same_policy_dev_20261103_codex as exp


def test_exact_budget_caps_and_counts():
    meter = exp.Budget()
    task = exp.panel.tasks()[0]
    meter.predict(task, np.zeros((160, 15, 2)))
    assert meter.token_samples == 480
    with pytest.raises(ValueError, match="cap"):
        meter.predict(task, np.zeros((1, 5, 2)))


def test_sealed_same_chain_interventions():
    if not exp.INTERVENTION.exists():
        pytest.skip("development result unavailable")
    plans = json.loads(exp.PLANS.read_text())
    audit = json.loads(exp.AUDIT.read_text())
    interventions = json.loads(exp.INTERVENTION.read_text())
    assert plans["dependencies"] == exp.dependencies()
    assert audit["plans_sha256"] == interventions["plans_sha256"] == exp.sha(exp.PLANS)
    assert interventions["audit_sha256"] == exp.sha(exp.AUDIT)
    assert len(plans["rows"]) == len(interventions["rows"]) == 24
    for plan, row in zip(plans["rows"], interventions["rows"], strict=True):
        task = plan["task"]
        goal = np.asarray(task["goal"])
        chain = np.asarray(plan["result"]["selected"]["chain"])
        np.testing.assert_array_equal(chain, row["chain"])
        first = np.asarray([x["prefix"] for x in plan["result"]["first_bank"]])
        distances = np.linalg.norm(exp.exact.endpoints(task, first) - goal, axis=1)
        np.testing.assert_allclose(distances, row["first_bank_distances"], atol=1e-12)
        prefixes = {"selected": chain[:5], "zero": np.zeros((5, 2)),
                    "greedy": first[int(np.argmin(distances))]}
        tails = [np.tile(x, (10, 1)) for x in ((0., 0.), (.5, 0.), (-.5, 0.), (.5, .5), (.5, -.5))]
        tails.append(chain[5:])
        for name, prefix in prefixes.items():
            control = row["controls"][name]
            np.testing.assert_array_equal(control["prefix"], prefix)
            actions = np.asarray([np.concatenate((prefix, tail)) for tail in tails])
            common = np.linalg.norm(exp.exact.endpoints(task, actions) - goal, axis=1)
            np.testing.assert_allclose(common, control["common_scores"], atol=1e-12)
            np.testing.assert_allclose(common - common[0], control["active_changes"], atol=1e-12)
            best, score, values = exp.optimize(
                prefix, 15, 120, np.random.default_rng(2026121000 + row["root"]),
                lambda actions: np.linalg.norm(exp.exact.endpoints(task, actions) - goal, axis=1))
            np.testing.assert_allclose(values, control["own_scores"], atol=1e-12)
            np.testing.assert_allclose(best, control["own_chain"], atol=1e-12)
            np.testing.assert_allclose(score, control["own_score"], atol=1e-12)
        sel = row["controls"]["selected"]
        assert row["nonprogress"] == (sel["prefix_distance"] >= row["initial_distance"] - 1e-10)
        assert row["active_response"] == all(sel["active_changes"][-1] <
            row["controls"][k]["active_changes"][-1] - 1e-10 for k in ("zero", "greedy"))
        assert row["own_advantage"] == all(sel["own_score"] <
            row["controls"][k]["own_score"] - 1e-10 for k in ("zero", "greedy"))
        assert row["limited_witness"] == (row["nonprogress"] and row["active_response"] and row["own_advantage"])
