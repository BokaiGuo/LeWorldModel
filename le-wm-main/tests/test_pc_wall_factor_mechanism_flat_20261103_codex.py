import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import pc_wall_factor_same_chain_mechanism_20261103_codex as mech
from experiments.spectral_reachability import pc_wall_factor_flat_budget_dev_20261103_codex as flat
from experiments.spectral_reachability import pc_wall_factor_joint_template_dev_20261103_codex as joint


def test_mechanism_selected_chain_and_controls_replay():
    report = json.loads(mech.OUTPUT.read_text())
    prior = mech.factor.load_prior()
    selected = json.loads(mech.factor.OUTPUT.read_text())
    assert len(report["rows"]) == 128
    assert all(mech.prior.base.digest(Path(p)) == digest for p, digest in report["dependencies"].items())
    for row in report["rows"]:
        task = prior["rows"][row["task"]]["task"]
        budget = next(b for b in selected["rows"][row["task"]]["budgets"] if b["cap"] == row["cap"])
        assert row["selected_actions"] == budget["arms"][row["arm"]]["actions"]
        chain = np.asarray(row["selected_actions"])
        tails = np.concatenate((mech.prior.probe_family(), chain[None, 6:]), axis=0)
        for control in row["controls"].values():
            actual = mech.prior.evaluate(task, np.asarray(control["prefix"]), tails)
            np.testing.assert_allclose(actual, control["common_suffix_distances"], atol=1e-12)
            np.testing.assert_allclose(actual - actual[0], control["relative_to_zero_suffix"], atol=1e-12)
            own = mech.prior.evaluate(task, np.asarray(control["prefix"]),
                                      np.asarray(control["own_optimized_suffix"])[None])[0]
            np.testing.assert_allclose(own, control["own_planned_distance"], atol=1e-12)
        assert row["limited_witness"] == (row["nonprogress"] and row["active_response"]
                                          and row["own_plan_advantage"])


def test_flat_budget_and_replay():
    report = json.loads(flat.OUTPUT.read_text())
    prior = flat.factor.load_prior()
    assert len(report["rows"]) == 16
    assert all(flat.base.digest(Path(p)) == digest for p, digest in report["dependencies"].items())
    for row in report["rows"]:
        task = prior["rows"][row["task"]]["task"]
        plan = row["flat"]
        assert plan["total_calls"] == row["rollout_budget"]
        trace, _ = flat.base.rollout(task["start"], [plan["actions"]])
        np.testing.assert_allclose(trace[0], plan["trajectory"], atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(trace[0, -1, :2] - task["goal"]),
                                   plan["terminal_distance"], atol=1e-12)


def test_joint_template_budget_and_replay():
    report = json.loads(joint.OUTPUT.read_text())
    prior = joint.factor.load_prior()
    assert len(report["rows"]) == 16
    assert all(joint.base.digest(Path(p)) == digest for p, digest in report["dependencies"].items())
    for row in report["rows"]:
        task = prior["rows"][row["task"]]["task"]
        plan = row["joint"]
        assert plan["rollout_calls"] == row["budget"]
        assert 1 + sum(len(step["scores"]) for step in plan["trace"]) == row["budget"]
        np.testing.assert_array_equal(joint.decode([plan["parameters"]])[0], plan["actions"])
        trajectory, _ = joint.base.rollout(task["start"], [plan["actions"]])
        np.testing.assert_allclose(trajectory[0], plan["trajectory"], atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(trajectory[0, -1, :2] - task["goal"]),
                                   plan["distance"], atol=1e-12)
