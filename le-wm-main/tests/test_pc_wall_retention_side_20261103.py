import json
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import pc_wall_retention_side_20261103 as exp
from experiments.spectral_reachability import pc_wall_retention_coverage_audit_side_20261103 as audit


def test_bank_and_probe_contract():
    bank, probes = exp.prefix_bank(), exp.probe_family()
    assert bank.shape == (33, 6, 2)
    assert probes.shape == (11, 10, 2)
    assert np.max(np.abs(bank) / exp.base.HIGH) <= 1
    assert np.max(np.abs(probes) / exp.base.HIGH) <= 1
    assert not bank[0].any() and not probes[0].any()
    with pytest.raises(ValueError):
        exp.suffixes([[0., 1.1]])


def test_retention_fixed_before_optimization_and_probe_binding():
    task = exp.base.tasks()[0]
    screened = exp.screen(task, 0)
    assert all(len(x) == len(set(x)) == 8 for x in screened["retained"].values())
    for i, prefix in enumerate(exp.prefix_bank()):
        np.testing.assert_array_equal(exp.evaluate(task, prefix, exp.probe_family()),
                                      screened["probe_costs"][i])
        assert screened["probe_costs"][i][0] == screened["prefix_costs"][i]
    assert screened == exp.screen(task, 0)


def test_optimizer_paired_noise_incumbent_and_budget():
    task, bank = exp.base.tasks()[0], exp.prefix_bank()
    screened = exp.screen(task, 0)
    first = exp.optimize(task, 0, bank[0], screened["probe_costs"][0], 128)
    second = exp.optimize(task, 0, bank[1], screened["probe_costs"][1], 128)
    assert first["trace"][0]["parameters"] == second["trace"][0]["parameters"]
    assert sum(len(x["scores"]) for x in first["trace"]) == first["optimizer_calls"] == 128
    assert first["terminal_distance"] <= min(screened["probe_costs"][0])
    error = exp.evaluate(task, bank[0], np.asarray(first["suffix"])[None])[0]
    assert abs(error - first["terminal_distance"]) < 1e-12


def test_saved_full_census_binding_retention_and_mechanism():
    if not exp.OUTPUT.exists():
        pytest.skip("panel not yet executed")
    report = json.loads(exp.OUTPUT.read_text())
    for path, expected in report["dependencies"].items():
        assert exp.base.digest(Path(path)) == expected
    assert report["summary"] == exp.summary(report["rows"])
    assert len(report["rows"]) == 8
    plans, selected = 0, 0
    for row in report["rows"]:
        assert row["screen"] == exp.screen(row["task"], row["index"])
        assert json.loads(row["sealed_retention"]) == row["screen"]["retained"]
        bank = np.asarray(row["bank"])
        for budget in row["budgets"]:
            census = budget["census"]
            assert len(census) == 33
            for i, result in enumerate(census):
                cost = exp.evaluate(row["task"], bank[i], np.asarray(result["suffix"])[None])[0]
                assert abs(cost - result["terminal_distance"]) < 1e-12
                assert sum(len(x["scores"]) for x in result["trace"]) == budget["cap"]
                plans += 1
            for name, arm in budget["arms"].items():
                idx = arm["selected_index"]
                retained = row["screen"]["retained"][name]
                assert idx in retained
                assert arm["terminal_distance"] == min(census[i]["terminal_distance"] for i in retained)
                assert arm["intervention"] == exp.intervention(
                    row["task"], bank, row["screen"], census, idx)
                full = np.concatenate((bank[idx], census[idx]["suffix"]))
                np.testing.assert_array_equal(full, arm["actions"])
                trajectory, blocked = exp.base.rollout(row["task"]["start"], full[None])
                np.testing.assert_array_equal(trajectory[0], arm["trajectory"])
                np.testing.assert_array_equal(blocked[0], arm["blocked"])
                assert arm["hypothetical_unique_planner_calls"] == (
                    33 + (363 if name == "lookahead" else 88) + 8 * budget["cap"])
                selected += 1
    assert plans == 528 and selected == 48


def test_privileged_coverage_artifact_replay_separate_from_planner():
    if not audit.OUTPUT.exists():
        pytest.skip("post-outcome coverage diagnostic not run")
    report = json.loads(audit.OUTPUT.read_text())
    for path, digest in report["dependencies"].items():
        assert exp.base.digest(Path(path)) == digest
    assert report["rows"] == audit.run()
    assert report["P1_admission"] is False
    for row, task in zip(report["rows"], exp.base.tasks(), strict=True):
        for item in row["witnesses"]:
            if item["constructed"]:
                trajectory, blocked = exp.base.rollout(task["start"], [item["actions"]])
                np.testing.assert_array_equal(trajectory[0], item["trajectory"])
                np.testing.assert_array_equal(blocked[0], item["blocked"])
                assert item["terminal_distance"] == pytest.approx(
                    np.linalg.norm(trajectory[0, -1, :2] - task["goal"]), abs=1e-12)
