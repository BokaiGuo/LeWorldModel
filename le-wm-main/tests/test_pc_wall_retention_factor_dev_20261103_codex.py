import json

import numpy as np

from experiments.spectral_reachability import pc_wall_retention_factor_dev_20261103_codex as exp


def test_diversity_is_deterministic_and_contains_greedy():
    task = exp.prior.base.tasks()[0]
    screen = exp.prior.screen(task, 0)
    retained = exp.diverse(task, screen)
    assert retained == exp.diverse(task, screen)
    assert retained[0] == screen["same_bank_greedy"]
    assert len(set(retained)) == 8


def test_stratified_budget_and_replay():
    task = exp.prior.base.tasks()[0]
    screen = exp.prior.screen(task, 0)
    prefix = exp.prior.prefix_bank()[0]
    plan = exp.stratified(task, 0, prefix, screen["probe_costs"][0], 128)
    assert sum(len(x["scores"]) for x in plan["trace"]) == 128
    assert plan["terminal_distance"] <= min(screen["probe_costs"][0])
    np.testing.assert_allclose(exp.prior.evaluate(task, prefix, np.asarray(plan["suffix"])[None])[0],
                               plan["terminal_distance"], atol=1e-12)


def test_saved_seal_and_actual_selected_chains():
    if not exp.OUTPUT.exists():
        return
    report, seal, old = json.loads(exp.OUTPUT.read_text()), json.loads(exp.SEAL.read_text()), exp.load_prior()
    assert report["seal_sha256"] == exp.prior.base.digest(exp.SEAL)
    assert len(report["rows"]) == len(seal["rows"]) == 8
    for row, frozen, original in zip(report["rows"], seal["rows"], old["rows"], strict=True):
        bank = exp.prior.prefix_bank()
        for budget, oldbudget in zip(row["budgets"], original["budgets"], strict=True):
            for name, arm in budget["arms"].items():
                retention, optimizer = name.rsplit("_", 1)
                assert arm["retained"] == frozen["retained"][retention]
                assert arm["selected_index"] in arm["retained"]
                plans = oldbudget["census"] if optimizer == "original" else budget["stratified_plans"]
                lookup = lambda i: plans[i] if optimizer == "original" else plans[str(i)]
                assert arm["selected_index"] == min(arm["retained"],
                                                     key=lambda i: (lookup(i)["terminal_distance"], i))
                actions = np.asarray(arm["actions"])
                np.testing.assert_array_equal(actions[:6], bank[arm["selected_index"]])
                cost = exp.prior.evaluate(original["task"], actions[:6], actions[None, 6:])[0]
                np.testing.assert_allclose(cost, arm["terminal_distance"], atol=1e-12)
                assert arm["screening_calls"] + arm["continuation_calls"] == (
                    33 + (33 * 11 if retention == "lookahead" else 8 * 11) + 8 * budget["cap"])
