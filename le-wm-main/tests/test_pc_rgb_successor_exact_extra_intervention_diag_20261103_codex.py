import json

import numpy as np
import pytest

from experiments import pc_rgb_successor_exact_extra_intervention_diag_20261103_codex as diag


def test_saved_extra_interventions_are_on_final_chains():
    if not diag.OUTPUT.exists():
        pytest.skip("diagnostic unavailable")
    result = json.loads(diag.OUTPUT.read_text())
    plans = json.loads(diag.prior.PLANS.read_text())
    controls = json.loads(diag.controls.PLANS.read_text())
    assert result["protocol_sha256"] == diag.prior.sha(diag.PROTOCOL)
    assert result["runner_sha256"] == diag.prior.sha(diag.Path(diag.__file__))
    assert result["plans_sha256"] == diag.prior.sha(diag.prior.PLANS)
    assert result["control_plans_sha256"] == diag.prior.sha(diag.controls.PLANS)
    index = {(x["root"], x["arm"]): x for x in controls["rows"]}
    assert len(result["rows"]) == len(plans["rows"]) == 24
    for row, plan in zip(result["rows"], plans["rows"], strict=True):
        root, task = row["root"], plan["task"]
        chain = np.asarray(plan["result"]["selected"]["chain"])
        np.testing.assert_array_equal(row["selected_chain"], chain)
        prefixes = {"successor": chain[:5]}
        for arm in ("random_bridge", "lookahead"):
            prefixes[arm] = np.asarray(index[root, arm]["selected"]["chain"])[:5]
        tails = [np.tile(x, (10, 1)) for x in ((0., 0.), (.5, 0.), (-.5, 0.), (.5, .5), (.5, -.5))]
        tails.append(chain[5:])
        for name, prefix in prefixes.items():
            actual = row["measurements"][name]
            np.testing.assert_array_equal(actual["prefix"], prefix)
            actions = np.asarray([np.concatenate((prefix, tail)) for tail in tails])
            values = np.linalg.norm(diag.prior.exact.endpoints(task, actions) - task["goal"], axis=1)
            np.testing.assert_allclose(actual["common_scores"], values, atol=1e-12)
            np.testing.assert_allclose(actual["relative_changes"], values - values[0], atol=1e-12)
            best, score, scores = diag.controls.optimize(
                prefix, 15, 120, np.random.default_rng(2026121400 + root),
                lambda a: np.linalg.norm(diag.prior.exact.endpoints(task, a) - task["goal"], axis=1))
            np.testing.assert_allclose(actual["own_scores"], scores, atol=1e-12)
            np.testing.assert_allclose(actual["own_chain"], best, atol=1e-12)
            np.testing.assert_allclose(actual["own_score"], score, atol=1e-12)
