import json

import numpy as np
import pytest

from experiments import pc_successor_floor_cem_exact_mechanism_dev_20261103_codex as diag


def test_all_final_chain_interventions_replay():
    if not diag.OUTPUT.exists():
        pytest.skip("mechanism artifact unavailable")
    data = json.loads(diag.OUTPUT.read_text())
    plans = json.loads(diag.experiment.PLANS.read_text())
    assert all(diag.experiment.prior.sha(p) == h for p, h in data["sources"].items())
    assert len(data["rows"]) == 24
    for row in data["rows"]:
        source = next(x for x in plans["rows"] if x["root"] == row["root"] and x["arm"] == "successor")
        assert row["chain"] == source["result"]["selected"]["chain"]
        goal = np.asarray(row["task"]["goal"])
        chain = np.asarray(row["chain"])
        np.testing.assert_allclose(row["initial_distance"], np.linalg.norm(np.asarray(row["task"]["start"])[:2] - goal), atol=1e-12, rtol=0)
        first = np.asarray([x["prefix"] for x in source["result"]["first_bank"]])
        first_values = np.linalg.norm(diag.experiment.prior.exact.endpoints(row["task"], first) - goal, axis=1)
        np.testing.assert_allclose(first_values, row["first_bank_distances"], atol=1e-12, rtol=0)
        prefixes = {"selected": chain[:5], "zero": np.zeros((5, 2)), "greedy": first[np.argmin(first_values)]}
        for name, control in row["controls"].items():
            np.testing.assert_array_equal(control["prefix"], prefixes[name])
            prefix_end = diag.experiment.prior.exact.endpoints(row["task"], prefixes[name][None])[0]
            np.testing.assert_allclose(control["prefix_distance"], np.linalg.norm(prefix_end - goal), atol=1e-12, rtol=0)
            actions = np.asarray(control["common_actions"])
            np.testing.assert_array_equal(actions[:, :5], np.broadcast_to(prefixes[name], (6, 5, 2)))
            values = np.linalg.norm(diag.experiment.prior.exact.endpoints(row["task"], actions) - goal, axis=1)
            np.testing.assert_allclose(values, control["common_scores"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(values - values[0], control["active_changes"], atol=1e-12, rtol=0)
            searched = np.concatenate([np.asarray(b["actions"]) for b in control["own_trace"]])
            scores = np.concatenate([np.asarray(b["scores"]) for b in control["own_trace"]])
            assert searched.shape == (120, 15, 2)
            np.testing.assert_array_equal(searched[:, :5], np.broadcast_to(prefixes[name], (120, 5, 2)))
            actual = np.linalg.norm(diag.experiment.prior.exact.endpoints(row["task"], searched) - goal, axis=1)
            np.testing.assert_allclose(scores, actual, atol=1e-12, rtol=0)
            idx = np.argmin(scores)
            assert control["own_score"] == scores[idx]
            np.testing.assert_array_equal(control["own_chain"], searched[idx])
        c = row["controls"]
        assert row["nonprogress"] == (c["selected"]["prefix_distance"] >= row["initial_distance"] - 1e-10)
        assert row["active_response"] == all(c["selected"]["active_changes"][-1] < c[k]["active_changes"][-1] - 1e-10 for k in ("zero", "greedy"))
        assert row["own_advantage"] == all(c["selected"]["own_score"] < c[k]["own_score"] - 1e-10 for k in ("zero", "greedy"))
        assert row["limited_witness"] == (row["nonprogress"] and row["active_response"] and row["own_advantage"])
