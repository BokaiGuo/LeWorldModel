import json

import numpy as np
import pytest

from experiments import pc_response_diverse_beam_same_chain_diag_20261103_codex as diag


def test_all_interventions_are_bound_and_replayed():
    if not diag.OUTPUT.exists():
        pytest.skip("mechanism diagnostic unavailable")
    data = json.loads(diag.OUTPUT.read_text())
    plans = json.loads(diag.exp.PLANS.read_text())
    previous = json.loads(diag.exp.previous.PLANS.read_text())
    assert data["dependencies"] == diag.dependencies()
    assert len(data["rows"]) == 24
    for row in data["rows"]:
        root, task = row["root"], row["task"]
        source = next(x for x in plans["rows"] if x["root"] == root and x["arm"] == "response_diverse")
        assert row["task"] == source["task"]
        assert row["chain"] == source["result"]["selected"]["chain"]
        goal, chain = np.asarray(task["goal"]), np.asarray(row["chain"])
        first = np.asarray([x["prefix"] for x in source["result"]["first_bank"]])
        scores = np.linalg.norm(diag.exp.prior.exact.endpoints(task, first.astype(np.float32)) - goal, axis=1)
        np.testing.assert_allclose(scores, row["first_scores"], atol=1e-12, rtol=0)
        expected = {"selected": chain[:5], "zero": np.zeros((5, 2)), "greedy": first[np.argmin(scores)]}
        for arm in ("random_bridge", "lookahead"):
            prior = next(x for x in previous["rows"] if x["root"] == root and x["arm"] == arm)
            assert prior["task"] == task
            expected[arm] = np.asarray(prior["result"]["selected"]["chain"])[:5]
        tails = [np.tile(x, (10, 1)) for x in ((0., 0.), (.5, 0.), (-.5, 0.), (.5, .5), (.5, -.5))]
        tails.append(chain[5:])
        for name, control in row["controls"].items():
            np.testing.assert_array_equal(control["prefix"], expected[name])
            prefix = expected[name]
            common = np.asarray([np.concatenate((prefix, tail)) for tail in tails])
            np.testing.assert_array_equal(common, control["common_actions"])
            points = diag.exp.prior.exact.endpoints(task, common.astype(np.float32))
            common_scores = np.linalg.norm(points - goal, axis=1)
            np.testing.assert_allclose(common_scores, control["common_scores"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(common_scores - common_scores[0], control["active_changes"], atol=1e-12, rtol=0)
            end = diag.exp.prior.exact.endpoints(task, prefix[None].astype(np.float32))[0]
            np.testing.assert_allclose(np.linalg.norm(end - goal), control["prefix_distance"], atol=1e-12, rtol=0)
            requested = np.concatenate([np.asarray(x["requested"]) for x in control["own_trace"]])
            model = np.concatenate([np.asarray(x["model"], dtype=np.float32) for x in control["own_trace"]])
            values = np.concatenate([np.asarray(x["scores"]) for x in control["own_trace"]])
            assert requested.shape == (120, 15, 2)
            np.testing.assert_array_equal(model, requested.astype(np.float32))
            np.testing.assert_array_equal(requested[:, :5], np.broadcast_to(prefix, (120, 5, 2)))
            exact = np.linalg.norm(diag.exp.prior.exact.endpoints(task, model) - goal, axis=1)
            np.testing.assert_allclose(values, exact, atol=1e-12, rtol=0)
            best = np.argmin(values)
            assert control["own_score"] == values[best]
            np.testing.assert_array_equal(control["own_chain"], requested[best])
        c = row["controls"]
        initial = np.linalg.norm(np.asarray(task["start"])[:2] - goal)
        np.testing.assert_allclose(initial, row["initial_distance"], atol=1e-12, rtol=0)
        assert row["nonprogress"] == (c["selected"]["prefix_distance"] >= initial - 1e-10)
        for name in diag.CONTROL_NAMES:
            assert row["active_better"][name] == (c["selected"]["active_changes"][-1] < c[name]["active_changes"][-1] - 1e-10)
            assert row["own_better"][name] == (c["selected"]["own_score"] < c[name]["own_score"] - 1e-10)
        assert row["base_intersection"] == (row["nonprogress"] and all(row["active_better"][x] and row["own_better"][x] for x in ("zero", "greedy")))
        assert row["all_controls_intersection"] == (row["nonprogress"] and all(row["active_better"][x] and row["own_better"][x] for x in diag.CONTROL_NAMES))
