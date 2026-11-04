import json

import numpy as np

from experiments import pc_multidirection_successor_same_chain_diag_20261103_codex as diag


def test_replay_sealed_mechanism_and_controls():
    data = json.loads(diag.OUTPUT.read_text())
    plans = json.loads(diag.exp.PLANS.read_text())
    previous = json.loads(diag.exp.previous.previous.PLANS.read_text())
    assert data["dependencies"] == diag.dependencies()
    assert len(data["rows"]) == 24
    for row in data["rows"]:
        root, task = row["root"], row["task"]
        source = next(r for r in plans["rows"] if r["root"] == root)
        assert task == source["task"]
        chain = np.asarray(source["result"]["selected"]["chain"])
        np.testing.assert_array_equal(row["chain"], chain)
        first = np.asarray([b["prefix"] for b in source["result"]["first_bank"]])
        goal = np.asarray(task["goal"])
        values = np.linalg.norm(diag.exp.previous.prior.exact.endpoints(task, first.astype(np.float32)) - goal, axis=1)
        expected = {"selected": chain[:5], "zero": np.zeros((5, 2)), "greedy": first[np.argmin(values)]}
        for name in ("random_bridge", "lookahead"):
            old = next(r for r in previous["rows"] if r["root"] == root and r["arm"] == name)
            assert old["task"] == task
            expected[name] = np.asarray(old["result"]["selected"]["chain"])[:5]
        tails = [np.tile(x, (10, 1)) for x in ((0., 0.), (.5, 0.), (-.5, 0.), (.5, .5), (.5, -.5))]
        tails.append(chain[5:])
        matrices = {}
        for name, control in row["controls"].items():
            prefix = expected[name]
            np.testing.assert_array_equal(control["prefix"], prefix)
            actions = np.asarray([np.concatenate((prefix, tail)) for tail in tails])
            np.testing.assert_array_equal(control["common_actions"], actions)
            scores = np.linalg.norm(diag.exp.previous.prior.exact.endpoints(task, actions.astype(np.float32)) - goal, axis=1)
            np.testing.assert_allclose(control["common_scores"], scores, atol=1e-12, rtol=0)
            np.testing.assert_allclose(control["active_changes"], scores - scores[0], atol=1e-12, rtol=0)
            requested = np.concatenate([np.asarray(b["requested"]) for b in control["own_trace"]])
            model = np.concatenate([np.asarray(b["model"], dtype=np.float32) for b in control["own_trace"]])
            observed = np.concatenate([np.asarray(b["scores"]) for b in control["own_trace"]])
            assert requested.shape == (120, 15, 2)
            np.testing.assert_array_equal(requested[:, :5], np.broadcast_to(prefix, (120, 5, 2)))
            np.testing.assert_array_equal(requested.astype(np.float32), model)
            recomputed = np.linalg.norm(diag.exp.previous.prior.exact.endpoints(task, model) - goal, axis=1)
            np.testing.assert_allclose(observed, recomputed, atol=1e-12, rtol=0)
            np.testing.assert_array_equal(control["own_chain"], requested[np.argmin(observed)])
            assert control["own_score"] == min(observed)
            matrices[name] = diag.response(task, prefix)
            np.testing.assert_allclose(control["successor_response"], matrices[name], atol=1e-12, rtol=0)
            np.testing.assert_allclose(control["response_singular_values"], np.linalg.svd(matrices[name], compute_uv=False), atol=1e-12, rtol=0)
        for name in diag.prior_diag.CONTROL_NAMES:
            np.testing.assert_allclose(row["response_difference_frobenius"][name],
                                       np.linalg.norm(matrices["selected"] - matrices[name]), atol=1e-12, rtol=0)
            assert row["active_better"][name] == (row["controls"]["selected"]["active_changes"][-1] < row["controls"][name]["active_changes"][-1] - 1e-10)
            assert row["own_better"][name] == (row["controls"]["selected"]["own_score"] < row["controls"][name]["own_score"] - 1e-10)
        initial = np.linalg.norm(np.asarray(task["start"])[:2] - goal)
        assert row["nonprogress"] == (row["controls"]["selected"]["prefix_distance"] >= initial - 1e-10)
        assert row["base_intersection"] == (row["nonprogress"] and all(row["active_better"][n] and row["own_better"][n] for n in ("zero", "greedy")))
        assert row["all_controls_intersection"] == (row["nonprogress"] and all(row["active_better"][n] and row["own_better"][n] for n in diag.prior_diag.CONTROL_NAMES))
