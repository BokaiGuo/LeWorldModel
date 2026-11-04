import json

import numpy as np

from experiments import pc_response_feedback_executed_tail_intervention_20261103_codex as run


def test_sealed_feedback_tail_interventions_replay():
    data = json.loads(run.OUTPUT.read_text())
    feedback = json.loads(run.feedback.PLANS.read_text())
    old = json.loads(run.diag.OUTPUT.read_text())
    assert data["dependencies"] == run.dependencies()
    assert len(data["rows"]) == 24
    for row in data["rows"]:
        source = next(r for r in feedback["rows"] if r["root"] == row["root"] and r["arm"] == "response_diverse")
        prior = next(r for r in old["rows"] if r["root"] == row["root"])
        assert source["task"] == row["task"] == prior["task"]
        chain = np.concatenate([np.asarray(s["dispatched"]) for s in source["stages"]])
        np.testing.assert_array_equal(chain, row["chain"])
        np.testing.assert_array_equal(chain[:5], prior["controls"]["selected"]["prefix"])
        goal = np.asarray(row["task"]["goal"])
        for name, c in row["controls"].items():
            previous = prior["controls"][name]
            np.testing.assert_array_equal(c["prefix"], previous["prefix"])
            prefix = np.asarray(c["prefix"])
            actions = np.asarray([np.concatenate((prefix, np.zeros((10, 2)))),
                                  np.concatenate((prefix, chain[5:]))], dtype=np.float32)
            scores = np.linalg.norm(run.diag.exp.prior.exact.endpoints(row["task"], actions) - goal, axis=1)
            np.testing.assert_allclose([c["zero_score"], c["executed_tail_score"]], scores, atol=1e-12, rtol=0)
            np.testing.assert_allclose(c["active_change"], scores[1] - scores[0], atol=1e-12, rtol=0)
            assert c["own_score"] == previous["own_score"]
        assert row["nonprogress"] == prior["nonprogress"]
        c = row["controls"]
        np.testing.assert_allclose(c["selected"]["executed_tail_score"], source["final_distance"], atol=1e-12, rtol=0)
        for name in run.diag.CONTROL_NAMES:
            assert row["active_better"][name] == (c["selected"]["active_change"] < c[name]["active_change"] - 1e-10)
            assert row["own_better"][name] == (c["selected"]["own_score"] < c[name]["own_score"] - 1e-10)
        assert row["base_intersection"] == (row["nonprogress"] and all(row["active_better"][n] and row["own_better"][n] for n in ("zero", "greedy")))
        assert row["all_controls_intersection"] == (row["nonprogress"] and all(row["active_better"][n] and row["own_better"][n] for n in run.diag.CONTROL_NAMES))
