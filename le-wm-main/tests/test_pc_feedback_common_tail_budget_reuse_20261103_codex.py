import json

import numpy as np

from experiments import pc_feedback_common_tail_budget_reuse_20261103_codex as run


def test_all_replacements_replayed_and_source_bound():
    result = json.loads(run.OUTPUT.read_text())
    tails = json.loads(run.tail.OUTPUT.read_text())
    original = json.loads(run.tail.diag.OUTPUT.read_text())
    assert result["dependencies"] == run.dependencies()
    assert len(result["rows"]) == 24
    for row in result["rows"]:
        context = next(r for r in tails["rows"] if r["root"] == row["root"])
        prior = next(r for r in original["rows"] if r["root"] == row["root"])
        chain, goal = np.asarray(context["chain"]), np.asarray(context["task"]["goal"])
        for name, item in row["controls"].items():
            record = prior["controls"][name]
            samples = np.concatenate([np.asarray(x["requested"]) for x in record["own_trace"]])
            assert samples.shape == (120, 15, 2)
            prefix = np.asarray(record["prefix"])
            np.testing.assert_array_equal(samples[:, :5], np.broadcast_to(prefix, (120, 5, 2)))
            points = run.tail.diag.exp.prior.exact.endpoints(context["task"], samples.astype(np.float32))
            scores = np.linalg.norm(points - goal, axis=1)
            shared = np.concatenate((prefix, chain[5:]))[None].astype(np.float32)
            shared_score = np.linalg.norm(run.tail.diag.exp.prior.exact.endpoints(context["task"], shared)[0] - goal)
            values = np.concatenate((scores[:119], [shared_score]))
            np.testing.assert_allclose(item["old_score"], scores.min(), atol=1e-12, rtol=0)
            np.testing.assert_allclose(item["shared_score"], shared_score, atol=1e-12, rtol=0)
            np.testing.assert_allclose(item["replacement_score"], values.min(), atol=1e-12, rtol=0)
            assert item["replacement_index"] == np.argmin(values)
            assert item["replacement_source"] == ("shared_tail" if np.argmin(values) == 119 else "recorded_cem")
        assert row["nonprogress"] == context["nonprogress"]
        assert row["active_better"] == context["active_better"]
        for name in run.tail.diag.CONTROL_NAMES:
            assert row["replacement_better"][name] == (
                row["controls"]["selected"]["replacement_score"] < row["controls"][name]["replacement_score"] - 1e-10)
        assert row["base_intersection"] == (row["nonprogress"] and all(
            row["active_better"][name] and row["replacement_better"][name] for name in ("zero", "greedy")))
