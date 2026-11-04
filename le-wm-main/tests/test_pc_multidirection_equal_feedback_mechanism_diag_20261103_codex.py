import json

import numpy as np

from experiments import pc_multidirection_equal_feedback_mechanism_diag_20261103_codex as exp


def test_feedback_sampling_and_sealed_prefixes():
    data, audit = json.loads(exp.PLANS.read_text()), json.loads(exp.AUDIT.read_text())
    source = json.loads(exp.source.OUTPUT.read_text())
    assert data["dependencies"] == exp.dependencies()
    assert audit["plans_sha256"] == exp.base.sha(exp.PLANS)
    assert len(data["rows"]) == len(audit["rows"]) == 120
    for row in data["rows"]:
        original = next(r for r in source["rows"] if r["root"] == row["root"])
        assert row["task"] == original["task"]
        np.testing.assert_array_equal(row["prefix"], original["controls"][row["name"]]["prefix"])
        for stage, horizon, count, seed, expected_chain, expected_score in (
            (0, 10, 240, 2027010100 + row["root"], row["ten_chain"], row["ten_score"]),
            (1, 5, 479, 2027011100 + row["root"], row["five_chain"], row["five_score"]),
        ):
            trace = iter(row["traces"][stage][1:] if stage else row["traces"][stage])

            def predict(actions):
                batch = next(trace)
                np.testing.assert_array_equal(actions, batch["requested"])
                return np.asarray(batch["scores"])

            chain, score, scores = exp.base.cem.optimize([], horizon, count, np.random.default_rng(seed), predict)
            np.testing.assert_array_equal(chain, expected_chain)
            np.testing.assert_allclose(score, expected_score, atol=1e-12, rtol=0)
            assert len(scores) == count and next(trace, None) is None
            assert sum(batch["tokens"] for batch in row["traces"][stage]) == 480
