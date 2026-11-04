import json

import numpy as np

from experiments import pc_fixed_prefix_equal_feedback_optimizer_20261103_codex as exp


def test_sealed_traces_reproduce_actual_cem_sampling():
    data, audit = json.loads(exp.PLANS.read_text()), json.loads(exp.AUDIT.read_text())
    assert data["dependencies"] == exp.dependencies()
    assert audit["plans_sha256"] == exp.sha(exp.PLANS)
    assert len(data["rows"]) == len(audit["rows"]) == 120
    for row in data["rows"]:
        for stage, horizon, count, seed, expected_chain, expected_score in (
            (0, 10, 240, 2027010100 + row["root"], row["ten_chain"], row["ten_score"]),
            (1, 5, 479, 2027011100 + row["root"], row["five_chain"], row["five_score"]),
        ):
            trace = iter(row["traces"][stage][1:] if stage else row["traces"][stage])

            def verify(actions):
                batch = next(trace)
                np.testing.assert_array_equal(actions, batch["requested"])
                return np.asarray(batch["scores"])

            chain, score, scores = exp.cem.optimize([], horizon, count, np.random.default_rng(seed), verify)
            np.testing.assert_array_equal(chain, expected_chain)
            np.testing.assert_allclose(score, expected_score, atol=1e-12, rtol=0)
            assert len(scores) == count
            assert next(trace, None) is None
        assert sum(batch["tokens"] for batch in row["traces"][0]) == 480
        assert sum(batch["tokens"] for batch in row["traces"][1]) == 480
