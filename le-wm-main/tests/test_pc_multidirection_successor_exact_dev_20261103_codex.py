import json

import numpy as np

from experiments import pc_multidirection_successor_exact_dev_20261103_codex as exp


def test_full_compiler_replays_sealed_decisions():
    data, audit = json.loads(exp.PLANS.read_text()), json.loads(exp.AUDIT.read_text())
    assert data["dependencies"] == exp.dependencies()
    assert audit["plans_sha256"] == exp.sha(exp.PLANS)
    assert len(data["rows"]) == 24
    for row in data["rows"]:
        meter = exp.previous.prior.Budget()
        trace = iter(row["trace"])

        def predict(actions):
            batch = next(trace)
            np.testing.assert_array_equal(actions, batch["requested"])
            positions, scores = meter.predict(row["task"], actions)
            np.testing.assert_allclose(positions, batch["positions"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(scores, batch["scores"], atol=1e-12, rtol=0)
            return positions, scores

        result = exp.compiler.decide(row["root"], row["task"]["goal"], predict, meter)
        result.pop("structure_seconds")
        expected = dict(row["result"])
        expected.pop("structure_seconds")
        assert result == expected
        assert next(trace, None) is None
        assert meter.token_samples == row["tokens"] <= 480
        for successor in result["successor_response"]:
            rank = successor["response"]["retained_rank"]
            assert len(successor["bank"]) == 1 + 2 * rank
            assert len(successor["retained"]) == min(2, len(successor["bank"]))
