import json

import numpy as np

from experiments import pc_multidirection_fixed_terminal_feedback_dev_20261103_codex as exp


def test_feedback_recreates_sealed_decisions():
    data, audit = json.loads(exp.PLANS.read_text()), json.loads(exp.AUDIT.read_text())
    assert data["dependencies"] == exp.dependencies()
    assert audit["plans_sha256"] == exp.sha(exp.PLANS)
    assert len(data["rows"]) == 24
    for row in data["rows"]:
        previous_chain = np.asarray(row["stages"][0]["selected"]["chain"])
        for step in (1, 2):
            stage = row["stages"][step]
            meter = exp.prior.previous.prior.Budget()
            trace = iter(stage["trace"])
            local = {"start": stage["start"], "goal": row["task"]["goal"]}

            def predict(actions):
                batch = next(trace)
                np.testing.assert_array_equal(actions, batch["requested"])
                positions, scores = meter.predict(local, actions)
                np.testing.assert_allclose(positions, batch["positions"], atol=1e-12, rtol=0)
                np.testing.assert_allclose(scores, batch["scores"], atol=1e-12, rtol=0)
                return positions, scores

            result = exp.feedback.decide(row["root"], step, row["task"]["goal"], predict, meter, previous_chain[5:])
            result.pop("structure_seconds")
            expected = dict(stage["result"])
            expected.pop("structure_seconds")
            assert result == expected
            assert next(trace, None) is None
            assert meter.token_samples == stage["tokens"] <= 480
            previous_chain = np.asarray(result["selected"]["chain"])
