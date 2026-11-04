import json

import numpy as np

from experiments import pc_fixed_prefix_multistart_feedback_diagnostic_20261103_codex as exp


def test_sealed_multistart_sampling_and_budget():
    data, audit = json.loads(exp.PLANS.read_text()), json.loads(exp.AUDIT.read_text())
    assert data["dependencies"] == exp.dependencies()
    assert audit["plans_sha256"] == exp.base.sha(exp.PLANS)
    assert len(data["rows"]) == 120
    for row in data["rows"]:
        for stage, horizon, counts, expected_chain, expected_score in (
            (0, 10, (60, 60, 60, 60), row["ten_chain"], row["ten_score"]),
            (1, 5, (120, 120, 120, 119), row["five_chain"], row["five_score"]),
        ):
            batches = iter(row["traces"][stage][1:] if stage else row["traces"][stage])

            def verify(actions, restart):
                batch = next(batches)
                assert batch["restart"] == restart
                np.testing.assert_array_equal(actions, batch["requested"])
                return np.asarray(batch["scores"])

            chain, score, starts = exp.optimize(row["root"], stage, horizon, counts, verify)
            np.testing.assert_array_equal(starts, row["starts"][stage])
            np.testing.assert_array_equal(chain, expected_chain)
            np.testing.assert_allclose(score, expected_score, atol=1e-12, rtol=0)
            assert next(batches, None) is None
            assert sum(b["tokens"] for b in row["traces"][stage]) == 480
