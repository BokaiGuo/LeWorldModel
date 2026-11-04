import json

import numpy as np
import pytest

from experiments import pc_response_diverse_fixed_terminal_exact_dev_20261103_codex as experiment


def test_sealed_feedback_realized_trajectory_and_incumbents():
    if not experiment.PLANS.exists() or not experiment.AUDIT.exists():
        pytest.skip("feedback artifacts unavailable")
    data = json.loads(experiment.PLANS.read_text())
    audit = json.loads(experiment.AUDIT.read_text())
    original = json.loads(experiment.prior.PLANS.read_text())
    assert data["dependencies"] == experiment.dependencies()
    assert audit["plans_sha256"] == experiment.prior.prior.sha(experiment.PLANS)
    assert len(data["rows"]) == len(audit["rows"]) == 48
    for row in data["rows"]:
        old = next(x for x in original["rows"] if x["root"] == row["root"] and x["arm"] == row["arm"])
        assert row["task"] == old["task"]
        assert row["stages"][0]["selected"] == old["result"]["selected"]
        actions = np.asarray([a for stage in row["stages"] for a in stage["dispatched"]], dtype=np.float32)
        assert actions.shape == (15, 2)
        end = experiment.prior.prior.exact.endpoints(row["task"], actions[None])[0]
        np.testing.assert_allclose(row["final_distance"], np.linalg.norm(end - row["task"]["goal"]), atol=1e-12, rtol=0)
        np.testing.assert_allclose(row["final_distance"], row["stages"][-1]["result"]["selected"]["terminal_score"],
                                   atol=1e-12, rtol=0)
        for index in (1, 2):
            stage = row["stages"][index]
            prev = row["stages"][index - 1]
            prior_selected = prev.get("selected") if index == 1 else prev["result"]["selected"]
            np.testing.assert_array_equal(stage["result"]["carry"], prior_selected["chain"][5:])
            assert sum(batch["tokens"] for batch in stage["trace"]) == stage["tokens"] <= 480
            scores = [score for batch in stage["trace"] if len(batch["requested"][0]) == stage["horizon"]
                      for score in batch["scores"]]
            np.testing.assert_allclose(stage["result"]["selected"]["terminal_score"], min(scores), atol=1e-12, rtol=0)
