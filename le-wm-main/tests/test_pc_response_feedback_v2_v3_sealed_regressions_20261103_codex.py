import json

import numpy as np
import pytest

from experiments import pc_response_feedback_v2_sealed_context_regression_20261103_codex as v2
from experiments import pc_response_feedback_v3_selection_only_regression_20261103_codex as v3


def test_regression_receipts_and_actual_contexts():
    if not all(path.exists() for path in (v2.PLANS, v2.AUDIT, v3.PLANS, v3.AUDIT)):
        pytest.skip("sealed regression artifacts unavailable")
    source = json.loads(v3.prior.PLANS.read_text())
    original = {r["root"]: r for r in source["rows"] if r["arm"] == "response_diverse"}
    for module in (v2, v3):
        data, audit = json.loads(module.PLANS.read_text()), json.loads(module.AUDIT.read_text())
        sha = v3.base.sha
        assert data["dependencies"] == module.dependencies()
        assert audit["plans_sha256"] == sha(module.PLANS)
        assert len(data["rows"]) == len(audit["rows"]) == 24
        for row in data["rows"]:
            old = original[row["root"]]
            assert row["task"] == {"start": old["stages"][2]["start"], "goal": old["task"]["goal"]}
            assert row["carry"] == old["stages"][2]["result"]["carry"]
            scores = []
            for batch in row["trace"]:
                requested = np.asarray(batch["requested"])
                actions = np.asarray(batch["model"], dtype=np.float32)
                np.testing.assert_array_equal(requested.astype(np.float32), actions)
                end = v3.base.exact.endpoints(row["task"], actions)
                actual = np.linalg.norm(end - row["task"]["goal"], axis=1)
                np.testing.assert_allclose(actual, batch["scores"], atol=1e-12, rtol=0)
                scores.extend(actual.tolist())
            assert len(scores) == row["tokens"] == 480
            np.testing.assert_allclose(row["result"]["selected"]["terminal_score"], min(scores), atol=1e-12, rtol=0)
            selected = np.asarray(row["result"]["selected"]["chain"], dtype=np.float32)
            endpoint = v3.base.exact.endpoints(row["task"], selected[None])[0]
            np.testing.assert_allclose(row["actual_distance"], np.linalg.norm(endpoint - row["task"]["goal"]), atol=1e-12, rtol=0)
            if module is v3:
                assert row["trace"][:-1] == old["stages"][2]["trace"][:-1]
                np.testing.assert_array_equal(selected, np.asarray(row["old_selected"]["chain"], dtype=np.float32))
