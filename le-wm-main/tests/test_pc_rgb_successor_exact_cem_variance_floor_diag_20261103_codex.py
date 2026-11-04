import json

import numpy as np
import pytest

from experiments import pc_rgb_successor_exact_cem_variance_floor_diag_20261103_codex as diag


def test_intervention_artifacts_and_all_scored_chains():
    if not diag.PLANS.exists() or not diag.AUDIT.exists():
        pytest.skip("intervention artifacts unavailable")
    result = json.loads(diag.PLANS.read_text())
    original = json.loads(diag.source.PLANS.read_text())
    receipt = json.loads(diag.AUDIT.read_text())
    assert result["dependencies"] == diag.dependencies()
    assert receipt == {"plans_sha256": diag.source.prior.sha(diag.PLANS), "replayed_searches": 144}
    assert len(result["rows"]) == len(original["rows"]) == 24
    for row, old in zip(result["rows"], original["rows"], strict=True):
        assert (row["root"], row["task"]) == (old["root"], old["task"])
        assert len(row["runs"]) == len(old["runs"]) == 6
        for run, old_run in zip(row["runs"], old["runs"], strict=True):
            assert (run["name"], run["prefix"]) == (old_run["name"], old_run["prefix"])
            prefix = np.asarray(run["prefix"])
            actions = np.concatenate([np.asarray(b["actions"]) for b in run["trace"]])
            scores = np.concatenate([np.asarray(b["scores"]) for b in run["trace"]])
            assert actions.shape == (384, 15, 2)
            assert len(run["trace"]) == 32
            assert np.max(np.abs(actions)) <= 1
            np.testing.assert_array_equal(actions[:, :len(prefix)], np.broadcast_to(prefix, actions[:, :len(prefix)].shape))
            expected = np.linalg.norm(diag.source.prior.exact.endpoints(row["task"], actions) - row["task"]["goal"], axis=1)
            np.testing.assert_allclose(scores, expected, atol=1e-12, rtol=0)
            for n in (192, 384):
                index = np.argmin(scores[:n])
                assert run["checkpoints"][str(n)]["score"] == scores[index]
                np.testing.assert_array_equal(run["checkpoints"][str(n)]["chain"], actions[index])
