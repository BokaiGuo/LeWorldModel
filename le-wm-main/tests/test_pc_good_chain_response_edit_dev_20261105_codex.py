import numpy as np
import pytest

from experiments import pc_good_chain_response_edit_dev_20261105_codex as run


@pytest.mark.parametrize("arm,expected", [
    ("response_fixed", 157), ("random_fixed", 97), ("cem_fixed", 84),
    ("response_total", 157), ("random_total", 157), ("cem_total", 157),
])
def test_budget_incumbent_and_boundary(arm, expected):
    seen = []
    parent = np.full((15, 2), .99, np.float32)
    parent[0, 0] = -1.

    def predict(actions):
        actions = np.asarray(actions)
        assert actions.dtype == np.float32 and np.abs(actions).max() <= 1
        seen.append(actions.copy())
        features = actions.sum(axis=1).astype(np.float64)
        return features, np.linalg.norm(features, axis=1)

    result = run.run_arm(predict, parent, np.zeros(2), arm, 19, 31)
    assert sum(len(batch) for batch in seen) == expected
    assert result["final_score"] <= result["parent_score"]
    if arm.startswith("response"):
        assert all(min(d) > 0 for d in result["denominators"])
        assert len(result["candidate_scores"]) == 12
    if arm.startswith("random"):
        assert result["matrices"] == []


def test_candidates_preserve_other_segments():
    parent = np.zeros((15, 2), np.float32)
    chains, edits = run.candidates(parent, [np.ones((5, 2)) for _ in range(3)])
    for chain, edit in zip(chains, edits, strict=True):
        keep = np.ones(15, dtype=bool)
        keep[edit["segment"]*5:(edit["segment"]+1)*5] = False
        np.testing.assert_array_equal(chain[keep], parent[keep])
