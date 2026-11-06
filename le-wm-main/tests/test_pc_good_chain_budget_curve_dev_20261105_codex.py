import numpy as np
import pytest

from experiments import pc_good_chain_budget_curve_dev_20261105_codex as run


@pytest.mark.parametrize("arm", run.ARMS)
@pytest.mark.parametrize("budget", run.BUDGETS)
def test_budget_and_returned_incumbent(arm, budget):
    parent = np.full((15, 2), .95, np.float32)
    parent[0, 0] = -1
    observed = []

    def predictor(actions):
        actions = np.asarray(actions, np.float32)
        assert np.isfinite(actions).all() and np.abs(actions).max() <= 1
        features = actions.sum(axis=1).astype(np.float64)
        scores = np.linalg.norm(features, axis=1)
        observed.extend((action.copy(), float(score)) for action, score in zip(actions, scores, strict=True))
        return features, scores

    result = run.run_arm(predictor, parent, np.zeros(2), arm, 12, budget)
    assert len(observed) == budget
    assert result["final_score"] == min(score for _, score in observed)
    assert any(np.array_equal(action, np.asarray(result["final_actions"], np.float32))
               and score == result["final_score"] for action, score in observed)
    if arm != "response":
        assert result["precompute"]["matrices"] == []
