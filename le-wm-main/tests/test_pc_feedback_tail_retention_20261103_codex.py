import numpy as np
import pytest

from experiments.pc_feedback_tail_retention_20261103_codex import select_feedback_tail


def test_old_tail_is_scored_in_same_budget_and_retained():
    calls = []

    def predict(batch):
        calls.append(batch.copy())
        return np.sum(batch ** 2, axis=(1, 2))

    old = np.zeros((5, 2))
    result = select_feedback_tail(old, total_samples=15,
                                  rng=np.random.default_rng(1), predict=predict)
    assert result["source"] == "old"
    assert sum(len(batch) for batch in calls) == 15
    np.testing.assert_array_equal(calls[0][0], old)
    assert all(batch.shape[1:] == (5, 2) for batch in calls)


def test_new_tail_can_replace_old_and_invalid_input_fails():
    result = select_feedback_tail(np.ones((5, 2)), total_samples=4,
                                  rng=np.random.default_rng(2),
                                  predict=lambda batch: np.sum(batch ** 2, axis=(1, 2)))
    assert result["source"] == "new"
    assert result["new_score"] < result["old_score"]
    with pytest.raises(ValueError):
        select_feedback_tail(np.full((5, 2), 2), total_samples=4,
                             rng=np.random.default_rng(2), predict=lambda x: x)
    with pytest.raises(ValueError):
        select_feedback_tail(np.zeros((5, 2)), total_samples=1,
                             rng=np.random.default_rng(2), predict=lambda x: x)
