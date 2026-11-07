import numpy as np
import pytest

from experiments.pc_tworoom_two_stage_query_20261107_codex import evaluate_two_stage


def test_queries_only_retained_b_and_counts():
    pool = np.zeros((13, 9, 25, 2), dtype=np.float32)
    for i in range(13):
        for j in range(9):
            pool[i, j, :, 0] = i / 20
            pool[i, j, :, 1] = j / 10
    calls = []

    def score(actions, stage):
        calls.append((stage, actions.copy()))
        return actions[:, 0, 0] + actions[:, 0, 1]

    result = evaluate_two_stage(pool, np.arange(13), score)
    assert [len(x[1]) for x in calls] == [65, 8, 8, 8]
    for stage, actions in calls[1:]:
        indices = result["arms"][stage]["retained"]
        np.testing.assert_array_equal(actions, pool[indices, 5:].reshape(8, 25, 2))
        assert result["arms"][stage]["selected"][0] in indices


def test_rejects_invalid_actions():
    pool = np.zeros((13, 9, 25, 2))
    pool[0, 0, 0, 0] = 2
    with pytest.raises(ValueError):
        evaluate_two_stage(pool, np.arange(13), lambda *_: [])
