import numpy as np
import pytest

from experiments.pc_tworoom_dedup_retention_20261107_codex import evaluate, unique_prefixes


def test_unique_first_occurrence_and_query_scope():
    prefixes = np.zeros((4, 10, 2), np.float32)
    prefixes[1] = .1
    prefixes[2] = .1
    prefixes[3] = .2
    tails = np.zeros((9, 15, 2), np.float32)
    calls = []

    def score(actions, stage):
        calls.append((stage, actions.copy()))
        return actions[:, 0, 0]

    result = evaluate(prefixes, tails, 7, score)
    assert result["representatives"] == [0, 1, 3]
    assert result["inverse"] == [0, 1, 1, 2]
    assert [len(actions) for _, actions in calls] == [15, 8, 8, 8]
    for name, actions in calls[1:]:
        indices = result["arms"][name]["retained"]
        assert len(set(indices)) == 2
        expected = np.stack([np.concatenate((prefixes[i], tails[j])) for i in indices for j in range(5, 9)])
        np.testing.assert_array_equal(actions, expected)


def test_fail_closed_single_unique_or_nonfinite():
    with pytest.raises(ValueError):
        unique_prefixes(np.zeros((2, 10, 2)))
    with pytest.raises(ValueError):
        unique_prefixes(np.full((2, 10, 2), np.nan))
