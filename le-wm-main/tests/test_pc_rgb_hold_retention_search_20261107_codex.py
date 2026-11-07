import numpy as np
import pytest

from experiments.pc_rgb_hold_retention_search_20261107_codex import choose_retained, search


def test_hold_relative_can_change_retention_with_same_scores():
    costs = np.array([[0., 1., 1.], [1., 2., 2.], [2., 2., 2.]])
    assert choose_retained(costs, "centered", 1) == [0, 1]
    assert choose_retained(costs, "effect", 1) == [0, 2]


def test_shared_candidate_shape_budget_and_selection():
    parents = np.zeros((3, 10, 2), dtype=np.float32)
    parents[1, :5] = .2
    parents[2, :5] = -.2
    probes = np.stack([np.full((5, 2), .1), np.full((5, 2), -.1)])
    tails = np.stack([np.full((5, 2), .4), np.full((5, 2), -.4)])
    batches = []
    def score(x):
        batches.append(x.copy())
        return x.sum(axis=(1, 2))
    result = search(score, parents, probes, tails, prefix=5, remaining=10, strategy="effect", seed=3)
    assert [len(x) for x in batches] == [3, 9, 4]
    assert result["model_queries"] == 16
    assert result["choice"].cost == min(x.sum(axis=(1, 2)).min() for x in batches)
    np.testing.assert_allclose(batches[1].reshape(3, 3, 10, 2)[:, :, :5],
                               np.broadcast_to(parents[:, None, :5], (3, 3, 5, 2)))


def test_invalid_actions_fail_before_scoring():
    calls = []
    with pytest.raises(ValueError):
        search(lambda x: calls.append(x), np.ones((3, 10, 2))*2, np.zeros((2, 5, 2)),
               np.zeros((2, 5, 2)), prefix=5, remaining=10, strategy="effect", seed=0)
    assert not calls
