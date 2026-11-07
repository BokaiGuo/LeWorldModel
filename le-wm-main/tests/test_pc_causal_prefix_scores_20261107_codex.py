import numpy as np
import pytest

from experiments.pc_causal_prefix_scores_20261107_codex import canonicalize, select


def test_same_prefix_canonicalized_but_different_tail_is_not():
    actions = np.zeros((2, 10, 1), dtype=np.float32)
    actions[1, 5:] = 0.5
    scores = np.array([[1.00001, 4], [1, 2]], dtype=np.float64)
    fixed, audit = canonicalize(actions, scores, packet_steps=5, max_disagreement=0.001)
    assert fixed[0, 0] == fixed[1, 0]
    np.testing.assert_array_equal(fixed[:, 1], [4, 2])
    assert audit["duplicate_prefix_groups"] == 1


def test_same_prefix_large_disagreement_fails_closed():
    actions = np.zeros((2, 10, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="exceeds tolerance"):
        canonicalize(actions, np.array([[1., 3.], [2., 3.]]),
                     packet_steps=5, max_disagreement=0.01)


def test_selection_is_invariant_to_input_order():
    actions = np.zeros((3, 10, 1), dtype=np.float32)
    actions[1, 5:] = 0.5
    actions[2, :] = 0.7
    scores = np.array([[1.00001, 4], [1, 2], [3, 3]], dtype=np.float64)
    first, value, _ = select(actions, scores, packet_steps=5, max_disagreement=0.001)
    order = np.array([2, 1, 0])
    second, value2, _ = select(actions[order], scores[order], packet_steps=5, max_disagreement=0.001)
    np.testing.assert_array_equal(actions[first], actions[order][second])
    assert value == value2


def test_invalid_clock_rejected():
    with pytest.raises(ValueError, match="time base"):
        canonicalize(np.zeros((1, 9, 1)), np.zeros((1, 2)), packet_steps=5, max_disagreement=0)
