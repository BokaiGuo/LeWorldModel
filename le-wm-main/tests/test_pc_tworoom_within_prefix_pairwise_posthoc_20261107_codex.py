import numpy as np
import pytest

from experiments.pc_tworoom_within_prefix_pairwise_posthoc_20261107_codex import evaluate, pair_counts, validate


def test_pairwise_agreement_reverse_ties_and_empty():
    a = np.tile(np.arange(8), (8, 1))
    yes = np.ones(8, dtype=bool)
    assert pair_counts(a, a, yes)["agree"] == 224
    assert pair_counts(a, -a, yes)["disagree"] == 224
    assert pair_counts(a, np.zeros_like(a), yes)["y_tie"] == 224
    assert pair_counts(a, a, ~yes)["accuracy"] is None
    assert pair_counts(a, a, np.array([True] + [False] * 7))["agree"] == 28
    permutation = [3, 1, 0, 7, 5, 6, 2, 4]
    assert pair_counts(a[:, permutation], -a[:, permutation], yes)["disagree"] == 224
    with pytest.raises(ValueError):
        pair_counts(a[:, :7], a, yes)


def test_action_and_prefix_identity_validation():
    actions = np.zeros((8, 8, 25, 2))
    outcomes = []
    for _ in range(64):
        trace = [dict(distance=1.0, terminated=False, truncated=False) for _ in range(25)]
        outcomes.append(dict(trace=trace, prefix_success=False, ever=False))
    assert not validate(actions, outcomes).any()
    actions[0, 1, 0, 0] = 1
    with pytest.raises(ValueError, match="prefix"):
        validate(actions, outcomes)


def test_success_pair_excludes_previously_successful_prefix():
    actions = np.zeros((8, 8, 25, 2))
    outcomes = []
    for row in range(8):
        for col in range(8):
            trace = [dict(distance=float(col if step >= 10 else 0), terminated=(row == 0 and step == 0)
                          or (row == 1 and col == 0 and step == 12), truncated=False)
                     for step in range(25)]
            outcomes.append(dict(trace=trace, prefix_success=row == 0,
                                 ever=row == 0 or (row == 1 and col == 0)))
    costs = np.tile(np.arange(8), 8)[:, None] * np.ones((1, 5))
    result = evaluate(actions, outcomes, costs, costs)
    subset = result["subsets"]["prefix_not_successful"]
    assert subset["prefix_count"] == 7
    success = subset["steps"]["25"]["prediction_vs_ever_success"]
    assert success["agree"] == 7 and success["disagree"] == 0
    assert success["accuracy"] == 1
