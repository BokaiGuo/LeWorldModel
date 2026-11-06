import numpy as np
import pytest

from experiments.pc_action_identity_allocation_dev_20261106_codex import descriptor, summarize


def test_descriptor_preserves_joint_reordering_but_not_action_identity():
    actions = np.array([[0., 0.], [1., 0.], [0., 2.], [2., 1.]])
    costs = np.array([0., 1., 4., 3.])
    permutation = np.array([2, 0, 3, 1])
    original = descriptor(actions, costs)
    assert descriptor(actions[permutation], costs[permutation]) == pytest.approx(original)
    assert descriptor(actions, costs[permutation]) != pytest.approx(original)
    assert sorted(costs) == sorted(costs[permutation])


def test_summarize_uses_only_retained_indices_for_physical_results():
    model = {"B_min": list(range(12)), "optimized_min": list(range(12))}
    physical = {"first_success_steps": {"B": [None, 5, None, None] + [1] * 8,
                                        "optimized": [None] * 12}}
    result = summarize(np.arange(12), model, physical)
    assert result["retained"] == [0, 1, 2, 3]
    assert result["B_success_chains"] == 1
    assert result["B_any_success"]
    assert not result["optimized_any_success"]


def test_descriptor_rejects_nonfinite_inputs():
    with pytest.raises(ValueError):
        descriptor(np.ones((2, 3)), [0., np.nan])
