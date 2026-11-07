import pytest

from experiments.pc_tworoom_geometry_pool_alignment_20261107_codex import success_pair_accuracy


def test_success_pair_accuracy_task_weighted_and_ties():
    assert success_pair_accuracy([0., 2., 1.], [True, False, False]) == 1.
    assert success_pair_accuracy([2., 0., 2.], [True, False, False]) == .25
    assert success_pair_accuracy([0., 1.], [True, True]) is None
    assert success_pair_accuracy([0., 1.], [False, False]) is None


def test_invalid_success_pair_values():
    with pytest.raises(ValueError):
        success_pair_accuracy([float("nan"), 2.], [True, False])
    with pytest.raises(ValueError):
        success_pair_accuracy([0.], [True, False])
