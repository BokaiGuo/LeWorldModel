import pytest

from experiments.pc_tworoom_effect_vector_physical_audit_20261107_codex import pairwise_accuracy


def test_pairwise_ties_and_direction():
    assert pairwise_accuracy([1, 2, 3], [1, 2, 3]) == 1
    assert pairwise_accuracy([3, 2, 1], [1, 2, 3]) == 0
    assert pairwise_accuracy([1, 1], [1, 2]) == .5
    assert pairwise_accuracy([1, 2], [1, 1]) is None


def test_invalid_shapes_and_nonfinite():
    with pytest.raises(ValueError):
        pairwise_accuracy([1], [1, 2])
    with pytest.raises(ValueError):
        pairwise_accuracy([float("nan")], [1])
