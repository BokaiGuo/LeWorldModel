import numpy as np
import pytest
from experiments.spectral_reachability.p1_return_target_cuda_20261101_main import return_goal


def test_target_preserves_object_and_returns_actuator():
    original = [10, 20, 30, 40, .5, 2, 3]
    target = [50, 60, 70, 80, 1.5, 4, 5]
    np.testing.assert_array_equal(return_goal(original, target), [10, 20, 70, 80, 1.5, 2, 3])
    assert target == [50, 60, 70, 80, 1.5, 4, 5]


def test_malformed_goal_rejected():
    with pytest.raises(ValueError):
        return_goal([0] * 6, [0] * 7)
