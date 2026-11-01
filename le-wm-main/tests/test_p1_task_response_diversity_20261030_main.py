import numpy as np
import pytest

from experiments.spectral_reachability.p1_task_response_diversity_20261030_main import select


def test_common_direction_and_deterministic_farthest():
    j = np.array([[[1., 0.], [100., 1.]],
                  [[2., 0.], [-100., 1.]],
                  [[1., 3.], [500., 1.]],
                  [[1., -3.], [0., 1.]]])
    result = select(j, [1., 0.], [0., 1., 1., 1.], 3)
    assert result["selected_indices"] == [0, 2, 3]
    assert result["selection_min_squared_distance"] == [None, 9., 9.]
    assert result["P1_admission"] is False


def test_ties_and_identical_responses_are_not_dropped():
    j = np.ones((3, 2, 2))
    result = select(j, [1., 1.], [2., 0., 0.], 3)
    assert result["selected_indices"] == [1, 0, 2]
    assert result["selection_min_squared_distance"] == [None, 0., 0.]


@pytest.mark.parametrize("direction,count", [([0., 0.], 1), ([1., 0.], 4), ([1., float("nan")], 1)])
def test_reject_invalid_inputs(direction, count):
    with pytest.raises(ValueError):
        select(np.ones((3, 2, 2)), direction, [0., 1., 2.], count)
