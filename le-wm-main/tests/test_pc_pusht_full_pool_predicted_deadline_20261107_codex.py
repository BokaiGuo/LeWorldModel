import numpy as np
import pytest

from experiments.pc_pusht_full_pool_predicted_deadline_20261107_codex import choose


def test_deadline_is_joint_candidate_time_choice():
    costs = np.array([[5, 4, 3, 2, 1], [0, 9, 9, 9, 2]], dtype=float)
    assert choose(costs, "endpoint") == (0, 4)
    assert choose(costs, "min_predicted") == (1, 0)


def test_stable_ties_and_invalid_data():
    assert choose(np.ones((2, 5)), "min_predicted") == (0, 0)
    with pytest.raises(ValueError):
        choose(np.full((2, 5), np.nan), "endpoint")
