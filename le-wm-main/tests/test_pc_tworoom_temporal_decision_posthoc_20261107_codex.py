import numpy as np
import pytest

from experiments.pc_tworoom_temporal_decision_posthoc_20261107_codex import selected_indices


def test_selection_uses_correct_clock_and_deterministic_ties():
    costs = np.ones((64, 5))
    costs[8, 2] = -1
    costs[13, 3] = -2
    costs[29, 4] = -3
    costs[7, 0] = -4
    assert selected_indices(costs) == {"step15": 8, "step20": 13, "step25": 29,
                                       "minimum_all_steps": 7}
    assert all(index == 0 for index in selected_indices(np.zeros((64, 5))).values())


def test_rejects_missing_or_nonfinite_costs():
    with pytest.raises(ValueError):
        selected_indices(np.ones((63, 5)))
    values = np.ones((64, 5))
    values[2, 0] = np.nan
    with pytest.raises(ValueError):
        selected_indices(values)
