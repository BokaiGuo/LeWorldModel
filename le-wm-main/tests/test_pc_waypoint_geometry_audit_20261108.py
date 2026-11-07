import numpy as np
import pytest

from experiments.pc_waypoint_geometry_audit_20261108_codex import check_selection


def test_valid_selection_and_strict_threshold():
    costs = np.ones(128)
    costs[9] = .05
    assert not check_selection(costs, 9)
    costs[9] = np.nextafter(.05, 0)
    assert check_selection(costs, 9)


def test_lost_best_and_noncanonical_tie_are_rejected():
    costs = np.ones(128)
    costs[7] = costs[9] = .01
    for choice in (1, 9, True):
        with pytest.raises(ValueError, match="selection"):
            check_selection(costs, choice)


def test_invalid_cost_support_is_rejected():
    with pytest.raises(ValueError, match="128 finite"):
        check_selection([.01], 0)
    costs = np.ones(128)
    costs[3] = np.nan
    with pytest.raises(ValueError, match="128 finite"):
        check_selection(costs, 0)
