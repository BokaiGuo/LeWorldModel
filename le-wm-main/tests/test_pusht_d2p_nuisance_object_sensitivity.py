import numpy as np
import pytest

from experiments.spectral_reachability.analyze_pusht_d2p_nuisance_object_sensitivity import residual_energy


def test_independent_lstsq_preserves_object_and_removes_agent():
    goals = np.zeros((5, 192))
    goals[1, 0], goals[2, 0], goals[3, 1], goals[4, 1] = -1., 1., -1., 1.
    vector = np.zeros(192)
    vector[:3] = [2., 3., 4.]
    result, rank = residual_energy(goals, vector)
    assert rank == 2
    assert result["retained_energy"] == pytest.approx(16.)
    assert result["retained_fraction"] == pytest.approx(16/29)


def test_zero_energy_has_no_fraction():
    result, rank = residual_energy(np.zeros((5, 192)), np.zeros(192))
    assert rank == 0
    assert result["retained_fraction"] is None


@pytest.mark.parametrize("goals,vector", [(np.zeros((4, 192)), np.zeros(192)),
    (np.zeros((5, 192)), np.zeros(191)), (np.full((5, 192), np.nan), np.zeros(192)),
    (np.zeros((5, 192)), np.full(192, np.inf))])
def test_invalid_geometry_rejected(goals, vector):
    with pytest.raises(ValueError): residual_energy(goals, vector)
