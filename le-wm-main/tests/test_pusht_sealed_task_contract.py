import numpy as np
import pytest
from experiments.spectral_reachability.audit_pusht_sealed_task_contract import labels


def test_agent_is_not_nuisance_for_official_task():
    goal = np.zeros(7)
    state = np.array([30., 0., 0., 0., 0., 0., 0.])
    result = labels(state, goal)
    assert result['object_only_success'] is True
    assert result['official_success'] is False
    assert result['official_state_distance'] == 30


def test_threshold_is_strict_and_joint_position():
    state = np.array([15., 0., 15., 0., 0., 0., 0.])
    assert labels(state, np.zeros(7))['official_success'] is False
    state[:4] = [0., 0., 20., 0.]
    assert labels(state, np.zeros(7))['object_only_success'] is False


@pytest.mark.parametrize('value', [np.zeros(5), np.full(7, np.nan), np.full(7, np.inf)])
def test_invalid_state(value):
    with pytest.raises(ValueError): labels(value, np.zeros(7))
