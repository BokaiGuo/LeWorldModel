import numpy as np
import pytest

from experiments.pc_pusht_native_success_metric_audit_20261107_codex import metrics


def test_agent_distance_is_part_of_success_even_when_object_is_exact():
    goal = np.zeros(7)
    state = goal.copy()
    state[0] = 30
    record = metrics(goal, state)
    assert record["object_pose_cost"] == 0
    assert not record["native_success"]
    assert record["success_component_proxy"] == 2.25


def test_native_distance_includes_velocity_while_success_does_not():
    goal = np.zeros(7)
    state = goal.copy()
    state[5] = 7
    record = metrics(goal, state)
    assert record["native_success"]
    assert record["native_state_distance"] == 7


def test_reject_nonfinite_state():
    with pytest.raises(ValueError):
        metrics(np.zeros(7), [0, 0, np.nan, 0, 0, 0, 0])
