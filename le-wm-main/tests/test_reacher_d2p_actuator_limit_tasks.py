import numpy as np
import pytest
import torch

from experiments.spectral_reachability.audit_reacher_d2p_actuator_limit_tasks import TASKS, action_limit_gate, task_bounds
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0


@pytest.mark.parametrize("task", list(TASKS))
def test_future_action_chart_and_physical_witness_use_identical_limits(task):
    mean, std = np.linspace(-.1, .1, 10, dtype=np.float32), np.linspace(.3, 1.4, 10, dtype=np.float32)
    lower, upper = task_bounds(mean, std, TASKS[task], device=torch.device("cpu"))
    params = torch.linspace(-2., 2., 30).reshape(1, 3, 10)
    raw, _ = p0._to_raw_actions(params, mean, std, lower, upper, np.array([-1., -1.]), np.array([1., 1.]))
    assert action_limit_gate({"action_witness_raw": raw.tolist()}, TASKS[task])
    full_lower, full_upper = task_bounds(mean, std, TASKS["full_drive"], device=torch.device("cpu"))
    full_raw, _ = p0._to_raw_actions(params, mean, std, full_lower, full_upper, np.array([-1., -1.]), np.array([1., 1.]))
    np.testing.assert_allclose(raw, full_raw * np.asarray(TASKS[task])[None], rtol=0, atol=1e-6)


@pytest.mark.parametrize("limits", [(0., 1.), (1., 2.), (float("nan"), 1.)])
def test_limits_must_define_a_finite_non_degenerate_future_chart(limits):
    with pytest.raises(ValueError):
        task_bounds(np.zeros(10), np.ones(10), limits, device=torch.device("cpu"))


def test_recorded_witness_rejects_a_freed_weak_actuator_or_wrong_horizon():
    method = {"action_witness_raw": np.zeros((15, 2)).tolist()}
    assert action_limit_gate(method, TASKS["weak_elbow"])
    method["action_witness_raw"][4][1] = .5
    assert not action_limit_gate(method, TASKS["weak_elbow"])
    assert not action_limit_gate({"action_witness_raw": np.zeros((14, 2)).tolist()}, TASKS["full_drive"])
