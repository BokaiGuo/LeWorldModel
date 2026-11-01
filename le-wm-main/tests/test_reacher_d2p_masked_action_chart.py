import numpy as np
import pytest
import torch

from experiments.spectral_reachability import reacher_d2p_masked_action_chart as chart
from experiments.spectral_reachability.run_reacher_d2p_masked_task_preflight import probe_spec


@pytest.mark.parametrize("task", chart.TASKS)
@pytest.mark.parametrize("horizon", (1, 2))
def test_exact_literal_zero_and_raw_time_order(task, horizon):
    width = len(chart.active_columns(task))
    value = torch.arange(horizon * width, dtype=torch.float64).reshape(1, horizon, width)
    full = chart.expand_commands(value, task)
    torch.testing.assert_close(full.ravel()[chart.active_columns(task, horizon)], value.ravel())
    inactive = sorted(set(range(10 * horizon)) - set(chart.active_columns(task, horizon)))
    assert torch.count_nonzero(full.ravel()[inactive]) == 0


def test_reduced_derivative_is_exact_column_selection():
    def expand(flat):
        return chart.expand_commands(flat.reshape(1, 2, 5), "wrist_only").ravel()
    j = torch.autograd.functional.jacobian(expand, torch.zeros(10))
    torch.testing.assert_close(j[chart.active_columns("wrist_only", 2)], torch.eye(10))
    assert torch.count_nonzero(j[chart.active_columns("shoulder_only", 2)]) == 0


def test_inactive_zero_is_normalized_once_not_set_to_normalized_zero():
    mean = torch.linspace(.1, 1, 10)
    std = torch.linspace(.2, 2, 10)
    raw = chart.expand_commands(torch.ones(1, 1, 5), "shoulder_only")
    result = chart.normalize_raw(raw, mean, std)
    torch.testing.assert_close(result[0, 0, 1::2], -mean[1::2] / std[1::2])
    np.testing.assert_allclose(result.numpy(), (raw.numpy() - mean.numpy()) / std.numpy())


@pytest.mark.parametrize("task,horizon", [("bad", 1), ("full_drive", 0), ("full_drive", True)])
def test_invalid_task_horizon(task, horizon):
    with pytest.raises(ValueError):
        chart.active_columns(task, horizon)


@pytest.mark.parametrize("std", [torch.zeros(10), torch.full((10,), float("nan"))])
def test_bad_statistics_rejected(std):
    with pytest.raises(ValueError):
        chart.normalize_raw(torch.zeros(1, 1, 10), torch.zeros(10), std)


def test_probe_inventory_fixed():
    assert len(probe_spec(1)) == 40 and len(probe_spec(2)) == 80
    assert len(set(probe_spec(2))) == 80
    with pytest.raises(ValueError):
        probe_spec(3)


def test_rollout_preserves_observed_history_and_packet_count():
    class Recorder:
        def __init__(self):
            self.actions = []
        def action_encoder(self, actions):
            return actions
        def predict(self, emb, actions):
            self.actions.append(actions.detach().clone())
            return emb + actions[..., :2]
    model = Recorder()
    history = torch.arange(30.).reshape(1, 3, 10)
    future = torch.ones(1, 2, 10)
    value = chart.terminal_normalized(model, torch.zeros(1, 3, 2), history, future)
    assert len(model.actions) == 2 and value.shape == (1, 2)
    torch.testing.assert_close(model.actions[0][:, :2], history[:, 1:])
    torch.testing.assert_close(model.actions[0][:, -1], future[:, 0])
    torch.testing.assert_close(model.actions[1][:, -2:], future)
