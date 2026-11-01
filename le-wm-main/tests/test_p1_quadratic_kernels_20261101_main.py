import pytest
import torch
from experiments.spectral_reachability import p1_quadratic_kernels_20261101_main as q


def test_quadratic_loss_and_gradient(monkeypatch):
    monkeypatch.setattr(q, "MATRIX", [[2., 0.], [0., .5]])
    monkeypatch.setattr(q, "GOAL", [0., 0.])
    prediction = torch.tensor([[1., 2.]], requires_grad=True)
    goal = torch.zeros_like(prediction)
    value = q.loss(prediction, goal)
    assert value.item() == 2.5
    value.backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([[4., .5]]))


def test_metric_cannot_be_used_on_different_goal(monkeypatch):
    monkeypatch.setattr(q, "MATRIX", [[1., 0.], [0., 1.]])
    monkeypatch.setattr(q, "GOAL", [0., 0.])
    with pytest.raises(ValueError, match="different goal"):
        q.loss(torch.zeros((1, 2)), torch.ones((1, 2)))
