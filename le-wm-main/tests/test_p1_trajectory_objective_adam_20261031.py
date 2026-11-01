import importlib

import pytest


MODULE = "experiments.spectral_reachability.p1_trajectory_objective_adam_20261031"


def test_softmin_prefers_lower_intermediate_endpoint():
    torch = pytest.importorskip("torch")
    mod = importlib.import_module(MODULE)
    values = [torch.tensor(0.2), torch.tensor(0.8)]
    result = mod.softmin(values, tau=0.02)
    assert float(result) < 0.2
    assert abs(float(result) - 0.2) < 0.001


def test_softmin_is_symmetric_and_differentiable():
    torch = pytest.importorskip("torch")
    mod = importlib.import_module(MODULE)
    a = torch.tensor(0.3, requires_grad=True)
    b = torch.tensor(0.7, requires_grad=True)
    value = mod.softmin([a, b])
    value.backward()
    assert a.grad is not None and b.grad is not None
    assert float(a.grad) > float(b.grad) > 0.0
