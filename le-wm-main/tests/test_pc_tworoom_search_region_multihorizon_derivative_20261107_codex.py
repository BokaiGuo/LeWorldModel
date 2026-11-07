import numpy as np
import pytest
import torch

from experiments.pc_tworoom_search_region_multihorizon_derivative_20261107_codex import (
    directions, measure, normalize_actions,
)


def test_physical_normalization_chain_rule_and_difference():
    raw = torch.tensor([[0.2, -0.1]] * 5, dtype=torch.float64)
    direction = torch.tensor(directions(9000, 5)[0], dtype=torch.float64)
    def score(x):
        return normalize_actions(x, [0.1, -0.2], [2.0, 0.5]).square().sum()
    result = measure(score, raw, direction, 0.001)
    expected = (((raw - torch.tensor([0.1, -0.2])) /
                 torch.tensor([2.0, 0.5]).square()) * 2 * direction).sum()
    assert result["ad"] == pytest.approx(float(expected), abs=1e-6)
    assert result["fd"] == pytest.approx(result["ad"], abs=1e-9)
    assert result["pass"]


def test_boundary_rejected_and_zero_derivative_allowed():
    raw = torch.ones((5, 2), dtype=torch.float64)
    direction = torch.ones_like(raw)
    with pytest.raises(ValueError, match="bounds"):
        measure(lambda x: x.sum(), raw, direction, 0.01)
    result = measure(lambda x: x.square().sum(), torch.zeros_like(raw), direction, 0.01)
    assert result["ad"] == 0 and result["fd"] == 0 and result["pass"]


def test_directions_deterministic_and_unit_norm():
    first = directions(9000, 15)
    assert np.array_equal(first, directions(9000, 15))
    assert np.allclose(np.linalg.norm(first.reshape(2, -1), axis=1), 1)
