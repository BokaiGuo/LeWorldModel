import numpy as np
import pytest
import torch
from copy import deepcopy

from experiments.pc_rgb_precise_spectral_proposals_20261107_codex import (
    jacobian_and_directions, make_candidates,
)
from experiments.pc_tworoom_precise_spectral_development_audit_20261107_codex import audit_arrays


def test_linear_jacobian_spectral_directions():
    matrix = torch.diag(torch.tensor([3.0, 2.0, 1.0, 0.5], dtype=torch.float64))
    raw = torch.zeros((2, 2), dtype=torch.float64)
    j, s, directions = jacobian_and_directions(lambda x: matrix @ x.flatten(), raw)
    torch.testing.assert_close(j, matrix)
    torch.testing.assert_close(s, torch.tensor([3., 2., 1., .5], dtype=torch.float64))
    assert abs(directions[0].flatten()[0]) == pytest.approx(1)
    assert abs(directions[1].flatten()[1]) == pytest.approx(1)


def test_candidates_same_parent_bounds_and_no_silent_clip():
    directions = np.arange(1, 21, dtype=float).reshape(2, 5, 2)
    raw = np.full((5, 2), 0.7)
    actions = make_candidates(raw, directions)
    assert actions.shape == (5, 5, 2)
    np.testing.assert_allclose(actions[0], raw)
    assert np.max(np.abs(actions)) <= 1
    with pytest.raises(ValueError, match="margin"):
        make_candidates(np.ones((5, 2)), directions)


def test_zero_rank_rejected():
    x = torch.ones((2, 2))
    with pytest.raises(ValueError, match="rank"):
        jacobian_and_directions(lambda a: a.flatten() * 0, x)


def test_audit_rejects_tampered_selection():
    episode = 9000
    base = np.zeros((25, 2))
    random = np.random.default_rng(27127000 + episode).normal(size=(2, 25, 2))
    spectral = np.ones((2, 25, 2))
    plan, physical = {"episode": episode, "arms": {}}, {"episode": episode, "arms": {}}
    for arm, direction in (("random", random), ("spectral", spectral)):
        plan["arms"][arm] = {"candidates": make_candidates(base, direction).tolist(),
                             "directions": direction.tolist(), "scores": [1, 2, 3, 4, 5], "selected_index": 0}
        physical["arms"][arm] = {"selected_index": 0,
                                 "candidates": [{"trace": [{"success": False}] * 25,
                                                 "ever_success": False, "complete": True} for _ in range(5)]}
    audit_arrays(plan, physical)
    corrupt = deepcopy(physical)
    corrupt["arms"]["random"]["selected_index"] = 1
    with pytest.raises(ValueError, match="selected"):
        audit_arrays(plan, corrupt)
