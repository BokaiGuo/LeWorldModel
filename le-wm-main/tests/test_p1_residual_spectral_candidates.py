import numpy as np
import pytest

from experiments.spectral_reachability.p1_residual_spectral_candidates import proposals


def test_residual_can_prioritize_nonleading_response_and_keeps_both_signs():
    jacobian = np.zeros((12, 10))
    jacobian[0, 0], jacobian[1, 1] = 4, 1
    residual = np.zeros(12)
    residual[1] = 1
    rows = proposals(np.zeros((5, 2)), jacobian, np.eye(12), residual,
                     radius=.25, max_modes=2)
    assert [r["mode"] for r in rows] == [1, 1, 0, 0]
    assert rows[0]["actions"].reshape(-1)[1] == -.25
    assert rows[1]["actions"].reshape(-1)[1] == .25
    assert all(np.max(np.abs(r["actions"])) <= 1 for r in rows)


def test_action_bounds_and_unresolved_modes():
    jacobian = np.zeros((2, 10))
    jacobian[0, 0] = 1
    rows = proposals(np.full((5, 2), .99), jacobian, np.eye(2), np.ones(2),
                     radius=.25, max_modes=3)
    assert len(rows) == 2
    assert np.max(rows[1]["actions"]) <= 1
    with pytest.raises(ValueError, match="no resolved"):
        proposals(np.zeros((5, 2)), np.zeros((2, 10)), np.eye(2), np.ones(2),
                  radius=.25, max_modes=1)


@pytest.mark.parametrize("max_modes", [0, True, 1.5])
def test_invalid_mode_budget_fails_closed(max_modes):
    with pytest.raises(ValueError):
        proposals(np.zeros((5, 2)), np.eye(10), np.eye(10), np.ones(10),
                  radius=.25, max_modes=max_modes)


def test_nonfinite_and_mismatched_inputs_fail_closed():
    with pytest.raises(ValueError):
        proposals(np.zeros((5, 2)), np.eye(10), np.eye(10), np.full(10, np.nan),
                  radius=.25, max_modes=1)
    with pytest.raises(ValueError):
        proposals(np.zeros((5, 2)), np.eye(10), np.eye(9), np.ones(10),
                  radius=.25, max_modes=1)
