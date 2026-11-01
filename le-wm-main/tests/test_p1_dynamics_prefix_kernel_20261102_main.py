import numpy as np
import pytest

from experiments.spectral_reachability import p1_dynamics_prefix_kernel_20261102_main as k


def test_prefixes_follow_model_response_and_respect_bounds():
    jac = np.zeros((12, 10))
    jac[0, 3] = 4
    pair, record = k.spectral_pair(np.zeros((5, 2)), jac, np.eye(12))
    assert pair[0].reshape(-1)[3] == -.25
    assert pair[1].reshape(-1)[3] == .25
    assert np.count_nonzero(pair[0]) == 1
    assert record["source"] == "task_weighted_model_action_jacobian"


def test_metric_changes_direction_without_outcomes():
    jac = np.zeros((12, 10))
    jac[0, 0], jac[1, 1] = 2, 1
    q = np.eye(12)
    q[1, 1] = 3
    pair, _ = k.spectral_pair(np.zeros((5, 2)), jac, q)
    assert pair[1].reshape(-1)[1] == .25
    assert pair[1].reshape(-1)[0] == 0


def test_suffix_bank_solves_simple_response_and_includes_zero():
    response = np.eye(20)
    bank = k.suffix_bank(response, np.ones(20), np.eye(20))
    assert bank.shape == (4, 10, 2)
    assert not bank[0].any()
    assert np.max(np.abs(bank)) <= 1
    scores = k.costs(bank.reshape(4, 20), np.ones(20), np.eye(20))
    assert k.select(scores) == 3


def test_selection_retains_control_and_stable_ties():
    assert k.select([1, 0, 2]) == 1
    assert k.select([0, 0]) == 0


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_invalid_scores_abstain(value):
    with pytest.raises(ValueError):
        k.select([0, value])


def test_no_response_is_not_a_program():
    with pytest.raises(ValueError):
        k.spectral_pair(np.zeros((5, 2)), np.zeros((12, 10)), np.eye(12))
