"""CPU tensor fixtures only; these tests do not instantiate a neural model."""
import numpy as np
import pytest
import torch

from experiments.spectral_reachability import p1_h4_parfit_20261101 as h4


@pytest.mark.parametrize("boundary,horizon,carry", [(0, 4, False), (5, 3, True),
                                                     (10, 2, True), (15, 1, True)])
def test_schedule(boundary, horizon, carry):
    assert h4.dimensions(boundary) == (horizon, carry)


@pytest.mark.parametrize("horizon", [1, 2, 3, 4])
def test_response_shape_and_inactive_columns(horizon):
    jac = np.zeros((192, 40))
    jac[:10*horizon, :10*horizon] = np.eye(10*horizon)
    bank, singular = h4.response_bank(jac, np.eye(192), np.ones(192), np.zeros(192), horizon)
    assert bank.shape == (8, 4, 5, 2)
    assert singular.shape == (10*horizon,)
    assert np.all(bank[:, horizon:] == 0)
    np.testing.assert_array_equal(bank[2], -bank[3])
    np.testing.assert_array_equal(bank[4], -bank[5])
    assert np.max(np.abs(bank)) <= 1
    jac[0, -1] = 1
    if horizon < 4:
        with pytest.raises(ValueError):
            h4.response_bank(jac, np.eye(192), np.ones(192), np.zeros(192), horizon)


def test_zero_response_and_invalid_numeric():
    z = np.zeros((192, 40))
    bank, _ = h4.response_bank(z, np.eye(192), np.ones(192), np.zeros(192), 4)
    assert np.count_nonzero(bank) == 0
    z[0, 0] = np.nan
    with pytest.raises(ValueError):
        h4.response_bank(z, np.eye(192), np.ones(192), np.zeros(192), 4)


@pytest.mark.parametrize("boundary", [0, 5, 10, 15])
def test_normalized_protection_and_inactive(boundary):
    horizon, _ = h4.dimensions(boundary)
    raw = torch.zeros(2, 4, 5, 2)
    gradient = torch.arange(80, dtype=torch.float32).reshape_as(raw) + 1
    gradient[:, horizon:] = 0
    out = h4.normalized_update(raw, gradient, horizon, boundary)
    assert torch.equal(out[:, horizon:], raw[:, horizon:])
    if boundary == 0:
        assert torch.equal(out[:, 0], raw[:, 0])
        assert torch.all(out[:, 1:] < 0)
    else:
        assert torch.all(out[:, 0] < 0)
    torch.testing.assert_close(out.abs().amax(dim=(1, 2, 3)), torch.full((2,), .05))
    if horizon < 4:
        gradient[:, horizon] = 1
        with pytest.raises(ValueError, match="inactive gradient"):
            h4.normalized_update(raw, gradient, horizon, boundary)


def test_exchange_prefix_diagonal_ties_and_64_calls():
    bank = torch.arange(8, dtype=torch.float32)[:, None, None, None].expand(8, 4, 5, 2) / 8
    calls = []
    def score(plan):
        calls.append(tuple(plan.shape))
        return 0.
    selected, scores, tails = h4.exchange(bank, 4, score)
    assert len(scores) == len(calls) == 64
    assert all(x == (4, 5, 2) for x in calls)
    assert tails == list(range(8))
    assert torch.equal(selected, bank)
    cross = h4.cross_bank(bank)
    assert torch.equal(cross[8*3+7, 0], bank[3, 0])
    assert torch.equal(cross[8*3+7, 1:], bank[7, 1:])
    chosen, _, tails = h4.exchange(bank, 3, lambda plan: float(plan[1:].square().sum()))
    assert tails == [0] * 8
    assert torch.equal(chosen[:, 0], bank[:, 0])
    assert torch.equal(chosen[:, 1:], bank[0:1, 1:].expand(8, 3, 5, 2))


@pytest.mark.parametrize("boundary", [5, 10, 15])
def test_feedback_rescores_old_tail_and_fresh_at_equal_horizon(boundary):
    horizon, _ = h4.dimensions(boundary)
    bank = torch.ones(8, 4, 5, 2) * .3
    previous = torch.zeros(horizon + 1, 5, 2)
    seen = []
    def score(plan):
        seen.append(plan.clone())
        return float(plan.square().sum())
    result, scores, index = h4.feedback_select(bank, previous, previous[0], boundary, score)
    assert len(seen) == len(scores) == 9
    assert all(x.shape == (horizon, 5, 2) for x in seen)
    assert index == 8 and torch.equal(result, previous[1:])
    with pytest.raises(ValueError):
        h4.feedback_select(bank, previous, torch.ones(5, 2), boundary, score)
    _, _, tie_index = h4.feedback_select(bank * 0, previous, previous[0], boundary, lambda x: 0)
    assert tie_index == 8


@pytest.mark.parametrize("arm", h4.ARMS)
@pytest.mark.parametrize("boundary", [0, 5, 10, 15])
def test_budget_matches_independent_event_schedule(arm, boundary):
    h, carry = h4.dimensions(boundary)
    actual = dict.fromkeys(h4.AXES, 0)
    def trajectory(length, replay=False):
        for _ in range(length):
            for axis in ("action_encode_calls", "predict_calls", "predict_samples"):
                actual[axis] += 1
        if replay:
            actual["replay_calls"] += 1
            actual["selection_evaluations"] += 1
    if arm == "structured":
        trajectory(h)
        actual["jacobian_calls"] += 1
        actual["svd_calls"] += 1
        actual["backward_calls"] += 192
        actual["vjp_rows"] += 192
    if arm == "immediate_progress":
        for _ in range(16):
            trajectory(1, replay=True)
    for _ in range(8):
        for _ in range(6):
            trajectory(h)
            actual["backward_calls"] += 1
            actual["vjp_rows"] += 1
        trajectory(h, replay=True)
    if h > 1:
        for _ in range(64):
            trajectory(h, replay=True)
    if carry:
        for _ in range(9):
            trajectory(h, replay=True)
    assert actual == h4.node_cost(arm, boundary)
