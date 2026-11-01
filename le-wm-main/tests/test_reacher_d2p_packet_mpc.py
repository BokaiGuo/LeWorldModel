"""Pure numerical/protocol fixtures; no CPU neural experiments."""
import pytest
import torch

from experiments.spectral_reachability.audit_reacher_d2p_packet_mpc import choose_plan


def cost(plan):
    return plan.square().sum((1, 2))


def test_carry_is_rescored_and_retained_without_inverse_tanh():
    searched = torch.full((1, 2, 10), 8.)
    carry = torch.full((1, 2, 10), -7.)
    chosen, receipt = choose_plan(cost, searched, carry)
    assert torch.equal(chosen, carry)
    assert receipt["selected"] == "carried"
    assert receipt["inverse_tanh_used"] is False
    assert receipt["current_model_costs"] == {"searched": 1280., "carried": 980.}
    chosen.zero_()
    assert bool((carry == -7).all())


def test_same_exact_plan_tie_keeps_carried():
    plan = torch.ones(1, 2, 10)
    _, receipt = choose_plan(cost, plan, plan.clone())
    assert receipt["selected"] == "carried"


def test_current_observation_can_reject_previous_plan():
    searched, carry = torch.ones(1, 1, 10), torch.zeros(1, 1, 10)
    calls = []
    def changed_cost(p):
        calls.append(p.clone())
        return (p - 1).square().sum((1, 2))
    chosen, receipt = choose_plan(changed_cost, searched, carry)
    assert torch.equal(chosen, searched)
    assert receipt["selected"] == "searched"
    assert len(calls) == 2


def test_no_carried_plan_is_single_candidate():
    chosen, receipt = choose_plan(cost, torch.zeros(1, 3, 10))
    assert chosen.shape == (1, 3, 10)
    assert receipt["current_model_costs"] == {"searched": 0.}
    assert receipt["carry_rescored_on_current_rgb"] is False


@pytest.mark.parametrize("shape", [(2, 2, 10), (1, 0, 10), (1, 3)])
def test_invalid_plan_shape_fails_closed(shape):
    with pytest.raises(ValueError):
        choose_plan(cost, torch.zeros(shape))


@pytest.mark.parametrize("bad", ["shape", "dtype", "nan"])
def test_invalid_carried_plan_fails_closed(bad):
    searched = torch.zeros(1, 2, 10)
    carry = torch.zeros(1, 1, 10) if bad == "shape" else searched.double() if bad == "dtype" else torch.full_like(searched, float("nan"))
    with pytest.raises(ValueError):
        choose_plan(cost, searched, carry)


@pytest.mark.parametrize("value", [-1., float("inf"), float("nan")])
def test_invalid_current_model_cost_fails_closed(value):
    with pytest.raises(ValueError):
        choose_plan(lambda p: torch.tensor([value]), torch.zeros(1, 1, 10))
