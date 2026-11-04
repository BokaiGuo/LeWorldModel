"""CPU fixtures only; no independent resource/provenance verification."""
from dataclasses import replace
from copy import deepcopy

import pytest

from experiments.spectral_reachability.d2p_planning_compute_ledger import (
    Binding, BudgetContract, Ledger, Resources,
)
from experiments.spectral_reachability.p3_schedule_ledger_guard_20261102_codexexclusive import (
    validate_p3_with_ledger,
)
from test_d2p_p1_p2_p3_interface_contract_20261102_codexexclusive import p3


def fixtures():
    receipt = p3()
    receipt["signal"].update(horizon=3, predict_budget=16)
    identity = receipt["identity"]
    binding = Binding(**{k: identity[k] for k in (
        "protocol_id", "task_id", "root_id", "frozen_model_id"
    )})
    values = {key: 0 for key in Resources.__dataclass_fields__}
    values.update(predict_calls=3, predict_samples=12,
                  max_planning_horizon_tokens=3)
    observed = Resources(**values)
    limits = replace(observed, predict_samples=16, predict_calls=4)
    return receipt, BudgetContract(binding, limits), Ledger(binding, "adaptive", observed)


def check(receipt, budget, ledger):
    return validate_p3_with_ledger(receipt, budget=budget, ledger=ledger)


def test_matching_full_ledger_and_budget():
    receipt, budget, ledger = fixtures()
    assert check(receipt, budget, ledger) == receipt
    assert receipt["P3_admission"] is False


@pytest.mark.parametrize("horizon", [-1, 0, -1.5, 1.5, 3.0, True, None, float("nan")])
def test_invalid_horizons_fail(horizon):
    receipt, budget, ledger = fixtures()
    receipt["signal"]["horizon"] = horizon
    with pytest.raises(ValueError):
        check(receipt, budget, ledger)


@pytest.mark.parametrize("value", [-1, 1.5, 16.0, True])
def test_predict_budget_requires_integer_samples(value):
    receipt, budget, ledger = fixtures()
    receipt["signal"]["predict_budget"] = value
    with pytest.raises(ValueError):
        check(receipt, budget, ledger)


def test_budget_and_actual_must_reconcile_without_imputing_work():
    receipt, budget, ledger = fixtures()
    receipt["signal"]["predict_budget"] = 15
    with pytest.raises(ValueError, match="planned allocation"):
        check(receipt, budget, ledger)
    receipt, budget, ledger = fixtures()
    ledger = replace(ledger, resources=replace(ledger.resources, predict_samples=11))
    with pytest.raises(ValueError, match="attempted ledger"):
        check(receipt, budget, ledger)


def test_identity_scope_and_every_resource_cap_are_checked():
    receipt, budget, ledger = fixtures()
    wrong = replace(ledger, binding=replace(ledger.binding, root_id="other"))
    with pytest.raises(ValueError, match="bindings"):
        check(receipt, budget, wrong)
    wrong = replace(ledger, resources=replace(ledger.resources, encoder_forwards=1))
    with pytest.raises(ValueError, match="resource exceeds"):
        check(receipt, budget, wrong)


def test_both_planned_and_observed_horizon_respect_caps():
    receipt, budget, ledger = fixtures()
    receipt["signal"]["horizon"] = 2
    with pytest.raises(ValueError, match="planning horizon"):
        check(receipt, budget, ledger)
    receipt, budget, ledger = fixtures()
    receipt["signal"]["horizon"] = 4
    with pytest.raises(ValueError, match="scheduled horizon"):
        check(receipt, budget, ledger)


def test_declared_allocation_cannot_exceed_predict_sample_cap():
    receipt, budget, ledger = fixtures()
    budget = replace(budget, limits=replace(budget.limits, predict_samples=15))
    with pytest.raises(ValueError, match="planned predict allocation"):
        check(receipt, budget, ledger)


def test_reject_missing_budget_and_ledger_and_preserve_input():
    receipt, budget, ledger = fixtures()
    original = deepcopy(receipt)
    with pytest.raises(TypeError):
        check(receipt, None, ledger)
    result = check(receipt, budget, ledger)
    result["signal"]["horizon"] = 1
    assert receipt == original
