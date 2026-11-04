"""P3 single-decision shape and caller-supplied ledger reconciliation.

No instrumentation or provenance authentication is performed here. The ledger
must cover the same declared work scope as the schedule allocation, including
attempted failed calls. A caller must not substitute unknown work with zero.
"""
from dataclasses import fields

from experiments.spectral_reachability.d2p_p1_p2_p3_interface_contract_20261102_codexexclusive import (
    validate_p3,
)
from experiments.spectral_reachability.d2p_planning_compute_ledger import (
    BudgetContract,
    Ledger,
    Resources,
)


def validate_p3_with_ledger(payload, *, budget, ledger):
    """Return a checked copy of a P3 receipt, not stage admission.

    This interface defines ``signal.horizon`` in planning tokens and
    ``signal.predict_budget`` / allocation in attempted predict-samples.
    The supplied ledger and budget refer to this same single-decision scope.
    Counts, identifiers, temporal order, and no-refund assertions remain supplied
    by the caller, not independently observed or authenticated by this function.
    """
    receipt = validate_p3(payload)
    if not isinstance(budget, BudgetContract) or not isinstance(ledger, Ledger):
        raise TypeError("BudgetContract and Ledger required")
    identity = receipt["identity"]
    binding = budget.binding
    if ledger.binding != binding or any(
        getattr(binding, name) != identity[name]
        for name in ("protocol_id", "task_id", "root_id", "frozen_model_id")
    ):
        raise ValueError("P3 ledger, budget and receipt bindings must match")

    signal, allocation = receipt["signal"], receipt["allocation"]
    horizon = signal["horizon"]
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("P3 horizon must be a positive integer number of tokens")
    predict_budget = signal["predict_budget"]
    if type(predict_budget) is not int or predict_budget < 0:
        raise ValueError("P3 predict budget must be nonnegative integer predict-samples")
    if predict_budget != allocation["planned"]:
        raise ValueError("P3 planned allocation must equal signal predict budget")
    if allocation["actual"] != ledger.resources.predict_samples:
        raise ValueError("P3 actual allocation must equal attempted ledger predict-samples")
    if ledger.resources.max_planning_horizon_tokens > horizon:
        raise ValueError("ledger planning horizon exceeds scheduled horizon")
    if allocation["planned"] > budget.limits.predict_samples:
        raise ValueError("planned predict allocation exceeds budget cap")
    if horizon > budget.limits.max_planning_horizon_tokens:
        raise ValueError("scheduled horizon exceeds budget cap")
    if any(
        getattr(ledger.resources, field.name) > getattr(budget.limits, field.name)
        for field in fields(Resources)
    ):
        raise ValueError("ledger resource exceeds budget cap")
    return receipt
