"""Pure, explicit D2P resource accounting; no model imports or execution.

One ledger covers one method on one frozen protocol/task/root/model binding.
Include setup, search, verification and evaluation encoding in the declared
scope, or seal their exclusion/shared-cost policy in protocol_id beforehand.
Never infer counters from historical ``forward_rollout_evaluations``: those
receipts mix batched calls and candidate counts. Unknown counters are NOT zero.

Predict calls count Python predict invocations; samples sum their batch sizes
at EACH predicted token (a B-sample H-token rollout usually contributes B*H).
Encoder forwards count model.encode invocations; encoder_samples sum input
batch sizes, NOT image/history tokens. Action/input VJP rows count logical
cotangent rows, while backward calls count autograd invocations, including a
vectorized Jacobian as one invocation, not one invocation per logical row.
Input derivatives (e.g. goal co-metric) are separate from action derivatives.

Rollout samples count candidate trajectories, imagined_action_tokens sum B*H
over those trajectories, and max_planning_horizon_tokens is a maximum, not an
additive counter. Physical tokens, env.step calls, and restored execution
branches are independent of imagined computation; warmup/reset scope must be
predeclared. Resources are never summed into a scalar 'equivalent budget'.

This module compares caller-supplied observations only. It does not instrument
existing runners, prove provenance/pre-outcome sealing, estimate FLOPs or wall
time, establish strong-baseline adequacy, or authorize P1/later stages.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields


def _label(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


@dataclass(frozen=True)
class Binding:
    """Opaque sealed identifiers, not outcome-dependent row positions.

    protocol_id should bind the frozen checkpoint config, task/metric/action
    semantics, horizon schedule, observation/feedback and accounting policy.
    root_id should bind the actual state/RGB/history/goal, not just its index.
    The caller owns creation and verification of these identifiers.
    """

    protocol_id: str
    task_id: str
    root_id: str
    frozen_model_id: str

    def __post_init__(self) -> None:
        for field in fields(self):
            _label(field.name, getattr(self, field.name))


@dataclass(frozen=True)
class Resources:
    """Complete nonnegative integer observations or per-resource budget caps.

    All fields are required: explicitly supply zero only for known zero work.
    Counts are descriptive; no architecture-independent cost ratios implied.
    """

    predict_calls: int
    predict_samples: int
    encoder_forwards: int
    encoder_samples: int
    action_vjp_rows: int
    action_backward_calls: int
    input_vjp_rows: int
    input_backward_calls: int
    rollout_samples: int
    imagined_action_tokens: int
    max_planning_horizon_tokens: int
    physical_action_tokens: int
    physical_env_steps: int
    physical_branches: int

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            # No coercion/truncation and no bool-as-int acceptance.
            if type(value) is not int or value < 0:
                raise ValueError(f"{field.name} must be a nonnegative Python integer")


@dataclass(frozen=True)
class Ledger:
    binding: Binding
    method: str
    resources: Resources

    def __post_init__(self) -> None:
        if not isinstance(self.binding, Binding) or not isinstance(self.resources, Resources):
            raise TypeError("Ledger requires Binding and complete Resources")
        _label("method", self.method)


@dataclass(frozen=True)
class BudgetContract:
    """Immutable per-root vector of predeclared caps shared by paired methods.

    Construct/freeze before outcomes and seal externally. Frozen dataclasses
    prevent ordinary mutation, not retrospective construction of a new budget.
    """

    binding: Binding
    limits: Resources

    def __post_init__(self) -> None:
        if not isinstance(self.binding, Binding) or not isinstance(self.limits, Resources):
            raise TypeError("BudgetContract requires Binding and complete Resources")


def _compare(observed: Resources, reference: Resources) -> dict[str, str]:
    return {
        field.name: ("under" if getattr(observed, field.name) < getattr(reference, field.name)
                     else "over" if getattr(observed, field.name) > getattr(reference, field.name)
                     else "equal")
        for field in fields(Resources)
    }


def compare_pair(contract: BudgetContract, baseline: Ledger, candidate: Ledger) -> dict:
    """Compare both methods to frozen caps AND candidate to baseline per axis.

    Under-budget is not equal use. Equal use above caps is not compliance.
    Neither implies equal runtime/FLOPs or an efficacy/stage-admission claim.
    """
    if not isinstance(contract, BudgetContract) or not all(
        isinstance(item, Ledger) for item in (baseline, candidate)
    ):
        raise TypeError("BudgetContract and two Ledger instances required")
    if not contract.binding == baseline.binding == candidate.binding:
        raise ValueError("mixed protocol/task/root/frozen-model binding")
    if baseline.method == candidate.method:
        raise ValueError("paired methods must have distinct identifiers")
    baseline_status = _compare(baseline.resources, contract.limits)
    candidate_status = _compare(candidate.resources, contract.limits)
    paired_status = _compare(candidate.resources, baseline.resources)
    return {
        "schema": "d2p_planning_compute_ledger.v1",
        "binding": asdict(contract.binding),
        "limits": asdict(contract.limits),
        "baseline": {"method": baseline.method, "resources": asdict(baseline.resources)},
        "candidate": {"method": candidate.method, "resources": asdict(candidate.resources)},
        "baseline_vs_budget": baseline_status,
        "candidate_vs_budget": candidate_status,
        "candidate_vs_baseline": paired_status,
        "both_within_budget": "over" not in baseline_status.values()
                              and "over" not in candidate_status.values(),
        "equal_resources": all(value == "equal" for value in paired_status.values()),
        "efficacy_claim": False,
    }
