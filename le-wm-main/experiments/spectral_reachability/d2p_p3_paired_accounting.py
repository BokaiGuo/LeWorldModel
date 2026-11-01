"""Pure P3 accounting contracts, not experiments or stage admission.

The caller must externally seal the cohort, branch census, caps and shared-cost
policy BEFORE outcomes. Receipts are assertions, not proof of instrumentation
coverage or CUDA execution. Unknown work must not be encoded as zero.
"""
from dataclasses import asdict, dataclass, fields
from math import isfinite

from experiments.spectral_reachability.d2p_planning_compute_ledger import (
    Binding, BudgetContract, Ledger, Resources, compare_pair,
)


def _label(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("nonempty sealed identifier required")


def _count(value):
    if type(value) is not int or value < 0:
        raise ValueError("nonnegative integer count required")


def _unique(items):
    if type(items) is not tuple or not items or len(set(items)) != len(items):
        raise ValueError("nonempty immutable unique census required")


@dataclass(frozen=True)
class RootPlan:
    contract: BudgetContract
    branch_ids: tuple[str, ...]

    def __post_init__(self):
        if not isinstance(self.contract, BudgetContract):
            raise TypeError("complete per-root budget contract required")
        _unique(self.branch_ids)
        for branch in self.branch_ids:
            _label(branch)


@dataclass(frozen=True)
class PairedPlan:
    """Externally presealed independent cohort; never derive from observed rows.

    Branch IDs describe common planned comparison slots, not successful
    candidates. Methods may search different numbers of imagined trajectories;
    those are counted in Resources, not censored by this branch census.
    """
    source_seal: str
    roots: tuple[RootPlan, ...]
    fixed_method: str = "fixed"
    adaptive_method: str = "adaptive"

    def __post_init__(self):
        for value in (self.source_seal, self.fixed_method, self.adaptive_method):
            _label(value)
        if self.fixed_method == self.adaptive_method:
            raise ValueError("distinct fixed and adaptive methods required")
        if type(self.roots) is not tuple or not self.roots or not all(
            isinstance(root, RootPlan) for root in self.roots
        ):
            raise ValueError("immutable nonempty RootPlan cohort required")
        _unique(tuple(root.contract.binding.root_id for root in self.roots))
        contexts = {(r.contract.binding.protocol_id, r.contract.binding.task_id,
                     r.contract.binding.frozen_model_id) for r in self.roots}
        if len(contexts) != 1:
            raise ValueError("mixed protocol/task/frozen-model cohort")


@dataclass(frozen=True)
class BranchReceipt:
    branch_id: str
    status: str
    task_success: bool
    resources: Resources

    def __post_init__(self):
        _label(self.branch_id)
        if self.status not in ("completed", "failed", "blocked", "aborted"):
            raise ValueError("terminal branch status required; pending is not complete accounting")
        if type(self.task_success) is not bool or (
            self.task_success and self.status != "completed"
        ):
            raise ValueError("only completed branches can report task success")
        if not isinstance(self.resources, Resources):
            raise TypeError("complete attempted resources required even for failures")


def _pool(resources):
    return Resources(**{
        field.name: (max if field.name == "max_planning_horizon_tokens" else sum)(
            getattr(row, field.name) for row in resources
        ) for field in fields(Resources)
    })


@dataclass(frozen=True)
class RunReceipt:
    binding: Binding
    method: str
    branches: tuple[BranchReceipt, ...]
    overhead: Resources
    prediction_receipt: dict
    walltime_seconds: float | None = None

    def __post_init__(self):
        if not isinstance(self.binding, Binding) or not isinstance(self.overhead, Resources):
            raise TypeError("binding and full overhead resources required")
        _label(self.method)
        if type(self.branches) is not tuple or not self.branches or not all(
            isinstance(branch, BranchReceipt) for branch in self.branches
        ):
            raise ValueError("nonempty immutable branch receipts required")
        _unique(tuple(b.branch_id for b in self.branches))
        if self.walltime_seconds is not None and (
            type(self.walltime_seconds) not in (int, float)
            or not isfinite(self.walltime_seconds) or self.walltime_seconds < 0
        ):
            raise ValueError("walltime must be measured finite nonnegative seconds or None")

    def resources(self):
        # Overhead includes signal construction, setup and any unassigned work.
        # Branch resources must be disjoint from it; no subtraction/refunds.
        return _pool((self.overhead, *(b.resources for b in self.branches)))


def _validate_prediction(run, contract):
    """Reconcile the existing PredictionBudget v2 receipt, including failures."""
    receipt = run.prediction_receipt
    if not isinstance(receipt, dict) or receipt.get("schema") != "d2p_adaptive_budget.v2":
        raise ValueError("PredictionBudget v2 receipt required")
    if receipt.get("binding") != asdict(run.binding) or receipt.get("method") != run.method:
        raise ValueError("prediction receipt binding/method mismatch")
    for name in ("predict_sample_limit", "attempted_predict_samples", "blocked_calls"):
        _count(receipt.get(name))
    if receipt["predict_sample_limit"] != contract.limits.predict_samples:
        raise ValueError("prediction guard cap differs from frozen per-root cap")
    phases = receipt.get("phases")
    if not isinstance(phases, list):
        raise ValueError("all prediction phases required")
    calls = samples = failures = 0
    for phase in phases:
        if not isinstance(phase, dict) or phase.get("method") != run.method or phase.get(
            "phase"
        ) not in ("planning", "evaluation"):
            raise ValueError("invalid prediction phase")
        for name in ("predict_calls", "predict_samples", "failed_predict_calls"):
            _count(phase.get(name))
        if type(phase.get("completed")) is not bool:
            raise ValueError("phase completion flag required")
        if phase["failed_predict_calls"] > phase["predict_calls"] or (
            phase["failed_predict_calls"] and phase["completed"]
        ) or phase["predict_samples"] < phase["predict_calls"]:
            raise ValueError("inconsistent attempted prediction counters")
        calls += phase["predict_calls"]
        samples += phase["predict_samples"]
        failures += phase["failed_predict_calls"]
    resources = run.resources()
    if samples != receipt["attempted_predict_samples"] or (
        samples != resources.predict_samples or calls != resources.predict_calls
    ):
        raise ValueError("all-phase attempted prediction totals mismatch; no success-only refunds")
    if samples > receipt["predict_sample_limit"]:
        raise ValueError("prediction guard cap exceeded")
    return {"attempted_predict_samples": samples, "failed_predict_calls": failures,
            "blocked_calls": receipt["blocked_calls"],
            "incomplete_phases": sum(not p["completed"] for p in phases)}


def compare_paired(plan, fixed, adaptive, *, comparison_scope="all_planned_branches",
                   walltime_advantage_claim=False):
    """Identity join with a pre-outcome manifest, never zip/intersection/filter.

    Reports vector compliance, not equal runtime, efficacy or admission. All
    failed/blocked/aborted branches stay in both denominator and attempted cost.
    This read-only validator launches no models, backward passes or env steps.
    """
    if not isinstance(plan, PairedPlan):
        raise TypeError("externally sealed PairedPlan required")
    if comparison_scope != "all_planned_branches":
        raise ValueError("successful-only or filtered comparisons forbidden")
    if walltime_advantage_claim is not False:
        raise ValueError("walltime advantage claims are not supported by this accounting validator")
    expected = {root.contract.binding.root_id: root for root in plan.roots}

    def index(rows, method):
        result = {}
        for run in rows:
            if not isinstance(run, RunReceipt) or run.method != method:
                raise ValueError("explicit matching method RunReceipt required")
            key = run.binding.root_id
            if key in result:
                raise ValueError("duplicate root identity")
            if key not in expected or run.binding != expected[key].contract.binding:
                raise ValueError("unexpected root or mixed protocol/task/model binding")
            if set(b.branch_id for b in run.branches) != set(expected[key].branch_ids):
                raise ValueError("missing/extra planned branches; failures must be retained")
            result[key] = run
        if set(result) != set(expected):
            raise ValueError("missing roots; never drop failed roots or use an intersection")
        return result

    fixed_index = index(fixed, plan.fixed_method)
    adaptive_index = index(adaptive, plan.adaptive_method)
    pairs = []
    for key, root in expected.items():
        left, right = fixed_index[key], adaptive_index[key]
        compared = compare_pair(root.contract, Ledger(left.binding, left.method, left.resources()),
                                Ledger(right.binding, right.method, right.resources()))
        for label, run in (("baseline", left), ("candidate", right)):
            compared[label]["prediction_audit"] = _validate_prediction(run, root.contract)
            compared[label]["overhead"] = asdict(run.overhead)
            compared[label]["branches"] = [asdict(b) for b in run.branches]
            compared[label]["successes"] = sum(b.task_success for b in run.branches)
            compared[label]["planned_branch_denominator"] = len(root.branch_ids)
            compared[label]["walltime_seconds"] = run.walltime_seconds
        compared["adaptive_minus_fixed"] = {
            f.name: getattr(right.resources(), f.name) - getattr(left.resources(), f.name)
            for f in fields(Resources)
        }
        pairs.append(compared)
    return {"schema": "d2p_p3_paired_accounting.v1", "source_seal": plan.source_seal,
            "root_denominator": len(expected), "pairs": pairs,
            "all_pairs_within_budget": all(p["both_within_budget"] for p in pairs),
            "walltime_measured_for_all": all(
                r.walltime_seconds is not None for r in (*fixed_index.values(), *adaptive_index.values())
            ),
            "p1_dependency": "not_passed", "p3_admitted": False,
            "compute_advantage": False, "walltime_advantage": False, "efficacy_claim": False}
