"""Fail closed on finite axes the sealed P2 accounting backend cannot enforce.

Pure arithmetic + explicitly requested reads of saved reports. No dispatch,
model imports, output files, cache, scientific admission, or hardware attestation.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from experiments.spectral_reachability import p2_full_resource_matching_20261027_v1 as old

NAME = 'p2_supported_resource_guard_20261027_v1'
BACKEND_SHA = 'fe1b6a3833d86918789d1227f773c56b298c30a6830f3bc64c22c3f15bd93018'
P2_PATH = old.PROJECT / 'results/p2_full_resource_matching_20261027_v1_audit01.json'
P2_SHA = '9d4a56ec78afc3c1704d29e92fb9ff6b52f5c00c258a01f045210dccdc8d884f'
P3_PATH = old.PROJECT / 'results/p3_adaptive_budget_saved_audit_20261027_v1_saved01/report.json'
P3_SHA = '29aa5458496a4d8118e78a173182e45d652739521d9664b6402668a94e602a2e'


def backend_binding():
    path = Path(old.__file__).resolve()
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    old.require(actual == BACKEND_SHA, 'sealed accounting backend bytes drifted')
    return dict(path=str(path), sha256=actual)


def supported_vector(value):
    """Reject even zero: it is a finite assertion, not unknown or measured work.

    UNKNOWN names are backend capability limits, not proof these quantities
    cannot ever be measured. Supporting them requires a new version with their
    correct aggregation rules (e.g. peak bytes max, not sum).
    """
    old.validate(value)
    finite = [axis for axis in old.UNKNOWN if value[axis] is not None]
    old.require(not finite, 'finite unsupported resource axes: ' + ','.join(finite))


def allocate(cap, fixed, unit, requested):
    """Same useful-work allocation, with a checked supported-axis boundary."""
    backend_binding()
    for value in (cap, fixed, unit):
        supported_vector(value)
    return old.allocate(cap, fixed, unit, requested)


class BudgetLedger:
    """NEW opt-in adapter, not a patch to sealed callers or a durable guard.

    Delegates supported-axis arithmetic and nonrefundable reservations. Rejects
    unsupported claims before changing state. No public mutable ledger state.
    """
    def __init__(self, cap, bindings):
        self._backend_binding = backend_binding()
        supported_vector(cap)
        self._ledger = old.BudgetLedger(cap, bindings)

    def reserve(self, work_id, work, purpose):
        backend_binding()
        supported_vector(work)
        self._ledger.reserve(work_id, work, purpose)

    def finish(self, work_id, completed):
        backend_binding()
        supported_vector(completed)
        self._ledger.finish(work_id, completed)

    def receipt(self):
        return dict(self._ledger.receipt(), schema=NAME + '.reservation',
            backend_binding=deepcopy(self._backend_binding),
            enforced_axes=list(old.KNOWN), unsupported_axes=list(old.UNKNOWN),
            actual_dispatch_verified=False, P1_admission=False, P3_admission=False)


def _bound_json(path, expected):
    raw = Path(path).read_bytes()
    old.require(hashlib.sha256(raw).hexdigest() == expected, 'saved report bytes drifted: ' + str(path))
    return json.loads(raw)


def audit_saved():
    """Reproduce the interface bug and verify unchanged saved allocations.

    Saved report hashes are checked, NOT their entire transitive source closure.
    Synthetic counterexamples do not alter the saved reports or count as actual
    GPU/physics work. Never call old.audit_current/run/main write workflows.
    """
    backend = backend_binding()
    p2 = _bound_json(P2_PATH, P2_SHA)
    p3 = _bound_json(P3_PATH, P3_SHA)
    old.require(p2['P2_admission'] is False and p3['P3_admission'] is False and
        p3['compute_advantage'] is False, 'saved scope must remain non-admission')
    saved = p2['next_protocol']
    allocations = {}
    for arm, requested in zip(old.ARMS, (80, 40)):
        recipe = old.candidate_recipe(arm)
        got = allocate(saved['common_per_root_cap'], recipe['fixed'],
            recipe['per_useful_candidate_cap'], requested)
        old.require(got == saved['allocations'][arm], 'saved useful allocation changed')
        allocations[arm] = got['candidates']

    # Counterexample inputs are deliberately synthetic, not historical AD costs.
    recipe = old.candidate_recipe(old.ARMS[1])
    cap, fixed, unit = deepcopy(saved['common_per_root_cap']), recipe['fixed'], recipe['per_useful_candidate_cap']
    axis = 'backward_engine_api_calls'
    cap[axis], fixed[axis], unit[axis] = 0, 0, 1
    legacy = old.allocate(cap, fixed, unit, 40)
    old.require(legacy['candidates'] == 40 and legacy['feasible'], 'expected legacy counterexample')
    rejected = False
    try:
        allocate(cap, fixed, unit, 40)
    except ValueError as exc:
        old.require('finite unsupported resource axes' in str(exc), 'unexpected rejection')
        rejected = True
    old.require(rejected, 'guard failed to reject unsupported finite cap')
    bindings = {key: 'a'*64 for key in ('root_sha256', 'task_sha256', 'model_sha256', 'protocol_sha256')}
    ledger = old.BudgetLedger(old.vector(predict_calls=1, **{axis: 0}), bindings)
    ledger.reserve('counterexample', old.vector(predict_calls=1, **{axis: 1}), 'candidate_evaluation')
    ledger.finish('counterexample', old.vector(predict_calls=1, **{axis: 2}))
    legacy_receipt = ledger.receipt()
    p3_mechanisms = {key: dict(mechanism=p3[key]['mechanism'],
        budget_saving_evidence=p3[key]['budget_saving_evidence']) for key in ('joint', 'fixed_h3', 'runtime')}
    old.require(all(v['budget_saving_evidence'] is False for v in p3_mechanisms.values()), 'do not upgrade enumeration to savings')
    return dict(schema=NAME + '.audit', status='saved_allocation_and_synthetic_regression_verified',
        bindings=[backend, dict(path=str(P2_PATH), sha256=P2_SHA), dict(path=str(P3_PATH), sha256=P3_SHA)],
        actual_saved_evidence=dict(allocations_unchanged=allocations,
            unsupported_caps_remain_unknown=all(saved['common_per_root_cap'][k] is None for k in old.UNKNOWN),
            p3_mechanisms=p3_mechanisms),
        synthetic_counterexample=dict(axis=axis, cap=0, per_candidate=1,
            legacy_candidates=legacy['candidates'], guarded_allocation_rejected=rejected,
            legacy_event_reserved=legacy_receipt['events'][0]['reserved'][axis],
            legacy_event_completed=legacy_receipt['events'][0]['completed'][axis],
            legacy_aggregate_reserved=legacy_receipt['reserved'][axis],
            legacy_aggregate_completed=legacy_receipt['completed'][axis]),
        transitive_source_closure_revalidated=False, sealed_callers_modified=False,
        runner_integration=False, cache_lookahead_freedom_certified=False,
        new_neural_calls=0, new_physical_steps=0, compute_advantage=False,
        P1_admission=False, P2_admission=False, P3_admission=False)


if __name__ == '__main__':
    print(json.dumps(audit_saved(), sort_keys=True, ensure_ascii=False, allow_nan=False))
