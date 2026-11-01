"""Arithmetic fixtures and hash-bound saved report regression; no tmp files."""
from copy import deepcopy
import pytest

from experiments.spectral_reachability import p2_supported_resource_guard_20261027_v1 as m

old = m.old


def bindings():
    return {k: 'a'*64 for k in ('root_sha256', 'task_sha256', 'model_sha256', 'protocol_sha256')}


def inputs():
    recipe = old.candidate_recipe(old.ARMS[1])
    return [old.next_protocol()['common_per_root_cap'], recipe['fixed'], recipe['per_useful_candidate_cap']]


@pytest.mark.parametrize('axis', old.UNKNOWN)
@pytest.mark.parametrize('slot', (0, 1, 2))
@pytest.mark.parametrize('finite', (0, 1))
def test_allocator_rejects_finite_unsupported_axes_without_mutation(axis, slot, finite):
    vectors = inputs()
    vectors[slot][axis] = finite
    before = deepcopy(vectors)
    with pytest.raises(ValueError, match='finite unsupported resource axes: ' + axis):
        m.allocate(*vectors, 40)
    assert vectors == before


@pytest.mark.parametrize('axis', old.UNKNOWN)
def test_ledger_rejects_finite_caps_even_zero(axis):
    with pytest.raises(ValueError, match='finite unsupported resource axes'):
        m.BudgetLedger(old.vector(predict_calls=2, **{axis: 0}), bindings())


@pytest.mark.parametrize('axis', old.UNKNOWN)
@pytest.mark.parametrize('stage', ('reserve', 'finish'))
def test_rejection_precedes_ledger_state_mutation(axis, stage):
    ledger = m.BudgetLedger(old.forward(2), bindings())
    if stage == 'finish':
        ledger.reserve('one', old.forward(1), 'candidate_evaluation')
    before = ledger.receipt()
    work = old.vector(predict_calls=1, **{axis: 1})
    with pytest.raises(ValueError, match='finite unsupported resource axes'):
        if stage == 'reserve':
            ledger.reserve('one', work, 'candidate_evaluation')
        else:
            ledger.finish('one', work)
    assert ledger.receipt() == before
    # A failed unsupported receipt does not consume the work ID or terminal slot.
    if stage == 'reserve':
        ledger.reserve('one', old.forward(1), 'candidate_evaluation')
    ledger.finish('one', old.forward(1))
    assert not ledger.receipt()['pending']


def test_exact_legacy_counterexample_and_new_rejection():
    cap, fixed, unit = inputs()
    axis = 'backward_engine_api_calls'
    cap[axis], fixed[axis], unit[axis] = 0, 0, 1
    legacy = old.allocate(cap, fixed, unit, 40)
    assert legacy['feasible'] and legacy['candidates'] == 40
    assert axis not in legacy['measured_cap_usage']
    with pytest.raises(ValueError, match='finite unsupported'):
        m.allocate(cap, fixed, unit, 40)
    ledger = old.BudgetLedger(old.vector(predict_calls=1, **{axis: 0}), bindings())
    ledger.reserve('x', old.vector(predict_calls=1, **{axis: 1}), 'candidate_evaluation')
    ledger.finish('x', old.vector(predict_calls=1, **{axis: 2}))
    assert ledger.receipt()['events'][0]['completed'][axis] == 2
    assert ledger.receipt()['completed'][axis] is None


def test_supported_AD_cap_still_binds_and_unknown_stays_unknown():
    cap, fixed, unit = inputs()
    cap['math_vjp_192x10_directions'] = 7*192
    result = m.allocate(cap, fixed, unit, 40)
    assert result['candidates'] == 7
    assert not result['strict_full_resource_ready']
    assert result['unknown_full_cost_axes'] == list(old.UNKNOWN)
    ledger = m.BudgetLedger(old.vector(math_vjp_192x10_directions=192), bindings())
    ledger.reserve('AD', old.vector(math_vjp_192x10_directions=192), 'candidate_generation')
    with pytest.raises(ValueError, match='predispatch'):
        ledger.reserve('more', old.vector(math_vjp_192x10_directions=1), 'candidate_generation')
    assert ledger.receipt()['reserved']['backward_engine_api_calls'] is None


def test_nonrefund_partial_failure_duplicate_ids_and_receipt_isolation():
    ledger = m.BudgetLedger(old.forward(2), bindings())
    ledger.reserve('x', old.forward(2), 'candidate_evaluation')
    ledger.finish('x', old.forward(1))
    receipt = ledger.receipt()
    assert receipt['reserved']['predict_calls'] == 2
    assert receipt['completed']['predict_calls'] == 1
    assert not receipt['pending']
    for key in ('actual_dispatch_verified', 'full_resource_equal', 'P1_admission', 'P2_admission', 'P3_admission'):
        assert receipt[key] is False
    receipt['reserved']['predict_calls'] = 0
    receipt['events'][0]['bindings']['root_sha256'] = 'tampered'
    assert ledger.receipt()['reserved']['predict_calls'] == 2
    assert ledger.receipt()['events'][0]['bindings'] == bindings()
    with pytest.raises(ValueError, match='predispatch'):
        ledger.reserve('extra', old.forward(1), 'candidate_evaluation')
    with pytest.raises(ValueError, match='unique work'):
        ledger.reserve('x', old.forward(1), 'candidate_evaluation')
    with pytest.raises(ValueError, match='one terminal'):
        ledger.finish('x', old.forward(1))


@pytest.mark.parametrize('bad', (-1, True, 1.5, '1'))
def test_malformed_counts_not_reinterpreted_as_unknown(bad):
    values = inputs()
    values[0]['backward_engine_api_calls'] = bad
    with pytest.raises(ValueError, match='nonnegative literal'):
        m.allocate(*values, 40)


def test_backend_drift_refuses_before_state_change(monkeypatch):
    ledger = m.BudgetLedger(old.forward(2), bindings())
    before = ledger.receipt()
    monkeypatch.setattr(m, 'BACKEND_SHA', '0'*64)
    with pytest.raises(ValueError, match='backend bytes drifted'):
        ledger.reserve('x', old.forward(1), 'candidate_evaluation')
    assert ledger.receipt() == before


def test_saved_report_hash_drift_is_not_silently_rebased(monkeypatch):
    monkeypatch.setattr(m, 'P2_SHA', '0'*64)
    with pytest.raises(ValueError, match='saved report bytes drifted'):
        m.audit_saved()


def test_actual_hash_bound_saved_reports_keep_allocations_and_nonadmission():
    r = m.audit_saved()
    assert list(r['actual_saved_evidence']['allocations_unchanged'].values()) == [80, 40]
    assert r['actual_saved_evidence']['unsupported_caps_remain_unknown']
    assert all(not x['budget_saving_evidence'] for x in r['actual_saved_evidence']['p3_mechanisms'].values())
    c = r['synthetic_counterexample']
    assert (c['cap'], c['legacy_candidates'], c['legacy_event_reserved'], c['legacy_event_completed']) == (0, 40, 1, 2)
    assert c['guarded_allocation_rejected']
    assert c['legacy_aggregate_reserved'] is c['legacy_aggregate_completed'] is None
    assert r['new_neural_calls'] == r['new_physical_steps'] == 0
    for key in ('P1_admission', 'P2_admission', 'P3_admission', 'compute_advantage',
                'runner_integration', 'sealed_callers_modified', 'transitive_source_closure_revalidated',
                'cache_lookahead_freedom_certified'):
        assert r[key] is False
