"""Pure saved/array regressions; not neural/physical efficacy tests."""
from copy import deepcopy
import json

import numpy as np
import pytest

from experiments.spectral_reachability import p3_adaptive_budget_saved_audit_20261027_v1 as m


def test_allocation_fixed_work_not_fixed_candidate_count():
    for mode in ('uniform', 'dynamic'):
        for c2, c3 in ((0, 0), (1, 0), (.8, .7)):
            a = m.allocate(mode, c2, c3)
            assert 2*a['2']+3*a['3'] == 60
    assert m.allocate('uniform', 1, 0) == {'2': 15, '3': 10}
    assert m.allocate('dynamic', 0, 0) == {'2': 21, '3': 6}


@pytest.mark.parametrize('x', [-1, 1.1, float('nan'), float('inf')])
def test_bad_signal(x):
    with pytest.raises(ValueError):
        m.allocate('dynamic', x, 0)


def test_spectral_residual_degeneracies():
    assert m.chi([[1], [0]], [0, 0], [1, 0], np.eye(2)) == 0
    assert m.chi([[1], [0]], [0, 0], [0, 1], np.eye(2)) == 1
    assert m.chi([[0], [0]], [0, 0], [1, 0], np.eye(2)) == 1
    assert m.chi([[0], [0]], [0, 0], [0, 0], np.eye(2)) == 0


def test_repeated_equal_attempts_and_start_return_not_double_charged():
    events = []
    for index in range(2):
        e = dict(kind='predict', role='arm', index=index, completed=False)
        events += [e, dict(e, completed=True)]
    ops, counts, anomalies = m.operation_ledger(events, ('predict',))
    assert counts['predict'] == dict(attempted=2, completed=2, unreturned=0)
    assert len(ops) == 2 and not anomalies


def test_failed_and_unreturned_attempts_remain_charged():
    start = dict(kind='predict', index=0, completed=False)
    events = [start, dict(start, error='abort'), dict(start, index=1)]
    ops, counts, anomalies = m.operation_ledger(events, ('predict',))
    assert not anomalies
    assert counts['predict'] == dict(attempted=2, completed=0, unreturned=1)
    assert ops[0]['error'] == 'abort'


def test_svd_retries_same_identity_are_not_deduplicated():
    e = dict(kind='svd', horizon=2, role='signal', completed=False)
    _, counts, anomalies = m.operation_ledger([e, dict(e, completed=True), e, dict(e, completed=True)], ('svd',))
    assert counts['svd']['attempted'] == counts['svd']['completed'] == 2 and not anomalies


def test_orphan_return_does_not_fabricate_attempt():
    _, counts, anomalies = m.operation_ledger([dict(kind='predict', index=1, completed=True)], ('predict',))
    assert counts['predict']['attempted'] == 0 and anomalies


def candidate(cost, h=2, index=0):
    return dict(cost=cost, horizon=h, index=index, completed=True)


def test_positive_or_stagnant_cost_is_not_stop_certificate():
    assert m.stop_certificate([candidate(1e-30), candidate(1e-30, index=1)])['first_certificate'] is None


def test_zero_tie_bound_checks_horizon_and_index():
    a = m.stop_certificate([candidate(0), candidate(2, 3)])
    assert a['first_certificate'] == dict(after_candidate=0, avoidable_search_predicts=3)
    assert a['realized_saved_predicts'] == 0
    # Future H2 could beat an H3 zero on stable tie: cannot stop yet.
    assert m.stop_certificate([candidate(0, 3), candidate(2, 2)])['first_certificate'] is None


def runtime_fixture(depth=1, max_depth=3, mode='adaptive'):
    policy = dict(base_beam=4, max_depth=max_depth, max_expansions=144,
                  expansion_factor=2, chi_threshold=.5, gain_threshold=.05)
    method = 'p3_adaptive_runtime_'+mode
    decision = dict(expanded_candidates=4, horizon_tokens=1, single_pass_predict_samples=4)
    layer = dict(depth=depth, expected_expansions=16, attempted_expansions=16,
        layer_complete=True, beam_width=4, retained=4,
        signals=[dict(horizon_gain=0., unreachable_fraction=.1)], decision=decision,
        kernels=[dict(parent_indices=[], receipt=dict(attempts=[dict(succeeded=True, cost=float(i), primitive_indices=[i]) for i in range(16)]))])
    row = dict(root_sha256='r', method=method, failure=None, action_encoder_samples=16+depth,
        processed_action_context_tokens=3*(16+depth), resources=dict(max_planning_horizon_tokens=depth),
        prediction_receipt=dict(attempted_predict_samples=16+depth,
            phases=[dict(predict_samples=16), dict(predict_samples=depth)]),
        search=dict(mode=mode, layers=[layer], best_plan=dict(depth=depth, cost=0., primitive_indices=[0]), status='dynamics_horizon_stop'))
    wrapper = dict(protocol_id='p', protocol=dict(schedule=policy, roots=['r'], methods=[method]))
    return dict(protocol_id='p', rows=[row]), wrapper


def test_runtime_final_layer_stop_not_effective_savings():
    r, p = runtime_fixture(depth=3)
    result = m.audit_runtime(r, p)
    assert result['rows'][0]['layers'][0]['signal_stop_triggered']
    assert not result['rows'][0]['effective_early_signal_stop']


def test_runtime_preterminal_stop_is_effective_but_not_efficacy():
    r, p = runtime_fixture()
    result = m.audit_runtime(r, p)
    assert result['rows'][0]['effective_early_signal_stop']
    assert result['budget_saving_evidence'] is False


def test_runtime_fixed_ignores_same_stop_signal():
    r, p = runtime_fixture(mode='fixed')
    result = m.audit_runtime(r, p)
    assert not result['rows'][0]['layers'][0]['signal_stop_triggered']


def test_runtime_tampered_threshold_decision_detected():
    r, p = runtime_fixture()
    r['rows'][0]['search']['layers'][0]['decision']['expanded_candidates'] = 8
    assert any(not c['ok'] for c in m.audit_runtime(r, p)['checks'])


def test_runtime_failed_kernel_preserved():
    r, p = runtime_fixture()
    failed = dict(succeeded=False, error='failed', primitive_indices=[0])
    r['rows'][0]['search']['layers'][0]['kernels'][0]['receipt']['attempts'][0] = failed
    assert m.audit_runtime(r, p)['rows'][0]['layers'][0]['failures'] == [failed]


def test_evidence_hash_drift_rejected(tmp_path):
    path = tmp_path/'input.json'; path.write_text('{}')
    with pytest.raises(ValueError):
        m.Evidence().load(path, '0'*64)


def test_nonfinite_json_not_emitted():
    with pytest.raises(ValueError):
        m.canonical(dict(value=float('nan')))


def test_real_saved_default_reports_and_counts():
    """Actual receipts, not new GPU evidence. Hash pins prevent silent drift."""
    ev = m.Evidence()
    r = ev.load(m.ROOT/'results'/m.DEFAULTS['runtime'][0], m.DEFAULTS['runtime'][1])
    p = ev.load(m.ROOT/'results/P3_ADAPTIVE_RUNTIME_20261027_cuda_v1/P3_ADAPTIVE_RUNTIME_protocol_20261027.json')
    result = m.audit_runtime(r, p)
    assert all(c['ok'] for c in result['checks'])
    assert result['predict_attempted'] == 1171
    assert not any(x['effective_early_signal_stop'] for x in result['rows'])
    assert sum(l['signal_stop_triggered'] for x in result['rows'] for l in x['layers']) == 1
    assert sum(l['attempted_edges'] == 0 for x in result['rows'] for l in x['layers']) == 2
    r3 = ev.load(m.ROOT/'results'/m.DEFAULTS['fixed_h3'][0], m.DEFAULTS['fixed_h3'][1])
    result3 = m.audit_fixed_h3(r3)
    assert all(c['ok'] for c in result3['checks'])
    assert result3['predict_attempted'] == 1176


def joint_saved():
    r = m.read(m.ROOT/'results'/m.DEFAULTS['joint'][0])
    e = m.read(m.ROOT/'results/p3_joint_horizon_goal_metric_scheduler_20261027_v1_cuda_main01/execution_protocol.json')
    p = m.read(m.Path(e['prepared'])/'protocol.json')
    journals = {x['root_sha256']: [json.loads(line) for line in m.Path(x['journal_path']).read_text().splitlines()] for x in r['roots']}
    return r, p, journals


def test_real_joint_accounting_and_same_q_strata():
    result = m.audit_joint(*joint_saved())
    assert all(c['ok'] for c in result['checks'])
    assert result['totals']['predict_attempted'] == result['totals']['action_encoder_attempted'] == 1066
    assert result['totals']['jacobian_completed'] == 8 and result['totals']['svd_completed'] == 32
    assert sum(p['stratum'] == 'mixedH' for p in result['pairings']) == 1
    assert not any(r['stopping']['first_certificate'] for r in result['rows'])


@pytest.mark.parametrize('field', ['cost', 'actions', 'metric', 'order'])
def test_joint_tampering_does_not_silently_pass(field):
    r, p, j = joint_saved(); arm = r['roots'][0]['arms']['old_Q__dynamic']
    if field == 'cost':
        arm['candidates'][0]['cost'] += 1
    elif field == 'actions':
        arm['candidates'][0]['actions'][0][0] = .1
    elif field == 'metric':
        arm['active_metric_id'] = 'wrong'
    else:
        arm['candidates'][0], arm['candidates'][1] = arm['candidates'][1], arm['candidates'][0]
    assert any(not c['ok'] for c in m.audit_joint(r, p, j)['checks'])


def test_joint_partial_compilation_preserves_actual_work_lower_bound():
    r, p, j = joint_saved(); r['roots'][0]['compilation'] = {}
    result = m.audit_joint(r, p, j)
    assert result['rows'][0]['known_work_lower_bound']
    assert result['rows'][0]['journal_ledger']['predict']['attempted'] > 0
    assert result['partials_discarded'] is False
