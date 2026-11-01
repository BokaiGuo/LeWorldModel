"""RAM metadata fixtures only: tests are not scientific or freshness evidence."""
from copy import deepcopy
import io
import json
from contextlib import redirect_stdout

import pytest

from experiments.spectral_reachability import moving_equal_budget_prospective_queue_20261027_v1 as q


def fixture():
    methods = []
    for i, arm in enumerate(q.ARMS):
        n = [272, 268, 144, 168, 3604, 176][i]
        costs = {k: q.known(dict(attempted=0, completed=0), 'explicit RAM zero') for k in q.COUNTERS}
        for k in ('predict', 'action_encoder', 'forward_samples'):
            costs[k] = q.known(dict(attempted=n, completed=n), 'RAM meter')
        for k in q.SCALARS:
            costs[k] = q.known(0, 'explicit RAM zero')
        costs['AD_forward_in_predict'] = q.known([128, 0, 0, 0, 3600, 4][i], 'RAM subset')
        costs['scalar_loss_reverse_sweeps'] = (q.known([64, 0, 0, 0, 1800, 0][i], 'RAM scalar loss')
            if i in (0, 4) else q.unknown('not separately metered'))
        costs['scalar_output_VJP_directions'] = (q.known(dict(attempted=384, completed=384), 'RAM VJP')
            if i == 5 else q.unknown('not separately metered'))
        costs['pred_projector'] = q.unknown('not separately metered')
        methods.append(dict(algorithm=arm, two_root_totals=costs,
            contract_fields=dict(per_root_budget=dict(predict=n // 2), horizon=2,
                                 selection='sealed rule', poison_not_an_outcome='configuration only'),
            future_packets=q.known(2, 'explicit future not H3'),
            root_receipts=[dict(source_slot=j, task_sha256=t) for j, t in enumerate(q.TASKS)],
            per_method_physics=q.unknown('multiarm cohort cannot be allocated'),
            model_invocation_latency=q.known(900, 'whole shared run, not per method'),
            model_Q='OUTCOME_POISON_MUST_NOT_READ'))
    return dict(schema='p1_moving_baseline_budget_matrix_20261027_v1.matrix',
        status='saved_resource_projection_not_admitted', input=dict(sha256=q.COMPARISON_SHA),
        equal_total_compute=False, P1_admission=False, methods=methods,
        inherited_encoding_shared_once=q.known({k: dict(attempted=8, completed=8)
            for k in ('encoder', 'projector')}, 'RAM shared once'),
        physical_cohorts=[dict(arms=list(q.ARMS[:3]), counts=dict(physics_step=dict(attempted=1224, completed=1224)))],
        RGB_cohorts=[dict(arms=list(q.ARMS[:3]), counts=dict(encoder=dict(attempted=12, completed=12)))])


def complete_contracts():
    """Algebraic equality fixture ONLY; never an approved prospective configuration."""
    costs = {}
    for k in q.UNITS:
        if q.kind(k) == 'CPU':
            value = dict(Q_scores=dict(attempted=2, completed=2))
        elif q.kind(k) == 'counter':
            value = dict(attempted=2, completed=2)
        else:
            value = 2
        costs[k] = q.known(value, 'RAM mock')
    c = dict(costs=costs, constraints={k: q.known('RAM identical contract', 'RAM only') for k in q.CONSTRAINTS})
    return [deepcopy(c) for _ in q.ARMS]


def certificate(plan_sha):
    return dict(schema=q.STEM + '.freshness', authority='MAIN', producer='external_mock',
        plan_sha256=plan_sha, calibration_matrix_sha256=q.MATRIX_SHA,
        independent_roots=True, not_used_for_design=True, no_candidate_cost_or_outcome_seen=True,
        predeclared_order=True, complete_prior_root_exclusion_checked=True,
        excluded_calibration_task_sha256=list(q.TASKS), ordered_fresh_root_sha256=['a'*64, 'b'*64],
        issued_at_ns=10, expires_at_ns=30)


def test_default_queue_is_prospective_fixed_complete_blocked_and_nonmutating():
    m = fixture(); original = deepcopy(m)
    x = q.assemble(m, now_ns=20)
    assert m == original
    assert x['status'] == 'blocked_planning_artifact'
    assert x['ready'] is x['launch_authorized'] is x['equal_total_compute'] is False
    assert x['scientific_evidence'] is x['P1_admission'] is False
    assert x['executable_queue'] == [] and x['freshness']['status'] == 'blocked'
    assert x['equality']['status'] == 'unestablished'
    assert [c['algorithm'] for c in x['plan']['candidates']] == list(q.ARMS)
    assert b'OUTCOME_POISON' not in q.encode(x)
    assert q.assemble(m, now_ns=99)['plan_sha256'] == x['plan_sha256']
    for i, c in enumerate(x['plan']['candidates']):
        assert c['sealed_config'] == m['methods'][i]['contract_fields']
        assert c['prospective_budget']['value'] is None
        assert c['sealed_config_sha256'] == q.sha(q.encode(c['sealed_config']))


def test_real_unit_shapes_no_added_scalar_and_no_shared_allocation():
    x = q.assemble(fixture())
    c = [r['reference_contract']['costs'] for r in x['plan']['candidates']]
    assert [z['predict']['value']['completed'] for z in c] == [272, 268, 144, 168, 3604, 176]
    assert c[4]['AD_forward_in_predict']['value'] == 3600
    assert c[4]['scalar_loss_reverse_sweeps']['value'] == 1800
    assert c[5]['scalar_output_VJP_directions']['value']['completed'] == 384
    assert c[5]['scalar_loss_reverse_sweeps']['value'] is None
    for row in c:
        assert all(row[k]['value'] is None for k in q.EXTRA)
        assert row['pred_projector']['value'] is None
        assert 'total_compute' not in row
    assert x['plan']['shared_accounting']['physical_cohorts'][0]['counts']['physics_step']['completed'] == 1224
    assert x['plan']['shared_accounting']['inherited_encoding_shared_once']['value']['encoder']['completed'] == 8


def test_exact_single_arm_physics_only_not_walltime():
    m = fixture()
    m['methods'][3]['per_method_physics'] = q.known(
        dict(physics_step=dict(attempted=408, completed=408), public_step=dict(attempted=40, completed=40)), 'single arm')
    c = q.assemble(m)['plan']['candidates'][3]['reference_contract']['costs']
    assert c['physics_step']['value']['completed'] == 408
    assert c['public_step']['value']['completed'] == 40
    assert c['pipeline_walltime_seconds']['value'] is None


def test_equal_only_for_complete_matching_vector_not_predict_alone():
    contracts = complete_contracts()
    assert q.assess(contracts)['status'] == 'equal'
    contracts[1]['costs']['svd']['value']['attempted'] = 3
    x = q.assess(contracts)
    assert x['status'] == 'unestablished'
    assert any(g['reason'] == 'incomplete attempts' for g in x['gaps'])


@pytest.mark.parametrize('section,axis', [('costs', k) for k in q.UNITS] + [('constraints', k) for k in q.CONSTRAINTS])
def test_every_dimension_required(section, axis):
    c = complete_contracts()
    c[3][section].pop(axis)
    assert q.assess(c)['status'] == 'unestablished'


@pytest.mark.parametrize('axis', q.CONSTRAINTS)
def test_support_scope_replay_or_device_mismatch_blocks(axis):
    c = complete_contracts()
    c[4]['constraints'][axis] = q.known('different RAM contract', 'RAM')
    assert q.assess(c)['status'] == 'unestablished'


@pytest.mark.parametrize('bad', [True, -1, float('nan'), float('inf')])
def test_invalid_scalar_rejected(bad):
    c = complete_contracts(); c[0]['costs']['AD_forward_in_predict']['value'] = bad
    with pytest.raises(ValueError): q.assess(c)


@pytest.mark.parametrize('bad', [True, -1, 1.5])
def test_invalid_counter_rejected(bad):
    c = complete_contracts(); c[0]['costs']['predict']['value']['attempted'] = bad
    with pytest.raises(ValueError): q.assess(c)


def test_unknown_zero_is_not_accepted_and_cpu_vector_not_scalar():
    c = complete_contracts(); c[0]['costs']['svd'] = dict(status='unmeasured', value=0, reason='absent')
    with pytest.raises(ValueError): q.assess(c)
    c = complete_contracts(); c[0]['costs']['CPU_work']['value'] = 42
    with pytest.raises(ValueError): q.assess(c)


def test_identical_partial_cpu_vectors_do_not_establish_equal():
    c = complete_contracts()
    for row in c:
        row['costs']['CPU_work']['value']['Q_scores']['attempted'] = 3
    x = q.assess(c)
    assert x['status'] == 'unestablished'
    assert any(g['reason'] == 'incomplete CPU attempts' for g in x['gaps'])


def test_valid_independent_attestation_still_cannot_create_equal_budgets_or_launch():
    m = fixture(); cert = certificate(q.assemble(m)['plan_sha256'])
    x = q.assemble(m, cert, q.sha(q.encode(cert)), 'MAIN', 20)
    assert x['freshness']['status'] == 'verified_external_attestation'
    assert x['ready'] is False and x['executable_queue'] == []
    assert x['equality']['status'] == 'unestablished'


@pytest.mark.parametrize('edit', ['self', 'plan', 'calibration', 'reuse', 'duplicates', 'expired',
    'future', 'bool_time', 'outcome_seen', 'not_independent', 'exclusion', 'authority', 'missing_field', 'sha'])
def test_freshness_failures(edit):
    binding = q.assemble(fixture())['plan_sha256']; cert = certificate(binding)
    if edit == 'self': cert['producer'] = 'MAIN'
    elif edit == 'plan': cert['plan_sha256'] = 'c'*64
    elif edit == 'calibration': cert['calibration_matrix_sha256'] = 'c'*64
    elif edit == 'reuse': cert['ordered_fresh_root_sha256'][0] = q.TASKS[0]
    elif edit == 'duplicates': cert['ordered_fresh_root_sha256'][1] = 'a'*64
    elif edit == 'expired': cert['expires_at_ns'] = 20
    elif edit == 'future': cert['issued_at_ns'] = 21
    elif edit == 'bool_time': cert['issued_at_ns'] = True
    elif edit == 'outcome_seen': cert['no_candidate_cost_or_outcome_seen'] = False
    elif edit == 'not_independent': cert['independent_roots'] = False
    elif edit == 'exclusion': cert['excluded_calibration_task_sha256'] = []
    elif edit == 'authority': cert['authority'] = 'local'
    elif edit == 'missing_field': cert.pop('producer')
    h = 'f'*64 if edit == 'sha' else q.sha(q.encode(cert))
    assert q.freshness(cert, h, 'MAIN', binding, 20)['status'] == 'blocked'


def test_no_trust_anchor_no_certification_and_plan_change_invalidates_cert():
    m = fixture(); cert = certificate(q.assemble(m)['plan_sha256'])
    assert q.freshness(cert, None, None, cert['plan_sha256'], 20)['status'] == 'blocked'
    m['methods'][0]['contract_fields']['selection'] = 'changed'
    assert q.assemble(m, cert, q.sha(q.encode(cert)), 'MAIN', 20)['freshness']['status'] == 'blocked'


@pytest.mark.parametrize('edit', ['schema', 'order', 'missing_arm', 'old_task', 'admitted'])
def test_bad_matrix_rejected(edit):
    m = fixture()
    if edit == 'schema': m['schema'] = 'old'
    elif edit == 'order': m['methods'].reverse()
    elif edit == 'missing_arm': m['methods'].pop()
    elif edit == 'old_task': m['methods'][0]['root_receipts'][0]['task_sha256'] = 'c'*64
    else: m['equal_total_compute'] = True
    with pytest.raises(ValueError): q.assemble(m)


def test_cli_stdout_only_with_mock_metadata_and_sha_gate(monkeypatch):
    m = fixture(); raw = q.encode(m)
    monkeypatch.setattr(q, 'MATRIX_SHA', q.sha(raw))
    monkeypatch.setattr(q, 'read_json', lambda _: (m, raw))
    monkeypatch.setattr(q.Path, 'read_bytes', lambda _: b'RAM accounting source')
    monkeypatch.setattr(q, 'SOURCE_SHA', q.sha(b'RAM accounting source'))
    monkeypatch.setattr('sys.argv', ['queue'])
    with redirect_stdout(io.StringIO()) as out: q.main()
    assert json.loads(out.getvalue())['status'] == 'blocked_planning_artifact'
    monkeypatch.setattr(q, 'MATRIX_SHA', 'f'*64)
    with pytest.raises(ValueError, match='matrix SHA'): q.main()


def test_parser_rejects_duplicate_keys():
    with pytest.raises(ValueError): q.unique([('a', 1), ('a', 2)])
