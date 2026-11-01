"""RAM and real saved metadata seams only. No model/environment instantiation."""
from copy import deepcopy
import pytest
from experiments.spectral_reachability import agent_p2_resume_20261027_next as m


@pytest.fixture(scope='module')
def saved():
    assert m.prior.file_sha(m.DEFAULT_WITNESS) == m.WITNESS_SHA
    w = m.prior.read(m.DEFAULT_WITNESS)
    directory = m.Path(w['inputs']['composition']['directory'])
    banks = [m.prior.read(directory/f'root_{i}_bank.json') for i in range(4)]
    return w, banks


@pytest.fixture(scope='module')
def plan(saved):
    return m.build(*saved, m.native_contract())


def test_real_saved_frontier_keeps_320_old_and_480_future_separate(plan):
    assert plan['old_witness_candidate_denominator'] == 320
    assert plan['future_candidate_denominator'] == sum(len(s['candidates']) for s in plan['slots']) == 480
    assert len(plan['slots']) == 8
    assert [s['root_slot'] for s in plan['slots']] == [0,0,1,1,2,2,3,3]
    assert [s['arm'] for s in plan['slots']] == list(m.ARMS)*4
    assert plan['queue_audit']['status'] == 'blocked'
    for slot in plan['slots']:
        assert slot['task_id'] is slot['blind_sha256'] is None
        assert len(slot['candidates']) == m.COUNTS[slot['arm']]
        for c in slot['candidates']:
            if slot['arm'] == m.ARMS[1]: assert c['commands'] is None and len(c['prefix_commands']) == 5
            else: assert len(c['commands']) == 10
            assert (c['old_saved_receipt_sha256'] is None) == (c['index'] >= 40)
    assert plan['P2_admission'] is plan['execution_authorized'] is plan['scientific_pass'] is False


@pytest.mark.parametrize('damage', ['root','goal','bank','missing','duplicate','action','history','suffix'])
def test_real_layout_seam_mutations_rejected(saved, damage):
    w, banks = deepcopy(saved); g = w['program_graph']
    if damage == 'root': g['root_slots'].reverse()
    elif damage == 'goal': g['root_slots'][0]['identity']['goal_vector_sha256'] = 'a'*64
    elif damage == 'bank': banks[0]['logits'][0][0] += .1
    elif damage == 'missing': g['candidates'].pop()
    elif damage == 'duplicate': g['candidates'][1] = deepcopy(g['candidates'][0])
    elif damage == 'action': g['candidates'][0]['edges'][0]['actions'][0][0] += .1
    elif damage == 'history': g['root_slots'][0]['identity']['history_sha256'] = 'a'*64
    else: g['candidates'][40]['edges'][0]['actions'][0][0] += .1
    with pytest.raises(ValueError): m.build(w, banks, m.native_contract())


def test_base_edge_resource_closure_and_not_total_equality():
    r = m.allocation()
    assert r['paired_unique_predict_upper']*4 == r['cohort_unique_predict_upper'] == 1328
    assert r['standalone_predict_upper'] == [168,168]
    assert 168+168-r['graph_base_shared_upper']['predict_calls'] == 332
    for a in m.ARMS:
        slot = r['arms'][a]
        assert m.cost.add(*slot['edge_evaluation_upper']) == slot['per_candidate_upper']
        assert slot['allocation']['candidates'] == m.COUNTS[a]
        assert slot['allocation']['measured_cap_usage']['predict_calls'] == 168
        assert all(r['common_standalone_cap'][k] is None for k in m.cost.UNKNOWN)
    assert r['arms'][m.ARMS[0]]['allocation']['unspent']['jacobian_192x10_calls'] == 40
    assert r['full_resource_equal'] is r['padding'] is r['transfer_unused_budget'] is False
    assert r['inherited_acquisition_cost'] is r['root_collection_cost'] is None


def queue(plan, arm=0):
    s = plan['slots'][arm]
    return m.ArmQueue(s['arm'], s['identity'], 'b'*64, 'c'*64)


def complete(q):
    row = q.begin(); q.finish(row['upper'], 'e'*64); return row


@pytest.mark.parametrize('arm', [0,1])
def test_ram_normal_path_full_denominator_and_caps(plan, arm):
    q = queue(plan, arm)
    for _ in range(m.COUNTS[m.ARMS[arm]]+3): complete(q)
    r = q.receipt(); ledger = r['resource_ledger']
    assert ledger['reserved']['predict_calls'] == ledger['completed']['predict_calls'] == 168
    assert len(r['candidates']) == r['candidate_denominator'] == m.COUNTS[m.ARMS[arm]]
    assert not ledger['pending']
    with pytest.raises(ValueError, match='exhausted'): q.begin()
    with pytest.raises(ValueError): q.finish(m.cost.vector(), 'e'*64)


def test_partial_failure_never_refunds_or_retries_or_drops_candidates(plan):
    q = queue(plan,1); complete(q); candidate = q.begin()
    q.finish(m.cost.forward(1), 'e'*64, 'suffix AD failed')
    r = q.receipt(); ledger = r['resource_ledger']
    assert r['candidates'][0]['status'] == 'failed'
    assert all(c['status'] == 'skipped' for c in r['candidates'][1:])
    assert ledger['reserved']['predict_calls'] == 8 and ledger['completed']['predict_calls'] == 5
    assert ledger['reserved']['jacobian_192x10_calls'] == 1
    with pytest.raises(ValueError): q.begin()
    assert candidate['work_id'] == 'candidate_00'


def test_fatal_attempt_keeps_unknown_completion_and_full8_census(plan):
    qs = m.new_queues(plan,'c'*64)
    qs[0].begin(); qs[0].stop('fatal external stop')
    r = m.census_queues(plan,qs)
    assert len(r['rows']) == 8 and r['candidate_denominator'] == 480
    first = r['rows'][0]['resource_ledger']
    assert first['events'][0]['completed'] is None and first['pending'] == ['base']
    assert r['rows'][0]['attempted_completion_total'] is None
    assert first['reserved']['predict_calls'] == 4
    assert all(c['status'] == 'skipped' for c in r['rows'][0]['candidates'])
    assert all(c['status'] == 'pending' for c in r['rows'][1]['candidates'])
    for bad in (qs[:-1], qs[::-1], [qs[0]]*8):
        with pytest.raises(ValueError): m.census_queues(plan,bad)


@pytest.mark.parametrize('bad', ['unknown_finite','overrun','bool','missing','sha','error'])
def test_invalid_terminal_transaction_keeps_attempt_and_charge(plan,bad):
    q = queue(plan); row = q.begin(); before = q.receipt()
    value = deepcopy(row['upper']); sha = 'e'*64; error = None
    if bad == 'unknown_finite': value['flops'] = 0
    elif bad == 'overrun': value['predict_calls'] += 1
    elif bad == 'bool': value['predict_calls'] = True
    elif bad == 'missing': del value['svd_192x20_calls']
    elif bad == 'sha': sha = 'wrong'
    else: error = ''
    with pytest.raises(ValueError): q.finish(value,sha,error)
    assert q.receipt() == before


def test_zero_radius_is_actual_zero_but_reservation_retained(plan):
    q = queue(plan,1); complete(q); row=q.begin(); actual=deepcopy(row['upper'])
    actual['svd_192x10_calls'] = actual['bounded_solve_10_calls'] = 0
    q.finish(actual,'e'*64)
    r=q.receipt()['resource_ledger']
    assert r['reserved']['svd_192x10_calls'] == 1 and r['completed']['svd_192x10_calls'] == 0


def test_native_contract_preserves_full_success_not_object_projection():
    c=m.native_contract()
    assert c['success']['positions_include'] == 'agent_xy AND object_xy'
    assert '[:4]' in c['success']['position'] and c['success']['raw_angle_preserved']
    assert 'np.minimum(angle_diff, 2 * np.pi - angle_diff)' in c['native_methods']['eval_state']
    assert c['goal']['physical_goal7'] is None and c['goal']['mutation_allowed'] is False
    assert c['controller']['action_scale'] == 100 and c['controller']['native_step_clips'] is False
    assert c['restore']['currently_certified'] is False


def test_native_source_drift_rejected(monkeypatch):
    monkeypatch.setattr(m.prior,'file_sha',lambda p: 'a'*64)
    with pytest.raises(ValueError,match='PushT source drift'): m.native_contract()


def test_confirmatory_denial_before_reads(monkeypatch):
    monkeypatch.setattr(m.prior,'read',lambda p: pytest.fail('no read permitted'))
    with pytest.raises(ValueError,match='confirmatory blocked'): m.main(['run-saved','--confirmatory'])
    with pytest.raises(ValueError,match='confirmatory blocked'): m.consume(None,None,confirmatory=True)


def test_prepare_verify_saved_api_and_mutation_seals(saved,monkeypatch):
    # Native full exporter was already tested; RAM substitute only this expensive boundary.
    w,banks=saved
    monkeypatch.setattr(m.prior,'verify',lambda value,expected: value)
    p=m.prepare(); sha=m.digest(p)
    assert m.verify(p,sha) == p
    out=m.consume(p,sha)
    assert out['candidate_denominator'] == 480 and len(out['slots']) == 8
    assert out['new_neural_calls'] == out['new_physics_steps'] == 0
    bad=deepcopy(p); bad['plan']['resources']['common_standalone_cap']['flops']=0
    with pytest.raises(ValueError,match='prepare SHA'): m.verify(bad,sha)
    with pytest.raises(ValueError,match='resource drift'): m.verify(bad,m.digest(bad))


def test_output_stem_rejected_before_write(monkeypatch):
    monkeypatch.setattr(m,'prepare',lambda *args: {'RAM':True})
    monkeypatch.setattr(m.prior,'write',lambda *args: pytest.fail('must not write'))
    with pytest.raises(ValueError,match='NEXT output stem'):
        m.main(['prepare','--output','results/agent_p2_resume_20261027_OLD.json'])


def test_no_shared_standalone_double_count_claim():
    r=m.allocation()
    left=m.cost.candidate_recipe(m.ARMS[0]); right=m.cost.candidate_recipe(m.ARMS[1])
    totals=[]
    for recipe,n in ((left,80),(right,40)):
        totals.append(m.cost.add(recipe['fixed'], *[recipe['per_useful_candidate_cap']]*n))
    assert m.cost.compare(*totals)['predict_frontier_equal']
    assert not m.cost.compare(*totals)['full_resource_equal']
    with pytest.raises(ValueError,match='unsupported full_resource_equal'):
        m.cost.validate_claim(*totals,'full_resource_equal')
