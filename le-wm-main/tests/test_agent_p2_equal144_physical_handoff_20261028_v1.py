"""RAM arrays/callbacks only; not physical or scientific evidence. No real prepare."""
from copy import deepcopy
import io
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p2_equal144_physical_handoff_20261028_v1 as m
from test_p1_constructive_moving_history_physical_20261027_v1 import ArrayEnv, fixture as moving_fixture, REQUEST
from test_p1_constructive_plan_physical_20261027_v1 import MemoryStore


def fixture():
    _, data = moving_fixture(); plans = []
    for s in (0, 1):
        for a in m.ARMS:
            p = dict(root_slot=s, root_sha256=data['references'][s]['root_sha256'], arm=a,
                     commands=np.full((10, 2), .25, np.float32).tolist())
            p['raw_float32_bits'] = m.bits(p['commands'])
            p['selection_digest'] = m.source.digest(p); plans.append(p)
    return dict(plans=plans, contract=m.contract(), bindings={}, sourcepack_sha256=m.SOURCE_SHA,
                source_prepare_id='b'*64), data


def execute(factory=ArrayEnv, *, p=None, data=None, store=None):
    fp, fd = fixture(); p = fp if p is None else p; data = fd if data is None else data
    store = MemoryStore() if store is None else store; envs = []
    def make(): e = factory(); envs.append(e); return e
    def digest(path):
        assert Path(path).name == 'journal.jsonl'
        return m.source.digest(store.events)
    with patch.object(m.kernel.moving, 'validate_env', lambda _: None), \
         patch.object(m.source, 'sample_state', lambda *a: REQUEST.copy()), patch.object(m, 'sha', digest), \
         patch.object(m.helper, 'execute', side_effect=AssertionError('old execute forbidden')), \
         patch.object(m.kernel.moving, 'episode', side_effect=AssertionError('source episode forbidden')), \
         patch.object(m.kernel.moving, 'witness', side_effect=AssertionError('witness forbidden')):
        result = m.execute(p, data, store, make, authorization=m.AUTH,
                           execution_binding=dict(prepared='/RAM/prepare.json', prepare_sha256='a'*64))
    return result, store, envs


def test_complete_four_NEW_branches_even_identical_commands():
    r, store, envs = execute()
    assert r['status'] == 'equal144_physical_recorded_not_admitted', r['errors']
    c = dict(reset=8, set_state=16, goal_render=8, goal_bind=4, public_step=80, physics_step=816,
             explicit_render=88, environment_construct=4, environment_close=4)
    assert r['counts'] == {k: dict(attempted=v, completed=v) for k, v in c.items()}
    assert len(envs) == r['fixed_branch_denominator'] == r['full_horizons'] == 4
    assert r['engineering_failures'] == r['early_stops'] == r['not_started'] == 0
    for row, env in zip(r['rows'], envs, strict=True):
        assert env.closed and len(env.commands) == 20 and row['initial_exact']
        assert m.kernel.moving.exact(np.asarray(env.commands[:10]), m.kernel.moving.warmup_commands())
        assert m.kernel.moving.exact(np.asarray(env.commands[10:]), np.asarray(row['selected_plan']['commands'], np.float32))
        assert [c['step'] for c in row['checkpoints']] == [0, 5, 10]
        assert all(len(c['state']) == 7 and c['png'] in store.files for c in row['checkpoints'])
        assert row['own_attempted'] == row['own_returned'] == row['warmup_returned'] == 10
        assert len(row['final_state']) == 7 and row['final_metrics']['object_success'] is None
        assert np.array_equal(env.goal_pose, [300., 300., 0.])
        assert 'reset' not in env.__dict__ and '_set_state' not in env.__dict__
    for k in ('P1_admission', 'P2_admission', 'P3_admission', 'scientific_admission', 'independent_confirmation',
              'equal_FLOPs', 'equal_total_compute', 'compute_advantage', 'walltime_advantage'):
        assert r[k] is False
    assert r['new_neural_calls'] == r['new_independent_samples'] == 0
    previous = None
    for e in store.events:
        assert e['previous_sha256'] == previous
        assert e['event_sha256'] == m.source.digest({k: v for k, v in e.items() if k != 'event_sha256'})
        previous = e['event_sha256']
    assert previous == r['journal_chain_tail']


@pytest.mark.parametrize('fault', ['terminal', 'truncate', 'guard', 'render', 'substitute', 'throw'])
def test_native_earlystop_engineering_failure_never_pad(fault):
    r, store, envs = execute(lambda: ArrayEnv(fault=fault))
    assert len(r['rows']) == 4 and all(e.closed and len(e.commands) == 11 for e in envs)
    assert r['full_horizons'] == 0 and all([c['step'] for c in row['checkpoints']] == [0] for row in r['rows'])
    if fault in ('terminal', 'truncate'):
        assert r['status'] == 'equal144_physical_recorded_not_admitted' and r['early_stops'] == 4
    else: assert r['status'] == 'partial_or_failed' and r['engineering_failures'] == 4
    assert all(row['own_attempted'] == 1 and row['own_returned'] == (0 if fault == 'throw' else 1) for row in r['rows'])
    if fault == 'throw': assert r['counts']['public_step'] == dict(attempted=44, completed=40)
    else: assert r['counts']['physics_step']['completed'] == 456
    if fault in ('guard', 'render', 'substitute'):
        assert all(row['last_observation']['png'] in store.files for row in r['rows'])


@pytest.mark.parametrize('field', ['raw', 'RGB', 'state', 'snapshot', 'goal'])
def test_precise_original_history_goal_binding_before_dispatch(field):
    p, d = fixture(); ref = d['references'][0]
    if field == 'raw': ref['raw_history'] = np.zeros((10, 2), np.float32)
    elif field == 'RGB': ref['history'][1, 0, 0, 0] ^= 1
    elif field == 'state': ref['states'][1, 0] += 1e-8
    elif field == 'snapshot': ref['dynamics'][1]['state_float64_bits'] = 'bad'
    else: ref['goal'][0] += 1
    r, _, _ = execute(p=p, data=d)
    assert all(row['own_attempted'] == 0 and not row['initial_exact'] for row in r['rows'][:2])
    assert all(row['full_horizon'] for row in r['rows'][2:])


@pytest.mark.parametrize('branch', range(4))
@pytest.mark.parametrize('fault', ['constructor', 'close', 'step', 'truncate'])
def test_each_one_failure_keeps_ALL_four_denominator(branch, fault):
    index = 0
    def factory():
        nonlocal index
        i = index; index += 1
        if i == branch:
            if fault == 'constructor': raise RuntimeError('partial constructor unknown')
            if fault == 'close': return ArrayEnv(close_error=True)
            return ArrayEnv(fault='throw' if fault == 'step' else 'truncate')
        return ArrayEnv()
    r, _, envs = execute(factory)
    assert index == len(r['rows']) == 4 and all(e.closed for e in envs)
    assert all(row['full_horizon'] for i, row in enumerate(r['rows']) if i != branch)
    row = r['rows'][branch]
    if fault == 'constructor':
        assert row['final_state'] is row['native_success'] is row['final_metrics'] is None
        assert row['constructor_internal_work'] == 'unknown'
        assert r['total_internal_physics_scope'] == 'unknown_constructor_work'
        assert r['counts']['environment_construct'] == dict(attempted=4, completed=3)
    elif fault == 'close': assert row['full_horizon'] and row['first_error']['stage'] == 'close'
    elif fault == 'step': assert row['own_attempted'] == 1 and row['own_returned'] == 0
    else: assert row['own_returned'] == 1 and row['status'] == 'native_stopped'


@pytest.mark.parametrize('stage', ['construct_return', 'step_intent', 'step_return', 'branch_closed', 'run_finished', 'report'])
def test_durable_failure_retains_actual_entries_returns_and_cleanup(stage):
    store = MemoryStore(); emit, put = store.emit, store.put
    def event(kind, **v):
        rec = v.get('receipt', {})
        fail = ((stage == 'construct_return' and kind == 'operation_returned' and rec.get('resource') == 'environment_construct') or
                (stage == 'step_intent' and kind == 'operation_dispatch_intent' and rec.get('phase') == 'plan') or
                (stage == 'step_return' and kind == 'operation_returned' and rec.get('phase') == 'plan') or stage == kind)
        if fail: raise OSError('mock durable write failure')
        return emit(kind, **v)
    def final(name, value, final=False):
        if stage == 'report' and final: raise OSError('mock final report failure')
        return put(name, value, final)
    store.emit, store.put = event, final
    r, _, envs = execute(store=store)
    assert r['status'] == 'partial_or_failed' and len(r['rows']) == 4 and all(e.closed for e in envs)
    if stage == 'step_intent': assert r['rows'][0]['own_attempted'] == r['rows'][0]['own_returned'] == 0
    if stage == 'step_return': assert r['rows'][0]['own_attempted'] == r['rows'][0]['own_returned'] == 1
    if stage == 'branch_closed': assert len(envs) == 1 and r['not_started'] == 3
    if stage == 'construct_return': assert len(envs) == 1 and r['counts']['environment_close']['completed'] == 1


@pytest.mark.parametrize('fault', [None, 'terminal', 'truncate', 'throw', 'guard'])
def test_downstream_assemble_actual7_checkpoints_and_partial_denominator(fault):
    p, d = fixture(); r, store, _ = execute(lambda: ArrayEnv(fault=fault), p=p, data=d)
    cp = m.assemble(r, store.events, p)
    assert len(r['rows']) == 4 and len(cp) == (12 if fault is None else 4)
    assert [(c['root_slot'], c['arm'], c['checkpoint']['step']) for c in cp] == [
        (s, a, step) for s in (0, 1) for a in m.ARMS for step in ((0, 5, 10) if fault is None else (0,))]
    assert all(len(c['checkpoint']['state']) == 7 for c in cp)
    assert all(c['selected_plan']['root_sha256'] == d['references'][c['root_slot']]['root_sha256'] for c in cp)


@pytest.mark.parametrize('fault', ['meter', 'receipt', 'raw', 'root', 'goal', 'state6', 'step', 'plan', 'guard', 'denominator', 'native', 'terminal'])
def test_downstream_assemble_rejects_cross_join_missing_axes_or_replacement(fault):
    p, d = fixture(); r, store, _ = execute(p=p, data=d); events = deepcopy(store.events)
    if fault == 'meter': r['counts']['physics_step']['completed'] -= 1
    elif fault == 'receipt': r['actual_operation_receipts'].pop()
    elif fault == 'raw':
        planstep = next(x for x in r['actual_operation_receipts'] if x.get('phase') == 'plan' and x['resource'] == 'public_step')
        planstep['raw_action'][0] = 0.
    elif fault == 'root': r['rows'][0]['root_state'][0] += 1
    elif fault == 'goal': r['rows'][0]['goal'][0] += 1
    elif fault == 'state6': r['rows'][0]['checkpoints'][1]['state'].pop()
    elif fault == 'step': r['rows'][0]['checkpoints'][1]['step'] = 4
    elif fault == 'plan': r['rows'][0]['selected_plan']['commands'][0][0] = 0.
    elif fault == 'guard':
        next(e for e in events if e['kind'] == 'guard_receipt' and e['phase'] == 'plan')['accepted'] = False
    elif fault == 'denominator': r['rows'].pop()
    elif fault == 'native': r['rows'][0]['final_metrics']['native_distance'] += 1
    else: events.pop()
    with pytest.raises(ValueError): m.assemble(r, events, p)


def test_downstream_consume_SHA_gate_before_any_prepare_source_or_images():
    with patch.object(Path, 'iterdir', return_value=[]), patch.object(m, 'sha', return_value='a'*64), \
         patch.object(m, 'verify', side_effect=AssertionError('no unchecked verify')), \
         patch.object(m, 'read', side_effect=AssertionError('no unchecked read')), pytest.raises(ValueError, match='external actual physical'):
        m.consume('/MOCK/output', 'b'*64)


def test_no_authorization_no_factory_or_io():
    with patch.object(m,'verify',side_effect=AssertionError('must not read')), pytest.raises(ValueError):
        m.run('/RAM/no','a'*64,'/RAM/out',lambda:pytest.fail('factory'),authorization='bad')


def test_actual_saved_material_prepare_verify_size_and_no_neural(tmp_path):
    # Actual saved authorities, not fake counters or real simulator.
    assert m.sha(m.REPORT)==m.REPORT_SHA
    assert m.sha(m.RECEIPT)==m.RECEIPT_SHA
    path=tmp_path/'prepare.json'
    result=m.prepare(path)
    p=m.verify(str(path),result['prepare_sha256'])
    assert result['bytes']==path.stat().st_size <= m.PREPARE_CAP
    assert [(x['root_slot'],x['arm']) for x in p['plans']]==[(s,a) for s in (0,1) for a in m.ARMS]
    assert all(x['raw_float32_bits']==m.bits(x['commands']) for x in p['plans'])
    assert p['inherited_resources']['predict_API']==p['inherited_resources']['predictor_samples']==576
    assert p['source_inputs']==m.SOURCE_INPUTS
    assert not any(n in sys.modules for n in ('torch','gymnasium','stable_worldmodel','pymunk'))
    with pytest.raises(ValueError):m.prepare(path)


@pytest.mark.parametrize('fault',['receipt','report','plan','slot','source','raw'])
def test_actual_saved_reject_mutation_without_modifying_input(fault):
    original_read=m.read
    def read(path,cap=m.PREPARE_CAP):
        v=original_read(path,cap)
        if Path(path)==m.RECEIPT:
            if fault=='receipt':v['rows'].pop()
            if fault=='plan':v['native_strict_receipt']['plans'][0]['commands'][0][0]+=.01
            if fault=='slot':v['native_strict_receipt']['plans'][0]['source_slot']=1
            if fault=='raw':v['native_strict_receipt']['plans'][0]['commands'][0][0]=-0.0
        if Path(path)==m.REPORT and fault=='report':v['prepare_sha256']='0'*64
        return v
    if fault=='source':
        with patch.object(m,'sha',return_value='0'*64),pytest.raises(ValueError):m.ownconsume()
    else:
        with patch.object(m,'read',side_effect=read),pytest.raises(ValueError):m.ownconsume()


def test_actual_task_bridge_rejects_context_goal_raw_swap():
    rows,bindings,_=m.ownconsume()
    d=m.kernel.load_source(*m.SOURCE_INPUTS.values())
    assert len(m.task_bridge(rows[0],d['references'][0],dict(bindings)))==64
    for field in ('raw_history','history','goal_rgb','blind_sha256'):
        ref=deepcopy(d['references'][0])
        if field=='blind_sha256':ref[field]='0'*64
        elif ref[field].dtype==np.uint8:ref[field].flat[0]^=1
        else:ref[field].flat[0]+=1
        with pytest.raises(ValueError):m.task_bridge(rows[0],ref,dict(bindings))


@pytest.mark.parametrize('tamper',[None,'PNG','journal','report'])
def test_real_producer_layout_full_consume_RAM_only(tmp_path,tamper):
    p,d=fixture();prepared=tmp_path/'prepare.json';prepared.write_bytes(m.raw(p))
    binding=dict(prepared=str(prepared),prepare_sha256=m.sha(prepared))
    out=tmp_path/'physical';store=m.Store(str(out));store.put('execution_binding.json',binding)
    with patch.object(m.kernel.moving,'validate_env',lambda _:None), \
         patch.object(m.source,'sample_state',lambda *a:REQUEST.copy()):
        result=m.execute(p,d,store,ArrayEnv,authorization=m.AUTH,execution_binding=binding)
    assert result['status']=='equal144_physical_recorded_not_admitted'
    report_sha=m.sha(out/'report.json')
    if tamper=='PNG':
        path=next(out.glob('*.png'));path.write_bytes(path.read_bytes()+b'changed')
    elif tamper=='journal':
        path=out/'journal.jsonl';path.write_bytes(path.read_bytes()+b'{}\n')
    elif tamper=='report':
        path=out/'report.json';path.write_bytes(path.read_bytes()+b' ')
    with patch.object(m,'verify',return_value=p):
        if tamper:
            with pytest.raises(ValueError):m.consume(str(out),report_sha)
        else:
            c=m.consume(str(out),report_sha)
            assert len(c['rows'])==c['fixed_branch_denominator']==4
            assert len(c['checkpoints'])==12 and c['checkpoint_complete'] is True
            assert c['new_neural_calls']==0 and c['scientific_admission'] is False


def test_raw_signed_zero_mutation_rejected_in_saved_meter():
    p,d=fixture()
    for plan in p['plans']:
        plan['commands'][0][0]=0.0
        plan['raw_float32_bits']=m.bits(plan['commands'])
        plan['selection_digest']=m.source.digest({k:v for k,v in plan.items() if k!='selection_digest'})
    r,store,_=execute(p=p,data=d)
    events=deepcopy(store.events)
    for e in events:
        rec=e.get('receipt',{})
        if rec.get('phase')=='plan' and rec.get('resource')=='public_step' and rec.get('step')==1:
            rec['raw_action'][0]=-0.0
    for rec in r['actual_operation_receipts']:
        if rec.get('phase')=='plan' and rec.get('resource')=='public_step' and rec.get('step')==1:
            rec['raw_action'][0]=-0.0
    with pytest.raises(ValueError,match='bits'):m.assemble(r,events,p)

