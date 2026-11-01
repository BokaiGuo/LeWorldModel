"""RAM state machine and saved-layout mutations; NOT native scientific evidence."""
from copy import deepcopy
from functools import lru_cache
import json
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from experiments.spectral_reachability import agent_observed_CEM_physical_20261028_v1 as m
from experiments.spectral_reachability import agent_baselines_resume_20261027_next as producer
import test_moving_history_observed_suffix_physical_20261027_v1 as prior_tests


def fixture():
    _, data = prior_tests.fixture()
    branches, plans = [], []
    for n, old in enumerate(data['paired']):
        slot = old['root_slot']; prefix = old['arm']
        b = dict(source_slot=slot, arm=prefix, branch_id=str(n)*64, parent_task_id='task'+str(slot),
                 blind_sha256=data['references'][slot]['blind_sha256'], planner_selection_sha256='a'*64,
                 physical_selection_digest=old['original_selection_digest'], prefix_commands=old['original_commands'][:5],
                 observed_context=dict(latent_history=[[0.]*192]*3, action_history=[[.1]*10]*2),
                 prefix_npy=dict(path='/RAM/prefix'+str(n)+'.npy', file='prefix.npy', sha256='b'*64, dtype='float32', shape=[192]),
                 checkpoint=deepcopy(old['prefix'][5]))
        row = dict(suffix_commands=np.full((5, 2), .125 if n % 2 == 0 else -.125, np.float32).tolist(), selected_index=n, model_Q=1.+n)
        plans.append(dict(producer.model_plan(b, row), whole_run_report_sha256='f'*64)); branches.append(b)
    accepted = dict(schema=producer.NAME+'.consume', status='observed_CEM_commands_verified_not_admitted', errors=[],
                    plans=plans, P1_admission=False, P3_admission=False, equal_total_compute=False, new_neural_calls=0, new_physics_calls=0)
    pp = dict(schema=producer.NAME+'.prepare', branches=branches, bindings={})
    # Actual producer/consumer inputs are independently deserialized JSON: no aliases.
    accepted = json.loads(m.raw(accepted)); pp = json.loads(m.raw(pp))
    rows = m.validate_plans(accepted, pp, data, 'f'*64)
    p = dict(schema=m.NAME+'.prepare', plans=rows, prefixes=m.prefix_references(data), contract=m.contract(60), bindings={},
             sourcepack_sha256='e'*64)
    return p, data, accepted, pp


def execute(factory=prior_tests.ArrayEnv, *, p=None, data=None, store=None):
    fp, fd, _, _ = fixture(); p = fp if p is None else p; data = fd if data is None else data
    store = prior_tests.MemoryStore() if store is None else store; envs = []
    def make():
        env = factory(); envs.append(env); return env
    def fake_sha(path):
        assert Path(path).name == 'journal.jsonl'; return m.source.digest(store.events)
    forbidden = AssertionError('old execute/branch/witness/episode forbidden')
    with patch.object(m.moving, 'validate_env', lambda _: None), \
         patch.object(m.source, 'sample_state', lambda *a: prior_tests.oldtests.REQUEST.copy()), \
         patch.object(m, 'sha', fake_sha), patch.object(m.observed, 'branch', side_effect=forbidden), \
         patch.object(m.observed, 'execute', side_effect=forbidden), patch.object(m.base, 'execute', side_effect=forbidden), \
         patch.object(m.base, 'branch', side_effect=forbidden), patch.object(m.moving, 'episode', side_effect=forbidden), \
         patch.object(m.moving, 'witness', side_effect=forbidden):
        report = m.execute(p, data, store, make, authorization=m.AUTH)
    return report, store, envs, p


@lru_cache(maxsize=1)
def complete():
    r, s, envs, p = execute()
    assert r['status'] == m.RECORDED, r
    return r, s, envs, p


def reseal(events, report):
    previous = None
    for i, e in enumerate(events):
        e.update(index=i, previous_sha256=previous)
        e['event_sha256'] = m.source.digest({k:v for k,v in e.items() if k != 'event_sha256'}); previous = e['event_sha256']
    report['journal_chain_tail'] = previous


def test_fixed4_real_layout_and_own_new_endpoints():
    r, s, envs, p = complete()
    out = m.validate_saved(r, s.events, p)
    assert out['status'] == 'saved_physical_verified_not_admitted'
    expected = dict(reset=8, public_step=80, explicit_render=88, goal_render=8, set_state=16, physics_step=816, goal_bind=4, constructor=4, close=4, goal_setter=12)
    assert r['counts'] == {k:dict(attempted=v,completed=v) for k,v in expected.items()}
    assert r['intents'] == expected and r['full_suffixes'] == 4 and r['engineering_failures'] == r['not_started'] == 0
    for row, env, plan in zip(r['rows'], envs, p['plans'], strict=True):
        assert env.closed and len(env.resets) == 2 and len(env.commands) == 20
        assert m.moving.exact(np.asarray(env.commands[:10]), m.moving.warmup_commands())
        assert m.moving.exact(np.asarray(env.commands[10:]), m.base.commands(plan['commands']))
        assert [c['step'] for c in row['checkpoints']] == [0, 5, 10]
        assert len(row['final_state']) == 7 and 'saved_old_terminal' not in row
        assert 'reset' not in env.__dict__ and '_set_state' not in env.__dict__
        assert '_set_goal_state' not in env.__dict__
        assert np.array_equal(env.goal_pose, [300, 300, 0])
    assert not out['scientific_admission'] and not out['equal_FLOPs']


@pytest.mark.parametrize('field', ['order','arm','slot','branch','task','parent','blind','context','npy','checkpoint','physical','prefix','suffix','report','selection','status','count','missing','q'])
def test_producer_consume_abi_mutations_fail_closed(field):
    _, d, c, pp = fixture(); x = c['plans'][0]
    if field == 'order': c['plans'][0], c['plans'][1] = c['plans'][1], c['plans'][0]
    elif field == 'arm': x['prefix_arm'] = 'line_search_GN_other'
    elif field == 'slot': x['source_slot'] = True
    elif field == 'branch': x['branch_id'] = 'bad'
    elif field == 'task': x['parent_task_id'] = 'bad'
    elif field == 'parent': x['parent_selection_sha256'] = 'bad'
    elif field == 'blind': x['blind_sha256'] = 'bad'
    elif field == 'context': x['observed_context_sha256'] = 'bad'
    elif field == 'npy': x['prefix_npy']['sha256'] = 'bad'
    elif field == 'checkpoint': pp['branches'][0]['checkpoint']['state'][0] += 1
    elif field == 'physical': x['physical_prefix_selection_digest'] = 'bad'
    elif field == 'prefix': x['commands'][0][0] = .5
    elif field == 'suffix': x['suffix_commands'][0][0] = .5
    elif field == 'report': x['whole_run_report_sha256'] = 'a'*64
    elif field == 'selection': x['selection_sha256'] = 'bad'
    elif field == 'status': c['status'] = 'commands_released'
    elif field == 'count': c['new_physics_calls'] = 1
    elif field == 'missing': c['plans'].pop()
    else: x['model_Q'] = float('nan')
    if field not in ('selection','q'):
        x['selection_sha256'] = m.core.digest({k:v for k,v in x.items() if k not in ('selection_sha256','whole_run_report_sha256')})
    with pytest.raises((ValueError,KeyError)): m.validate_plans(c, pp, d, 'f'*64)


@pytest.mark.parametrize('field', ['history', 'state', 'body', 'signedzero', 'goal', 'prefixpng'])
def test_exact_recipe_certification_never_savedpoststate(field):
    p, d, _, _ = fixture()
    if field == 'history': d['references'][0]['history'][1,2,2,0] ^= 1
    elif field == 'state': d['references'][0]['states'][1,0] += 1e-10
    elif field == 'body': d['references'][0]['dynamics'][2]['bodies']['agent']['velocity'][0] += 1e-10
    elif field == 'signedzero': d['paired'][0]['prefix'][5]['snapshot']['state_float64_bits'] = 'bad'
    elif field == 'goal': d['paired'][0]['goal'][0] += 1
    else: d['paired'][0]['prefix'][5]['file_sha256'] = 'bad'
    p['prefixes'] = m.prefix_references(d)
    r, _, envs, _ = execute(p=p,data=d)
    assert r['status'] == 'partial_or_failed' and len(r['rows']) == 4
    assert len(envs[0].commands) <= 15 and not r['rows'][0]['prefix_exact']
    assert all(e.closed for e in envs)


@pytest.mark.parametrize('fault', ['terminal','truncate'])
def test_native_suffix_early_stop_retains4_and_no_padding(fault):
    r, s, envs, p = execute(lambda: prior_tests.ArrayEnv(fault=fault, at=16))
    assert r['status'] == m.RECORDED and r['early_stops'] == 4 and r['full_suffixes'] == 0
    assert all(len(e.commands) == 16 and e.closed for e in envs)
    assert r['counts']['physics_step']['completed'] == 656
    assert m.validate_saved(r, s.events, p)['status'] == 'saved_physical_verified_not_admitted'


@pytest.mark.parametrize('at', [5,12,15])
def test_pre_suffix_stop_or_substitution_cannot_release(at):
    r, _, envs, _ = execute(lambda: prior_tests.ArrayEnv(fault='substitute', at=at))
    assert r['status'] == 'partial_or_failed' and all(len(e.commands) == at for e in envs)
    assert all(not r['prefix_exact'] for r in r['rows'])


def test_step_failure_actual_attempt_no_refund_and_cleanup():
    r, s, envs, p = execute(lambda: prior_tests.ArrayEnv(fault='throw', at=16, close_error=True))
    assert r['status'] == 'partial_or_failed' and all(e.closed for e in envs)
    assert r['counts']['public_step'] == dict(attempted=64,completed=60)
    assert all(x['own_attempted'] == 6 and x['own_returned'] == 5 and x['prefix_exact'] for x in r['rows'])
    assert r['counts']['close'] == dict(attempted=4,completed=0)
    assert m.validate_saved(r, s.events, p)['status'] == 'saved_partial_verified_not_admitted'


@pytest.mark.parametrize('field', ['state', 'goal', 'PNG', 'commands', 'guard', 'admission', 'metric', 'rowcount', 'counter', 'scope', 'terminal', 'warmbits', 'oldendpoint', 'earlysummary', 'hashchain'])
def test_saved_validator_mutations_even_with_resealed_chain(field):
    r0,s,_,p0 = complete(); r,p,events = deepcopy(r0),deepcopy(p0),deepcopy(s.events)
    if field == 'state': next(e for e in events if e['kind']=='fresh_observation' and e['receipt']['phase']=='plan' and e['receipt']['step']==5)['receipt']['state'][0] += 1e-10
    elif field == 'goal': next(e for e in events if e['kind']=='returned_transition' and e['phase']=='plan')['returned_goal_state'][0] += 1
    elif field == 'PNG': r['rows'][0]['PNG'] -= 1
    elif field == 'commands': next(e for e in events if e['kind']=='operation_attempted' and e['resource']=='public_step' and e.get('phase')=='plan')['raw_action'][0] += .1
    elif field == 'guard': next(e for e in events if e['kind']=='guard_receipt')['accepted'] = False
    elif field == 'admission': events[:] = [e for e in events if e['kind']!='suffix_admitted_after_exact_prefix']
    elif field == 'metric': r['rows'][0]['final_metrics']['native_distance'] = 0
    elif field == 'rowcount': r['rows'].pop()
    elif field == 'counter': r['counts']['physics_step']['completed'] -= 1
    elif field == 'scope': next(e for e in events if e.get('branch')==0)['prefix_arm'] = 'other'
    elif field == 'terminal': next(e for e in events if e['kind']=='returned_transition' and e['phase']=='plan')['terminated'] = True
    elif field == 'warmbits': next(e for e in events if e['kind']=='warmup_return')['raw_float32_bits'] = '00'
    elif field == 'oldendpoint': r['rows'][0]['saved_old_terminal'] = {'state':[0]*7}
    elif field == 'earlysummary': r['early_stops'] = 1
    else: events[3]['event_sha256'] = '0'*64
    if field != 'hashchain': reseal(events, r)
    with pytest.raises((ValueError,KeyError)): m.validate_saved(r, events, p)


def test_authorization_and_caps_before_any_callback():
    p,d,_,_ = fixture(); store = prior_tests.MemoryStore(); ops = m.Ops(p,store)
    p['contract']['caps']['public_step'] = 0
    with pytest.raises(ValueError): ops.call('public_step',lambda:pytest.fail('dispatch'))
    ops.started -= 1000
    with pytest.raises(ValueError): ops.call('constructor',lambda:pytest.fail('dispatch'))
    with pytest.raises(ValueError): m.execute(p,d,store,lambda:pytest.fail('factory'),authorization='wrong')
    with pytest.raises(ValueError): m.run('x','x','x',lambda:pytest.fail('factory'),authorization='wrong')


def test_constructor_failure_keeps_unknowns_and4_denominator():
    def fail(): raise RuntimeError('constructor partial internal work unknown')
    r, _, envs, _ = execute(fail)
    assert len(r['rows']) == 4 and r['engineering_failures'] == 4 and not envs
    assert r['counts']['constructor'] == dict(attempted=4,completed=0) and r['counts']['close']['attempted'] == 0
    assert all(x['native_success'] is x['final_state'] is x['final_metrics'] is None for x in r['rows'])


@pytest.mark.parametrize('event', ['operation_intent','operation_attempted','operation_completed'])
def test_constructor_return_journal_fault_still_closes(event):
    s = prior_tests.MemoryStore(); emit = s.emit; fired=[]
    def fail(kind, **kw):
        if kind == event and kw.get('resource')=='constructor' and not fired:
            fired.append(1); raise OSError('journal write fault')
        emit(kind, **kw)
    s.emit = fail
    r, _, envs, _ = execute(store=s)
    assert r['status'] == 'partial_or_failed' and all(e.closed for e in envs)
    assert len(envs) == (4 if event=='operation_completed' else 3)


def test_cleanup_close_runs_despite_failed_cleanup_intent():
    s=prior_tests.MemoryStore(); emit=s.emit
    def fail(kind, **kw):
        if kind=='operation_intent' and kw.get('resource')=='close': raise OSError('no cleanup journal')
        emit(kind, **kw)
    s.emit=fail; r,_,envs,_=execute(store=s)
    assert r['status']=='partial_or_failed' and all(e.closed for e in envs)
    assert r['counts']['close']==dict(attempted=4,completed=4)


def test_actual_return_meter_survives_return_journal_error_without_refund():
    p,_,_,_=fixture(); s=prior_tests.MemoryStore(); ops=m.Ops(p,s)
    emit=s.emit; callbacks=[]
    def fail(kind,**kw):
        if kind=='operation_completed':raise OSError('after return journal')
        emit(kind,**kw)
    s.emit=fail
    with pytest.raises(OSError):ops.call('public_step',lambda:callbacks.append('returned'))
    assert callbacks==['returned'] and ops.counts['public_step']==dict(attempted=1,completed=1)
    assert ops.intents['public_step']==1 and not ops.active


@pytest.mark.parametrize('fault',['none','file','pixel','shape','mode'])
def test_actual_png_bytes_and_pixels_memory_seam(fault):
    from PIL import Image
    arr=np.zeros((224,224,3),np.uint8)
    if fault=='shape':arr=np.zeros((224,223,3),np.uint8)
    if fault=='mode':arr=np.zeros((224,224),np.uint8)
    stream=io.BytesIO();Image.fromarray(arr).save(stream,format='PNG');payload=stream.getvalue()
    receipt=dict(file_sha256=hashlib.sha256(payload).hexdigest(),pixel_sha256=m.moving.image_sha(arr))
    if fault=='file':receipt['file_sha256']='0'*64
    if fault=='pixel':receipt['pixel_sha256']='0'*64
    if fault=='none':m.validate_png(payload,receipt)
    else:
        with pytest.raises(ValueError):m.validate_png(payload,receipt)


def test_no_old_endpoints_in_execution_inputs():
    p,d,_,_ = fixture()
    for x in d['paired']:
        x['old_terminal'] = {'state':[float('nan')]*7}; x['old_final_metrics'] = {'native_distance':-999}
    r,_,_,_ = execute(p=p,data=d)
    assert r['status']==m.RECORDED and all(x['final_metrics']['native_distance']>0 for x in r['rows'])


@pytest.mark.parametrize('fault', ['setter','physics','render','pose'])
def test_offline_goal_cannot_restore_or_add_work(fault):
    prior_tests.original_fixture(); original=m.moving.attach_goal
    def attach(body, goal, rgb):
        original(body,goal,rgb)
        if fault=='setter':body._set_state(body._get_obs())
        elif fault=='physics':body.space.step(.01)
        elif fault=='render':body.render()
        else:body.goal_pose[0]+=1
    with patch.object(m.moving,'attach_goal',attach):r,_,envs,_=execute()
    assert r['status']=='partial_or_failed' and all(len(e.commands)==10 and e.closed for e in envs)


def authority_fixture():
    p,d,c,pp = fixture(); pp['bindings']={}; d['bindings']={}
    a=dict(compiler_output='/RAM/CUDA',compiler_report_sha256='f'*64,producer_source_sha256='a'*64,consumer_source_sha256='a'*64,
           consumer_report='/RAM/consume.json',consumer_report_sha256='b'*64,
           consumer_wrapper_source='/RAM/consume_observed_CEM_MAIN_20261028_v1.py',consumer_wrapper_source_sha256='c'*64,
           original_physical_report='/RAM/old/report.json',original_physical_report_sha256=m.observed.ORIGINAL_SHA,wall_seconds=600)
    report=dict(prepared='/RAM/CUDA/prepare.json',prepare_sha256='d'*64,artifact_sha256={})
    d['original_prepared']['sourcepack_sha256']='e'*64
    def read(path): return c if Path(path).name=='consume.json' else report
    def sha(path):
        path=str(path)
        if path.endswith('/agent_baselines_resume_20261027_next.py'): return 'a'*64
        if path==a['consumer_report']:return a['consumer_report_sha256']
        if path==a['consumer_wrapper_source']:return a['consumer_wrapper_source_sha256']
        if path==str(Path(m.observed.__file__).resolve()):return m.HELPER_SHA
        if path=='/RAM/CUDA/report.json':return a['compiler_report_sha256']
        if path==a['original_physical_report']:return m.observed.ORIGINAL_SHA
        if path==report['prepared']:return report['prepare_sha256']
        return 'e'*64
    return a,d,c,pp,report,read,sha


def checked_material(a,d,c,pp,read,sha):
    module=SimpleNamespace(consume=lambda *args:deepcopy(c),verify=lambda *args:deepcopy(pp))
    with patch.object(m,'sha',side_effect=sha),patch.object(m,'read',side_effect=read),\
         patch.object(m.importlib,'import_module',return_value=module),patch.object(m.observed,'load_original',return_value=d):
        return m.material(a)


def test_actual_authority_layout_without_future_sha_or_execution():
    a,d,c,pp,_,read,sha=authority_fixture(); p=checked_material(a,d,c,pp,read,sha)
    assert len(p['plans'])==4 and p['arguments']==a and a['consumer_wrapper_source'] in p['bindings']


@pytest.mark.parametrize('fault',['missing','zero','wrongsource','wrongwrapper','oldphysical','wrapjson','reportsha','sourcehash','consumehash'])
def test_external_actual_authority_required(fault):
    a,d,c,pp,r,read,sha=authority_fixture()
    if fault=='missing':del a['consumer_report_sha256']
    elif fault=='zero':a['compiler_report_sha256']='0'*64
    elif fault=='wrongsource':a['consumer_source_sha256']='e'*64
    elif fault=='wrongwrapper':a['consumer_wrapper_source']='/RAM/other.py'
    elif fault=='oldphysical':a['original_physical_report_sha256']='e'*64
    elif fault=='wrapjson':
        read=lambda path:dict(c,MAIN_metadata=True) if Path(path).name=='consume.json' else r
    elif fault=='reportsha':a['compiler_report_sha256']='e'*64
    else:
        real=sha
        sha=lambda path:'9'*64 if (fault=='sourcehash' and Path(path).name==m.PRODUCER+'.py') or (fault=='consumehash' and str(path)==a['consumer_report']) else real(path)
    with pytest.raises((ValueError,KeyError)):checked_material(a,d,c,pp,read,sha)
