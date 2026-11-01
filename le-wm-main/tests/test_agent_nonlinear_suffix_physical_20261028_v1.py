"""RAM metadata and native-array mocks only; no model, physics or production consume."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import agent_nonlinear_suffix_physical_20261028_v1 as m
from test_agent_allmode_physical_20261028_v1 import fixture as old_fixture, MemoryStore, Env, fake_place, h, reseal
from test_agent_allmode_prefix_20261028_v1 import slot as model_slot


def fixture():
    prior, data, _, original, _ = old_fixture()
    original['schema'] = m.suffix.source.NAME + '.prepare'
    arguments = dict(producer_output='/RAM/suffix', producer_report_sha256=h('report'),
        producer_prepared='/RAM/suffix_prepare.json', producer_prepare_sha256=h('prepare'),
        producer_source_sha256=h('source'), producer_receipt='/RAM/suffix_consume.json',
        producer_receipt_sha256=h('receipt'))
    compiled = dict(schema=m.suffix.NAME + '.prepare', contract=m.suffix.contract(),
        slots=[], expected_model_hash=h('model'), full_upstream_rebuilt_here=True,
        original_compute=m.suffix.source.contract()['total'], bindings={},
        arguments=dict(parent_output='/RAM/parent', parent_sha=h('parent report')))
    rows = []; finals = []; selected = []
    for i, old in enumerate(original['slots']):
        example = model_slot(i)
        old.update(packet=example['packet'], q=example['q'])
        task = m.suffix.task_from(old, compiled['expected_model_hash'])
        commands = deepcopy(prior['plans'][i]['commands'])
        candidates = [dict(index=j, prefix=deepcopy(commands[:5])) for j in range(42)]
        slot = dict(index=i, root_slot=old['root_slot'], condition=old['condition'],
            task=task, candidates=candidates)
        compiled['slots'].append(slot)
        # Explicit fixed full168 RAM witness, no real score or neural compute.
        group = []
        for j in range(42):
            result = dict(index=j, winner_iteration=2, commands=deepcopy(commands),
                selected_Q=float(abs(j-i)), selected_context=deepcopy(task['root_context']))
            group.append(dict(slot=i, prefix_index=j, status='sealed', result=result))
        rows.extend(group); winner = group[i]['result']
        plan = dict(prefix_index=i, iterate_index=2, commands=deepcopy(commands), Q=0.,
            context=deepcopy(task['root_context']), physical_release=False,
            selection_digest=m.suffix.digest([m.suffix.NAME, task['binding'], winner]))
        finals.append(dict(slot=i, status='sealed', plan=plan))
        selected.append(dict(plan, root_slot=old['root_slot'], condition=old['condition'],
            task_id=old['task_id'], planner_report_sha256=arguments['producer_report_sha256']))
    report = dict(schema=m.suffix.NAME+'.report',status='all168_nonlinear_suffix_sealed_not_admitted',
        errors=[],prepared=arguments['producer_prepared'],prepare_sha256=arguments['producer_prepare_sha256'],
        contract=m.suffix.contract(), rows=rows, finals=finals, fixed_denominator=168, slot_denominator=4,
        known_roots=2, source_verified=True,model_before=dict(original_state_dict=compiled['expected_model_hash']),
        model_after=dict(original_state_dict=compiled['expected_model_hash']),
        P1_admission=False,freshness=False,physical_release=False,scientific_admission=False,artifact_sha256={})
    consumed = dict(schema=m.suffix.NAME+'.consume',status='all168_verified_science_blocked',complete=True,
        report_sha256=arguments['producer_report_sha256'],fixed_denominator=168,slot_denominator=4,
        known_roots=2,rows=deepcopy(rows),plans=selected,
        actual_new_compute={k:m.suffix.caps(3376,1344) for k in ('attempted','completed')},
        inherited_compute=compiled['original_compute'],P1_admission=False,freshness=False,
        physical_release=False,scientific_admission=False)
    plans = m.plans_from_saved(report,compiled,consumed,original,data,arguments)
    authority = dict(producer=m.suffix.NAME,**arguments,complete_all168=True,full_source_build_verified=True,
        persisted_receipt_equal=True,new_algorithm=True,selection_uses_physics=False,P1_admission=False,
        actual_new_compute=deepcopy(consumed['actual_new_compute']),
        inherited_compute=deepcopy(consumed['inherited_compute']),plans_sha256=m.mixed.core.digest(plans))
    p = dict(schema=m.NAME+'.prepare',contract=m.contract(),arguments=arguments,plans=plans,
        planner_authority=authority,bindings={},source_report_sha256=m.mixed.SOURCE_SHA,
        source_slots=deepcopy(prior['source_slots']),source_roots=deepcopy(prior['source_roots']),
        warm_traces=deepcopy(prior['warm_traces']))
    return p,data,report,compiled,consumed,original


def test_real_new_authority_full168_and_four_finals():
    p,data,r,c,u,o=fixture();m.validate_execution(p)
    assert m.suffix.NAME=='agent_p1_nonlinear_suffix_library_20261028_v2'
    assert [x['selected_index'] for x in p['plans']]==[0,1,2,3]
    assert p['contract']['inheritance'] is False and len(u['rows'])==168
    assert all(x['schema']==m.NAME+'.execution_plan' for x in p['plans'])
    assert 'selector_authority' not in p


@pytest.mark.parametrize('kind',['prepare','report','consume'])
def test_v1_schema_not_relabelled_or_accepted(kind):
    p,data,r,c,u,o=fixture()
    target={'prepare':c,'report':r,'consume':u}[kind]
    target['schema']='agent_p1_nonlinear_suffix_library_20261028_v1.'+kind
    with pytest.raises(ValueError):m.plans_from_saved(r,c,u,o,data,p['arguments'])


@pytest.mark.parametrize('damage',['partial','missingrow','badrow','final_count','final_status','final_action',
    'consume_action','consume_context','consume_task','report_sha','prepare_sha','source_schema','goal','context','Q',
    'blind','cost','model','admission','winner','iterate','prefix','digest','report_contract'])
def test_full168_final_action_source_mutations_refused(damage):
    p,data,r,c,u,o=fixture()
    if damage=='partial':u['complete']=False
    elif damage=='missingrow':r['rows'].pop();u['rows']=deepcopy(r['rows'])
    elif damage=='badrow':r['rows'][5]['status']='failed';u['rows']=deepcopy(r['rows'])
    elif damage=='final_count':r['finals'].pop()
    elif damage=='final_status':r['finals'][0]['status']='failed'
    elif damage=='final_action':r['finals'][0]['plan']['commands'][0][0]=.125
    elif damage=='consume_action':u['plans'][0]['commands'][0][0]=.125
    elif damage=='consume_context':u['plans'][0]['context']['latent_history'][0][0]+=.1
    elif damage=='consume_task':u['plans'][0]['task_id']=h('wrong')
    elif damage=='report_sha':u['report_sha256']=h('wrong')
    elif damage=='prepare_sha':r['prepare_sha256']=h('wrong')
    elif damage=='source_schema':o['schema']=m.mixed.PRODUCER+'.prepare'
    elif damage=='goal':o['slots'][0]['packet']['goal'][0]+=.1
    elif damage=='context':o['slots'][0]['packet']['root_context']['latent_history'][0][0]+=.1
    elif damage=='Q':o['slots'][0]['q'][0][0]=2
    elif damage=='blind':o['slots'][0]['blind_sha256']=h('wrong')
    elif damage=='cost':u['actual_new_compute']['completed']['predict_API']-=1
    elif damage=='model':r['model_after']['original_state_dict']=h('wrong')
    elif damage=='admission':u['P1_admission']=True
    elif damage=='winner':r['rows'][0]['result']['selected_Q']=100;u['rows']=deepcopy(r['rows'])
    elif damage=='iterate':r['finals'][0]['plan']['iterate_index']=9
    elif damage=='prefix':c['slots'][0]['candidates'][0]['prefix'][0][0]=.125
    elif damage=='digest':r['finals'][0]['plan']['selection_digest']=h('wrong')
    elif damage=='report_contract':r['contract']['updates']=7
    with pytest.raises(ValueError):m.plans_from_saved(r,c,u,o,data,p['arguments'])


@pytest.mark.parametrize('damage',['raw','digest','authority','native_goal','cost','schema','order'])
def test_execution_contract_mutations_refused(damage):
    p,*_=fixture()
    if damage=='raw':p['plans'][0]['commands'][0][0]=.125
    elif damage=='digest':p['plans'][0]['selection_digest']=h('wrong')
    elif damage=='authority':p['planner_authority']['complete_all168']=False
    elif damage=='native_goal':
        p['plans'][0]['native_goal'][0]+=.1
        p['plans'][0]['selection_digest']=m.mixed.core.digest({k:v for k,v in p['plans'][0].items() if k!='selection_digest'})
        p['planner_authority']['plans_sha256']=m.mixed.core.digest(p['plans'])
    elif damage=='cost':p['planner_authority']['actual_new_compute']['attempted']['predict_API']-=1
    elif damage=='schema':p['schema']=m.terminal.NAME+'.prepare'
    elif damage=='order':p['plans'].reverse()
    with pytest.raises(ValueError):m.validate_execution(p)


def execute(*, fault=None, at=11, close_error=False):
    p, data, *_ = fixture(); store = MemoryStore(); envs = []
    def factory():
        env = Env(fault=fault, at=at, close_error=close_error); envs.append(env); return env
    with pytest.MonkeyPatch.context() as mp:
        # Explicit RAM-only placement and validation seams; production source
        # NEVER patches imported globals or rewrites an old schema/receipt.
        mp.setattr(m.mixed, 'place', fake_place); mp.setattr(m.mixed, 'validate_env', lambda _: None)
        mp.setattr(m, 'sha', lambda f: hashlib.sha256(store.files[Path(f).name]).hexdigest())
        report = m.execute(p, data, store, factory, authorization=m.AUTH, prepared='/RAM_ONLY/native_prepare.json',
                           prepare_sha256=h('native prepare'))
    return report, store, p, data, envs


@lru_cache(maxsize=1)
def complete(): return execute()


def test_actual_lowlevel_episode_full4_all10_own_selector_arm():
    r, s, p, data, envs = complete()
    assert r['status'] == m.RECORDED, r['errors']
    assert len(envs) == len(m.validate_saved(r, s.events, p)) == 4
    assert r['counts'] == {k: dict(attempted=v, completed=v) for k, v in m.contract()['caps'].items()}
    assert r['counts']['public_step']['completed'] == 80 and r['counts']['physics_step']['completed'] == 816
    assert len([x for x in s.files if x.endswith('.png')]) == 264
    assert all(x['own_returned'] == 10 and x['warmup_returned'] == 10 and
               x['prefix51']['complete'] and x['missing_checkpoints'] == [] for x in r['rows'])
    for env, plan in zip(envs, p['plans'], strict=True):
        assert env.closed and env.dispatch_flags == [False] * 20 and env.with_target is False
        assert np.asarray(env.commands[10:], np.float32).tobytes() == m.mixed.raw10(plan['commands']).tobytes()
        assert '_set_state' not in env.__dict__ and 'reset' not in env.__dict__


@pytest.mark.parametrize('fault', ['terminal', 'truncate', 'throw', 'substitute', 'guard', 'render'])
def test_native_or_engineering_stop_preserves4_and_does_not_pad(fault):
    r, s, p, _, envs = execute(fault=fault, at=15)
    assert len(r['rows']) == 4 and all(x.closed for x in envs)
    assert all(row['own_returned'] <= 10 for row in r['rows'])
    if fault in ('terminal', 'truncate'):
        assert r['status'] == m.RECORDED, r['errors']
        assert all(len(x.commands) == 15 for x in envs)
        assert all(row['own_returned'] == 5 and row['missing_checkpoints'] == [10] for row in r['rows'])
        m.validate_saved(r, s.events, p)
    else:
        assert r['status'] == 'partial_or_failed'


@pytest.mark.parametrize('damage', ['raw', 'scope', 'task', 'state', 'snapshot', 'reward', 'goal', 'triplet',
    'RGB', 'trajectory', 'unknown_event', 'counts', 'seal', 'first5only', 'old_schema', 'admission'])
def test_saved_resealed_mutations_refused(damage):
    original, store, prepared, _, _ = complete()
    r, p, es = deepcopy(original), deepcopy(prepared), deepcopy(store.events)
    def event(kind): return next(e for e in es if e['kind'] == kind)
    if damage == 'raw': next(e for e in es if e['kind'] == 'operation_dispatch_intent' and
        e['receipt']['resource'] == 'public_step')['receipt']['raw_bits'] = '00'
    elif damage == 'scope': event('fresh_observation')['condition'] = 'on'
    elif damage == 'task': r['rows'][0]['task_id'] = h('wrong')
    elif damage == 'state': event('fresh_observation')['receipt']['state'][0] += 1
    elif damage == 'snapshot': event('fresh_observation')['receipt']['snapshot']['state'][0] += 1
    elif damage == 'reward': r['rows'][0]['responses'][0]['reward'] = 0
    elif damage == 'goal': r['rows'][0]['goal_pose'][0] += 1
    elif damage == 'triplet': event('fresh_observation')['receipt']['overlay_index'] = 2
    elif damage == 'RGB': event('fresh_observation')['receipt']['pixel_sha256'] = h('bad')
    elif damage == 'trajectory': r['rows'][0]['trajectory'].pop()
    elif damage == 'unknown_event': event('fresh_observation')['kind'] = 'unknown'
    elif damage == 'counts': r['counts']['physics_step']['completed'] -= 1
    elif damage == 'seal': r['execution_binding']['planner_authority']['producer_report_sha256'] = h('bad')
    elif damage == 'first5only': r['rows'][0]['own_returned'] = 5
    elif damage == 'old_schema': r['schema'] = m.mixed.NAME + '.report'
    elif damage == 'admission': r['P1_admission'] = True
    reseal(es, r)
    with pytest.raises((ValueError, KeyError, StopIteration)): m.validate_saved(r, es, p)


def test_saved_complete_consume_png_bytes_and_extra_artifact(monkeypatch):
    r, store, p, data, _ = complete(); files = deepcopy(store.files)
    expected = hashlib.sha256(files['report.json']).hexdigest()
    monkeypatch.setattr(m, 'verify', lambda *a: (deepcopy(p), deepcopy(data)))
    monkeypatch.setattr(m.mixed, 'saved_artifacts', lambda *a: deepcopy(files))
    out = m.consume('/RAM_ONLY', expected)
    assert out['fixed_denominator'] == 4 and out['saved_PNG_verified'] == 264
    assert out['P1_admission'] is out['freshness'] is False
    files['unexpected.txt'] = b'unknown'
    with pytest.raises(ValueError, match='unaccounted'): m.consume('/RAM_ONLY', expected)
    files.pop('unexpected.txt')
    name = next(x for x in files if x.endswith('.png')); files[name] += b'bad'
    with pytest.raises(ValueError): m.consume('/RAM_ONLY', expected)


def test_MAIN_authority_rejected_before_reads_or_imports(monkeypatch):
    monkeypatch.setattr(m, 'verify', lambda *a: pytest.fail('no unauthorized reads'))
    with pytest.raises(ValueError): m.run(None, None, None, lambda: pytest.fail('no env'), authorization=None)
    with pytest.raises(ValueError): m.main(['run', '--authorization', 'wrong'])


def test_cold_import_no_foreign_runtime():
    script = '''
import builtins
old=builtins.__import__
def guard(name,*a,**k):
    if name.split('.')[0] in ('torch','gymnasium','pymunk','stable_worldmodel'):raise AssertionError(name)
    return old(name,*a,**k)
builtins.__import__=guard
from experiments.spectral_reachability import agent_nonlinear_suffix_physical_20261028_v1 as m
assert m.contract()['inheritance'] is False
try:m.main(['--help'])
except SystemExit as e:assert e.code==0
'''
    result = subprocess.run([sys.executable, '-B', '-c', script], cwd=m.ROOT, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_native_joint_success_not_native7_reward_threshold():
    value = m.mixed.metrics([0.] * 7, [1., 1., 1., 1., .01, 10000., -10000.])
    assert value['native_success'] is True and value['native7_reward'] < -10000


def test_material_and_verify_call_real_consume_entrypoint_and_exact_saved_receipt(monkeypatch):
    p,data,r,c,u,o=fixture();args=p['arguments'];data['bindings']={}
    parent=dict(prepared='/RAM/original.json',prepare_sha256=h('original prepare'))
    c['bindings']={'/RAM/parent/report.json':h('parent report'),'/RAM/original.json':h('original prepare')}
    records={args['producer_prepared']:c,args['producer_output']+'/report.json':r,
        args['producer_receipt']:u,'/RAM/parent/report.json':parent,'/RAM/original.json':o}
    hashes={str(Path(m.suffix.__file__).resolve()):args['producer_source_sha256'],
        args['producer_prepared']:args['producer_prepare_sha256'],
        args['producer_output']+'/report.json':args['producer_report_sha256'],
        args['producer_receipt']:args['producer_receipt_sha256'],**c['bindings']}
    hashes.update({s['task_file']:s['task_sha256'] for s in o['slots']})
    calls=[]
    monkeypatch.setattr(m,'read',lambda path,*a:deepcopy(records[str(path)]))
    monkeypatch.setattr(m,'sha',lambda path:hashes.get(str(path),h(str(path))))
    monkeypatch.setattr(m.suffix,'consume',lambda output,expected:
        calls.append((str(output),expected)) or deepcopy(u))
    monkeypatch.setattr(m.mixed,'source_material',lambda:deepcopy(data))
    monkeypatch.setattr(m.mixed.allmode,'constructor_contract',lambda:('/RAM/native.py',h('/RAM/native.py')))
    built,_=m.material(args)
    assert calls==[(args['producer_output'],args['producer_report_sha256'])]
    assert built['planner_authority']['full_source_build_verified'] is True
    records['/RAM/native_prepare.json']=built
    verified,_=m.verify('/RAM/native_prepare.json',m.sha('/RAM/native_prepare.json'))
    assert verified==built and len(calls)==2
    records[args['producer_receipt']]=dict(u,complete=False)
    with pytest.raises(ValueError,match='persisted real receipt'):m.material(args)
    assert len(calls)==3


def test_external_source_sha_rejected_before_consume(monkeypatch):
    p,*_=fixture()
    monkeypatch.setattr(m,'sha',lambda path:h('different'))
    monkeypatch.setattr(m.suffix,'consume',lambda *a:pytest.fail('no consume after bad source'))
    with pytest.raises(ValueError,match='external producer SHA'):m.material(p['arguments'])


def test_prepare_existing_refused_before_expensive_full_build(tmp_path,monkeypatch):
    file=tmp_path/'exists.json';file.write_text('{}')
    monkeypatch.setattr(m,'material',lambda *a:pytest.fail('no full build'))
    with pytest.raises(ValueError,match='exclusive prepare'):m.prepare(str(file),{})


def test_cli_requires_actual_seven_producer_arguments(monkeypatch):
    p,*_=fixture();called=[]
    monkeypatch.setattr(m,'prepare',lambda out,args:called.append((out,args)) or dict(status='RAM'))
    cli=['prepare','--output','/RAM/new.json']
    for key,value in p['arguments'].items():
        cli+=['--'+key.replace('_sha256','_sha').replace('_','-'),value]
    assert m.main(cli)==0
    assert called==[('/RAM/new.json',p['arguments'])]


def test_no_outcome_input_changes_model_plan_choice():
    p,data,r,c,u,o=fixture()
    r['native_outcomes']=[1e99,-1e99,None,0]
    for row in r['rows']:row['physical_reward']=-1e99
    u['rows']=deepcopy(r['rows'])
    assert m.plans_from_saved(r,c,u,o,data,p['arguments'])==p['plans']

