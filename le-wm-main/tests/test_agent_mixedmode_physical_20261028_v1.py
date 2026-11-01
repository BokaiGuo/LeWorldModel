"""RAM mocks and tiny metadata only. No neural/native physics or real consume."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import agent_mixedmode_physical_20261028_v1 as m
from test_agent_allmode_physical_20261028_v1 import fixture as old_fixture, MemoryStore, Env, fake_place, h
from test_agent_allmode_physical_20261028_v1 import completed as old_completed, reseal


def fixture():
    old,data,report,compiled,consumed=old_fixture();p=deepcopy(old)
    p.update(schema=m.NAME+'.prepare',contract=m.contract())
    p['old_physical_consume']=dict(schema=m.OLD+'.consume',rows=[])
    p['old_report_sha256']=h('old report')
    p['arguments']=dict(producer_output='/RAM/mixed',producer_report_sha256=h('mixed report'),
        producer_source_sha256=m.PRODUCER_SHA,producer_prepared='/RAM/mixed_prepare.json',producer_prepare_sha256=h('mixed prepare'),
        old_output='/RAM/allmode',old_report_sha256=h('old report'),old_source_sha256=m.OLD_SHA)
    p['producer_report_sha256']=h('mixed report');p['producer_prepare_sha256']=h('mixed prepare')
    for i,plan in enumerate(p['plans']):
        plan.update(schema=m.PRODUCER+'.model_plan',arm=m.ARM,whole_run_report_sha256=h('mixed report'))
        plan['selection_digest']=m.core.digest({k:v for k,v in plan.items() if k not in ('selection_digest','whole_run_report_sha256')})
        report['rows'][i]['plan']={k:v for k,v in plan.items() if k!='whole_run_report_sha256'}
    consumed.update(schema=m.PRODUCER+'.consume',status='all4_mixedmode_commands_verified_not_admitted',plans=deepcopy(p['plans']))
    p['producer_consume']=consumed
    report.update(schema=m.PRODUCER+'.report',status='all4_mixedmode_plans_sealed_not_admitted',
        resources={k:dict(attempted=5352,completed=5352) for k in ('predict','action_encoder','pred_proj')})
    compiled['schema']=m.PRODUCER+'.prepare'
    for slot in compiled['slots']:slot.update(packet=dict(context=h(slot['index'])),q=[[1.]])
    oldcompiled=deepcopy(compiled);oldcompiled['schema']=m.allmode.PRODUCER+'.prepare'
    # Original kernel execution fixtures: all rows default require actual new work.
    p['routes']=[dict(index=i,mode='execute',reason='raw10_changed',old_index=i) for i in range(4)]
    p['bindings']={}
    return p,data,report,compiled,old,oldcompiled


def test_mixed222_ABI_complete_and_own_selection_seals():
    p,data,report,compiled,*_=fixture()
    assert m.validate_plans(report,compiled,p['producer_consume'],data,p['producer_report_sha256'])==p['plans']


@pytest.mark.parametrize('fault',['partial','resources','task','raw','digest','schema','index','denominator','zero_sha'])
def test_mixed222_ABI_rejects_corruption(fault):
    p,data,r,c,*_=fixture();v=p['producer_consume']
    if fault=='partial':r['status']='partial_or_failed'
    elif fault=='resources':r['resources']['predict']['completed']-=1
    elif fault=='task':v['plans'][0]['task_sha256']=h('wrong task')
    elif fault=='raw':v['plans'][0]['commands'][0][0]=.5
    elif fault=='digest':v['plans'][0]['selection_digest']=h('bad')
    elif fault=='schema':r['schema']=m.allmode.PRODUCER+'.report'
    elif fault=='index':v['plans'][0]['selected_index']=222
    elif fault=='denominator':v['plans'].pop()
    else:p['producer_report_sha256']='0'*64
    with pytest.raises((ValueError,KeyError)):m.validate_plans(r,c,v,data,p['producer_report_sha256'])


def test_inheritance_raw_context_source_and_denominator_not_outcomes():
    p,data,_,compiled,old,oldcompiled=fixture()
    oldreceipt=dict(rows=[dict(status='horizon_completed',native_success=False) for _ in range(4)])
    routes=m.routes(p['plans'],compiled,old,oldcompiled,oldreceipt,p)
    assert len(routes)==4 and all(x['mode']=='inherit' for x in routes)
    p['plans'][1]['commands'][0][0]=.5
    oldcompiled['slots'][2]['packet']['context']='changed'
    routes=m.routes(p['plans'],compiled,old,oldcompiled,oldreceipt,p)
    assert [r['mode'] for r in routes]==['inherit','execute','execute','inherit']


@pytest.mark.parametrize('fault',['source','blind','goal','warmup','slot','oldfailure'])
def test_inheritance_mismatch_requires_new_whole_branch(fault):
    p,data,_,compiled,old,oldcompiled=fixture()
    receipt=dict(rows=[dict(status='horizon_completed') for _ in range(4)])
    if fault=='source':old['source_report_sha256']=h('other source')
    elif fault=='blind':old['plans'][0]['blind_sha256']=h('other blind')
    elif fault=='goal':old['source_roots'][0]['goal'][0]+=1
    elif fault=='warmup':old['warm_traces'][0][0]['receipt']['state'][0]+=1
    elif fault=='slot':old['source_slots'][0]['reset_seed']+=1
    else:receipt['rows'][0]['status']='engineering_failed'
    result=m.routes(p['plans'],compiled,old,oldcompiled,receipt,p)
    assert len(result)==4 and result[0]['mode']=='execute'


def test_run_authority_before_reads_or_factory(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('unauthorized read'))
    with pytest.raises(ValueError):m.run('/missing',h('x'),'/missing',lambda:pytest.fail('factory'),authorization=None)


def test_native_success_is_not_reward7_threshold():
    goal=np.zeros(7);state=np.array([1.,1.,1.,1.,.01,10000.,-10000.])
    metric=m.metrics(goal.tolist(),state.tolist())
    assert metric['native_success'] is True and metric['joint_xy_error']==2
    assert metric['native7_reward'] < -10000 and metric['native7_distance']>10000
    state[0]=20
    assert m.metrics(goal.tolist(),state.tolist())['native_success'] is False


def test_cli_prepare_external_argument_names(monkeypatch):
    seen=[]
    monkeypatch.setattr(m,'prepare',lambda path,args:seen.append((path,args)) or dict(status='RAM_only'))
    assert m.main(['prepare','--output','/RAM/prepare','--producer-report-sha',h('report'),
        '--producer-source-sha',m.PRODUCER_SHA,'--producer-prepare-sha',h('prepare'),'--old-source-sha',m.OLD_SHA,
        '--old-report-sha',m.OLD_REPORT_SHA])==0
    assert set(seen[0][1])==m.ARGUMENTS and seen[0][1]['producer_report_sha256']==h('report')


def execute(monkeypatch,*,modes=('execute',)*4,factory=Env,started=None):
    p,data,*_=fixture();s=MemoryStore();envs=[]
    p['routes']=[dict(index=i,old_index=i,mode=mode,reason='RAM') for i,mode in enumerate(modes)]
    if 'inherit' in modes:p['old_physical_consume']['rows']=deepcopy(old_completed()[0]['rows'])
    def make():env=factory();envs.append(env);return env
    monkeypatch.setattr(m,'place',fake_place);monkeypatch.setattr(m,'validate_env',lambda _:None)
    monkeypatch.setattr(m,'sha',lambda f:hashlib.sha256(s.files[Path(f).name]).hexdigest())
    r=m.execute(p,data,s,make,authorization=m.AUTH,prepared='/RAM/mixed_prepare.json',prepare_sha256=h('own prepare'),started=started)
    return r,s,p,data,envs


@lru_cache(maxsize=1)
def completed():
    with pytest.MonkeyPatch.context() as mp:return execute(mp,modes=('execute','execute','inherit','inherit'))


def test_root0_new_root1_inherited_all4_native_receipts_no_splicing():
    r,s,p,data,envs=completed()
    assert r['status']==m.RECORDED and len(r['rows'])==4 and len(envs)==2 and r['inherited']==2
    assert r['counts']['public_step']==dict(attempted=40,completed=40)
    assert r['counts']['physics_step']==dict(attempted=408,completed=408)
    assert [x['own_returned'] for x in r['rows']]==[10,10,0,0]
    assert all(x['prefix51']['complete'] for x in r['rows'][:2])
    assert all(x['inherited_evidence']==p['old_physical_consume']['rows'][i] for i,x in enumerate(r['rows']) if i>=2)
    assert m.validate_saved(r,s.events,p)==r['rows']


@pytest.mark.parametrize('fault',['denominator','rowkey','raw','goal','renderer','pixels','metric','reward','earlystop','inherit_splice','inherit_replace','route','scope','counts','admission','chain'])
def test_resealed_saved_boundary_tamper_failclosed(fault):
    r0,s,p0,*_=completed();r,p,events=deepcopy(r0),deepcopy(p0),deepcopy(s.events)
    if fault=='denominator':r['rows'].pop()
    elif fault=='rowkey':r['rows'][0]['condition']='on'
    elif fault=='raw':next(e for e in events if e['kind']=='operation_dispatch_intent' and e['receipt']['resource']=='public_step')['receipt']['raw_bits']='00'
    elif fault=='goal':r['rows'][0]['goal_pose'][0]+=1
    elif fault=='renderer':next(e for e in events if e['kind']=='fresh_observation')['pre']['renderer']['action_scale']=1
    elif fault=='pixels':next(e for e in events if e['kind']=='fresh_observation')['receipt']['pixel_sha256']=h('bad')
    elif fault=='metric':r['rows'][0]['responses'][0]['native_metrics']['native_distance']=0.
    elif fault=='reward':r['rows'][0]['responses'][0]['reward']=0.
    elif fault=='earlystop':r['rows'][0]['responses'][0]['truncated']=True
    elif fault=='inherit_splice':r['rows'][2]['checkpoints']=[dict(step=10,state=[0]*7)]
    elif fault=='inherit_replace':r['rows'][2]['inherited_evidence']['final_state'][0]+=1
    elif fault=='route':r['rows'][0]['route']='inherit'
    elif fault=='scope':
        for e in events:
            if e['kind'].startswith('operation_') and e['receipt']['resource']=='constructor' and e['receipt']['branch']==0:e['receipt']['condition']='on'
        r['actual_operation_receipts'][0]['condition']='on'
    elif fault=='counts':r['counts']['physics_step']['completed']-=1
    elif fault=='admission':r['freshness']=True
    else:events[2]['event_sha256']=h('bad')
    if fault!='chain':reseal(events,r)
    with pytest.raises((ValueError,KeyError,StopIteration)):m.validate_saved(r,events,p)


def test_saved_consume_complete_PNG_byte_boundary_and_unknown(monkeypatch):
    r,s,p,data,*_=completed();files=deepcopy(s.files);expected=hashlib.sha256(files['report.json']).hexdigest()
    monkeypatch.setattr(m,'saved_artifacts',lambda _:deepcopy(files))
    monkeypatch.setattr(m,'verify',lambda *a:(deepcopy(p),deepcopy(data)))
    receipt=m.consume('/RAM_ONLY',expected)
    assert len(receipt['rows'])==len(receipt['metric_breakdown'])==4 and receipt['inherited']==2
    assert receipt['new_neural_calls']==receipt['new_physics_calls']==0 and receipt['P1_admission'] is receipt['freshness'] is False
    name=next(n for n in files if n.endswith('.png'));files[name]+=b'tampered'
    with pytest.raises(ValueError):m.consume('/RAM_ONLY',expected)


def test_all_inherited_no_env_or_new_operations(monkeypatch):
    r,s,p,_,envs=execute(monkeypatch,modes=('inherit',)*4,factory=lambda:pytest.fail('no new env'))
    assert not envs and r['inherited']==4 and r['status']==m.RECORDED
    assert all(v==dict(attempted=0,completed=0) for v in r['counts'].values())
    assert m.validate_saved(r,s.events,p)==r['rows']


def test_constructor_failure_keeps_unknown_all_four(monkeypatch):
    def failed():raise RuntimeError('RAM constructor failure')
    r,s,p,_,_=execute(monkeypatch,factory=failed)
    assert r['status']=='partial_or_failed' and len(r['rows'])==4
    assert all(x['final_state'] is x['native_success'] is None for x in r['rows'])
    assert r['total_internal_physics_scope']=='unknown_constructor_work'
    assert m.validate_saved(r,s.events,p)==r['rows']


@pytest.mark.parametrize('fault,at',[('truncate',5),('terminal',11),('truncate',18)])
def test_new_native_earlystop_never_pads_or_borrows_old_suffix(monkeypatch,fault,at):
    r,s,p,_,envs=execute(monkeypatch,modes=('execute','execute','inherit','inherit'),factory=lambda:Env(fault=fault,at=at))
    assert r['status']==m.RECORDED and len(r['rows'])==4
    assert all(e.closed and len(e.commands)==at for e in envs)
    assert all(x['own_returned']==max(0,at-10) and 10 in x['missing_checkpoints'] and
        x['inherited_evidence'] is None for x in r['rows'][:2])
    assert m.validate_saved(r,s.events,p)==r['rows']


@pytest.mark.parametrize('fault',['zero','order','scale','dtype','shape'])
def test_fixed_nonzero_warmup_rejected_before_any_public_call(fault,monkeypatch):
    p,data,*_=fixture();root=data['roots'][0];packet=root['packets']['off'];a=packet['raw_action_history']
    if fault=='zero':a=np.zeros_like(a)
    elif fault=='order':a=a[::-1].copy()
    elif fault=='scale':a=a*100
    elif fault=='dtype':a=a.astype(np.float64)
    else:a=a.reshape(5,4)
    packet['raw_action_history']=a
    monkeypatch.setattr(m,'place',lambda *a:pytest.fail('bad history before placement'))
    with pytest.raises(ValueError):m.episode(SimpleNamespace(unwrapped=object()),root,data['source_plan'],p['plans'][0],{},None,None)


def material_boundary(monkeypatch,tmp_path):
    p,data,r,c,old,oldc=fixture();calls=[];out=tmp_path/'mixed';out.mkdir()
    pp='/RAM/mixed_prepare.json';oldpp='/RAM/allmode_prepare.json';oldcompiler=old['arguments']['producer_prepared']
    rp=str(out/'report.json');oldout=Path('/RAM/allmode')
    code=str(m.ROOT/'experiments/spectral_reachability'/(m.PRODUCER+'.py'))
    args=dict(p['arguments'],producer_output=str(out),producer_report_sha256=h(rp),producer_prepare_sha256=h(pp),
        old_report_sha256=m.OLD_REPORT_SHA)
    oldreceipt=dict(schema=m.OLD+'.consume',status=m.allmode.RECORDED,errors=[],fixed_denominator=4,
        inputs_final_verified=True,physical_report_sha256=m.OLD_REPORT_SHA,rows=deepcopy(old_completed()[0]['rows']),
        bindings={oldpp:h(oldpp),oldcompiler:old['arguments']['producer_prepare_sha256']})
    c.update(contract={'RAM':'mixed'},bindings={})
    r.update(contract=c['contract'],prepared=pp,prepare_sha256=h(pp))
    verified=deepcopy(p['producer_consume'])
    for plan in verified['plans']:plan['whole_run_report_sha256']=h(rp)
    oldreport=dict(rows=deepcopy(oldreceipt['rows']),execution_binding=dict(prepared=oldpp,prepare_sha256=h(oldpp)))
    docs={rp:r,pp:c,str(oldout/'report.json'):oldreport,oldpp:old,oldcompiler:oldc}
    hashes={code:m.PRODUCER_SHA,str(Path(m.allmode.__file__)):m.OLD_SHA,rp:h(rp),pp:h(pp),
        str(oldout/'report.json'):m.OLD_REPORT_SHA,oldpp:h(oldpp),oldcompiler:old['arguments']['producer_prepare_sha256'],
        **{slot['task_file']:slot['task_sha256'] for slot in c['slots']}}
    def newconsume(*a):calls.append('mixed.consume');return deepcopy(verified)
    def oldconsume(*a):calls.append('allmode.consume');return deepcopy(oldreceipt)
    module=SimpleNamespace(__file__=code,contract=lambda:c['contract'],consume=newconsume)
    monkeypatch.setattr(m,'REPORT_SHA',h(rp));monkeypatch.setattr(m,'PREP_SHA',h(pp));monkeypatch.setattr(m,'PREPARED',Path(pp))
    monkeypatch.setattr(m,'OLD_OUTPUT',oldout);monkeypatch.setattr(m,'consume_old',oldconsume)
    monkeypatch.setattr(m,'source_material',lambda:deepcopy(data));monkeypatch.setattr(m,'own_files',lambda:[Path('/RAM/own.py')])
    monkeypatch.setattr(m,'read',lambda path,cap=m.PREP_CAP:deepcopy(docs[str(path)]))
    monkeypatch.setattr(m,'sha',lambda path:hashes.get(str(path),h(str(path))))
    monkeypatch.setattr(m.importlib,'import_module',lambda name:module)
    return args,calls,module,hashes,docs,oldreceipt


def test_material_complete_both_real_consume_ABIs_and_actual_external_seals(monkeypatch,tmp_path):
    args,calls,module,hashes,docs,old=material_boundary(monkeypatch,tmp_path)
    p,_=m.material(args)
    assert calls==['mixed.consume','allmode.consume'] and p['old_physical_consume']==old
    assert len(p['plans'])==len(p['routes'])==4 and all(r['mode']=='inherit' for r in p['routes'])
    assert p['arguments']==args and '/RAM/own.py' in p['bindings']


@pytest.mark.parametrize('fault',['source','report','prepare','partial','new_consume','old_consume','conflicting_binding'])
def test_material_failclosed_no_partial_release_or_bypass(monkeypatch,tmp_path,fault):
    args,calls,module,hashes,docs,old=material_boundary(monkeypatch,tmp_path)
    if fault=='source':hashes[str(module.__file__)]=h('bad')
    elif fault=='report':hashes[str(Path(args['producer_output'])/'report.json')]=h('bad')
    elif fault=='prepare':hashes[args['producer_prepared']]=h('bad')
    elif fault=='partial':docs[str(Path(args['producer_output'])/'report.json')]['status']='partial_or_failed'
    elif fault=='new_consume':module.consume=lambda *a:(_ for _ in ()).throw(ValueError('actual mixed consume failure'))
    elif fault=='old_consume':monkeypatch.setattr(m,'consume_old',lambda *a:(_ for _ in ()).throw(ValueError('actual old consume failure')))
    else:
        docs[args['producer_prepared']]['bindings']['/RAM/conflict']=h('new')
        old['bindings']['/RAM/conflict']=h('old')
    with pytest.raises(ValueError):m.material(args)
    if fault in ('source','report','prepare','partial','new_consume'):assert calls==[]


def test_new_store_prefix_exclusive_floor_and_bound(monkeypatch,tmp_path):
    monkeypatch.setattr(m.shutil,'disk_usage',lambda _:SimpleNamespace(free=m.FLOOR+m.CAP))
    path=tmp_path/(m.NAME+'_RAM');store=m.Store(str(path));store.put('tiny.json',dict(RAM=True))
    with pytest.raises(ValueError):m.Store(str(path))
    with pytest.raises(ValueError):store.check(m.CAP)
    with pytest.raises(ValueError):m.Store(str(tmp_path/(m.OLD+'_forbidden')))


def test_prepare_verify_reconstructs_and_rejects_reentry_tamper(monkeypatch,tmp_path):
    p,data,*_=fixture();calls=[]
    monkeypatch.setattr(m,'material',lambda a:calls.append(a) or (deepcopy(p),deepcopy(data)))
    monkeypatch.setattr(m.shutil,'disk_usage',lambda _:SimpleNamespace(free=m.FLOOR+m.CAP))
    path=tmp_path/(m.NAME+'_prepare_RAM.json');receipt=m.prepare(str(path),p['arguments'])
    assert m.verify(str(path),receipt['sha256'])[0]==p
    before=len(calls)
    with pytest.raises(ValueError):m.prepare(str(path),p['arguments'])
    assert len(calls)==before
    changed=deepcopy(p);changed['routes'][0]['mode']='inherit';path.write_bytes(m.raw(changed))
    with pytest.raises(ValueError):m.verify(str(path),m.sha(path))


def test_cli_run_refuses_without_MAIN_before_any_native_import():
    with pytest.raises(ValueError):m.main(['run','--prepared','/missing','--output','/missing'])
