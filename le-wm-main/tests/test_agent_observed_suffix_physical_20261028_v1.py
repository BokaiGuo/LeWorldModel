"""RAM-only native ABI fixtures; never instantiate real physics or neural models."""
from copy import deepcopy
from functools import lru_cache
import hashlib
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import agent_observed_suffix_physical_20261028_v1 as m
import test_agent_nonlinear_suffix_physical_20261028_v1 as old


def fixture():
    bp,data,pr,pc,pu,_=old.fixture();br=deepcopy(old.complete()[0])
    args=dict(compile_output='/RAM/observed',compile_report_sha=old.h('observed report'),
        compile_receipt='/RAM/observed_receipt.json',compile_receipt_sha=old.h('observed receipt'),compile_source_sha=old.h('observed source'))
    c=dict(schema=m.compiler.NAME+'.prepare',contract=m.compiler.contract(),production_ready=True,rows=[],
        expected_model_hash=pc['expected_model_hash'],bindings={},midpoint_authority={},
        original_physical_authority=deepcopy(bp['planner_authority']),inherited_resources={})
    rows=[];plans=[]
    for i,s in enumerate(pc['slots']):
        parent=pr['finals'][i]['plan'];task=deepcopy(s['task']);commands=deepcopy(parent['commands'])
        commands[5:]=np.full((5,2),np.float32(.125)).tolist()
        spec=dict(index=i,root_slot=m.ORDER[i][0],condition=m.ORDER[i][1],status='ready',
            parent_plan_sha256=m.compiler.digest(parent),parent_selection_digest=parent['selection_digest'],
            midpoint_row_binding_sha256=old.h(('mid',i)),request=dict(task=task,observed_context=deepcopy(task['root_context']),
                incumbent=dict(prefix=deepcopy(parent['commands'][:5]),prefix_bits=m.compiler.F.bits(parent['commands'][:5]),
                    parent_prefix_index=parent['prefix_index'],parent_iteration=parent['iterate_index'])))
        c['rows'].append(spec)
        plan=dict(schema=m.compiler.NAME+'.plan',root_slot=spec['root_slot'],condition=spec['condition'],slot=i,
            task_id=task['binding']['task_id'],model_task_binding=deepcopy(task['binding']),commands=commands,
            raw_float32_bits=m.compiler.F.bits(commands),prefix_bits=spec['request']['incumbent']['prefix_bits'],
            parent_plan_sha256=spec['parent_plan_sha256'],parent_selection_digest=spec['parent_selection_digest'],
            observed_context_sha256=m.compiler.digest(spec['request']['observed_context']),goal_sha256=m.compiler.digest(task['goal']),
            new_suffix_Q=.25,observed_incumbent_Q=.5,new_terminal_context=deepcopy(task['root_context']),
            winner_iteration=2,midpoint_row_binding_sha256=spec['midpoint_row_binding_sha256'],
            physical_release=False,P1_admission=False,freshness=False)
        plan['selection_digest']=m.compiler.digest(plan);plans.append(plan)
        rows.append(dict(index=i,status='sealed',result=dict(plan=deepcopy(plan)),resources={}))
    flags=dict(P1_admission=False,scientific_admission=False,freshness=False,physical_release=False)
    r=dict(schema=m.compiler.NAME+'.report',status='observed_suffix4_recompiled_not_admitted',contract=m.compiler.contract(),
        prepared='/RAM/observed_prepare.json',prepare_sha256=old.h('observed prepare'),rows=rows,errors=[],
        fixed_denominator=4,known_roots=2,source_verified=True,model_unchanged=True,
        model_before=dict(original_state_dict=c['expected_model_hash']),model_after=dict(original_state_dict=c['expected_model_hash']),**flags)
    u=dict(schema=m.compiler.NAME+'.consume',status='observed_suffix4_verified_science_blocked',complete=True,
        report_sha256=args['compile_report_sha'],fixed_denominator=4,known_roots=2,plans=plans,rows=deepcopy(rows),
        actual_new_compute={k:m.compiler.F.caps(72,32) for k in ('attempted','completed')},
        original_physical_authority=deepcopy(c['original_physical_authority']),**flags)
    p=m.assemble(args,c,r,u,bp,br,data)
    return p,data,c,r,u,bp,br,args


def test_actual_schema_join_four_newplans_and_complete_task():
    p,data,*_=fixture();m.validate_execution(p)
    assert len(p['plans'])==4 and p['contract']['inheritance'] is False
    assert all(q['schema']==m.NAME+'.execution_plan' and len(q['model_task']['goal'])==192 for q in p['plans'])
    assert m.META_CAP==128*1024**2 and m.contract()['caps']['explicit_render']==264


@pytest.mark.parametrize('bad',['schema','partial','plan_count','order','prefix','goal','Q','normalizer','parent','plan_digest','cost','physical_goal','cp5','reference_prefix','admission'])
def test_compile_and_reference_mutations_fail_closed(bad):
    _,data,c,r,u,bp,br,args=fixture()
    if bad=='schema':r['schema']=m.base.suffix.NAME+'.report'
    elif bad=='partial':u['complete']=False
    elif bad=='plan_count':u['plans'].pop()
    elif bad=='order':c['rows'].reverse()
    elif bad=='prefix':u['plans'][0]['commands'][0][0]=.125
    elif bad=='goal':c['rows'][0]['request']['task']['goal'][0]+=1
    elif bad=='Q':c['rows'][0]['request']['task']['projection'][0][0]=2
    elif bad=='normalizer':c['rows'][0]['request']['task']['action_std'][0]=1
    elif bad=='parent':c['rows'][0]['parent_selection_digest']=old.h('bad')
    elif bad=='plan_digest':u['plans'][0]['selection_digest']=old.h('bad')
    elif bad=='cost':u['actual_new_compute']['completed']['predict_API']-=1
    elif bad=='physical_goal':br['rows'][0]['goal'][0]+=1
    elif bad=='cp5':br['rows'][0]['checkpoints']=[x for x in br['rows'][0]['checkpoints'] if x['step']!=5]
    elif bad=='reference_prefix':bp['plans'][0]['commands'][0][0]=.125
    else:u['P1_admission']=True
    with pytest.raises((ValueError,KeyError)):m.assemble(args,c,r,u,bp,br,data)


def execute(fault=None,at=15):
    p,data,*_=fixture();store=old.MemoryStore();envs=[]
    def factory():e=old.Env(fault=fault,at=at);envs.append(e);return e
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m.mixed,'place',old.fake_place);mp.setattr(m.mixed,'validate_env',lambda _:None)
        mp.setattr(m,'sha',lambda path:hashlib.sha256(store.files[Path(path).name]).hexdigest())
        r=m.execute(p,data,store,factory,authorization=m.AUTH,prepared='/RAM_ONLY/prepare.json',prepare_sha256=old.h('own prepare'))
    return r,store,p,data,envs


@lru_cache(maxsize=1)
def complete():return execute()


def test_real_native_callbacks_four_fresh_full10_and_exact_cp5():
    r,s,p,_,envs=complete();rows=m.validate_saved(r,s.events,p)
    assert r['status']==m.RECORDED,r['errors']
    assert len(envs)==len(rows)==4 and all(e.closed for e in envs)
    assert r['counts']['public_step']['completed']==80 and r['counts']['physics_step']['completed']==816
    assert len([x for x in s.files if x.endswith('.png')])==264
    for row,env,plan in zip(rows,envs,p['plans'],strict=True):
        assert row['own_returned']==10 and row['prefix_reference_exact'] is True
        assert np.asarray(env.commands[10:],np.float32).tobytes()==m.mixed.raw10(plan['commands']).tobytes()
        assert env.dispatch_flags==[False]*20


@pytest.mark.parametrize('bad',['unknown','state','snapshot','raw','reward','goal','counts','PNG','cp5','prefix_claim','missing_guard','oldschema','timing'])
def test_resealed_native_mutations_rejected(bad):
    original,s,pp,_,_=complete();r,p,es=deepcopy(original),deepcopy(pp),deepcopy(s.events)
    obs=next(e for e in es if e['kind']=='fresh_observation')
    if bad=='unknown':obs['kind']='unknown'
    elif bad=='state':obs['receipt']['state'][0]+=1
    elif bad=='snapshot':obs['receipt']['snapshot']['state'][0]+=1
    elif bad=='raw':next(e for e in es if e['kind']=='operation_dispatch_intent' and e['receipt']['resource']=='public_step')['receipt']['raw_bits']='00'
    elif bad=='reward':r['rows'][0]['responses'][0]['reward']=0
    elif bad=='goal':r['rows'][0]['goal_pose'][0]+=1
    elif bad=='counts':r['counts']['physics_step']['completed']-=1
    elif bad=='PNG':obs['receipt']['pixel_sha256']=old.h('bad')
    elif bad=='cp5':p['prefix_reference'][0]['frames'][-1]['receipt']['state'][0]+=1
    elif bad=='prefix_claim':r['rows'][0]['prefix_reference_exact']=False
    elif bad=='missing_guard':r['rows'][0]['trajectory'].pop()
    elif bad=='oldschema':r['schema']=m.base.NAME+'.report'
    else:r['wall_seconds']=m.WALL
    old.reseal(es,r)
    with pytest.raises((ValueError,KeyError,StopIteration)):m.validate_saved(r,es,p)


def test_authorization_before_reads(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('unauthorized reads'))
    with pytest.raises(ValueError):m.run(None,None,None,lambda:pytest.fail('physics'),authorization=None)


def test_cp5_mismatch_blocks_suffix_before_any_step6_intent(monkeypatch):
    original=m.prefix_match
    def mismatch(row,ref):
        if row['prefix51']['complete']:raise ValueError('RAM injected cp5 mismatch')
        return original(row,ref)
    monkeypatch.setattr(m,'prefix_match',mismatch)
    r,s,_,_,envs=execute()
    assert r['status']=='partial_or_failed'
    assert all(len(e.commands)==15 for e in envs)
    assert not any(e['kind']=='operation_dispatch_intent' and e['receipt']['resource']=='public_step'
        and e['receipt']['phase']=='plan' and e['receipt']['step']>=6 for e in s.events)
    guards=[e for e in s.events if e['kind']=='cp5_guard']
    assert len(guards)==4 and all(e['accepted'] is False for e in guards)


def test_cp5_guard_chronology_strictly_precedes_suffix_intent():
    r,s,p,_,_=complete()
    for i in range(4):
        guard=next(e for e in s.events if e['kind']=='cp5_guard' and e['branch']==i)
        cp5=[e for e in s.events if e['kind']=='fresh_observation' and e.get('branch')==i and
            e['receipt']['phase']=='plan' and e['receipt']['step']==5]
        step6=next(e for e in s.events if e['kind']=='operation_dispatch_intent' and e['receipt']['branch']==i
            and e['receipt']['resource']=='public_step' and e['receipt'].get('phase')=='plan' and e['receipt'].get('step')==6)
        assert len(cp5)==3 and max(e['index'] for e in cp5)<guard['index']<step6['index']
    es,rr=deepcopy(s.events),deepcopy(r)
    g=next(e for e in es if e['kind']=='cp5_guard');es.remove(g);es.insert(len(es)-1,g);old.reseal(es,rr)
    with pytest.raises(ValueError):m.validate_saved(rr,es,p)


def test_cold_import_no_neural_or_physics():
    code='''
import builtins
old=builtins.__import__
def guard(name,*a,**k):
    if name.split('.')[0] in ('torch','gymnasium','pymunk','stable_worldmodel'):raise AssertionError(name)
    return old(name,*a,**k)
builtins.__import__=guard
from experiments.spectral_reachability import agent_observed_suffix_physical_20261028_v1 as m
assert m.contract()['no_neural'] is True
'''
    p=subprocess.run([sys.executable,'-B','-c',code],cwd=m.ROOT,capture_output=True,text=True,timeout=30)
    assert p.returncode==0,p.stderr


def test_saved_actual_compile_native_schema_join_without_production():
    root=m.ROOT/'results';out=root/(m.compiler.NAME+'_cuda_MAIN01')
    receipt=root/(m.compiler.NAME+'_consume_MAIN01.json')
    if not receipt.exists():pytest.skip('MAIN actual observed compilation not present')
    assert m.sha(out/'report.json')=='3de58e380acbef66269e69150cab7bca312604b7df88a0f1ae5663764d691b45'
    assert m.sha(receipt)=='09d617dbece12ce3ff2ca7246390e433454b1d8ac29042ab07eadd2be75238ec'
    r=m.read(out/'report.json',m.META_CAP);c=m.read(r['prepared'],m.META_CAP);u=m.read(receipt,m.META_CAP)
    assert m.sha(r['prepared'])==r['prepare_sha256']=='9aa3d63db85857b80c8a614a00a5a7b22c77c26e688e149970a83825c0d74411'
    bp,br=m.reference(c);data=m.mixed.source_material()
    args=dict(compile_output=str(out),compile_report_sha=m.sha(out/'report.json'),compile_receipt=str(receipt),
        compile_receipt_sha=m.sha(receipt),compile_source_sha=m.sha(m.compiler.__file__))
    p=m.assemble(args,c,r,u,bp,br,data)
    assert len(p['plans'])==4 and all(p['prefix_reference'][i]['complete'] for i in range(4))
    assert [x['commands'] for x in p['plans']]==[x['commands'] for x in u['plans']]


def test_saved_complete_consume_verifies_all264_png_and_guard(monkeypatch):
    r,store,p,data,_=complete();files=deepcopy(store.files)
    expected=hashlib.sha256(files['report.json']).hexdigest()
    monkeypatch.setattr(m,'verify',lambda *a:(deepcopy(p),deepcopy(data)))
    monkeypatch.setattr(m.mixed,'saved_artifacts',lambda *a:deepcopy(files))
    out=m.consume('/RAM_ONLY',expected)
    assert out['saved_PNG_verified']==264 and len(out['rows'])==4 and not out['scientific_admission']
    name=next(x for x in files if x.endswith('.png'));body=files.pop(name)
    with pytest.raises((ValueError,KeyError)):m.consume('/RAM_ONLY',expected)
    files[name]=body;files['unknown.txt']=b'unknown'
    with pytest.raises(ValueError):m.consume('/RAM_ONLY',expected)


@pytest.mark.parametrize('fault',['terminal','truncate','guard','throw'])
def test_native_stop_or_engineering_error_never_pads(fault):
    r,s,p,_,envs=execute(fault=fault,at=18)
    assert len(r['rows'])==len(envs)==4 and all(e.closed for e in envs)
    assert all(len(e.commands)==18 for e in envs)
    if fault in ('terminal','truncate'):
        assert r['status']==m.RECORDED and all(row['missing_checkpoints']==[10] for row in r['rows'])
        m.validate_saved(r,s.events,p)
    else:assert r['status']=='partial_or_failed'


def test_compiler_hash_encoding_is_not_own_physical_encoding():
    assert m.digest([1,2])!=m.compiler.digest([1,2])
    p,_,c,_,u,*_=fixture();plan=p['plans'][0];chosen=u['plans'][0]
    assert plan['selection_digest']==m.digest({k:v for k,v in plan.items() if k!='selection_digest'})
    assert chosen['selection_digest']==m.compiler.digest({k:v for k,v in chosen.items() if k!='selection_digest'})
    assert chosen['selection_digest']!=m.digest({k:v for k,v in chosen.items() if k!='selection_digest'})
    assert chosen['goal_sha256']==m.compiler.digest(c['rows'][0]['request']['task']['goal'])


@pytest.mark.parametrize('field',['selection_digest','goal_sha256','observed_context_sha256'])
def test_physical_canonical_digest_cannot_substitute_producer_hash(field):
    _,data,c,r,u,bp,br,args=fixture();chosen=u['plans'][0]
    value={'selection_digest':{k:v for k,v in chosen.items() if k!='selection_digest'},
        'goal_sha256':c['rows'][0]['request']['task']['goal'],
        'observed_context_sha256':c['rows'][0]['request']['observed_context']}[field]
    chosen[field]=m.digest(value);r['rows'][0]['result']['plan']=deepcopy(chosen);u['rows']=deepcopy(r['rows'])
    if field!='selection_digest':
        chosen['selection_digest']=m.compiler.digest({k:v for k,v in chosen.items() if k!='selection_digest'})
        r['rows'][0]['result']['plan']=deepcopy(chosen);u['rows']=deepcopy(r['rows'])
    with pytest.raises(ValueError):m.assemble(args,c,r,u,bp,br,data)
