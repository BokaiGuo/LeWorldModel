"""RAM/fakeenv only: no simulator, neural library, actual prepare or output."""
from copy import deepcopy
import io
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import p1_moving_extended_baseline_physical_20261027_v1 as m
from test_p1_constructive_moving_history_physical_20261027_v1 import ArrayEnv,fixture as moving_fixture,REQUEST
from test_p1_constructive_plan_physical_20261027_v1 import MemoryStore

NEW=['GRASP_H3_bounded','joint_H2_spectral_packet_mapped']
ALL=[a for _,_,arms,_ in m.AUTHORITIES.values() for a in arms]


def fixture(arms=None):
    arms=NEW if arms is None else arms;_,data=moving_fixture();plans=[]
    for s in (0,1):
        for a in arms:
            plan=dict(root_slot=s,root_sha256=data['references'][s]['root_sha256'],arm=a,commands=np.full((10,2),.25,np.float32).tolist())
            plan['selection_digest']=m.source.digest(plan);plans.append(plan)
    return dict(plans=plans,contract=m.contract(arms,600),bindings={},sourcepack_sha256=m.SOURCE_SHA,source_prepare_id='b'*64),data


def execute(factory=ArrayEnv,*,arms=None,p=None,data=None,store=None):
    fp,fd=fixture(arms);p=fp if p is None else p;data=fd if data is None else data
    store=MemoryStore() if store is None else store;envs=[]
    def make():e=factory();envs.append(e);return e
    def digest(path):
        assert Path(path).name=='journal.jsonl'
        return m.source.digest(store.events)
    forbidden=AssertionError('old whole runner/witness/episode forbidden')
    with patch.object(m.kernel.moving,'validate_env',lambda _:None),patch.object(m.source,'sample_state',lambda *a:REQUEST.copy()), \
         patch.object(m,'sha',digest),patch.object(m.previous,'execute',side_effect=forbidden), \
         patch.object(m.kernel.moving,'episode',side_effect=forbidden),patch.object(m.kernel.moving,'witness',side_effect=forbidden):
        r=m.execute(p,data,store,make,authorization=m.AUTH,execution_binding=dict(prepared='/RAM/prepare.json',prepare_sha256='a'*64))
    return r,store,envs


@pytest.mark.parametrize('arms',[NEW])
def test_full_fixed_denominator_calls_and_RGB_interface(arms):
    r,store,envs=execute(arms=arms);n=2*len(arms)
    assert r['status']=='moving_extended_baseline_physical_recorded',r
    values=dict(reset=2*n,set_state=4*n,goal_render=2*n,goal_bind=n,public_step=20*n,physics_step=204*n,
        explicit_render=22*n,environment_construct=n,environment_close=n)
    assert r['counts']=={k:dict(attempted=v,completed=v) for k,v in values.items()}
    assert r['full_horizons']==r['fixed_branch_denominator']==len(r['rows'])==n
    assert r['new_neural_calls']==0 and not r['P1_admission']
    assert r['prepared']=='/RAM/prepare.json' and r['prepare_sha256']=='a'*64
    for row,e in zip(r['rows'],envs,strict=True):
        assert e.closed and row['initial_exact'] and row['full_horizon'] and row['endpoint_scope']=='full10'
        assert [x['step'] for x in row['checkpoints']]==[0,5,10]
        assert all(x['phase']=='plan' and x['png'] in store.files for x in row['checkpoints'])
        assert m.kernel.moving.exact(np.asarray(e.commands[:10]),m.kernel.moving.warmup_commands())
        assert m.kernel.moving.exact(np.asarray(e.commands[10:]),np.full((10,2),.25,np.float32))
        assert row['selection_digest']==row['selected_plan']['selection_digest']
    previous=None
    for event in store.events:
        assert event['previous_sha256']==previous
        assert event['event_sha256']==m.source.digest({k:v for k,v in event.items() if k!='event_sha256'})
        previous=event['event_sha256']
    assert r['journal_chain_tail']==previous


@pytest.mark.parametrize('fault',['truncate','terminal','guard','render','substitute','throw'])
def test_partial_keeps_all_denominators_and_fresh_evidence(fault):
    r,store,envs=execute(lambda:ArrayEnv(fault=fault));n=4
    assert r['fixed_branch_denominator']==len(r['rows'])==n and r['full_horizons']==0
    assert all(e.closed for e in envs) and all(len(e.commands)==11 for e in envs)
    assert r['status']==('moving_extended_baseline_physical_recorded' if fault in ('truncate','terminal') else 'partial_or_failed')
    assert all(not any(c['step'] in (5,10) for c in row['checkpoints']) for row in r['rows'])
    if fault=='throw':assert r['counts']['public_step']==dict(attempted=44,completed=40)
    else:assert r['counts']['physics_step']['completed']==456
    if fault in ('guard','render','substitute'):
        assert all(row['last_observation']['png'] in store.files for row in r['rows'])


@pytest.mark.parametrize('field',['raw','history','states','snapshot'])
def test_warmup_exact_guard_before_any_selected_action(field):
    p,data=fixture();ref=data['references'][0]
    if field=='raw':ref['raw_history']=np.zeros((10,2),np.float32)
    elif field=='history':ref['history'][1,0,0,0]^=1
    elif field=='states':ref['states'][1,0]+=1e-8
    else:ref['dynamics'][1]['state_float64_bits']='bad'
    r,_,_=execute(p=p,data=data)
    assert all(row['own_attempted']==0 and not row['initial_exact'] for row in r['rows'][:2])
    assert all(row['full_horizon'] for row in r['rows'][2:])


def test_primary_preserved_and_close_unconditional_after_error():
    r,_,envs=execute(lambda:ArrayEnv(fault='throw',close_error=True))
    assert all(e.closed for e in envs)
    assert all('primary step' in row['first_error']['message'] and 'close' in str(row['secondary_errors']) for row in r['rows'])
    assert r['counts']['environment_close']==dict(attempted=4,completed=0)


@pytest.mark.parametrize('fault',['construct_return','close_attempt','branch_closed','run_finished','report'])
def test_durable_failure_not_success_and_returned_environment_closed(fault):
    store=MemoryStore();emit,put=store.emit,store.put
    def emit_fault(kind,**v):
        bad=(fault=='construct_return' and kind=='operation_completed' and v.get('resource')=='environment_construct') or \
            (fault=='close_attempt' and kind=='operation_attempted' and v.get('resource')=='environment_close') or kind==fault
        if bad:raise OSError('mock fsync failure')
        return emit(kind,**v)
    def put_fault(name,value,final=False):
        if fault=='report' and final:raise OSError('mock report fsync failure')
        return put(name,value,final)
    store.emit=emit_fault;store.put=put_fault
    r,_,envs=execute(store=store)
    assert r['status']=='partial_or_failed' and all(e.closed for e in envs)
    if fault=='branch_closed':assert len(envs)==1 and all(row['status']=='not_started' for row in r['rows'][1:])
    if fault=='construct_return':assert all(len(e.commands)==0 for e in envs) and r['counts']['environment_close']['completed']==4


def manifest_fixture(arms=None):
    arms=NEW if arms is None else arms;_,data=fixture(arms)
    authorities=[dict(authority=k,output='/MOCK/'+k,report_sha256=m.APPROVED_REPORTS[k])
        for k,(_,_,names,_) in m.AUTHORITIES.items() if any(a in arms for a in names)]
    manifest=dict(schema=m.NAME+'.selection',sourcepack='/MOCK/sourcepack.json',sourcepack_sha256=m.SOURCE_SHA,
        source_prepared='/MOCK/source_prepare.json',source_prepare_sha256='b'*64,
        roots=[dict(root_slot=i,root_sha256=r['root_sha256']) for i,r in enumerate(data['references'])],arms=arms,
        authorities=authorities,selection_fixed_before_physics=True,physical_outcome_selection=False,wall_seconds=600)
    rows={e['authority']:[dict(source_slot=s,arm=a,commands=np.full((10,2),.25,np.float32).tolist(),task_id='task'+str(s),
        blind_sha256=data['references'][s]['blind_sha256'],task_export='/MOCK/task_'+str(s),task_sha256='f'*64)
        for s in (0,1) for a in m.AUTHORITIES[e['authority']][2] if a in arms] for e in authorities}
    return manifest,data,rows


def select(manifest,data,rows):
    with patch.object(m,'ownconsume',side_effect=lambda e:(deepcopy(rows[e['authority']]),{})),patch.object(m,'task_bridge'):
        return m.selection(manifest,data)


@pytest.mark.parametrize('arms',[NEW])
def test_saved_selection_all_ordered_slots_and_explicit_authorities(arms):
    manifest,data,rows=manifest_fixture(arms);plans,_=select(manifest,data,rows)
    assert [(p['root_slot'],p['arm']) for p in plans]==[(s,a) for s in (0,1) for a in arms]
    assert all(p['selection_digest']==m.source.digest({k:v for k,v in p.items() if k!='selection_digest'}) for p in plans)


@pytest.mark.parametrize('fault',['missing','duplicate','boolslot','unknownarm','rounded','outside','rootorder','outcome','authorityorder','extraschema'])
def test_saved_selection_failclosed_no_dispatch(fault):
    manifest,data,rows=manifest_fixture();values=rows['grasp']
    if fault=='missing':values.pop()
    elif fault=='duplicate':values.append(deepcopy(values[0]))
    elif fault=='boolslot':values[0]['source_slot']=True
    elif fault=='unknownarm':values[0]['arm']='GN'
    elif fault=='rounded':values[0]['commands'][0][0]=.1
    elif fault=='outside':values[0]['commands'][0][0]=1.1
    elif fault=='rootorder':manifest['roots'].reverse()
    elif fault=='outcome':manifest['physical_outcome_selection']=True
    elif fault=='authorityorder':manifest['authorities'].reverse()
    else:manifest['unbound_outcome']='do not use'
    with pytest.raises(ValueError):select(manifest,data,rows)


def test_own_authority_hash_before_import():
    with patch.object(m.importlib,'import_module',side_effect=AssertionError('must not import')):
        with pytest.raises(ValueError):m.ownconsume(dict(authority='arbitrary',output='/MOCK',report_sha256='f'*64))
        with patch.object(m,'sha',return_value='wrong'),pytest.raises(ValueError):
            m.ownconsume(dict(authority='grasp',output='/MOCK',report_sha256='f'*64))


@pytest.mark.parametrize('authority',['grasp','no_composition'])
def test_real_ownconsume_signature_plan_hash_and_saved_provenance(authority):
    name,expected,arms,key=m.AUTHORITIES[authority];path=m.ROOT/'experiments/spectral_reachability'/(name+'.py');plans=[];called=[]
    actual=m.APPROVED_REPORTS[authority]
    for s in (0,1):
        plan=dict(schema=name+'.model_plan',source_slot=s,arm=arms[0],commands=np.full((10,2),.25,np.float32).tolist(),
            with_target=False,physical_execution=False,P1_admission=False)
        plans.append(dict(plan,selection_sha256=m.source.digest(plan),whole_run_report_sha256=actual))
    module=NS(__file__=str(path),consume=lambda output,h:(called.append((output,h)) or dict(schema=name+'.consume',P1_admission=False,rows=plans)))
    report=dict(prepared='/MOCK/prepare.json',prepare_sha256='p'*64,artifact_sha256={})
    def hashfile(p):return expected if Path(p)==path else 'p'*64 if str(p).endswith('prepare.json') else actual
    with patch.object(m,'sha',side_effect=hashfile),patch.object(m,'read',side_effect=lambda p:report if str(p).endswith('report.json') else dict(bindings_sha256={})), \
         patch.object(m.importlib,'import_module',return_value=module):
        rows,b=m.ownconsume(dict(authority=authority,output='/MOCK/run',report_sha256=actual))
        assert rows==plans and called==[(Path('/MOCK/run'),actual)] and str(path) in b
        plans[0]['commands'][0][0]=.5
        with pytest.raises(ValueError,match='digest'):m.ownconsume(dict(authority=authority,output='/MOCK/run',report_sha256=actual))


def test_taskbridge_pixels_history_Q_tanh_contract():
    from experiments.spectral_reachability import p1_constructive_moving_history_encode_cuda_20261027_v1 as encoder
    _,data=fixture();ref=data['references'][0]
    t=dict(root_raw_action_history=ref['raw_history'].astype(float).tolist(),projection=np.eye(192).tolist(),packet_logit_cap=float(.75*np.sqrt(5)),task_contract_sha256='d'*64)
    tc=dict(pixel_sha256=[m.kernel.moving.image_sha(a) for a in [*ref['history'],ref['goal_rgb']]],raw_action_sha256=m.kernel.moving.image_sha(ref['raw_history']))
    plan=dict(schema='own.model_plan',source_slot=0,blind_sha256=ref['blind_sha256'],task_id='task',task_export='/MOCK/task',task_sha256='f'*64,
        task_contract_sha256='d'*64,logits=np.zeros((2,10)).tolist(),commands=np.zeros((10,2)).tolist())
    with patch.object(encoder,'load_task',return_value=dict(gn_task=t,task_id='task')), \
         patch.object(encoder,'read',return_value=dict(source_slot=0,blind_sha256=ref['blind_sha256'],task_contract=tc)):
        bindings={};m.task_bridge(plan,ref,bindings);assert bindings['/MOCK/task']=='f'*64
        tc['pixel_sha256'][-1]='badgoal'
        with pytest.raises(ValueError):m.task_bridge(plan,ref,{})


def test_authorization_cap_and_floor_before_dispatch():
    p,data=fixture()
    with pytest.raises(ValueError):m.execute(p,data,MemoryStore(),lambda:pytest.fail('factory'),authorization='wrong')
    with pytest.raises(ValueError):m.run('none','none','none',lambda:pytest.fail('factory'),authorization='wrong')
    ops=m.kernel.Ops(p,MemoryStore());p['contract']['caps']['public_step']=0
    with pytest.raises(ValueError):ops.call('public_step',lambda:pytest.fail('step'))
    store=object.__new__(m.Store);store.path=NS(iterdir=lambda:[])
    with patch.object(m.shutil,'disk_usage',return_value=NS(free=m.FLOOR-1)),pytest.raises(ValueError):store.check(1)


def test_four_branch_saved_artifact_capacity_RAM():
    from PIL import Image
    r,store,_=execute(arms=ALL);sizes=[]
    for value in store.files.values():
        if isinstance(value,np.ndarray):
            b=io.BytesIO();Image.fromarray(value).save(b,format='PNG');sizes.append(len(b.getvalue()))
        else:sizes.append(len(value))
    sizes.append(sum(len(m.raw(e)) for e in store.events))
    assert len([n for n in store.files if n.endswith('.png')])==88
    assert sum(sizes)<m.CAP and len(m.raw(r))<m.FINAL


def test_source_closure_and_no_neural_simulator_import():
    files=m.physical_files()
    assert Path(m.__file__).resolve() in files and Path(m.kernel.__file__).resolve() in files
    assert not any(x in sys.modules for x in ('torch','gymnasium','stable_worldmodel','pymunk'))


@pytest.mark.parametrize('arms',[[NEW[0]],[NEW[1]],list(reversed(NEW)),NEW+NEW,[]])
def test_no_single_arm_or_outcome_subselection(arms):
    with pytest.raises(ValueError,match='BOTH'):m.checked_arms(arms)


@pytest.mark.parametrize('module_name',[
    'p1_moving_grasp_baseline_cuda_20261027_v1','p1_moving_no_composition_baseline_cuda_20261027_v1'])
def test_exact_real_producer_core_canonical_compatible(module_name):
    module=m.importlib.import_module('experiments.spectral_reachability.'+module_name)
    content=dict(schema=module.NAME+'.model_plan',source_slot=0,arm=module.ARM,
        commands=np.asarray([[.1,0]]*10,np.float32).astype(float).tolist(),with_target=False,
        physical_execution=False,P1_admission=False,extra={'unicode':'真实历史','value':-.0})
    assert m.source.digest(content)==module.core.digest(content)


def test_native_constructor_source_only_guard():
    path,expected=m.native_constructor_contract()
    assert path.endswith('stable_worldmodel/envs/pusht/env.py')
    assert expected=='c6d2104adc368ff1c352edc0e35ac74da9f07fda3a748ddbef0ba6b7a2d1daa2'
    assert not any(x in sys.modules for x in ('torch','gymnasium','stable_worldmodel','pymunk'))


def test_constructor_pre_reset_rejected_before_hooks():
    def bad():
        env=ArrayEnv();env.latest_action=np.zeros(2,np.float32);return env
    r,_,envs=execute(bad)
    assert r['status']=='partial_or_failed' and all(e.closed and not e.resets for e in envs)
    assert r['counts']['public_step']==dict(attempted=0,completed=0)
    assert r['fixed_branch_denominator']==4
