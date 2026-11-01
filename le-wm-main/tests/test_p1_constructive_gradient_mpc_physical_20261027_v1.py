"""RAM-only selection/dispatch adapter tests, no native simulator or model."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.spectral_reachability import p1_constructive_gradient_mpc_physical_20261027_v1 as m


def consumed_fixture():
    rows=[]
    for s in (0,1):
        # Identical commands deliberately remain two independently executed plans.
        body=dict(schema=m.compiler.NAME+'.model_plan',source_slot=s,arm=m.ARM,blind_sha256=m.BLIND_SHAS[s],
            task_id=m.TASK_IDS[s],task_export=m.compiler.EXPORTS[s],task_sha256=m.compiler.EXPORT_SHAS[s],
            task_contract_sha256='a'*64,with_target=False,physical_execution=False,selection_uses_physics=False,
            P1_admission=False,commands=np.full((10,2),.125,np.float32).astype(float).tolist())
        rows.append(dict(body,selection_sha256=m.compiler.core.digest(body),whole_run_report_sha256=m.REPORT_SHA))
    return dict(schema=m.compiler.NAME+'.consume',status='own_gradient_model_commands_verified_not_admitted',
        new_physics=0,P1_admission=False,rows=rows)


def test_exact_two_plans_no_dedup_no_padding_or_old_endpoint():
    c=consumed_fixture();p=m.selection_from(c)
    assert p['schema']==m.physical.NAME+'.selection' and p['arms']==[m.ARM] and len(p['plans'])==2
    assert [(x['root_slot'],x['arm']) for x in p['plans']]==[(0,m.ARM),(1,m.ARM)]
    assert all(x['commands']==c['rows'][i]['commands'] and len(x['commands'])==10 for i,x in enumerate(p['plans']))
    assert p['sourcepack_sha256']==m.SOURCEPACK_SHA and p['source_prepare_sha256']==m.SOURCE_PREPARE_SHA
    assert p['roots']==[dict(root_slot=s,root_sha256=h) for s,h in enumerate(m.ROOT_SHAS)]
    assert p['selection_fixed_before_physics'] and not p['physical_outcome_selection']
    assert all('endpoint' not in x and 'prediction' not in x for x in p['plans'])


def test_generic_JSON_pointer_mapping_matches_own_actual_report_fields(monkeypatch):
    c=consumed_fixture();p=m.selection_from(c)
    report=dict(schema=m.compiler.NAME+'.report',status='gradient_mpc_model_plans_sealed_not_admitted',errors=[],
                rows=[{k:v for k,v in row.items() if k in ('source_slot','arm','blind_sha256','commands')} for row in c['rows']])
    data=dict(references=[dict(root_sha256=m.ROOT_SHAS[s],blind_sha256=m.BLIND_SHAS[s]) for s in (0,1)])
    monkeypatch.setattr(m.physical,'read',lambda _:report);monkeypatch.setattr(m.physical,'sha',lambda _:m.REPORT_SHA)
    plans,bindings=m.physical.selection(p,data)
    assert plans==p['plans'] and bindings=={str(m.REPORT):m.REPORT_SHA}
    for s,x in enumerate(plans):
        assert x['commands_pointer']==f'/rows/{s}/commands'
        assert m.physical.source.digest({k:v for k,v in x.items() if k!='selection_digest'})==x['selection_digest']


@pytest.mark.parametrize('damage',('partial','missing','order','boolslot','arm','blind','task','export','exportsha','reportsha','digest','outcome','target','executed','schema'))
def test_only_exact_consume_and_approved_identity(damage):
    c=consumed_fixture();row=c['rows'][0]
    if damage=='partial':c['status']='partial_or_failed'
    elif damage=='missing':c['rows'].pop()
    elif damage=='order':c['rows'].reverse()
    elif damage=='boolslot':row['source_slot']=False
    elif damage=='arm':row['arm']='line_search_GN'
    elif damage=='blind':row['blind_sha256']=m.BLIND_SHAS[1]
    elif damage=='task':row['task_id']=m.TASK_IDS[1]
    elif damage=='export':row['task_export']=m.compiler.EXPORTS[1]
    elif damage=='exportsha':row['task_sha256']=m.compiler.EXPORT_SHAS[1]
    elif damage=='reportsha':row['whole_run_report_sha256']='0'*64
    elif damage=='digest':row['selection_sha256']='0'*64
    elif damage=='outcome':row['selection_uses_physics']=True
    elif damage=='target':row['with_target']=True
    elif damage=='executed':row['physical_execution']=True
    else:row['schema']='other.model_plan'
    with pytest.raises(ValueError):m.selection_from(c)


@pytest.mark.parametrize('raw',(np.zeros((9,2)),np.full((10,2),1.001,np.float32),np.full((10,2),np.nan),np.full((10,2),.1),np.zeros((10,2),bool)))
def test_raw_commands_rejected_not_clipped_or_rounded(raw):
    c=consumed_fixture();row=c['rows'][0];row['commands']=raw.tolist()
    # No self-seal bypass: command validator must itself reject invalid packets.
    if np.isfinite(raw).all():
        body={k:v for k,v in row.items() if k not in ('selection_sha256','whole_run_report_sha256')}
        row['selection_sha256']=m.compiler.core.digest(body)
    with pytest.raises(ValueError):m.selection_from(c)


def material_fixture(monkeypatch):
    manifest=m.selection_from(consumed_fixture());h='a'*64
    generic=dict(schema=m.physical.NAME+'.prepare',manifest='/RAM/selection.json',manifest_sha256=h,
        plans=manifest['plans'],contract=m.physical.contract(2,m.WALL),bindings={},source_prepare_id='b'*64,
        sourcepack_sha256=m.SOURCEPACK_SHA,source_slots=manifest['roots'])
    calls=[]
    # The expected compiler-derived manifest must not alias the caller's
    # simulated on-disk manifest, otherwise tampering changes both sides.
    expected_manifest=deepcopy(manifest)
    monkeypatch.setattr(m,'build',lambda:(deepcopy(expected_manifest),{'/RAM/adapter.py':h}))
    monkeypatch.setattr(m,'read',lambda _:deepcopy(manifest));monkeypatch.setattr(m,'sha',lambda _:h)
    monkeypatch.setattr(m.physical,'material',lambda path,expected:(calls.append((path,expected)) or deepcopy(generic)))
    return manifest,generic,calls,h


def test_prepare_delegation_adds_source_seal_without_altering_safety_contract(monkeypatch):
    manifest,generic,calls,h=material_fixture(monkeypatch)
    p=m.material('/RAM/selection.json',h)
    assert calls==[('/RAM/selection.json',h)] and p['schema']==m.NAME+'.prepare'
    assert p['contract']==generic['contract'] and p['plans']==generic['plans'] and p['bindings']=={'/RAM/adapter.py':h}
    assert p['adapter']['compiler_report_sha256']==m.REPORT_SHA and p['adapter']['compiler_prepare_sha256']==m.PREPARE_SHA


def test_prepare_rejects_modified_manifest_before_generic_source_access(monkeypatch):
    manifest,_,calls,h=material_fixture(monkeypatch);manifest['plans'][0]['commands'][0][0]=.5
    monkeypatch.setattr(m,'read',lambda _:manifest)
    with pytest.raises(ValueError,match='exact own'):m.material('/RAM/selection.json',h)
    assert calls==[]


def test_prepare_rejects_generic_contract_drift(monkeypatch):
    _,generic,_,h=material_fixture(monkeypatch);generic['contract']['own_horizon']=5
    monkeypatch.setattr(m.physical,'material',lambda *_:generic)
    with pytest.raises(ValueError,match='original generic'):m.material('/RAM/selection.json',h)


def test_run_authorization_precedes_verify_source_or_factory(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('unauthorized verify'))
    with pytest.raises(ValueError,match='MAIN-only'):m.run('x','x','x',lambda:pytest.fail('factory'),authorization='no')


@pytest.mark.parametrize('status',('physical_calibration_recorded','partial_or_failed'))
def test_run_calls_generic_executor_once_and_preserves_result(monkeypatch,status):
    manifest=m.selection_from(consumed_fixture());p=dict(manifest='/RAM/selection.json',bindings={'adapter':'a'*64})
    calls=[];data=dict(mock='reference data, not model input');payload=dict(schema=m.physical.NAME+'.report',status=status,rows=[{},{}])
    class Store:
        def __init__(self,output):self.output=output
        def put(self,name,value):calls.append(('put',name,value))
    monkeypatch.setattr(m,'verify',lambda *a:p);monkeypatch.setattr(m,'read',lambda _:manifest)
    monkeypatch.setattr(m.physical,'load_source',lambda *a:(calls.append(('source',a)) or data))
    monkeypatch.setattr(m.physical,'Store',Store)
    factory=object()
    def execute(actual,refs,store,make,*,authorization):
        assert actual is p and refs is data and make is factory and authorization==m.physical.AUTH
        calls.append(('execute',));return payload
    monkeypatch.setattr(m.physical,'execute',execute)
    result=m.run('/RAM/prepare.json','a'*64,'/RAM/unused',factory,authorization=m.physical.AUTH)
    assert result is payload and sum(c[0]=='execute' for c in calls)==1
    assert calls[0]==('source',(str(m.SOURCEPACK),m.SOURCEPACK_SHA,str(m.SOURCE_PREPARED),m.SOURCE_PREPARE_SHA))
    assert calls[1][1]=='execution_binding.json'


def test_verify_requires_full_current_reconstruction(monkeypatch):
    p=dict(manifest='/RAM/selection.json',manifest_sha256='b'*64)
    monkeypatch.setattr(m,'sha',lambda _:'a'*64);monkeypatch.setattr(m,'read',lambda _:p)
    monkeypatch.setattr(m,'material',lambda *a:dict(p,changed=True))
    with pytest.raises(ValueError,match='entire own'):m.verify('/RAM/prepare.json','a'*64)


def array_physical_fixture():
    spec=importlib.util.spec_from_file_location('gradient_generic_array_fixture',Path(__file__).with_name('test_p1_constructive_plan_physical_20261027_v1.py'))
    f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
    p,data=f.fixture();p['plans']=[deepcopy(p['plans'][i]) for i in (0,2)]
    for row in p['plans']:row['arm']=m.ARM
    p['contract']=m.physical.contract(2,60)
    return f,p,data


def test_generic_array_full_two_roots_20own_20warmup():
    f,p,data=array_physical_fixture();r,store,envs=f.execute(p=p,data=data)
    assert r['status']=='physical_calibration_recorded' and len(r['rows'])==2 and r['native_successes']==0
    assert r['counts']=={k:dict(attempted=v,completed=v) for k,v in dict(reset=4,public_step=40,explicit_render=44,
        goal_render=4,set_state=8,physics_step=408,goal_bind=2).items()}
    for i,(row,env) in enumerate(zip(r['rows'],envs,strict=True)):
        assert row['own_returned']==10 and row['warmup_returned']==10 and row['full_horizon']
        assert np.array_equal(env.sent[10:],p['plans'][i]['commands']) and len(env.sent)==20 and env.closed
    assert r['new_neural_calls']==0 and not r['P1_admission']


def test_generic_array_guard_keeps_both_partial_branches():
    f,p,data=array_physical_fixture();r,store,envs=f.execute(lambda:f.ArrayEnv(guard_fail=11),p=p,data=data)
    assert r['status']=='partial_or_failed' and len(r['rows'])==2
    assert all(row['own_returned']==1 and not row['full_horizon'] for row in r['rows'])
    assert all(len(env.sent)==11 for env in envs)
    assert len([e for e in store.events if e['kind']=='guard_receipt' and not e['accepted']])==2


def test_write_gate_has_no_tmp_and_rejects_insufficient_reserve(monkeypatch):
    class Path:
        parent='RAM_PARENT'
        def absolute(self):return self
        def exists(self):return False
        def open(self,*a,**kw):pytest.fail('must not write with inadequate reserve')
    monkeypatch.setattr(m,'Path',lambda _:Path())
    monkeypatch.setattr(m.shutil,'disk_usage',lambda _:SimpleNamespace(free=m.physical.FLOOR))
    with pytest.raises(ValueError,match='NEW small'):m.write_new('RAM',{})
