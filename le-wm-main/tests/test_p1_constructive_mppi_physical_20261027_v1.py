"""Pure RAM adapter/array mocks, no simulator, model or pytest tmp fixtures."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.spectral_reachability import p1_constructive_mppi_physical_20261027_v1 as m


def receipt():
    rows=[]
    for s in (0,1):
        body=dict(schema=m.compiler.NAME+'.model_plan',source_slot=s,arm=m.ARM,task_id=m.TASK_IDS[s],
            blind_sha256=m.BLIND_SHAS[s],task_export=m.compiler.sampling.EXPORTS[s],task_sha256=m.compiler.sampling.EXPORT_SHAS[s],
            with_target=False,physical_execution=False,selection_uses_physics=False,P1_admission=False,official_MPPI=False,
            selected_index=34,selection='last_updated_nominal_not_best',commands=np.full((10,2),.125,np.float32).tolist())
        rows.append(dict(body,selection_sha256=m.compiler.core.digest(body),whole_run_report_sha256=m.REPORT_SHA))
    return dict(schema=m.compiler.NAME+'.consume',rows=rows,new_physics=0,P1_admission=False,
        official_MPPI=False,paper_strong_MPPI_requirement_closed=False)


def test_two_fixed_roots_one_arm_no_dedup():
    c=receipt();p=m.selection_from(c)
    assert p['arms']==[m.ARM] and len(p['plans'])==2
    assert [(r['root_slot'],r['arm']) for r in p['plans']]==[(0,m.ARM),(1,m.ARM)]
    assert p['roots']==[dict(root_slot=s,root_sha256=h) for s,h in enumerate(m.ROOT_SHAS)]
    assert [r['commands'] for r in p['plans']]==[r['commands'] for r in c['rows']]
    assert p['selection_fixed_before_physics'] and not p['physical_outcome_selection']


@pytest.mark.parametrize('field,value',[
    ('source_slot',False),('source_slot',1),('arm','best_sample'),('schema','other'),('task_id','x'),
    ('blind_sha256','0'*64),('whole_run_report_sha256','0'*64),('task_export','other'),('task_sha256','0'*64),
    ('with_target',True),('physical_execution',True),('selection_uses_physics',True),('P1_admission',True),
    ('official_MPPI',True),('selected_index',7),('selection','best_sample'),('selection_sha256','0'*64)])
def test_identity_nominal_and_seal_rejected(field,value):
    c=receipt();c['rows'][0][field]=value
    with pytest.raises(ValueError):m.selection_from(c)


@pytest.mark.parametrize('damage',('missing','reverse','official','admitted'))
def test_census_and_claims(damage):
    c=receipt()
    if damage=='missing':c['rows'].pop()
    elif damage=='reverse':c['rows'].reverse()
    elif damage=='official':c['official_MPPI']=True
    else:c['paper_strong_MPPI_requirement_closed']=True
    with pytest.raises(ValueError):m.selection_from(c)


@pytest.mark.parametrize('raw',(np.zeros((9,2)),np.full((10,2),1.001,np.float32),np.full((10,2),np.nan),np.full((10,2),.1),np.zeros((10,2),bool)))
def test_no_command_clipping_or_rounding(raw):
    c=receipt();c['rows'][0]['commands']=raw.tolist()
    with pytest.raises(ValueError):m.selection_from(c)


def test_actual_report_pointer_grammar(monkeypatch):
    c=receipt();p=m.selection_from(c)
    report=dict(schema=m.compiler.NAME+'.report',status='kernel_nominal_plans_sealed_not_admitted',errors=[],rows=c['rows'])
    monkeypatch.setattr(m.physical,'sha',lambda _:m.REPORT_SHA);monkeypatch.setattr(m.physical,'read',lambda _:report)
    refs=dict(references=[dict(root_sha256=m.ROOT_SHAS[s],blind_sha256=m.BLIND_SHAS[s]) for s in (0,1)])
    plans,bindings=m.physical.selection(p,refs)
    assert plans==p['plans'] and bindings=={str(m.REPORT):m.REPORT_SHA}


def material_fixture(monkeypatch):
    manifest=m.selection_from(receipt());wanted=deepcopy(manifest);h='a'*64
    generic=dict(manifest='/RAM/selection',manifest_sha256=h,plans=deepcopy(manifest['plans']),
        sourcepack_sha256=m.SOURCEPACK_SHA,source_slots=manifest['roots'],contract=m.physical.contract(2,m.WALL),bindings={})
    monkeypatch.setattr(m,'build',lambda:(deepcopy(wanted),{'adapter':h}))
    monkeypatch.setattr(m,'sha',lambda _:h);monkeypatch.setattr(m,'read',lambda _:deepcopy(manifest))
    monkeypatch.setattr(m.physical,'material',lambda *_:deepcopy(generic))
    return manifest,generic,h


def test_prepare_preserves_generic_contract_and_search_cost(monkeypatch):
    _,g,h=material_fixture(monkeypatch);p=m.material('/RAM/selection',h)
    assert p['schema']==m.NAME+'.prepare' and p['contract']==g['contract'] and p['plans']==g['plans']
    assert p['bindings']=={'adapter':h} and p['adapter']['search_cost_per_root']==dict(predict=72,action_encoder=72)
    assert not p['adapter']['official_MPPI'] and not p['adapter']['P1_admission']


def test_tampered_manifest_rejected_before_source_access(monkeypatch):
    manifest,_,h=material_fixture(monkeypatch);manifest['plans'][0]['commands'][0][0]=.5
    monkeypatch.setattr(m.physical,'material',lambda *_:pytest.fail('source accessed'))
    with pytest.raises(ValueError,match='exact own'):m.material('/RAM/selection',h)


def test_generic_contract_drift_rejected(monkeypatch):
    _,g,h=material_fixture(monkeypatch);g['contract']['own_horizon']=5
    with pytest.raises(ValueError,match='original generic'):m.material('/RAM/selection',h)


def test_verify_reconstructs_entire_prepare(monkeypatch):
    p=dict(manifest='RAM',manifest_sha256='a'*64)
    monkeypatch.setattr(m,'sha',lambda _:'a'*64);monkeypatch.setattr(m,'read',lambda _:p)
    monkeypatch.setattr(m,'material',lambda *_:dict(p,changed=True))
    with pytest.raises(ValueError,match='entire own'):m.verify('RAM','a'*64)


def test_unauthorized_cannot_open_source_or_factory(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *_:pytest.fail('verify reached'))
    with pytest.raises(ValueError,match='MAIN-only'):m.run('x','x','x',lambda:pytest.fail('factory'),authorization='no')


@pytest.mark.parametrize('status',('physical_calibration_recorded','partial_or_failed'))
def test_single_delegate_preserves_partial_and_full(monkeypatch,status):
    manifest=m.selection_from(receipt());p=dict(manifest='RAM');data=object();factory=object();calls=[]
    class Store:
        def __init__(self,output):self.output=output
        def put(self,name,value):calls.append((name,value))
    monkeypatch.setattr(m,'verify',lambda *_:p);monkeypatch.setattr(m,'read',lambda _:manifest)
    monkeypatch.setattr(m.physical,'load_source',lambda *args:data);monkeypatch.setattr(m.physical,'Store',Store)
    result=dict(status=status,rows=[{},{}])
    def execute(actual,refs,store,make,*,authorization):
        assert actual is p and refs is data and make is factory and authorization==m.physical.AUTH
        calls.append('execute');return result
    monkeypatch.setattr(m.physical,'execute',execute)
    assert m.run('RAM','a'*64,'RAM',factory,authorization=m.physical.AUTH) is result
    assert calls[0][0]=='execution_binding.json' and calls.count('execute')==1


@pytest.mark.parametrize('guard_fail',(None,11))
def test_generic_array_safety_and_complete_denominator(guard_fail):
    spec=importlib.util.spec_from_file_location('mppi_generic_array_fixture',Path(__file__).with_name('test_p1_constructive_plan_physical_20261027_v1.py'))
    f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
    p,data=f.fixture();p['plans']=[deepcopy(p['plans'][i]) for i in (0,2)]
    for r in p['plans']:r['arm']=m.ARM
    p['contract']=m.physical.contract(2,60)
    r,store,envs=f.execute((lambda:f.ArrayEnv(guard_fail=guard_fail)) if guard_fail else f.ArrayEnv,p=p,data=data)
    assert len(r['rows'])==2 and r['new_neural_calls']==0 and not r['P1_admission']
    if guard_fail:
        assert r['status']=='partial_or_failed' and all(x['own_returned']==1 for x in r['rows'])
        assert len([e for e in store.events if e['kind']=='guard_receipt' and not e['accepted']])==2
    else:
        assert r['status']=='physical_calibration_recorded'
        assert r['counts']=={k:dict(attempted=v,completed=v) for k,v in dict(reset=4,public_step=40,explicit_render=44,goal_render=4,set_state=8,physics_step=408,goal_bind=2).items()}
        for plan,env in zip(p['plans'],envs,strict=True):assert np.array_equal(env.sent[10:],plan['commands'])
    assert all(env.closed for env in envs)


def test_disk_floor_blocks_writes(monkeypatch):
    class RAMPath:
        parent='RAM'
        def absolute(self):return self
        def exists(self):return False
        def open(self,*a,**kw):pytest.fail('disk floor must precede write')
    monkeypatch.setattr(m,'Path',lambda _:RAMPath());monkeypatch.setattr(m.shutil,'disk_usage',lambda _:SimpleNamespace(free=m.physical.FLOOR))
    with pytest.raises(ValueError,match='NEW small'):m.write_new('RAM',{})
