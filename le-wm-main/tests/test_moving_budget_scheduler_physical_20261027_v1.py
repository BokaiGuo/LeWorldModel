"""RAM/fakeenv only. No actual prepare, sourcepack, neural or simulator."""
from copy import deepcopy
import io
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import moving_budget_scheduler_physical_20261027_v1 as m
from test_p1_constructive_moving_history_physical_20261027_v1 import ArrayEnv,fixture as moving_fixture,REQUEST
from test_p1_constructive_plan_physical_20261027_v1 import MemoryStore


def fixture():
    _,data=moving_fixture();plans=[]
    for s in (0,1):
        for a in m.ARMS:
            p=dict(root_slot=s,arm=a,root_sha256=data['references'][s]['root_sha256'],commands=np.full((10,2),.25,np.float32).tolist())
            p['selection_digest']=m.source.digest(p);plans.append(p)
    return dict(plans=plans,contract=m.contract(600),bindings={},sourcepack_sha256=m.SOURCE_SHA,source_prepare_id='b'*64),data


def execute(factory=ArrayEnv,*,p=None,data=None,store=None):
    fp,fd=fixture();p=fp if p is None else p;data=fd if data is None else data;store=MemoryStore() if store is None else store;envs=[]
    def make():e=factory();envs.append(e);return e
    def digest(path):
        assert Path(path).name=='journal.jsonl'
        return m.source.digest(store.events)
    with patch.object(m.kernel.moving,'validate_env',lambda _:None),patch.object(m.source,'sample_state',lambda *a:REQUEST.copy()), \
         patch.object(m,'sha',digest),patch.object(m.helper,'execute',side_effect=AssertionError('old runner forbidden')), \
         patch.object(m.kernel.moving,'episode',side_effect=AssertionError('source simulation forbidden')), \
         patch.object(m.kernel.moving,'witness',side_effect=AssertionError('witness forbidden')):
        r=m.execute(p,data,store,make,authorization=m.AUTH,execution_binding=dict(prepared='/RAM/prepare.json',prepare_sha256='a'*64))
    return r,store,envs


def test_complete_eight_distinct_dispatches_even_identical_commands():
    r,store,envs=execute();assert r['status']=='moving_budget_scheduler_physical_recorded',r
    counts=dict(reset=16,set_state=32,goal_render=16,goal_bind=8,public_step=160,physics_step=1632,
        explicit_render=176,environment_construct=8,environment_close=8)
    assert r['counts']=={k:dict(attempted=v,completed=v) for k,v in counts.items()}
    assert len(envs)==r['fixed_branch_denominator']==r['full_horizons']==8
    assert [(r['root_slot'],r['arm']) for r in r['rows']]==[(s,a) for s in (0,1) for a in m.ARMS]
    assert not any(r[k] for k in ('P1_admission','P2_admission','P3_admission','independent_confirmation','compute_advantage'))
    assert r['new_neural_calls']==0
    for row,e in zip(r['rows'],envs,strict=True):
        assert e.closed and row['initial_exact'] and row['endpoint_scope']=='full10'
        assert row['constructor_internal_work']=='sealed_native_pristine_no_physics'
        assert m.kernel.moving.exact(np.asarray(e.commands[:10]),m.kernel.moving.warmup_commands())
        assert len(e.commands)==20 and [c['step'] for c in row['checkpoints']]==[0,5,10]
        assert all(c['png'] in store.files for c in row['checkpoints'])
    previous=None
    for e in store.events:
        assert e['previous_sha256']==previous and e['event_sha256']==m.source.digest({k:v for k,v in e.items() if k!='event_sha256'})
        previous=e['event_sha256']
    assert previous==r['journal_chain_tail']


@pytest.mark.parametrize('fault',['truncate','terminal','guard','render','substitute','throw'])
def test_partial_full_denominator_and_no_padded_terminal(fault):
    r,store,envs=execute(lambda:ArrayEnv(fault=fault))
    assert len(r['rows'])==8 and r['full_horizons']==0 and all(e.closed and len(e.commands)==11 for e in envs)
    assert all(not any(c['step'] in (5,10) for c in row['checkpoints']) for row in r['rows'])
    assert r['status']==('moving_budget_scheduler_physical_recorded' if fault in ('terminal','truncate') else 'partial_or_failed')
    if fault=='throw':assert r['counts']['public_step']==dict(attempted=88,completed=80)
    else:assert r['counts']['physics_step']['completed']==912
    if fault in ('guard','render','substitute'):assert all(row['last_observation']['png'] in store.files for row in r['rows'])


@pytest.mark.parametrize('field',['raw','RGB','state','body'])
def test_original_history_guards_precede_ALL_four_root_actions(field):
    p,data=fixture();r=data['references'][0]
    if field=='raw':r['raw_history']=np.zeros((10,2),np.float32)
    elif field=='RGB':r['history'][1,0,0,0]^=1
    elif field=='state':r['states'][1,0]+=1e-8
    else:r['dynamics'][1]['state_float64_bits']='bad'
    result,_,_=execute(p=p,data=data)
    assert all(row['own_attempted']==0 and not row['initial_exact'] for row in result['rows'][:4])
    assert all(row['full_horizon'] for row in result['rows'][4:])


def test_constructor_exception_is_machine_readable_unknown_not_exact_zero():
    def fail():raise RuntimeError('mock constructor half failure')
    r,_,envs=execute(fail)
    assert not envs and len(r['rows'])==8 and r['status']=='partial_or_failed'
    assert all(row['constructor_internal_work']=='unknown' for row in r['rows'])
    assert r['total_internal_physics_scope']=='unknown_constructor_work'
    assert r['counts']['environment_construct']==dict(attempted=8,completed=0)
    assert r['counts']['physics_step']==dict(attempted=0,completed=0)  # observed hook count only, NOT constructor total


@pytest.mark.parametrize('stage',['construct_return','close_attempt','branch_closed','run_finished','report'])
def test_durable_failure_keeps_fixed_denominator_and_cleanup(stage):
    store=MemoryStore();emit,put=store.emit,store.put
    def event(kind,**v):
        if (stage=='construct_return' and kind=='operation_completed' and v.get('resource')=='environment_construct') or \
           (stage=='close_attempt' and kind=='operation_attempted' and v.get('resource')=='environment_close') or stage==kind:
            raise OSError('mock fsync failed')
        return emit(kind,**v)
    def final(name,value,final=False):
        if stage=='report' and final:raise OSError('mock report failed')
        return put(name,value,final)
    store.emit=event;store.put=final;r,_,envs=execute(store=store)
    assert r['status']=='partial_or_failed' and len(r['rows'])==8 and all(e.closed for e in envs)
    if stage=='branch_closed':assert len(envs)==1 and all(row['status']=='not_started' for row in r['rows'][1:])
    if stage=='construct_return':assert r['total_internal_physics_scope']=='unknown_constructor_work'


def test_primary_step_survives_close_failure():
    r,_,envs=execute(lambda:ArrayEnv(fault='throw',close_error=True))
    assert all(e.closed for e in envs) and r['counts']['environment_close']==dict(attempted=8,completed=0)
    assert all('primary step' in row['first_error']['message'] and 'close' in str(row['secondary_errors']) for row in r['rows'])


def source_rows():
    _,data=fixture();rows=[]
    for a in m.ARMS:  # scheduler method-major receipt -> physical root-major identity join
        for s in (0,1):
            p=dict(schema=m.AUTHORITY+'.model_plan',source_slot=s,root_sha256=data['references'][s]['root_sha256'],arm=a,
                commands=np.full((10,2),.25,np.float32).tolist(),task_id='task'+str(s),task_export='/MOCK/task_'+str(s),task_sha256='f'*64,
                task_contract_sha256='d'*64,bank_sha256='e'*64,blind_sha256=data['references'][s]['blind_sha256'],physical_execution=False,
                selection_uses_physics=False,P1_admission=False,P3_admission=False,independent_confirmation=False,compute_advantage=False,walltime_advantage=False)
            rows.append(dict(p,selection_sha256=m.source.digest(p),whole_run_report_sha256='c'*64))
    return data,rows


def manifest_fixture():
    data,rows=source_rows()
    return dict(schema=m.NAME+'.selection',sourcepack='/MOCK/sourcepack.json',sourcepack_sha256=m.SOURCE_SHA,
        source_prepared='/MOCK/source_prepare.json',source_prepare_sha256='b'*64,
        roots=[dict(root_slot=i,root_sha256=r['root_sha256']) for i,r in enumerate(data['references'])],arms=list(m.ARMS),
        authorities=[dict(authority=m.AUTHORITY,output='/MOCK/scheduler',report_sha256='c'*64)],
        selection_fixed_before_physics=True,physical_outcome_selection=False,wall_seconds=600),data,rows


def test_strict_saved_selection_and_provenance_reorder_not_outcome_order():
    manifest,data,rows=manifest_fixture()
    with patch.object(m,'ownconsume',return_value=(rows,{})),patch.object(m,'task_bridge'):
        plans,bindings=m.selection(manifest,data)
    assert [(p['root_slot'],p['arm']) for p in plans]==[(s,a) for s in (0,1) for a in m.ARMS]
    for p in plans:
        row=next(x for x in rows if (x['source_slot'],x['arm'])==(p['root_slot'],p['arm']))
        assert p['own_plan_sha256']==m.source.digest(row) and p['compiler_selection_sha256']==row['selection_sha256']


@pytest.mark.parametrize('fault',['single','reorder','missing','duplicate','boolslot','unknownarm','rounded','outside','rootorder','outcome'])
def test_saved_join_failclosed_all_eight(fault):
    manifest,data,rows=manifest_fixture()
    if fault=='single':manifest['arms']=m.ARMS[:1]
    elif fault=='reorder':manifest['arms']=list(reversed(m.ARMS))
    elif fault=='missing':rows.pop()
    elif fault=='duplicate':rows.append(deepcopy(rows[0]))
    elif fault=='boolslot':rows[0]['source_slot']=True
    elif fault=='unknownarm':rows[0]['arm']='old_GN'
    elif fault=='rounded':rows[0]['commands'][0][0]=.1
    elif fault=='outside':rows[0]['commands'][0][0]=1.1
    elif fault=='rootorder':manifest['roots'].reverse()
    else:manifest['physical_outcome_selection']=True
    with patch.object(m,'ownconsume',return_value=(rows,{})),patch.object(m,'task_bridge'),pytest.raises(ValueError):m.selection(manifest,data)


def test_pending_authority_refuses_before_import_or_report_read():
    with patch.object(m,'AUTHORITY_SHA',None),patch.object(m.importlib,'import_module',side_effect=AssertionError('no unfinished import')), \
         patch.object(m,'read',side_effect=AssertionError('no read')),pytest.raises(ValueError,match='pending final'):
        m.ownconsume(dict(authority=m.AUTHORITY,output='/MOCK/run',report_sha256='f'*64))


def test_eight_modelplan_ownconsume_API_with_mock_FINAL_authority_only():
    _,rows=source_rows();calls=[];slots=[{k:v for k,v in rows[s].items() if k in ('source_slot','root_sha256','task_id','blind_sha256','task_export','task_sha256','task_contract_sha256','bank_sha256')} for s in (0,1)]
    path=m.ROOT/'experiments/spectral_reachability'/(m.AUTHORITY+'.py');source_sha='a'*64
    flags=dict(P1_admission=False,P3_admission=False,independent_confirmation=False,compute_advantage=False,walltime_advantage=False)
    receipt=dict(schema=m.AUTHORITY+'.consume',status='four_policy_commands_verified_not_admitted',errors=[],plans=rows,**flags)
    mod=NS(__file__=str(path),consume=lambda output,h:(calls.append((output,h)) or receipt))
    report=dict(prepared='/MOCK/prepare.json',prepare_sha256='b'*64,artifact_sha256={})
    prep=dict(schema=m.AUTHORITY+'.prepare',slots=slots,bindings_sha256={})
    def hashfile(p):return m.authority_files().get(str(Path(p)), 'b'*64 if str(p).endswith('prepare.json') else 'c'*64)
    with patch.object(m,'AUTHORITY_SHA',source_sha),patch.object(m,'sha',side_effect=hashfile), \
         patch.object(m,'read',side_effect=lambda p:report if str(p).endswith('report.json') else prep),patch.object(m.importlib,'import_module',return_value=mod):
        selected,bindings=m.ownconsume(dict(authority=m.AUTHORITY,output='/MOCK/run',report_sha256='c'*64))
        assert len(selected)==8 and calls==[(Path('/MOCK/run'),'c'*64)] and str(path) in bindings
        rows[0]['commands'][0][0]=.5
        with pytest.raises(ValueError,match='seal'):m.ownconsume(dict(authority=m.AUTHORITY,output='/MOCK/run',report_sha256='c'*64))


def test_auth_caps_and_floor_before_dispatch():
    p,data=fixture()
    with pytest.raises(ValueError):m.execute(p,data,MemoryStore(),lambda:pytest.fail('factory'),authorization='wrong')
    with pytest.raises(ValueError):m.run('none','none','none',lambda:pytest.fail('factory'),authorization='wrong')
    ops=m.kernel.Ops(p,MemoryStore());p['contract']['caps']['public_step']=0
    with pytest.raises(ValueError):ops.call('public_step',lambda:pytest.fail('step'))
    store=object.__new__(m.Store);store.path=NS(iterdir=lambda:[])
    with patch.object(m.helper.shutil,'disk_usage',return_value=NS(free=m.FLOOR-1)),pytest.raises(ValueError):store.check(1)


def test_saved_eight_branch_capacity_and_checkpoint_schema_RAM():
    from PIL import Image
    r,store,_=execute();sizes=[]
    for value in store.files.values():
        if isinstance(value,np.ndarray):
            b=io.BytesIO();Image.fromarray(value).save(b,format='PNG');sizes.append(len(b.getvalue()))
        else:sizes.append(len(value))
    sizes.append(sum(len(m.raw(e)) for e in store.events))
    assert len([n for n in store.files if n.endswith('.png')])==176 and sum(sizes)<m.CAP
    assert len(m.raw(r))<m.FINAL
    assert set(r['rows'][0]['checkpoints'][1])==set('phase step png file_sha256 pixel_sha256 state snapshot observation_state'.split())


def test_no_neural_or_simulator_import():
    assert not any(x in sys.modules for x in ('torch','gymnasium','stable_worldmodel','pymunk'))


@pytest.mark.parametrize('branch',range(8))
@pytest.mark.parametrize('fault',['constructor','close','step','truncate'])
def test_each_single_failed_or_partial_branch_retains_all_eight_identities(branch,fault):
    index=0
    def factory():
        nonlocal index
        i=index;index+=1
        if i==branch:
            if fault=='constructor':raise RuntimeError('target constructor partial work unknown')
            if fault=='close':return ArrayEnv(close_error=True)
            if fault=='step':return ArrayEnv(fault='throw')
            return ArrayEnv(fault='truncate')
        return ArrayEnv()
    r,store,envs=execute(factory)
    assert index==len(r['rows'])==r['fixed_branch_denominator']==8
    assert [(row['root_slot'],row['arm']) for row in r['rows']]==[(s,a) for s in (0,1) for a in m.ARMS]
    target=r['rows'][branch]
    others=[row for i,row in enumerate(r['rows']) if i!=branch]
    assert all(row['status']=='horizon_completed' and row['full_horizon'] and row['initial_exact'] for row in others)
    assert all(e.closed for e in envs)
    assert [e['receipt']['branch_index'] for e in store.events if e['kind']=='branch_closed']==list(range(8))
    if fault=='constructor':
        assert target['status']=='failed' and not target['initial_exact'] and target['own_attempted']==target['own_returned']==0
        assert target['constructor_internal_work']=='unknown' and not target['checkpoints']
        assert r['total_internal_physics_scope']=='unknown_constructor_work'
        assert r['counts']['environment_construct']==dict(attempted=8,completed=7)
        assert r['counts']['environment_close']==dict(attempted=7,completed=7)
        assert r['counts']['public_step']==dict(attempted=140,completed=140)
        assert r['counts']['physics_step']==dict(attempted=1428,completed=1428)
        assert r['full_horizons']==7
    elif fault=='close':
        assert target['status']=='failed' and target['full_horizon'] and target['own_returned']==10
        assert target['first_error']['stage']=='close'
        assert r['counts']['environment_close']==dict(attempted=8,completed=7)
        assert r['counts']['physics_step']==dict(attempted=1632,completed=1632)
        assert r['full_horizons']==8  # physical execution completed, cleanup failed: distinct axes
        assert [c['step'] for c in target['checkpoints']]==[0,5,10]
    elif fault=='step':
        assert target['status']=='failed' and target['initial_exact'] and not target['full_horizon']
        assert target['own_attempted']==1 and target['own_returned']==0
        assert 'primary step' in target['first_error']['message']
        assert r['counts']['public_step']==dict(attempted=151,completed=150)
        assert r['counts']['physics_step']==dict(attempted=1542,completed=1542)
        assert [c['step'] for c in target['checkpoints']]==[0]
        assert r['full_horizons']==7
    else:
        assert target['status']=='native_stopped' and target['initial_exact'] and target['truncated']
        assert target['own_attempted']==target['own_returned']==1 and not target['full_horizon']
        assert r['counts']['public_step']==dict(attempted=151,completed=151)
        assert r['counts']['physics_step']==dict(attempted=1542,completed=1542)
        assert [c['step'] for c in target['checkpoints']]==[0]
        assert r['full_horizons']==7
    assert r['status']==('moving_budget_scheduler_physical_recorded' if fault=='truncate' else 'partial_or_failed')


@pytest.mark.parametrize('kind',['source','tests','doc'])
def test_final_authority_any_source_test_doc_drift_rejected_before_import(kind):
    pins=m.authority_files();paths=list(pins);wrong=paths[['source','tests','doc'].index(kind)]
    def digest(path):return '0'*64 if str(path)==wrong else pins[str(path)]
    with patch.object(m,'sha',side_effect=digest),patch.object(m.importlib,'import_module',side_effect=AssertionError('drift must reject first')), \
         pytest.raises(ValueError,match='before import'):
        m.ownconsume(dict(authority=m.AUTHORITY,output='/MOCK/run',report_sha256='c'*64))


def test_final_real_scheduler_modelplan_canonical_and_whole_run_exclusion():
    # Import pure source, no producer material/consume/run or actual task data.
    module=m.importlib.import_module('experiments.spectral_reachability.'+m.AUTHORITY)
    assert m.sha(module.__file__)==m.AUTHORITY_SHA
    for path,h in m.authority_files().items():assert m.sha(path)==h
    slot=dict(source_slot=0,root_sha256='a'*64,task_id='b'*64,blind_sha256='c'*64,
        task_export='/RAM/task_0',task_sha256='d'*64,task_contract_sha256='e'*64,bank_sha256='f'*64,anchors=[])
    winner=dict(commands=np.zeros((10,2)).tolist(),index=0,cost=1.25)
    plan=module.model_plan(slot,m.ARMS[0],winner)
    content={k:v for k,v in plan.items() if k!='selection_sha256'}
    assert m.source.digest(content)==module.core.digest(content)==plan['selection_sha256']
    receipt=dict(plan,whole_run_report_sha256='9'*64)
    unsealed={k:v for k,v in receipt.items() if k not in ('selection_sha256','whole_run_report_sha256')}
    assert m.source.digest(unsealed)==receipt['selection_sha256']
    assert m.source.digest({k:v for k,v in receipt.items() if k!='selection_sha256'})!=receipt['selection_sha256']
    assert not any(x in sys.modules for x in ('torch','gymnasium','stable_worldmodel','pymunk'))
