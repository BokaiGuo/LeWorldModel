"""In-memory fake transitions only, no neural/simulator/temporary files."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from experiments.spectral_reachability import agent_experiment1_common_controller_physical_20261028_v1 as m
from test_p1_constructive_moving_history_physical_20261027_v1 import ArrayEnv,fixture as moving_fixture,REQUEST
from test_p1_constructive_plan_physical_20261027_v1 import MemoryStore


def fixture():
    _,data=moving_fixture();plans=[]
    for s in (0,1):
        for a in m.ARMS:
            p=dict(root_slot=s,root_sha256=data['references'][s]['root_sha256'],arm=a,
                commands=np.full((10,2),.25 if a==m.ARMS[0] else -.25,np.float32).tolist(),
                target=np.full(192,.5,np.float32).tolist(),predicted_contexts=[{'RAM':'prefix'},{'RAM':'terminal'}],
                model_terminal_cost=2.,prefix_target_cost=1.,tracking_cost=1.5)
            p['selection_digest']=m.source.digest(p);plans.append(p)
    return dict(plans=plans,contract=m.contract(),bindings={},sourcepack_sha256=m.SOURCE_SHA,source_prepare_id='b'*64),data


def execute(factory=ArrayEnv,p=None,data=None,store=None):
    fp,fd=fixture();p=fp if p is None else p;data=fd if data is None else data
    store=MemoryStore() if store is None else store;envs=[]
    def make():e=factory();envs.append(e);return e
    def digest(path):
        assert Path(path).name=='journal.jsonl';return m.source.digest(store.events)
    with patch.object(m.kernel.moving,'validate_env',lambda _:None),patch.object(m.source,'sample_state',lambda *a:REQUEST.copy()),\
        patch.object(m,'sha',digest),patch.object(m.primitives,'execute',side_effect=AssertionError('old execute forbidden')),\
        patch.object(m.kernel.moving,'episode',side_effect=AssertionError('witness episode forbidden')):
        r=m.execute(p,data,store,make,authorization=m.AUTH,execution_binding=dict(prepared='/RAM/prepare',prepare_sha256='a'*64))
    return r,store,envs


def test_full_four_tracking_six_slots_native7():
    r,store,envs=execute();assert r['status']=='common_controller_physical_calibration_recorded',r['errors']
    expected=dict(reset=8,set_state=16,goal_render=8,goal_bind=4,public_step=80,physics_step=816,
        explicit_render=88,environment_construct=4,environment_close=4)
    assert r['counts']=={k:dict(attempted=v,completed=v) for k,v in expected.items()}
    assert len(r['rows'])==r['full_horizons']==r['fixed_branch_denominator']==4
    assert len(r['scientific_rows'])==r['scientific_slot_denominator']==6
    assert r['learned_unknown_slots']==2 and r['new_neural_calls']==0 and not r['P1_admission']
    for row,env in zip(r['rows'],envs,strict=True):
        assert row['initial_exact'] and env.closed and row['PNG']==22
        assert m.kernel.moving.exact(np.asarray(env.commands[:10]),m.kernel.moving.warmup_commands())
        assert m.kernel.moving.exact(np.asarray(env.commands[10:]),np.asarray(row['selected_plan']['commands'],np.float32))
        assert [c['step'] for c in row['checkpoints']]==[0,5,10]
        assert [c['step'] for c in row['tracking']]==list(range(11))
        for cp in row['tracking']:
            assert cp['png'] in store.files and len(cp['state'])==7
            assert cp['native']==m.kernel.metric(row['goal'],cp['state'])
            assert cp['observed_latent192'] is cp['latent_tracking_error'] is None
            assert cp['model_successor']==(row['selected_plan']['predicted_contexts'][cp['step']//5-1] if cp['step'] in (5,10) else None)
        assert np.array_equal(env.goal_pose,[300,300,0])
    unknown=[x for x in r['scientific_rows'] if x['arm']==m.LEARNED]
    assert len(unknown)==2 and all(x['status']=='unknown' and x['commands'] is x['resources'] is x['native_success'] is None for x in unknown)


@pytest.mark.parametrize('fault',['terminal','truncate','guard','render','substitute','throw'])
def test_stop_and_failure_denominators(fault):
    r,s,envs=execute(lambda:ArrayEnv(fault=fault))
    assert len(r['rows'])==4 and len(r['scientific_rows'])==6 and all(e.closed and len(e.commands)==11 for e in envs)
    assert r['full_horizons']==0
    assert r['status']==('common_controller_physical_calibration_recorded' if fault in ('terminal','truncate') else 'partial_or_failed')
    if fault=='throw':
        assert r['counts']['public_step']==dict(attempted=44,completed=40)
        # This is a last available frame, not a guaranteed post-fault scene.
        assert all(row['last_observation']['step']==0 for row in r['rows'])
    if fault in ('guard','render','substitute'):
        assert all(row['last_observation']['png'] in s.files for row in r['rows'])
    if fault=='guard':
        for i in range(4):
            e=[e for e in s.events if e.get('branch')==i]
            obs=next(x for x in reversed(e) if x['kind']=='fresh_observation')
            guard=next(x for x in reversed(e) if x['kind']=='guard_receipt')
            assert obs['index']<guard['index'] and guard['accepted'] is False


@pytest.mark.parametrize('field',['raw','RGB','state','snapshot','goal'])
def test_identity_rejection_before_own_commands(field):
    p,d=fixture();ref=d['references'][0]
    if field=='raw':ref['raw_history']=np.zeros((10,2),np.float32)
    elif field=='RGB':ref['history'][1,0,0,0]^=1
    elif field=='state':ref['states'][1,0]+=1e-8
    elif field=='snapshot':ref['dynamics'][1]['state_float64_bits']='bad'
    else:ref['goal'][0]+=1
    r,_,_=execute(p=p,data=d)
    assert all(row['own_attempted']==0 for row in r['rows'][:2])
    assert all(row['full_horizon'] for row in r['rows'][2:]) and len(r['scientific_rows'])==6


@pytest.mark.parametrize('change',['missing','arm','digest','root','target'])
def test_no_reselection_or_subset(change):
    p,d=fixture()
    if change=='missing':p['plans'].pop()
    elif change=='arm':p['plans'][0]['arm']='GN'
    elif change=='root':p['plans'][0]['root_sha256']='a'*64
    elif change=='target':p['plans'][0]['target'][0]=1.
    else:p['plans'][0]['commands'][0][0]=0.
    with pytest.raises(ValueError):m.execute(p,d,MemoryStore(),ArrayEnv,authorization=m.AUTH)


def test_no_authority_no_factory():
    p,d=fixture();calls=[]
    with pytest.raises(ValueError):m.execute(p,d,MemoryStore(),lambda:calls.append(1),authorization='wrong')
    assert not calls


def test_constructor_close_fault_and_partial():
    def fail():raise RuntimeError('constructor fault')
    r,_,_=execute(fail)
    assert len(r['rows'])==4 and len(r['scientific_rows'])==6 and r['total_internal_physics_scope']=='unknown_constructor_work'
    assert r['counts']['environment_construct']==dict(attempted=4,completed=0)
    r,_,_=execute(lambda:ArrayEnv(fault='throw',close_error=True))
    assert all('step' in str(row['first_error']) and row['secondary_errors'] for row in r['rows'])


def test_journal_chain_and_report_write_failure():
    r,s,_=execute();previous=None
    for e in s.events:
        assert e['previous_sha256']==previous and e['event_sha256']==m.source.digest({k:v for k,v in e.items() if k!='event_sha256'})
        previous=e['event_sha256']
    assert previous==r['journal_chain_tail']
    s=MemoryStore();put=s.put
    def broken(name,value,final=False):
        if name=='report.json':raise OSError('report fault')
        return put(name,value,final)
    s.put=broken;r,_,_=execute(store=s)
    assert r['status']=='partial_or_failed' and r['errors'][-1]['stage']=='final_report'


def test_selection_explicit_future_authority(monkeypatch):
    _,data=fixture();rows=[]
    for s in (0,1):
        for a in m.ARMS:
            rows.append(dict(source_slot=s,arm=a,commands=np.full((10,2),.25,np.float32).tolist(),
                task_id='id'+str(s),blind_sha256=str(s)*64,task_export='/RAM/task'+str(s),task_sha256='e'*64,
                task_contract_sha256='b'*64,target=[.5]*192,contexts=[{},{}],prefix_target_cost=1.,model_terminal_cost=2.,
                tracking_cost=1.5,selection_sha256='a'*64))
    monkeypatch.setattr(m,'ownconsume',lambda _: (rows,{}))
    monkeypatch.setattr(m.primitives.helper,'task_bridge',lambda *a:None)
    manifest=dict(schema=m.NAME+'.selection',sourcepack='/RAM/source',sourcepack_sha256=m.SOURCE_SHA,
        source_prepared='/RAM/prepare',source_prepare_sha256='a'*64,
        roots=[dict(root_slot=i,root_sha256=r['root_sha256']) for i,r in enumerate(data['references'])],arms=list(m.ARMS),
        authority=dict(output='/RAM/compiler',report_sha256='c'*64),selection_fixed_before_physics=True,physical_outcome_selection=False,wall_seconds=600)
    plans,b=m.selection(manifest,data);assert len(plans)==4
    assert all(p['selection_digest']==m.source.digest({k:v for k,v in p.items() if k!='selection_digest'}) for p in plans)
    manifest['arms']=[m.ARMS[0]]
    with pytest.raises(ValueError):m.selection(manifest,data)


def test_native7_velocity_and_native_wrapped_discrepancy():
    goal=np.zeros(7);state=goal.copy();state[-1]=10
    metric=m.kernel.metric(goal,state)
    assert metric['native_distance']==10 and metric['native_success']
    state=np.zeros(7);state[4]=3*np.pi/1.2
    metric=m.kernel.metric(goal,state)
    assert metric['negative_native_angle'] and metric['native_success'] and not metric['wrapped_allxy_success']


def test_producer_seals_no_future_report_constant():
    files=(Path(m.producer.__file__),m.ROOT/'tests'/('test_'+m.producer.NAME+'.py'),m.ROOT/'innerdocs'/(m.producer.NAME+'.md'))
    assert [m.sha(f) for f in files]==list(m.PRODUCER_SEALS)
    c=m.contract();assert c['caps']['public_step']==80 and c['caps']['physics_step']==816
    assert c['new_neural_calls']==0 and c['scientific_slots']==6 and c['learned_unknown_slots']==2
    assert m.sha(m.primitives.__file__)==m.PRIMITIVES_SHA


@pytest.mark.parametrize('field',['authority','report','prepare','drift'])
def test_ownconsume_external_authority_failclosed(monkeypatch,field):
    entry=dict(authority=m.producer.NAME,output='/RAM/compiler',report_sha256=m.REPORT_SHA,prepare_sha256=m.CUDA_PREPARE_SHA,
        strict_receipt='/RAM/strict.json',strict_sha256=m.STRICT_SHA)
    calls=[]
    def hash_file(path):
        path=str(path)
        if path.endswith('/report.json'):return 'c'*64
        for suffix,h in zip((m.producer.NAME+'.py','test_'+m.producer.NAME+'.py',m.producer.NAME+'.md'),m.PRODUCER_SEALS):
            if Path(path).name==suffix:return h
        return '0'*64
    monkeypatch.setattr(m,'sha',hash_file)
    monkeypatch.setattr(m.producer,'consume',lambda *a:calls.append(1))
    if field=='authority':entry['authority']='old_sampler'
    elif field=='report':entry['report_sha256']='a'*64
    elif field=='prepare':entry['prepare_sha256']='pending'
    elif field=='drift':monkeypatch.setattr(m,'sha',lambda path:'0'*64)
    with pytest.raises(ValueError):m.ownconsume(entry)
    assert not calls


def test_material_external_manifest_sha_before_source(monkeypatch):
    monkeypatch.setattr(m,'sha',lambda path:'a'*64)
    calls=[];monkeypatch.setattr(m.kernel,'load_source',lambda *a:calls.append(1))
    # Real existing, bounded source file is only used as a read-only path fixture.
    with pytest.raises(ValueError,match='manifest SHA'):m.material(str(Path(m.__file__).resolve()),'b'*64)
    assert not calls
