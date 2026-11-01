"""RAM callback dynamics only, no model/simulator/prepare/temp output."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from experiments.spectral_reachability import agent_kinodynamic_matched_physical_20261028_v1 as m
from test_p1_constructive_moving_history_physical_20261027_v1 import ArrayEnv,fixture as moving_fixture,REQUEST
from test_p1_constructive_plan_physical_20261027_v1 import MemoryStore


def fixture():
    _,data=moving_fixture();plans=[]
    for slot in (0,1):
        b=m.producer.bank(slot);p=dict(root_slot=slot,root_sha256=data['references'][slot]['root_sha256'],arm=m.ARM,
            commands=b['commands'][1]+b['commands'][2],compiler_authority=m.producer.NAME)
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
        patch.object(m.kernel.moving,'episode',side_effect=AssertionError('source witness execution forbidden')):
        r=m.execute(p,data,store,make,authorization=m.AUTH,execution_binding=dict(prepared='/RAM/prepare.json',prepare_sha256='a'*64))
    return r,store,envs


def test_two_new_arms_full_accounting_and_real_nonzero_fixture():
    r,s,envs=execute();assert r['status']=='matched_domain_beam_physical_recorded_not_admitted',r['errors']
    expected=dict(reset=4,set_state=8,goal_render=4,goal_bind=2,public_step=40,physics_step=408,
        explicit_render=44,environment_construct=2,environment_close=2)
    assert r['counts']=={k:dict(attempted=v,completed=v) for k,v in expected.items()}
    assert len(r['rows'])==r['full_horizons']==r['fixed_branch_denominator']==2
    for row,env in zip(r['rows'],envs,strict=True):
        assert row['arm']==m.ARM and row['initial_exact'] and env.closed
        assert m.kernel.moving.exact(np.asarray(env.commands[:10]),m.kernel.moving.warmup_commands())
        assert m.kernel.moving.exact(np.asarray(env.commands[10:]),np.asarray(row['selected_plan']['commands'],np.float32))
        assert row['PNG']==22 and [c['step'] for c in row['checkpoints']]==[0,5,10]
        assert len(row['final_state'])==7 and all(len(c['state'])==7 for c in row['checkpoints'])
        assert np.array_equal(env.goal_pose,[300,300,0])
    assert not r['P1_admission'] and not r['scientific_admission'] and r['new_neural_calls']==0


@pytest.mark.parametrize('fault',['terminal','truncate','guard','render','substitute','throw'])
def test_earlystop_and_engineering_keep_two(fault):
    r,s,envs=execute(lambda:ArrayEnv(fault=fault))
    assert len(r['rows'])==2 and all(e.closed and len(e.commands)==11 for e in envs)
    assert r['full_horizons']==0 and all([c['step'] for c in row['checkpoints']]==[0] for row in r['rows'])
    assert r['status']==('matched_domain_beam_physical_recorded_not_admitted' if fault in ('terminal','truncate') else 'partial_or_failed')
    if fault=='throw':assert r['counts']['public_step']==dict(attempted=22,completed=20)
    if fault in ('guard','render','substitute'):
        assert all(row['last_observation']['png'] in s.files for row in r['rows'])


@pytest.mark.parametrize('field',['raw','RGB','state','snapshot','goal'])
def test_reset_context_guard_before_selected(field):
    p,d=fixture();ref=d['references'][0]
    if field=='raw':ref['raw_history']=np.zeros((10,2),np.float32)
    elif field=='RGB':ref['history'][1,0,0,0]^=1
    elif field=='state':ref['states'][1,0]+=1e-8
    elif field=='snapshot':ref['dynamics'][1]['state_float64_bits']='bad'
    else:ref['goal'][0]+=1
    r,_,_=execute(p=p,data=d)
    assert r['rows'][0]['own_attempted']==0 and not r['rows'][0]['initial_exact']
    assert r['rows'][1]['full_horizon'] and len(r['rows'])==2


def test_close_fault_retains_primary():
    r,_,_=execute(lambda:ArrayEnv(fault='throw',close_error=True))
    assert r['status']=='partial_or_failed' and len(r['rows'])==2
    for row in r['rows']:
        assert 'step' in str(row['first_error']) and row['secondary_errors']


def test_construct_failure_no_fake_zero_work():
    def fail():raise RuntimeError('constructor fault')
    r,_,_=execute(fail)
    assert len(r['rows'])==2 and r['total_internal_physics_scope']=='unknown_constructor_work'
    assert r['counts']['environment_construct']==dict(attempted=2,completed=0)
    assert r['counts']['reset']==dict(attempted=0,completed=0)


def test_authorization_before_factory():
    p,d=fixture();calls=[]
    with pytest.raises(ValueError):m.execute(p,d,MemoryStore(),lambda:calls.append(1),authorization='wrong')
    assert calls==[]


@pytest.mark.parametrize('change',['missing','oldarm','digest','root'])
def test_no_old_arm_subset_or_unsealed_command(change):
    p,d=fixture()
    if change=='missing':p['plans'].pop()
    elif change=='oldarm':p['plans'][0]['arm']='CEM'
    elif change=='root':p['plans'][0]['root_sha256']='a'*64
    else:p['plans'][0]['commands'][0][0]=0.
    with pytest.raises(ValueError):m.execute(p,d,MemoryStore(),ArrayEnv,authorization=m.AUTH)


def test_journal_chain_and_caps():
    r,s,_=execute();previous=None
    for e in s.events:
        assert e['previous_sha256']==previous
        assert e['event_sha256']==m.source.digest({k:v for k,v in e.items() if k!='event_sha256'})
        previous=e['event_sha256']
    assert previous==r['journal_chain_tail']
    assert all(v['attempted']<=m.contract()['caps'][k] for k,v in r['counts'].items())


def test_final_report_failure_no_success_claim():
    store=MemoryStore();original=store.put
    def put(name,value,final=False):
        if name=='report.json':raise OSError('final write fault')
        return original(name,value,final)
    store.put=put;r,_,_=execute(store=store)
    assert r['status']=='partial_or_failed' and r['errors'][-1]['stage']=='final_report'


def test_contract_pins_and_no_equivalence_claim():
    c=m.contract();assert c['branches']==2 if 'branches' in c else c['caps']['public_step']==40
    assert c['inherited_prediction_budget']['predict_API']==288
    assert c['new_neural_calls']==0 and not c['P1_admission'] and not c['equal_FLOPs']
    assert m.sha(m.primitives.__file__)==m.PRIMITIVES_SHA and m.sha(m.producer.__file__)==m.PRODUCER_SHA


def test_selection_interface_pure_join(monkeypatch):
    _,d=fixture();rows=[dict(source_slot=s,arm=m.ARM,commands=m.producer.bank(s)['commands'][1]*2,
        task_id='id'+str(s),blind_sha256=str(s)*64,task_export='/RAM/task'+str(s),task_sha256='e'*64,
        selected_path=[1,1],bank_sha256='f'*64,selection_sha256='a'*64) for s in (0,1)]
    monkeypatch.setattr(m,'ownconsume',lambda _: (rows,{}));monkeypatch.setattr(m,'task_bridge',lambda *a:'b'*64)
    manifest=dict(schema=m.NAME+'.selection',sourcepack='/RAM/source',sourcepack_sha256=m.SOURCE_SHA,
        source_prepared='/RAM/prepare',source_prepare_sha256='a'*64,
        roots=[dict(root_slot=i,root_sha256=r['root_sha256']) for i,r in enumerate(d['references'])],arms=[m.ARM],
        authorities=[dict(output='/RAM/compiler')],selection_fixed_before_physics=True,physical_outcome_selection=False,wall_seconds=600)
    plans,b=m.selection(manifest,d);assert len(plans)==2
    for p in plans:assert p['selection_digest']==m.source.digest({k:v for k,v in p.items() if k!='selection_digest'})
    manifest['arms']=['CEM']
    with pytest.raises(ValueError):m.selection(manifest,d)
