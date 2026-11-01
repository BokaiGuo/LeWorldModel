"""RAM/fake-env only, not actual physics or model evidence."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from experiments.spectral_reachability import chat_p3_20261027_physical as m
from test_p1_constructive_moving_history_physical_20261027_v1 import ArrayEnv,fixture as moving_fixture,REQUEST
from test_p1_constructive_plan_physical_20261027_v1 import MemoryStore


def fixture():
    _,data=moving_fixture();plans=[]
    for s in (0,1):
        for a in m.ARMS:
            p=dict(root_slot=s,arm=a,root_sha256=data['references'][s]['root_sha256'],commands=np.full((10,2),.25,np.float32).tolist())
            p['selection_digest']=m.kernel.source.digest(p);plans.append(p)
    return dict(plans=plans,contract=m.contract(),bindings={},root_goals=[r['goal'].tolist() for r in data['references']]),data


def execute(factory=ArrayEnv):
    p,data=fixture();store=MemoryStore();envs=[]
    def make():e=factory();envs.append(e);return e
    def fake_sha(path):
        assert Path(path).name=='journal.jsonl'
        return m.kernel.source.digest(store.events)
    with patch.object(m.kernel.moving,'validate_env',lambda _:None),patch.object(m.kernel.source,'sample_state',lambda *a:REQUEST.copy()), \
         patch.object(m,'sha',fake_sha),patch.object(m.helper,'execute',side_effect=AssertionError('old runner forbidden')):
        r=m.execute(p,data,store,make,m.AUTH,dict(prepared='/RAM/prepare',prepare_sha256='a'*64))
    return p,r,store,envs


@pytest.fixture(scope='module')
def full():return execute()


def validate(p,r,store):
    def frame_reader(name,expected):
        a=store.files[name]
        assert m.kernel.moving.image_sha(a)==expected
        return a.copy()
    return m.validate_saved(p,r,store.events,frame_reader)


def test_full8_native_queue_and_saved_consumer(full):
    p,r,store,envs=full
    assert r['status']=='rank_physical_recorded_not_admitted',r
    receipt=validate(p,r,store)
    assert receipt['native_full10_pair_available'] and receipt['full_horizons']==8 and len(receipt['filenames'])==176
    assert len(envs)==8 and all(e.closed and len(e.commands)==20 for e in envs)
    counts=dict(environment_construct=8,environment_close=8,reset=16,set_state=32,goal_render=16,goal_bind=8,public_step=160,physics_step=1632,explicit_render=176)
    assert receipt['counts']=={k:dict(attempted=v,completed=v) for k,v in counts.items()}
    assert all(v==[0.,0.] for metrics in receipt['D_minus_control'].values() for v in metrics.values())


@pytest.mark.parametrize('fault',['duplicate','command','metric','goal','count','image','checkpoint'])
def test_saved_invalid_receipts_fail(full,fault):
    p,r,store,_=full;bad=deepcopy(r)
    if fault=='duplicate':bad['rows'][1]=deepcopy(bad['rows'][0])
    if fault=='command':bad['rows'][0]['selected_plan']['commands'][0][0]+=1
    if fault=='metric':bad['rows'][0]['final_metrics']['native_distance']+=1
    if fault=='goal':bad['rows'][0]['goal'][0]+=1
    if fault=='count':bad['counts']['public_step']['completed']-=1
    if fault=='image':bad['rows'][0]['PNG']-=1
    if fault=='checkpoint':bad['rows'][0]['checkpoints'].pop()
    with pytest.raises(ValueError):validate(p,bad,store)


@pytest.mark.parametrize('fault',['truncate','terminal','guard','substitute','throw'])
def test_partial_denominator_no_retries_or_padding(fault):
    p,r,store,envs=execute(lambda:ArrayEnv(fault=fault))
    assert len(r['rows'])==8 and r['full_horizons']==0 and all(e.closed and len(e.commands)==11 for e in envs)
    assert all(not any(c['step'] in (5,10) for c in row['checkpoints']) for row in r['rows'])
    if fault in ('truncate','terminal'):
        receipt=validate(p,r,store)
        assert receipt['D_minus_control'] is None and not receipt['native_full10_pair_available']
    else:
        assert r['status']=='partial_or_failed'
        with pytest.raises(ValueError):validate(p,r,store)


def test_constructor_failure_retains_unknown_work():
    def broken():raise RuntimeError('mock constructor failure')
    _,r,_,envs=execute(broken)
    assert not envs and len(r['rows'])==8 and r['total_internal_physics_scope']=='unknown_constructor_work'
    assert r['counts']['environment_construct']==dict(attempted=8,completed=0)


def test_authorization_precedes_factory():
    p,data=fixture()
    with pytest.raises(ValueError):m.execute(p,data,MemoryStore(),lambda:pytest.fail('factory called'),None,{})
