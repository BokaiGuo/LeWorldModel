"""RAM saved-data verification; no model/simulator/temp output."""
from copy import deepcopy
import numpy as np
import pytest
from experiments.spectral_reachability import agent_experiment1_common_controller_physical_consume_20261028_v1 as m
from test_agent_experiment1_common_controller_physical_20261028_v1 import execute,fixture
from test_p1_constructive_moving_history_physical_20261027_v1 import ArrayEnv


@pytest.fixture(scope='module')
def full():
    p,_=fixture();r,s,_=execute(p=p);return p,r,s.events


def test_complete_saved_full_four_and_six(full):
    p,r,e=full;cp=m.assemble(r,e,p)
    assert len(cp)==12 and len(r['scientific_rows'])==6
    assert [(c['root_slot'],c['arm'],c['checkpoint']['step']) for c in cp]==[(s,a,k) for s in (0,1) for a in m.physical.ARMS for k in (0,5,10)]
    assert all(len(c['checkpoint']['state'])==7 and len(c['target'])==192 for c in cp)


@pytest.mark.parametrize('fault',['terminal','truncate','guard','render','substitute','throw'])
def test_actual_earlystop_or_partial_not_dropped(fault):
    p,_=fixture();r,s,_=execute(lambda:ArrayEnv(fault=fault),p=p)
    cp=m.assemble(r,s.events,p)
    assert len(cp)==4 and len(r['scientific_rows'])==6
    assert all(c['checkpoint']['step']==0 for c in cp)
    if fault=='render':assert r['status']=='partial_or_failed'


@pytest.mark.parametrize('change',['missingrow','unknown','denominator','nativevelocity','reward','command','target','predicted',
    'tracking','latentfill','counter','receipt','missingevent','checkpoint','guard','selection','terminal','order','extraoperation'])
def test_corruption_rejected(full,change):
    p,r,e=map(deepcopy,full)
    if change=='missingrow':r['rows'].pop()
    elif change=='unknown':r['scientific_rows'][1]['resources']={}
    elif change=='denominator':r['scientific_slot_denominator']=4
    elif change=='nativevelocity':r['rows'][0]['final_state'][-1]+=1
    elif change=='reward':next(x for x in e if x['kind']=='returned_transition' and x['phase']=='plan')['reward']+=1
    elif change=='command':next(x for x in e if x['kind']=='returned_transition' and x['phase']=='plan')['command'][0]+=.1
    elif change=='target':p['plans'][0]['target'][0]+=1
    elif change=='predicted':p['plans'][0]['predicted_contexts'][0]={'fake':'context'}
    elif change=='tracking':r['rows'][0]['tracking'][0]['native']['native_distance']+=1
    elif change=='latentfill':next(x for x in e if x['kind']=='tracking_observation')['receipt']['observed_latent192']=[0.]*192
    elif change=='counter':r['counts']['physics_step']['completed']-=1
    elif change=='receipt':next(x for x in e if x['kind']=='operation_returned')['receipt']['root_slot']=1
    elif change=='missingevent':e.pop(next(i for i,x in enumerate(e) if x['kind']=='operation_returned'))
    elif change=='checkpoint':r['rows'][0]['checkpoints'].pop()
    elif change=='guard':next(x for x in e if x['kind']=='guard_receipt' and x['phase']=='plan')['accepted']=False
    elif change=='selection':r['rows'][0]['selection_digest']='a'*64
    elif change=='terminal':r['rows'][0]['terminated']=True
    elif change=='order':r['rows'][0],r['rows'][1]=r['rows'][1],r['rows'][0]
    else:e.insert(-1,deepcopy(next(x for x in e if x['kind']=='operation_dispatch_intent')))
    with pytest.raises((ValueError,IndexError)):m.assemble(r,e,p)


def test_native7_not_five_and_velocity_matters():
    goal=np.zeros(7);state=goal.copy();state[-2:]=[3.,4.]
    assert m.metric(goal,state)['native_distance']==5
    assert m.metric(goal,state)['native_success']
    with pytest.raises(ValueError):m.metric(goal[:5],state)
    state=np.zeros(7);state[4]=2.5*np.pi
    native=m.metric(goal,state)
    assert native['negative_native_angle'] and native['native_success'] and not native['wrapped_allxy_success']


def test_no_science_upgrade(full):
    p,r,e=map(deepcopy,full);r['P1_admission']=True
    with pytest.raises(ValueError):m.assemble(r,e,p)


def test_incomplete_journal_rejects_even_if_partial(full):
    p,r,e=map(deepcopy,full);r['status']='partial_or_failed'
    e.pop();e[-1]['kind']='run_finished';e[-1]['status']=r['status'];e[-1]['counts']=r['counts'];e[-1]['errors']=r['errors']
    # Removing a real operation receipt must not be guessed from final counters.
    e.pop(next(i for i,x in enumerate(e) if x['kind']=='operation_returned'))
    with pytest.raises(ValueError):m.assemble(r,e,p)


def test_sealed_physical_three_files():
    from pathlib import Path
    files=(Path(m.physical.__file__),m.ROOT/'tests'/('test_'+m.physical.NAME+'.py'),m.ROOT/'innerdocs'/(m.physical.NAME+'.md'))
    assert [m.sha(f) for f in files]==list(m.SEALS)
