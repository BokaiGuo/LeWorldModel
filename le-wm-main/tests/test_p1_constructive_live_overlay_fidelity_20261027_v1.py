"""RAM-only replay/pair contracts. No environment, model, prepare or tmp files."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.spectral_reachability import p1_constructive_live_overlay_fidelity_20261027_v1 as m


def fixture():
    p=dict(contract=m.physical.contract(4,600),plans=[],source_slots=[dict(root_sha256=str(i)) for i in range(2)])
    report=dict(schema=m.physical.NAME+'.report',status='physical_calibration_recorded',errors=[],contract=p['contract'],rows=[],
        counts={k:dict(attempted=v,completed=v) for k,v in p['contract']['caps'].items()})
    ev=[dict(kind='run_started',prepare=p)]
    for resource,n in p['contract']['caps'].items():
        for _ in range(n):ev.extend([dict(kind='operation_attempted',resource=resource),dict(kind='operation_completed',resource=resource)])
    for b,(slot,arm) in enumerate((i,a) for i in range(2) for a in m.base.ARMS):
        plan=dict(root_slot=slot,arm=arm,commands=np.zeros((10,2),np.float32).tolist())
        plan['selection_digest']=m.physical.source.digest(plan);p['plans'].append(plan)
        row=dict(root_slot=slot,arm=arm,full_horizon=True,own_returned=10,own_attempted=10,warmup_returned=10,
            first_error=None,secondary_errors=[],PNG=22,goal=[0.]*7,selection_digest=plan['selection_digest'])
        report['rows'].append(row)
        ev.extend([dict(kind='branch_started',branch=b,plan=plan),
            dict(kind='operation_attempted',resource='reset',branch=b,kind_of_reset='original_seed',seed=100+slot),
            dict(kind='operation_attempted',resource='reset',branch=b,kind_of_reset='placement',seed=0),
            dict(kind='placement_request',branch=b,original_goal=[0.]*7,requested=[1.]*7),
            dict(kind='goal_attached',branch=b,goal=[0.]*7,goal_rgb_sha256='goal')])
        # Reset identities above need balanced resources and belong to original census.
        for phase,s in m.ALL:
            c=dict(phase=phase,step=s,png=f'b{b}_{phase}_{s:02}.png',file_sha256='file',pixel_sha256='pixel',
                state=[float(s)]*7,snapshot=dict(x=s),observation_state=[float(s)]*7)
            ev.extend([dict(kind='fresh_observation',branch=b,receipt=c),
                dict(kind='guard_receipt',branch=b,accepted=True,render_preserved_dynamics=True)])
            if s:
                e=dict(kind='returned_transition',branch=b,phase=phase,step=s,state=c['state'],reward=0.,terminated=False,truncated=False)
                if phase=='plan':e.update(command=plan['commands'][s-1],returned_goal_state=[0.]*7,returned_goal_rgb_sha256='goal')
                ev.append(e)
    # Move 8 generic reset receipts to branch-specific identity receipts (no extra calls).
    ev=[e for e in ev if not(e['kind'] in ('operation_attempted','operation_completed') and e.get('resource')=='reset' and 'branch' not in e)]
    expanded=[]
    for e in ev:
        expanded.append(e)
        if e['kind']=='operation_attempted' and e.get('resource')=='reset':expanded.append(dict(kind='operation_completed',resource='reset',branch=e['branch']))
    ev=expanded+[dict(kind='run_finished',status=report['status'],counts=report['counts'])]
    return report,ev,p


def test_original_full_denominator_and_commands():
    r,e,p=fixture();items=m.assemble(r,e,p)
    assert len(items)==4 and sum(len(x['receipts']) for x in items)==88
    assert [(x['source_slot'],x['arm']) for x in items]==[(i,a) for i in range(2) for a in m.base.ARMS]
    assert all(x['placement_seed']==0 for x in items)


@pytest.mark.parametrize('fault',['missing_off','reorder','command','short','guard','reset','resource','digest','goal'])
def test_original_corruption_rejected(fault):
    r,e,p=fixture()
    if fault=='missing_off':e.pop(next(i for i,x in enumerate(e) if x['kind']=='fresh_observation'))
    elif fault=='reorder':p['plans'].reverse()
    elif fault=='command':next(x for x in e if x['kind']=='returned_transition' and x['phase']=='plan')['command']=[1.,1.]
    elif fault=='short':r['rows'][0]['own_returned']=9
    elif fault=='guard':next(x for x in e if x['kind']=='guard_receipt')['accepted']=False
    elif fault=='reset':next(x for x in e if x.get('kind_of_reset')=='placement')['seed']=1
    elif fault=='resource':e.pop(next(i for i,x in enumerate(e) if x['kind']=='operation_completed'))
    elif fault=='digest':p['plans'][0]['selection_digest']='bad'
    elif fault=='goal':next(x for x in e if x['kind']=='returned_transition' and x['phase']=='plan')['returned_goal_state']=[9.]*7
    with pytest.raises((ValueError,KeyError)):m.assemble(r,e,p)


class Sink:
    def __init__(self):self.events=[]
    def check(self,*a):return None
    def emit(self,k,**kw):self.events.append(dict(kind=k,**kw))


def pair_fixture():
    body=SimpleNamespace(with_target=False,state=0)
    env=SimpleNamespace(unwrapped=body,render=lambda:np.full((2,2,3),int(body.with_target),np.uint8))
    sink=Sink();ops=m.Ops(dict(contract=m.contract()),sink)
    return body,env,ops,sink


def test_pair_same_env_no_steps_and_restore():
    b,e,ops,sink=pair_fixture();frames=[]
    proof=m.pair(e,ops,lambda a:frames.append(a),lambda a:frames.append(a),snap=lambda b:b.state,check=lambda b:None)
    assert proof['snapshot_equal'] and b.with_target is False
    assert np.all(frames[0]==0) and np.all(frames[1]==1)
    assert ops.counts['explicit_render']==dict(attempted=2,completed=2)
    assert ops.counts['physics_step']==dict(attempted=0,completed=0)


@pytest.mark.parametrize('mutate_view',[False,True])
def test_pair_mutation_retained_and_no_retry(mutate_view):
    b,e,ops,sink=pair_fixture();frames=[]
    def render():
        if b.with_target==mutate_view:b.state+=1
        return np.zeros((2,2,3),np.uint8)
    e.render=render
    with pytest.raises(ValueError,match='mutated snapshot'):
        m.pair(e,ops,lambda a:frames.append(a),lambda a:frames.append(a),snap=lambda b:b.state,check=lambda b:None)
    assert len(frames)==(2 if mutate_view else 1) and b.with_target is False and not ops.in_pair


@pytest.mark.parametrize('operation',['physics_step','public_step','set_state','goal_bind','reset'])
def test_pair_denies_physics_setter_goal_and_step(operation):
    b,e,ops,sink=pair_fixture();executed=[]
    def render():return ops.call(operation,lambda:executed.append(True))
    e.render=render
    with pytest.raises(ValueError,match='NO reset'):m.pair(e,ops,lambda a:None,lambda a:None,snap=lambda b:b.state,check=lambda b:None)
    assert not executed and b.with_target is False


def test_pose_unproven_stops_before_render():
    b,e,ops,sink=pair_fixture()
    def reject(b):raise ValueError('pose unknown')
    with pytest.raises(ValueError,match='pose unknown'):m.pair(e,ops,lambda a:None,lambda a:None,snap=lambda b:b.state,check=reject)
    assert ops.counts['explicit_render']['attempted']==0


def test_actual_native_default_contract_not_custom_goal():
    v=lambda a:SimpleNamespace(value=np.asarray(a))
    b=SimpleNamespace(goal_pose=np.asarray(m.DEFAULT_POSE),variation_space={'goal':{'position':v([256.,256.]),'angle':v(np.pi/4)},'rendering':{'render_goal':v(1)}})
    m.default_pose(b);b.goal_pose=np.zeros(3)
    with pytest.raises(ValueError):m.default_pose(b)


def test_snapshot_signed_zero_not_tolerance():
    assert m.numeric([0.])!=m.numeric([-0.])


def test_budget_internal_resets_and_extra_render():
    c=m.contract();assert c['caps']['physics_step']==2*8+10*80
    assert c['caps']['explicit_render']==88+16
    assert c['selected_images']==32 and c['selected_pairs']==16 and c['P1_admission'] is False


def test_no_privileged_loader_calls_in_new_source():
    import ast
    tree=ast.parse(m.Path(m.__file__).read_text())
    called={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
    assert not {'load_source','material','witness','sample_state','read_prepared'} & called


@pytest.mark.parametrize('bad_off',[None,3])
def test_complete_replay_kernel_or_first_off_mismatch_in_ram(monkeypatch,bad_off):
    report,events,p=fixture();item=m.assemble(report,events,p)[0];item['goal_rgb_file']='IN_MEMORY_ONLY'
    for c in item['receipts']:
        t=c['step']+(10 if c['phase']=='plan' else 0)
        c.update(state=[float(t)]*7,observation_state=[float(t)]*7,snapshot=dict(x=t))
    for e in item['transitions']:
        t=e['step']+(10 if e['phase']=='plan' else 0);e['state']=[float(t)]*7
    body=SimpleNamespace(t=0,goal_pose=np.asarray(m.DEFAULT_POSE),with_target=False,goal_state=np.zeros(7),_goal=np.zeros((224,224,3),np.uint8))
    body._get_obs=lambda:np.full(7,body.t,dtype=float)
    class Env:
        unwrapped=body
        def reset(self,seed=None,options=None):body.t=0;return dict(state=body._get_obs()),dict(goal_state=np.zeros(7))
        def step(self,command):
            body.t+=1
            return dict(state=body._get_obs()),0.,False,False,dict(goal_state=body.goal_state,goal=body._goal)
        def render(self):return body._goal.copy()
    class Archive:
        def __enter__(self):return dict(goal_rgb=body._goal.copy())
        def __exit__(self,*a):return False
    class Store(Sink):
        def __init__(self):super().__init__();self.frames=[]
        def frame(self,name,a):
            self.frames.append(name)
            return 'wrong' if bad_off is not None and name==f'b0_warmup_{bad_off:02}.png' else 'file'
    def attach(b,goal,image):b.goal_state=goal.copy();b._goal=image.copy()
    monkeypatch.setattr(m,'default_pose',lambda b:None)
    monkeypatch.setattr(m,'snapshot',lambda b:dict(bodies=[b.t]))
    monkeypatch.setattr(m.physical.builder,'snapshot',lambda b:dict(x=b.t))
    monkeypatch.setattr(m.physical.builder,'image_sha',lambda a:'pixel')
    monkeypatch.setattr(m.physical.builder,'attach_goal',attach)
    monkeypatch.setattr(m.physical.source,'root_hash',lambda a,b:item['root_sha256'])
    monkeypatch.setattr(m.np,'load',lambda *a,**kw:Archive())
    item['goal_rgb_sha256']='pixel'
    store=Store();ops=m.Ops(dict(contract=m.contract()),store)
    row=dict(pairs=[],off_exact=0,public_returned=0)
    if bad_off is None:
        m.branch(Env(),item,row,ops,store)
        assert row['status']=='replayed_and_paired' and row['off_exact']==22 and len(row['pairs'])==4
        assert len(store.frames)==26 and row['public_returned']==20
        assert ops.counts['explicit_render']==dict(attempted=26,completed=26)
    else:
        with pytest.raises(ValueError,match='off full trajectory'):m.branch(Env(),item,row,ops,store)
        assert row['public_returned']==bad_off and len(store.frames)==bad_off+1
        assert row['last_observation']['file_sha256']=='wrong' and row['off_exact']==bad_off
