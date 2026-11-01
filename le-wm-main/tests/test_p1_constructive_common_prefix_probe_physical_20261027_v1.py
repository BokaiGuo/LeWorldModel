"""Pure saved/NumPy/RAM simulator mocks; never instantiate native physics/GPU."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.spectral_reachability import p1_constructive_common_prefix_probe_physical_20261027_v1 as m


def task():
    t=dict(environment='pusht',task_contract_sha256='a'*64,root_context=dict(layout='states_with_between_state_actions_v1',
        context_length=3,latent_dim=192,action_dim=10,latent_history=np.zeros((3,192)).tolist(),action_history=np.zeros((2,10)).tolist()),
        root_raw_action_history=np.zeros((10,2)).tolist(),goal=np.ones(192).tolist(),projection=np.eye(192).tolist(),
        model_sha256='b'*64,model_config_sha256='c'*64,metric_source_sha256='d'*64,goal_source_sha256='e'*64,
        representation='native_projected_cls',latent_dim=192,horizon=2,packet_steps=5,control_dim=2,
        action_mean=[.1,-.2],action_std=[.7,1.3],packet_logit_cap=m.LOGIT_CAP,feedback='open_loop_model_only')
    t['root_context']['action_history']=m.core.normalized_tokens(t,np.zeros((10,2))).tolist()
    return t


def candidates(zero_winner=False):
    cs=[]; params=[]; kinds=[]
    for i in range(10):
        x=np.zeros(10) if i==0 else np.full(10,i*.02); u=np.tanh(x.astype(np.float32)).reshape(5,2)
        target=0 if zero_winner else 1; z=np.ones(192)*(1 if i==target else i+3)
        cost=float(np.mean((z-1)**2)); kind='candidate_'+str(i); params.append(x.tolist()); kinds.append(kind)
        cs.append(dict(index=i,completed=True,independent_model_replay_exact=True,prefix_params=x.tolist(),proposal_kind=kind,
            prefix_commands=u.tolist(),commands=np.concatenate((u,np.zeros((5,2)))).tolist(),predicted_prefix=z.tolist(),stage_cost=cost,
            own_reach_value=.1,objective=cost+.1))
    w=min(cs,key=lambda c:(c['objective'],c['index']))
    return dict(status='completed',bank=dict(params=params,kinds=kinds),candidates=cs,winner=deepcopy(w))


def records(allzero=False):
    out=[]
    for k in (0,1):
        root=np.array([100.+k,120.,200.,220.,.2,0.,0.]); goal=np.array([350.,360.,330.,340.,.7,0.,0.])
        roles=m.choose_roles(task(),candidates(allzero))
        out.append(dict(root_slot=k,root_sha256=m.source.root_hash(root,goal),goal=goal.tolist(),roles=roles,
            recipe_contract=dict(root_seed=k,with_target=False),action_history_contract=dict(zero10=True),direction=m.direction(root,goal)))
    return out


def protocol(allzero=False):
    rs=records(allzero); positions,aliases=m.deduplicate(rs); plans=[]
    for pos in positions:
        for probe in m.probes():
            commands=np.concatenate((pos['prefix_commands'],probe['commands']))
            plans.append(dict(index=len(plans),root_slot=pos['root_slot'],position_id=pos['position_id'],probe_index=probe['index'],
                arm='mock',commands=commands.tolist(),selection_digest=m.digest(dict(pos=pos['position_id'],probe=probe['index']))))
    return dict(roots=rs,positions=positions,role_aliases=aliases,probes=m.probes(),plans=plans,contract=m.contract(len(positions)),bindings={},inherited_costs={})


def test_probe21_order_ball_tanh_time_major_and_budget():
    b=m.probes(); assert len(b)==21 and b[0]['coordinate'] is None
    for j in range(10):
        for offset,sign in ((1,1),(2,-1)):
            r=b[2*j+offset]; x=np.asarray(r['logits'],np.float32)
            assert r['coordinate']==j and r['sign']==sign and x[j]==sign*.25 and np.count_nonzero(x)==1
            assert np.array_equal(r['commands'],np.tanh(x).reshape(5,2))
            assert np.linalg.norm(x)<=m.LOGIT_CAP
    c=m.contract(4); assert c['branches']==84 and c['caps']==dict(reset=168,public_step=1680,physics_step=17136,
        explicit_render=1848,goal_render=168,set_state=336,goal_bind=84)
    assert c['new_GPU']==c['new_AD']==c['new_predict']==0


def test_G_D_alias_and_all_three_zero_alias_not_extra_samples():
    for allzero,wanted in ((False,4),(True,2)):
        p=protocol(allzero); assert len(p['positions'])==wanted and len(p['role_aliases'])==6 and len(p['plans'])==21*wanted
        for k in (0,1):
            roles=[r for r in p['role_aliases'] if r['root_slot']==k]
            assert roles[1]['position_id']==roles[2]['position_id']
        assert p['contract']['independent_roots']==2


def test_no_crossroot_or_state_approximate_dedup_and_signed_zero_preserved():
    rs=records(True); positions,_=m.deduplicate(rs); assert len(positions)==2
    rs=records(True); rs[0]['roles'][0]['prefix_commands'][0][0]=-0.
    positions,_=m.deduplicate(rs); assert len(positions)==3
    assert m.bits([0.],'<f4')!=m.bits([-0.],'<f4')
    # History is included explicitly, not inferred from a similar endpoint.
    key=deepcopy(positions[0]['dedup_key']); other=deepcopy(key); other['action_history_contract']={'different':True}
    assert m.digest(key)!=m.digest(other)


@pytest.mark.parametrize('damage',('slot','duplicate_root','roles','different_G_D','candidate_count','cost','winner','tanh'))
def test_failclosed_role_inputs(damage):
    if damage in ('slot','duplicate_root','roles'):
        rs=records()
        if damage=='slot': rs[0],rs[1]=rs[1],rs[0]
        elif damage=='duplicate_root': rs[1]['root_sha256']=rs[0]['root_sha256']
        else: rs[0]['roles'].pop()
        with pytest.raises(ValueError): m.deduplicate(rs)
    else:
        r=candidates()
        if damage=='candidate_count': r['candidates'].pop()
        elif damage=='cost': r['candidates'][1]['stage_cost']+=1
        elif damage=='winner': r['winner']=r['candidates'][0]
        elif damage=='tanh': r['candidates'][0]['prefix_commands'][0][0]=.1
        else:
            r['candidates'][2]['own_reach_value']=-100; r['candidates'][2]['objective']=r['candidates'][2]['stage_cost']-100
            r['winner']=deepcopy(r['candidates'][2])
        with pytest.raises(ValueError): m.choose_roles(task(),r)


def completed_rows(p):
    out=[]
    for plan in p['plans']:
        s=np.array([100.,120.,200.,220.,.2,0.,0.]); probe=p['probes'][plan['probe_index']]
        if probe['coordinate'] is not None:
            s[probe['coordinate']%4]+=probe['sign']*.25*(plan['position_id']+1)
        out.append(dict(index=plan['index'],position_id=plan['position_id'],probe_index=plan['probe_index'],
            root_slot=plan['root_slot'],status='horizon_completed',prefix_exact=True,full_horizon=True,first_error=None,
            final_state=s.tolist(),prefix_checkpoint=dict(state=[99.,120.,200.,220.,.2,0.,0.])))
    return out


def test_common_direction_B_E_full_denominator_and_partial_columns():
    p=protocol(); rows=completed_rows(p); a=m.analyze(p,rows)
    assert len(a['positions'])==4 and all(len(v['responses'])==21 and len(v['columns'])==10 for v in a['positions'])
    col=a['positions'][0]['columns'][0]; assert col['B']==[1.,0.,0.,0.] and col['E']==[0.]*4
    assert a['positions'][1]['columns'][0]['B']==[2.,0.,0.,0.]
    assert len(a['position_comparisons'])==2 and 'structural zero' in a['G_minus_D']
    rows[1]['status']='failed_partial'; a=m.analyze(p,rows)
    assert a['positions'][0]['columns'][0]['B'] is None and a['positions'][0]['a_L2'] is None
    assert len(a['positions'][0]['responses'])==21 and a['positions'][0]['complete']==20
    rows[0]['status']='not_started'; a=m.analyze(p,rows)
    assert not a['positions'][0]['zero_endpoint_available'] and all(v['delta_position4'] is None for v in a['positions'][0]['responses'])


def test_zero_task_direction_undefined_without_dropping_root():
    p=protocol(True)
    for r in p['roots']:
        s=r['direction']['root_state']; r['direction']=m.direction(s,s)
    a=m.analyze(p,completed_rows(p))
    assert len(a['positions'])==2 and all(v['common_direction_T'] is None for x in a['positions'] for v in x['responses'])
    assert all(x['a_L2'] is None for x in a['positions'])


class RAMStore:
    def __init__(self): self.events=[]; self.files={}; self.previous=None; self.index=0; self.broken=False; self.path=RAMPath(self)
    def check(self,size=4096): return None
    def write(self,name,value,append=False,final=False):
        self.files[name]=(self.files.get(name,b'') if append else b'')+value
        return hashlib.sha256(self.files[name]).hexdigest()
    def put(self,name,v,final=False): return self.write(name,m.raw(v),final=final)
    def emit(self,kind,**fields):
        e=dict(index=self.index,previous_sha256=self.previous,kind=kind,**deepcopy(fields)); e['event_sha256']=m.digest(e)
        self.events.append(e); self.previous=e['event_sha256']; self.index+=1
        # Avoid quadratic append copies; actual Store has its own fsync/hashchain tests.
    def frame(self,name,v): return self.write(name,np.asarray(v).tobytes())
    def arrays(self,name,**values): return self.put(name,{k:np.asarray(v).tolist() for k,v in values.items()})


class RAMPath:
    def __init__(self,store,name=''): self.store,self.name=store,name
    def iterdir(self): return [RAMPath(self.store,k) for k in self.store.files]
    def is_file(self): return self.name in self.store.files
    def __truediv__(self,name): return RAMPath(self.store,name)


class Space:
    def step(self,*a): return None


class Part:
    def __init__(self,body,name): self.body,self.name=body,name; self.force=(0.,0.); self.torque=0.; self.is_sleeping=False; self.shapes=[]
    @property
    def position(self): return self.body.state[:2] if self.name=='agent' else self.body.state[2:4]
    @property
    def velocity(self): return self.body.state[5:] if self.name=='agent' else [0.,0.]
    @property
    def angle(self): return 0. if self.name=='agent' else float(self.body.state[4])
    @property
    def angular_velocity(self): return 0.


class Env:
    def __init__(self,stop=None):
        self.unwrapped=self; self.space=Space(); self.state=np.array([100.,120.,200.,220.,.2,0.,0.]); self.goal_state=np.array([400.,400.,400.,400.,1.,0.,0.])
        self.agent=Part(self,'agent'); self.block=Part(self,'block'); self.render_action=False; self.render_size=224; self.render_mode='rgb_array'
        self.relative=True; self.action_scale=100; self.dt=.01; self.control_hz=10; self.with_target=False; self.closed=False; self.steps=0; self.stop=stop; self._goal=self.render()
    def _get_obs(self): return self.state.copy()
    def _set_state(self,s): self.state=np.asarray(s).copy(); self.space.step(.01)
    def _set_goal_state(self,g): self.goal_state=np.asarray(g).copy()
    def reset(self,seed,options=None):
        self.steps=0
        s=np.array([100.+max(seed-100,0),120.,200.,220.,.2,0.,0.]) if options is None else options['state']
        if options is not None: self.goal_state=options['goal_state'].copy()
        self._set_state(s); self._set_state(s); self._goal=self.render()
        return dict(state=self._get_obs()),dict(goal_state=self.goal_state.copy(),goal=self._goal.copy())
    def render(self):
        a=np.zeros((224,224,3),np.uint8); a[:]=[65,105,225]
        if self.stop=='guard' and self.steps==16: a[:]=255
        return a
    def eval_state(self,goal,state):
        x=m.old.metric(goal,state); return x['native_success'],x['native_distance']
    def step(self,u):
        self.steps+=1
        for _ in range(10): self.space.step(.01)
        self.state[:2]+=np.asarray(u)*3; self.state[5:]=np.asarray(u)*100
        yes,d=self.eval_state(self.goal_state,self.state)
        return dict(state=self._get_obs()),-d,yes,bool(self.stop=='terminal' and self.steps==16),dict(goal_state=self.goal_state.copy(),goal=self._goal.copy())
    def close(self): self.closed=True


def runtime_data(p):
    refs=[]; oldprefix={}
    for k,r in enumerate(p['roots']):
        s=np.asarray(r['direction']['root_state']); e=Env(); e.state=s.copy(); frame=e.render()
        snap=m.builder.snapshot(e)
        refs.append(dict(slot=dict(index=k,reset_seed=100+k,root_seed=k),root_sha256=r['root_sha256'],goal=np.asarray(r['goal']),
            goal_rgb=frame,history=np.stack([frame]*3),states=np.stack([s]*3),dynamics=[deepcopy(snap) for _ in range(3)]))
        prefix=next(x for x in p['role_aliases'] if x['root_slot']==k and x['role']=='D')['prefix_commands']
        for u in prefix: e.state[:2]+=np.asarray(u)*3; e.state[5:]=np.asarray(u)*100
        oldprefix[k]=dict(state=e.state.tolist(),snapshot=m.builder.snapshot(e),pixel_sha256=m.builder.image_sha(e.render()))
    return dict(references=refs,old_prefix=oldprefix,plan={})


def mock_native(monkeypatch):
    monkeypatch.setattr(m.builder,'validate_env',lambda body:None)
    monkeypatch.setattr(m.source,'sample_state',lambda goal,shapes,slot,*args:np.array([100.+slot['root_seed'],120.,200.,220.,.2,0.,0.]))
    real=m.sha
    monkeypatch.setattr(m,'sha',lambda path:hashlib.sha256(path.store.files[path.name]).hexdigest() if isinstance(path,RAMPath) else real(path))


def test_full_recipe_rebuilt_every_probe_with_exact_actual_mock_hooks(monkeypatch):
    mock_native(monkeypatch); p=protocol(True); data=runtime_data(p); store=RAMStore(); envs=[]
    def factory(): e=Env(); envs.append(e); return e
    result=m.execute(p,data,store,factory,authorization=m.AUTH)
    assert result['status']=='common_prefix_probes_completed',result['errors']
    assert result['planned']==result['completed']==42 and len(envs)==42 and all(e.closed for e in envs)
    assert result['counts']=={k:dict(attempted=v,completed=v) for k,v in p['contract']['caps'].items()}
    assert sum(e['kind']=='prefix_pairing' for e in store.events)==42
    assert all(r['prefix_exact'] for r in result['rows']) and all(r['own_returned']==10 for r in result['rows'])
    assert result['P1_admission'] is False


@pytest.mark.parametrize('stop',('guard','terminal'))
def test_real_partial_mock_denominator_kept_and_no_post_stop_dispatch(monkeypatch,stop):
    mock_native(monkeypatch); p=protocol(True); data=runtime_data(p); store=RAMStore(); envs=[]
    def factory(): e=Env(stop); envs.append(e); return e
    r=m.execute(p,data,store,factory,authorization=m.AUTH)
    assert r['status']=='partial_or_failed' and r['planned']==42 and r['completed']==0
    assert all(x['own_returned']==6 for x in r['rows']) and all(e.closed for e in envs)
    assert all(not v['zero_endpoint_available'] for v in r['analysis']['positions'])
    if stop=='guard':
        assert any(e['kind']=='guard_receipt' and e['accepted'] is False for e in store.events)
        assert all(r['PNG']==18 for r in r['rows']) # warmup11 + plan0..6 saved BEFORE rejected guard


def test_prefix_lowest_bit_and_hidden_body_difference_stop_before_suffix():
    p=protocol(True); plan=p['plans'][0]; row={}; store=RAMStore(); e=Env(); data=runtime_data(p); anchors={}
    sink=m.BranchSink(store,p,data,plan,row,e,anchors); sink.returned=np.zeros((5,2)).tolist()
    receipt=dict(state=e.state.tolist(),pixel_sha256=m.builder.image_sha(e.render()),snapshot=m.builder.snapshot(e))
    sink.emit('fresh_observation',receipt=receipt); sink.emit('guard_receipt',phase='plan',step=5,accepted=True)
    assert row['prefix_exact']
    for which in ('state','body'):
        plan2=p['plans'][1]; sink=m.BranchSink(store,p,data,plan2,{},e,anchors); sink.returned=np.zeros((5,2)).tolist()
        receipt=dict(state=e.state.tolist(),pixel_sha256=m.builder.image_sha(e.render()),snapshot=m.builder.snapshot(e))
        sink.emit('fresh_observation',receipt=receipt)
        if which=='state': receipt['state'][0]=np.nextafter(receipt['state'][0],np.inf)
        else: receipt['body_dynamics']['block']['float64_bits']='wrong'
        with pytest.raises(m.PrefixMismatch): sink.emit('guard_receipt',phase='plan',step=5,accepted=True)


def test_auth_budget_wall_disk_and_nonrefund_errors(monkeypatch):
    p=protocol(True); store=RAMStore()
    with pytest.raises(ValueError,match='authorization'): m.execute(p,{},store,lambda:pytest.fail('factory'),authorization=None)
    with pytest.raises(ValueError,match='authorization'): m.run('notread','notread','notcreated',authorization=None)
    meter=m.Ops(p,store,clock=lambda:0.)
    def fail(): raise RuntimeError('primary native failure')
    with pytest.raises(RuntimeError): meter.call('reset',fail)
    assert meter.counts['reset']==dict(attempted=1,completed=0)
    with pytest.raises(ValueError): meter.call('reset',lambda:None)
    clock=iter((0.,601.)); meter=m.Ops(p,store,clock=lambda:next(clock))
    with pytest.raises(m.FatalBudget): meter.call('reset',lambda:pytest.fail('late dispatch'))
    meter=m.Ops(p,store,clock=lambda:0.); meter.counts['reset']['attempted']=p['contract']['caps']['reset']
    with pytest.raises(m.FatalBudget): meter.call('reset',lambda:pytest.fail('over cap'))
    monkeypatch.setattr(store,'check',lambda *a:(_ for _ in ()).throw(OSError('disk floor')))
    meter=m.Ops(p,store,clock=lambda:0.)
    with pytest.raises(OSError): meter.call('reset',lambda:pytest.fail('low disk'))
    assert meter.counts['reset']['attempted']==0


def test_first_error_cleanup_not_overwritten_and_remaining_plans_not_started(monkeypatch):
    mock_native(monkeypatch); p=protocol(True); store=RAMStore()
    class Broken(Env):
        def reset(self,*a,**kw): raise RuntimeError('first reset')
        def close(self): self.closed=True; raise RuntimeError('second close')
    e=Broken(); r=m.execute(p,runtime_data(p),store,lambda:e,authorization=m.AUTH)
    assert r['planned']==42 and r['not_started']==41 and r['first_error']['message']=='first reset'
    assert r['rows'][0]['first_error']['message']=='first reset'
    assert any(x['message']=='second close' for x in r['rows'][0]['secondary_errors']) and e.closed


def actual_arguments():
    root=m.ROOT/'results'
    return dict(sourcepack=str(root/'p1_constructive_reachable_source_20261027_v1_main01/sourcepack.json'),source_sha=m.old.SOURCE_SHA,
        source_prepared=str(root/'p1_constructive_reachable_source_20261027_v1_prepare01.json'),source_prepare_sha='98ffb806018a270f5ef27ffa76be929393a74355c213bc274bf158c72d80b086',
        planner_report=str(root/'p1_constructive_blind_planners_cuda_20261027_v1_cuda_main01/report.json'),planner_sha=m.PLANNER_SHA,
        physical_report=str(root/'p1_constructive_plan_physical_20261027_v1_main01/report.json'),physical_sha=m.PHYSICAL_SHA,wall_seconds=600)


def test_actual_saved_RAM_collect_no_prepare_no_environment():
    p,data=m.collect_saved(**actual_arguments())
    assert len(p['positions'])==4 and len(p['plans'])==84 and len(p['role_aliases'])==6
    assert [a['candidate_index'] for a in p['role_aliases']]==[None,2,2,None,7,7]
    assert len(m.raw(p))<=256*1024
    assert len(data['references'])==2 and p['P1_admission'] is False
    assert p['contract']['caps']['physics_step']==17136
