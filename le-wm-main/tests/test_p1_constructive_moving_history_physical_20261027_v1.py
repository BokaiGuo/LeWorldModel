"""RAM-only regression: artificial array transitions, never native physics/neural."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import p1_constructive_moving_history_physical_20261027_v1 as m
from test_p1_constructive_moving_history_source_20261027_v1 import FakeEnv
from test_p1_constructive_plan_physical_20261027_v1 import MemoryStore, manifest_fixture as old_manifest


REQUEST=np.array([100.,100.,150.,150.,0.,0.,0.])


class ArrayEnv(FakeEnv):
    def __init__(self,*,fault=None,at=11,close_error=False):
        super().__init__(close_error=close_error)
        self.render_action=False;self.render_size=224;self.render_mode='rgb_array'
        self.goal_pose=np.array([300.,300.,0.]);self.fault=fault;self.at=at;self.closed=False
    def render(self):
        a=super().render()
        if len(self.commands)==self.at:
            if self.fault=='guard':a[0,0]=0
            if self.fault=='render':self.agent.velocity=(1.,0.)
        return a
    def step(self,c):
        v=super().step(c)
        if len(self.commands)==self.at:
            if self.fault=='throw':raise RuntimeError('primary step error')
            if self.fault=='substitute':self.latest_action=np.zeros_like(c)
            if self.fault=='terminal':
                self.state=self.goal_state.copy();self.agent.position=tuple(self.state[:2]);self.block.position=tuple(self.state[2:4])
                return dict(state=self._get_obs()),0.,True,False,self.info()
            if self.fault=='truncate':return v[0],v[1],False,True,v[4]
        return v
    def close(self):self.closed=True;return super().close()


def fixture():
    env=ArrayEnv();env.reset(seed=1);obs,_=env.reset(seed=0,options=dict(state=REQUEST.copy(),goal_state=env.goal_state.copy()))
    frames=[env.render()];states=[obs['state']];snaps=[m.moving.snapshot(env)]
    for n,c in enumerate(m.moving.warmup_commands(),1):
        obs,*_=env.step(c.copy())
        if n in (5,10):frames.append(env.render());states.append(obs['state']);snaps.append(m.moving.snapshot(env))
    goal=np.array([400.,400.,350.,350.,0.,0.,0.]);goal_rgb=np.full((224,224,3),31,np.uint8)
    refs=[dict(slot=dict(index=i,root_seed=11+i,reset_seed=21+i),root_sha256=m.source.root_hash(states[-1],goal),
        goal=goal.copy(),goal_rgb=goal_rgb.copy(),history=np.asarray(frames),states=np.asarray(states),
        dynamics=deepcopy(snaps),raw_history=m.moving.warmup_commands(),blind_sha256=str(i)*64) for i in (0,1)]
    plans=[dict(root_slot=i,arm=a,commands=np.full((10,2),.25 if a=='GN' else -.25,np.float32).tolist(),selection_digest=str(i)+a)
        for i in (0,1) for a in ('GN','D2P')]
    return dict(plans=plans,contract=m.contract(60),bindings={},sourcepack_sha256='e'*64),dict(references=refs,plan={})


def execute(factory=ArrayEnv,*,p=None,data=None,store=None):
    fp,fd=fixture();p=fp if p is None else p;data=fd if data is None else data;store=MemoryStore() if store is None else store;envs=[]
    def make():e=factory();envs.append(e);return e
    def fake_sha(path):
        assert Path(path).name=='journal.jsonl'
        return m.source.digest(store.events)
    forbidden=AssertionError('old branch/witness/episode forbidden')
    with patch.object(m.moving,'validate_env',lambda _:None),patch.object(m.source,'sample_state',lambda *a:REQUEST.copy()),\
         patch.object(m,'sha',fake_sha),patch.object(m.low,'branch',side_effect=forbidden),\
         patch.object(m.moving,'witness',side_effect=forbidden),patch.object(m.moving,'episode',side_effect=forbidden):
        r=m.execute(p,data,store,make,authorization=m.AUTH)
    return r,store,envs


def test_complete_actual_nonzero_and_own_commands_census():
    r,s,envs=execute();assert r['status']=='moving_history_physical_recorded',r
    expected=dict(reset=8,public_step=80,explicit_render=88,goal_render=8,set_state=16,physics_step=816,goal_bind=4)
    assert r['counts']=={k:dict(attempted=v,completed=v) for k,v in expected.items()}
    assert len(r['rows'])==4 and not r['P1_admission'] and r['new_neural_calls']==0
    for n,(row,e) in enumerate(zip(r['rows'],envs,strict=True)):
        assert row['initial_exact'] and row['PNG']==22 and row['full_horizon'] and e.closed
        assert m.moving.exact(np.asarray(e.commands[:10]),m.moving.warmup_commands())
        assert m.moving.exact(np.asarray(e.commands[10:]),np.asarray(fixture()[0]['plans'][n]['commands'],np.float32))
        assert [x['step'] for x in row['checkpoints']]==[0,5,10]
        assert 'reset' not in e.__dict__ and '_set_state' not in e.__dict__
        assert np.array_equal(e.goal_pose,[300,300,0])
    prev=None
    for e in s.events:
        assert e['previous_sha256']==prev and e['event_sha256']==m.source.digest({k:v for k,v in e.items() if k!='event_sha256'})
        prev=e['event_sha256']
    assert r['journal_chain_tail']==prev and 'report.json' in s.files


@pytest.mark.parametrize('a',[np.zeros((10,2),np.float32),m.moving.warmup_commands().astype(np.float64),
    np.tanh(m.moving.warmup_commands()),np.asarray([[.1,0]]*5+[[0,.1]]*4,np.float32)])
def test_wrong_raw_payload_before_reset(a):
    p,d=fixture();d['references'][0]['raw_history']=a
    r,_,envs=execute(p=p,data=d)
    assert r['status']=='partial_or_failed' and all(len(e.resets)==0 for e in envs[:2])
    assert len(r['rows'])==4 and r['rows'][2]['full_horizon']


@pytest.mark.parametrize('field',['history','states','velocity','signedzero'])
def test_all_three_boundaries_exact_including_body_bits(field):
    p,d=fixture();ref=d['references'][0]
    if field=='history':ref['history'][1,2,2,0]^=1
    elif field=='states':ref['states'][1,0]+=1e-10
    elif field=='velocity':ref['dynamics'][0]['bodies']['agent']['velocity'][0]=1e-10
    else:ref['dynamics'][2]['state_float64_bits']='bad_signed_zero_bits'
    r,_,envs=execute(p=p,data=d)
    assert all(len(e.commands)==10 for e in envs[:2]) and not r['rows'][0]['initial_exact']
    assert r['counts']['goal_bind']['completed']==2


@pytest.mark.parametrize('at',[0,5,11])
def test_failed_guard_png_snapshot_precedes_rejection(at):
    r,s,envs=execute(lambda:ArrayEnv(fault='guard',at=at))
    assert r['status']=='partial_or_failed' and len(r['rows'])==4
    for i,e in enumerate(envs):
        assert len(e.commands)==at
        row=r['rows'][i];receipt=row['last_observation'];assert receipt['png'] in s.files
        observed=[x for x in s.events if x['kind']=='fresh_observation' and x['branch']==i][-1]
        guard=[x for x in s.events if x['kind']=='guard_receipt' and x['branch']==i][-1]
        assert observed['index']<guard['index'] and not guard['accepted']


@pytest.mark.parametrize('fault',['substitute','render'])
@pytest.mark.parametrize('at',[5,11])
def test_hidden_transition_or_render_mutation_retained(fault,at):
    r,s,envs=execute(lambda:ArrayEnv(fault=fault,at=at))
    assert all(len(e.commands)==at for e in envs) and r['status']=='partial_or_failed'
    assert all(x['last_observation']['png'] in s.files for x in r['rows'])


@pytest.mark.parametrize('fault',['terminal','truncate'])
def test_native_stop_no_pad_no_retry(fault):
    r,_,envs=execute(lambda:ArrayEnv(fault=fault))
    assert r['status']=='moving_history_physical_recorded',r
    assert len(envs)==4 and all(len(e.commands)==11 for e in envs)
    assert all(x['own_returned']==1 and not x['full_horizon'] for x in r['rows'])
    assert r['counts']['physics_step']['completed']==456


def test_warmup_terminal_never_executes_plan():
    r,_,envs=execute(lambda:ArrayEnv(fault='truncate',at=5))
    assert all(len(e.commands)==5 for e in envs) and all(x['own_attempted']==0 for x in r['rows'])
    assert r['counts']['goal_bind']['attempted']==0


def test_step_exception_keeps_first_and_last_available_frame():
    r,s,envs=execute(lambda:ArrayEnv(fault='throw',close_error=True))
    assert all(e.closed for e in envs) and len(r['rows'])==4
    for row in r['rows']:
        assert 'primary step' in row['first_error']['message'] and 'secondary fake close' in str(row['secondary_errors'])
        assert row['last_observation']['step']==0 and row['last_observation']['phase']=='plan'
        assert row['own_attempted']==1 and row['own_returned']==0
    assert r['counts']['public_step']==dict(attempted=44,completed=40)


@pytest.mark.parametrize('fault',['setter','step','render','pose'])
def test_goal_bind_has_no_poststate_or_extra_physics_or_render(fault):
    original=m.moving.attach_goal
    def attach(body,goal,rgb):
        original(body,goal,rgb)
        if fault=='setter':body._set_state(body._get_obs())
        elif fault=='step':body.space.step(.01)
        elif fault=='render':body.render()
        else:body.goal_pose[0]+=1
    with patch.object(m.moving,'attach_goal',attach):r,_,_=execute()
    assert r['status']=='partial_or_failed' and all(x['own_attempted']==0 for x in r['rows'])


def test_authorization_caps_wall_storage_before_dispatch():
    p,d=fixture();s=MemoryStore();ops=m.Ops(p,s);p['contract']['caps']['public_step']=0
    with pytest.raises(ValueError):ops.call('public_step',lambda:pytest.fail('dispatch'))
    ops.started-=1000
    with pytest.raises(ValueError):ops.call('reset',lambda:pytest.fail('dispatch'))
    with pytest.raises(ValueError):m.execute(p,d,s,lambda:pytest.fail('factory'),authorization='wrong')
    with pytest.raises(ValueError):m.run('no','no','no',lambda:pytest.fail('factory'),authorization='wrong')
    store=object.__new__(m.Store);store.path=NS(iterdir=lambda:[])
    with patch.object(m.low.shutil,'disk_usage',return_value=NS(free=m.FLOOR-1)),pytest.raises(ValueError):store.check(1)


@pytest.mark.parametrize('stage',['final_report','final_journal','branch_closed'])
def test_durability_failure_never_reports_success(stage):
    s=MemoryStore();put,emit=s.put,s.emit
    def newput(n,v,final=False):
        if stage=='final_report' and final:raise OSError('final write failed')
        return put(n,v,final)
    def newemit(kind,**kw):
        if (stage=='final_journal' and kind=='run_finished') or kind==stage:raise OSError('journal failed')
        return emit(kind,**kw)
    s.put=newput;s.emit=newemit;r,_,envs=execute(store=s)
    assert r['status']=='partial_or_failed'
    if stage=='branch_closed':assert len(envs)==1 and all(x['status']=='not_started' for x in r['rows'][1:])


def test_explicit_manifest_mapping_no_old_source_or_native_identity_in_plans():
    a,d,r=old_manifest();a['schema']=m.NAME+'.selection'
    with patch.object(m.low,'sha',return_value='c'*64),patch.object(m.low,'read',return_value=r):
        rows,_=m.selection(a,d)
    assert len(rows)==4 and all('root_sha256' not in x for x in r['rows'])
    for field in ('source_slot','blind_sha','commands','arm'):
        bad=deepcopy(r);bad['rows'][0][field]=None
        with patch.object(m.low,'sha',return_value='c'*64),patch.object(m.low,'read',return_value=bad),pytest.raises(ValueError):m.selection(a,d)
    a['arms']=['GN']
    with pytest.raises(ValueError):m.selection(a,d)


def test_exact_moving_consume_handoff_shape_and_default_spaces_digest():
    a,_,_=old_manifest();_,d=fixture();arms=['line_search_GN','full10_conditional_D2P']
    consume=dict(schema='moving_history_planners_cuda_20261027_v1.consume',
        status='moving_commands_verified_not_admitted',new_physics=0,P1_admission=False,P2_admission=False,plans=[])
    a.update(schema=m.NAME+'.selection',arms=arms,roots=[dict(root_slot=i,root_sha256=r['root_sha256']) for i,r in enumerate(d['references'])])
    for n,row in enumerate(a['plans']):
        slot=n//2;arm=arms[n%2];command=fixture()[0]['plans'][n]['commands']
        consume['plans'].append(dict(source_slot=slot,arm=arm,commands=command,blind_sha256=d['references'][slot]['blind_sha256'],selection_sha256='f'*64))
        row.update(arm=arm,commands=command,compiler_schema='moving_history_planner_handoff.v1',compiler_status='commands_verified_before_physics',
            commands_pointer=f'/data/plans/{n}/commands',source_slot_pointer=f'/data/plans/{n}/source_slot',
            blind_sha_pointer=f'/data/plans/{n}/blind_sha256',arm_pointer=f'/data/plans/{n}/arm')
        row['selection_digest']=m.source.digest({k:v for k,v in row.items() if k!='selection_digest'})
    envelope=dict(schema='moving_history_planner_handoff.v1',status='commands_verified_before_physics',errors=[],data=consume)
    with patch.object(m.low,'sha',return_value='c'*64),patch.object(m.low,'read',return_value=envelope):rows,_=m.selection(a,d)
    assert len(rows)==4 and rows[3]['arm']==arms[1]


def test_final_source_edit_fails_closed():
    p,d=fixture();p['bindings']={'/MOCK/sealed.py':'old'}
    # execute's mock SHA permits only the final journal: any source recheck fails.
    r,_,_=execute(p=p,data=d)
    assert r['status']=='partial_or_failed' and r['errors'][0]['stage']=='final_bindings'


def saved_fixture():
    _,d=fixture();p=dict(slots=[x['slot'] for x in d['references']],raw_warmup=m.moving.warmup_commands().tolist());rows=[];events=[dict(kind='run_started',prepare=p)];blind=[]
    for i,ref in enumerate(d['references']):
        rows.append(dict(slot=ref['slot'],reachable=True,status='reachable_native_replayed',replay_native_success=[True,True],root_sha256=ref['root_sha256'],blind_payload_sha256=ref['blind_sha256']))
        blind.append(dict(raw_action_history=ref['raw_history'],rgb_history=ref['history'],goal_rgb=ref['goal_rgb']))
        for phase in ('replay0','replay1'):
            for j,step in enumerate((0,5,10)):
                events.append(dict(kind='fresh_observation',slot=i,phase=phase,step=step,state=ref['states'][j].tolist(),snapshot=ref['dynamics'][j],pixel_sha256=m.moving.image_sha(ref['history'][j])))
            for step,c in enumerate(ref['raw_history'],1):events.append(dict(kind='warmup_return',slot=i,phase=phase,step=step,raw_action=c.tolist(),raw_float32_bits=c.tobytes().hex(),terminated=False,truncated=False))
        events.append(dict(kind='native_return',slot=i,phase='construction',step=1,state=ref['goal'].tolist()))
        events.append(dict(kind='fresh_observation',slot=i,phase='construction',step=11,pixel_sha256=m.moving.image_sha(ref['goal_rgb'])))
    counts={'mock':'not actual'};events.append(dict(kind='run_finished',status='fixed_slots_processed',counts=counts,errors=[]))
    pack=dict(schema=m.moving.NAME+'.sourcepack',status='fixed_slots_processed',errors=[],first_error=None,sourcepack_complete=True,reachable_count=2,prepare_id=m.source.digest(p),rows=rows,counts=counts)
    return pack,p,events,blind


def test_saved_source_adapter_does_not_need_certificate_or_witness_arrays():
    args=saved_fixture();refs=m.references(*args)
    assert len(refs)==2 and m.moving.exact(refs[0]['raw_history'],m.moving.warmup_commands())
    assert refs[0]['root_sha256']==fixture()[1]['references'][0]['root_sha256']


@pytest.mark.parametrize('field',['bits','replay','goal','order','raw','prepare'])
def test_saved_source_adapter_rejects_receipt_drift(field):
    pack,p,e,b=saved_fixture()
    if field=='bits':next(x for x in e if x['kind']=='warmup_return')['raw_float32_bits']='bad'
    elif field=='replay':next(x for x in e if x.get('phase')=='replay1' and x['kind']=='fresh_observation')['snapshot']=dict(state=[])
    elif field=='goal':b[0]['goal_rgb']=np.zeros_like(b[0]['goal_rgb'])
    elif field=='order':pack['rows'].reverse()
    elif field=='raw':b[0]['raw_action_history']=np.zeros((10,2),np.float32)
    else:pack['prepare_id']='bad'
    with pytest.raises(ValueError):m.references(pack,p,e,b)


def test_z_import_no_native_modules():
    assert not any(n in sys.modules for n in ('torch','gymnasium','pymunk','stable_worldmodel'))
