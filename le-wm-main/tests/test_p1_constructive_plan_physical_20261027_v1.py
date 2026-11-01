"""In-memory array dynamics only. Never invoke a simulator/model or make tmp files."""
from copy import deepcopy
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
import pytest
from experiments.spectral_reachability import p1_constructive_plan_physical_20261027_v1 as m


class MemoryStore:
    def __init__(self):self.files={};self.events=[];self.path=Path('/MEMORY_ONLY');self.previous=None
    def check(self,size=4096):return None
    def put(self,name,value,final=False):
        assert name not in self.files
        self.files[name]=m.raw(value)
    def emit(self,kind,**fields):
        e=dict(index=len(self.events),previous_sha256=self.previous,kind=kind,**deepcopy(fields))
        e['event_sha256']=m.source.digest(e);self.previous=e['event_sha256'];self.events.append(e)
    def frame(self,name,value):
        assert name not in self.files
        self.files[name]=value.copy();return hashlib.sha256(value.tobytes()).hexdigest()
    def arrays(self,name,**values):
        self.files[name]=deepcopy(values)
        return hashlib.sha256(b''.join(x.tobytes() for x in values.values())).hexdigest()


class ArrayEnv:
    """An explicitly artificial numeric state machine, NOT native physics."""
    def __init__(self,*,guard_fail=None,step_fail=None,close_fail=False,terminal=False,render_mutates=False):
        self.unwrapped=self;self.render_action=False;self.render_size=224;self.render_mode='rgb_array'
        self.block=NS(velocity=(0.,0.),angular_velocity=0.,angle=0.,shapes=[])
        self.state=np.array([100.,100.,150.,150.,0.,0.,0.]);self.goal_state=None
        self.guard_fail=guard_fail;self.step_fail=step_fail;self.close_fail=close_fail;self.terminal=terminal
        self.render_mutates=render_mutates;self.steps=0;self.closed=False;self.sent=[]
    def _get_obs(self):return self.state.copy()
    def _set_state(self,state):self.state=np.asarray(state).copy();self.block.angle=self.state[4];self.space.step(.01)
    def _set_goal_state(self,goal):self.goal_state=goal.copy()
    def info(self):return dict(goal_state=self.goal_state.copy(),goal=self._goal.copy())
    def reset(self,seed,options=None):
        self.space=NS(step=lambda dt:None);self.original_space_step=self.space.step
        goal=np.array([400.,400.,350.,350.,0.,0.,0.]) if options is None else options['goal_state']
        self._set_state(goal);self._set_goal_state(goal);self._goal=self.render()
        self._set_state(np.array([100.,100.,150.,150.,0.,0.,0.]) if options is None else options['state'])
        return dict(state=self._get_obs()),self.info()
    def render(self):
        rgb=np.zeros((224,224,3),np.uint8)
        if self.steps!=self.guard_fail:rgb[0,0]=[65,105,225]
        if self.render_mutates and self.steps>10:self.state[0]+=1
        return rgb
    def eval_state(self,goal,state):
        d=state-goal;a=abs(d[4]);return bool(np.linalg.norm(d[:4])<20 and min(a,2*np.pi-a)<np.pi/9),float(np.linalg.norm(d))
    def step(self,c):
        self.steps+=1;self.sent.append(c.copy())
        for _ in range(10):self.space.step(.01)
        if self.steps==self.step_fail:raise RuntimeError('primary step failed')
        self.state[:4]+=np.tile(np.asarray(c),2)
        if self.terminal and self.steps==11:self.state=self.goal_state.copy()
        success,distance=self.eval_state(self.goal_state,self.state)
        return dict(state=self._get_obs()),-distance,success,False,self.info()
    def close(self):
        self.closed=True
        if self.close_fail:raise RuntimeError('secondary close failed')


def fixture():
    env=ArrayEnv();obs,_=env.reset(seed=0);s=obs['state'];rgb=env.render();snap=m.builder.snapshot(env)
    refs=[]
    for i in range(2):
        goal=np.array([300.,300.,300.,300.,0.,0.,0.])
        refs.append(dict(slot=dict(index=i,root_seed=11+i,reset_seed=21+i),root_sha256=m.source.root_hash(s,goal),
            goal=goal,goal_rgb=np.full_like(rgb,12),history=np.array([rgb]*3),states=np.array([s]*3),
            dynamics=[deepcopy(snap) for _ in range(3)],blind_sha256=str(i)*64))
    plans=[dict(root_slot=i,arm=arm,commands=np.full((10,2),.25 if arm=='GN' else -.25).tolist(),
        selection_digest=str(i)+arm) for i in range(2) for arm in ('GN','GD')]
    return dict(plans=plans,contract=m.contract(4,60),bindings={},sourcepack_sha256=m.SOURCE_SHA),dict(references=refs,plan={})


def execute(factory=ArrayEnv,*,p=None,data=None,store=None):
    fp,fd=fixture();p=fp if p is None else p;data=fd if data is None else data;store=MemoryStore() if store is None else store
    envs=[]
    def make():env=factory();envs.append(env);return env
    def fake_sha(path):
        assert Path(path).name=='journal.jsonl'
        return m.source.digest(store.events)
    with patch.object(m.builder,'validate_env',lambda _:None),patch.object(m.source,'sample_state',lambda *a:np.array([100.,100.,150.,150.,0.,0.,0.])),patch.object(m,'sha',fake_sha),patch.object(m.builder,'episode',side_effect=AssertionError('witness episode forbidden')),patch.object(m.builder,'witness',side_effect=AssertionError('witness forbidden')):
        result=m.execute(p,data,store,make,authorization=m.AUTH)
    return result,store,envs


def test_complete_own_commands_and_actual_hook_census():
    r,store,envs=execute();assert r['status']=='physical_calibration_recorded',r
    assert len(r['rows'])==4 and r['native_successes']==0 and not r['P1_admission']
    expected=dict(reset=8,public_step=80,explicit_render=88,goal_render=8,set_state=16,physics_step=816,goal_bind=4)
    assert r['counts']=={k:dict(attempted=v,completed=v) for k,v in expected.items()}
    for index,(row,env) in enumerate(zip(r['rows'],envs)):
        assert row['initial_exact'] and row['own_returned']==10 and row['PNG']==22
        assert [x['step'] for x in row['checkpoints']]==[0,5,10]
        assert np.array_equal(env.sent[:10],np.zeros((10,2)))
        assert np.array_equal(env.sent[10:],fixture()[0]['plans'][index]['commands'])
        assert env.closed and '_set_state' not in env.__dict__ and env.space.step is env.original_space_step
    previous=None
    for e in store.events:
        assert e['previous_sha256']==previous
        assert e['event_sha256']==m.source.digest({k:v for k,v in e.items() if k!='event_sha256'})
        previous=e['event_sha256']


@pytest.mark.parametrize('step',[0,7,11])
def test_failing_rgb_state_snapshot_durable_before_guard_and_no_next_dispatch(step):
    r,store,envs=execute(lambda:ArrayEnv(guard_fail=step))
    assert r['status']=='partial_or_failed'
    assert all(e.steps==step for e in envs)
    for branch in range(4):
        guard=next(e for e in store.events if e['kind']=='guard_receipt' and e['branch']==branch and not e['accepted'])
        observed=next(e for e in store.events if e['kind']=='fresh_observation' and e['branch']==branch and e['receipt']['png']==guard['png'])
        assert observed['index']<guard['index'] and guard['png'] in store.files
        assert observed['receipt']['snapshot']['state']==observed['receipt']['state']
        assert r['rows'][branch]['last_observation']['png']==guard['png']
        assert not any(e['kind']=='operation_attempted' and e.get('branch')==branch and e['index']>guard['index'] for e in store.events)


def test_blind_mismatch_prevents_goal_and_own_commands():
    p,d=fixture();d['references'][0]['history'][0,2,2,0]=5
    r,_,envs=execute(p=p,data=d)
    assert envs[0].steps==envs[1].steps==10 and not r['rows'][0]['initial_exact']
    assert r['rows'][2]['initial_exact']


def test_native_terminal_honored_no_padding_or_replacement():
    r,_,envs=execute(lambda:ArrayEnv(terminal=True))
    assert r['status']=='physical_calibration_recorded' and r['native_successes']==4
    assert all(e.steps==11 for e in envs)
    assert all(x['own_returned']==1 and not x['full_horizon'] for x in r['rows'])


def test_render_must_not_change_dynamics_after_incident_persisted():
    r,s,_=execute(lambda:ArrayEnv(render_mutates=True))
    assert all(x['own_returned']==1 for x in r['rows'])
    assert len([e for e in s.events if e['kind']=='guard_receipt' and not e['render_preserved_dynamics']])==4


def test_first_error_survives_cleanup_and_full_denominator():
    r,_,envs=execute(lambda:ArrayEnv(step_fail=11,close_fail=True))
    assert len(r['rows'])==4 and all(e.closed for e in envs)
    assert all('primary step' in x['first_error']['message'] and 'secondary close' in str(x['secondary_errors']) for x in r['rows'])
    assert r['counts']['public_step']==dict(attempted=44,completed=40)


def test_branch_close_journal_failure_keeps_remaining_not_started():
    s=MemoryStore();emit=s.emit
    def fail(kind,**kw):
        if kind=='branch_closed':raise OSError('journal closed failure')
        return emit(kind,**kw)
    s.emit=fail;r,_,envs=execute(store=s)
    assert len(envs)==1 and len(r['rows'])==4
    assert [x['status'] for x in r['rows'][1:]]==['not_started']*3
    assert r['errors'][0]['stage']=='execution' and 'report.json' in s.files


@pytest.mark.parametrize('stage',['final_journal','journal_hash','final_report'])
def test_finalizers_attempted_independently(stage):
    s=MemoryStore();emit=s.emit;put=s.put
    def fail_emit(kind,**kw):
        if stage=='final_journal' and kind=='run_finished':raise OSError('end journal failed')
        return emit(kind,**kw)
    def fail_put(name,value,final=False):
        if stage=='final_report' and name=='report.json':raise OSError('report failed')
        return put(name,value,final)
    s.emit=fail_emit;s.put=fail_put
    if stage=='journal_hash':
        # execute() supplies an in-memory hash function: force its underlying digest only at final hash.
        original=m.source.digest
        def digest(v):
            if isinstance(v,list):raise OSError('hash failed')
            return original(v)
        with patch.object(m.source,'digest',digest):r,_,_=execute(store=s)
    else:r,_,_=execute(store=s)
    assert r['status']=='partial_or_failed' and any(e['stage']==stage for e in r['errors'])
    if stage!='final_report':assert 'report.json' in s.files


def manifest_fixture():
    _,data=fixture();report=dict(schema='MOCK.blind',status='mock_completed',errors=[],rows=[])
    plans=[]
    for i in range(2):
        for arm in ('GN','GD'):
            idx=len(plans);a=np.zeros((10,2)).tolist()
            report['rows'].append(dict(source_slot=i,arm=arm,commands=a,blind_sha=data['references'][i]['blind_sha256']))
            row=dict(root_slot=i,arm=arm,commands=a,compiler_report='/MOCK/report.json',compiler_report_SHA='c'*64,
                compiler_schema='MOCK.blind',compiler_status='mock_completed',commands_pointer=f'/rows/{idx}/commands',
                source_slot_pointer=f'/rows/{idx}/source_slot',arm_pointer=f'/rows/{idx}/arm',blind_sha_pointer=f'/rows/{idx}/blind_sha')
            row['selection_digest']=m.source.digest(row);plans.append(row)
    manifest=dict(schema=m.NAME+'.selection',sourcepack='/MOCK/sourcepack.json',sourcepack_sha256=m.SOURCE_SHA,
        source_prepared='/MOCK/prepare.json',source_prepare_sha256='d'*64,
        roots=[dict(root_slot=i,root_sha256=r['root_sha256']) for i,r in enumerate(data['references'])],arms=['GN','GD'],plans=plans,
        selection_fixed_before_physics=True,physical_outcome_selection=False,wall_seconds=60)
    return manifest,data,report


def test_selection_has_no_native_rootsha_requirement_in_compiler():
    manifest,data,report=manifest_fixture()
    with patch.object(m,'sha',return_value='c'*64),patch.object(m,'read',return_value=report):
        rows,_=m.selection(manifest,data)
    assert len(rows)==4 and all('root_sha256' not in x for x in report['rows'])


@pytest.mark.parametrize('mutation',['slot','boolslot','blind','arm','commands','missing','reorder','digest','outcome','status'])
def test_selection_failclosed_mutations(mutation):
    a,d,r=manifest_fixture()
    if mutation=='slot':r['rows'][0]['source_slot']=1
    elif mutation=='boolslot':r['rows'][0]['source_slot']=False
    elif mutation=='blind':r['rows'][0]['blind_sha']='bad'
    elif mutation=='arm':r['rows'][0]['arm']='other'
    elif mutation=='commands':r['rows'][0]['commands']=[[1,1]]*10
    elif mutation=='missing':a['plans'].pop()
    elif mutation=='reorder':a['plans'].reverse()
    elif mutation=='digest':a['plans'][0]['selection_digest']='bad'
    elif mutation=='outcome':a['physical_outcome_selection']=True
    elif mutation=='status':r['status']='running'
    with patch.object(m,'sha',return_value='c'*64),patch.object(m,'read',return_value=r),pytest.raises(ValueError):m.selection(a,d)


@pytest.mark.parametrize('value',[np.zeros((9,2)),np.full((10,2),np.nan),np.full((10,2),1.001),np.full((10,2),.1),np.zeros((10,2),bool)])
def test_raw_commands_not_silently_clipped_rounded_or_padded(value):
    with pytest.raises(ValueError):m.commands(value)


def test_pointer_exact_and_not_executable():
    assert m.pointer({'a/b':{'~':[3]}},'/a~1b/~0/0')==3
    for p in ('rows/0','/*','/a~2','/00','/-1'):
        with pytest.raises(ValueError):m.pointer([0],p)


def test_caps_wall_disk_and_authorization_before_dispatch():
    p,d=fixture();s=MemoryStore();ops=m.Ops(p,s);p['contract']['caps']['public_step']=0
    with pytest.raises(ValueError):ops.call('public_step',lambda:pytest.fail('dispatched'))
    ops.started-=1000
    with pytest.raises(ValueError):ops.call('reset',lambda:pytest.fail('dispatched'))
    with pytest.raises(ValueError):m.execute(p,d,s,lambda:pytest.fail('factory'),authorization='no')
    with pytest.raises(ValueError):m.run('no','no','no',lambda:pytest.fail('factory'),authorization='no')


def test_store_predispatch_cap_and_floor_in_memory():
    s=object.__new__(m.Store);s.path=NS(iterdir=lambda:[])
    with patch.object(m.shutil,'disk_usage',return_value=NS(free=m.FLOOR-1)),pytest.raises(ValueError):s.check(1)
    with patch.object(m.shutil,'disk_usage',return_value=NS(free=10**12)),pytest.raises(ValueError):s.check(m.CAP)


def test_native_wrapped_discrepancy_not_relabelled():
    goal=np.zeros(7);state=np.zeros(7);state[4]=4*np.pi+1
    result=m.metric(goal,state)
    assert result['native_success'] and result['negative_native_angle'] and not result['wrapped_allxy_success']
    assert result['object_success'] is None


def test_actual_saved_source_read_only_whitelisted_bridge():
    pack=m.ROOT/'results/p1_constructive_reachable_source_20261027_v1_main01/sourcepack.json'
    prepared=m.ROOT/'results/p1_constructive_reachable_source_20261027_v1_prepare01.json'
    data=m.load_source(pack,m.SOURCE_SHA,prepared,'98ffb806018a270f5ef27ffa76be929393a74355c213bc274bf158c72d80b086')
    assert len(data['references'])==2 and all(x['history'].shape==(3,224,224,3) for x in data['references'])
    assert all('witness_commands' not in x for x in data['references'])


def test_invalid_render_retains_raw_before_failure():
    class Invalid(ArrayEnv):
        def render(self):
            return np.zeros((2,2,3),np.uint8) if self.steps==11 else super().render()
    r,s,_=execute(Invalid)
    assert all(x['own_returned']==1 and x['status']=='failed' for x in r['rows'])
    assert all(f'b{i}_plan_01_invalid.npz' in s.files for i in range(4))
    assert len([e for e in s.events if e['kind']=='invalid_render'])==4
    incidents=[e for e in s.events if e['kind']=='guard_receipt' and e.get('reason')=='invalid render shape/dtype']
    assert len(incidents)==4
    for e in incidents:
        assert e['incident']['raw_render'] in s.files
        assert e['incident']['snapshot']['state']==e['incident']['state']


def test_no_neural_simulator_imports():
    import sys
    assert not any(x in sys.modules for x in ('torch','gymnasium','pymunk','stable_worldmodel'))


def test_store_exclusive_directory_and_startup_reserve_before_create():
    path=NS(absolute=lambda:None)
    # Reject before mkdir, with no real directory created.
    class FakePath:
        parent='/MEMORY_PARENT'
        def absolute(self):return self
        def exists(self):return False
        def mkdir(self):pytest.fail('reserve gate must precede mkdir')
    with patch.object(m.builder,'Path',return_value=FakePath()),patch.object(m.shutil,'disk_usage',return_value=NS(free=m.FLOOR+m.CAP-1)),pytest.raises(ValueError):m.Store(path)


def test_verify_reconstructs_not_merely_accept_self_consistent_manifest():
    p=dict(manifest='/MEMORY/selection.json',manifest_sha256='b'*64)
    with patch.object(m,'sha',return_value='a'*64),patch.object(m,'read',return_value=p),patch.object(m,'material',return_value={**p,'changed':True}),pytest.raises(ValueError):m.verify('/MEMORY/prepare.json','a'*64)


def test_returned_state_mismatch_has_guard_receipt_before_stop():
    class Mismatch(ArrayEnv):
        def step(self,c):
            obs,*other=super().step(c)
            if self.steps==11:obs['state'][0]+=1
            return (obs,*other)
    r,s,_=execute(Mismatch)
    rejected=[e for e in s.events if e['kind']=='guard_receipt' and not e['returned_state_equal']]
    assert len(rejected)==4 and all(x['own_returned']==1 for x in r['rows'])
    assert all(e['png'] in s.files for e in rejected)


def test_bad_goal_attachment_cannot_execute_own_actions():
    attach=m.builder.attach_goal
    def bad(body,goal,rgb):
        attach(body,goal,rgb);body.state[0]+=1
    with patch.object(m.builder,'attach_goal',bad):r,s,envs=execute()
    assert all(e.steps==10 for e in envs)
    assert all(x['status']=='failed' and x['own_attempted']==0 for x in r['rows'])
    assert all('goal attachment no physics' in x['first_error']['message'] for x in r['rows'])


def test_original_native_terminal_during_warmup_stops_without_own_dispatch():
    class WarmTerminal(ArrayEnv):
        def step(self,c):
            obs,reward,terminal,truncated,info=super().step(c)
            if self.steps==3:
                self.state=self.goal_state.copy();obs=dict(state=self._get_obs());reward=0.;terminal=True
            return obs,reward,terminal,truncated,info
    r,s,envs=execute(WarmTerminal)
    assert all(e.steps==3 for e in envs)
    assert all(x['warmup_returned']==3 and x['own_returned']==0 and x['last_observation']['step']==3 for x in r['rows'])
    assert not any(e['kind']=='goal_attached' for e in s.events)
