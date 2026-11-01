"""RAM-only artificial array replay; no simulator, model, prepare or disk tmp."""
from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import sys
import numpy as np
import pytest

from experiments.spectral_reachability import p1_moving_target_overlay_plans_physical_20261027_v1 as m
from test_p1_constructive_moving_history_physical_20261027_v1 import ArrayEnv, fixture as moving_fixture, REQUEST
from test_p1_constructive_plan_physical_20261027_v1 import MemoryStore


class Env(ArrayEnv):
    """Explicit numeric mock: no learned/native computation or hidden physics."""
    def __init__(self, *, overlay_fault=None, **kwargs):
        super().__init__(**kwargs)
        self.window_size=512; self.variation_space=NS(value={'rendering':{'render_goal':True},'color':[0,255,0]})
        self.overlay_fault=overlay_fault; self.on_calls=0; self.dispatch_flags=[]

    def render(self):
        a=super().render()
        if self.with_target:
            self.on_calls+=1; a[2,2]=[0,255,0]
            if self.overlay_fault == 'guard': a[0,0]=0
            if self.overlay_fault == 'mutation': self.agent.velocity=(1.,0.)
            if self.overlay_fault == 'pose': self.goal_pose[0]+=1
            if self.overlay_fault == 'goal': self.goal_state[0]+=1
            if self.overlay_fault == 'config': self.render_action=True
            if self.overlay_fault == 'throw': raise RuntimeError('on render primary failure')
            if self.overlay_fault == 'invalid': return np.zeros((3,4),np.float64)
            if self.overlay_fault == 'physics': self.space.step(.01)
        return a

    def step(self,c):
        self.dispatch_flags.append(self.with_target)
        return super().step(c)


def reference_saved(plans, data):
    """Independent artificial old off trace, not a call to any producer kernel."""
    saved=[]
    for i,plan in enumerate(plans):
        env=Env(); env.reset(seed=21+i//2)
        obs,_=env.reset(seed=0,options=dict(state=REQUEST.copy(),goal_state=env.goal_state.copy()))
        off={}; transitions={}
        def observe(phase,step):
            a=env.render(); h=hashlib.sha256(a.tobytes()).hexdigest()
            return dict(phase=phase,step=step,png=f'b{i}_{phase}_{step:02}.png',file_sha256=h,
                pixel_sha256=h,state=m.source.vector(env._get_obs()).tolist(),snapshot=m.moving.snapshot(env),
                observation_state=np.asarray(obs['state']).tolist())
        off[('warmup',0)]=observe('warmup',0)
        for step,c in enumerate(m.moving.warmup_commands(),1):
            obs,reward,terminal,truncated,_=env.step(c.copy())
            off[('warmup',step)]=observe('warmup',step)
            transitions[('warmup',step)]=dict(phase='warmup',step=step,state=obs['state'].tolist(),reward=reward,terminated=terminal,truncated=truncated)
        r=data['references'][i//2]; m.moving.attach_goal(env,r['goal'],r['goal_rgb'])
        off[('plan',0)]=observe('plan',0)
        for step,c in enumerate(m.low.commands(plan['commands']),1):
            obs,reward,terminal,truncated,_=env.step(c.copy())
            off[('plan',step)]=observe('plan',step)
            transitions[('plan',step)]=dict(phase='plan',step=step,command=c.tolist(),state=obs['state'].tolist(),reward=reward,terminated=terminal,truncated=truncated)
        saved.append(dict(off=off,transitions=transitions))
    return saved


def fixture():
    old,data=moving_fixture(); plans=deepcopy(old['plans'])
    for i,plan in enumerate(plans): plan['arm']=m.ARMS[i%2]
    p=dict(schema=m.NAME+'.prepare',contract=m.contract(),bindings={},source=dict(plans=plans),fixed_denominator=4)
    return p,data,reference_saved(plans,data)


def execute(factory=Env, *, p=None, data=None, saved=None, store=None):
    fp,fd,fs=fixture(); p=fp if p is None else p; data=fd if data is None else data; saved=fs if saved is None else saved
    store=MemoryStore() if store is None else store; envs=[]
    def make():
        env=factory(); envs.append(env); return env
    def fake_sha(path):
        assert Path(path).name == 'journal.jsonl'
        return m.source.digest(store.events)
    forbidden=AssertionError('old execution/witness/episode forbidden')
    with patch.object(m.moving,'validate_env',lambda _:None),patch.object(m.source,'sample_state',lambda *a:REQUEST.copy()),\
        patch.object(m,'sha',fake_sha),patch.object(m.prior,'execute',side_effect=forbidden),\
        patch.object(m.prior,'branch',side_effect=forbidden),patch.object(m.low,'execute',side_effect=forbidden),\
        patch.object(m.low,'branch',side_effect=forbidden),patch.object(m.moving,'execute',side_effect=forbidden),\
        patch.object(m.moving,'episode',side_effect=forbidden),patch.object(m.moving,'witness',side_effect=forbidden):
        result=m.execute(p,data,saved,store,make,authorization=m.AUTH)
    return result,store,envs


def test_full112_not124_exact_internal_census_and_fixed4():
    r,s,envs=execute()
    assert r['status'] == 'moving_target_overlay_plans_recorded',r
    expected=dict(reset=8,public_step=80,physics_step=816,explicit_render=112,goal_render=8,set_state=16,goal_bind=4)
    assert r['counts'] == {k:dict(attempted=v,completed=v) for k,v in expected.items()}
    assert r['fixed_denominator'] == len(r['rows']) == 4 and r['full_horizon_count'] == 4 and r['complete_triplets'] == 12
    assert r['new_neural_calls'] == r['new_independent_samples'] == 0 and not r['P1_admission'] and not r['source_goal_on_certified']
    assert len([n for n in s.files if n.endswith('.png')]) == 112
    for i,(row,env) in enumerate(zip(r['rows'],envs,strict=True)):
        assert row['off_verified'] == 22 and row['PNG'] == 28 and row['missing_checkpoints'] == []
        assert env.closed and env.with_target is False and env.on_calls == 3
        assert env.dispatch_flags == [False]*20 and np.array_equal(env.goal_pose,[300,300,0])
        assert m.moving.exact(np.asarray(env.commands[:10]),m.moving.warmup_commands())
        assert m.moving.exact(np.asarray(env.commands[10:]),m.low.commands(fixture()[0]['source']['plans'][i]['commands']))
        assert 'reset' not in env.__dict__ and '_set_state' not in env.__dict__
        for c in row['checkpoints']:
            assert c['complete'] and [c[k]['mode'] for k in ('off_before','on','off_after')] == ['off','on','off_after']
            a,b,d=(c[k] for k in ('off_before','on','off_after'))
            assert a['before'] == a['after'] == b['before'] == b['after'] == d['after']
            assert a['file_sha256'] == d['file_sha256'] and a['pixel_sha256'] != b['pixel_sha256']
            assert all(a['old_off_match'].values()) and all(d['old_off_match'].values())
            events=[e for e in s.events if e['kind']=='fresh_observation' and e.get('branch')==i and e['receipt']['phase']=='plan' and e['receipt']['step']==c['step']]
            assert len(events) == 3  # first existing off reused, not rendered twice
    previous=None
    for e in s.events:
        assert e['previous_sha256'] == previous
        assert e['event_sha256'] == m.source.digest({k:v for k,v in e.items() if k!='event_sha256'})
        previous=e['event_sha256']
    assert r['journal_chain_tail'] == previous and r['inputs_final_verified'] and 'report.json' in s.files


@pytest.mark.parametrize('phase,step',[('warmup',0),('warmup',5),('warmup',10),('plan',0),('plan',4),('plan',10)])
@pytest.mark.parametrize('field',['file_sha256','pixel_sha256','state','snapshot'])
def test_any_original_off_mismatch_failclosed_after_actual_frame_persisted(phase,step,field):
    p,d,saved=fixture(); target=saved[0]['off'][(phase,step)]
    if field == 'state': target[field][0]+=1e-8
    elif field == 'snapshot': target[field]['state_float64_bits']='changed'
    else: target[field]='f'*64
    r,s,envs=execute(p=p,data=d,saved=saved)
    row=r['rows'][0]; assert row['status']=='failed' and r['fixed_denominator']==len(r['rows'])==4
    assert row['last_observation']['phase']==phase and row['last_observation']['step']==step
    assert row['last_observation']['png'] in s.files and envs[0].with_target is False
    guards=[e for e in s.events if e['kind']=='guard_receipt' and e.get('branch')==0]
    assert guards[-1]['accepted'] is False and any(e['kind']=='fresh_observation' and e['index']<guards[-1]['index'] and
        e.get('branch')==0 and e['receipt']['png']==guards[-1]['png'] for e in s.events)
    assert envs[0].commands.__len__() == step+(10 if phase=='plan' else 0)


@pytest.mark.parametrize('fault',['guard','mutation','pose','goal','config','invalid','throw','physics'])
def test_on_failure_first_incident_retained_flag_restored_no_own_dispatch(fault):
    r,s,envs=execute(lambda:Env(overlay_fault=fault,close_error=True))
    assert r['status']=='partial_or_failed' and len(r['rows'])==4
    for i,(row,e) in enumerate(zip(r['rows'],envs,strict=True)):
        assert row['first_error'] is not None and row['own_attempted']==row['own_returned']==0
        assert 'secondary fake close' in str(row['secondary_errors']) and e.with_target is False and e.closed
        assert row['missing_checkpoints']==[0,5,10] and not row['checkpoints'][0]['complete']
        if fault not in ('throw','physics'):
            receipt=row['last_observation']; assert receipt.get('png',receipt.get('raw_render')) in s.files
            obs=[x for x in s.events if x['kind']=='fresh_observation' and x.get('branch')==i][-1]
            guard=[x for x in s.events if x['kind']=='guard_receipt' and x.get('branch')==i][-1]
            assert obs['index']<guard['index'] and not guard['accepted']


def test_native_early_stop_reduces_actual_operations_never_four_arm_denominator():
    r,s,envs=execute(lambda:Env(fault='truncate',at=11))
    assert r['status']=='moving_target_overlay_plans_recorded',r
    assert r['fixed_denominator']==len(r['rows'])==4 and r['full_horizon_count']==0 and r['complete_triplets']==4
    assert r['counts']['public_step']==dict(attempted=44,completed=44)
    assert r['counts']['physics_step']==dict(attempted=456,completed=456)
    assert r['counts']['explicit_render']==dict(attempted=60,completed=60)
    assert all(row['missing_checkpoints']==[5,10] and row['status']=='native_stopped' and not row['full_horizon'] for row in r['rows'])
    assert all(len(e.commands)==11 for e in envs)
    assert not any('_plan_05_' in name or '_plan_10_' in name for name in s.files)


def test_internal_step_exception_has_no_fabricated_postfault_frame_and_first_error_survives():
    r,s,envs=execute(lambda:Env(fault='throw',at=11,close_error=True))
    assert r['status']=='partial_or_failed' and r['fixed_denominator']==4
    assert r['counts']['public_step']==dict(attempted=44,completed=40)
    for row in r['rows']:
        assert 'primary step' in row['first_error']['message'] and 'secondary fake close' in str(row['secondary_errors'])
        assert row['last_observation']['step']==0 and row['last_observation']['mode']=='off_after'
        assert row['own_attempted']==1 and row['own_returned']==0
    assert 'do not guarantee' in r['incident_scope']


@pytest.mark.parametrize('fault',['substitute','guard','render'])
def test_own_dispatch_guard_and_render_faults_stop_before_next_command(fault):
    r,s,envs=execute(lambda:Env(fault=fault,at=11))
    assert r['status']=='partial_or_failed' and all(len(e.commands)==11 for e in envs)
    assert all(row['last_observation']['png'] in s.files for row in r['rows'])


@pytest.mark.parametrize('stage',['branch_closed','run_finished','final_hash','final_report'])
def test_durable_finalization_attempts_independent_and_never_claim_success(stage):
    s=MemoryStore(); emit,put=s.emit,s.put
    def fail_emit(kind,**kw):
        if kind==stage: raise OSError(stage+' injected')
        return emit(kind,**kw)
    def fail_put(name,value,final=False):
        if final and stage=='final_report': raise OSError('final_report injected')
        return put(name,value,final=final)
    s.emit=fail_emit; s.put=fail_put
    if stage=='final_hash':
        def hash_failure(*a): raise OSError('final hash injected')
        with patch.object(m.source,'digest',wraps=m.source.digest):
            # execute() helper supplies journal hash; fail its events serialization only at final read.
            original_execute=m.execute
            def wrapped(*a,**kw):
                with patch.object(m,'sha',side_effect=hash_failure): return original_execute(*a,**kw)
            with patch.object(m,'execute',wrapped): r,_,envs=execute(store=s)
    else: r,_,envs=execute(store=s)
    assert r['status']=='partial_or_failed' and len(r['rows'])==4
    if stage!='final_report': assert 'report.json' in s.files
    if stage=='branch_closed': assert len(envs)==1 and [x['status'] for x in r['rows'][1:]]==['not_started']*3


def test_authorization_contract_before_factory_and_operation_caps_wall():
    p,d,saved=fixture(); s=MemoryStore()
    with pytest.raises(ValueError): m.execute(p,d,saved,s,lambda:pytest.fail('factory'),authorization='wrong')
    with pytest.raises(ValueError): m.run('no','no','no',lambda:pytest.fail('factory'),authorization='wrong')
    bad=deepcopy(p); bad['contract']['caps']['explicit_render']=124
    with pytest.raises(ValueError): m.execute(bad,d,saved,s,lambda:pytest.fail('factory'),authorization=m.AUTH)
    ops=m.Ops(p,s); p['contract']['caps']['public_step']=0
    with pytest.raises(ValueError): ops.call('public_step',lambda:pytest.fail('dispatch'))
    ops.started-=1000
    with pytest.raises(ValueError): ops.call('reset',lambda:pytest.fail('dispatch'))
    store=object.__new__(m.Store); store.path=NS(iterdir=lambda:[])
    with patch.object(m.low.shutil,'disk_usage',return_value=NS(free=m.FLOOR-1)),pytest.raises(ValueError): store.check(1)


def test_literal_goalpose_default_outline_does_not_become_task_goal():
    r,_,_=execute()
    for row in r['rows']:
        assert row['goal_pose']==[300,300,0] and row['goal'][2:5]!=row['goal_pose']
    assert 'NOT constructed task goal' in r['contract']['on_semantics']
    assert not r['source_goal_on_certified']


def saved_events_fixture():
    p,d,saved=fixture(); oldp=dict(plans=p['source']['plans'],sourcepack_sha256=m.SOURCEPACK_SHA)
    report=dict(schema=m.prior.NAME+'.report',status='moving_history_physical_recorded',errors=[],sourcepack_sha256=m.SOURCEPACK_SHA,
        counts={},rows=[])
    events=[dict(kind='run_started',prepare=oldp)]
    for i,(plan,trace) in enumerate(zip(oldp['plans'],saved,strict=True)):
        report['rows'].append(dict(root_slot=plan['root_slot'],arm=plan['arm'],selection_digest=plan['selection_digest'],
            status='horizon_completed',full_horizon=True,initial_exact=True,first_error=None,secondary_errors=[],
            own_returned=10,own_attempted=10,warmup_returned=10,PNG=22,checkpoints=[trace['off'][('plan',s)] for s in (0,5,10)]))
        events.append(dict(kind='branch_started',branch=i,plan=plan))
        for ph in ('warmup','plan'):
            for step in range(11):
                events.append(dict(kind='fresh_observation',branch=i,receipt=trace['off'][(ph,step)]))
                if step:
                    events.append(dict(kind='returned_transition',branch=i,**trace['transitions'][(ph,step)]))
                    if ph=='warmup': events.append(dict(kind='warmup_return',branch=i,step=step,raw_float32_bits=m.moving.warmup_commands()[step-1].tobytes().hex()))
    events.append(dict(kind='run_finished',counts={},errors=[]))
    return oldp,report,events


def test_collect_saved_strict_moving_identity_all88_and_raw_actions():
    p,r,e=saved_events_fixture()
    traces=reference_saved(p['plans'],fixture()[1])
    def pixels(path,expected):
        branch=int(Path(path).name[1]); parts=Path(path).stem.split('_'); phase=parts[1]; step=int(parts[2])
        trace=traces[branch]['off'][(phase,step)]
        # Reference renderer depends on x; reconstruct all off pixels from saved state.
        a=np.full((224,224,3),int(trace['state'][0])%200,np.uint8); a[0,0]=[65,105,225]
        assert m.moving.image_sha(a)==expected; return a
    with patch.object(m,'png',pixels):
        saved,bindings=m.collect_saved(p,r,e,Path('/RAM'))
    assert len(saved)==4 and len(bindings)==88
    for mutate in ('zero','missing','command','slot'):
        badr,bade=deepcopy(r),deepcopy(e)
        if mutate=='zero': badr['sourcepack_sha256']='9'*64
        elif mutate=='missing': bade.remove(next(x for x in bade if x['kind']=='fresh_observation'))
        elif mutate=='command':
            x=next(x for x in bade if x['kind']=='returned_transition' and x['phase']=='plan'); x['command']=[0.,0.]
        else: badr['rows'][0]['root_slot']=1
        with patch.object(m,'png',pixels),pytest.raises(ValueError): m.collect_saved(p,badr,bade,Path('/RAM'))


def test_build_ram_no_execution_and_three_file_closure():
    value=dict(plans=[],bindings={'old':'a'*64})
    with patch.object(m,'inputs',return_value=(value,None,None)),patch.object(m,'sha',return_value='b'*64):
        p=m.build()
    assert p['schema']==m.NAME+'.prepare' and p['fixed_denominator']==4 and len(p['bindings'])==4
    assert p['source_goal_on_authority'].startswith('external SHA bridge') and not p['P1_admission']
    with patch.object(m,'sha',return_value='c'*64),patch.object(m,'read',return_value=p),patch.object(m,'build',return_value=p):
        assert m.verify('/RAM','c'*64)==p
        with pytest.raises(ValueError): m.verify('/RAM','d'*64)


def test_no_native_neural_imports_or_cpu_fallback():
    assert not any(x in sys.modules for x in ('torch','gymnasium','pymunk','stable_worldmodel'))
