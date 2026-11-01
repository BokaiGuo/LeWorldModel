"""RAM mock/metadata only. No actual prepare, neural runtime, or physics."""
from copy import deepcopy
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace as NS
import sys

import numpy as np
from PIL import Image
import pytest

from experiments.spectral_reachability import p1_moving_overlay_recompile_physical_20261027_v1 as m
from test_p1_moving_target_overlay_plans_physical_20261027_v1 import Env
from test_p1_constructive_moving_history_physical_20261027_v1 import REQUEST


def h(value): return hashlib.sha256(str(value).encode()).hexdigest()


def png(value):
    out=io.BytesIO(); Image.fromarray(value).save(out,format='PNG'); return out.getvalue()


class MemoryStore(m.Store):
    """All bytes and the real JSON hashchain remain in RAM, never a directory."""
    def __init__(self):
        self.path=Path('/RAM_ONLY'); self.files={}; self.previous=None; self.index=0
        self.events=[]; self.free=m.FLOOR+m.CAP; self.used=0; self.hashers={}

    def check(self,size=4096,final=False):
        m.require(self.used+size<=m.CAP-(0 if final else m.FINAL) and self.free>=m.FLOOR+size,'RAM mocked cap/floor')

    def write(self,name,value,append=False,final=False):
        self.check(len(value),final)
        assert append or name not in self.files
        self.files[name]=self.files.get(name,b'')+value if append else value
        self.used+=len(value)
        self.hashers.setdefault(name,hashlib.sha256()).update(value)
        return self.hashers[name].hexdigest()

    def emit(self,kind,**fields):
        offset=len(self.files.get('journal.jsonl',b''))
        super().emit(kind,**fields)
        self.events.append(__import__('json').loads(self.files['journal.jsonl'][offset:]))


def fixture():
    roots=[]; exports={}; images={}
    for s in (0,1):
        e=Env(); e.reset(seed=21+s)
        obs,_=e.reset(seed=0,options=dict(state=REQUEST.copy(),goal_state=e.goal_state.copy()))
        frames={'off':[],'on':[]}; warm={}
        for step in range(11):
            if step: obs,*_=e.step(m.moving.warmup_commands()[step-1].copy())
            e.with_target=False; off=e.render(); e.with_target=True; on=e.render(); e.with_target=False
            name=f'/RAM/source_s{s}_{step}.png'; images[name]=off.copy()
            warm[step]=dict(path=name,file_sha256=hashlib.sha256(png(off)).hexdigest(),pixel_sha256=m.moving.image_sha(off),
                state=obs['state'].tolist(),snapshot=m.moving.snapshot(e))
            if step in (0,5,10):
                frames['off'].append(off.copy()); frames['on'].append(on.copy())
        goal=np.array([400.,400.,350.,350.,0.,0.,0.])
        goal_off=np.full((224,224,3),31,np.uint8); goal_on=goal_off.copy(); goal_on[2,2]=[0,255,0]
        packets={c:dict(rgb_history=np.asarray(frames[c]),raw_action_history=m.moving.warmup_commands(),
                        goal_rgb=goal_off.copy() if c=='off' else goal_on.copy()) for c in m.CONDITIONS}
        roots.append(dict(slot=dict(index=s,reset_seed=21+s,root_seed=11+s),goal=goal,
                          root_sha256=m.source.root_hash(obs['state'],goal),packets=packets,warm=warm))
        for c in m.CONDITIONS: exports[s,c]=dict(path=f'/RAM/s{s}_{c}_blind_inputs.npz',sha256=h(('blind',s,c)))
    planner=h('SYNTHETIC planner report, not a real SHA authority')
    plans=[dict(root_slot=s,condition=c,arm=a,commands=np.full((10,2),(.25 if a==m.ARMS[0] else -.25)*(1 if c=='off' else -1),np.float32).tolist(),
                selection_digest=h((s,c,a)),task_id=h(('task',s,c)),blind_path=exports[s,c]['path'],blind_sha256=exports[s,c]['sha256'],
                source_report_sha256=m.SOURCE_SHA,planner_report_sha256=planner)
           for s in (0,1) for c in m.CONDITIONS for a in m.ARMS]
    p=dict(schema=m.NAME+'.prepare',contract=m.contract(),plans=plans,bindings={},planner_report_sha256=planner,source_report_sha256=m.SOURCE_SHA)
    data=dict(source_plan={},roots=roots,exports=exports,bindings={})
    return p,data,images


def fake_place(env,plan,slot,meter,store):
    _,info=meter.call('reset',lambda:env.reset(seed=slot['reset_seed']))
    goal=info['goal_state'].copy()
    obs,_=meter.call('reset',lambda:env.reset(seed=0,options=dict(state=REQUEST.copy(),goal_state=goal.copy())))
    return obs,goal


def execute(monkeypatch,factory=Env,*,p=None,data=None,images=None,store=None,started=None):
    fp,fd,fi=fixture(); p=fp if p is None else p; data=fd if data is None else data; images=fi if images is None else images
    store=MemoryStore() if store is None else store; envs=[]
    def make():
        env=factory(); envs.append(env); return env
    def fake_sha(path):
        name=Path(path).name
        assert name in store.files
        return hashlib.sha256(store.files[name]).hexdigest()
    monkeypatch.setattr(m,'place',fake_place)
    monkeypatch.setattr(m,'validate_env',lambda body:None)
    monkeypatch.setattr(m,'old_image',lambda ref:images[ref['path']].copy())
    monkeypatch.setattr(m,'sha',fake_sha)
    binding=dict(prepared='/RAM/prepare',prepare_sha256=h('SYNTHETIC prepare'),planner_report_sha256=p['planner_report_sha256'],source_report_sha256=m.SOURCE_SHA)
    result=m.execute(p,data,store,make,authorization=m.AUTH,prepare_sha256=binding['prepare_sha256'],execution_binding=binding,started=started)
    return result,store,envs


def test_full8_newplans_metered_native_off_and_real7_labels(monkeypatch):
    p,data,_=fixture(); r,s,envs=execute(monkeypatch)
    assert r['status']=='moving_overlay_recompile_physical_recorded',r['errors']
    assert r['fixed_denominator']==len(r['rows'])==len(envs)==8
    assert r['counts']=={k:dict(attempted=v,completed=v) for k,v in m.contract()['caps'].items()}
    assert r['new_neural_calls']==0 and not r['P1_admission']
    assert len([n for n in s.files if n.endswith('.png')])==528
    assert len({id(e) for e in envs})==8
    for row,plan,env in zip(r['rows'],p['plans'],envs,strict=True):
        assert row['status']=='horizon_completed' and row['initial_exact'] and row['missing_checkpoints']==[]
        assert row['own_returned']==row['own_attempted']==row['warmup_returned']==10
        assert env.closed and env.with_target is False and env.dispatch_flags==[False]*20
        assert env.on_calls==22 and m.moving.exact(np.asarray(env.commands[:10]),m.moving.warmup_commands())
        assert m.moving.exact(np.asarray(env.commands[10:]),m.previous.commands(plan['commands']))
        assert m.moving.exact(env._goal,data['roots'][plan['root_slot']]['packets'][plan['condition']]['goal_rgb'])
        assert env.goal_pose.tolist()==[300.,300.,0.] and '_set_state' not in env.__dict__ and 'reset' not in env.__dict__
        assert len(row['trajectory'])==22
        for cp in row['checkpoints']:
            assert set(cp)=={'step','png','file_sha256','pixel_sha256','state','snapshot'} and len(cp['state'])==7
            image=np.asarray(Image.open(io.BytesIO(s.files[cp['png']])))
            assert image[2,2].tolist()==([0,255,0] if row['condition']=='on' else image[3,3].tolist())
            assert hashlib.sha256(s.files[cp['png']]).hexdigest()==cp['file_sha256']
    previous=None
    for event in s.events:
        assert event['previous_sha256']==previous and event['event_sha256']==m.source.digest({k:v for k,v in event.items() if k!='event_sha256'})
        previous=event['event_sha256']
    assert r['journal_chain_tail']==previous
    assert r['journal_sha256']==hashlib.sha256(s.files['journal.jsonl']).hexdigest()
    assert r['execution_binding_sha256']==hashlib.sha256(s.files['execution_binding.json']).hexdigest()


@pytest.mark.parametrize('field',['file_sha256','pixel_sha256','state','snapshot'])
def test_off_root_identity_mismatch_persists_actual_image_and_no_own_actions(monkeypatch,field):
    p,data,images=fixture(); ref=data['roots'][0]['warm'][10]
    if field=='state': ref[field][0]+=1e-8
    elif field=='snapshot': ref[field]['state_float64_bits']='NOT_SAME'
    else: ref[field]=h('NOT_SAME')
    r,s,envs=execute(monkeypatch,p=p,data=data,images=images)
    assert len(r['rows'])==8 and all(row['status']=='engineering_failed' for row in r['rows'][:4])
    assert all(row['own_returned']==0 and row['last_observation']['png'] in s.files for row in r['rows'][:4])
    assert all(len(e.commands)==10 for e in envs[:4])
    assert all(row['status']=='horizon_completed' for row in r['rows'][4:])


def test_on_history_packet_mismatch_is_not_fixed_by_using_off_packet(monkeypatch):
    p,data,images=fixture(); data['roots'][0]['packets']['on']['rgb_history'][1,3,3,0]+=1
    r,_,envs=execute(monkeypatch,p=p,data=data,images=images)
    assert all(r['rows'][i]['own_returned']==0 and len(envs[i].commands)==10 for i in (2,3))
    assert r['rows'][0]['status']=='horizon_completed'


@pytest.mark.parametrize('fault',['truncate','terminal'])
def test_native_stop_never_pads_missing5_10_or_changes8_denominator(monkeypatch,fault):
    r,_,envs=execute(monkeypatch,lambda:Env(fault=fault,at=11))
    assert r['status']=='moving_overlay_recompile_physical_recorded' and len(r['rows'])==8
    for row,env in zip(r['rows'],envs,strict=True):
        assert row['missing_checkpoints']==[5,10] and [c['step'] for c in row['checkpoints']]==[0]
        assert row['own_returned']==1 and len(env.commands)==11 and env.closed
        assert row['native_success'] is (fault=='terminal') and len(row['final_state'])==7
    assert r['counts']['public_step']==dict(attempted=88,completed=88)
    assert r['counts']['explicit_render']==dict(attempted=312,completed=312)


@pytest.mark.parametrize('fault',['substitute','guard','render','throw'])
def test_incident_no_action_repair_or_next_dispatch_and_first_error_retained(monkeypatch,fault):
    r,s,envs=execute(monkeypatch,lambda:Env(fault=fault,at=11,close_error=True))
    assert r['status']=='partial_or_failed' and len(r['rows'])==8
    assert all(len(env.commands)==11 and env.closed for env in envs)
    for row in r['rows']:
        assert row['first_error'] and row['secondary_errors'] and row['missing_checkpoints']==[5,10]
        assert 'secondary fake close' in str(row['secondary_errors'])
        assert row['last_observation']['png'] in s.files
    if fault=='throw': assert r['counts']['public_step']==dict(attempted=88,completed=80)


@pytest.mark.parametrize('fault',['guard','mutation','pose','goal','config','invalid','throw','physics'])
def test_temporary_on_render_failure_restores_off_before_fresh_next_arm(monkeypatch,fault):
    r,s,envs=execute(monkeypatch,lambda:Env(overlay_fault=fault))
    assert r['status']=='partial_or_failed' and len(envs)==8
    assert all(env.with_target is False and env.closed and not env.commands for env in envs)
    assert all(row['status']=='engineering_failed' and row['missing_checkpoints']==[0,5,10] for row in r['rows'])
    assert not any(e['resource']=='public_step' for e in s.events if e['kind']=='operation_attempted')


@pytest.mark.parametrize('field', sorted(m.PLAN_KEYS))
def test_incomplete_handoff_rejected_before_factory(field):
    p,data,_=fixture(); del p['plans'][0][field]
    with pytest.raises(ValueError): m.validate_plans(p['plans'],p['planner_report_sha256'],data['exports'])


@pytest.mark.parametrize('mutation',[
    lambda p:p.reverse(),lambda p:p.pop(),lambda p:p[0].update(root_slot=True),
    lambda p:p[0].update(condition='on'),lambda p:p[0].update(blind_path='relative.npz'),
    lambda p:p[0].update(blind_sha256=h('other packet')),lambda p:p[0].update(planner_report_sha256=h('future fake')),
    lambda p:p[0].update(source_report_sha256=h('old report')),lambda p:p[0].update(commands=np.zeros((5,2)).tolist()),
    lambda p:p[0].update(commands=np.full((10,2),.1).tolist()),
    lambda p:p[0].update(selection_uses_physics=True),lambda p:p[0].update(P1_admission=True),
])
def test_order_identity_exactfloat32_and_no_physics_selection(mutation):
    p,data,_=fixture(); mutation(p['plans'])
    with pytest.raises(ValueError): m.validate_plans(p['plans'],p['planner_report_sha256'],data['exports'])


def test_richer_Nash_metadata_projected_without_recomputing_selection_digest():
    p,data,_=fixture(); rich=deepcopy(p['plans'])
    for plan in rich:
        plan.update(schema='p1_moving_overlay_recompile_cuda_20261027_v1.plan',with_target=plan['condition']=='on',
                    selection_uses_physics=False,P1_admission=False,logits='SYNTHETIC saved metadata',prediction={},model_terminal_cost=1.)
    assert m.validate_plans(rich,p['planner_report_sha256'],data['exports'])==p['plans']


@pytest.mark.parametrize('stage',['branch_closed','run_finished','journal_hash','binding_hash','report'])
def test_independent_finalization_and_full_denominator(monkeypatch,stage):
    store=MemoryStore(); emit,put=store.emit,store.put
    def fail_emit(kind,**fields):
        if kind==stage: raise OSError('INJECTED '+stage)
        return emit(kind,**fields)
    def fail_put(name,value,final=False):
        if stage=='report' and name=='report.json': raise OSError('INJECTED report')
        return put(name,value,final)
    store.emit=fail_emit; store.put=fail_put
    if stage in ('journal_hash','binding_hash'):
        original=hashlib.sha256
        target='journal.jsonl' if stage=='journal_hash' else 'execution_binding.json'
        # Our execute helper's own-module sha callback is wrapped, never old globals.
        p,data,images=fixture(); envs=[]
        monkeypatch.setattr(m,'place',fake_place); monkeypatch.setattr(m,'validate_env',lambda _:None)
        monkeypatch.setattr(m,'old_image',lambda ref:images[ref['path']])
        def hash_fail(path):
            if Path(path).name==target: raise OSError('INJECTED '+stage)
            return original(store.files[Path(path).name]).hexdigest()
        monkeypatch.setattr(m,'sha',hash_fail)
        binding=dict(prepared='/RAM/p',prepare_sha256=h('prepare'),planner_report_sha256=p['planner_report_sha256'],source_report_sha256=m.SOURCE_SHA)
        r=m.execute(p,data,store,Env,authorization=m.AUTH,prepare_sha256=binding['prepare_sha256'],execution_binding=binding)
    else: r,_,envs=execute(monkeypatch,store=store)
    assert r['status']=='partial_or_failed' and len(r['rows'])==8
    if stage!='report': assert 'report.json' in store.files
    if stage=='branch_closed': assert len(envs)==1 and all(row['status']=='not_started' for row in r['rows'][1:])


def test_caps_wall_disk_floor_before_dispatch(monkeypatch):
    store=MemoryStore(); ops=m.Meter(store); ops.counts['public_step']['attempted']=160
    with pytest.raises(ValueError): ops.call('public_step',lambda:pytest.fail('dispatch'))
    ops.started-=901
    with pytest.raises(ValueError): ops.call('reset',lambda:pytest.fail('dispatch'))
    store.free=m.FLOOR-1; ops=m.Meter(store)
    with pytest.raises(ValueError): ops.call('constructor',lambda:pytest.fail('constructor'))
    assert ops.counts['constructor']==dict(attempted=0,completed=0)
    prod=object.__new__(m.Store); prod.path=NS(iterdir=lambda:[])
    monkeypatch.setattr(m.shutil,'disk_usage',lambda _:NS(free=m.FLOOR-1))
    with pytest.raises(ValueError): prod.check(1)


def test_no_authority_no_prepare_read_or_environment(monkeypatch):
    with pytest.raises(ValueError): m.run('/NO','x','/NO',lambda:pytest.fail('factory'),authorization='not_MAIN')


def test_protocol_readonly_binding_and_saved_material_all8_consume(monkeypatch):
    p,data,_=fixture(); called=[]
    protocol=m.ROOT.parent/'docs/D2P_MOVING_OVERLAY_RECOMPILE_PROTOCOL_20261027.md'
    assert protocol in m.own_files() and protocol.is_file()
    monkeypatch.setattr(m,'source_material',lambda:data)
    monkeypatch.setattr(m,'sha',lambda path:p['planner_report_sha256'] if Path(path).name=='report.json' else h(path))
    def consume(output,report_sha):
        called.append((output,report_sha)); return deepcopy(p['plans'])
    prepared,returned=m.material('/RAM/compiler',p['planner_report_sha256'],consume)
    assert returned is data and len(prepared['plans'])==8
    assert called==[(Path('/RAM/compiler'),p['planner_report_sha256'])]
    assert str(protocol) in prepared['bindings']
    assert prepared['compiler_handoff_sha256']==m.source.digest(p['plans'])
    with pytest.raises(ValueError):
        m.material('/RAM/compiler',p['planner_report_sha256'],lambda *_:p['plans'][:7])
    monkeypatch.setattr(m,'sha',lambda _:h('wrong actual report'))
    with pytest.raises(ValueError): m.material('/RAM/compiler',p['planner_report_sha256'],consume)
    assert len(called)==1


def test_verify_reconstructs_entire_material_without_prepare_or_physics(monkeypatch):
    p,data,_=fixture(); p['planner_output']='/RAM/compiler'; called=[]
    expected=h('RAM prepare only')
    monkeypatch.setattr(m,'sha',lambda _:expected)
    monkeypatch.setattr(m,'read',lambda _:deepcopy(p))
    def material(output,report_sha):
        called.append((output,report_sha)); return deepcopy(p),data
    monkeypatch.setattr(m,'material',material)
    assert m.verify('/RAM/prepare',expected)==(p,data)
    assert called==[(p['planner_output'],p['planner_report_sha256'])]
    changed=deepcopy(p); changed['plans'][0]['commands'][0][0]=.5
    monkeypatch.setattr(m,'material',lambda *_:(changed,data))
    with pytest.raises(ValueError): m.verify('/RAM/prepare',expected)


def test_cleanup_still_closes_once_with_wall_disk_and_journal_exhausted():
    store=MemoryStore(); meter=m.Meter(store); meter.started-=901; store.free=0
    calls=[]
    meter.call('close',lambda:calls.append('closed'),cleanup=True)
    assert calls==['closed'] and meter.counts['close']==dict(attempted=1,completed=1)
    assert len(meter.secondary)==2
    with pytest.raises(ValueError): meter.call('public_step',lambda:None,cleanup=True)


def test_native_stop_hook_restore_error_is_not_successful(monkeypatch):
    from contextlib import contextmanager
    original=m.native_hooks
    @contextmanager
    def faulty_hooks(env,meter,errors):
        try:
            with original(env,meter,errors): yield
        finally: errors.append(dict(stage='restore_hook',type='OSError',message='SYNTHETIC restore failure'))
    monkeypatch.setattr(m,'native_hooks',faulty_hooks)
    def stop(*args): raise m.NativeStop('SYNTHETIC native stop before dispatch')
    monkeypatch.setattr(m,'episode',stop)
    r,_,envs=execute(monkeypatch)
    assert r['status']=='partial_or_failed' and all(e.closed for e in envs)
    assert all(row['status']=='engineering_failed' and row['first_error']['stage']=='cleanup' for row in r['rows'])


def test_finalization_wall_limit_is_failure_without_zero_fill(monkeypatch):
    r,_,_=execute(monkeypatch,started=m.time.monotonic()-901)
    assert r['status']=='partial_or_failed' and any(e['stage']=='final_wall' for e in r['errors'])
    assert all(row['native_success'] is None and row['checkpoints']==[] for row in r['rows'])


def test_no_neural_or_native_runtime_loaded_and_no_old_execution_call():
    assert not any(name in sys.modules for name in ('torch','gymnasium','pymunk','stable_worldmodel'))
    import ast
    tree=ast.parse(Path(m.__file__).read_text())
    forbidden={'execute','branch','episode','witness','source_episode'}
    assert not [n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                and n.func.attr in forbidden]
