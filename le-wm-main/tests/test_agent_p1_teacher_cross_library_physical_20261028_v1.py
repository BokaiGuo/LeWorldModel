"""RAM-only fake environment/PNG/NPZ tests; no actual prepare/model/physics."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
from PIL import Image
import pytest
from experiments.spectral_reachability import agent_p1_teacher_cross_library_physical_20261028_v1 as m
from test_agent_mixedmode_capability_physical_20261028_v1 import (
    MemoryStore as ParentStore, fixture as parent_fixture, Env, fake_place, file_store)
from test_p1_constructive_moving_history_physical_20261027_v1 import REQUEST


def h(x):return hashlib.sha256(str(x).encode()).hexdigest()


class MemoryStore(ParentStore):
    def __init__(self):super().__init__();self.free=m.CAP+m.FLOOR
    def check(self,size=4096,final=False):
        m.require(self.used+size<=m.CAP-(0 if final or self.terminal else m.FINAL) and self.free>=m.FLOOR+size,'RAM own cap/reserve/floor')


def frame(e,phase,step,k):
    e.with_target=k==1;pre=m.invariant(e);pixels=e.render();post=m.invariant(e)
    b=io.BytesIO();Image.fromarray(pixels).save(b,format='PNG')
    return dict(receipt=dict(phase=phase,step=step,condition='on' if k==1 else 'off',overlay_index=k,
        file_sha256=hashlib.sha256(b.getvalue()).hexdigest(),pixel_sha256=m.moving.image_sha(pixels),
        state=m.source.vector(e._get_obs()).tolist(),snapshot=post['snapshot']),pre=pre,post=post,
        source_png=f'/RAM/original_{phase}_{step}_{k}.png')


def raw_ref(e,a,phase,step):
    obs,*_=e.step(a.copy());return dict(phase=phase,step=step,raw_bits=a.tobytes().hex(),actual_bits=e.latest_action.tobytes().hex(),
        actual_dtype=e.latest_action.dtype.str,state=obs['state'].tolist(),snapshot=m.moving.snapshot(e))


@lru_cache(maxsize=1)
def _fixture():
    _,p,data=parent_fixture();anchors=[];references=[];teacher_inputs=[];compiler_rows=[]
    for ti,t in enumerate(p['tasks']):
        t['slot']['packet']=dict(root_context=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
            latent_history=np.full((3,192),ti,np.float32).tolist(),action_history=m.teacher.kernel.normalized_history(m.moving.warmup_commands()).tolist()),
            goal=np.zeros(192).tolist())
        t['slot']['q']=np.eye(192).tolist();t['context_digest']=m.digest(t['slot'])
        slot,condition=m.probe.ORDER[ti];root=data['roots'][slot]
        for role in m.teacher.ROLES:
            prefix=np.zeros((5,2),'<f4') if role=='Z' else m.probe.raw(t['plan']['prefix_commands'],5)
            e=Env();e.reset(seed=root['slot']['reset_seed']);e.reset(seed=0,options=dict(state=REQUEST.copy(),goal_state=e.goal_state.copy()))
            refs=[frame(e,'warmup',0,k) for k in range(3)];raw=[];e.with_target=False
            for i,a in enumerate(m.moving.warmup_commands(),1):
                raw.append(raw_ref(e,a,'warmup',i));refs.extend(frame(e,'warmup',i,k) for k in range(3));e.with_target=False
            m.moving.attach_goal(e,root['goal'],root['packets'][condition]['goal_rgb'])
            refs.extend(frame(e,'root',0,k) for k in range(3));e.with_target=False
            for i,a in enumerate(prefix,1):
                raw.append(raw_ref(e,a,'prefix',10+i));refs.extend(frame(e,'prefix',i,k) for k in range(3));e.with_target=False
            last=refs[-2 if condition=='on' else -3]
            a=dict(index=len(anchors),task=ti,root_slot=slot,condition=condition,role=role,task_id=t['slot']['task_id'],
                blind_sha256=t['slot']['blind_sha256'],task_file=t['slot']['task_file'],task_sha256=t['slot']['task_sha256'],
                root_context=t['slot']['packet']['root_context'],goal=t['slot']['packet']['goal'],q=t['slot']['q'],prefix_commands=prefix.tolist(),
                png=last['source_png'],png_sha256=last['receipt']['file_sha256'],pixel_sha256=last['receipt']['pixel_sha256'])
            a['anchor_id']=m.digest(a);anchors.append(a);references.append(dict(anchor_id=a['anchor_id'],raw15=raw,frames51=refs))
            b=m.teacher.blind_anchor(a,np.full(192,a['index'],np.float32));teacher_inputs.append(b)
            compiler_rows.append(dict(status='all_five_replayed',blind_input=b,evaluations=[]))
    for ti in range(4):
        bank=m.teacher.common_bank(np.tile(np.asarray([.05,.02],'<f4'),(5,1)),np.tile(np.asarray([.02,-.04],'<f4'),(5,1)))
        for row in compiler_rows[2*ti:2*ti+2]:row['evaluations']=[dict(raw_commands=a.tolist(),cost=float(5-i)) for i,a in enumerate(bank)]
    compiled=dict(schema=m.teacher.NAME+'.prepare',contract=m.teacher.contract(),anchors=anchors,
        ancestry=dict(probe_digest=m.probe.digest(p)))
    release=dict(schema=m.teacher.NAME+'.consume',status='teacher_cross_library_commands_verified_not_admitted',P1_admission=False,
        C2_non_greedy_premise=False,new_neural_calls=0,new_physics_calls=0,plans=m.teacher.plans_for(compiled,compiler_rows))
    package=dict(schema=m.NAME+'.prepare',contract=m.contract(),jobs=m.jobs_for(compiled,release,p),
        references=references,teacher_inputs=teacher_inputs,anchors=anchors,probe_digest=m.probe.digest(p),
        arguments=dict(compiler_output='/RAM/compiler',compiler_report_sha=m.COMPILER_REPORT_SHA,
            compiler_consume='/RAM/accepted.json',compiler_consume_sha=m.COMPILER_CONSUME_SHA),
        bindings={str(Path(m.__file__).resolve()):h('source')})
    return package,p,data,compiled,release


def fixture():return deepcopy(_fixture())


def execute(monkeypatch,*,factory=Env,store=None,started=None,package=None):
    base,p,data,_,_=fixture();package=base if package is None else package;store=MemoryStore() if store is None else store;envs=[]
    def make():e=factory();envs.append(e);return e
    monkeypatch.setattr(m,'place',fake_place);monkeypatch.setattr(m,'validate_env',lambda _:None)
    monkeypatch.setattr(m.probe,'verify_bindings',lambda _:None)
    r=m.execute(package,p,data,store,make,authorization=m.AUTH,prepared='/RAM/own.json',prepare_sha=h('prepare'),started=started)
    return r,store,envs,package,p


@lru_cache(maxsize=1)
def completed():
    with pytest.MonkeyPatch.context() as mp:return execute(mp)


def test_full40_literal_execution_and_all_resource_axes():
    r,s,envs,package,p=completed()
    assert r['status']=='complete',r['errors'][:2]
    assert len(envs)==len(r['rows'])==40
    assert r['counts']=={('internal_physics_step' if k=='physics_step' else k):dict(attempted=v,completed=v) for k,v in m.CAPS.items()}
    assert len([n for n in s.blobs if n.endswith('.png')])==2640
    assert len([n for n in s.blobs if n.endswith('.npz')])==800
    for j,row,env in zip(package['jobs'],r['rows'],envs,strict=True):
        raw=np.asarray(env.commands,np.float32)
        assert raw.tobytes()==np.concatenate((m.moving.warmup_commands(),m.probe.raw(j['commands'],10))).tobytes()
        assert len(env.resets)==2 and env.closed and env.dispatch_flags==[False]*20
        assert row['public']==dict(warmup=dict(attempted=10,completed=10),prefix=dict(attempted=5,completed=5),suffix=dict(attempted=5,completed=5))
        assert row['reference_raw_verified']==15 and row['reference_frames_verified']==51
        assert row['teacher_history_verified'] and row['initial_exact'] and row['history_exact'] and row['start_verified']
        assert row['missing_checkpoints']==[] and [cp['absolute_step'] for cp in row['checkpoints']]==[10,15,20]
        assert row['actual_latent_Q'] is None
        assert '_set_state' not in env.__dict__ and 'reset' not in env.__dict__
    assert m.zero.CAPS['public_step']==400 and m.helper.CAPS['public_step']==700


def test_full_saved_validators_and_dense_sizes():
    r,s,_,package,p=completed();m.validate_trace(package,p,r,s.events)
    m.validate_payloads(package,r,s.events,lambda n:bytes(s.blobs[n]))
    assert len(s.blobs['journal.jsonl'])<m.JOURNAL_CAP
    assert len(m.canonical(r))<m.REPORT_CAP
    assert len(m.canonical(package))<m.PREP_CAP
    assert s.used<m.CAP-m.FINAL


def test_common_bank_full_denominator_including_zero_duplicates():
    package,p,_,compiled,release=fixture()
    assert [(j['task'],j['role'],j['library_index']) for j in package['jobs']]==[(t,r,i) for t in range(4) for r in m.teacher.ROLES for i in range(5)]
    assert len({j['job_digest'] for j in package['jobs']})==40
    # Arbitrarily reverse model scores WITHOUT changing commands: no score selection.
    changed=deepcopy(release)
    for i,plan in enumerate(changed['plans']):
        plan['model_Q']=1e10-i;plan['plan_digest']=m.digest({k:v for k,v in plan.items() if k!='plan_digest'})
    alternative=m.jobs_for(compiled,changed,p)
    assert len(alternative)==40 and [j['commands'] for j in alternative]==[j['commands'] for j in package['jobs']]
    assert 'model_Q' not in package['jobs'][0]


@pytest.mark.parametrize('field',['missing','order','role','root','condition','raw','inverse','prefix','seal','admission'])
def test_release_seam_fail_closed(field):
    _,p,_,compiled,release=fixture()
    if field=='missing':release['plans'].pop()
    elif field=='order':release['plans'].reverse()
    elif field=='role':release['plans'][0]['role']='S'
    elif field=='root':release['plans'][0]['root_slot']=1
    elif field=='condition':release['plans'][0]['condition']='on'
    elif field=='raw':release['plans'][0]['commands'][9][0]=.1
    elif field=='inverse':
        plan=release['plans'][2];plan['suffix_commands']=deepcopy(plan['suffix_commands'][::-1]);plan['suffix_commands'][0][0]=float(np.float32(.1))
        plan['commands']=plan['prefix_commands']+plan['suffix_commands'];plan['plan_digest']=m.digest({k:v for k,v in plan.items() if k!='plan_digest'})
    elif field=='prefix':compiled['anchors'][0]['prefix_commands'][0][0]=float(np.float32(.5))
    elif field=='seal':release['plans'][0]['plan_digest']=h('bad')
    else:release['P1_admission']=True
    with pytest.raises((ValueError,KeyError)):m.jobs_for(compiled,release,p)


@pytest.mark.parametrize('field',['count','identity','goal','raw','return','history','snapshot','guard','checkpoint','metric','endpoint','flags'])
def test_strict_trace_corruption_rejected(field):
    original,s,_,package,p=completed();r=deepcopy(original);events=deepcopy(s.events)
    if field=='count':r['counts']['internal_physics_step']['completed']-=1
    elif field=='identity':r['rows'][0]['anchor_id']=h('bad')
    elif field=='goal':r['rows'][0]['goal'][0]+=1
    elif field=='raw':next(e for e in events if e['kind']=='operation_attempted' and e['resource']=='public_step')['fields']['raw_bits']='bad'
    elif field=='return':next(e for e in events if e['kind']=='raw_return')['actual_bits']='bad'
    elif field=='history':next(e for e in events if e['kind']=='teacher_history_verified')['tokens'][0][0]+=1
    elif field=='snapshot':next(e for e in events if e['kind']=='raw_return')['snapshot']['state'][0]+=1
    elif field=='guard':r['rows'][0]['guards']['task']['primitive_commands_authorized']=5
    elif field=='checkpoint':r['rows'][0]['checkpoints'][0]['absolute_step']=15
    elif field=='metric':r['rows'][0]['checkpoints'][0]['diagnostics']['independent']['all_xy_position_error']+=1
    elif field=='endpoint':r['rows'][0]['end_state'][0]+=1
    else:r['P1_admission']=True
    with pytest.raises((ValueError,KeyError)):m.validate_trace(package,p,r,events)


@pytest.mark.parametrize('field',['png','pixel','triplet','prefix','npz','framejoin'])
def test_all_payloads_checked(field):
    original,s,_,package,p=completed();r=deepcopy(original);events=deepcopy(s.events)
    def load(n):return bytes(s.blobs[n])+ (b'bad' if field=='png' and n.endswith('.png') else b'')
    if field=='pixel':next(e for e in events if e['kind']=='fresh_observation')['receipt']['pixel_sha256']=h('bad')
    elif field=='triplet':next(e for e in events if e['kind']=='fresh_observation' and e['receipt']['phase']=='suffix')['receipt']['overlay_index']=1
    elif field=='prefix':package=deepcopy(package);package['references'][0]['frames51'][0]['receipt']['file_sha256']=h('bad')
    elif field=='npz':next(e for e in events if e['kind']=='raw_return')['info']['sha256']=h('bad')
    elif field=='framejoin':r['rows'][0]['checkpoints'][0]['frame']['file']='bad.png'
    with pytest.raises((ValueError,KeyError)):m.validate_payloads(package,r,events,load)


@pytest.mark.parametrize('fault',['terminal','truncate','throw','substitute'])
def test_suffix_failure_or_native_stop_keeps_all40(monkeypatch,fault):
    class Incident(Env):
        def step(self,c):
            result=super().step(c)
            if fault=='substitute' and len(self.commands)==16:
                # Unlike replacing with zero, this is a REAL substitution even
                # for the mandatory zero library slot (including signed zero).
                self.latest_action=c.copy();self.latest_action[0]+=np.float32(.125)
            return result
    r,s,envs,package,p=execute(monkeypatch,factory=lambda:Incident(fault=fault,at=16))
    assert r['status']=='partial' and len(envs)==len(r['rows'])==40
    assert all(e.closed and len(e.commands)==16 for e in envs)
    assert all(row['end_state'] is None and row['missing_checkpoints']==[10] for row in r['rows'])
    assert all(row['status']==('native_stopped' if fault in ('terminal','truncate') else 'failed') for row in r['rows'])
    m.validate_trace(package,p,r,s.events)


def test_prefix_mismatch_stops_before_new_suffix(monkeypatch):
    package,*_=fixture();package['references'][0]['frames51'][-1]['receipt']['pixel_sha256']=h('wrong')
    r,s,envs,package,p=execute(monkeypatch,package=package)
    assert all(len(envs[i].commands)==15 and r['rows'][i]['status']=='failed' for i in range(5))
    assert all(row['status']=='horizon_completed' for row in r['rows'][5:])
    assert all(env.with_target is False for env in envs)


def test_guard_is_official224_and_not_state_only(monkeypatch):
    r,s,envs,package,p=execute(monkeypatch,factory=lambda:Env(fault='guard',at=16))
    assert len(envs)==40 and all(len(e.commands)==16 for e in envs)
    for row in r['rows']:
        assert row['status']=='failed' and row['error']['message']=='fresh RGB/canvas guard'
        # The earlier capture guard rejects before g.observe is reached: the
        # retained RGB guard receipt must NOT invent an observation/abstention.
        assert row['guards']['task']['abstained'] is False
        assert row['guards']['task']['primitive_commands_authorized']==6
        assert [o['primitive_step'] for o in row['guards']['task']['observations']]==list(range(6))
        scoped=[e for e in s.events if e.get('job_digest')==row['job_digest']]
        incident=[e for e in scoped if e['kind']=='fresh_observation' and e['receipt']['phase']=='suffix']
        assert len(incident)==1 and incident[0]['receipt']['png'] in s.blobs
        pixels=np.asarray(Image.open(io.BytesIO(s.blobs[incident[0]['receipt']['png']])))
        assert pixels.shape==(224,224,3) and not np.any(np.all(pixels==[65,105,225],axis=-1))
        assert not any(e['kind']=='rgb_guard_observed' and e['scope']=='task' and e['primitive_step']==6 for e in scoped)
    m.validate_trace(package,p,r,s.events)


def test_wall_keeps_all_unstarted_without_factory(monkeypatch):
    r,s,envs,package,p=execute(monkeypatch,started=time.monotonic()-m.WALL-1)
    assert not envs and r['counts']=={} and len(r['rows'])==40 and r['status']=='partial'
    assert all(row['status']=='unstarted' for row in r['rows']);m.validate_trace(package,p,r,s.events)


def test_authority_before_reads_or_output(monkeypatch):
    def forbidden(*args):pytest.fail('unauthorized work')
    monkeypatch.setattr(m,'verify',forbidden);monkeypatch.setattr(m,'Store',forbidden)
    with pytest.raises(ValueError):m.run('x','x','x',forbidden,authorization='bad')


def test_meter_actual_attempt_and_return_and_nonphysical_cleanup_only():
    s=MemoryStore();meter=m.Meter(s);meter.where=dict(job_digest=h('job'));seen=[]
    meter.call('constructor',lambda:seen.append(s.events[-1]['kind']));assert seen==['operation_attempted']
    def bad():raise RuntimeError('first callback')
    with pytest.raises(RuntimeError):meter.call('reset',bad)
    assert meter.receipt()['reset']==dict(attempted=1,completed=0)
    meter.counts['public_step']['attempted']=800
    with pytest.raises(ValueError):meter.call('public_step',lambda:pytest.fail('cap'))
    with pytest.raises(ValueError):meter.call('reset',lambda:None,cleanup=True)
    meter.fault=True;meter.call('close',lambda:None,cleanup=True)


def test_first_error_survives_final_source_hash_failure(monkeypatch):
    package,p,data,_,_=fixture();s=MemoryStore()
    monkeypatch.setattr(m.probe,'verify_bindings',lambda _:(_ for _ in ()).throw(FileNotFoundError('LATER')))
    def fail():raise RuntimeError('FIRST')
    r=m.execute(package,p,data,s,fail,authorization=m.AUTH,prepared='/RAM/own',prepare_sha=h('p'))
    assert r['errors'][0]['message']=='FIRST' and any(e['phase']=='final_source_hash' for e in r['errors'])
    assert len(r['rows'])==40 and r['status']=='partial'


def test_store_append_partial_actual_byte_account(monkeypatch):
    old,fs=file_store(monkeypatch);s=object.__new__(m.Store);s.__dict__.update(old.__dict__);s.check=lambda *a:None
    fs.fileno=lambda:91
    s.write('journal.jsonl',b'abc',append=True);s.write('journal.jsonl',b'def',append=True);assert s.used==6
    fs.partial=True
    with pytest.raises(OSError):s.write('partial',b'abcdef')
    assert s.used==9 and s.files['partial'][2]==3


def test_capacity_bounds_and_no_old_global_changes():
    r,s,_,package,p=completed()
    assert len(m.canonical(package))<m.PREP_CAP and len(m.canonical(r))<m.REPORT_CAP
    assert m.CAPS['public_step']==800 and m.CAPS['physics_step']==8160 and m.CAPS['explicit_render']==2640
    assert m.CAPS['reset']==80 and m.WALL==3600 and m.JOURNAL_CAP==128*1024**2
    assert m.zero.NAME=='agent_mixedmode_zero_time_control_20261028_v1' and m.zero.CAPS['explicit_render']==1320
    assert m.contract()['actual_latent_Q'] is None and m.contract()['C2_non_greedy_premise'] is False


@pytest.mark.parametrize('shift',[-4*np.pi,-2*np.pi,0,2*np.pi,4*np.pi])
def test_wrapped_angle_geometry_invariant_not_native_override(shift):
    goal=np.zeros(7);state=np.asarray([0,0,0,0,.1+shift,0,0.])
    d=m.pose.pose_diagnostics(goal,state)
    assert d['independent']['wrapped_angle_error']==pytest.approx(.1)
    assert d['independent']['all_xy_position_error']==0
    if abs(shift)>2*np.pi:assert d['native_formula']['negative_angle_term'] is True


def test_cli_help_and_run_authority(capsys):
    with pytest.raises(SystemExit) as e:m.main(['--help'])
    assert e.value.code==0 and 'consume' in capsys.readouterr().out
    with pytest.raises(ValueError):m.main(['run','--prepared','x','--prepare-sha','x','--output','x'])
