"""Paired moving-history rendering, MAIN-only bounded physical producer.

Own construction/replay kernels. Imports only sealed utilities,
never calls old episode/execute or changes module globals. Off remains native
runtime semantics; on exists only during render. No neural consumer or training.
"""
from __future__ import annotations
import argparse
import base64
from contextlib import contextmanager
from copy import deepcopy
import gzip
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import p1_constructive_moving_history_source_20261027_v1 as moving
from experiments.spectral_reachability import p1_constructive_moving_history_physical_20261027_v1 as previous

NAME='p1_moving_target_overlay_source_20261027_v1'
ROOT=moving.ROOT
require,sha,raw,digest=moving.require,previous.sha,moving.raw,moving.source.digest
CAP,FLOOR,FINAL=8*1024**2,200*1024**2,512*1024
AUTH='MAIN_MOVING_TARGET_OVERLAY_PHYSICS_ONLY'
SD=ROOT/'results/p1_constructive_moving_history_source_20261027_v1_main01'
SP=ROOT/'results/p1_constructive_moving_history_source_20261027_v1_prepare01.json'
PINNED={str(SD/'sourcepack.json'):'c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6',
    str(SP):'c1292f0ebfb1827e5da92b32ab13d2e4a12c025c4741277ab4c4f478d6ee8cb9',
    str(Path(moving.__file__).resolve()):'9fb76ecfa054f878ab9778fc29c59bc708728b240dd2cbc3060b166e52622f55',
    str(Path(previous.__file__).resolve()):'44d90599eab881c773d5efa09177dfe6b395fa3f4eef41070340681fea5ec96e'}


class Store(moving.Store):
    """Same exclusive 8MiB store, larger explicit final-report reservation."""
    def write(self,name,value,append=False,final=False):
        require(Path(name).name==name and name not in ('','.','..'),'flat output artifact')
        used=sum(f.stat().st_size for f in self.path.iterdir() if f.is_file())
        require(used+len(value)<=CAP-(0 if final else FINAL),'own8MiB incl512KiB final reserve')
        return super().write(name,value,append=append,final=final)


def flat(directory,name):
    require(type(name) is str and Path(name).name==name and name not in ('','.','..'),'flat sealed reference')
    return Path(directory)/name


def read(path):
    path=Path(path);require(path.stat().st_size<2*1024**2,'bounded metadata');return previous.read(path)


def seal(bindings,path,expected=None):
    path=str(Path(path).resolve());h=sha(path)
    if expected is not None:require(h==expected,'external source seal '+path)
    require(path not in bindings or bindings[path]==h,'nonconflicting source identity');bindings[path]=h
    return path


def inputs():
    """Saved evaluator data only. Privileged data never enters blind NPZ."""
    bindings={}
    for path,h in PINNED.items():seal(bindings,path,h)
    data=previous.load_source(SD/'sourcepack.json',PINNED[str(SD/'sourcepack.json')],SP,PINNED[str(SP)])
    sp=data['plan'];bindings.update(data['bindings'])
    se=previous.journal(SD/'journal.jsonl',data['pack']['journal_chain_tail']);seal(bindings,SD/'journal.jsonl')
    for directory,prepared in ((SD,SP),):
        path=seal(bindings,directory/'execution_binding.json');b=read(path)
        require(b==dict(prepared=str(prepared),prepare_file_sha256=PINNED[str(prepared)]),'old execution prepare binding')
    references={};witnesses=[]
    for i,ref in enumerate(data['references']):
        require(ref['slot']==sp['slots'][i] and ref['slot']['index']==i,'fixed source slots')
        logits,commands=moving.witness(ref['states'][-1],ref['slot']['sign'],sp['packet_logit_cap'])
        witnesses.append(commands.copy())
        for phase in ('construction','replay0','replay1'):
            key=f's{i}_{phase}';obs=[e for e in se if e['kind']=='fresh_observation' and e.get('slot')==i and e['phase']==phase]
            require([e['step'] for e in obs]==list(range(21)),'all21 original source frames')
            warm=[e for e in se if e['kind']=='warmup_return' and e.get('slot')==i and e['phase']==phase]
            require([e['raw_float32_bits'] for e in warm]==[a.tobytes().hex() for a in moving.warmup_commands()], 'original10 actual raw warmup')
            cp=seal(bindings,SD/f'{key}_commit.json');commit=read(cp)
            require(moving.exact(np.asarray(commit['commands'],np.float32),commands) and
                moving.exact(np.asarray(commit['logits'],np.float32),logits) and commit['root_snapshot']==obs[10]['snapshot'], 'original precommitted witness per phase')
            refs={}
            for e in obs:
                path=seal(bindings,flat(SD,e['png']),e['file_sha256'])
                refs[str(e['step'])]=dict(path=path,file_sha256=e['file_sha256'],pixel_sha256=e['pixel_sha256'],state=e['state'],snapshot=e['snapshot'])
            references[key]=refs
        row=data['pack']['rows'][i];seal(bindings,flat(SD,row['certificate']),row['certificate_sha256']) # hash only
    for path,h in bindings.items():require(sha(path)==h,'saved source/recipe/plan final seal')
    return dict(source_plan=sp,refs=data['references'],witnesses=witnesses,references=references,bindings=bindings)


def contract():
    return dict(slots=2,source_phases=['construction','replay0','replay1'],
        warmup=10,horizon=10,source_triplets=[0,5,10,15,20],flags=[False,True,False],
        caps=dict(reset=12,public_step=120,physics_step=1224,set_state=24,goal_render=12,goal_setter=16,goal_bind=4,explicit_render=186),
        wall_seconds=600,output_cap_bytes=CAP,final_reserve_bytes=FINAL,disk_floor_bytes=FLOOR,
        with_target_default=False,no_goal_pose_edit=True,no_postreset_state_setter=True,
        new_neural_calls=0,training=False,P1_admission=False,new_independent_roots=0,
        semantics='same old tasks/actions; renderer default goal_pose is NOT constructed task goal; no effect claim')


def files():
    return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def build():
    data=inputs();bindings=deepcopy(data['bindings']);bindings.update({str(p):sha(p) for p in files()})
    return dict(schema=NAME+'.prepare',contract=contract(),bindings=bindings,
        old_inputs=deepcopy(PINNED),old_source_prepare_id=digest(data['source_plan']),slots=deepcopy(data['source_plan']['slots']),
        reference_index_sha256=digest(data['references']),
        command_authority='old sealed warmup/witness only; zero model or outcome selection')


def verify(p):
    require(p['schema']==NAME+'.prepare' and p['contract']==contract(),'own paired overlay prepare')
    for f,h in p['bindings'].items():require(sha(f)==h,'prepare/source seal drift '+f)
    data=inputs();bs=deepcopy(data['bindings']);bs.update({str(f):sha(f) for f in files()})
    expected=dict(schema=NAME+'.prepare',contract=contract(),bindings=bs,old_inputs=deepcopy(PINNED),
        old_source_prepare_id=digest(data['source_plan']),slots=deepcopy(data['source_plan']['slots']),
        reference_index_sha256=digest(data['references']),
        command_authority='old sealed warmup/witness only; zero model or outcome selection')
    require(p==expected,'complete source/reference/contract reconstruction');return data


def prepare(path):
    p=build();blob=raw(dict(schema=NAME+'.envelope',plan_sha256=digest(p),
        gzip_base64=base64.b64encode(gzip.compress(raw(p),mtime=0)).decode()))
    path=Path(path).absolute();require(not path.exists() and len(blob)<200_000 and shutil.disk_usage(path.parent).free>=FLOOR+len(blob)+4096,'exclusive small prepare +200MiB')
    with path.open('xb') as f:require(f.write(blob)==len(blob),'short prepare');f.flush();os.fsync(f.fileno())
    fsync_dir(path.parent);return dict(prepared=str(path),prepare_file_sha256=sha(path),bytes=len(blob))


def fsync_dir(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def load_prepared(path,expected):
    require(sha(path)==expected and Path(path).stat().st_size<200_000,'external own prepare SHA')
    e=read(path);require(set(e)=={'schema','plan_sha256','gzip_base64'} and e['schema']==NAME+'.envelope','own envelope only')
    with gzip.GzipFile(fileobj=BytesIO(base64.b64decode(e['gzip_base64'],validate=True))) as f:b=f.read(2*1024**2+1)
    require(len(b)<=2*1024**2,'bounded prepare expansion');p=json.loads(b)
    require(digest(p)==e['plan_sha256'] and sha(path)==expected,'prepare semantic SHA/TOCTOU');return p


def literal(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:literal(a) for k,a in v.items()}
    if isinstance(v,(tuple,list)):return [literal(a) for a in v]
    require(v is None or type(v) in (bool,int,float,str),'literal renderer metadata');return v


def invariant(body):
    pose=np.asarray(body.goal_pose,float);require(pose.shape==(3,) and np.isfinite(pose).all(),'native goal_pose3')
    return dict(snapshot=moving.snapshot(body),goal_state=moving.source.vector(body.goal_state).tolist(),goal_pose=pose.tolist(),
        goal_rgb_sha256=moving.image_sha(body._goal),renderer=dict(render_mode=body.render_mode,render_size=body.render_size,
        window_size=body.window_size,render_action=body.render_action,relative=body.relative,action_scale=body.action_scale,
        dt=body.dt,control_hz=body.control_hz,variations=literal(body.variation_space.value)))


def old_image(ref):
    from PIL import Image
    p=Path(ref['path']);require(sha(p)==ref['file_sha256'] and p.stat().st_size<=1024**2,'old PNG file seal')
    with Image.open(p) as im:
        require(im.mode=='RGB' and im.format=='PNG' and im.size==(224,224),'unaltered old PNG')
        a=np.asarray(im).copy()
    require(moving.image_sha(a)==ref['pixel_sha256'] and sha(p)==ref['file_sha256'],'old PNG pixels/TOCTOU');return a


class StopNative(Exception):
    """A legal native stop, not an engineering exception or retry permission."""


class Meter(moving.Ops):
    def __init__(self,store):super().__init__(contract(),store);self.where={};self.secondary=[]
    def call(self,kind,fn,**fields):
        fields={**self.where,**fields};require(time.monotonic()-self.started<self.p['wall_seconds'],'wall cap')
        c=self.counts[kind];require(c['attempted']<self.p['caps'][kind],'predispatch '+kind+' cap')
        c['attempted']+=1;self.store.emit('operation_attempted',resource=kind,**fields)
        try:value=fn()
        except BaseException as exc:
            try:self.store.emit('operation_failed',resource=kind,message=str(exc),**fields)
            except BaseException as e:self.secondary.append(dict(phase='operation_error_journal',type=type(e).__name__,message=str(e)))
            raise
        c['completed']+=1;self.store.emit('operation_completed',resource=kind,**fields);return value


@contextmanager
def native_hooks(env,meter,errors):
    body=env.unwrapped;undo=[];spaces=[];resetting=False;default=body.with_target
    old_set,old_goal,old_render,old_reset=body._set_state,body._set_goal_state,body.render,env.reset
    def patch(obj,key,new):undo.append((obj,key,key in obj.__dict__,getattr(obj,key)));setattr(obj,key,new)
    def install():
        space=body.space
        if any(s[0] is space for s in spaces):return
        old=space.step;spaces.append((space,'step' in space.__dict__,old))
        space.step=lambda *a,**kw:meter.call('physics_step',lambda:old(*a,**kw))
    def setter(v):
        require(resetting,'no post-reset state setter');install();return meter.call('set_state',lambda:old_set(v))
    def goalsetter(v):
        before=moving.snapshot(body);pose=np.asarray(body.goal_pose).copy()
        result=meter.call('goal_setter',lambda:old_goal(v))
        require(moving.snapshot(body)==before and moving.exact(np.asarray(body.goal_pose),pose),'goal setter changed body/goal_pose');return result
    def render():return meter.call('goal_render' if resetting else 'explicit_render',old_render)
    def reset(*a,**kw):
        nonlocal resetting
        require(body.with_target is False,'native default off at reset');resetting=True
        try:return old_reset(*a,**kw)
        finally:resetting=False
    try:
        for obj,key,new in ((body,'_set_state',setter),(body,'_set_goal_state',goalsetter),(body,'render',render),(env,'reset',reset)):patch(obj,key,new)
        yield
    finally:
        try:body.with_target=default
        except BaseException as e:errors.append(dict(phase='restore_flag',type=type(e).__name__,message=str(e)))
        for obj,owned,old in reversed(spaces):
            try:
                if owned:obj.step=old
                else:delattr(obj,'step')
            except BaseException as e:errors.append(dict(phase='restore_space',type=type(e).__name__,message=str(e)))
        for obj,key,owned,old in reversed(undo):
            try:
                if owned:setattr(obj,key,old)
                else:delattr(obj,key)
            except BaseException as e:errors.append(dict(phase='restore_hook',type=type(e).__name__,message=str(e)))


def capture(env,obs,key,reference,triplet,row,meter,store):
    """Every fresh image/receipt durable BEFORE any exactness/guard rejection."""
    body=env.unwrapped;fixed=invariant(body);first=None;result={}
    flags=(False,True,False) if triplet else (False,)
    try:
        for index,flag in enumerate(flags):
            body.with_target=flag;before=invariant(body);a=moving.image(env.render());after=invariant(body)
            name=f'{row["key"]}_{key.replace(":","_")}_{index}_{"on" if flag else "off"}.png';h=store.frame(name,a)
            checks=None;reference_error=None
            if not flag:
                try:
                    old=old_image(reference);checks=dict(file=h==reference['file_sha256'],pixels=moving.exact(a,old),
                        state=moving.exact(np.asarray(after['snapshot']['state']),np.asarray(reference['state'])),snapshot=after['snapshot']==reference['snapshot'])
                except BaseException as e:reference_error=e
            replay=moving.exact(first,a) if index==2 else None
            receipt=dict(reference_key=key,overlay_index=index,with_target=flag,png=name,file_sha256=h,
                pixel_sha256=moving.image_sha(a),pre=before,post=after,old_off_match=checks,off_replay_match=replay,
                reference_error=None if reference_error is None else dict(type=type(reference_error).__name__,message=str(reference_error)))
            row['frames'].append(dict(reference_key=key,overlay_index=index,png=name,file_sha256=h,pixel_sha256=moving.image_sha(a)))
            store.emit('fresh_overlay_frame',**meter.where,receipt=receipt)
            if reference_error is not None:raise reference_error
            require(before==after==fixed and body.with_target is flag and moving.exact(np.asarray(obs['state']),np.asarray(body._get_obs())),
                'render/flag changed full body/goal/renderer or observation')
            if checks is not None:require(all(checks.values()),'old off RGB/state/snapshot/file identity mismatch')
            moving.guard(a,obs['state'])
            if index==0:first=a.copy();result['off']=a.copy()
            if index==1:result['on']=a.copy()
            if index==2:require(replay,'off/on/off replay differs')
    finally:body.with_target=False
    if triplet:row['missing_triplets'].remove(key)
    return result


def place(env,plan,slot,meter,store):
    body=env.unwrapped
    _,info=meter.call('reset',lambda:env.reset(seed=slot['reset_seed']),role='seed',seed=slot['reset_seed'])
    goal=moving.source.vector(info['goal_state']).copy()
    requested=moving.source.sample_state(goal,list(body.block.shapes),dict(root_seed=slot['root_seed']),0,0,plan)
    store.emit('placement_recipe',**meter.where,slot=slot,native_goal=goal.tolist(),requested_state=requested.tolist())
    obs,info=meter.call('reset',lambda:env.reset(seed=0,options=dict(state=requested.copy(),goal_state=goal.copy())),role='placement',seed=0)
    return obs,goal


def bind_goal(body,goal,rgb,meter,store):
    before=invariant(body);counts=deepcopy(meter.counts)
    meter.call('goal_bind',lambda:moving.attach_goal(body,goal,rgb))
    after=invariant(body)
    require(before['snapshot']==after['snapshot'] and before['goal_pose']==after['goal_pose'] and before['renderer']==after['renderer'] and
        all(v==meter.counts[k] for k,v in counts.items() if k not in ('goal_bind','goal_setter')) and
        np.array_equal(body.goal_state,goal) and moving.exact(body._goal,rgb),'offline task attachment changed dynamics/renderer/pose')
    store.emit('goal_attached',**meter.where,goal=goal.tolist(),goal_rgb_sha256=moving.image_sha(rgb),snapshot=after['snapshot'],goal_pose=after['goal_pose'])


def step_raw(env,command,phase,step,row,meter,store):
    require(command.dtype==np.float32 and command.shape==(2,) and env.unwrapped.with_target is False,'literal raw2/native-off control')
    obs,reward,terminated,truncated,info=meter.call('public_step',lambda:env.step(command.copy()),phase=phase,step=step,raw_bits=command.tobytes().hex())
    row['public_returned']+=1;actual=np.asarray(env.unwrapped.latest_action).copy()
    store.emit('native_return',**meter.where,phase=phase,step=step,requested_raw=command.tolist(),requested_bits=command.tobytes().hex(),
        actual_raw=actual.tolist(),actual_dtype=actual.dtype.str,actual_bits=actual.tobytes().hex(),state=moving.source.vector(obs['state']).tolist(),
        snapshot=moving.snapshot(env.unwrapped),reward=float(reward),terminated=bool(terminated),truncated=bool(truncated))
    return obs,info,bool(terminated),bool(truncated),float(reward),actual


def check_return(env,command,actual,info,goal,reward,terminated):
    require(moving.exact(command,actual),'actual raw action substitution')
    require(np.array_equal(info['goal_state'],goal),'native task goal changed')
    success,distance=env.unwrapped.eval_state(goal,moving.source.vector(env.unwrapped._get_obs()))
    require(bool(terminated)==bool(success) and reward==-float(distance),'native reward/terminal mismatch')
    return bool(success)


def source_episode(env,data,index,phase,row,meter,store,construction=None):
    plan=data['source_plan'];slot=plan['slots'][index];refs=data['references'][row['key']];body=env.unwrapped
    obs,native_goal=place(env,plan,slot,meter,store);history={'off':[],'on':[]};warm=moving.warmup_commands()
    pair=capture(env,obs,'0',refs['0'],True,row,meter,store)
    for flag in history:history[flag].append(pair[flag])
    for k,command in enumerate(warm,1):
        if body.eval_state(native_goal,moving.source.vector(obs['state']))[0]:raise StopNative('warmup root already native terminal')
        obs,info,t,tr,reward,actual=step_raw(env,command,'warmup',k,row,meter,store)
        pair=capture(env,obs,str(k),refs[str(k)],k in (5,10),row,meter,store)
        check_return(env,command,actual,info,native_goal,reward,t)
        if t or tr:raise StopNative('warmup stopped; no witness or payload padding')
        if k in (5,10):
            for flag in history:history[flag].append(pair[flag])
    root=moving.source.vector(obs['state']);logits,commands=moving.witness(root,slot['sign'],plan['packet_logit_cap'])
    require(moving.exact(commands,data['witnesses'][index]),'same presealed witness at exact old root')
    store.put(row['key']+'_commands.json',dict(schema=NAME+'.witness_commit',commands=commands.tolist(),logits=logits.tolist(),root_snapshot=moving.snapshot(body)))
    goal=native_goal
    if construction is not None:
        require(moving.exact(np.asarray(history['off']),construction['history']['off']) and moving.exact(np.asarray(history['on']),construction['history']['on']) and
            moving.exact(root,construction['root']), 'independent history/root reconstruction')
        goal=construction['goal'];bind_goal(body,goal,construction['goal_rgb']['off'],meter,store)
        if body.eval_state(goal,root)[0]:raise StopNative('constructed goal already native terminal at root')
    reached=False;future=[];snapshots=[]
    for k,command in enumerate(commands,1):
        obs,info,t,tr,reward,actual=step_raw(env,command,'witness',k,row,meter,store)
        pair=capture(env,obs,str(10+k),refs[str(10+k)],k in (5,10),row,meter,store)
        success=check_return(env,command,actual,info,goal,reward,t)
        state=moving.source.vector(obs['state']);future.append(state.copy());snapshots.append(moving.snapshot(body))
        if construction is not None:
            require(moving.exact(state,construction['future'][k-1]) and snapshots[-1]==construction['dynamics'][k-1] and
                moving.exact(info['goal'],construction['goal_rgb']['off']),'independent witness dynamics/task goal')
        if t or tr:
            if construction is None or tr or not success or k!=10:raise StopNative('witness stopped before required full10 terminal')
            reached=True;break
    if construction is not None:require(reached,'same fixed witness did not return native success at full10')
    result=dict(root=root,goal=moving.source.vector(obs['state']),history={k:np.asarray(v) for k,v in history.items()},
        goal_rgb=pair,future=np.asarray(future),dynamics=snapshots)
    oldref=data['refs'][index]
    require(moving.exact(result['goal'],oldref['goal']) and moving.exact(result['history']['off'],oldref['history']) and
        moving.exact(result['goal_rgb']['off'],oldref['goal_rgb']) and moving.source.root_hash(root,result['goal'])==oldref['root_sha256'],
        'same old source task root/history/goal identity, not independent roots')
    if construction is not None:
        require(moving.exact(result['goal'],construction['goal']) and all(moving.exact(result['goal_rgb'][f],construction['goal_rgb'][f]) for f in ('off','on')),
            'true terminal RGB pair matches construction, never synthesized')
    row.update(status='source_episode_completed',native_success=reached,goal_pose=invariant(body)['goal_pose'],root_goal_sha256=moving.source.root_hash(root,result['goal']))
    return result


def write_blind(store,index,construction,replay0,replay1):
    artifacts={}
    for flag in ('off','on'):
        history=replay0['history'][flag];goal=construction['goal_rgb'][flag]
        require(moving.exact(history,replay1['history'][flag]) and moving.exact(goal,replay1['goal_rgb'][flag]),'both independent replay RGBs')
        name=f's{index}_{flag}_blind_inputs.npz';h=store.arrays(name,rgb_history=history,raw_action_history=moving.warmup_commands(),goal_rgb=goal)
        artifacts[flag]=dict(path=name,sha256=h,fields=['rgb_history','raw_action_history','goal_rgb'],
            rgb_history_pixel_sha256=[moving.image_sha(a) for a in history],goal_pixel_sha256=moving.image_sha(goal),
            raw_action_sha256=moving.image_sha(moving.warmup_commands()))
    return artifacts


def execute(p,store,factory,*,authorization):
    require(authorization==AUTH,'explicit MAIN paired moving overlay physics authorization')
    rows=[dict(key=f's{s}_{phase}',source_slot=s,kind='source',phase=phase,status='not_started',public_returned=0,frames=[],missing_triplets=['0','5','10','15','20'])
        for s in range(2) for phase in ('construction','replay0','replay1')]
    meter=Meter(store);errors=[];exports=[];source_ok=False;data=None
    def error(phase,e):errors.append(dict(phase=phase,type=type(e).__name__,message=str(e)))
    try:
        data=verify(p);store.emit('run_started',prepare_id=digest(p),contract=contract())
        for s in range(2):
            results=[]
            for j,phase in enumerate(('construction','replay0','replay1')):
                row=rows[3*s+j];env=None;meter.where=dict(key=row['key'],source_slot=s)
                try:
                    env=factory();moving.validate_env(env.unwrapped)
                    require(env.unwrapped.render_action is False and env.unwrapped.render_size==224 and env.unwrapped.render_mode=='rgb_array','original native off renderer')
                    row['status']='running'
                    with native_hooks(env,meter,errors):r=source_episode(env,data,s,phase,row,meter,store,results[0] if j else None)
                    results.append(r)
                except StopNative as e:row.update(status='native_stopped',reason=str(e))
                except BaseException as e:error('source_episode',e);row.update(status='engineering_failed')
                finally:
                    if env is not None:
                        try:env.close()
                        except BaseException as e:error('close',e)
                    store.emit('branch_finished',receipt=row)
                if len(results)!=j+1:break
            if len(results)==3:
                exports.append(dict(source_slot=s,artifacts=write_blind(store,s,*results),old_root_sha256=data['refs'][s]['root_sha256']))
    except BaseException as e:error('execution',e)
    finally:
        try:verify(p);source_ok=True
        except BaseException as e:error('final_source',e)
        errors.extend(meter.secondary)
        if not errors:
            try:
                c={k:v['completed'] for k,v in meter.counts.items()}
                require(all(v['attempted']==v['completed'] for v in meter.counts.values()) and c['physics_step']==2*c['reset']+10*c['public_step'] and
                    c['set_state']==2*c['reset'] and c['goal_render']==c['reset'] and c['goal_setter']==c['reset']+c['goal_bind'] and
                    c['public_step']==sum(r['public_returned'] for r in rows) and c['explicit_render']==sum(len(r['frames']) for r in rows), 'actual returned resource identities')
                if all(r['status']=='source_episode_completed' for r in rows):require(c==contract()['caps'],'complete exact full caps')
            except BaseException as e:error('final_census',e)
        complete=not errors and len(exports)==2 and all(r['status']=='source_episode_completed' for r in rows)
        status='paired_moving_overlay_recorded' if complete else 'engineering_failed' if errors else 'partial_native_stop'
        try:store.emit('run_finished',status=status,counts=meter.counts,errors=errors)
        except BaseException as e:error('final_journal',e);status='engineering_failed';complete=False
        result=dict(schema=NAME+'.report',status=status,prepare_id=digest(p),contract=contract(),rows=rows,blind_exports=exports,
            fixed_denominator=2,expected_episodes=6,expected_frames=186,
            counts=deepcopy(meter.counts),errors=errors,first_error=deepcopy(errors[0]) if errors else None,
            journal_chain_tail=store.previous,source_verified=source_ok,full_protocol=complete,new_neural_calls=0,
            new_independent_roots=0,P1_admission=False,scientific_winner=None,wall_seconds=time.monotonic()-meter.started)
        try:store.put('report.json',result,final=True)
        except BaseException as e:error('final_report',e);result.update(status='engineering_failed',full_protocol=False,first_error=deepcopy(errors[0]))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','run'))
    for k in ('prepared','prepare-sha','output'):parser.add_argument('--'+k)
    parser.add_argument('--authorize-physics',action='store_true');a=parser.parse_args()
    if a.command=='prepare':result=prepare(a.output)
    else:
        p=load_prepared(a.prepared,a.prepare_sha)
        if a.command=='verify':verify(p);result=dict(status='verified_no_execution',contract=p['contract'])
        else:
            require(a.authorize_physics,'MAIN explicit CLI authorization')
            def factory():
                import gymnasium as gym
                import stable_worldmodel  # noqa: F401
                return gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224,with_target=False)
            store=Store(a.output);store.put('execution_binding.json',dict(prepared=str(Path(a.prepared).resolve()),prepare_file_sha256=a.prepare_sha))
            result=execute(p,store,factory,authorization=AUTH)
    print(moving.source.canonical({k:v for k,v in result.items() if k!='rows'}))
    if result.get('status')=='engineering_failed':raise SystemExit(1)


if __name__=='__main__':main()
