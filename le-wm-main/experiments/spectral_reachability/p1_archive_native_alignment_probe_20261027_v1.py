"""Fixed saved archive -> native action-time feasibility probe, MAIN physics only.

Ordinals1/6229 x lag0/1, no winner/label shift/retry/scientific admission.
Prepare/verify read only selected H5 slices; mock tests never import a simulator.
"""
from __future__ import annotations
import argparse
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import p1_constructive_moving_history_source_20261027_v1 as base

NAME='p1_archive_native_alignment_probe_20261027_v1'
ROOT=base.ROOT
DATA=Path('/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5')
ORDINALS=(1,6229)
CHECKPOINTS=(0,5,10,15,20)
ROWS,EPISODES=2336736,18685
CAP,FLOOR,FINAL=8*1024**2,200*1024**2,128*1024
AUTH='MAIN_ARCHIVE_NATIVE_ALIGNMENT_PHYSICS_ONLY'
ENV,ENV_SHA=base.ENV,base.ENV_SHA
require,raw,sha,digest=base.require,base.raw,base.source.file_sha,base.source.digest
Store=base.Store


def array_id(a):
    a=np.asarray(a)
    return dict(shape=list(a.shape),dtype=a.dtype.str,sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())


def file_stat(path):
    p=Path(path);require(p.is_file() and not p.is_symlink(),'regular nonsymlink source')
    s=p.stat()
    return dict(path=str(p.resolve()),device=s.st_dev,inode=s.st_ino,size_bytes=s.st_size,mtime_ns=s.st_mtime_ns)


def selected_episode(f,index):
    """No outcome screening. Short fixed slot retained, not replaced."""
    require(type(index) is int and index in ORDINALS,'fixed episode ordinal')
    off,n=int(f['ep_offset'][index]),int(f['ep_len'][index])
    nxt=int(f['ep_offset'][index+1])
    require(n>=0 and off>=0 and off+n==nxt<=ROWS and
        int(f['ep_offset'][index-1])+int(f['ep_len'][index-1])==off,'exact neighboring episode boundaries')
    record=dict(episode_ordinal=index,offset=off,length=n,next_offset=nxt,
        boundary_metadata={k:array_id(np.asarray(f[k][index-1:index+2])) for k in ('ep_offset','ep_len')})
    if n<22:
        record.update(status='rejected_short_fixed_slot',reason='length<22; both lag branches retained without execution')
        return record,None
    ids=np.asarray(f['episode_idx'][off:off+22]);steps=np.asarray(f['step_idx'][off:off+22])
    require(ids.shape==steps.shape==(22,) and np.array_equal(ids,np.full(22,index)) and
        np.array_equal(steps,np.arange(22)),'selected episode/time IDs, no crossings')
    actions=np.asarray(f['action'][off:off+21]);states=np.asarray(f['state'][off:off+21])
    pixels=np.asarray(f['pixels'][[off+k for k in CHECKPOINTS]])
    require(actions.dtype==np.float32 and actions.shape==(21,2) and np.isfinite(actions).all(),
        'literal finite raw actions; no tanh/clip/ball rejection')
    require(states.dtype.kind=='f' and states.shape==(21,7) and np.isfinite(states).all(),'finite literal archive state7')
    require(pixels.dtype==np.uint8 and pixels.shape==(5,224,224,3),'unaltered checkpoint RGB224')
    record.update(status='selected',episode_idx=array_id(ids),step_idx=array_id(steps),
        action_slice=dict(local=[0,21],global_slice=[off,off+21],array=array_id(actions)),
        state_slice=dict(local=[0,21],global_slice=[off,off+21],array=array_id(states)),
        pixels=[dict(step=k,global_row=off+k,array=array_id(pixels[i])) for i,k in enumerate(CHECKPOINTS)],
        raw_action_min=actions.min(axis=0).astype(float).tolist(),raw_action_max=actions.max(axis=0).astype(float).tolist())
    return record,dict(actions=actions,states=states,pixels=pixels)


def read_archive():
    """Read-only H5, bounded selected slices; no whole-file hash/large copy."""
    import h5py
    import hdf5plugin  # noqa: F401 -- register installed archive pixel codec; no install/network
    before=file_stat(DATA)
    with h5py.File(DATA,'r') as f:
        require(len(f.attrs)==0,'known archive attrs{}; collection render config remains unknown')
        required={'action','state','pixels','proprio','ep_len','ep_offset','episode_idx','step_idx'}
        require(set(f.keys())==required,'known local dataset fields')
        schema={k:dict(shape=list(f[k].shape),dtype=f[k].dtype.str) for k in sorted(required)}
        require(f['action'].shape==(ROWS,2) and f['action'].dtype==np.float32 and
            f['state'].shape==(ROWS,7) and f['state'].dtype.kind=='f' and
            f['pixels'].shape==(ROWS,224,224,3) and f['pixels'].dtype==np.uint8 and
            f['proprio'].shape==(ROWS,4) and f['proprio'].dtype==np.float32,'fixed archive layout')
        for k in ('ep_len','ep_offset','episode_idx','step_idx'):
            require(f[k].dtype.kind in 'iu' and f[k].shape==((EPISODES,) if k in ('ep_len','ep_offset') else (ROWS,)),
                'fixed integer time metadata '+k)
        selected=[selected_episode(f,index) for index in ORDINALS]
    require(file_stat(DATA)==before,'H5 stat changed while reading selected slices')
    return dict(file_stat=before,attrs={},schema=schema,episodes=[r for r,_ in selected],
        identity_scope='stat version + selected slice SHA, NOT whole-file SHA or certified dataset revision'),[v for _,v in selected]


def contract():
    return dict(ordinals=list(ORDINALS),lags=[0,1],branches=4,checkpoints=list(CHECKPOINTS),
        horizon=20,min_episode_length=22,seed_reset=0,placement_reset_seed=0,
        environment='swm/PushT-v1',resolution=224,with_target=False,render_action=False,
        relative=True,action_scale=100,dt=.01,control_hz=10,wall_seconds=300,
        caps=dict(reset=8,public_step=80,physics_step=816,explicit_render=84,
                  goal_render=8,set_state=16,goal_setter=8),
        goal_setter_basis='native reset lines279-281: one _set_goal_state per reset; no external goal attachment',
        output_cap_bytes=CAP,final_reserve_bytes=FINAL,disk_floor_bytes=FLOOR,
        comparison='observed0 vs archive[0]; after kth actual command vs archive[k], both lags same labels',
        raw='literal float32 archive action[lag:lag+20], no /100/tanh/clip/recentering',
        stop='honor returned terminated/truncated, no padding; never move labels or select lag',
        unknown=['block hidden velocities','collector render/target/action configuration','training membership/global freshness'],
        P1_admission=False,causal_claim=False,scientific_winner=False,neural=0,torch=0,GPU=0,training=False)


def closure():
    import h5py
    import hdf5plugin
    pending=[Path(__file__),Path(base.__file__),ENV,ENV.parent.parent/'utils.py',
        Path(h5py.__file__),Path(hdf5plugin.__file__),
        ROOT/'.venv/lib/python3.12/site-packages/stable_worldmodel/spaces.py',
        ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]
    seen=set()
    while pending:
        path=pending.pop().resolve()
        if path in seen:continue
        seen.add(path)
        if path.suffix!='.py' or not path.is_relative_to(ROOT/'experiments'):continue
        for node in ast.walk(ast.parse(path.read_text())):
            names=[]
            if isinstance(node,ast.Import):names=[v.name for v in node.names]
            elif isinstance(node,ast.ImportFrom) and node.module:
                names=[node.module]+[node.module+'.'+v.name for v in node.names]
            for name in names:
                child=ROOT/Path(*name.split('.')).with_suffix('.py')
                if name.startswith('experiments.') and child.is_file():pending.append(child)
    return {str(p):sha(p) for p in sorted(seen)}


def make_plan(archive):
    bindings=closure();require(bindings[str(ENV)]==ENV_SHA,'sealed native environment')
    return dict(schema=NAME+'.prepare',contract=contract(),archive=deepcopy(archive),bindings=bindings,
        selection_fixed_before_execution=True,model_or_physical_outcomes_read=False,
        claims='action-time reconstruction feasibility only; no diagnosis/theory/cause/repair/freshness assertion')


def build():
    archive,_=read_archive();return make_plan(archive)


def verify(p):
    require(p['schema']==NAME+'.prepare','new independent prepare schema')
    for path,h in p['bindings'].items():require(sha(path)==h,'source seal drift '+path)
    archive,payload=read_archive()
    require(p==make_plan(archive),'complete source/time/selected slice/contract reconstruction')
    return payload


def prepare(path):
    p=build();value=raw(p);path=Path(path)
    require(len(value)<=128*1024 and not path.exists() and shutil.disk_usage(path.parent).free>=FLOOR+len(value)+4096,
        'exclusive small prepare + floor')
    with path.open('xb') as f:f.write(value);f.flush();os.fsync(f.fileno())
    return sha(path)


def compare(state,rgb,archive_state,archive_rgb):
    s=base.source.vector(state);a=base.source.vector(archive_state)
    rgb=base.image(rgb);reference=base.image(archive_rgb)
    d=s-a;p=rgb.astype(np.int16)-reference.astype(np.int16)
    return dict(state_exact=bool(np.array_equal(s,archive_state)),state_bit_exact=base.exact(np.asarray(state),archive_state),
        state_difference=d.tolist(),state_absolute_error=abs(d).tolist(),
        state_max_absolute_error=float(np.max(abs(d))),
        rgb_exact=bool(np.array_equal(rgb,reference)),different_pixels=int(np.any(p!=0,axis=-1).sum()),
        different_channels=int((p!=0).sum()),pixel_MAE=float(np.mean(abs(p))),pixel_max_absolute_error=int(np.max(abs(p))),
        archive_state=array_id(archive_state),archive_pixels=array_id(archive_rgb),
        actual_state=array_id(np.asarray(state)),actual_pixels=array_id(rgb))


class Meter(base.Ops):
    def __init__(self,p,store):
        super().__init__(p,store);self.secondary=[]
    def call(self,kind,fn,**fields):
        require(time.monotonic()-self.started<self.p['wall_seconds'],'wall cap before dispatch')
        c=self.counts[kind];require(c['attempted']<self.p['caps'][kind],'predispatch '+kind+' cap')
        c['attempted']+=1;self.store.emit('operation_attempted',resource=kind,**fields)
        try:value=fn()
        except BaseException as exc:
            try:self.store.emit('operation_failed',resource=kind,error_type=type(exc).__name__,message=str(exc),**fields)
            except BaseException as err:self.secondary.append(dict(phase='failed_operation_journal',type=type(err).__name__,message=str(err)))
            raise
        c['completed']+=1;self.store.emit('operation_completed',resource=kind,**fields)
        return value


def validate_env(body):
    base.validate_env(body)
    require(body.render_action is False and body.render_size==224,'fixed no-action-overlay RGB224')


def branch(env,slot,payload,row,meter,store):
    body=env.unwrapped;ordinal,lag=slot['episode_ordinal'],row['lag']
    tag=dict(episode_ordinal=ordinal,lag=lag)
    initial,info=meter.call('reset',lambda:env.reset(seed=0),reset_role='seed_goal',**tag)
    goal=base.source.vector(info['goal_state'])
    store.emit('reset_return',**tag,reset_role='seed_goal',seed=0,
        state=base.source.vector(initial['state']).tolist(),snapshot=base.snapshot(body),goal=goal.tolist())
    obs,info=meter.call('reset',lambda:env.reset(seed=0,options=dict(state=payload['states'][0].copy(),goal_state=goal.copy())),
        reset_role='archive_placement',**tag)
    store.emit('reset_return',**tag,reset_role='archive_placement',seed=0,
        state=base.source.vector(obs['state']).tolist(),snapshot=base.snapshot(body),
        goal=base.source.vector(info['goal_state']).tolist())
    commands=payload['actions'][lag:lag+20].copy()
    store.put(f'e{ordinal}_lag{lag}_commands.json',dict(**tag,source_action_slice=[lag,lag+20],
        raw_float32_bits=commands.tobytes().hex(),raw_commands=commands.tolist(),requested_state=array_id(payload['states'][0]),
        requested_state_values=payload['states'][0].astype(float).tolist(),native_reset_goal=goal.tolist()))
    row.update(status='running',steps_completed=0,checkpoints=[],missing_checkpoints=list(CHECKPOINTS),
        terminated=False,truncated=False,initial_native_success=bool(body.eval_state(goal,obs['state'])[0]))
    def capture(k):
        before=base.snapshot(body);rgb=base.image(env.render());after=base.snapshot(body)
        name=f'e{ordinal}_lag{lag}_{k:02}.png';h=store.frame(name,rgb)
        receipt=dict(**tag,step=k,png=name,png_sha256=h,pixels=array_id(rgb),
            state=base.source.vector(obs['state']).tolist(),snapshot=after)
        # Fresh returned image/state durable BEFORE subsequent exactness gates.
        store.emit('fresh_observation',**receipt)
        require(before==after and base.exact(np.asarray(obs['state']),np.asarray(body._get_obs())),
            'render changed dynamics or observation disagrees with native state')
        if k in CHECKPOINTS:
            c=compare(obs['state'],rgb,payload['states'][k],payload['pixels'][CHECKPOINTS.index(k)])
            record=dict(step=k,archive_global_row=slot['offset']+k,observation=receipt,comparison=c)
            row['checkpoints'].append(record);row['missing_checkpoints'].remove(k)
            store.emit('checkpoint_comparison',**tag,checkpoint=record)
    capture(0)
    for k,command in enumerate(commands,1):
        obs,reward,terminated,truncated,info=meter.call('public_step',lambda c=command:env.step(c.copy()),
            step=k,source_action_local_index=lag+k-1,raw_action=command.tolist(),**tag)
        actual=np.asarray(body.latest_action).copy()
        store.emit('native_return',**tag,step=k,source_action_local_index=lag+k-1,
            requested_raw=array_id(command),requested_float32_bits=command.tobytes().hex(),
            actual_raw=array_id(actual),actual_raw_bits=actual.tobytes().hex(),
            state=base.source.vector(obs['state']).tolist(),snapshot=base.snapshot(body),
            reward=float(reward),terminated=bool(terminated),truncated=bool(truncated))
        row.update(steps_completed=k,terminated=bool(terminated),truncated=bool(truncated))
        capture(k)
        require(base.exact(actual,command),'native latest_action not exact literal archive raw float32')
        require(np.array_equal(info['goal_state'],goal),'native goal changed after placement')
        success,distance=body.eval_state(goal,obs['state'])
        require(bool(terminated)==bool(success) and float(reward)==-float(distance),'unchanged native reward/termination')
        if terminated or truncated:
            row['status']='native_stopped';break
    else:row['status']='horizon_completed'
    store.emit('branch_finished',receipt=row)


def execute(p,store,factory,*,authorization):
    require(authorization==AUTH,'explicit MAIN physics authorization')
    meter=Meter(contract(),store)
    rows=[dict(episode_ordinal=i,lag=lag,status='not_started',steps_completed=0,checkpoints=[],
        missing_checkpoints=list(CHECKPOINTS)) for i in ORDINALS for lag in (0,1)]
    result=dict(schema=NAME+'.report',prepare_id=digest(p),status='running',errors=[],rows=rows,
        fixed_denominator=4,expected_checkpoints=20,contract=contract(),archive=p.get('archive'),
        started_unix_ns=time.time_ns(),P1_admission=False,scientific_winner=None,new_neural_calls=0)
    env=body=None;hooks=[];spaces=[];in_reset=False;active={}
    def error(phase,exc):result['errors'].append(dict(phase=phase,type=type(exc).__name__,message=str(exc)))
    def install_space():
        space=body.space
        if any(x[0] is space for x in spaces):return
        old=space.step;owned='step' in space.__dict__;spaces.append((space,owned,old))
        space.step=lambda *a,**kw:meter.call('physics_step',lambda:old(*a,**kw),**active)
    try:
        store.emit('run_started',prepare_id=digest(p),contract=contract(),archive=p.get('archive'))
        payloads=verify(p)
        require(p['contract']==contract(),'fixed predispatch contract')
        if any(x is not None for x in payloads):
            env=factory();body=env.unwrapped;validate_env(body)
            set_state,set_goal,render,reset=body._set_state,body._set_goal_state,body.render,env.reset
            def setter(value):
                install_space()
                return meter.call('set_state',lambda:set_state(value),requested_state=array_id(value),**active)
            def goalsetter(value):
                before=base.snapshot(body)
                v=meter.call('goal_setter',lambda:set_goal(value),requested_goal=array_id(value),**active)
                after=base.snapshot(body)
                store.emit('goal_setter_return',before=before,after=after,goal=base.source.vector(body.goal_state).tolist(),**active)
                require(before==after,'native goal setter mutated body dynamics')
                return v
            def renderer():
                kind='goal_render' if in_reset else 'explicit_render'
                value=meter.call(kind,render,**active)
                store.emit('render_return',resource=kind,pixels=array_id(value),snapshot=base.snapshot(body),**active)
                return value
            def resetter(*args,**kwargs):
                nonlocal in_reset
                in_reset=True
                try:return reset(*args,**kwargs)
                finally:in_reset=False
            for obj,key,new in ((body,'_set_state',setter),(body,'_set_goal_state',goalsetter),(body,'render',renderer),(env,'reset',resetter)):
                hooks.append((obj,key,key in obj.__dict__,getattr(obj,key)));setattr(obj,key,new)
        for index,(slot,payload) in enumerate(zip(p['archive']['episodes'],payloads,strict=True)):
            for lag in (0,1):
                row=rows[2*index+lag];active.clear();active.update(episode_ordinal=slot['episode_ordinal'],lag=lag)
                if payload is None:
                    row.update(status='rejected_short_fixed_slot',reason=slot['reason'])
                    store.emit('branch_finished',receipt=row);continue
                row['status']='running';branch(env,slot,payload,row,meter,store)
        result['status']='alignment_probe_processed'
    except BaseException as exc:
        error('execution',exc)
        for row in rows:
            if row['status']=='running':row['status']='engineering_failed'
    finally:
        for obj,owned,original in reversed(spaces):
            try:
                if owned:obj.step=original
                else:delattr(obj,'step')
            except BaseException as exc:error('restore_space',exc)
        for obj,key,owned,original in reversed(hooks):
            try:
                if owned:setattr(obj,key,original)
                else:delattr(obj,key)
            except BaseException as exc:error('restore_hook',exc)
        if env is not None:
            try:env.close()
            except BaseException as exc:error('close',exc)
        try:verify(p)
        except BaseException as exc:error('final_source_seal',exc)
        if not result['errors']:
            try:
                c={k:v['completed'] for k,v in meter.counts.items()}
                require(all(v['attempted']==v['completed'] for v in meter.counts.values()) and
                    c['physics_step']==2*c['reset']+10*c['public_step'] and c['goal_render']==c['reset'] and
                    c['set_state']==2*c['reset'] and c['goal_setter']==c['reset'],'native returned resource census')
            except BaseException as exc:error('final_census',exc)
        result['errors'].extend(meter.secondary)
        result.update(counts=deepcopy(meter.counts),horizon_completed=sum(r['status']=='horizon_completed' for r in rows),
            native_stopped=sum(r['status']=='native_stopped' for r in rows),
            completed_checkpoints=sum(len(r['checkpoints']) for r in rows),wall_seconds=time.monotonic()-meter.started,
            inputs_final_verified=not any(x['phase']=='final_source_seal' for x in result['errors']))
        if result['errors']:result['status']='engineering_failed'
        try:store.emit('run_finished',status=result['status'],counts=result['counts'],errors=result['errors'])
        except BaseException as exc:error('final_journal',exc);result['status']='engineering_failed'
        result['journal_chain_tail']=store.previous
        result['first_error']=deepcopy(result['errors'][0]) if result['errors'] else None
        result['run_processed']=result['status']=='alignment_probe_processed'
        try:store.put('report.json',result,final=True)
        except BaseException as exc:
            error('final_report',exc);result.update(status='engineering_failed',run_processed=False,first_error=deepcopy(result['errors'][0]))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','verify','run'))
    parser.add_argument('--prepared',type=Path);parser.add_argument('--prepare-sha');parser.add_argument('--output',type=Path)
    parser.add_argument('--authorize-physics',action='store_true');args=parser.parse_args()
    if args.command=='prepare':value=prepare(args.output)
    else:
        require(sha(args.prepared)==args.prepare_sha,'external prepare SHA')
        require(args.prepared.stat().st_size<=128*1024,'small prepared JSON');p=json.loads(args.prepared.read_bytes())
        if args.command=='verify':verify(p);value=p['schema']
        else:
            require(args.authorize_physics,'MAIN explicit authorization required')
            def factory():
                import gymnasium as gym
                import stable_worldmodel  # Registration only; MAIN physics, no model.
                return gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224,with_target=False,render_action=False,relative=True)
            store=Store(args.output)
            store.put('execution_binding.json',dict(prepared=str(args.prepared.resolve()),prepare_file_sha256=args.prepare_sha))
            result=execute(p,store,factory,authorization=AUTH)
            print(base.source.canonical(dict(status=result['status'],counts=result['counts'],horizon_completed=result['horizon_completed'])))
            if result['status']=='engineering_failed':raise SystemExit(1)
            return
    print(base.source.canonical(value))


if __name__=='__main__':main()
