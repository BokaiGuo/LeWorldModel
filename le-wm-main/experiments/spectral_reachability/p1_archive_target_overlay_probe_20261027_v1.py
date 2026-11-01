"""Fixed archive lag0 full20, native off/on/off at five checkpoints; MAIN only.

Actions are original raw action[0:20], not a function of future state or metrics.
Only rendering intervention is with_target=False/True/False, then restoration.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import p1_archive_native_alignment_probe_20261027_v1 as prior

NAME='p1_archive_target_overlay_probe_20261027_v1'
ROOT,base=prior.ROOT,prior.base
require,sha,raw,digest=prior.require,prior.sha,prior.raw,prior.digest
REPORT=ROOT/'results/p1_archive_native_alignment_probe_20261027_v1_main01/report.json'
REPORT_SHA='affb90a6e67c096623b62ac1d4a1f9f2e11ceaa5d5fe8436f365d39536de2e05'
PREPARE_SHA='899485c8ae93827ef02100f64429dd293422f617055e81c37dc1f1859b6db0d7'
PRIOR_SHA='62f8fcbffacbc14cdf3641654eb2c4adb9fcadea1c41a6e762a1a0c79034d172'
AUTH='MAIN_ARCHIVE_TARGET_OVERLAY_PHYSICS_ONLY'
CAP,FLOOR,FINAL=prior.CAP,prior.FLOOR,prior.FINAL
Store=prior.Store


def read(path,expected):
    require(sha(path)==expected and Path(path).stat().st_size<128*1024,'bounded sealed JSON SHA')
    p=json.loads(Path(path).read_bytes());require(sha(path)==expected,'JSON read TOCTOU');return p


def old_png(path,expected):
    from PIL import Image
    from io import BytesIO
    require(sha(path)==expected and path.stat().st_size<=1024**2,'original off checkpoint PNG seal')
    data=path.read_bytes()
    with Image.open(BytesIO(data)) as im:
        require(im.format=='PNG' and im.mode=='RGB' and im.size==(224,224),'original unconverted RGB224 PNG')
        a=np.asarray(im).copy()
    require(sha(path)==expected,'original PNG read TOCTOU');return base.image(a)


def read_payloads(archive,report):
    """Bounded approved H5 reader, then fixed lag0 reference PNGs; no old execute.

    Future state/RGB is comparison-only. Controller uses only original actions.
    """
    actual,arrays=prior.read_archive();require(actual==archive,'complete approved H5 stat/schema/slice identities')
    require([s['episode_ordinal'] for s in archive['episodes']]==list(prior.ORDINALS),'fixed two episodes')
    payload=[];bindings={}
    for slot,a in zip(archive['episodes'],arrays,strict=True):
        require(slot['status']=='selected' and a is not None,'both fixed episodes available, no substitution')
        rows=[r for r in report['rows'] if r['episode_ordinal']==slot['episode_ordinal'] and r['lag']==0]
        require(len(rows)==1 and rows[0]['status']=='horizon_completed' and rows[0]['steps_completed']==20,
            'original fixed lag0 full20 reference')
        cps=rows[0]['checkpoints'];require([c['step'] for c in cps]==list(prior.CHECKPOINTS),'all original checkpoints')
        references=[]
        for index,c in enumerate(cps):
            o=c['observation'];step=c['step'];name=o['png']
            require(type(name) is str and Path(name).name==name,'flat old checkpoint filename')
            path=REPORT.parent/name;image=old_png(path,o['png_sha256']);bindings[str(path)]=o['png_sha256']
            require(prior.array_id(image)==o['pixels'] and c['archive_global_row']==slot['offset']+step and
                prior.array_id(a['states'][step])==c['comparison']['archive_state'] and
                prior.array_id(a['pixels'][index])==c['comparison']['archive_pixels'], 'original checkpoint/reference/slice identities')
            references.append(dict(receipt=deepcopy(o),pixels=image))
        payload.append(dict(actions=a['actions'][:20].copy(),initial_state=a['states'][0].copy(),
            comparison_states=a['states'].copy(),comparison_pixels=a['pixels'].copy(),references=references))
    return payload,bindings


def inputs():
    require(sha(prior.__file__)==PRIOR_SHA,'frozen original probe source')
    report=read(REPORT,REPORT_SHA);bp=REPORT.parent/'execution_binding.json';bh=sha(bp);binding=read(bp,bh)
    require(binding['prepare_file_sha256']==PREPARE_SHA,'original probe external prepare SHA')
    pp=Path(binding['prepared']).absolute();p=read(pp,PREPARE_SHA)
    require(p['schema']==prior.NAME+'.prepare' and p['contract']==prior.contract() and
        report['schema']==prior.NAME+'.report' and report['status']=='alignment_probe_processed' and
        report['errors']==[] and report['horizon_completed']==4 and report['inputs_final_verified'] is True and
        report['prepare_id']==digest(p) and report['archive']==p['archive'],'approved complete original probe chain')
    bindings={str(REPORT):REPORT_SHA,str(bp):bh,str(pp):PREPARE_SHA,**p['bindings']}
    for path,h in bindings.items():require(sha(path)==h,'original probe source seal '+path)
    payload,frames=read_payloads(p['archive'],report);bindings.update(frames)
    return dict(original_archive=deepcopy(p['archive']),bindings=bindings,
        source_identity_scope='approved H5 stat + original slices + lag0 checkpoint PNG/state/snapshot; not whole H5 SHA'),payload


def contract():
    return dict(ordinals=list(prior.ORDINALS),branches=2,lag=0,horizon=20,checkpoints=list(prior.CHECKPOINTS),flags=[False,True,False],seed=0,
        placement_seed=0,environment='swm/PushT-v1',resolution=224,render_action=False,render_mode='rgb_array',
        wall_seconds=300,caps=dict(reset=4,set_state=8,physics_step=408,goal_setter=4,goal_render=4,explicit_render=30,public_step=40),
        output_cap_bytes=CAP,final_reserve_bytes=FINAL,disk_floor_bytes=FLOOR,
        interventions='only with_target off/on/off; native goals, goal_pose, colors and camera unchanged',
        green_mask='int16 G>R and G>B, descriptive counts only',neural=0,GPU=0,training=False,
        raw='literal archive action[0:20] float32; no clip/tanh/rescale; no future-state controller',
        stop='honor termination/truncation; retain missing checkpoints/partial, no padding',P1_admission=False,scientific_winner=None)


def build():
    source,_=inputs();bindings=deepcopy(source['bindings'])
    for f in [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]:bindings[str(f)]=sha(f)
    return dict(schema=NAME+'.prepare',source=source,bindings=bindings,contract=contract(),
        selection_fixed_before_execution=True,archive_future_state_for_evaluation_only=True)


def verify(p):
    require(p['schema']==NAME+'.prepare' and p['contract']==contract(),'own fixed overlay prepare')
    for path,h in p['bindings'].items():require(sha(path)==h,'overlay closure seal')
    source,payload=inputs();require(source==p['source'],'original probe input chain unchanged')
    # Recreate local bindings without a second H5 read.
    bs=deepcopy(source['bindings'])
    for f in [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]:bs[str(f)]=sha(f)
    require(p==dict(schema=NAME+'.prepare',source=source,bindings=bs,contract=contract(),
        selection_fixed_before_execution=True,archive_future_state_for_evaluation_only=True),'entire overlay prepare reconstruction')
    return payload


def prepare(path):
    p=build();value=raw(p);path=Path(path).absolute()
    require(not path.exists() and len(value)<128*1024 and shutil.disk_usage(path.parent).free>=FLOOR+len(value)+4096,'exclusive small prepare +200MiB')
    with path.open('xb') as f:require(f.write(value)==len(value),'short prepare');f.flush();os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(prepared=str(path),prepare_file_sha256=sha(path),bytes=len(value))


def json_value(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:json_value(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [json_value(v) for v in value]
    require(value is None or type(value) in (str,int,float,bool),'literal renderer configuration');return value


def renderer_config(body):
    return dict(render_mode=body.render_mode,render_size=body.render_size,window_size=body.window_size,
        render_action=body.render_action,relative=body.relative,action_scale=body.action_scale,
        dt=body.dt,control_hz=body.control_hz,variations=json_value(body.variation_space.value))


def invariant(body):
    pose=np.asarray(body.goal_pose,dtype=float);require(pose.shape==(3,) and np.isfinite(pose).all(),'native goal_pose3')
    return dict(snapshot=base.snapshot(body),goal_state=base.source.vector(body.goal_state).tolist(),
        goal_pose=pose.tolist(),goal_rgb=prior.array_id(body._goal),renderer=renderer_config(body))


def compare(rgb,reference):
    a,b=base.image(rgb).astype(np.int16),base.image(reference).astype(np.int16);d=a-b
    ga=(a[:,:,1]>a[:,:,0])&(a[:,:,1]>a[:,:,2]);gb=(b[:,:,1]>b[:,:,0])&(b[:,:,1]>b[:,:,2])
    return dict(rgb_MAE=float(np.mean(np.abs(d))),rgb_max_absolute_error=int(np.max(np.abs(d))),
        different_pixels=int(np.any(d!=0,axis=-1).sum()),rgb_exact=bool(np.array_equal(a,b)),
        green_actual=int(ga.sum()),green_archive=int(gb.sum()),green_intersection=int((ga&gb).sum()),
        green_actual_only=int((ga&~gb).sum()),green_archive_only=int((gb&~ga).sum()),green_xor=int((ga^gb).sum()))


def branch(env,ordinal,payload,row,meter,store,do_reset):
    body=env.unwrapped
    _,info=do_reset(None,'seed_goal');goal=base.source.vector(info['goal_state']).copy()
    obs,info=do_reset(dict(state=payload['initial_state'].copy(),goal_state=goal.copy()),'archive_placement')
    require(np.array_equal(info['goal_state'],goal),'original native reset goal retained')
    fixed=invariant(body);row.update(status='running',native_goal=goal.tolist(),goal_pose=fixed['goal_pose'],
        renderer=fixed['renderer'],placed_snapshot=fixed['snapshot'],requested_state0=prior.array_id(payload['initial_state']),
        state0_absolute_error=np.abs(base.source.vector(obs['state'])-payload['initial_state']).tolist())
    row['state0_max_absolute_error']=float(np.max(row['state0_absolute_error']))
    def capture(step):
        reference=payload['references'][prior.CHECKPOINTS.index(step)]
        before_triplet=invariant(body);off=None;first=len(row['frames']);original=body.with_target
        try:
            for index,flag in enumerate((False,True,False)):
                body.with_target=flag;before=invariant(body)
                image=base.image(env.render());after=invariant(body)
                name=f'e{ordinal}_s{step:02}_{index}_'+('on' if flag else 'off')+'.png';h=store.frame(name,image)
                old=reference['receipt']
                match=None if flag else dict(file=h==old['png_sha256'],pixels=base.exact(image,reference['pixels']) and
                    prior.array_id(image)==old['pixels'],state=base.exact(np.asarray(after['snapshot']['state']),np.asarray(old['state'])),
                    snapshot=after['snapshot']==old['snapshot'])
                replay=None if index!=2 else base.exact(off,image)
                receipt=dict(step=step,overlay_index=index,with_target=flag,png=name,png_sha256=h,pixels=prior.array_id(image),
                    pre=before,post=after,comparison=compare(image,payload['comparison_pixels'][prior.CHECKPOINTS.index(step)]),
                    old_off_match=match,off_replay_match=replay)
                row['frames'].append(receipt)
                if index==2:
                    row['off_replay_checks'].append(dict(step=step,first_frame_index=first,last_frame_index=len(row['frames'])-1,bit_exact=replay))
                store.emit('fresh_overlay_frame',episode_ordinal=ordinal,receipt=receipt)
                require(after==before==before_triplet and body.with_target is flag,'render changed full snapshot/goals/configuration/flag')
                # Goals/config remain equal across actual actions, dynamics may move.
                require({k:v for k,v in after.items() if k!='snapshot'}=={k:v for k,v in fixed.items() if k!='snapshot'},
                    'goal or renderer configuration changed across actual replay')
                if match is not None:require(all(match.values()),'new off differs from original sealed lag0 checkpoint')
                if index==0:off=image.copy()
                if index==2:require(replay,'off->on->off replay pixels differ')
            row['missing_checkpoints'].remove(step)
        finally:
            body.with_target=original
    capture(0)
    # Only this fixed raw array is used for control. Future states and image
    # comparisons cannot change a command, lag, target, number of candidates.
    commands=payload['actions'];require(commands.dtype==np.float32 and commands.shape==(20,2),'fixed literal lag0 full20 commands')
    for step,command in enumerate(commands,1):
        obs,reward,terminated,truncated,info=meter.call('public_step',lambda c=command:env.step(c.copy()),
            episode_ordinal=ordinal,step=step,source_action_index=step-1,requested_bits=command.tobytes().hex())
        actual=np.asarray(body.latest_action).copy()
        store.emit('native_return',episode_ordinal=ordinal,step=step,requested_raw=prior.array_id(command),
            requested_float32_bits=command.tobytes().hex(),actual_raw=prior.array_id(actual),actual_raw_bits=actual.tobytes().hex(),
            state=base.source.vector(obs['state']).tolist(),snapshot=base.snapshot(body),reward=float(reward),
            terminated=bool(terminated),truncated=bool(truncated))
        row.update(steps_completed=step,terminated=bool(terminated),truncated=bool(truncated))
        if step in prior.CHECKPOINTS:capture(step)
        require(base.exact(actual,command),'actual action differs from literal float32 archive command')
        require(base.exact(np.asarray(obs['state']),np.asarray(body._get_obs())) and np.array_equal(info['goal_state'],goal),
            'native returned state or goal mismatch')
        success,distance=body.eval_state(goal,obs['state'])
        require(bool(terminated)==bool(success) and float(reward)==-float(distance),'native termination/reward contract changed')
        if terminated or truncated:
            row['status']='native_stopped';return
    row['status']='overlay_full20_completed'


def execute(p,store,factory,*,authorization):
    require(authorization==AUTH,'explicit MAIN overlay physics authorization')
    meter=prior.Meter(contract(),store);rows=[dict(episode_ordinal=o,status='not_started',frames=[],steps_completed=0,
        missing_checkpoints=list(prior.CHECKPOINTS),off_replay_checks=[],terminated=False,truncated=False) for o in prior.ORDINALS]
    errors=[];env=body=None;hooks=[];spaces=[];resetting=False;active={};original_flag=None
    def error(phase,e):errors.append(dict(phase=phase,type=type(e).__name__,message=str(e)))
    def patch(obj,key,new):hooks.append((obj,key,key in obj.__dict__,getattr(obj,key)));setattr(obj,key,new)
    def install_space():
        space=body.space
        if any(s[0] is space for s in spaces):return
        old=space.step;spaces.append((space,'step' in space.__dict__,old))
        space.step=lambda *a,**kw:meter.call('physics_step',lambda:old(*a,**kw),**active)
    def do_reset(options,role):
        nonlocal resetting
        def call():
            nonlocal resetting
            resetting=True
            try:return env.reset(seed=0,**({} if options is None else dict(options=options)))
            finally:resetting=False
        result=meter.call('reset',call,role=role,**active)
        store.emit('reset_return',role=role,seed=0,snapshot=base.snapshot(body),
            goal_state=base.source.vector(result[1]['goal_state']).tolist(),goal_pose=np.asarray(body.goal_pose,dtype=float).tolist(),**active)
        return result
    try:
        store.emit('run_started',prepare_id=digest(p),contract=contract())
        payloads=verify(p);require(len(payloads)==2,'fixed two lag0 replay inputs')
        env=factory();body=env.unwrapped;prior.validate_env(body);original_flag=body.with_target
        set_state,set_goal,render=body._set_state,body._set_goal_state,body.render
        def setter(value):
            install_space();return meter.call('set_state',lambda:set_state(value),requested=prior.array_id(value),**active)
        def goalsetter(value):
            before=base.snapshot(body);result=meter.call('goal_setter',lambda:set_goal(value),**active)
            require(base.snapshot(body)==before,'native goal setter changed body dynamics');return result
        def renderer():return meter.call('goal_render' if resetting else 'explicit_render',render,**active)
        # Public calls are metered at branch() exactly once; native step is not
        # replaced or borrowed from any prior execute implementation.
        for obj,key,new in ((body,'_set_state',setter),(body,'_set_goal_state',goalsetter),(body,'render',renderer)):patch(obj,key,new)
        for ordinal,payload,row in zip(prior.ORDINALS,payloads,rows,strict=True):
            active.clear();active['episode_ordinal']=ordinal
            row['status']='running';branch(env,ordinal,payload,row,meter,store,do_reset)
            store.emit('branch_finished',receipt=row)
    except BaseException as e:
        error('execution',e)
        for row in rows:
            if row['status']=='running':row['status']='engineering_failed'
    finally:
        if body is not None and original_flag is not None:
            try:body.with_target=original_flag
            except BaseException as e:error('restore_flag',e)
        for space,owned,old in reversed(spaces):
            try:
                if owned:space.step=old
                else:delattr(space,'step')
            except BaseException as e:error('restore_space',e)
        for obj,key,owned,old in reversed(hooks):
            try:
                if owned:setattr(obj,key,old)
                else:delattr(obj,key)
            except BaseException as e:error('restore_hook',e)
        if env is not None:
            try:env.close()
            except BaseException as e:error('close',e)
        try:verify(p)
        except BaseException as e:error('final_source',e)
        if not errors:
            try:
                c={k:v['completed'] for k,v in meter.counts.items()}
                require(all(v['attempted']==v['completed'] for v in meter.counts.values()) and
                    c['physics_step']==2*c['reset']+10*c['public_step'] and c['set_state']==2*c['reset'] and
                    c['goal_setter']==c['goal_render']==c['reset'] and
                    c['public_step']==sum(r['steps_completed'] for r in rows) and
                    c['explicit_render']==sum(len(r['frames']) for r in rows),'exact actual overlay resource census')
                if all(r['status']=='overlay_full20_completed' for r in rows):
                    require(c==contract()['caps'],'exact complete full20 caps')
            except BaseException as e:error('final_census',e)
        errors.extend(meter.secondary)
        status=('overlay_probe_processed' if all(r['status']=='overlay_full20_completed' for r in rows)
            else 'overlay_probe_partial_native_stop') if not errors else 'engineering_failed'
        try:store.emit('run_finished',status=status,counts=meter.counts,errors=errors)
        except BaseException as e:error('final_journal',e);status='engineering_failed'
        result=dict(schema=NAME+'.report',status=status,prepare_id=digest(p),approved_probe_report_sha256=REPORT_SHA,
            approved_probe_prepare_sha256=PREPARE_SHA,rows=rows,fixed_denominator=2,expected_frames=30,
            counts=deepcopy(meter.counts),contract=contract(),errors=errors,first_error=deepcopy(errors[0]) if errors else None,
            journal_chain_tail=store.previous,inputs_final_verified=not any(e['phase']=='final_source' for e in errors),
            wall_seconds=time.monotonic()-meter.started,new_neural_calls=0,P1_admission=False,scientific_winner=None)
        try:store.put('report.json',result,final=True)
        except BaseException as e:error('final_report',e);result.update(status='engineering_failed',first_error=deepcopy(errors[0]))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','verify','run'))
    for key in ('prepared','prepare-sha','output'):p.add_argument('--'+key)
    p.add_argument('--authorize-physics',action='store_true');a=p.parse_args()
    if a.command=='prepare':result=prepare(a.output)
    else:
        prepared=read(a.prepared,a.prepare_sha)
        if a.command=='verify':verify(prepared);result=dict(status='verified_no_execution')
        else:
            require(a.authorize_physics,'MAIN CLI authorization')
            def factory():
                import gymnasium as gym
                import stable_worldmodel  # noqa: F401
                return gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224,with_target=False,render_action=False,relative=True)
            store=Store(a.output);store.put('execution_binding.json',dict(prepared=str(Path(a.prepared).resolve()),prepare_file_sha256=a.prepare_sha))
            result=execute(prepared,store,factory,authorization=AUTH)
    print(base.source.canonical({k:v for k,v in result.items() if k!='rows'}))
    if result.get('status')=='engineering_failed':raise SystemExit(1)


if __name__=='__main__':main()
