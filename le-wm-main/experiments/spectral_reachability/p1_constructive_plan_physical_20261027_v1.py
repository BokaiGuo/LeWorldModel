"""MAIN-only consumer of a presealed constructive blind-compiler selection.

No model imports. Certificate access is offline evaluator-only: witness arrays
are NEVER read or used as candidate commands. Future compiler field extraction
is explicit JSON Pointer data, never executable adapters or monkey patches.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
import numpy as np
from experiments.spectral_reachability import p1_constructive_reachable_source_20261027_v1 as builder

source=builder.source
ROOT=builder.ROOT
NAME='p1_constructive_plan_physical_20261027_v1'
AUTH='MAIN_CONSTRUCTIVE_SELECTED_PLAN_PHYSICS_ONLY'
CAP,FLOOR,FINAL=8*1024**2,200*1024**2,256*1024
SOURCE_SHA='9ad3a414fd2148258524d4a7e9d4f070f59532ec2e5943cfa61f9145b3a41bd9'
require=source.require


def raw(value):return (source.canonical(value)+'\n').encode()
def read(path):return json.loads(Path(path).read_bytes())
def sha(path):
    path=Path(path);require(path.is_file() and not path.is_symlink(),'regular nonsymlink input: '+str(path))
    return source.file_sha(path)


def commands(value):
    a=np.asarray(value)
    require(a.shape==(10,2) and a.dtype.kind in 'fi' and np.isfinite(a).all() and (abs(a)<=1).all() and
        np.array_equal(a,a.astype(np.float32)),'exact finite float32 raw10x2 in[-1,1]; never clip/round')
    return a.astype(np.float32)


def pointer(value,path):
    """RFC6901 subset: explicit absolute path; no eval, wildcard or search."""
    require(type(path) is str and (path=='' or path.startswith('/')),'explicit JSON pointer')
    for part in ([] if path=='' else path[1:].split('/')):
        require(not re.search(r'~(?![01])',part),'valid JSON pointer escaping')
        key=part.replace('~1','/').replace('~0','~')
        if isinstance(value,list):
            require(bool(re.fullmatch(r'0|[1-9][0-9]*',key)) and int(key)<len(value),'in-range literal array pointer')
            value=value[int(key)]
        else:require(isinstance(value,dict) and key in value,'existing object pointer');value=value[key]
    return value


def journal(path,tail):
    data=Path(path).read_bytes();require(len(data)<8*1024**2 and data.endswith(b'\n'),'bounded complete source journal')
    events=[json.loads(l) for l in data.splitlines()];previous=None
    for i,e in enumerate(events):
        require(e['index']==i and e['previous_sha256']==previous and
            e['event_sha256']==source.digest({k:v for k,v in e.items() if k!='event_sha256'}),'source hashchain')
        previous=e['event_sha256']
    require(previous==tail and events[-1]['kind']=='run_finished','source terminal tail');return events


def load_source(pack_path,pack_sha,prepared,prepare_sha):
    """Authenticate audited source; no episode, witness(), simulator or model."""
    bindings={}
    def bind(path,h):
        path=str(Path(path).resolve());require(sha(path)==h,'source SHA mismatch: '+path);bindings[path]=h;return path
    require(pack_sha==SOURCE_SHA,'this v1 binds the reviewed2-root actual source; other sources need a new contract')
    pack_path=Path(bind(pack_path,pack_sha));pack=read(pack_path)
    bind(prepared,prepare_sha);p=builder.read_prepared(Path(prepared).read_bytes())
    require(pack['sourcepack_complete'] is True and pack['status']=='fixed_slots_processed' and pack['errors']==[] and
        pack['first_error'] is None and pack['prepare_id']==source.digest(p) and len(pack['rows'])==len(p['slots'])==2 and
        pack['reachable_count']==2,'reviewed complete reachable source')
    binding=read(pack_path.parent/'execution_binding.json')
    require(binding==dict(prepared=str(Path(prepared).resolve()),prepare_file_sha256=prepare_sha),'source execution prepare binding')
    bind(pack_path.parent/'execution_binding.json',sha(pack_path.parent/'execution_binding.json'))
    bindings.update(p['bindings'])
    events=journal(pack_path.parent/'journal.jsonl',pack['journal_chain_tail'])
    bind(pack_path.parent/'journal.jsonl',sha(pack_path.parent/'journal.jsonl'))
    references=[]
    for i,row in enumerate(pack['rows']):
        require(row['slot']==p['slots'][i] and row['slot']['index']==i and row['reachable'] is True and
            row['status']=='reachable_native_replayed' and row['replay_native_success']==[True,True],'source slot reach identity')
        blind_path=bind(pack_path.parent/row['blind_payload'],row['blind_payload_sha256'])
        blind=builder.load_blind(blind_path,row['blind_payload_sha256'])
        cert_path=bind(pack_path.parent/row['certificate'],row['certificate_sha256'])
        # Deliberately never access witness_commands, witness_logits or future_states.
        with np.load(cert_path,allow_pickle=False) as z:
            cert={k:z[k].copy() for k in ('goal','goal_rgb','observed_states','root_history')}
        obs=[e for e in events if e['kind']=='fresh_observation' and e['slot']==i and e['phase']=='replay0' and e['step'] in (0,5,10)]
        require([e['step'] for e in obs]==[0,5,10] and np.array_equal([e['state'] for e in obs],cert['observed_states']) and
            np.array_equal(cert['root_history'],blind['rgb_history']) and np.array_equal(cert['goal_rgb'],blind['goal_rgb']) and
            source.root_hash(cert['observed_states'][-1],cert['goal'])==row['root_sha256'],'source blind/state/RGB/root bridge')
        references.append(dict(slot=deepcopy(row['slot']),root_sha256=row['root_sha256'],goal=cert['goal'],goal_rgb=cert['goal_rgb'],
            history=blind['rgb_history'],states=cert['observed_states'],dynamics=[e['snapshot'] for e in obs],
            blind_sha256=row['blind_payload_sha256']))
    return dict(plan=p,pack=pack,references=references,bindings=bindings)


def selection(manifest,data):
    required={'schema','sourcepack','sourcepack_sha256','source_prepared','source_prepare_sha256','roots','arms','plans',
        'selection_fixed_before_physics','physical_outcome_selection','wall_seconds'}
    require(set(manifest)==required and manifest['schema']==NAME+'.selection','exact NEW selection manifest schema')
    require(manifest['selection_fixed_before_physics'] is True and manifest['physical_outcome_selection'] is False,'preselected no-outcome workflow declaration')
    require(type(manifest['wall_seconds']) is int and 0<manifest['wall_seconds']<=600,'explicit wall<=600 seconds')
    arms=manifest['arms'];require(isinstance(arms,list) and 1<=len(arms)<=4 and len(set(arms))==len(arms) and
        all(type(x) is str and re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,39}',x) for x in arms),'fixed1--4 unique safe arms')
    roots=[dict(root_slot=i,root_sha256=r['root_sha256']) for i,r in enumerate(data['references'])]
    require(manifest['roots']==roots and all(type(x['root_slot']) is int for x in manifest['roots']),'fixed original2-root order')
    require([(x['root_slot'],x['arm']) for x in manifest['plans']]==[(i,arm) for i in range(2) for arm in arms], 'complete predeclared2roots x fixedarms; no duplicate/missing/reordering')
    bindings={};plans=[]
    for row in manifest['plans']:
        fields={'root_slot','arm','commands','compiler_report','compiler_report_SHA','compiler_schema','compiler_status',
            'commands_pointer','source_slot_pointer','blind_sha_pointer','arm_pointer','selection_digest'}
        require(set(row)==fields,'explicit compiler mapping, no implicit future adapter')
        require(type(row['root_slot']) is int and row['root_slot'] in (0,1),'literal evaluator root_slot')
        require(source.digest({k:v for k,v in row.items() if k!='selection_digest'})==row['selection_digest'],'external row selection digest')
        commands(row['commands']);path=str(Path(row['compiler_report']).resolve())
        require(sha(path)==row['compiler_report_SHA'],'actual sealed compiler report SHA')
        report=read(path)
        require(type(row['compiler_schema']) is str and report['schema']==row['compiler_schema'] and
            type(row['compiler_status']) is str and row['compiler_status'] not in ('running','partial_or_failed','engineering_failed') and
            report['status']==row['compiler_status'] and report['errors']==[],'explicit reviewed compiler schema/status and no errors')
        slot=pointer(report,row['source_slot_pointer'])
        require(type(slot) is int and slot==row['root_slot'] and
            pointer(report,row['commands_pointer'])==row['commands'] and pointer(report,row['arm_pointer'])==row['arm'] and
            pointer(report,row['blind_sha_pointer'])==data['references'][slot]['blind_sha256'],
            'literal own compiler source_slot/arm/selected raw/blind SHA binding; no native rootsha in controls')
        bindings[path]=row['compiler_report_SHA'];plans.append(deepcopy(row))
    return plans,bindings


def contract(n,wall):
    return dict(branches=n,root_slots=2,own_horizon=10,warmup=10,with_target=False,wall_seconds=wall,
        caps=dict(reset=2*n,public_step=20*n,explicit_render=22*n,goal_render=2*n,set_state=4*n,physics_step=204*n,goal_bind=n),
        output_cap_bytes=CAP,disk_floor_bytes=FLOOR,final_reserve_bytes=FINAL,
        PNG='allwarmup0..10 + post-attach plan0..10 (22/complete branch); terminal/guard failure frames retained before stop',
        scope='same2 calibration tasks; no witness candidate, retry, outcome selection, fresh confirmation or P1 admission')


def material(manifest_path,manifest_sha):
    require(sha(manifest_path)==manifest_sha,'external selection manifest SHA')
    m=read(manifest_path);data=load_source(m['sourcepack'],m['sourcepack_sha256'],m['source_prepared'],m['source_prepare_sha256'])
    plans,bindings=selection(m,data);bindings.update(data['bindings']);bindings[str(Path(manifest_path).resolve())]=manifest_sha
    paths=(Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md'))
    bindings.update({str(p):sha(p) for p in paths})
    return dict(schema=NAME+'.prepare',manifest=str(Path(manifest_path).resolve()),manifest_sha256=manifest_sha,
        plans=plans,contract=contract(len(plans),m['wall_seconds']),bindings=bindings,source_prepare_id=source.digest(data['plan']),
        sourcepack_sha256=m['sourcepack_sha256'],source_slots=m['roots'])


def prepare(manifest,manifest_sha,output):
    p=material(manifest,manifest_sha);value=raw(p);output=Path(output)
    require(len(value)<256*1024 and shutil.disk_usage(output.parent).free>=FLOOR+len(value),'small prepare200MiB floor')
    with output.open('xb') as f:f.write(value);f.flush();os.fsync(f.fileno())
    return dict(prepared=str(output.resolve()),sha256=sha(output),bytes=len(value))


def verify(path,expected):
    require(sha(path)==expected,'external physical prepare SHA');p=read(path)
    require(p==material(p['manifest'],p['manifest_sha256']),'full new selection/source/closure reconstruction');return p


class Store(builder.Store):
    def check(self,size=4096):
        used=sum(x.stat().st_size for x in self.path.iterdir() if x.is_file())
        require(used+size<=CAP-FINAL and shutil.disk_usage(self.path).free>=FLOOR+size,'predispatch artifact cap /200MiB floor')
    def write(self,name,value,append=False,final=False):
        require(Path(name).name==name and name not in ('','.','..') and not (self.path/name).is_symlink(),'flat exclusive output')
        used=sum(x.stat().st_size for x in self.path.iterdir() if x.is_file())
        require(used+len(value)<=CAP-(0 if final else FINAL),'8MiB total includes partial writes/final reserve')
        require(shutil.disk_usage(self.path).free>=FLOOR+len(value),'200MiB write floor')
        with (self.path/name).open('ab' if append else 'xb') as f:f.write(value);f.flush();os.fsync(f.fileno())
        fd=os.open(self.path,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
        return sha(self.path/name)


class Ops:
    def __init__(self,p,store):
        self.p=p;self.store=store;self.started=time.monotonic();self.where={}
        self.counts={k:dict(attempted=0,completed=0) for k in p['contract']['caps']}
    def call(self,kind,fn,**fields):
        require(time.monotonic()-self.started<self.p['contract']['wall_seconds'],'wall cap before dispatch')
        count=self.counts[kind];require(count['attempted']<self.p['contract']['caps'][kind],'predispatch cap: '+kind)
        self.store.check(4096);count['attempted']+=1
        self.store.emit('operation_attempted',resource=kind,**self.where,**fields)
        value=fn();count['completed']+=1
        self.store.emit('operation_completed',resource=kind,**self.where,**fields);return value


def metric(goal,state):
    state=source.vector(state);d=state-source.vector(goal);angle=float(abs(d[4]));native_angle=min(angle,2*np.pi-angle)
    wrapped=float(abs(np.arctan2(np.sin(d[4]),np.cos(d[4]))));xy=float(np.linalg.norm(d[:4]))
    return dict(native_success=bool(xy<20 and native_angle<np.pi/9),native_distance=float(np.linalg.norm(d)),
        native_angle_term=float(native_angle),negative_native_angle=bool(native_angle<0),
        wrapped_allxy_success=bool(xy<20 and wrapped<np.pi/9),wrapped_object_angle_error=wrapped,
        object_position_error=float(np.linalg.norm(d[2:4])),object_pose=state[2:5].tolist(),object_success=None)


def retain(row,exc,stage):
    e=dict(stage=stage,type=type(exc).__name__,message=str(exc));row['status']='failed'
    if row['first_error'] is None:row['first_error']=e
    else:row['secondary_errors'].append(e)


def branch(env,reference,source_plan,plan,row,ops,store,index):
    body=env.unwrapped;slot=reference['slot'];history=[];states=[];dynamics=[]
    ops.where=dict(branch=index,root_slot=plan['root_slot'],arm=plan['arm'],selection_digest=plan['selection_digest'])
    store.emit('branch_started',**ops.where,plan=plan)
    def capture(phase,step,obs):
        # Save even invisible/out-of-canvas valid RGB BEFORE any fresh guard.
        before_render=builder.snapshot(body)
        rendered=ops.call('explicit_render',env.render);candidate=np.asarray(rendered)
        name=f'b{index}_{phase}_{step:02}.png'
        if candidate.shape!=(224,224,3) or candidate.dtype!=np.uint8:
            artifact=f'b{index}_{phase}_{step:02}_invalid.npz';artifact_sha=None
            if candidate.dtype.kind in 'buif':artifact_sha=store.arrays(artifact,raw_render=candidate)
            receipt=dict(phase=phase,step=step,raw_render=artifact if artifact_sha is not None else None,
                file_sha256=artifact_sha,shape=list(candidate.shape),dtype=str(candidate.dtype),
                state=np.asarray(body._get_obs()).tolist(),snapshot=builder.snapshot(body),observation_state=np.asarray(obs['state']).tolist())
            row['last_observation']=receipt
            store.emit('invalid_render',**ops.where,receipt=receipt)
            store.emit('guard_receipt',**ops.where,phase=phase,step=step,accepted=False,reason='invalid render shape/dtype',incident=receipt)
            raise ValueError('invalid native render, incident retained')
        frame=candidate.copy();file_sha=store.frame(name,frame)
        actual=source.vector(body._get_obs());snap=builder.snapshot(body)
        receipt=dict(phase=phase,step=step,png=name,file_sha256=file_sha,pixel_sha256=builder.image_sha(frame),
            state=actual.tolist(),snapshot=snap,observation_state=np.asarray(obs['state']).tolist())
        row['last_observation']=receipt
        store.emit('fresh_observation',**ops.where,receipt=receipt);row['PNG']+=1
        pixels=int(np.count_nonzero(np.all(frame==[65,105,225],axis=-1)))
        same_state=np.array_equal(actual,obs['state']);pure_render=snap==before_render
        legal=pixels>0 and bool(((actual[:4]>=0)&(actual[:4]<=512)).all()) and same_state and pure_render
        store.emit('guard_receipt',**ops.where,phase=phase,step=step,canonical_agent_pixels=pixels,accepted=legal,
            returned_state_equal=bool(same_state),render_preserved_dynamics=pure_render,png=name,file_sha256=file_sha)
        require(legal,'fresh RGB/canvas/state/render guard failure; scene retained; no next dispatch')
        if phase=='plan' and step in (0,5,10):row['checkpoints'].append(receipt)
        return frame,snap
    _,info=ops.call('reset',lambda:env.reset(seed=slot['reset_seed']),kind_of_reset='original_seed',seed=slot['reset_seed'])
    original_goal=source.vector(info['goal_state'])
    requested=source.sample_state(original_goal,list(body.block.shapes),dict(root_seed=slot['root_seed']),0,0,source_plan)
    store.emit('placement_request',**ops.where,original_goal=original_goal.tolist(),requested=requested.tolist(),placement_seed=0)
    obs,info=ops.call('reset',lambda:env.reset(seed=0,options=dict(state=requested.copy(),goal_state=original_goal.copy())),kind_of_reset='placement',seed=0)
    frame,snap=capture('warmup',0,obs);history.append(frame);states.append(source.vector(obs['state']));dynamics.append(snap)
    for step in range(1,11):
        require(not body.eval_state(original_goal,source.vector(obs['state']))[0],'original native terminal before warmup dispatch')
        obs,reward,terminated,truncated,info=ops.call('public_step',lambda:env.step(np.zeros(2,np.float32)),phase='warmup',step=step,raw_action=[0.,0.])
        row['warmup_returned']+=1
        store.emit('returned_transition',**ops.where,phase='warmup',step=step,state=source.vector(obs['state']).tolist(),reward=float(reward),terminated=bool(terminated),truncated=bool(truncated),metrics=metric(original_goal,obs['state']))
        frame,snap=capture('warmup',step,obs)
        require(not terminated and not truncated,'warmup native terminal/truncation; no plan execution')
        if step in (5,10):history.append(frame);states.append(source.vector(obs['state']));dynamics.append(snap)
    require(np.array_equal(history,reference['history']) and np.array_equal(states,reference['states']) and dynamics==reference['dynamics'],'exact blind0/5/10 state/RGB/dynamics reconstruction')
    root=source.vector(obs['state']);goal=reference['goal'];goal_rgb=reference['goal_rgb']
    require(source.root_hash(root,goal)==reference['root_sha256'],'exact original root+constructed goal identity')
    before=builder.snapshot(body);before_counts=deepcopy(ops.counts)
    ops.call('goal_bind',lambda:builder.attach_goal(body,goal,goal_rgb))
    require(all(ops.counts[k]==before_counts[k] for k in ops.counts if k!='goal_bind') and builder.snapshot(body)==before,'goal attachment no physics/render/reset/setter')
    frame,_=capture('plan',0,obs);require(np.array_equal(frame,history[-1]),'goal bind must not change target-free root pixels')
    require(np.array_equal(body.goal_state,goal) and np.array_equal(body._goal,goal_rgb),'actual goal/goalRGB attachment')
    row.update(initial_exact=True,initial_metrics=metric(goal,root),goal=goal.tolist(),root_state=root.tolist())
    require(not row['initial_metrics']['native_success'],'native task already reached at reconstructed root')
    store.emit('goal_attached',**ops.where,goal=goal.tolist(),goal_rgb_sha256=builder.image_sha(goal_rgb),snapshot=before)
    for step,command in enumerate(commands(plan['commands']),1):
        def dispatch(c=command):
            row['own_attempted']+=1
            return env.step(c.copy())
        obs,reward,terminated,truncated,info=ops.call('public_step',dispatch,phase='plan',step=step,raw_action=command.tolist())
        row['own_returned']+=1;state=source.vector(obs['state']);m=metric(goal,state);native_success,distance=body.eval_state(goal,state)
        row.update(final_state=state.tolist(),final_metrics=m,native_success=bool(native_success),terminated=bool(terminated),truncated=bool(truncated))
        store.emit('returned_transition',**ops.where,phase='plan',step=step,command=command.tolist(),state=state.tolist(),
            reward=float(reward),terminated=bool(terminated),truncated=bool(truncated),metrics=m,actual_native_success=bool(native_success),actual_native_distance=float(distance),
            returned_goal_state=np.asarray(info['goal_state']).tolist(),returned_goal_rgb_sha256=builder.image_sha(info['goal']))
        capture('plan',step,obs)
        require(np.array_equal(info['goal_state'],goal) and np.array_equal(info['goal'],goal_rgb),'native goal must remain bound')
        require(bool(native_success)==m['native_success'] and float(distance)==m['native_distance'] and
            bool(terminated)==bool(native_success) and float(reward)==-float(distance),'unchanged native eval/reward/termination')
        if terminated or truncated:
            row['status']='native_success' if terminated and native_success and not truncated else 'native_stopped';break
    else:row['status']='horizon_completed'
    row['full_horizon']=row['own_returned']==10
    row['endpoint_scope']='native_terminal' if row['terminated'] else 'full10' if row['full_horizon'] else 'partial'
    row['object_position_error_change']=row['final_metrics']['object_position_error']-row['initial_metrics']['object_position_error']
    row['object_angle_error_change']=row['final_metrics']['wrapped_object_angle_error']-row['initial_metrics']['wrapped_object_angle_error']


def execute(p,data,store,factory,*,authorization):
    require(authorization==AUTH,'explicit MAIN physics authorization')
    rows=[dict(root_slot=x['root_slot'],arm=x['arm'],selection_digest=x['selection_digest'],status='not_started',first_error=None,
        secondary_errors=[],warmup_returned=0,own_attempted=0,own_returned=0,initial_exact=False,full_horizon=False,native_success=False,
        terminated=False,truncated=False,PNG=0,checkpoints=[]) for x in p['plans']]
    result=dict(schema=NAME+'.report',status='running',rows=rows,errors=[],contract=p['contract'],sourcepack_sha256=p['sourcepack_sha256'],
        new_neural_calls=0,training=False,P1_admission=False,new_independent_samples=0,global_freshness='unknown_calibration_only')
    ops=Ops(p,store)
    def error(stage,exc):result['errors'].append(dict(stage=stage,type=type(exc).__name__,message=str(exc)))
    try:
        store.emit('run_started',prepare=p)
        for index,(plan,row) in enumerate(zip(p['plans'],rows,strict=True)):
            env=None;originals=[];spaces=[];resetting=False;body=None
            try:
                store.check(262144);env=factory();body=env.unwrapped;builder.validate_env(body)
                require(body.render_action is False and body.render_size==224 and body.render_mode=='rgb_array','original target-free/no action-marker RGB224')
                ops.where=dict(branch=index,root_slot=plan['root_slot'],arm=plan['arm'],selection_digest=plan['selection_digest'])
                def install_space():
                    space=body.space
                    if any(x[0] is space for x in spaces):return
                    owned='step' in space.__dict__;old=space.step;spaces.append((space,owned,old))
                    space.step=lambda *a,**kw:ops.call('physics_step',lambda:old(*a,**kw))
                old_set,old_render,old_reset=body._set_state,body.render,env.reset
                def setter(value):install_space();return ops.call('set_state',lambda:old_set(value))
                def render():return ops.call('goal_render',old_render) if resetting else old_render()
                def reset(*a,**kw):
                    nonlocal resetting
                    resetting=True
                    try:return old_reset(*a,**kw)
                    finally:resetting=False
                for obj,key,new in ((body,'_set_state',setter),(body,'render',render),(env,'reset',reset)):
                    originals.append((obj,key,key in obj.__dict__,getattr(obj,key)));setattr(obj,key,new)
                row['status']='running';branch(env,data['references'][plan['root_slot']],data['plan'],plan,row,ops,store,index)
            except (Exception,KeyboardInterrupt) as exc:retain(row,exc,'branch')
            finally:
                for space,owned,old in reversed(spaces):
                    try:
                        if owned:space.step=old
                        else:delattr(space,'step')
                    except (Exception,KeyboardInterrupt) as exc:retain(row,exc,'restore_space')
                for obj,key,owned,old in reversed(originals):
                    try:
                        if owned:setattr(obj,key,old)
                        else:delattr(obj,key)
                    except (Exception,KeyboardInterrupt) as exc:retain(row,exc,'restore_hook')
                if env is not None:
                    try:env.close()
                    except (Exception,KeyboardInterrupt) as exc:retain(row,exc,'close')
                store.emit('branch_closed',receipt=row)
    except (Exception,KeyboardInterrupt) as exc:error('execution',exc)
    # No operation is retried after a partial nested native call.
    try:
        for path,h in p['bindings'].items():require(sha(path)==h,'final seal drift: '+path)
    except (Exception,KeyboardInterrupt) as exc:error('final_bindings',exc)
    result['counts']=ops.counts;result['wall_seconds']=time.monotonic()-ops.started
    clean=not result['errors'] and all(x['status'] in ('horizon_completed','native_success','native_stopped') for x in rows)
    if clean:
        try:
            c=ops.counts;require(all(v['attempted']==v['completed'] for v in c.values()) and
                c['reset']['completed']==2*len(rows) and c['set_state']['completed']==2*c['reset']['completed'] and
                c['goal_render']['completed']==c['reset']['completed'] and c['goal_bind']['completed']==len(rows) and
                c['public_step']['completed']==sum(x['warmup_returned']+x['own_returned'] for x in rows) and
                c['physics_step']['completed']==2*c['reset']['completed']+10*c['public_step']['completed'] and
                c['explicit_render']['completed']==sum(x['PNG'] for x in rows),'actual complete native census')
        except (Exception,KeyboardInterrupt) as exc:error('final_census',exc);clean=False
    result['status']='physical_calibration_recorded' if clean else 'partial_or_failed'
    result['native_successes']=sum(x['status']=='native_success' for x in rows)
    result['journal_sha256']=None
    try:store.emit('run_finished',status=result['status'],counts=ops.counts,errors=result['errors'])
    except (Exception,KeyboardInterrupt) as exc:error('final_journal',exc);result['status']='partial_or_failed'
    try:result['journal_sha256']=sha(store.path/'journal.jsonl')
    except (Exception,KeyboardInterrupt) as exc:error('journal_hash',exc);result['status']='partial_or_failed'
    try:store.put('report.json',result,final=True)
    except (Exception,KeyboardInterrupt) as exc:error('final_report',exc);result['status']='partial_or_failed'
    return result


def run(prepared,expected,output,factory,*,authorization):
    require(authorization==AUTH,'explicit MAIN physics authorization')
    p=verify(prepared,expected);m=read(p['manifest'])
    data=load_source(m['sourcepack'],m['sourcepack_sha256'],m['source_prepared'],m['source_prepare_sha256'])
    store=Store(output);store.put('execution_binding.json',dict(prepared=str(Path(prepared).resolve()),prepare_sha256=expected))
    return execute(p,data,store,factory,authorization=authorization)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','run'))
    for key in ('manifest','manifest-sha','prepared','prepare-sha','output'):parser.add_argument('--'+key)
    parser.add_argument('--authorize-physics',action='store_true');a=parser.parse_args()
    if a.command=='prepare':r=prepare(a.manifest,a.manifest_sha,a.output)
    elif a.command=='verify':r=dict(schema=verify(a.prepared,a.prepare_sha)['schema'])
    else:
        require(a.authorize_physics,'MAIN CLI authorization')
        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401
            return gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224,with_target=False,render_action=False)
        r=run(a.prepared,a.prepare_sha,a.output,factory,authorization=AUTH)
    print(source.canonical({k:v for k,v in r.items() if k!='rows'}))
    if a.command=='run' and r['status']!='physical_calibration_recorded':raise SystemExit(2)


if __name__=='__main__':main()
