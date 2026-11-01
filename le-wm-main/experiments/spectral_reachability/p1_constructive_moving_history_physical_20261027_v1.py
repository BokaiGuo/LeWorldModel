"""MAIN-only moving-history selected-plan consumer; no zero-branch patching.

Source adapter follows current moving source, fails closed on schema drift.
Compiler mapping is explicit JSON-pointer manifest, not guessed producer fields.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import shutil
import os
import time
import numpy as np
from experiments.spectral_reachability import p1_constructive_plan_physical_20261027_v1 as low
from experiments.spectral_reachability import p1_constructive_moving_history_source_20261027_v1 as moving

NAME='p1_constructive_moving_history_physical_20261027_v1'
AUTH='MAIN_MOVING_HISTORY_SELECTED_PHYSICS_ONLY'
MOVING_SOURCE_SHA='9fb76ecfa054f878ab9778fc29c59bc708728b240dd2cbc3060b166e52622f55'
ROOT=moving.ROOT
require,sha,read,raw,source=low.require,low.sha,low.read,low.raw,low.source
commands,pointer,journal,Store,Ops,metric,retain=low.commands,low.pointer,low.journal,low.Store,low.Ops,low.metric,low.retain
FLOOR=low.FLOOR


def raw_history(value):
    require(isinstance(value,np.ndarray) and moving.exact(value,np.asarray([[.1,0]]*5+[[0,.1]]*5,np.float32)),
        'exact moving payload float32 raw10, never zero/tanh/normalize/round')
    require(moving.exact(value,moving.warmup_commands()),'frozen source warmup_commands equals actual payload')
    return value.copy()


def flat(directory,name):
    require(type(name) is str and Path(name).name==name and name not in ('','.','..'),'flat source artifact')
    return Path(directory)/name


def references(pack,p,events,blind):
    """Saved evaluator metadata only; no certificate or witness candidate extraction."""
    require(pack['schema']==moving.NAME+'.sourcepack' and pack['status']=='fixed_slots_processed' and pack['errors']==[] and
        pack['first_error'] is None and pack['sourcepack_complete'] is True and pack['reachable_count']==2 and
        pack['prepare_id']==source.digest(p) and len(pack['rows'])==len(p['slots'])==len(blind)==2,'complete new moving source')
    require(events[0]['prepare']==p and events[-1]['status']==pack['status'] and events[-1]['counts']==pack['counts'] and events[-1]['errors']==[],'source prepare/terminal receipts')
    refs=[]
    for i,(row,b) in enumerate(zip(pack['rows'],blind,strict=True)):
        require(row['slot']==p['slots'][i] and row['slot']['index']==i and row['reachable'] is True and
            row['status']=='reachable_native_replayed' and row['replay_native_success']==[True,True],'two fixed independent source replays')
        history=raw_history(b['raw_action_history'])
        require(moving.exact(np.asarray(p['raw_warmup'],np.float32),history),'source fixed raw-history contract')
        observed=[]
        for phase in ('replay0','replay1'):
            obs=[e for e in events if e['kind']=='fresh_observation' and e.get('slot')==i and e['phase']==phase and e['step'] in (0,5,10)]
            warm=[e for e in events if e['kind']=='warmup_return' and e.get('slot')==i and e['phase']==phase]
            require([e['step'] for e in obs]==[0,5,10] and [e['step'] for e in warm]==list(range(1,11)) and
                [e['raw_float32_bits'] for e in warm]==[a.tobytes().hex() for a in history] and
                moving.exact(np.asarray([e['raw_action'] for e in warm],np.float32),history),'all source actual10 raw and observed boundaries')
            require([e['pixel_sha256'] for e in obs]==[moving.image_sha(a) for a in b['rgb_history']] and
                all(e['snapshot']['state']==e['state'] for e in obs),'source RGB/state/snapshot bridge')
            observed.append(obs)
        require(all(observed[0][j]['state']==observed[1][j]['state'] and observed[0][j]['snapshot']==observed[1][j]['snapshot'] for j in range(3)),'both replays exact exported dynamics')
        future=[e for e in events if e['kind']=='native_return' and e.get('slot')==i and e['phase']=='construction']
        require(future and [e['step'] for e in future]==list(range(1,len(future)+1)) and len(future)<=10,'original derived task endpoint only')
        goal=source.vector(future[-1]['state']);root=source.vector(observed[0][-1]['state'])
        end=[e for e in events if e['kind']=='fresh_observation' and e.get('slot')==i and e['phase']=='construction' and e['step']==10+len(future)]
        require(len(end)==1 and end[0]['pixel_sha256']==moving.image_sha(b['goal_rgb']) and
            source.root_hash(root,goal)==row['root_sha256'],'derived new goal RGB/state/root identity')
        refs.append(dict(slot=deepcopy(row['slot']),root_sha256=row['root_sha256'],goal=goal,goal_rgb=b['goal_rgb'].copy(),
            history=b['rgb_history'].copy(),raw_history=history,states=np.asarray([e['state'] for e in observed[0]]),
            dynamics=[e['snapshot'] for e in observed[0]],blind_sha256=row['blind_payload_sha256']))
    return refs


def load_source(pack_path,pack_sha,prepared,prepare_sha):
    require(sha(moving.__file__)==MOVING_SOURCE_SHA,'MAIN frozen moving source interface')
    bindings={}
    def bind(path,h):
        path=Path(path).resolve();require(sha(path)==h,'source seal: '+str(path));bindings[str(path)]=h;return path
    path=bind(pack_path,pack_sha);pack=read(path)
    p=moving.read_prepared(bind(prepared,prepare_sha).read_bytes())
    binding=read(bind(path.parent/'execution_binding.json',sha(path.parent/'execution_binding.json')))
    require(binding==dict(prepared=str(Path(prepared).resolve()),prepare_file_sha256=prepare_sha),'source execution binding')
    events=journal(bind(path.parent/'journal.jsonl',sha(path.parent/'journal.jsonl')),pack['journal_chain_tail'])
    blind=[]
    for row in pack['rows']:
        file=bind(flat(path.parent,row['blind_payload']),row['blind_payload_sha256'])
        require(file.stat().st_size<2*1024**2,'small approved blind NPZ')
        blind.append(moving.load_blind(file,row['blind_payload_sha256']))
    refs=references(pack,p,events,blind);bindings.update(p['bindings'])
    return dict(plan=p,pack=pack,references=refs,bindings=bindings)


def selection(manifest,data):
    require(manifest['schema']==NAME+'.selection' and len(manifest['arms'])==2,'new manifest, exactly two fixed arms')
    # Reuse ONLY schema-independent saved mapping checks. No old material/load_source/branch/execute.
    adapted=deepcopy(manifest);adapted['schema']=low.NAME+'.selection'
    return low.selection(adapted,data)


def contract(wall):
    c=low.contract(4,wall);c.update(warmup_raw=np.asarray([[.1,0]]*5+[[0,.1]]*5,np.float32).tolist(),
        scope='reused two seed/placement recipes, NEW moving roots/tasks; fixed two arms, no witness candidate or P1 admission',
        source_interface='current moving source; external prepare/source/sourcepack seals required; reject drift')
    return c


def material(manifest_path,manifest_sha):
    require(sha(manifest_path)==manifest_sha,'external new selection manifest SHA');m=read(manifest_path)
    require(m['schema']==NAME+'.selection','new moving manifest, not old zero source')
    data=load_source(m['sourcepack'],m['sourcepack_sha256'],m['source_prepared'],m['source_prepare_sha256'])
    plans,bindings=selection(m,data);bindings.update(data['bindings']);bindings[str(Path(manifest_path).resolve())]=manifest_sha
    files=[Path(__file__),Path(low.__file__),Path(low.builder.__file__),Path(source.__file__),Path(moving.__file__),
        ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]
    bindings.update({str(f.resolve()):sha(f) for f in files})
    return dict(schema=NAME+'.prepare',manifest=str(Path(manifest_path).resolve()),manifest_sha256=manifest_sha,
        plans=plans,contract=contract(m['wall_seconds']),bindings=bindings,source_prepare_id=source.digest(data['plan']),
        sourcepack_sha256=m['sourcepack_sha256'],source_slots=m['roots'])


def prepare(manifest,manifest_sha,output):
    p=material(manifest,manifest_sha);value=raw(p);path=Path(output)
    require(len(value)<256*1024 and shutil.disk_usage(path.parent).free>=FLOOR+len(value),'small exclusive prepare/200MiB reserve')
    with path.open('xb') as f:f.write(value);f.flush();os.fsync(f.fileno())
    return dict(prepared=str(path.resolve()),prepare_sha256=sha(path),bytes=len(value))


def verify(path,expected):
    require(sha(path)==expected,'external consumer prepare SHA');p=read(path)
    require(p==material(p['manifest'],p['manifest_sha256']),'full moving source/commands/closure reconstruction');return p


def capture(env,obs,phase,step,row,ops,store,index):
    body=env.unwrapped;before=moving.snapshot(body);a=np.asarray(ops.call('explicit_render',env.render))
    if a.dtype!=np.uint8 or a.shape!=(224,224,3):
        artifact=f'b{index}_{phase}_{step:02}_invalid.npz';h=None
        if a.dtype.kind in 'buif':h=store.arrays(artifact,raw_render=a)
        row['last_observation']=dict(phase=phase,step=step,shape=list(a.shape),dtype=str(a.dtype),raw_render=artifact if h else None,file_sha256=h,snapshot=moving.snapshot(body))
        store.emit('invalid_render',**ops.where,receipt=row['last_observation']);raise ValueError('invalid render retained')
    a=a.copy();name=f'b{index}_{phase}_{step:02}.png';h=store.frame(name,a);row['PNG']+=1
    state=source.vector(body._get_obs());snap=moving.snapshot(body)
    receipt=dict(phase=phase,step=step,png=name,file_sha256=h,pixel_sha256=moving.image_sha(a),state=state.tolist(),snapshot=snap,observation_state=np.asarray(obs['state']).tolist())
    row['last_observation']=receipt;store.emit('fresh_observation',**ops.where,receipt=receipt)
    pixels=int(np.count_nonzero(np.all(a==[65,105,225],axis=-1)))
    accepted=pixels>0 and bool(((state[:4]>=0)&(state[:4]<=512)).all()) and moving.exact(state,np.asarray(obs['state'])) and snap==before
    store.emit('guard_receipt',**ops.where,phase=phase,step=step,accepted=accepted,canonical_agent_pixels=pixels,render_preserved_dynamics=snap==before)
    require(accepted,'RGB/canvas/state/pure-render guard; failed image retained')
    if phase=='plan' and step in (0,5,10):row['checkpoints'].append(receipt)
    return a,snap


def branch(env,ref,source_plan,plan,row,ops,store,index):
    """Own new kernel: replay actual payload10, then own selected commands10."""
    body=env.unwrapped;slot=ref['slot'];warm=raw_history(ref['raw_history']);history=[];states=[];dynamics=[];executed=[]
    ops.where=dict(branch=index,root_slot=plan['root_slot'],arm=plan['arm'],selection_digest=plan['selection_digest'])
    store.emit('branch_started',**ops.where,plan=plan,raw_warmup=warm.tolist(),raw_float32_bits=warm.tobytes().hex())
    _,info=ops.call('reset',lambda:env.reset(seed=slot['reset_seed']),kind_of_reset='original_seed',seed=slot['reset_seed'])
    original_goal=source.vector(info['goal_state'])
    requested=source.sample_state(original_goal,list(body.block.shapes),dict(root_seed=slot['root_seed']),0,0,source_plan)
    store.emit('placement_request',**ops.where,original_goal=original_goal.tolist(),requested=requested.tolist(),placement_seed=0)
    obs,info=ops.call('reset',lambda:env.reset(seed=0,options=dict(state=requested.copy(),goal_state=original_goal.copy())),kind_of_reset='placement',seed=0)
    frame,snap=capture(env,obs,'warmup',0,row,ops,store,index);history.append(frame);states.append(source.vector(obs['state']));dynamics.append(snap)
    for step,command in enumerate(warm,1):
        require(not body.eval_state(original_goal,source.vector(obs['state']))[0],'native terminal before moving warmup dispatch')
        obs,reward,terminated,truncated,info=ops.call('public_step',lambda c=command:env.step(c.copy()),phase='warmup',step=step,raw_action=command.tolist())
        row['warmup_returned']+=1
        store.emit('returned_transition',**ops.where,phase='warmup',step=step,state=source.vector(obs['state']).tolist(),reward=float(reward),terminated=bool(terminated),truncated=bool(truncated))
        frame,snap=capture(env,obs,'warmup',step,row,ops,store,index)
        require(moving.exact(np.asarray(body.latest_action),command),'native raw warmup substitution')
        executed.append(np.asarray(body.latest_action).copy());store.emit('warmup_return',**ops.where,step=step,raw_action=executed[-1].tolist(),raw_float32_bits=executed[-1].tobytes().hex())
        require(not terminated and not truncated,'moving warmup stopped; no plan execution')
        if step in (5,10):history.append(frame);states.append(source.vector(obs['state']));dynamics.append(snap)
    require(moving.exact(np.asarray(executed),warm) and moving.exact(np.asarray(history),ref['history']) and
        moving.exact(np.asarray(states),ref['states']) and dynamics==ref['dynamics'],'exact moving0/5/10 RGB/state/snapshot/raw reconstruction')
    goal,goal_rgb=ref['goal'],ref['goal_rgb'];root=source.vector(obs['state'])
    require(source.root_hash(root,goal)==ref['root_sha256'],'NEW moving root+goal identity')
    before=moving.snapshot(body);counts=deepcopy(ops.counts);pose=deepcopy(getattr(body,'goal_pose',None))
    ops.call('goal_bind',lambda:moving.attach_goal(body,goal,goal_rgb))
    require(moving.snapshot(body)==before and all(v==ops.counts[k] for k,v in counts.items() if k!='goal_bind') and
        np.array_equal(getattr(body,'goal_pose',None),pose),'offline goal bind no step/setter/render or pose change')
    frame,_=capture(env,obs,'plan',0,row,ops,store,index)
    require(moving.exact(frame,history[-1]) and np.array_equal(body.goal_state,goal) and np.array_equal(body._goal,goal_rgb),'target-free pixels and actual goal attachment')
    row.update(initial_exact=True,root_state=root.tolist(),goal=goal.tolist(),initial_metrics=metric(goal,root))
    require(not row['initial_metrics']['native_success'],'derived goal already reached at moving root')
    store.emit('goal_attached',**ops.where,goal=goal.tolist(),goal_rgb_sha256=moving.image_sha(goal_rgb),snapshot=before)
    for step,command in enumerate(commands(plan['commands']),1):
        def dispatch(c=command):row['own_attempted']+=1;return env.step(c.copy())
        obs,reward,terminated,truncated,info=ops.call('public_step',dispatch,phase='plan',step=step,raw_action=command.tolist())
        row['own_returned']+=1;state=source.vector(obs['state']);m=metric(goal,state);success,distance=body.eval_state(goal,state)
        row.update(final_state=state.tolist(),final_metrics=m,native_success=bool(success),terminated=bool(terminated),truncated=bool(truncated))
        store.emit('returned_transition',**ops.where,phase='plan',step=step,command=command.tolist(),state=state.tolist(),reward=float(reward),terminated=bool(terminated),truncated=bool(truncated),
            returned_goal_state=np.asarray(info['goal_state']).tolist(),returned_goal_rgb_sha256=moving.image_sha(info['goal']))
        capture(env,obs,'plan',step,row,ops,store,index)
        require(moving.exact(np.asarray(body.latest_action),command),'own raw selected command substitution')
        require(np.array_equal(info['goal_state'],goal) and np.array_equal(info['goal'],goal_rgb),'native goal remains bound')
        require(bool(success)==m['native_success'] and float(distance)==m['native_distance'] and bool(terminated)==bool(success) and float(reward)==-float(distance),'native eval/reward/terminal unchanged')
        if terminated or truncated:row['status']='native_success' if terminated and success and not truncated else 'native_stopped';break
    else:row['status']='horizon_completed'
    row['full_horizon']=row['own_returned']==10
    row['endpoint_scope']='native_terminal' if row['terminated'] else 'full10' if row['full_horizon'] else 'partial'


@contextmanager
def native_hooks(env,ops,row):
    body=env.unwrapped;undo=[];spaces=[];resetting=False
    old_set,old_render,old_reset=body._set_state,body.render,env.reset
    def outside_goal_bind():
        c=ops.counts['goal_bind'];require(c['attempted']==c['completed'],'no physics/render during offline goal bind')
    def install():
        space=body.space
        if any(x[0] is space for x in spaces):return
        old=space.step;spaces.append((space,'step' in space.__dict__,old))
        def physics(*a,**kw):
            outside_goal_bind();return ops.call('physics_step',lambda:old(*a,**kw))
        space.step=physics
    def setter(value):
        require(resetting,'post-reset state setter forbidden');install();return ops.call('set_state',lambda:old_set(value))
    def render():
        outside_goal_bind();return ops.call('goal_render',old_render) if resetting else old_render()
    def reset(*a,**kw):
        nonlocal resetting
        resetting=True
        try:return old_reset(*a,**kw)
        finally:resetting=False
    try:
        for obj,key,new in ((body,'_set_state',setter),(body,'render',render),(env,'reset',reset)):
            undo.append((obj,key,key in obj.__dict__,getattr(obj,key)));setattr(obj,key,new)
        yield
    except BaseException as exc:retain(row,exc,'branch')
    finally:
        for obj,owned,old in reversed(spaces):
            try:
                if owned:obj.step=old
                else:delattr(obj,'step')
            except BaseException as exc:retain(row,exc,'restore_space')
        for obj,key,owned,old in reversed(undo):
            try:
                if owned:setattr(obj,key,old)
                else:delattr(obj,key)
            except BaseException as exc:retain(row,exc,'restore_hook')


def execute(p,data,store,factory,*,authorization):
    require(authorization==AUTH,'explicit MAIN physics authorization')
    require(len(p['plans'])==4 and len(data['references'])==2,'fixed four branches/two roots')
    rows=[dict(root_slot=x['root_slot'],arm=x['arm'],selection_digest=x['selection_digest'],status='not_started',first_error=None,secondary_errors=[],
        warmup_returned=0,own_attempted=0,own_returned=0,initial_exact=False,full_horizon=False,native_success=False,terminated=False,truncated=False,PNG=0,checkpoints=[]) for x in p['plans']]
    result=dict(schema=NAME+'.report',status='running',rows=rows,errors=[],contract=p['contract'],sourcepack_sha256=p['sourcepack_sha256'],
        P1_admission=False,training=False,new_neural_calls=0,new_independent_samples=0,global_freshness='reused_recipe_new_moving_tasks_calibration')
    ops=Ops(p,store)
    def error(stage,exc):result['errors'].append(dict(stage=stage,type=type(exc).__name__,message=str(exc)))
    try:
        store.emit('run_started',prepare=p)
        for i,(plan,row) in enumerate(zip(p['plans'],rows,strict=True)):
            env=None
            try:
                store.check(262144);env=factory();moving.validate_env(env.unwrapped)
                require(env.unwrapped.render_action is False and env.unwrapped.render_size==224 and env.unwrapped.render_mode=='rgb_array','native target-free/no action-marker RGB224')
                row['status']='running'
                with native_hooks(env,ops,row):branch(env,data['references'][plan['root_slot']],data['plan'],plan,row,ops,store,i)
            except BaseException as exc:retain(row,exc,'setup_or_branch')
            finally:
                if env is not None:
                    try:env.close()
                    except BaseException as exc:retain(row,exc,'close')
                store.emit('branch_closed',receipt=row)
    except BaseException as exc:error('execution',exc)
    try:
        for path,h in p['bindings'].items():require(sha(path)==h,'final source seal: '+path)
    except BaseException as exc:error('final_bindings',exc)
    clean=not result['errors'] and all(r['status'] in ('horizon_completed','native_success','native_stopped') for r in rows)
    if clean:
        try:
            c=ops.counts
            require(all(v['attempted']==v['completed'] for v in c.values()) and c['reset']['completed']==8 and c['set_state']['completed']==16 and c['goal_render']['completed']==8 and c['goal_bind']['completed']==4 and
                c['public_step']['completed']==sum(r['warmup_returned']+r['own_returned'] for r in rows) and
                c['physics_step']['completed']==16+10*c['public_step']['completed'] and c['explicit_render']['completed']==sum(r['PNG'] for r in rows),'actual full-denominator internal resource census')
        except BaseException as exc:error('final_census',exc);clean=False
    result.update(status='moving_history_physical_recorded' if clean else 'partial_or_failed',counts=ops.counts,native_successes=sum(r['status']=='native_success' for r in rows),wall_seconds=time.monotonic()-ops.started)
    try:store.emit('run_finished',status=result['status'],counts=ops.counts,errors=result['errors']);result['journal_sha256']=sha(store.path/'journal.jsonl');result['journal_chain_tail']=store.previous
    except BaseException as exc:error('final_journal',exc);result['status']='partial_or_failed'
    try:store.put('report.json',result,final=True)
    except BaseException as exc:error('final_report',exc);result['status']='partial_or_failed'
    return result


def run(prepared,expected,output,factory,*,authorization):
    require(authorization==AUTH,'MAIN authorization');p=verify(prepared,expected);m=read(p['manifest'])
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
        if r['status']!='moving_history_physical_recorded':print(source.canonical(r));raise SystemExit(2)
    print(source.canonical({k:v for k,v in r.items() if k!='rows'}))


if __name__=='__main__':main()
