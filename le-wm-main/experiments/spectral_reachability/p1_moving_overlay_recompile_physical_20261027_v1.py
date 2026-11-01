"""MAIN-only eight newly recompiled moving-overlay plans; zero neural calls.

Independent physical kernel. Reuses only saved-source/placement/goal utilities,
never any old execute/episode/branch and never changes another module's globals.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import p1_moving_target_overlay_source_20261027_v1 as paired

NAME = 'p1_moving_overlay_recompile_physical_20261027_v1'
AUTH = 'MAIN_MOVING_OVERLAY_RECOMPILE_PHYSICS_ONLY'
ROOT, moving, previous = paired.ROOT, paired.moving, paired.previous
source = moving.source
require, sha, raw = paired.require, previous.sha, moving.raw
place, bind_goal, invariant = paired.place, paired.bind_goal, paired.invariant
old_image, validate_env = paired.old_image, moving.validate_env
CAP, FLOOR, FINAL = 64*1024**2, 200*1024**2, 2*1024**2
SOURCE_REPORT = ROOT/'results/p1_moving_target_overlay_source_20261027_v1_main01/report.json'
SOURCE_SHA = '85ca1df3de910628bfd6c5bd1a0a482f61880a2ee76bfd404732a77dd8c7666a'
ARMS = ('line_search_GN', 'full10_conditional_D2P')
CONDITIONS = ('off', 'on')
PLAN_KEYS = {'root_slot','condition','arm','commands','selection_digest','task_id',
             'blind_path','blind_sha256','source_report_sha256','planner_report_sha256'}


def read(path):
    def pairs(items):
        result = {}
        for k,v in items:
            require(k not in result, 'duplicate JSON key'); result[k] = v
        return result
    def constant(v):
        raise ValueError('nonfinite JSON constant: '+v)
    path = Path(path)
    require(path.stat().st_size <= 2*1024**2, 'bounded metadata')
    result = json.loads(path.read_bytes(), object_pairs_hook=pairs, parse_constant=constant)
    raw(result)
    return result


def hash_value(h):
    require(type(h) is str and len(h)==64 and all(c in '0123456789abcdef' for c in h), 'literal SHA256 required')


def contract():
    return dict(fixed_denominator=8, slots=[0,1], conditions=list(CONDITIONS), arms=list(ARMS),
        warmup=10, horizon=10, checkpoints=[0,5,10], state_dimension=7,
        with_target=False, flags=[False,True,False], no_goal_pose_edit=True,
        wall_seconds=900, output_cap_bytes=CAP, disk_floor_bytes=FLOOR, final_reserve_bytes=FINAL,
        caps=dict(constructor=8,close=8,reset=16,public_step=160,physics_step=1632,
                  set_state=32,goal_setter=24,goal_render=16,goal_bind=8,explicit_render=528),
        rendering='every warmup0..10 and plan0..10 uses off/on/off; chosen condition is the label',
        on_semantics='native default outline is NOT constructed task goal; goalRGB comes from paired true source endpoint',
        no_neural=True,training=False,Q_fit=False,P1_admission=False)


def validate_plans(plans, planner_sha, exports):
    hash_value(planner_sha)
    require(type(plans) is list and len(plans)==8, 'all eight new compiler plans required')
    expected=[(s,c,a) for s in (0,1) for c in CONDITIONS for a in ARMS]
    for p,key in zip(plans,expected,strict=True):
        require(type(p) is dict and PLAN_KEYS<=set(p), 'required compiler consume plan fields')
        require(type(p['root_slot']) is int and (p['root_slot'],p['condition'],p['arm'])==key, 'fixed slot-major/condition/arm order')
        previous.commands(p['commands'])  # Lossless raw float32 only; never repair.
        for k in ('selection_digest','task_id','blind_sha256','source_report_sha256','planner_report_sha256'):
            hash_value(p[k])
        require(p['source_report_sha256']==SOURCE_SHA and p['planner_report_sha256']==planner_sha, 'externally pinned source/planner join')
        e=exports[(key[0],key[1])]
        require(Path(p['blind_path']).is_absolute() and p['blind_path']==e['path'] and p['blind_sha256']==e['sha256'], 'same paired blind path/SHA, not a cross-condition packet')
        if 'with_target' in p: require(p['with_target'] is (key[1]=='on'), 'compiler rendering condition mismatch')
        if 'selection_uses_physics' in p: require(p['selection_uses_physics'] is False, 'no physical-outcome selection')
        if 'P1_admission' in p: require(p['P1_admission'] is False, 'compiler does not admit P1')
    return [{k:deepcopy(p[k]) for k in PLAN_KEYS} for p in plans]


def source_material():
    """Read saved metadata/PNG/NPZ only; never old producer execution or witness()."""
    require(sha(SOURCE_REPORT)==SOURCE_SHA, 'fixed paired source report external SHA')
    report=read(SOURCE_REPORT); directory=SOURCE_REPORT.parent
    require(report['schema']==paired.NAME+'.report' and report['status']=='paired_moving_overlay_recorded'
            and report['errors']==[] and report['full_protocol'] is True, 'complete paired moving source required')
    bp=directory/'execution_binding.json'; binding=read(bp)
    require(set(binding)=={'prepared','prepare_file_sha256'} and Path(binding['prepared']).is_absolute(), 'paired source execution binding')
    p=paired.load_prepared(binding['prepared'],binding['prepare_file_sha256'])
    require(p['contract']==paired.contract() and report['prepare_id']==source.digest(p), 'paired prepare identity')
    data=previous.load_source(paired.SD/'sourcepack.json',paired.PINNED[str(paired.SD/'sourcepack.json')],
                              paired.SP,paired.PINNED[str(paired.SP)])
    require(p['slots']==data['plan']['slots'], 'original moving placement recipe slots')
    bindings={**p['bindings'],**data['bindings']}
    def bind(path, expected=None):
        path=str(Path(path).resolve()); h=sha(path)
        require(expected is None or h==expected, 'saved source artifact SHA')
        require(path not in bindings or bindings[path]==h, 'source seal conflict')
        bindings[path]=h; return path
    bind(SOURCE_REPORT,SOURCE_SHA); bind(bp); bind(binding['prepared'],binding['prepare_file_sha256'])
    jp=bind(directory/'journal.jsonl'); events=previous.journal(jp,report['journal_chain_tail'])
    require([(r['source_slot'],r['phase']) for r in report['rows']]==
            [(s,phase) for s in (0,1) for phase in ('construction','replay0','replay1')]
            and all(r['status']=='source_episode_completed' for r in report['rows']), 'source six fixed phases')
    require(len(report['blind_exports'])==2, 'two paired source exports')
    roots=[]; exports={}
    for s,export in enumerate(report['blind_exports']):
        require(type(export['source_slot']) is int and export['source_slot']==s and
                export['old_root_sha256']==data['references'][s]['root_sha256'], 'source stable slot/root')
        packets={}
        for condition in CONDITIONS:
            e=export['artifacts'][condition]; path=bind(paired.flat(directory,e['path']),e['sha256'])
            packets[condition]=moving.load_blind(path,e['sha256'])
            exports[(s,condition)]=dict(path=path,sha256=e['sha256'])
        warm={}
        for step in range(11):
            found=[e['receipt'] for e in events if e['kind']=='fresh_overlay_frame' and
                   e.get('key')==f's{s}_replay0' and e['receipt']['reference_key']==str(step) and e['receipt']['overlay_index']==0]
            require(len(found)==1, 'all paired off warmup observations')
            f=found[0]; require(f['pre']==f['post'] and all(f['old_off_match'].values()), 'paired off source invariants')
            path=bind(paired.flat(directory,f['png']),f['file_sha256'])
            warm[step]=dict(path=path,file_sha256=f['file_sha256'],pixel_sha256=f['pixel_sha256'],
                            state=f['post']['snapshot']['state'],snapshot=f['post']['snapshot'])
        ref=data['references'][s]
        require(moving.exact(packets['off']['rgb_history'],ref['history']) and
                moving.exact(packets['off']['goal_rgb'],ref['goal_rgb']), 'paired off original root/history/goal')
        roots.append(dict(slot=deepcopy(ref['slot']),goal=ref['goal'].copy(),root_sha256=ref['root_sha256'],
                          packets=packets,warm=warm))
    for path,h in bindings.items(): require(sha(path)==h, 'source closure drift')
    return dict(source_plan=data['plan'],roots=roots,exports=exports,bindings=bindings)


def own_files():
    return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md'),
            ROOT.parent/'docs/D2P_MOVING_OVERLAY_RECOMPILE_PROTOCOL_20261027.md',
            Path(paired.__file__),Path(previous.__file__),Path(previous.low.__file__),
            Path(moving.__file__),Path(source.__file__),Path(moving.old.__file__),Path(previous.low.builder.__file__)]


def material(planner_output, planner_report_sha, consume=None):
    """Saved-only MAIN preparation boundary; no invented future report SHA."""
    hash_value(planner_report_sha)
    output=Path(planner_output).resolve(); rp=output/'report.json'
    require(sha(rp)==planner_report_sha, 'external real compiler report SHA')
    compiler_file=None
    if consume is None:
        from experiments.spectral_reachability import p1_moving_overlay_recompile_cuda_20261027_v1 as compiler
        consume=compiler.consume; compiler_file=Path(compiler.__file__).resolve()
    data=source_material()
    handoff=consume(output,planner_report_sha)
    plans=validate_plans(handoff,planner_report_sha,data['exports'])
    bindings=deepcopy(data['bindings']); bindings[str(rp)]=planner_report_sha
    for plan in plans: bindings[plan['blind_path']]=plan['blind_sha256']
    for path in own_files()+([] if compiler_file is None else [compiler_file]): bindings[str(path.resolve())]=sha(path)
    p=dict(schema=NAME+'.prepare',contract=contract(),planner_output=str(output),
           planner_report_sha256=planner_report_sha,source_report_sha256=SOURCE_SHA,
           plans=plans,compiler_handoff_sha256=source.digest(handoff),bindings=bindings)
    return p,data


def prepare(output, planner_output, planner_report_sha):
    p,_=material(planner_output,planner_report_sha); value=raw(p); path=Path(output).absolute()
    require(len(value)<=1024**2 and not path.exists() and shutil.disk_usage(path.parent).free>=FLOOR+len(value), 'exclusive bounded prepare +200MiB floor')
    with path.open('xb') as f: require(f.write(value)==len(value),'short prepare'); f.flush(); os.fsync(f.fileno())
    paired.fsync_dir(path.parent)
    return dict(prepared=str(path),prepare_sha256=sha(path),bytes=len(value))


def verify(prepared, expected):
    hash_value(expected); require(sha(prepared)==expected, 'external consumer prepare SHA')
    p=read(prepared); now,data=material(p['planner_output'],p['planner_report_sha256'])
    require(p==now and sha(prepared)==expected, 'complete own prepare closure/commands/source binding')
    return p,data


class Store(moving.Store):
    """64MiB exclusive durable store, not inherited old 8MiB write limits."""
    def __init__(self,path):
        self.path=Path(path).absolute()
        require(not self.path.exists() and not self.path.is_symlink() and shutil.disk_usage(self.path.parent).free>=FLOOR+CAP, 'new64MiB output +200MiB reserve')
        self.path.mkdir(); paired.fsync_dir(self.path.parent); self.previous=None; self.index=0

    def check(self,size=4096,final=False):
        used=sum(f.stat().st_size for f in self.path.iterdir() if f.is_file())
        require(used+size<=CAP-(0 if final else FINAL) and shutil.disk_usage(self.path).free>=FLOOR+size, '64MiB output /200MiB disk floor')

    def write(self,name,value,append=False,final=False):
        require(type(name) is str and Path(name).name==name and name not in ('','.','..') and not (self.path/name).is_symlink(), 'flat exclusive nonsymlink output')
        self.check(len(value),final)
        with (self.path/name).open('ab' if append else 'xb') as f:
            require(f.write(value)==len(value),'short artifact write'); f.flush(); os.fsync(f.fileno())
        paired.fsync_dir(self.path); return sha(self.path/name)


class Meter:
    def __init__(self,store,started=None):
        self.store=store; self.started=time.monotonic() if started is None else started; self.where={}; self.secondary=[]
        self.counts={k:dict(attempted=0,completed=0) for k in contract()['caps']}
        self.active={k:0 for k in self.counts}

    def call(self,kind,fn,*,cleanup=False,**fields):
        require(not cleanup or kind=='close', 'only nonphysical close may bypass exhausted predispatch budgets')
        if not cleanup:
            require(time.monotonic()-self.started<900, '900s wall cap before dispatch')
            self.store.check()
        c=self.counts[kind]
        require(c['attempted']<contract()['caps'][kind], 'predispatch resource cap: '+kind)
        fields={**self.where,**fields}; c['attempted']+=1
        def emit(event):
            try: self.store.emit(event,resource=kind,**fields)
            except BaseException as e:
                if not cleanup: raise
                self.secondary.append(error('cleanup_journal',e))
        emit('operation_attempted')
        self.active[kind]+=1
        try:
            result=fn(); c['completed']+=1
        except BaseException as e:
            try: self.store.emit('operation_failed',resource=kind,type=type(e).__name__,message=str(e),**fields)
            except BaseException as journal_error: self.secondary.append(error('operation_error_journal',journal_error))
            raise
        finally: self.active[kind]-=1
        emit('operation_completed')
        return result


def error(stage,e): return dict(stage=stage,type=type(e).__name__,message=str(e))


@contextmanager
def native_hooks(env,meter,errors):
    body=env.unwrapped; undo=[]; spaces=[]; resetting=False
    old_set,old_goal,old_render,old_reset=body._set_state,body._set_goal_state,body.render,env.reset
    def idle():
        require(not meter.active['explicit_render'] and not meter.active['goal_bind'], 'no hidden physics/setter during render/goal bind')
    def install():
        space=body.space
        if any(item[0] is space for item in spaces): return
        old=space.step; spaces.append((space,'step' in space.__dict__,old))
        def physics(*a,**kw):
            idle(); return meter.call('physics_step',lambda:old(*a,**kw))
        space.step=physics
    def setter(value):
        require(resetting and body.with_target is False, 'state setter only original off reset')
        install(); return meter.call('set_state',lambda:old_set(value))
    def goalsetter(value):
        require(resetting or meter.active['goal_bind'], 'goal setter only reset or single goal bind')
        before=moving.snapshot(body); pose=np.asarray(body.goal_pose).copy()
        result=meter.call('goal_setter',lambda:old_goal(value))
        require(moving.snapshot(body)==before and moving.exact(np.asarray(body.goal_pose),pose), 'goal setter changed dynamics/goal_pose')
        return result
    def render():
        require(not meter.active['goal_bind'], 'no hidden rendering during goal bind')
        return meter.call('goal_render' if resetting else 'explicit_render',old_render)
    def reset(*a,**kw):
        nonlocal resetting
        require(body.with_target is False, 'native resets off'); resetting=True
        try: return old_reset(*a,**kw)
        finally: resetting=False
    try:
        for obj,key,new in ((body,'_set_state',setter),(body,'_set_goal_state',goalsetter),(body,'render',render),(env,'reset',reset)):
            undo.append((obj,key,key in obj.__dict__,getattr(obj,key))); setattr(obj,key,new)
        yield
    finally:
        try: body.with_target=False
        except BaseException as e: errors.append(error('restore_flag',e))
        for obj,owned,old in reversed(spaces):
            try:
                if owned: obj.step=old
                else: delattr(obj,'step')
            except BaseException as e: errors.append(error('restore_space',e))
        for obj,key,owned,old in reversed(undo):
            try:
                if owned: setattr(obj,key,old)
                else: delattr(obj,key)
            except BaseException as e: errors.append(error('restore_hook',e))


def capture(env,obs,phase,step,row,meter,store,reference=None):
    """Fresh triplet durable before rejection; primary labels use row condition."""
    body=env.unwrapped; fixed=invariant(body); frames={}; receipts={}; first=None
    try:
        for index,flag in enumerate((False,True,False)):
            body.with_target=flag; before=invariant(body)
            a=np.asarray(env.render()); after=invariant(body)
            name=f'b{row["index"]}_{phase}_{step:02}_{index}.png'
            if a.shape!=(224,224,3) or a.dtype!=np.uint8:
                name=name[:-4]+'_invalid.npz'; h=store.arrays(name,raw_render=a)
                store.emit('invalid_observation',**meter.where,phase=phase,step=step,file=name,file_sha256=h)
                raise ValueError('native uint8 RGB224 required')
            a=a.copy(); h=store.frame(name,a)
            state=source.vector(obs['state'])
            require(state.shape==(7,), 'true seven-dimensional state required')
            receipt=dict(phase=phase,step=step,condition='on' if flag else 'off',overlay_index=index,
                         png=name,file_sha256=h,pixel_sha256=moving.image_sha(a),state=state.tolist(),snapshot=after['snapshot'])
            row['last_observation']=deepcopy(receipt); row['PNG']+=1
            store.emit('fresh_observation',**meter.where,receipt=receipt,pre=before,post=after)
            require(before==after==fixed and moving.exact(state,source.vector(body._get_obs())), 'render changed dynamics/goal/renderer/observation')
            if not flag and reference is not None:
                require(h==reference['file_sha256'] and moving.image_sha(a)==reference['pixel_sha256'] and
                        moving.exact(a,old_image(reference)) and moving.exact(state,np.asarray(reference['state'])) and
                        after['snapshot']==reference['snapshot'], 'paired source off physical root/history mismatch')
            moving.guard(a,state)
            if index==0: first=a.copy(); frames['off']=a.copy(); receipts['off']=deepcopy(receipt)
            if index==1: frames['on']=a.copy(); receipts['on']=deepcopy(receipt)
            if index==2: require(moving.exact(first,a), 'off/on/off pixels changed')
    finally: body.with_target=False
    selected=receipts[row['condition']]
    row['trajectory'].append(deepcopy(selected))
    if phase=='plan' and step in (0,5,10):
        row['checkpoints'].append({k:deepcopy(selected[k]) for k in ('step','png','file_sha256','pixel_sha256','state','snapshot')})
        row['missing_checkpoints'].remove(step)
    return frames[row['condition']]


class NativeStop(Exception):
    """Legal native early stop; never pad, retry or substitute an action."""


def dispatch(env,command,phase,step,row,meter,store,goal):
    require(env.unwrapped.with_target is False and command.dtype==np.float32 and command.shape==(2,), 'sealed raw2 off dispatch')
    if phase=='plan': row['own_attempted']+=1
    obs,reward,t,tr,info=meter.call('public_step',lambda:env.step(command.copy()),phase=phase,step=step,raw_bits=command.tobytes().hex())
    row['warmup_returned' if phase=='warmup' else 'own_returned']+=1
    actual=np.asarray(env.unwrapped.latest_action).copy(); state=source.vector(obs['state'])
    store.emit('returned_transition',**meter.where,phase=phase,step=step,requested_bits=command.tobytes().hex(),
        actual_bits=actual.tobytes().hex(),actual_dtype=actual.dtype.str,state=state.tolist(),snapshot=moving.snapshot(env.unwrapped),
        reward=float(reward),terminated=bool(t),truncated=bool(tr),goal_state=np.asarray(info['goal_state']).tolist())
    # Checks happen after fresh capture so returned incident images are retained.
    return obs,info,bool(t),bool(tr),float(reward),actual


def episode(env,root,source_plan,plan,row,meter,store):
    body=env.unwrapped; packet=root['packets'][plan['condition']]
    obs,native_goal=place(env,source_plan,root['slot'],meter,store)
    frames=[capture(env,obs,'warmup',0,row,meter,store,root['warm'][0])]
    for step,command in enumerate(moving.warmup_commands(),1):
        if body.eval_state(native_goal,source.vector(obs['state']))[0]: raise NativeStop('warmup already native terminal')
        obs,info,t,tr,reward,actual=dispatch(env,command,'warmup',step,row,meter,store,native_goal)
        image=capture(env,obs,'warmup',step,row,meter,store,root['warm'][step])
        paired.check_return(env,command,actual,info,native_goal,reward,t)
        if t or tr: raise NativeStop('warmup native stop; no plan or padded history')
        if step in (5,10): frames.append(image)
    require(moving.exact(np.asarray(frames),packet['rgb_history']) and moving.exact(packet['raw_action_history'],moving.warmup_commands()), 'same-condition true source history3/raw10')
    state=source.vector(obs['state']); goal=source.vector(root['goal'])
    require(state.shape==goal.shape==(7,) and source.root_hash(state,goal)==root['root_sha256'], 'exact original moving physical root/goal identity')
    bind_goal(body,goal,packet['goal_rgb'],meter,store)
    bound=invariant(body)
    row.update(initial_exact=True,root_state=state.tolist(),goal=goal.tolist(),goal_pose=bound['goal_pose'],goal_pixel_sha256=moving.image_sha(packet['goal_rgb']))
    capture(env,obs,'plan',0,row,meter,store)
    row['native_success']=bool(body.eval_state(goal,state)[0])
    if row['native_success']: raise NativeStop('native task already terminal at root')
    for step,command in enumerate(previous.commands(plan['commands']),1):
        obs,info,t,tr,reward,actual=dispatch(env,command,'plan',step,row,meter,store,goal)
        capture(env,obs,'plan',step,row,meter,store)
        row['native_success']=paired.check_return(env,command,actual,info,goal,reward,t)
        require(moving.exact(np.asarray(info['goal']),packet['goal_rgb']), 'same-condition attached goalRGB changed')
        now=invariant(body)
        require({k:v for k,v in now.items() if k!='snapshot'}=={k:v for k,v in bound.items() if k!='snapshot'}, 'goal/goal_pose/renderer changed during own plan')
        row.update(terminated=t,truncated=tr,final_state=source.vector(obs['state']).tolist())
        if t or tr:
            row['status']='native_success' if t and not tr and row['native_success'] else 'native_stopped'
            return
    row['status']='horizon_completed'


def census(rows,counts):
    c={k:v['completed'] for k,v in counts.items()}
    require(all(v['attempted']==v['completed'] for v in counts.values()), 'unreturned operation attempts retained')
    require(c['physics_step']==2*c['reset']+10*c['public_step'] and c['set_state']==2*c['reset'] and
            c['goal_render']==c['reset'] and c['goal_setter']==c['reset']+c['goal_bind'] and
            c['public_step']==sum(r['warmup_returned']+r['own_returned'] for r in rows) and
            c['explicit_render']==sum(r['PNG'] for r in rows), 'actual native resource identities')
    require(c['constructor']==c['close']==8, 'eight fresh independent metered environments')


def execute(p,data,store,factory,*,authorization,prepare_sha256,execution_binding,started=None):
    require(authorization==AUTH and p['schema']==NAME+'.prepare' and p['contract']==contract(), 'explicit MAIN authorization and own contract')
    hash_value(prepare_sha256)
    require(p['source_report_sha256']==SOURCE_SHA, 'own prepare source identity')
    validate_plans(p['plans'],p['planner_report_sha256'],data['exports'])
    require(execution_binding['prepare_sha256']==prepare_sha256 and execution_binding['planner_report_sha256']==p['planner_report_sha256'] and
            execution_binding['source_report_sha256']==SOURCE_SHA, 'execution binding identities')
    rows=[dict(index=i,root_slot=plan['root_slot'],condition=plan['condition'],arm=plan['arm'],selection_digest=plan['selection_digest'],
          task_id=plan['task_id'],status='not_started',checkpoints=[],native_success=None,missing_checkpoints=[0,5,10],
          warmup_returned=0,own_attempted=0,own_returned=0,PNG=0,initial_exact=False,
          trajectory=[],last_observation=None,first_error=None,secondary_errors=[]) for i,plan in enumerate(p['plans'])]
    meter=Meter(store,started); errors=[]; source_ok=False; seen_envs=[]
    def fail(stage,e,row=None):
        item=error(stage,e); errors.append(item)
        if row is not None:
            if row['first_error'] is None: row['first_error']=deepcopy(item)
            else: row['secondary_errors'].append(deepcopy(item))
            row['status']='engineering_failed'
    try:
        store.put('execution_binding.json',execution_binding)
        store.emit('run_started',prepare_sha256=prepare_sha256,plans=p['plans'],contract=contract())
        for row,plan in zip(rows,p['plans'],strict=True):
            meter.where={k:row[k] for k in ('root_slot','condition','arm','selection_digest')}; meter.where['branch']=row['index']
            env=None; row['status']='running'
            try:
                env=meter.call('constructor',factory)
                require(all(env is not old and env.unwrapped is not old.unwrapped for old in seen_envs), 'each plan requires a fresh independent environment')
                seen_envs.append(env); validate_env(env.unwrapped)
                require(env.unwrapped.render_action is False and env.unwrapped.render_size==224 and env.unwrapped.render_mode=='rgb_array', 'native renderer/action semantics')
                with native_hooks(env,meter,row['secondary_errors']): episode(env,data['roots'][plan['root_slot']],data['source_plan'],plan,row,meter,store)
                require(not row['secondary_errors'], 'native hook restoration failed')
            except NativeStop as e: row.update(status='native_stopped',stop_reason=str(e))
            except BaseException as e: fail('branch',e,row)
            finally:
                if env is not None:
                    try: meter.call('close',env.close,cleanup=True)
                    except BaseException as e: fail('close',e,row)
                if row['secondary_errors'] and row['first_error'] is None:
                    fail('cleanup',ValueError('native hook restoration failed; see secondary_errors'),row)
            store.emit('branch_closed',**meter.where,status=row['status'],own_returned=row['own_returned'],first_error=row['first_error'])
    except BaseException as e: fail('execution',e)
    finally:
        try:
            for path,h in p['bindings'].items(): require(sha(path)==h, 'final input/source seal drift')
            source_ok=True
        except BaseException as e: fail('final_inputs',e)
        errors.extend(meter.secondary)
        if time.monotonic()-meter.started>=900:
            fail('final_wall',ValueError('900s wall cap exhausted; cleanup/finalization retained'))
        if not errors:
            try: census(rows,meter.counts)
            except BaseException as e: fail('census',e)
        complete=not errors and source_ok and all(r['status'] in ('horizon_completed','native_success','native_stopped') for r in rows)
        status='moving_overlay_recompile_physical_recorded' if complete else 'partial_or_failed'
        try: store.emit('run_finished',status=status,counts=meter.counts,errors=errors)
        except BaseException as e: fail('final_journal',e); status='partial_or_failed'
        journal_sha=None
        try: journal_sha=sha(store.path/'journal.jsonl')
        except BaseException as e: fail('journal_hash',e); status='partial_or_failed'
        report=dict(schema=NAME+'.report',status=status,rows=rows,errors=errors,
            prepare_sha256=prepare_sha256,planner_report_sha256=p['planner_report_sha256'],source_report_sha256=SOURCE_SHA,
            counts=deepcopy(meter.counts),journal_chain_tail=store.previous,journal_sha256=journal_sha,
            execution_binding=deepcopy(execution_binding),execution_binding_sha256=None,
            fixed_denominator=8,inputs_final_verified=source_ok,P1_admission=False,new_neural_calls=0,
            missing_labels_are_null=True,wall_seconds=time.monotonic()-meter.started)
        try: report['execution_binding_sha256']=sha(store.path/'execution_binding.json')
        except BaseException as e: fail('binding_hash',e); report['status']='partial_or_failed'
        report['wall_seconds']=time.monotonic()-meter.started
        if report['wall_seconds']>=900 and not any(e['stage']=='final_wall' for e in errors):
            fail('final_wall',ValueError('900s wall cap exhausted during finalization'))
            report['status']='partial_or_failed'
        try: store.put('report.json',report,final=True)
        except BaseException as e: fail('final_report',e); report['status']='partial_or_failed'
    return report


def run(prepared,expected,output,factory,*,authorization):
    require(authorization==AUTH, 'MAIN physical authorization required before prepare/read/output')
    started=time.monotonic()
    p,data=verify(prepared,expected)
    binding=dict(prepared=str(Path(prepared).resolve()),prepare_sha256=expected,
                 planner_report_sha256=p['planner_report_sha256'],source_report_sha256=SOURCE_SHA)
    return execute(p,data,Store(output),factory,authorization=authorization,prepare_sha256=expected,execution_binding=binding,started=started)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','verify','run'))
    for key in ('prepared','prepare-sha','planner-output','planner-report-sha','output'): parser.add_argument('--'+key)
    parser.add_argument('--authorize-physics',action='store_true'); a=parser.parse_args(argv)
    if a.command=='prepare': result=prepare(a.output,a.planner_output,a.planner_report_sha)
    elif a.command=='verify': p,_=verify(a.prepared,a.prepare_sha); result=dict(status='verified_no_execution',contract=p['contract'])
    else:
        require(a.authorize_physics,'MAIN explicit --authorize-physics')
        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401
            return gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224,with_target=False)
        result=run(a.prepared,a.prepare_sha,a.output,factory,authorization=AUTH)
    print(source.canonical({k:v for k,v in result.items() if k!='rows'}))
    return 1 if result.get('status')=='partial_or_failed' else 0


if __name__=='__main__': raise SystemExit(main())
