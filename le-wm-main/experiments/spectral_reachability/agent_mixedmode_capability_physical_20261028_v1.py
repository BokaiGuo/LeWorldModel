"""MAIN-only true40 pre/post physical probes; frozen probe-v2 ABI, no neural.

No old post-only run is called or relabelled. Uses public placement, render,
goal binding, observability guard and native-hook utilities without patching
their module globals. All causal replay is in this module.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import io
import os
from pathlib import Path
import shutil
import stat
import time
import numpy as np
from experiments.spectral_reachability import agent_mixedmode_capability_probe_20261028_v2 as probe
from experiments.spectral_reachability import agent_allmode_physical_20261028_v1 as adapter
from experiments.spectral_reachability import pusht_rgb_observability_guard as rgb_guard

NAME='agent_mixedmode_capability_physical_20261028_v1'
AUTH='MAIN_MIXEDMODE40_BEFORE_AFTER_PROBES_ONLY'
ROOT=probe.ROOT
CAP,FINAL,FLOOR=512*1024**2,32*1024**2,200*1024**2
JOURNAL_CAP,PREP_CAP,REPORT_CAP=64*1024**2,64*1024**2,8*1024**2
WALL=1800
require,sha,canonical=probe.require,probe.sha,probe.canonical
moving,source=adapter.moving,adapter.source
place,bind_goal,invariant=adapter.place,adapter.bind_goal,adapter.invariant
native_hooks,validate_env=adapter.native_hooks,adapter.validate_env
CAPS=dict(constructor=40,close=40,reset=80,public_step=700,physics_step=7160,
          set_state=160,goal_setter=120,goal_render=80,goal_bind=40,explicit_render=2340)


def contract():
    return dict(schema=NAME+'.contract',probe_schema=probe.NAME+'.prepare',
        authority=AUTH,fixed_denominator=40,order='four source tasks, before/after, zero,+x,-x,+y,-y',
        caps=deepcopy(CAPS),wall_seconds=WALL,output_cap_bytes=CAP,final_reserve_bytes=FINAL,
        journal_uncompressed_cap_bytes=JOURNAL_CAP,prepare_cap_bytes=PREP_CAP,report_cap_bytes=REPORT_CAP,
        disk_floor_bytes=FLOOR,rendering='every warmup0..10, root-goalbind0, prefix1..5, probe1..5: off/on/off',
        frames=2340,probe_initial='reuse immediately preceding fresh root/prefix frame, no extra render',
        guard='original RgbObservabilityGuard; fresh per primitive; separate warmup/task sequence at goal binding',
        raw='literal float32 original warmup/prefix/probe; no clip/tanh/renormalization',
        no_neural=True,training=False,Q_fit=False,retries=0,replace_failed=False,
        time_confound=True,known_roots=2,fresh_roots=0,P1_admission=False,
        scientific_admission=False,independent_confirmation=False,equal_total_compute=False,
        constructor_partial_internal_work='unknown on constructor failure; constructors verified inert before hooks',
        cleanup='close only bypasses exhausted wall/fault; never physics/reset/render',
        reuse='none; all40 independently reconstructed; old42 equality does not skip a branch')


def error(phase,exc):
    return dict(phase=phase,type=type(exc).__name__,message=str(exc))


def own_files():
    return [Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md'),
            Path(probe.__file__),Path(rgb_guard.__file__),*adapter.own_files()]


def material(probe_prepared,probe_prepare_sha):
    # Full sealed input checks, metadata/PNG/arrays only, no env constructor.
    p=probe.verify(probe_prepared,probe_prepare_sha)
    data=adapter.source_material()
    require(p['schema']==probe.NAME+'.prepare' and p['contract']==probe.contract() and
        len(p['jobs'])==40 and p['source_slots']==[r['slot'] for r in data['roots']] and
        p['warm_traces']==[r['warm_trace'] for r in data['roots']] and
        [t['native'] for t in p['tasks']]==[adapter.root_metadata(data)[s] for s,c in probe.ORDER],
        'original full source recipe/context/native metadata, never zero loader')
    bindings=deepcopy(p['bindings'])
    adapter.merge(bindings,data['bindings'])
    native,native_sha=adapter.constructor_contract()
    adapter.merge(bindings,{native:native_sha,str(Path(probe_prepared).resolve()):probe_prepare_sha})
    for path in own_files():adapter.merge(bindings,{str(path.resolve()):sha(path)})
    package=dict(schema=NAME+'.prepare',contract=contract(),probe_prepared=str(Path(probe_prepared).resolve()),
        probe_prepare_sha256=probe_prepare_sha,probe_digest=probe.digest(p),bindings=bindings)
    require(len(canonical(package))<=PREP_CAP,'complete own64MiB prepare')
    probe.verify_bindings(package)
    return package,p,data


def prepare(output,probe_prepared,probe_prepare_sha):
    package,_,_=material(probe_prepared,probe_prepare_sha)
    path=Path(output).absolute();raw=canonical(package)
    require(path.name.startswith(NAME+'_') and path.suffix=='.json' and
            shutil.disk_usage(path.parent).free>=FLOOR+len(raw),'own prepare prefix and diskfloor')
    with path.open('xb') as f:
        require(f.write(raw)==len(raw),'complete prepare write');f.flush();os.fsync(f.fileno())
    adapter.paired.fsync_dir(path.parent)
    return dict(prepared=str(path),prepare_sha256=sha(path),bytes=len(raw))


def verify(prepared,expected):
    require(sha(prepared)==probe.hash_value(expected),'external own prepare SHA')
    p=probe.read(prepared)
    now,pp,data=material(p['probe_prepared'],p['probe_prepare_sha256'])
    require(p==now and sha(prepared)==expected,'complete own64MiB closure/limits')
    return p,pp,data


class Store:
    """Exclusive durable output, O(1) byte account; full terminal inode census."""
    def __init__(self,path):
        self.path=Path(path).absolute()
        require(self.path.name.startswith(NAME+'_') and not self.path.exists() and
            not self.path.is_symlink() and shutil.disk_usage(self.path.parent).free>=FLOOR+CAP,
            'new512MiB output plus200MiB floor')
        self.path.mkdir();adapter.paired.fsync_dir(self.path.parent)
        self.used=0;self.files={};self.previous=None;self.index=0;self.terminal=False;self.drift=False

    def check(self,size=4096,final=False):
        require(self.used+size<=CAP-(0 if final or self.terminal else FINAL) and
            shutil.disk_usage(self.path).free>=FLOOR+size,'output cap/final reserve/floor')

    def write(self,name,raw,append=False,final=False):
        require(type(raw) is bytes and Path(name).name==name and name not in ('','.','..'), 'flat bytes only')
        path=self.path/name
        previous=self.files.get(name)
        require(not append or previous is not None or name=='journal.jsonl','append only own files')
        require(append or previous is None,'exclusive artifact no overwrite')
        self.check(len(raw),final)
        if name=='journal.jsonl':
            require((previous[2] if previous else 0)+len(raw)<=JOURNAL_CAP,'journal64MiB uncompressed')
        flags=os.O_WRONLY|os.O_NOFOLLOW|(os.O_APPEND if append and previous else os.O_CREAT|os.O_EXCL)
        fd=os.open(path,flags,0o600)
        try:
            st=os.fstat(fd)
            if not (stat.S_ISREG(st.st_mode) and (previous is None or
                (st.st_dev,st.st_ino,st.st_size)==previous)):
                self.drift=True
                raise ValueError('owned inode/size before append')
            with os.fdopen(fd,'ab' if append else 'wb',closefd=False) as f:
                require(f.write(raw)==len(raw),'short write');f.flush();os.fsync(fd)
        finally:
            # Actual partial writes remain charged, including raised flush/fsync.
            st=os.fstat(fd);os.close(fd)
            old=previous[2] if previous else 0
            self.used+=st.st_size-old;self.files[name]=(st.st_dev,st.st_ino,st.st_size)
        if previous is None:adapter.paired.fsync_dir(self.path)

    def emit(self,kind,**fields):
        e=dict(index=self.index,previous_sha256=self.previous,kind=kind,**deepcopy(fields))
        e['event_sha256']=probe.digest(e)
        self.write('journal.jsonl',canonical(e)+b'\n',append=True)
        self.index+=1;self.previous=e['event_sha256']

    def put(self,name,value,final=False):self.write(name,canonical(value),final=final)

    def frame(self,name,value):
        from PIL import Image
        b=io.BytesIO();Image.fromarray(value).save(b,format='PNG');self.write(name,b.getvalue())
        return sha(self.path/name)

    def arrays(self,name,**arrays):
        b=io.BytesIO();np.savez_compressed(b,**arrays);self.write(name,b.getvalue());return sha(self.path/name)

    def census(self):
        require(not self.drift,'prior inode/size drift cannot be re-certified')
        actual={}
        for f in self.path.iterdir():
            st=f.lstat();require(stat.S_ISREG(st.st_mode),'no foreign dir/symlink')
            actual[f.name]=(st.st_dev,st.st_ino,st.st_size)
        require(actual==self.files and sum(v[2] for v in actual.values())==self.used,'full name/inode/size/byte census')
        self.check(0,final=True)
        return {name:sha(self.path/name) for name in sorted(actual)}


class Meter:
    def __init__(self,store,started=None):
        self.store=store;self.started=time.monotonic() if started is None else started
        self.where={};self.counts={k:dict(attempted=0,completed=0) for k in CAPS}
        self.active={k:0 for k in CAPS};self.index=0;self.fault=False;self.secondary=[]

    def call(self,kind,fn,*,cleanup=False,**fields):
        require(not cleanup or kind=='close','only nonphysical close may bypass exhausted budget')
        if not cleanup:
            require(not self.fault and time.monotonic()-self.started<WALL,'fault/wall before actual callback')
            self.store.check()
        c=self.counts[kind];require(c['attempted']<CAPS[kind],'resource cap before callback: '+kind)
        i=self.index;self.index+=1
        kw=dict(job_digest=self.where['job_digest'],operation_index=i,
            resource='internal_physics_step' if kind=='physics_step' else kind)
        old_terminal=self.store.terminal
        if cleanup:self.store.terminal=True
        try:
            try:self.store.emit('operation_attempted',**kw)
            except BaseException as exc:
                self.fault=True
                if not cleanup:raise
                self.secondary.append(error('close_intent',exc))
            c['attempted']+=1;self.active[kind]+=1
            try:
                result=fn();c['completed']+=1
            except BaseException as exc:
                try:self.store.emit('operation_failed',**kw,error=error(kind,exc),fields=fields)
                except BaseException as extra:self.fault=True;self.secondary.append(error('failure_journal',extra))
                raise
            finally:self.active[kind]-=1
            try:self.store.emit('operation_returned',**kw)
            except BaseException as exc:
                self.fault=True
                if not cleanup:raise
                self.secondary.append(error('close_return',exc))
            return result
        finally:self.store.terminal=old_terminal

    def receipt(self):
        return {('internal_physics_step' if k=='physics_step' else k):deepcopy(v)
                for k,v in self.counts.items() if v['attempted']}


class NativeStop(Exception):
    """Native terminal/truncation, retain denominator without substitution."""


def save_info(store,name,info):
    """Original returned arrays stay lossless, never RGB lists in the journal."""
    arrays={}
    def pack(v):
        if isinstance(v,np.ndarray):
            require(not v.dtype.hasobject,'no object/pickle info arrays')
            key='a'+str(len(arrays));arrays[key]=v.copy()
            return dict(array=key,dtype=v.dtype.str,shape=list(v.shape))
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,dict):return {k:pack(x) for k,x in v.items()}
        if isinstance(v,(tuple,list)):return [pack(x) for x in v]
        require(v is None or type(v) in (str,bool,int,float),'literal native info')
        return v
    result=pack(info)
    if arrays:
        h=store.arrays(name,**arrays)
        return dict(tree=result,file=name,sha256=h)
    return dict(tree=result,file=None,sha256=None)


def observe(env,obs,phase,step,row,meter,store,trace=None,guard=None):
    frames=[]
    class Sink:
        def __getattr__(self,name):return getattr(store,name)
        def emit(self,kind,**kw):
            store.emit(kind,**kw)
            if kind=='fresh_observation':frames.append(deepcopy(kw))
    rgb=adapter.base.capture(env,obs,phase,step,row,meter,Sink())
    require(len(frames)==3,'fresh off/on/off invariant triplet')
    if trace is not None:
        for frame in frames:
            ref=adapter.reference_for(trace,step,frame['receipt']['overlay_index'])
            if ref is not None:adapter.check_warm_frame(frame,ref)
    selected=frames[1 if row['condition']=='on' else 0]
    r=selected['receipt']
    if guard is not None:
        g,scope,n=guard;accepted=False
        try:g.observe(rgb,primitive_step=n);accepted=True
        finally:
            row.setdefault('guards',{})[scope]=g.receipt()
            store.emit('rgb_guard_observed',**meter.where,scope=scope,primitive_step=n,
                accepted=accepted,receipt=g.receipt())
    return rgb,dict(step=step,file=r['png'],sha256=r['file_sha256'],pixel_sha256=r['pixel_sha256'],
        state=r['state'],pre=selected['pre'],post=selected['post'])


def step_raw(env,a,phase,step,row,meter,store,goal,guard=None):
    require(a.dtype==np.float32 and a.shape==(2,) and env.unwrapped.with_target is False,'literal native-off raw2')
    if env.unwrapped.eval_state(goal,source.vector(env.unwrapped._get_obs()))[0]:
        raise NativeStop('native terminal before '+phase+' dispatch')
    if guard is not None:
        g,scope=guard
        try:require(g.dispatch(a).tobytes()==a.tobytes(),'guard authorizes same literal raw buffer')
        finally:row.setdefault('guards',{})[scope]=g.receipt()
    bits=a.tobytes().hex()
    if phase=='probe':
        store.emit('probe_attempted',**meter.where,step=step,raw_bits=bits);row['attempted']+=1
    obs,reward,t,tr,info=meter.call('public_step',lambda:env.step(a.copy()),phase=phase,step=step,raw_bits=bits)
    actual=np.asarray(env.unwrapped.latest_action)
    state=source.vector(obs['state'])
    # A returned callback is accounted even when a subsequent consistency check fails.
    if phase=='probe':row['completed']+=1
    if phase=='probe':
        store.emit('probe_returned',**meter.where,step=step,raw_bits=bits,state=state.tolist(),terminated=bool(t),truncated=bool(tr))
    else:
        store.emit('causal_replay_returned',**meter.where,step=step,raw_bits=bits,state=state.tolist(),terminated=bool(t),truncated=bool(tr))
    info_receipt=save_info(store,f'b{row["index"]}_{phase}_{step:02}_info.npz',info)
    store.emit('raw_return',**meter.where,phase=phase,step=step,raw_bits=bits,actual_bits=actual.tobytes().hex(),
        actual_dtype=actual.dtype.str,state=state.tolist(),snapshot=moving.snapshot(env.unwrapped),
        reward=float(reward),terminated=bool(t),truncated=bool(tr),info=info_receipt)
    require(actual.dtype==np.float32 and actual.shape==(2,) and actual.tobytes()==a.tobytes(),'actual dispatch raw buffer exact')
    adapter.paired.check_return(env,a,actual,info,goal,float(reward),bool(t))
    row['last_return']=dict(state=state.tolist(),reward=float(reward),terminated=bool(t),truncated=bool(tr))
    return obs,bool(t),bool(tr)


def episode(env,root,source_plan,task,job,row,meter,store):
    body=env.unwrapped;packet=root['packets'][row['condition']]
    require(moving.exact(packet['raw_action_history'],moving.warmup_commands()),'source native warmup raw10')
    warmguard=rgb_guard.RgbObservabilityGuard(root_id=root['root_sha256'],agent_rgb=(65,105,225))
    taskguard=rgb_guard.RgbObservabilityGuard(root_id=root['root_sha256'],agent_rgb=(65,105,225))
    obs,native_goal=place(env,source_plan,root['slot'],meter,store)
    history=[];rgb,_=observe(env,obs,'warmup',0,row,meter,store,root['warm_trace'],(warmguard,'warmup',0));history.append(rgb)
    for i,a in enumerate(moving.warmup_commands(),1):
        obs,t,tr=step_raw(env,a,'warmup',i,row,meter,store,native_goal,(warmguard,'warmup'))
        rgb,_=observe(env,obs,'warmup',i,row,meter,store,root['warm_trace'],(warmguard,'warmup',i))
        if t or tr:raise NativeStop('native stop during warmup')
        if i in (5,10):history.append(rgb)
    state=source.vector(obs['state']);goal=source.vector(root['goal'])
    require(moving.exact(np.asarray(history),packet['rgb_history']) and source.root_hash(state,goal)==root['root_sha256'],
        'original same-condition history3 and full7 root exact')
    native=task['native']
    require(state.tolist()==native['root_state'],'original root7, no object-only identity')
    bind_goal(body,goal,packet['goal_rgb'],meter,store)
    bound=invariant(body)
    require(bound['goal_pose']==native['goal_pose'] and bound['renderer']==native['renderer'] and
        moving.image_sha(packet['goal_rgb'])==native['goal_pixels'][row['condition']], 'original true goal/renderer')
    row.update(root_state=state.tolist(),goal=goal.tolist(),root_sha256=root['root_sha256'],goal_pose=bound['goal_pose'],
        goal_pixel_sha256=native['goal_pixels'][row['condition']])
    _,frame=observe(env,obs,'root',0,row,meter,store,guard=(taskguard,'task',0))
    for i,a in enumerate(probe.raw(job['prefix_commands'],len(job['prefix_commands'])),1):
        obs,t,tr=step_raw(env,a,'prefix',10+i,row,meter,store,goal,(taskguard,'task'))
        _,frame=observe(env,obs,'prefix',i,row,meter,store,guard=(taskguard,'task',i))
        if t or tr:raise NativeStop('native stop during prefix')
    require({k:v for k,v in invariant(body).items() if k!='snapshot'}==
        {k:v for k,v in bound.items() if k!='snapshot'},'task metadata unchanged before probe')
    row.update(start_verified=True,start_state=source.vector(obs['state']).tolist(),
        causal_replay_verified=True,causal_prefix_commands=deepcopy(job['prefix_commands']))
    frame['step']=0;row['frames'].append(frame)
    for i,a in enumerate(probe.raw(job['probe_commands'],5),1):
        obs,t,tr=step_raw(env,a,'probe',i,row,meter,store,goal,(taskguard,'task'))
        _,frame=observe(env,obs,'probe',i,row,meter,store,guard=(taskguard,'task',len(job['prefix_commands'])+i));row['frames'].append(frame)
        require({k:v for k,v in invariant(body).items() if k!='snapshot'}==
            {k:v for k,v in bound.items() if k!='snapshot'},'task/renderer invariant through probe')
        if t or tr:raise NativeStop('native stop during probe (even step5 retained as stopped)')
    row.update(status='completed',end_state=source.vector(obs['state']).tolist(),rgb_guard_accepted=True)


def execute(package,p,data,store,factory,*,authorization,prepared,prepare_sha,started=None):
    require(authorization==AUTH,'explicit MAIN authority before any simulator call')
    require(package['schema']==NAME+'.prepare' and package['contract']==contract() and
        package['probe_digest']==probe.digest(p),'sealed own limits and complete probe tasks')
    meter=Meter(store,started);errors=[];seen=[]
    rows=[dict(index=j['index'],job_digest=j['job_digest'],task=j['task'],role=j['role'],probe_id=j['probe_id'],
        condition=p['tasks'][j['task']]['plan']['condition'],status='unstarted',attempted=0,completed=0,
        error=None,secondary_errors=[],end_state=None,last_return=None,start_state=None,start_verified=False,
        context_digest=j['context_digest'],causal_prefix_commands=deepcopy(j['prefix_commands']),
        causal_replay_verified=False,rgb_guard_accepted=False,frames=[],PNG=0,trajectory=[],
        checkpoints=[],missing_checkpoints=[],last_observation=None) for j in p['jobs']]
    def fail(phase,exc,row=None):
        e=error(phase,exc);errors.append(e)
        if row is not None:
            if row['error'] is None:row['error']=deepcopy(e)
            else:row['secondary_errors'].append(deepcopy(e))
            row.update(status='failed',end_state=None)
    binding=dict(prepared=str(Path(prepared).resolve()),prepare_sha256=prepare_sha,
        probe_prepared=package['probe_prepared'],probe_prepare_sha256=package['probe_prepare_sha256'])
    try:
        store.put('execution_binding.json',binding);store.emit('run_started',contract=contract(),execution_binding=binding)
        for j,row in zip(p['jobs'],rows,strict=True):
            if meter.fault or time.monotonic()-meter.started>=WALL:break
            meter.where=dict(job_digest=j['job_digest']);env=None
            try:
                env=meter.call('constructor',factory)
                require(all(env is not old and env.unwrapped is not old.unwrapped for old in seen),'independent fresh environment per probe')
                seen.append(env);validate_env(env.unwrapped)
                require(env.unwrapped.render_action is False and env.unwrapped.render_size==224 and
                    env.unwrapped.render_mode=='rgb_array','native official224 renderer')
                with native_hooks(env,meter,row['secondary_errors']):
                    episode(env,data['roots'][p['tasks'][j['task']]['plan']['root_slot']],
                        data['source_plan'],p['tasks'][j['task']],j,row,meter,store)
                if row['secondary_errors']:raise ValueError('hook restoration errors preserved')
            except NativeStop as exc:row.update(status='native_stopped',stop_reason=str(exc),end_state=None)
            except BaseException as exc:fail('branch',exc,row)
            finally:
                if env is not None:
                    try:meter.call('close',env.close,cleanup=True)
                    except BaseException as exc:fail('close',exc,row)
                if row['secondary_errors'] and row['error'] is None:fail('hook_cleanup',ValueError('see secondary_errors'),row)
                # Immediate durable per-branch partial arrays/frames, no replacement.
                try:store.put(f'branch_{row["index"]:02}.json',row)
                except BaseException as exc:meter.fault=True;fail('branch_receipt',exc,row)
    except BaseException as exc:fail('execution',exc)
    inputs=False
    try:probe.verify_bindings(package);inputs=True
    except BaseException as exc:fail('final_source_hash',exc)
    errors.extend(meter.secondary)
    if time.monotonic()-meter.started>=WALL:fail('wall',ValueError('1800s exhausted; no further dispatch'))
    store.terminal=True
    complete=inputs and not errors and all(row['status']=='completed' for row in rows)
    counts=meter.receipt()
    status='complete' if complete else 'partial'
    try:store.emit('run_finished',status=status,counts=counts,errors=errors)
    except BaseException as exc:fail('terminal_journal',exc);status='partial'
    report=dict(schema=probe.NAME+'.evidence',executor_schema=NAME+'.report',status=status,
        prepare_sha256=package['probe_prepare_sha256'],executor_prepare_sha256=prepare_sha,
        executor_source_sha256=package['bindings'][str(Path(__file__).resolve())],
        execution_binding=binding,contract=contract(),rows=rows,errors=errors,counts=counts,
        inputs_final_verified=inputs,new_neural_calls=0,P1_admission=False,independent_confirmation=False,
        time_confound=True,wall_seconds=time.monotonic()-meter.started,
        journal=dict(file='journal.jsonl',sha256=None),journal_tail=store.previous,
        artifact_sha256={},artifact_bytes_before_report=store.used)
    try:report['artifact_sha256']=store.census();report['journal']['sha256']=report['artifact_sha256']['journal.jsonl']
    except BaseException as exc:fail('terminal_census',exc);report['status']='partial'
    try:
        require(len(canonical(report))<=REPORT_CAP,'report8MiB final reserve')
        store.put('report.json',report,final=True);store.census()
    except BaseException as exc:
        # Never replace the primary exception. Best-effort independent terminal
        # failure receipt uses reserved bytes, never rewrites a published report.
        fail('final_report',exc);report['status']='partial'
        try:store.put('terminal_failure.json',dict(errors=errors,counts=counts,rows=rows),final=True)
        except BaseException as extra:errors.append(error('terminal_failure_write',extra))
    return report


def run(prepared,prepare_sha,output,factory,*,authorization):
    require(authorization==AUTH,'MAIN authority before input reads/output/factory')
    started=time.monotonic();package,p,data=verify(prepared,prepare_sha)
    require(time.monotonic()-started<WALL,'saved verification included in wall')
    return execute(package,p,data,Store(output),factory,authorization=authorization,
        prepared=prepared,prepare_sha=prepare_sha,started=started)


def consume(output,report_sha):
    directory=Path(output).resolve();require(sha(directory/'report.json')==probe.hash_value(report_sha),'external actual physical report SHA')
    r=probe.read(directory/'report.json')
    package,p,_=verify(r['execution_binding']['prepared'],r['executor_prepare_sha256'])
    require(r['executor_schema']==NAME+'.report' and r['contract']==contract() and r['inputs_final_verified'] is True,
        'own source/limits/final bindings')
    require({f.name for f in directory.iterdir()}==set(r['artifact_sha256'])|{'report.json'} and
        sum(f.stat().st_size for f in directory.iterdir())<=CAP,'complete own artifact census')
    for name,h in r['artifact_sha256'].items():
        require(Path(name).name==name and not (directory/name).is_symlink() and sha(directory/name)==h,'all artifacts exact')
    require(probe.read(directory/'execution_binding.json')==r['execution_binding'] and
        r['execution_binding']['probe_prepare_sha256']==package['probe_prepare_sha256'] and
        r['prepare_sha256']==package['probe_prepare_sha256'] and
        r['executor_source_sha256']==package['bindings'][str(Path(__file__).resolve())], 'exact two-prepare/source execution binding')
    events=read_events(directory/'journal.jsonl',r['journal_tail'])
    validate_trace(p,r,events)
    # The frozen probe reader expects all axes to have been entered. Early
    # pre-constructor failures legitimately have an empty resource census;
    # retain these as partial receipts, not fabricated zero-valued successes.
    if r['status']=='partial':
        return dict(schema=NAME+'.consume',status='partial_evidence_preserved_not_admitted',
            executor_report_sha256=report_sha,rows=deepcopy(r['rows']),errors=deepcopy(r['errors']),
            counts=deepcopy(r['counts']),contrasts=None,P1_admission=False,new_neural_calls=0,new_physics_calls=0)
    result=probe.consume(package['probe_prepared'],package['probe_prepare_sha256'],str(directory/'report.json'),report_sha,
        str(Path(__file__).resolve()),package['bindings'][str(Path(__file__).resolve())])
    require(r['wall_seconds']<WALL and all(v['attempted']<=CAPS['physics_step' if k=='internal_physics_step' else k]
        for k,v in r['counts'].items()),'actual wall/caps, not only declarations')
    result.update(executor_report_sha256=report_sha,executor_prepare_sha256=r['executor_prepare_sha256'])
    return result


def read_events(path,tail):
    require(Path(path).stat().st_size<=JOURNAL_CAP,'own journal64MiB')
    raw=Path(path).read_bytes();require(raw.endswith(b'\n'),'complete journal lines')
    import json
    events=[json.loads(line) for line in raw.splitlines()];previous=None
    for i,e in enumerate(events):
        require(e['index']==i and e['previous_sha256']==previous and
            e['event_sha256']==probe.digest({k:v for k,v in e.items() if k!='event_sha256'}),'own complete event hashchain')
        previous=e['event_sha256']
    require(previous==tail and events[-1]['kind']=='run_finished','terminal chain')
    return events


def validate_trace(p,r,events):
    """Own ABI validation also works for all-unstarted/failed denominators."""
    require(len(r['rows'])==len(p['jobs'])==40 and r['status'] in ('complete','partial'),'complete fixed denominator')
    counts={};pending={}
    for e in events:
        if e['kind'] not in ('operation_attempted','operation_returned'):continue
        k=(e['job_digest'],e['operation_index']);resource=e['resource']
        require(e['job_digest'] in {j['job_digest'] for j in p['jobs']},'no extra or replacement job')
        c=counts.setdefault(resource,dict(attempted=0,completed=0))
        if e['kind']=='operation_attempted':
            require(k not in pending,'no operation retry');pending[k]=[resource,False];c['attempted']+=1
        else:
            require(k in pending and pending[k]==[resource,False],'unique return after actual attempted operation')
            pending[k][1]=True;c['completed']+=1
    require(counts==r['counts'] and all(v['attempted']<=CAPS['physics_step' if k=='internal_physics_step' else k]
        for k,v in counts.items()),'full actual resource census and caps')
    for j,row in zip(p['jobs'],r['rows'],strict=True):
        require(row['job_digest']==j['job_digest'] and row['context_digest']==j['context_digest'] and
            row['status'] in ('completed','failed','native_stopped','unstarted'),'original jobs/context and failure denominator')
        scoped=[e for e in events if e.get('job_digest')==j['job_digest']]
        attempts=[e for e in scoped if e['kind']=='probe_attempted']
        returns=[e for e in scoped if e['kind']=='probe_returned']
        require(row['attempted']==len(attempts) and row['completed']==len(returns) and
            len(returns)<=len(attempts)<=5,'partial attempt/return lower bounds')
        for i,e in enumerate(attempts):
            require(e['step']==i+1 and e['raw_bits']==probe.raw(j['probe_commands'],5)[i].tobytes().hex(),'attempted raw5 exact')
        for i,e in enumerate(returns):
            require(e['step']==i+1 and e['raw_bits']==attempts[i]['raw_bits'],'returned raw5 exact')
        raw_returns=[e for e in scoped if e['kind']=='raw_return'];stopped=False
        for e in raw_returns:
            require(not stopped and e['raw_bits']==e['actual_bits'] and e['actual_dtype']==np.dtype(np.float32).str,
                'actual buffer/no dispatch after native stop')
            stopped=e['terminated'] or e['truncated']
        if row['status']=='completed':
            require(len(returns)==5 and row['end_state']==returns[-1]['state'] and row['error'] is None and
                row['rgb_guard_accepted'] is row['causal_replay_verified'] is row['start_verified'] is True,'full actual successful response')
            for scope,n in [('warmup',10),('task',len(j['prefix_commands'])+5)]:
                g=row['guards'][scope]
                require(g['primitive_commands_authorized']==n and g['abstained'] is False and
                    [o['primitive_step'] for o in g['observations']]==list(range(n+1)) and
                    all(o['canonical_agent_pixels']>0 for o in g['observations']), 'existing fresh RGB guard full command sequence')
        else:require(row['end_state'] is None,'unknown failed/stopped endpoint, not zero fill')
    if r['status']=='complete':
        require(r['errors']==[] and all(x['status']=='completed' for x in r['rows']),'no false complete')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    b=sub.add_parser('prepare')
    for k in ('output','probe-prepared','probe-prepare-sha'):b.add_argument('--'+k,required=True)
    v=sub.add_parser('verify')
    for k in ('prepared','prepare-sha'):v.add_argument('--'+k,required=True)
    r=sub.add_parser('run')
    for k in ('prepared','prepare-sha','output'):r.add_argument('--'+k,required=True)
    r.add_argument('--authorize-physics',action='store_true')
    c=sub.add_parser('consume')
    for k in ('output','report-sha'):c.add_argument('--'+k,required=True)
    a=vars(parser.parse_args(argv));cmd=a.pop('command')
    if cmd=='prepare':result=prepare(**a)
    elif cmd=='verify':
        p,_,_=verify(a['prepared'],a['prepare_sha']);result=dict(status='verified_no_execution',contract=p['contract'])
    elif cmd=='consume':result=consume(**a)
    else:
        require(a.pop('authorize_physics'),'MAIN explicit --authorize-physics required')
        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401, only behind explicit run authority
            return gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224,with_target=False)
        result=run(**a,factory=factory,authorization=AUTH)
    print(canonical(result).decode());return 1 if result.get('status')=='partial' else 0


if __name__=='__main__':raise SystemExit(main())
