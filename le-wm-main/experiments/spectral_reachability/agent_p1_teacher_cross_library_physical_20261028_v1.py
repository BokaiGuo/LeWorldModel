"""MAIN-only forty fresh teacher common-library physical branches, zero neural.

External accepted compiler receipt fixes ALL raw10 plans. Old probe trajectories
authenticate only warmup/prefix identity; they are never new suffix outcomes.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import time
import numpy as np
from experiments.spectral_reachability import agent_p1_teacher_cross_library_20261028_v1 as teacher
from experiments.spectral_reachability import agent_mixedmode_zero_time_control_20261028_v1 as zero
from experiments.spectral_reachability import p1_witness_physical_20261027_v1 as pose

NAME='agent_p1_teacher_cross_library_physical_20261028_v1'
AUTH='MAIN_TEACHER_COMMON_LIBRARY40_PHYSICS_ONLY'
ROOT=teacher.ROOT
probe,helper,adapter=zero.probe,zero.helper,zero.adapter
moving,source,rgb_guard=zero.moving,zero.source,zero.rgb_guard
place,bind_goal,invariant=zero.place,zero.bind_goal,zero.invariant
native_hooks,validate_env=zero.native_hooks,zero.validate_env
require,sha,canonical,digest=teacher.require,teacher.sha,teacher.canonical,teacher.digest
COMPILER_SOURCE_SHA='ca80a146b602406a1ef960fd2fa064afabc9a2bfd02d5396983f328d7f443b4a'
COMPILER_REPORT_SHA='d5ab99d55bacd73c8d9cf0fb774819bb46af63100053198858e639d8fa746738'
COMPILER_CONSUME_SHA='d82109eb8de582e59a4f0485307c01c9b91f9f84551c218c61cca4da24335213'
CAP,FINAL,FLOOR=1024**3,64*1024**2,200*1024**2
PREP_CAP,JOURNAL_CAP,REPORT_CAP,WALL=64*1024**2,128*1024**2,16*1024**2,3600
CAPS=dict(constructor=40,close=40,reset=80,public_step=800,physics_step=8160,
          set_state=160,goal_setter=120,goal_render=80,goal_bind=40,explicit_render=2640)
FRAME_FIELDS=('phase','step','condition','overlay_index','file_sha256','pixel_sha256','state','snapshot')


def contract():
    return dict(schema=NAME+'.contract',authority=AUTH,fixed_denominator=40,
        order='task(0off,0on,1off,1on),role(Z,S),library(zero,uZ,-uZ,uS,-uS)',
        task_order=[list(k) for k in probe.ORDER],roles=list(teacher.ROLES),library=list(teacher.LIBRARY),
        recipe='fresh original random seed + placement request; warm10 -> literal release prefix5 -> literal release suffix5',
        caps=deepcopy(CAPS),public_breakdown=dict(warmup=400,prefix=200,suffix=200),
        reset_internal_physics=160,public_internal_physics=8000,
        rendering='off/on/off each warmup0..10, root0, prefix1..5, suffix1..5; 66 PNG per full branch',
        raw='complete release float32 raw10, no tanh/clip/rescale/second /100; negative zero preserved',
        identity='all51 actual warm/root/prefix frames + all15 actual raw/state/snapshots exact before new suffix',
        guard='original RGB224 guard, warmup0..10 and task0..10; native earlystop obeyed',
        prior_evidence='prefix acquisition only, NEVER suffix effects; no replay reuse, padding, retry or replacement',
        new_neural_calls=0,training=False,Qfit=False,selection_uses_outcome=False,
        actual_latent_Q=None,actual_RGB_encoding='not performed; future separate MAIN CUDA job',
        checkpoints=[0,5,10],absolute_steps=[10,15,20],state_dimension=7,
        metrics='official native flags/reward plus independent wrapped/all jointXY/objectXY diagnostics; no override',
        time_matched=True,state_velocity_contact_mediation_isolated=False,
        known_roots=2,fresh_roots=0,P1_admission=False,C2_non_greedy_premise=False,
        independent_confirmation=False,equal_FLOPs=False,equal_total_compute=False,
        wall_seconds=WALL,output_cap_bytes=CAP,final_reserve_bytes=FINAL,
        prepare_cap_bytes=PREP_CAP,journal_uncompressed_cap_bytes=JOURNAL_CAP,
        report_cap_bytes=REPORT_CAP,disk_floor_bytes=FLOOR,
        constructor_failure_internal_work='unknown before native hooks; constructor contract must be inert',
        cleanup='close only may bypass wall/fault, never reset/render/physics')


def error(phase,exc):return dict(phase=phase,type=type(exc).__name__,message=str(exc))


def jobs_for(compiled,release,p):
    """Fixed forty literal actions, no model_Q/physical score enters this interface."""
    require(compiled['schema']==teacher.NAME+'.prepare' and compiled['contract']==teacher.contract() and
        len(compiled['anchors'])==8 and release['schema']==teacher.NAME+'.consume' and
        release['status']=='teacher_cross_library_commands_verified_not_admitted' and len(release['plans'])==40 and
        release['P1_admission'] is release['C2_non_greedy_premise'] is False and
        release['new_neural_calls']==release['new_physics_calls']==0,'complete real teacher release ABI')
    require(compiled['ancestry']['probe_digest']==probe.digest(p) and len(p['tasks'])==4,'original task ancestry')
    jobs=[]
    for ai,a in enumerate(compiled['anchors']):
        ti,ri=divmod(ai,2);slot,condition=probe.ORDER[ti];role=teacher.ROLES[ri];t=p['tasks'][ti]
        require(a['index']==ai and a['task']==ti and a['role']==role and (a['root_slot'],a['condition'])==(slot,condition) and
            a['anchor_id']==digest({k:v for k,v in a.items() if k!='anchor_id'}) and
            a['task_id']==t['slot']['task_id'] and a['blind_sha256']==t['slot']['blind_sha256'] and
            a['root_context']==t['slot']['packet']['root_context'] and a['goal']==t['slot']['packet']['goal'],
            'same four original tasks/conditions, all eight teacher anchor seals')
        prefix=np.zeros((5,2),'<f4') if role=='Z' else probe.raw(t['plan']['prefix_commands'],5)
        require(probe.raw(a['prefix_commands'],5).tobytes()==prefix.tobytes(),'actual zero or original selected prefix')
        for li,label in enumerate(teacher.LIBRARY):
            plan=release['plans'][ai*5+li]
            require(plan['index']==ai*5+li and plan['schema']==teacher.NAME+'.model_plan' and
                all(plan[k]==a[k] for k in ('task','root_slot','condition','role','task_id','blind_sha256','anchor_id')) and
                plan['library_index']==li and plan['library_label']==label and
                plan['plan_digest']==digest({k:v for k,v in plan.items() if k!='plan_digest'}) and
                plan['selection_uses_probe_response'] is plan['selection_uses_physics'] is plan['P1_admission'] is False,
                'literal complete fixed library release, never choose by score')
            raw=probe.raw(plan['commands'],10)
            require(np.max(abs(raw))<=1 and raw[:5].tobytes()==prefix.tobytes() and
                raw.tobytes()==probe.raw(plan['prefix_commands']+plan['suffix_commands'],10).tobytes(),'full raw10 buffer authority')
            j=dict(index=len(jobs),task=ti,root_slot=slot,condition=condition,role=role,anchor_index=ai,anchor_id=a['anchor_id'],
                library_index=li,library_label=label,task_id=a['task_id'],blind_sha256=a['blind_sha256'],
                plan_digest=plan['plan_digest'],teacher_context_sha256=plan['teacher_context_sha256'],
                parent_selection_digest=t['plan']['selection_digest'],commands=raw.tolist(),
                prefix_commands=raw[:5].tolist(),suffix_commands=raw[5:].tolist())
            j['job_digest']=digest(j);jobs.append(j)
    for ti in range(4):
        z,s=jobs[ti*10:ti*10+5],jobs[ti*10+5:ti*10+10]
        bank=teacher.common_bank(z[1]['suffix_commands'],s[3]['suffix_commands'])
        for rows in (z,s):
            require(all(probe.raw(j['suffix_commands'],5).tobytes()==b.tobytes() for j,b in zip(rows,bank,strict=True)),
                    'same symmetric five raw buffers across both roles, including sign bits')
    return jobs


def reference_material(compiled,p,bindings):
    reports={};events={};bases={}
    for role,key in (('Z','zero_report'),('S','selected_report')):
        name,h=teacher.INPUTS[key];path=ROOT/name;r=teacher.read(path,h)
        journal=teacher.flat(path.parent,r['journal']['file'])
        require(sha(journal)==r['journal']['sha256'],'original acquired physical journal SHA')
        reports[role]=r;events[role]=helper.read_events(journal,r['journal_tail']);bases[role]=path.parent
        adapter.merge(bindings,{str(path):h,str(journal):r['journal']['sha256']})
    out=[]
    for a in compiled['anchors']:
        role=a['role'];ti=a['task'];row=reports[role]['rows'][ti*5 if role=='Z' else ti*10+5]
        require(row['status']=='completed' and row['probe_id']=='zero' and row['task']==ti and row['condition']==a['condition'],
                'fixed prior zero-probe representative, not selected by outcome')
        scoped=[e for e in events[role] if e.get('job_digest')==row['job_digest']]
        frames=[deepcopy(e) for e in scoped if e['kind']=='fresh_observation' and e['receipt']['phase']!='probe']
        raw=[deepcopy(e) for e in scoped if e['kind']=='raw_return'][:15]
        expected=np.concatenate((moving.warmup_commands(),probe.raw(a['prefix_commands'],5)))
        require(len(frames)==51 and len(raw)==15,'original complete warm/root/prefix triplets and raw15')
        raw_refs=[]
        for i,(e,command) in enumerate(zip(raw,expected,strict=True)):
            require(e['phase']==('warmup' if i<10 else 'zero' if role=='Z' else 'prefix') and e['step']==i+1 and
                e['raw_bits']==e['actual_bits']==command.tobytes().hex() and e['actual_dtype']==np.dtype('<f4').str and
                not e['terminated'] and not e['truncated'] and e['snapshot']['state']==e['state'],'original real raw15 and full state7')
            raw_refs.append(dict(phase='warmup' if i<10 else 'prefix',step=i+1,raw_bits=e['raw_bits'],actual_bits=e['actual_bits'],
                actual_dtype=e['actual_dtype'],state=e['state'],snapshot=e['snapshot']))
        refs=[]
        for e in frames:
            f=e['receipt'];path=teacher.flat(bases[role],f['png']).resolve()
            require(sha(path)==f['file_sha256']==reports[role]['artifact_sha256'][f['png']] and e['pre']==e['post'],
                    'original full PNG bytes and render invariants')
            adapter.merge(bindings,{str(path):f['file_sha256']})
            f['phase']='prefix' if f['phase']=='zero' else f['phase']
            refs.append(dict(receipt={k:deepcopy(f[k]) for k in FRAME_FIELDS},pre=e['pre'],post=e['post'],source_png=str(path)))
        expected_frames=[(phase,step,k) for phase,steps in [('warmup',range(11)),('root',[0]),('prefix',range(1,6))]
            for step in steps for k in range(3)]
        require([(f['receipt']['phase'],f['receipt']['step'],f['receipt']['overlay_index']) for f in refs]==expected_frames,'complete51 ordered reference frames')
        last=refs[-2 if a['condition']=='on' else -3]
        require(last['source_png']==a['png'] and last['receipt']['file_sha256']==a['png_sha256'] and
            last['receipt']['pixel_sha256']==a['pixel_sha256'] and last['receipt']['state']==raw[-1]['state'] and
            last['receipt']['snapshot']==raw[-1]['snapshot'],'true same-condition teacher15 acquisition, not terminal20')
        out.append(dict(anchor_id=a['anchor_id'],raw15=raw_refs,frames51=refs))
    return out


def own_files():
    return [Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md'),
            Path(teacher.__file__),Path(pose.__file__),*zero.own_files()]


def material(compiler_output,compiler_report_sha,compiler_consume,compiler_consume_sha):
    require(sha(teacher.__file__)==COMPILER_SOURCE_SHA and compiler_report_sha==COMPILER_REPORT_SHA and
            compiler_consume_sha==COMPILER_CONSUME_SHA,'fixed actual compiler source/report/accepted consume seals')
    output=Path(compiler_output).resolve();r=teacher.read(output/'report.json',compiler_report_sha,teacher.REPORT_CAP)
    release=teacher.read(compiler_consume,compiler_consume_sha);compiled=teacher.read(r['prepared'],r['prepare_sha256'])
    require(r['schema']==teacher.NAME+'.report' and r['status']=='common_library_sealed_not_admitted' and
        r['errors']==[] and r['first_error'] is None and r['source_verified'] is r['commands_released'] is r['model_unchanged'] is True and
        r['contract']==compiled['contract']==teacher.contract() and
        release['report_sha256']==compiler_report_sha and release['prepare_sha256']==r['prepare_sha256'] and
        r['model_hash_before']==r['model_hash_after']==compiled['expected_model_hash'] and
        release['plans']==teacher.plans_for(compiled,r['rows']), 'external accepted whole40 release binds actual source/prepare/report')
    # MAIN's external receipt already ran the full saved compiler consumer. Do NOT rerun it/model here.
    p=teacher.read(ROOT/teacher.INPUTS['probe_prepare'][0],teacher.INPUTS['probe_prepare'][1])
    jobs=jobs_for(compiled,release,p);data=adapter.source_material()
    require(p['source_slots']==[x['slot'] for x in data['roots']] and p['warm_traces']==[x['warm_trace'] for x in data['roots']] and
        [t['native'] for t in p['tasks']]==[adapter.root_metadata(data)[s] for s,c in probe.ORDER],'original placement/root/goal/renderer/complete histories')
    bindings=deepcopy(compiled['bindings']);adapter.merge(bindings,data['bindings'])
    references=reference_material(compiled,p,bindings)
    adapter.merge(bindings,{str(output/'report.json'):compiler_report_sha,str(Path(compiler_consume).resolve()):compiler_consume_sha,
        str(Path(r['prepared']).resolve()):r['prepare_sha256']})
    native,native_sha=adapter.constructor_contract();adapter.merge(bindings,{native:native_sha})
    for f in own_files():adapter.merge(bindings,{str(f.resolve()):sha(f)})
    package=dict(schema=NAME+'.prepare',contract=contract(),arguments=dict(compiler_output=str(output),compiler_report_sha=compiler_report_sha,
        compiler_consume=str(Path(compiler_consume).resolve()),compiler_consume_sha=compiler_consume_sha),jobs=jobs,references=references,
        teacher_inputs=[deepcopy(row['blind_input']) for row in r['rows']],anchors=compiled['anchors'],
        probe_digest=probe.digest(p),probe_prepared=str(ROOT/teacher.INPUTS['probe_prepare'][0]),probe_prepare_sha256=teacher.INPUTS['probe_prepare'][1],
        compiler_prepared=r['prepared'],compiler_prepare_sha256=r['prepare_sha256'],bindings=bindings)
    require(len(canonical(package))<=PREP_CAP,'complete own64MiB prepare, no source closure trimming')
    probe.verify_bindings(package);return package,p,data


def prepare(output,**arguments):
    package,_,_=material(**arguments);raw=canonical(package);path=Path(output).absolute()
    require(path.name.startswith(NAME+'_') and path.suffix=='.json' and not path.is_symlink() and
        shutil.disk_usage(path.parent).free>=FLOOR+len(raw),'own prepare prefix/200MiB floor')
    with path.open('xb') as f:require(f.write(raw)==len(raw),'short prepare');f.flush();os.fsync(f.fileno())
    adapter.paired.fsync_dir(path.parent);return dict(prepared=str(path),prepare_sha256=sha(path),bytes=len(raw))


def verify(prepared,prepare_sha):
    saved=teacher.read(prepared,prepare_sha);now,p,data=material(**saved['arguments'])
    require(saved==now and sha(prepared)==prepare_sha,'own complete reconstruction and external prepare binding');return saved,p,data


class Store(helper.Store):
    """Only public emitter/PNG/NPZ/census reused; own limits and exclusive writer."""
    def __init__(self,path):
        self.path=Path(path).absolute();require(self.path.name.startswith(NAME+'_') and not self.path.exists() and
            not self.path.is_symlink() and shutil.disk_usage(self.path.parent).free>=FLOOR+CAP,'new1GiB output +200MiB floor')
        self.path.mkdir();adapter.paired.fsync_dir(self.path.parent)
        self.used=0;self.files={};self.previous=None;self.index=0;self.terminal=False;self.drift=False
    def check(self,size=4096,final=False):
        require(self.used+size<=CAP-(0 if final or self.terminal else FINAL) and shutil.disk_usage(self.path).free>=FLOOR+size,'own cap/reserve/floor')
    def write(self,name,raw,append=False,final=False):
        require(type(raw) is bytes and isinstance(name,str) and Path(name).name==name and name not in ('','.','..'),'flat literal bytes')
        path=self.path/name;old=self.files.get(name)
        require(not path.is_symlink() and (append or old is None) and (not append or old is not None or name=='journal.jsonl'),'exclusive/owned append only')
        self.check(len(raw),final)
        if name=='journal.jsonl':require((old[2] if old else 0)+len(raw)<=JOURNAL_CAP,'own128MiB uncompressed journal')
        fd=os.open(path,os.O_WRONLY|os.O_NOFOLLOW|(os.O_APPEND if old else os.O_CREAT|os.O_EXCL),0o600)
        try:
            st=os.fstat(fd)
            if not (stat.S_ISREG(st.st_mode) and (old is None or (st.st_dev,st.st_ino,st.st_size)==old)):
                self.drift=True;raise ValueError('owned inode/size before append')
            with os.fdopen(fd,'ab' if old else 'wb',closefd=False) as f:
                require(f.write(raw)==len(raw),'short write');f.flush();os.fsync(f.fileno())
        finally:
            st=os.fstat(fd);os.close(fd);self.used+=st.st_size-(old[2] if old else 0);self.files[name]=(st.st_dev,st.st_ino,st.st_size)
        if old is None:adapter.paired.fsync_dir(self.path)


class Meter:
    def __init__(self,store,started=None):
        self.store=store;self.started=time.monotonic() if started is None else started;self.where={}
        self.counts={k:dict(attempted=0,completed=0) for k in CAPS};self.active={k:0 for k in CAPS};self.index=0;self.fault=False;self.secondary=[]
    def call(self,kind,fn,*,cleanup=False,**fields):
        require(not cleanup or kind=='close','close only cleanup bypass')
        if not cleanup:require(not self.fault and time.monotonic()-self.started<WALL,'fault/wall before callback');self.store.check()
        c=self.counts[kind];require(c['attempted']<CAPS[kind],'resource cap before callback '+kind)
        i=self.index;self.index+=1
        kw=dict(**self.where,operation_index=i,resource='internal_physics_step' if kind=='physics_step' else kind,fields=deepcopy(fields))
        terminal=self.store.terminal
        if cleanup:self.store.terminal=True
        try:
            try:self.store.emit('operation_attempted',**kw)
            except BaseException as exc:
                self.fault=True
                if not cleanup:raise
                self.secondary.append(error('close_intent',exc))
            c['attempted']+=1;self.active[kind]+=1
            try:result=fn();c['completed']+=1
            except BaseException as exc:
                try:self.store.emit('operation_failed',**kw,error=error(kind,exc))
                except BaseException as extra:self.fault=True;self.secondary.append(error('failed_operation_journal',extra))
                raise
            finally:self.active[kind]-=1
            try:self.store.emit('operation_returned',**kw)
            except BaseException as exc:
                self.fault=True
                if not cleanup:raise
                self.secondary.append(error('close_return',exc))
            return result
        finally:self.store.terminal=terminal
    def receipt(self):return {('internal_physics_step' if k=='physics_step' else k):deepcopy(v) for k,v in self.counts.items() if v['attempted']}


def check_frame(actual,reference):
    require(actual['pre']==actual['post']==reference['pre']==reference['post'] and
        all(actual['receipt'][k]==reference['receipt'][k] for k in FRAME_FIELDS),
        'original warmup/prefix full PNG bytes,pixels,state,snapshot,goal,renderer identity')


def observe(env,obs,phase,step,row,meter,store,reference,guard):
    class Sink:
        def __getattr__(self,k):return getattr(store,k)
        def emit(self,kind,**fields):
            store.emit(kind,**fields)
            if kind=='fresh_observation' and phase!='suffix':
                ref=next((r for r in reference['frames51'] if (r['receipt']['phase'],r['receipt']['step'],r['receipt']['overlay_index'])==
                    (phase,step,fields['receipt']['overlay_index'])),None)
                require(ref is not None,'all prior warmup/prefix references required; no missing-on allowance')
                check_frame(fields,ref);row['reference_frames_verified']+=1
                store.emit('reference_frame_verified',**meter.where,phase=phase,step=step,
                    overlay_index=fields['receipt']['overlay_index'],source_png=ref['source_png'],file_sha256=ref['receipt']['file_sha256'])
    return helper.observe(env,obs,phase,step,row,meter,Sink(),guard=guard)


def dispatch(env,command,phase,step,row,meter,store,goal,guard,reference=None):
    body=env.unwrapped;require(command.dtype==np.float32 and command.shape==(2,) and body.with_target is False,'literal raw2 off dispatch')
    if body.eval_state(goal,source.vector(body._get_obs()))[0]:raise helper.NativeStop('native terminal before '+phase+'; no further dispatch')
    g,scope=guard
    try:require(g.dispatch(command).tobytes()==command.tobytes(),'same literal action authorized by RGB guard')
    finally:row.setdefault('guards',{})[scope]=g.receipt()
    bits=command.tobytes().hex()
    def callback():
        row['public'][phase]['attempted']+=1
        result=env.step(command.copy());row['public'][phase]['completed']+=1;return result
    obs,reward,t,tr,info=meter.call('public_step',callback,phase=phase,step=step,raw_bits=bits)
    state=source.vector(obs['state']);actual=np.asarray(body.latest_action);snapshot=moving.snapshot(body)
    native_success,native_distance=body.eval_state(goal,state)
    last=dict(phase=phase,step=step,state=state.tolist(),reward=float(reward),terminated=bool(t),truncated=bool(tr),
        native_success=bool(native_success),native_distance=float(native_distance),diagnostics=pose.pose_diagnostics(goal,state))
    row['last_return']=deepcopy(last);row['last_state']=state.tolist();row['native_success']=bool(native_success)
    info_receipt=helper.save_info(store,f'b{row["index"]}_{phase}_{step:02}_info.npz',info)
    event=dict(phase=phase,step=step,raw_bits=bits,actual_bits=actual.tobytes().hex(),actual_dtype=actual.dtype.str,
        state=state.tolist(),snapshot=snapshot,reward=float(reward),terminated=bool(t),truncated=bool(tr),
        native_success=bool(native_success),native_distance=float(native_distance),info=info_receipt)
    store.emit('raw_return',**meter.where,**event)
    require(actual.dtype==np.float32 and actual.shape==(2,) and actual.tobytes()==command.tobytes(),'actual literal raw buffer exact')
    adapter.paired.check_return(env,command,actual,info,goal,float(reward),bool(t))
    if phase!='suffix':
        require(reference is not None and all(event[k]==reference[k] for k in reference),'original actual raw/state/snapshot replay exact')
        row['reference_raw_verified']+=1;store.emit('reference_raw_verified',**meter.where,phase=phase,step=step,raw_bits=bits)
    return obs,bool(t),bool(tr)


def checkpoint(row,frame,step,goal,body,store,meter):
    state=source.vector(frame['state']);success,distance=body.eval_state(goal,state)
    cp=dict(step=step,absolute_step=10+step,frame=deepcopy(frame),native_success=bool(success),native_distance=float(distance),
        diagnostics=pose.pose_diagnostics(goal,state))
    row['checkpoints'].append(cp);row['missing_checkpoints'].remove(step)
    store.emit('checkpoint_recorded',**meter.where,checkpoint=deepcopy(cp))


def episode(env,root,source_plan,task,job,reference,teacher_input,anchor,row,meter,store):
    body=env.unwrapped;packet=root['packets'][job['condition']]
    require(moving.exact(packet['raw_action_history'],moving.warmup_commands()),'original true warmup10 raw bytes')
    warmguard=rgb_guard.RgbObservabilityGuard(root_id=root['root_sha256'],agent_rgb=(65,105,225))
    taskguard=rgb_guard.RgbObservabilityGuard(root_id=root['root_sha256'],agent_rgb=(65,105,225))
    obs,native_goal=place(env,source_plan,root['slot'],meter,store)
    history=[];image,_=observe(env,obs,'warmup',0,row,meter,store,reference,(warmguard,'warmup',0));history.append(image)
    for i,a in enumerate(moving.warmup_commands(),1):
        obs,t,tr=dispatch(env,a,'warmup',i,row,meter,store,native_goal,(warmguard,'warmup'),reference['raw15'][i-1])
        image,_=observe(env,obs,'warmup',i,row,meter,store,reference,(warmguard,'warmup',i))
        if t or tr:raise helper.NativeStop('native stop warmup; no replacement')
        if i in (5,10):history.append(image)
    state=source.vector(obs['state']);goal=source.vector(root['goal']);native=task['native']
    require(moving.exact(np.asarray(history),packet['rgb_history']) and source.root_hash(state,goal)==root['root_sha256'] and
            state.tolist()==native['root_state'],'same-condition source history3 and complete root7 exact')
    bind_goal(body,goal,packet['goal_rgb'],meter,store);bound=invariant(body)
    require(bound['goal_pose']==native['goal_pose'] and bound['renderer']==native['renderer'] and
        moving.image_sha(packet['goal_rgb'])==native['goal_pixels'][job['condition']],'original goal/renderer identity')
    row.update(root_state=state.tolist(),goal=goal.tolist(),root_sha256=root['root_sha256'],goal_pose=bound['goal_pose'],
        goal_pixel_sha256=native['goal_pixels'][job['condition']],initial_exact=True,history_exact=True)
    _,frame=observe(env,obs,'root',0,row,meter,store,reference,(taskguard,'task',0));checkpoint(row,frame,0,goal,body,store,meter)
    for i,a in enumerate(probe.raw(job['prefix_commands'],5),1):
        obs,t,tr=dispatch(env,a,'prefix',10+i,row,meter,store,goal,(taskguard,'task'),reference['raw15'][9+i])
        _,frame=observe(env,obs,'prefix',i,row,meter,store,reference,(taskguard,'task',i))
        if i==5:checkpoint(row,frame,5,goal,body,store,meter)
        if t or tr:raise helper.NativeStop('native stop prefix; no replacement')
    require(row['reference_frames_verified']==51 and row['reference_raw_verified']==15 and
        frame['sha256']==anchor['png_sha256'] and frame['pixel_sha256']==anchor['pixel_sha256'],'all51/15 exact and true teacher15 PNG')
    raw_history=np.concatenate((moving.warmup_commands()[5:],probe.raw(job['prefix_commands'],5)))
    tokens=teacher.kernel.normalized_history(raw_history)
    require(probe.raw(teacher_input['raw_history'],10).tobytes()==raw_history.tobytes() and
        np.array_equal(teacher_input['context']['action_history'],tokens) and
        digest(teacher_input['context'])==job['teacher_context_sha256'],'actual raw history reconstructs own teacher context tokens')
    row.update(start_verified=True,start_state=source.vector(obs['state']).tolist(),teacher_history_verified=True)
    store.emit('teacher_history_verified',**meter.where,raw_history=raw_history.tolist(),tokens=tokens.tolist(),
        context_sha256=job['teacher_context_sha256'],teacher_png_sha256=anchor['png_sha256'])
    fixed=lambda x:{k:v for k,v in x.items() if k!='snapshot'}
    for i,a in enumerate(probe.raw(job['suffix_commands'],5),1):
        require(fixed(invariant(body))==fixed(bound),'task metadata invariant before suffix')
        obs,t,tr=dispatch(env,a,'suffix',i,row,meter,store,goal,(taskguard,'task'))
        _,frame=observe(env,obs,'suffix',i,row,meter,store,reference,(taskguard,'task',5+i))
        require(fixed(invariant(body))==fixed(bound),'original task/renderer invariant through suffix')
        if i==5:checkpoint(row,frame,10,goal,body,store,meter)
        if t or tr:raise helper.NativeStop('native stop suffix, no padding even at step5')
    row.update(status='horizon_completed',end_state=source.vector(obs['state']).tolist(),rgb_guard_accepted=True)


def execute(package,p,data,store,factory,*,authorization,prepared,prepare_sha,started=None):
    require(authorization==AUTH,'MAIN authority before callbacks')
    require(package['schema']==NAME+'.prepare' and package['contract']==contract() and len(package['jobs'])==40 and
        package['probe_digest']==probe.digest(p),'new complete40 ABI, not old probe evidence')
    meter=Meter(store,started);errors=[];seen=[]
    rows=[dict(**deepcopy(j),status='unstarted',public={k:dict(attempted=0,completed=0) for k in ('warmup','prefix','suffix')},
        error=None,secondary_errors=[],root_state=None,goal=None,end_state=None,last_state=None,last_return=None,
        initial_exact=False,history_exact=False,start_verified=False,start_state=None,teacher_history_verified=False,
        reference_frames_verified=0,reference_raw_verified=0,rgb_guard_accepted=False,native_success=None,
        checkpoints=[],missing_checkpoints=[0,5,10],trajectory=[],last_observation=None,PNG=0,actual_latent_Q=None) for j in package['jobs']]
    def fail(phase,exc,row=None):
        e=error(phase,exc);errors.append(e)
        if row is not None:
            if row['error'] is None:row['error']=deepcopy(e)
            else:row['secondary_errors'].append(deepcopy(e))
            row.update(status='failed',end_state=None)
    binding=dict(prepared=str(Path(prepared).resolve()),prepare_sha256=prepare_sha,
        compiler_report_sha256=package['arguments']['compiler_report_sha'],compiler_consume_sha256=package['arguments']['compiler_consume_sha'])
    try:
        store.put('execution_binding.json',binding);store.emit('run_started',contract=contract(),execution_binding=binding)
        for j,row in zip(package['jobs'],rows,strict=True):
            if meter.fault or time.monotonic()-meter.started>=WALL:break
            meter.where=dict(job_digest=j['job_digest']);env=None
            try:
                env=meter.call('constructor',factory);require(all(env is not old and env.unwrapped is not old.unwrapped for old in seen),'fresh independent environment for every library slot')
                seen.append(env);validate_env(env.unwrapped)
                require(env.unwrapped.render_action is False and env.unwrapped.render_size==224 and env.unwrapped.render_mode=='rgb_array','official224 renderer')
                with native_hooks(env,meter,row['secondary_errors']):
                    ai=j['anchor_index'];episode(env,data['roots'][j['root_slot']],data['source_plan'],p['tasks'][j['task']],j,
                        package['references'][ai],package['teacher_inputs'][ai],package['anchors'][ai],row,meter,store)
                if row['secondary_errors']:raise ValueError('native hook cleanup errors retained')
            except helper.NativeStop as exc:row.update(status='native_stopped',stop_reason=str(exc),end_state=None)
            except BaseException as exc:fail('branch',exc,row)
            finally:
                if env is not None:
                    try:meter.call('close',env.close,cleanup=True)
                    except BaseException as exc:fail('close',exc,row)
                if row['secondary_errors'] and row['error'] is None:fail('hook_cleanup',ValueError('see secondary_errors'),row)
                try:store.put(f'branch_{row["index"]:02}.json',row)
                except BaseException as exc:meter.fault=True;fail('branch_receipt',exc,row)
    except BaseException as exc:fail('execution',exc)
    inputs=False
    try:probe.verify_bindings(package);inputs=True
    except BaseException as exc:fail('final_source_hash',exc)
    errors.extend(meter.secondary)
    if time.monotonic()-meter.started>=WALL:fail('wall',ValueError('3600s exhausted, no additional dispatch'))
    store.terminal=True;counts=meter.receipt()
    status='complete' if inputs and not errors and all(row['status']=='horizon_completed' for row in rows) else 'partial'
    try:store.emit('run_finished',status=status,counts=counts,errors=errors)
    except BaseException as exc:fail('terminal_journal',exc);status='partial'
    report=dict(schema=NAME+'.report',status=status,fixed_denominator=40,execution_binding=binding,
        executor_source_sha256=package['bindings'][str(Path(__file__).resolve())],contract=contract(),rows=rows,
        counts=counts,errors=errors,first_error=errors[0] if errors else None,inputs_final_verified=inputs,
        new_neural_calls=0,P1_admission=False,C2_non_greedy_premise=False,independent_confirmation=False,
        time_matched=True,state_velocity_contact_mediation_isolated=False,wall_seconds=time.monotonic()-meter.started,
        journal=dict(file='journal.jsonl',sha256=None),journal_tail=store.previous,artifact_sha256={})
    try:report['artifact_sha256']=store.census();report['journal']['sha256']=report['artifact_sha256']['journal.jsonl']
    except BaseException as exc:fail('terminal_census',exc);report['status']='partial'
    try:
        require(len(canonical(report))<=REPORT_CAP,'bounded complete16MiB report');store.put('report.json',report,final=True);store.census()
    except BaseException as exc:
        fail('final_report',exc);report['status']='partial'
        try:store.put('terminal_failure.json',dict(schema=NAME+'.failure',errors=errors,counts=counts,rows=rows),final=True)
        except BaseException as extra:errors.append(error('terminal_failure_write',extra))
    return report


def run(prepared,prepare_sha,output,factory,*,authorization):
    require(authorization==AUTH,'MAIN explicit authority before reads/output/factory')
    started=time.monotonic();package,p,data=verify(prepared,prepare_sha)
    require(time.monotonic()-started<WALL,'verification included in3600s wall')
    return execute(package,p,data,Store(output),factory,authorization=authorization,prepared=prepared,prepare_sha=prepare_sha,started=started)


def expected_raw(job):
    return [('warmup',i+1,a) for i,a in enumerate(moving.warmup_commands())]+[
        ('prefix',11+i,a) for i,a in enumerate(probe.raw(job['prefix_commands'],5))]+[
        ('suffix',i+1,a) for i,a in enumerate(probe.raw(job['suffix_commands'],5))]


def validate_trace(package,p,r,events):
    require(r['schema']==NAME+'.report' and r['status'] in ('complete','partial') and r['contract']==contract() and
        r['fixed_denominator']==len(r['rows'])==len(package['jobs'])==40 and r['P1_admission'] is r['C2_non_greedy_premise'] is False and
        r['new_neural_calls']==0 and package['probe_digest']==probe.digest(p),'own40 fixed full denominator and no admission')
    require(events[0]['kind']=='run_started' and events[0]['contract']==contract() and
        events[0]['execution_binding']==r['execution_binding'],'initial new authority binding')
    counts={};pending={};known={j['job_digest'] for j in package['jobs']}
    for e in events:
        if 'job_digest' in e:require(e['job_digest'] in known,'no added/substitute job')
        if e['kind'] not in ('operation_attempted','operation_returned','operation_failed'):continue
        key=e['operation_index'];kind=e['resource'];capkey='physics_step' if kind=='internal_physics_step' else kind
        require(capkey in CAPS,'all physical axes known')
        if e['kind']=='operation_attempted':
            require(key not in pending and key==len(pending),'unique sequential operation attempt, no retry')
            pending[key]=dict(job_digest=e['job_digest'],resource=kind,fields=e['fields'],outcome=None)
            counts.setdefault(kind,dict(attempted=0,completed=0))['attempted']+=1
        else:
            require(key in pending and all(pending[key][k]==e[k] for k in ('job_digest','resource','fields')) and pending[key]['outcome'] is None,'same operation return/failure after actual attempt')
            pending[key]['outcome']=e['kind']
            if e['kind']=='operation_returned':counts[kind]['completed']+=1
    require(counts==r['counts'] and all(v['attempted']<=CAPS['physics_step' if k=='internal_physics_step' else k] for k,v in counts.items()),'actual complete attempted/returned resource ledger')
    for j,row in zip(package['jobs'],r['rows'],strict=True):
        require(all(row[k]==v for k,v in j.items()) and row['status'] in ('unstarted','failed','native_stopped','horizon_completed') and row['actual_latent_Q'] is None,'literal library identity, no neural outcome inferred')
        scoped=[e for e in events if e.get('job_digest')==j['job_digest']];expected=expected_raw(j)
        intents=[e for e in scoped if e['kind']=='operation_attempted' and e['resource']=='public_step']
        returns=[e for e in scoped if e['kind']=='raw_return'];complete=[e for e in scoped if e['kind']=='operation_returned' and e['resource']=='public_step']
        require(len(returns)<=len(complete)<=len(intents)<=20,'actual raw-return lower bound')
        for e,(phase,step,a) in zip(intents,expected):require(e['fields']==dict(phase=phase,step=step,raw_bits=a.tobytes().hex()),'every attempted literal action/order')
        stopped=False
        for i,e in enumerate(returns):
            phase,step,a=expected[i]
            require(not stopped and e['phase']==phase and e['step']==step and e['raw_bits']==a.tobytes().hex(),'no extra/changed raw action after native terminal')
            source.vector(e['state']);require(e['snapshot']['state']==e['state'],'actual full state7/snapshot join')
            stopped=e['terminated'] or e['truncated']
            if row['status']=='horizon_completed':require(e['actual_bits']==e['raw_bits'] and e['actual_dtype']==np.dtype('<f4').str,'actual raw float32 bytes')
        for phase in ('warmup','prefix','suffix'):
            a=sum(e['fields']['phase']==phase for e in intents);c=sum(e['fields']['phase']==phase for e in complete)
            require(row['public'][phase]['attempted']<=a and row['public'][phase]['completed']<=row['public'][phase]['attempted'] and
                row['public'][phase]['completed']>=c,'callback census and durable lower bounds')
        refraw=[e for e in scoped if e['kind']=='reference_raw_verified'];refframes=[e for e in scoped if e['kind']=='reference_frame_verified']
        require(row['reference_raw_verified']==len(refraw)<=15 and row['reference_frames_verified']==len(refframes)<=51,'actual prior-only reference checks')
        ref=package['references'][j['anchor_index']]
        for e,rr in zip(returns[:15],ref['raw15']):
            if row['status']=='horizon_completed':require(all(e[k]==v for k,v in rr.items()),'exact actual warm/prefix15 prior binding')
        history=[e for e in scoped if e['kind']=='teacher_history_verified']
        if row['start_verified']:
            b=package['teacher_inputs'][j['anchor_index']];raw=np.concatenate((moving.warmup_commands()[5:],probe.raw(j['prefix_commands'],5)))
            require(len(history)==1 and history[0]['raw_history']==raw.tolist() and history[0]['tokens']==teacher.kernel.normalized_history(raw).tolist() and
                history[0]['context_sha256']==j['teacher_context_sha256']==digest(b['context']) and
                np.array_equal(b['context']['action_history'],history[0]['tokens']) and
                row['start_state']==ref['raw15'][-1]['state'] and row['teacher_history_verified'] is True,'actual pre-suffix history arithmetic, not repeated receipt equality')
        require([e['checkpoint'] for e in scoped if e['kind']=='checkpoint_recorded']==row['checkpoints'],'actual checkpoints durable join')
        steps=[cp['step'] for cp in row['checkpoints']]
        require(steps==[n for n in (0,5,10) if n in steps] and row['missing_checkpoints']==[n for n in (0,5,10) if n not in steps],'reached checkpoints only')
        native=p['tasks'][j['task']]['native']
        for cp in row['checkpoints']:
            require(cp['absolute_step']==10+cp['step'] and cp['frame']['pre']==cp['frame']['post'] and
                cp['diagnostics']==pose.pose_diagnostics(native['goal'],cp['frame']['state']),'actual original jointXY/wrapped native diagnostics')
        if row['status']!='horizon_completed':require(row['end_state'] is None,'partial/stopped endpoint20 not fabricated');continue
        require(len(intents)==len(complete)==len(returns)==20 and not stopped and row['error'] is None and
            row['initial_exact'] is row['history_exact'] is row['start_verified'] is row['teacher_history_verified'] is row['rgb_guard_accepted'] is True and
            len(refraw)==15 and len(refframes)==51 and steps==[0,5,10] and row['PNG']==66 and len(row['trajectory'])==22,'all actual warm/prefix/suffix/full guard denominator')
        require(all(row[k]==native[k] for k in ('root_state','goal','root_sha256','goal_pose')) and
            row['goal_pixel_sha256']==native['goal_pixels'][j['condition']] and returns[9]['state']==row['root_state'] and
            returns[14]['state']==row['start_state'] and returns[19]['state']==row['end_state']==row['checkpoints'][-1]['frame']['state'],'original initial/teacher/terminal full7 identity')
        for scope in ('warmup','task'):
            guard=row['guards'][scope]
            require(guard['primitive_commands_authorized']==10 and guard['abstained'] is False and
                [o['primitive_step'] for o in guard['observations']]==list(range(11)) and
                all(o['canonical_agent_pixels']>0 for o in guard['observations']),'fresh official RGB guard each primitive')
    if r['status']=='complete':
        require(r['errors']==[] and r['first_error'] is None and r['inputs_final_verified'] is True and r['wall_seconds']<WALL and
            all(row['status']=='horizon_completed' for row in r['rows']) and counts=={
                ('internal_physics_step' if k=='physics_step' else k):dict(attempted=v,completed=v) for k,v in CAPS.items()},'complete exact all resource axes; no refunded zero/collisions')


def validate_payloads(package,r,events,load):
    from PIL import Image
    for e in events:
        if e['kind']=='raw_return':
            info=e['info']
            if info['file'] is not None:
                raw=load(info['file']);require(hashlib.sha256(raw).hexdigest()==info['sha256'],'lossless native info NPZ SHA')
                with np.load(io.BytesIO(raw),allow_pickle=False) as z:
                    used=set()
                    def visit(v):
                        if isinstance(v,dict) and set(v)=={'array','dtype','shape'}:
                            a=z[v['array']];used.add(v['array']);require(a.dtype.str==v['dtype'] and list(a.shape)==v['shape'] and not a.dtype.hasobject,'all info arrays original dtype/shape')
                        elif isinstance(v,dict):
                            for item in v.values():visit(item)
                        elif isinstance(v,list):
                            for item in v:visit(item)
                    visit(info['tree']);require(used==set(z.files),'complete info array inventory')
        if e['kind']!='fresh_observation':continue
        f=e['receipt'];raw=load(f['png']);require(hashlib.sha256(raw).hexdigest()==f['file_sha256'],'actual PNG bytes SHA')
        with Image.open(io.BytesIO(raw)) as im:
            require(im.mode=='RGB' and im.format=='PNG','official RGB PNG');pixels=np.asarray(im)
        require(pixels.dtype==np.uint8 and pixels.shape==(224,224,3) and moving.image_sha(pixels)==f['pixel_sha256'] and
            e['pre']==e['post'] and f['snapshot']==e['post']['snapshot'] and f['state']==f['snapshot']['state'],'fresh RGB224 and actual invariant full7')
    for j,row in zip(package['jobs'],r['rows'],strict=True):
        scoped=[e for e in events if e.get('job_digest')==j['job_digest']];frames=[e for e in scoped if e['kind']=='fresh_observation']
        ref=package['references'][j['anchor_index']]
        for e in frames:
            f=e['receipt']
            if f['phase']!='suffix' and row['status']=='horizon_completed':
                reference=next((x for x in ref['frames51'] if (x['receipt']['phase'],x['receipt']['step'],x['receipt']['overlay_index'])==
                    (f['phase'],f['step'],f['overlay_index'])),None)
                require(reference is not None,'known original reference');check_frame(e,reference)
        for i in range(0,len(frames),3):
            triple=frames[i:i+3]
            if len(triple)<3:require(row['status']!='horizon_completed','full off/on/off triplet');continue
            a,b,c=[e['receipt'] for e in triple]
            if row['status']=='horizon_completed':
                require([a['overlay_index'],b['overlay_index'],c['overlay_index']]==[0,1,2] and
                    a['phase']==b['phase']==c['phase'] and a['step']==b['step']==c['step'] and
                    a['file_sha256']==c['file_sha256'] and a['pixel_sha256']==c['pixel_sha256'],'actual off/on/off exact bytes')
        if row['status']!='horizon_completed':continue
        selected=[e for e in frames if e['receipt']['overlay_index']==(1 if j['condition']=='on' else 0)]
        expected=[('warmup',i) for i in range(11)]+[('root',0)]+[('prefix',i) for i in range(1,6)]+[('suffix',i) for i in range(1,6)]
        require([e['receipt'] for e in selected]==row['trajectory'] and
                [(e['receipt']['phase'],e['receipt']['step']) for e in selected]==expected,'all22 actual selected fresh frames')
        for cp,idx in zip(row['checkpoints'],(11,16,21),strict=True):
            e=selected[idx];f=cp['frame'];q=e['receipt']
            require(f['file']==q['png'] and f['sha256']==q['file_sha256'] and f['pixel_sha256']==q['pixel_sha256'] and
                f['pre']==e['pre'] and f['post']==e['post'] and f['state']==q['state'],'actual0/5/10 checkpoint full receipt join')


def read_events(path,tail):
    require(Path(path).stat().st_size<=JOURNAL_CAP,'bounded own128MiB uncompressed journal');events=[];previous=None
    with Path(path).open('rb') as f:
        for line in f:
            require(line.endswith(b'\n'),'complete durable event line');e=adapter.decode(line)
            require(e['index']==len(events) and e['previous_sha256']==previous and
                e['event_sha256']==digest({k:v for k,v in e.items() if k!='event_sha256'}) and line==canonical(e)+b'\n','canonical complete hashchain')
            previous=e['event_sha256'];events.append(e)
    require(events and previous==tail and events[-1]['kind']=='run_finished','terminal whole journal binding');return events


def consume(output,report_sha):
    directory=Path(output).absolute();require(not directory.is_symlink(),'no output directory symlink')
    r=teacher.read(directory/'report.json',report_sha,REPORT_CAP);binding=r['execution_binding']
    package,p,_=verify(binding['prepared'],binding['prepare_sha256'])
    require(binding['compiler_report_sha256']==package['arguments']['compiler_report_sha'] and
        binding['compiler_consume_sha256']==package['arguments']['compiler_consume_sha'] and
        r['executor_source_sha256']==package['bindings'][str(Path(__file__).resolve())],'exact physical/compiler/accepted release bindings')
    require({f.name for f in directory.iterdir()}==set(r['artifact_sha256'])|{'report.json'} and
        sum(f.stat().st_size for f in directory.iterdir())<=CAP,'whole artifact census/output cap')
    def load(name):
        path=teacher.flat(directory,name);require(name in r['artifact_sha256'] and path.is_file() and sha(path)==r['artifact_sha256'][name],'flat whole artifact external SHA')
        return path.read_bytes()
    for name in r['artifact_sha256']:load(name)
    require(adapter.decode(load('execution_binding.json'))==binding and
        r['journal']==dict(file='journal.jsonl',sha256=r['artifact_sha256']['journal.jsonl']),'durable actual bindings')
    events=read_events(directory/'journal.jsonl',r['journal_tail'])
    require(events[-1]['counts']==r['counts'] and events[-1]['status']==r['status'] and events[-1]['errors']==r['errors'],'terminal report/event identity')
    validate_trace(package,p,r,events);validate_payloads(package,r,events,load)
    for row in r['rows']:
        name=f'branch_{row["index"]:02}.json'
        if name in r['artifact_sha256']:require(adapter.decode(load(name))==row,'durable whole branch receipt')
        else:require(row['status']=='unstarted' or r['status']=='partial','missing branch only partial')
    probe.verify_bindings(package);require(sha(directory/'report.json')==report_sha,'report unchanged during consume')
    return dict(schema=NAME+'.consume',status='teacher_cross_library40_physical_verified_not_admitted' if r['status']=='complete'
        else 'partial_teacher_cross_library40_preserved_not_admitted',fixed_denominator=40,report_sha256=report_sha,
        prepare_sha256=binding['prepare_sha256'],compiler_report_sha256=binding['compiler_report_sha256'],
        compiler_consume_sha256=binding['compiler_consume_sha256'],rows=deepcopy(r['rows']),counts=deepcopy(r['counts']),errors=deepcopy(r['errors']),
        P1_admission=False,C2_non_greedy_premise=False,known_roots=2,time_matched=True,state_velocity_contact_mediation_isolated=False,
        independent_confirmation=False,new_neural_calls=0,new_physics_calls=0,actual_latent_Q=None)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare')
    for k in ('output','compiler-output','compiler-report-sha','compiler-consume','compiler-consume-sha'):a.add_argument('--'+k,required=True)
    a=sub.add_parser('verify')
    for k in ('prepared','prepare-sha'):a.add_argument('--'+k,required=True)
    a=sub.add_parser('run')
    for k in ('prepared','prepare-sha','output'):a.add_argument('--'+k,required=True)
    a.add_argument('--authorize-physics',action='store_true')
    a=sub.add_parser('consume')
    for k in ('output','report-sha'):a.add_argument('--'+k,required=True)
    args=vars(ap.parse_args(argv));command=args.pop('command')
    if command=='prepare':result=prepare(**args)
    elif command=='verify':p,_,_=verify(**args);result=dict(status='verified_no_execution',contract=p['contract'])
    elif command=='consume':result=consume(**args)
    else:
        require(args.pop('authorize_physics'),'MAIN explicit --authorize-physics required')
        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401
            return gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224,with_target=False)
        result=run(**args,factory=factory,authorization=AUTH)
    print(canonical(result).decode());return int(result.get('status')=='partial')


if __name__=='__main__':raise SystemExit(main())
