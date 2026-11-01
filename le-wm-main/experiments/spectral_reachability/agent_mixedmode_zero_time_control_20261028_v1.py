"""Independent MAIN-only zero5 matched-time intervention, twenty physical branches.

No model witness is selected here. Parent mixed selections authenticate ancestry
only; every branch reconstructs placement, warmup10, actual zero5 and probe5.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import agent_mixedmode_capability_physical_20261028_v1 as helper

NAME='agent_mixedmode_zero_time_control_20261028_v1'
AUTH='MAIN_ZERO5_MATCHED_TIME20_ONLY'
probe,adapter,rgb_guard=helper.probe,helper.adapter,helper.rgb_guard
ROOT=probe.ROOT
require,sha,canonical=probe.require,probe.sha,probe.canonical
moving,source=adapter.moving,adapter.source
place,bind_goal,invariant=adapter.place,adapter.bind_goal,adapter.invariant
native_hooks,validate_env=adapter.native_hooks,adapter.validate_env
observe,step_raw,NativeStop=helper.observe,helper.step_raw,helper.NativeStop
CAP,FINAL,FLOOR=512*1024**2,32*1024**2,200*1024**2
PREP_CAP,JOURNAL_CAP,REPORT_CAP=64*1024**2,64*1024**2,8*1024**2
WALL=1800
CAPS=dict(constructor=20,close=20,reset=40,public_step=400,physics_step=4080,
    set_state=80,goal_setter=60,goal_render=40,goal_bind=20,explicit_render=1320)


def contract():
    return dict(schema=NAME+'.contract',authority=AUTH,fixed_tasks=4,fixed_denominator=20,
        task_order=[[s,c] for s,c in probe.ORDER],probe_order=[k for k,_ in probe.PROBES],
        intervention='literal_zero5_experimental_intervention_not_model_witness',
        recipe='independent original placement recipe -> original warmup10 -> actual zero5 -> original probe5',
        parent_selection='ancestry_only_never_dispatched',zero_commands=np.zeros((5,2),dtype='<f4').tolist(),
        caps=deepcopy(CAPS),wall_seconds=WALL,output_cap_bytes=CAP,final_reserve_bytes=FINAL,
        prepare_cap_bytes=PREP_CAP,journal_uncompressed_cap_bytes=JOURNAL_CAP,report_cap_bytes=REPORT_CAP,
        disk_floor_bytes=FLOOR,rendering='off/on/off at warmup0..10, root0, zero1..5, probe1..5',
        probe_initial='reuse immediately preceding fresh zero5 frame',
        guard='original RGB224 guard; fresh warmup0..10 and task0..10 sequences',
        units='original literal float32 raw; no /100, tanh, clipping, or model calls',
        public_breakdown=dict(warmup=200,zero_intervention=100,probe=100),
        reset_physics=80,public_physics=4000,render_count=1320,retries=0,replace_failed=False,
        reuse=False,reset_saved_poststate=False,new_neural_calls=0,training=False,Q_fit=False,
        time_matched=True,state_velocity_contact_mediation_isolated=False,
        time_confound='elapsed commanded time matched to selected-prefix after; state/velocity/contact/cache mediated',
        known_roots=2,fresh_roots=0,P1_admission=False,independent_confirmation=False,equal_total_compute=False,
        constructor_partial_internal_work='unknown if inert constructor fails before hooks',
        cleanup='close only may bypass wall/fault; never reset/physics/render')


def error(phase,exc):return dict(phase=phase,type=type(exc).__name__,message=str(exc))


def jobs_for(p):
    """No outcome input. Preserve all fixed parent tasks and probe amplitudes."""
    require(p['schema']==probe.NAME+'.prepare' and p['contract']==probe.contract() and
        len(p['tasks'])==4 and len(p['jobs'])==40,'complete original four-task contract')
    jobs=[]
    for task,(slot,condition) in enumerate(probe.ORDER):
        t=p['tasks'][task]
        require(t['index']==task and (t['plan']['root_slot'],t['plan']['condition'])==(slot,condition) and
            t['context_digest']==probe.digest(t['slot']),'full original context/order')
        for k,(probe_id,_) in enumerate(probe.PROBES):
            parent=p['jobs'][task*10+5+k]
            expected=probe.contract()['probes'][k]['commands']
            require(parent['index']==task*10+5+k and parent['task']==task and parent['role']=='after' and
                parent['probe_id']==probe_id and parent['context_digest']==t['context_digest'] and
                parent['selection_digest']==t['plan']['selection_digest'] and
                parent['probe_commands']==expected and
                parent['prefix_commands']==t['plan']['prefix_commands'] and
                parent['job_digest']==probe.digest({a:b for a,b in parent.items() if a!='job_digest'}),
                'original ordered probes and parent-only ancestry')
            j=dict(index=len(jobs),task=task,root_slot=slot,condition=condition,probe_id=probe_id,
                intervention='zero5_time_control',model_selected_witness=False,
                context_digest=t['context_digest'],parent_after_job_digest=parent['job_digest'],
                parent_selection_digest=t['plan']['selection_digest'],
                intervention_commands=contract()['zero_commands'],probe_commands=deepcopy(expected))
            j['job_digest']=probe.digest(j);jobs.append(j)
    return jobs


def own_files():
    return [Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md'),*helper.own_files()]


def material(probe_prepared,probe_prepare_sha):
    p=probe.verify(probe_prepared,probe_prepare_sha);jobs=jobs_for(p)
    data=adapter.source_material()
    require(p['source_slots']==[r['slot'] for r in data['roots']] and
        p['warm_traces']==[r['warm_trace'] for r in data['roots']] and
        [t['native'] for t in p['tasks']]==[adapter.root_metadata(data)[s] for s,c in probe.ORDER],
        'exact original recipes/root7/goal/renderer and complete histories')
    bindings=deepcopy(p['bindings']);adapter.merge(bindings,data['bindings'])
    native,native_sha=adapter.constructor_contract()
    adapter.merge(bindings,{native:native_sha,str(Path(probe_prepared).resolve()):probe_prepare_sha})
    for f in own_files():adapter.merge(bindings,{str(f.resolve()):sha(f)})
    package=dict(schema=NAME+'.prepare',contract=contract(),jobs=jobs,
        probe_prepared=str(Path(probe_prepared).resolve()),probe_prepare_sha256=probe_prepare_sha,
        probe_digest=probe.digest(p),bindings=bindings)
    require(len(canonical(package))<=PREP_CAP,'complete64MiB prepare, no closure pruning')
    probe.verify_bindings(package)
    return package,p,data


def prepare(output,probe_prepared,probe_prepare_sha):
    package,_,_=material(probe_prepared,probe_prepare_sha);raw=canonical(package);path=Path(output).absolute()
    require(path.name.startswith(NAME+'_') and path.suffix=='.json' and not path.is_symlink() and
        shutil.disk_usage(path.parent).free>=FLOOR+len(raw),'own new prepare prefix/floor')
    with path.open('xb') as f:
        require(f.write(raw)==len(raw),'complete prepare write');f.flush();os.fsync(f.fileno())
    adapter.paired.fsync_dir(path.parent)
    return dict(prepared=str(path),prepare_sha256=sha(path),bytes=len(raw))


def verify(prepared,prepare_sha):
    require(sha(prepared)==probe.hash_value(prepare_sha),'external own prepare SHA')
    saved=probe.read(prepared)
    now,p,data=material(saved['probe_prepared'],saved['probe_prepare_sha256'])
    require(saved==now and sha(prepared)==prepare_sha,'complete regeneration/source closure')
    return saved,p,data


class Store(helper.Store):
    """Public durable byte writer reused; own prefix and independent limits."""
    def __init__(self,path):
        self.path=Path(path).absolute()
        require(self.path.name.startswith(NAME+'_') and not self.path.exists() and not self.path.is_symlink() and
            shutil.disk_usage(self.path.parent).free>=FLOOR+CAP,'own new512MiB output plus200MiB floor')
        self.path.mkdir();adapter.paired.fsync_dir(self.path.parent)
        self.used=0;self.files={};self.previous=None;self.index=0;self.terminal=False;self.drift=False

    def check(self,size=4096,final=False):
        require(self.used+size<=CAP-(0 if final or self.terminal else FINAL) and
            shutil.disk_usage(self.path).free>=FLOOR+size,'own output cap/reserve/floor')


class Meter:
    """New twenty-branch caps; no modification of helper module globals."""
    def __init__(self,store,started=None):
        self.store=store;self.started=time.monotonic() if started is None else started
        self.where={};self.counts={k:dict(attempted=0,completed=0) for k in CAPS}
        self.active={k:0 for k in CAPS};self.index=0;self.fault=False;self.secondary=[]

    def call(self,kind,fn,*,cleanup=False,**fields):
        require(not cleanup or kind=='close','only close can bypass wall/fault')
        if not cleanup:
            require(not self.fault and time.monotonic()-self.started<WALL,'fault/wall before callback')
            self.store.check()
        c=self.counts[kind];require(c['attempted']<CAPS[kind],'resource cap before callback: '+kind)
        i=self.index;self.index+=1
        kw=dict(**self.where,operation_index=i,resource='internal_physics_step' if kind=='physics_step' else kind,
            fields=deepcopy(fields))
        old=self.store.terminal
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
                except BaseException as extra:self.fault=True;self.secondary.append(error('failure_journal',extra))
                raise
            finally:self.active[kind]-=1
            try:self.store.emit('operation_returned',**kw)
            except BaseException as exc:
                self.fault=True
                if not cleanup:raise
                self.secondary.append(error('close_return',exc))
            return result
        finally:self.store.terminal=old

    def receipt(self):
        return {('internal_physics_step' if k=='physics_step' else k):deepcopy(v)
            for k,v in self.counts.items() if v['attempted']}


def episode(env,root,source_plan,task,job,row,meter,store):
    body=env.unwrapped;packet=root['packets'][row['condition']]
    require(moving.exact(packet['raw_action_history'],moving.warmup_commands()),'original literal warmup10')
    warmguard=rgb_guard.RgbObservabilityGuard(root_id=root['root_sha256'],agent_rgb=(65,105,225))
    taskguard=rgb_guard.RgbObservabilityGuard(root_id=root['root_sha256'],agent_rgb=(65,105,225))
    obs,native_goal=place(env,source_plan,root['slot'],meter,store)
    history=[]
    rgb,_=observe(env,obs,'warmup',0,row,meter,store,root['warm_trace'],(warmguard,'warmup',0));history.append(rgb)
    for i,a in enumerate(moving.warmup_commands(),1):
        obs,t,tr=step_raw(env,a,'warmup',i,row,meter,store,native_goal,(warmguard,'warmup'))
        rgb,_=observe(env,obs,'warmup',i,row,meter,store,root['warm_trace'],(warmguard,'warmup',i))
        if t or tr:raise NativeStop('native stop during warmup; no replacement')
        if i in (5,10):history.append(rgb)
    state=source.vector(obs['state']);goal=source.vector(root['goal']);native=task['native']
    require(moving.exact(np.asarray(history),packet['rgb_history']) and
        source.root_hash(state,goal)==root['root_sha256'] and state.tolist()==native['root_state'],
        'same-condition history3 and full7 root exact')
    bind_goal(body,goal,packet['goal_rgb'],meter,store);bound=invariant(body)
    require(bound['goal_pose']==native['goal_pose'] and bound['renderer']==native['renderer'] and
        moving.image_sha(packet['goal_rgb'])==native['goal_pixels'][row['condition']],'original goal/renderer')
    row.update(root_state=state.tolist(),goal=goal.tolist(),root_sha256=root['root_sha256'],
        goal_pose=bound['goal_pose'],goal_pixel_sha256=native['goal_pixels'][row['condition']])
    _,frame=observe(env,obs,'root',0,row,meter,store,guard=(taskguard,'task',0))
    store.emit('intervention_declared',**meter.where,intervention=job['intervention'],
        model_selected_witness=False,parent_selection_digest=job['parent_selection_digest'],
        raw5=job['intervention_commands'])
    for i,a in enumerate(probe.raw(job['intervention_commands'],5),1):
        require(a.tobytes()==np.zeros(2,dtype='<f4').tobytes(),'literal positive zero raw2')
        obs,t,tr=step_raw(env,a,'zero',10+i,row,meter,store,goal,(taskguard,'task'))
        _,frame=observe(env,obs,'zero',i,row,meter,store,guard=(taskguard,'task',i))
        if t or tr:raise NativeStop('native stop during actual zero intervention')
    fixed=lambda x:{k:v for k,v in x.items() if k!='snapshot'}
    require(fixed(invariant(body))==fixed(bound),'unchanged task metadata before probe')
    row.update(start_verified=True,start_state=source.vector(obs['state']).tolist(),zero5_executed=True)
    frame['step']=0;row['frames'].append(frame)
    for i,a in enumerate(probe.raw(job['probe_commands'],5),1):
        obs,t,tr=step_raw(env,a,'probe',i,row,meter,store,goal,(taskguard,'task'))
        _,frame=observe(env,obs,'probe',i,row,meter,store,guard=(taskguard,'task',5+i));row['frames'].append(frame)
        require(fixed(invariant(body))==fixed(bound),'goal/renderer invariant through probe')
        if t or tr:raise NativeStop('native stop during probe, including final step')
    row.update(status='completed',end_state=source.vector(obs['state']).tolist(),rgb_guard_accepted=True)


def execute(package,p,data,store,factory,*,authorization,prepared,prepare_sha,started=None):
    require(authorization==AUTH,'explicit MAIN authority before simulator callbacks')
    require(package['schema']==NAME+'.prepare' and package['contract']==contract() and
        package['probe_digest']==probe.digest(p) and package['jobs']==jobs_for(p),'own20 sealed intervention')
    meter=Meter(store,started);errors=[];seen=[]
    rows=[dict(**deepcopy(j),status='unstarted',attempted=0,completed=0,error=None,secondary_errors=[],
        end_state=None,last_return=None,start_state=None,start_verified=False,zero5_executed=False,
        rgb_guard_accepted=False,frames=[],PNG=0,trajectory=[],checkpoints=[],missing_checkpoints=[],
        last_observation=None) for j in package['jobs']]
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
        for j,row in zip(package['jobs'],rows,strict=True):
            if meter.fault or time.monotonic()-meter.started>=WALL:break
            meter.where=dict(job_digest=j['job_digest']);env=None
            try:
                env=meter.call('constructor',factory)
                require(all(env is not e and env.unwrapped is not e.unwrapped for e in seen),'fresh independent constructor')
                seen.append(env);validate_env(env.unwrapped)
                require(env.unwrapped.render_action is False and env.unwrapped.render_size==224 and
                    env.unwrapped.render_mode=='rgb_array','official224 RGB renderer')
                with native_hooks(env,meter,row['secondary_errors']):
                    episode(env,data['roots'][j['root_slot']],data['source_plan'],p['tasks'][j['task']],j,row,meter,store)
                if row['secondary_errors']:raise ValueError('hook restoration errors retained')
            except NativeStop as exc:row.update(status='native_stopped',stop_reason=str(exc),end_state=None)
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
    if time.monotonic()-meter.started>=WALL:fail('wall',ValueError('1800s exhausted; no more dispatch'))
    store.terminal=True;counts=meter.receipt()
    status='complete' if inputs and not errors and all(r['status']=='completed' for r in rows) else 'partial'
    try:store.emit('run_finished',status=status,counts=counts,errors=errors)
    except BaseException as exc:fail('terminal_journal',exc);status='partial'
    report=dict(schema=NAME+'.report',status=status,fixed_denominator=20,execution_binding=binding,
        executor_source_sha256=package['bindings'][str(Path(__file__).resolve())],
        contract=contract(),rows=rows,errors=errors,counts=counts,inputs_final_verified=inputs,
        new_neural_calls=0,P1_admission=False,independent_confirmation=False,time_matched=True,
        state_velocity_contact_mediation_isolated=False,wall_seconds=time.monotonic()-meter.started,
        journal=dict(file='journal.jsonl',sha256=None),journal_tail=store.previous,artifact_sha256={})
    try:report['artifact_sha256']=store.census();report['journal']['sha256']=report['artifact_sha256']['journal.jsonl']
    except BaseException as exc:fail('terminal_census',exc);report['status']='partial'
    try:
        require(len(canonical(report))<=REPORT_CAP,'bounded final report8MiB')
        store.put('report.json',report,final=True);store.census()
    except BaseException as exc:
        fail('final_report',exc);report['status']='partial'
        try:store.put('terminal_failure.json',dict(errors=errors,counts=counts,rows=rows),final=True)
        except BaseException as extra:errors.append(error('terminal_failure_write',extra))
    return report


def run(prepared,prepare_sha,output,factory,*,authorization):
    require(authorization==AUTH,'MAIN authority before reads/output/factory')
    started=time.monotonic();package,p,data=verify(prepared,prepare_sha)
    require(time.monotonic()-started<WALL,'saved verification counts against wall')
    return execute(package,p,data,Store(output),factory,authorization=authorization,
        prepared=prepared,prepare_sha=prepare_sha,started=started)


def validate_trace(package,p,r,events):
    """Own new ABI; no renaming into old forty-branch evidence."""
    jobs=jobs_for(p);require(package['jobs']==jobs and r['schema']==NAME+'.report' and
        r['contract']==contract() and r['fixed_denominator']==len(r['rows'])==20 and
        r['status'] in ('complete','partial') and r['new_neural_calls']==0 and r['P1_admission'] is False,
        'own20 full denominator/schema and admission flags')
    require(events[0]['kind']=='run_started' and events[0]['contract']==contract() and
        events[0]['execution_binding']==r['execution_binding'],'initial sealed authority')
    counts={};pending={};known={j['job_digest'] for j in jobs}
    for e in events:
        if 'job_digest' in e:require(e['job_digest'] in known,'no extra/replacement branch')
        if e['kind'] not in ('operation_attempted','operation_returned','operation_failed'):continue
        key=e['operation_index'];resource=e['resource']
        capkey='physics_step' if resource=='internal_physics_step' else resource
        require(capkey in CAPS,'known physical axis')
        if e['kind']=='operation_attempted':
            require(key not in pending and key==len(pending),'unique sequential attempt, no retry')
            pending[key]=dict(job_digest=e['job_digest'],resource=resource,fields=e['fields'],outcome=None)
            counts.setdefault(resource,dict(attempted=0,completed=0))['attempted']+=1
        else:
            require(key in pending and all(pending[key][k]==e[k] for k in ('job_digest','resource','fields')) and
                pending[key]['outcome'] is None,'unique returned/failed callback after intent')
            pending[key]['outcome']=e['kind']
            if e['kind']=='operation_returned':counts[resource]['completed']+=1
    require(counts==r['counts'] and all(c['attempted']<=CAPS['physics_step' if k=='internal_physics_step' else k]
        for k,c in counts.items()),'actual attempted/returned complete budget')
    for j,row in zip(jobs,r['rows'],strict=True):
        require(all(row[k]==v for k,v in j.items()) and row['status'] in ('completed','failed','native_stopped','unstarted'),
            'new intervention identity and parent ancestry, no model selection')
        scoped=[e for e in events if e.get('job_digest')==j['job_digest']]
        intents=[e for e in scoped if e['kind']=='operation_attempted' and e['resource']=='public_step']
        expected=[('warmup',i+1,a) for i,a in enumerate(moving.warmup_commands())]+[
            ('zero',11+i,a) for i,a in enumerate(probe.raw(j['intervention_commands'],5))]+[
            ('probe',i+1,a) for i,a in enumerate(probe.raw(j['probe_commands'],5))]
        require(len(intents)<=20,'no extra public action')
        for e,(phase,step,a) in zip(intents,expected):
            require(e['fields']==dict(phase=phase,step=step,raw_bits=a.tobytes().hex()),'every attempted literal buffer/order')
        rawreturns=[e for e in scoped if e['kind']=='raw_return']
        successful=[e for e in scoped if e['kind']=='operation_returned' and e['resource']=='public_step']
        require(len(rawreturns)<=len(successful)<=len(intents),'durable return lower bound')
        stopped=False
        for i,e in enumerate(rawreturns):
            phase,step,a=expected[i]
            require(not stopped and e['phase']==phase and e['step']==step and
                e['raw_bits']==a.tobytes().hex(),'no dispatch after terminal; true replay order')
            # Incorrect actual buffer is evidence of a failed branch, not fabricated success.
            if row['status']=='completed':require(e['actual_bits']==e['raw_bits'] and
                e['actual_dtype']==np.dtype('<f4').str,'actual command same float32 authority')
            source.vector(e['state']);require(e['snapshot']['state']==e['state'],'native7 snapshot/return join')
            stopped=e['terminated'] or e['truncated']
        attempts=[e for e in scoped if e['kind']=='probe_attempted'];returns=[e for e in scoped if e['kind']=='probe_returned']
        require(row['attempted']==len(attempts) and row['completed']==len(returns) and
            len(returns)<=len(attempts)<=5,'probe attempt/return lower bounds')
        for i,e in enumerate(attempts):require(e['step']==i+1 and
            e['raw_bits']==probe.raw(j['probe_commands'],5)[i].tobytes().hex(),'probe intent original amplitude')
        for i,e in enumerate(returns):require(e['step']==i+1 and e['raw_bits']==attempts[i]['raw_bits'],'probe return order')
        if row['status']!='completed':
            require(row['end_state'] is None,'failed/stopped/unstarted endpoint null');continue
        require(len(intents)==len(successful)==len(rawreturns)==20 and len(returns)==5 and not stopped and
            row['error'] is None and row['start_verified'] is row['zero5_executed'] is row['rgb_guard_accepted'] is True,
            'twenty actual returned actions, not reused zero prefix')
        native=p['tasks'][j['task']]['native']
        require(all(row[k]==native[k] for k in ('root_state','goal','root_sha256','goal_pose')) and
            row['goal_pixel_sha256']==native['goal_pixels'][j['condition']] and
            rawreturns[9]['state']==native['root_state'] and row['start_state']==rawreturns[14]['state'] and
            row['end_state']==rawreturns[19]['state']==returns[-1]['state'],'full original root7 and actual start/end')
        for scope in ('warmup','task'):
            g=row['guards'][scope]
            require(g['primitive_commands_authorized']==10 and g['abstained'] is False and
                [o['primitive_step'] for o in g['observations']]==list(range(11)) and
                all(o['canonical_agent_pixels']>0 for o in g['observations']),'existing fresh RGB guard every primitive')
        require(len(row['frames'])==6 and len(row['trajectory'])==22 and row['PNG']==66,'complete fresh frame denominator')
        for i,f in enumerate(row['frames']):
            require(f['step']==i and f['pre']==f['post'] and f['state']==f['post']['snapshot']['state'] and
                f['state']==(row['start_state'] if i==0 else returns[i-1]['state']) and
                f['pre']['goal_pose']==native['goal_pose'] and f['pre']['renderer']==native['renderer'],
                'actual probe frame alignment/renderer/goal')
    if r['status']=='complete':
        require(r['errors']==[] and r['inputs_final_verified'] is True and r['wall_seconds']<WALL and
            all(x['status']=='completed' for x in r['rows']) and
            counts=={('internal_physics_step' if k=='physics_step' else k):dict(attempted=v,completed=v)
                for k,v in CAPS.items()},'complete exact all resource axes, not just upper bounds')


def validate_payloads(package,p,r,events,load):
    """Validate every saved PNG/NPZ, including available partial branch data."""
    from PIL import Image
    for e in events:
        if e['kind']=='raw_return':
            receipt=e['info']
            if receipt['file'] is not None:
                raw=load(receipt['file']);require(probe.hashlib.sha256(raw).hexdigest()==receipt['sha256'],'info NPZ SHA')
                with np.load(io.BytesIO(raw),allow_pickle=False) as z:
                    used=set()
                    def visit(v):
                        if isinstance(v,dict) and set(v)=={'array','dtype','shape'}:
                            a=z[v['array']];used.add(v['array'])
                            require(a.dtype.str==v['dtype'] and list(a.shape)==v['shape'] and not a.dtype.hasobject,
                                'lossless returned info arrays')
                        elif isinstance(v,dict):
                            for x in v.values():visit(x)
                        elif isinstance(v,list):
                            for x in v:visit(x)
                    visit(receipt['tree']);require(used==set(z.files),'complete info inventory')
        if e['kind']!='fresh_observation':continue
        f=e['receipt'];raw=load(f['png'])
        require(probe.hashlib.sha256(raw).hexdigest()==f['file_sha256'],'PNG bytes SHA')
        with Image.open(io.BytesIO(raw)) as im:a=np.asarray(im)
        require(a.shape==(224,224,3) and a.dtype==np.uint8 and moving.image_sha(a)==f['pixel_sha256'],
            'official full precision RGB224')
        require(e['pre']==e['post'] and f['snapshot']==e['post']['snapshot'] and
            f['state']==f['snapshot']['state'],'render invariant actual state7')
    for j,row in zip(package['jobs'],r['rows'],strict=True):
        scoped=[e for e in events if e.get('job_digest')==j['job_digest']]
        frames=[e for e in scoped if e['kind']=='fresh_observation']
        for n in range(0,len(frames),3):
            trip=frames[n:n+3]
            if len(trip)<3:
                require(row['status']!='completed','complete triplet');continue
            a,b,c=[e['receipt'] for e in trip]
            require([a['overlay_index'],b['overlay_index'],c['overlay_index']]==[0,1,2] and
                a['phase']==b['phase']==c['phase'] and a['step']==b['step']==c['step'] and
                a['file_sha256']==c['file_sha256'] and a['pixel_sha256']==c['pixel_sha256'],
                'fresh off/on/off exact replay')
            if a['phase']=='warmup':
                for e in trip:
                    ref=adapter.reference_for(p['warm_traces'][j['root_slot']],e['receipt']['step'],e['receipt']['overlay_index'])
                    if ref is not None:adapter.check_warm_frame(e,ref)
        selected=[e for e in frames if e['receipt']['overlay_index']==(1 if j['condition']=='on' else 0)]
        if row['status']=='completed':
            require([e['receipt'] for e in selected]==row['trajectory'],'all selected fresh receipts join')
            expected=[('warmup',i) for i in range(11)]+[('root',0)]+[('zero',i) for i in range(1,6)]+[('probe',i) for i in range(1,6)]
            require([(e['receipt']['phase'],e['receipt']['step']) for e in selected]==expected,'all zero observations actual')
            for f,e in zip(row['frames'],selected[16:],strict=True):
                q=e['receipt'];require(f['file']==q['png'] and f['sha256']==q['file_sha256'] and
                    f['pixel_sha256']==q['pixel_sha256'] and f['pre']==e['pre'] and f['post']==e['post'],
                    'probe frame full receipt join, initial is true zero5')


def consume(output,report_sha):
    directory=Path(output).resolve();report_path=directory/'report.json'
    require(sha(report_path)==probe.hash_value(report_sha),'external actual report SHA')
    r=probe.read(report_path);binding=r['execution_binding']
    package,p,_=verify(binding['prepared'],binding['prepare_sha256'])
    require(binding['probe_prepared']==package['probe_prepared'] and
        binding['probe_prepare_sha256']==package['probe_prepare_sha256'] and
        r['executor_source_sha256']==package['bindings'][str(Path(__file__).resolve())],'own input/source binding')
    require(not directory.is_symlink() and
        {f.name for f in directory.iterdir()}==set(r['artifact_sha256'])|{'report.json'} and
        sum(f.stat().st_size for f in directory.iterdir())<=CAP,'complete output file census/cap')
    def load(name):
        require(Path(name).name==name and name not in ('','.','..') and name in r['artifact_sha256'], 'own flat artifact')
        path=directory/name;require(path.is_file() and not path.is_symlink() and
            sha(path)==r['artifact_sha256'][name],'complete actual file SHA')
        return path.read_bytes()
    for name in r['artifact_sha256']:load(name)
    require(json.loads(load('execution_binding.json'))==binding and r['journal']==dict(file='journal.jsonl',
        sha256=r['artifact_sha256']['journal.jsonl']),'durable binding/journal SHA')
    events=helper.read_events(directory/'journal.jsonl',r['journal_tail'])
    require(events[-1]['counts']==r['counts'] and events[-1]['status']==r['status'] and
        events[-1]['errors']==r['errors'],'terminal receipt exact')
    validate_trace(package,p,r,events);validate_payloads(package,p,r,events,load)
    for row in r['rows']:
        name=f'branch_{row["index"]:02}.json'
        if name in r['artifact_sha256']:require(json.loads(load(name))==row,'durable branch exact')
        else:require(row['status']=='unstarted' or r['status']=='partial','missing branch only partial')
    probe.verify_bindings(package);require(sha(report_path)==report_sha,'report unchanged during consume')
    return dict(schema=NAME+'.consume',status='zero_time_control_responses_verified_not_admitted' if r['status']=='complete'
        else 'partial_zero_time_evidence_preserved_not_admitted',fixed_denominator=20,report_sha256=report_sha,
        prepare_sha256=binding['prepare_sha256'],rows=deepcopy(r['rows']),counts=deepcopy(r['counts']),
        errors=deepcopy(r['errors']),time_matched=True,state_velocity_contact_mediation_isolated=False,
        P1_admission=False,independent_confirmation=False,new_neural_calls=0,new_physics_calls=0)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
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
        p,_,_=verify(**a);result=dict(status='verified_no_execution',contract=p['contract'])
    elif cmd=='consume':result=consume(**a)
    else:
        require(a.pop('authorize_physics'),'MAIN explicit --authorize-physics required')
        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401, only behind MAIN authorization
            return gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224,with_target=False)
        result=run(**a,factory=factory,authorization=AUTH)
    print(canonical(result).decode());return int(result.get('status')=='partial')


if __name__=='__main__':raise SystemExit(main())
