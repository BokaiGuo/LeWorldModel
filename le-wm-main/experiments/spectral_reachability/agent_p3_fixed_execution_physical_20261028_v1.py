"""MAIN-only fixed requested five-step native execution; no model/child loader.

All four task/method requests stay. No suffix extension, padding, source-schema
projection or old outcome reuse. Saved verification is not scientific release.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib
import io
import os
from pathlib import Path
import shutil
import time

import numpy as np

from experiments.spectral_reachability import agent_p3_fixed_execution_contract_20261028_v1 as fixed
from experiments.spectral_reachability import agent_allmode_physical_20261028_v1 as allmode
from experiments.spectral_reachability import p1_constructive_moving_history_encode_cuda_20261027_v1 as codec

NAME='agent_p3_fixed_execution_physical_20261028_v1'
PRODUCER='agent_p3_fixed_execution_cuda_20261028_v1'
SUPPORTED_SOURCE_SHA='a307615d0366a475f876c2cdf0676a03b202f7470a5a96f5d2b4988e9ff67a31'
CUDA_PREPARE='results/agent_p3_fixed_execution_cuda_20261028_v1_prepare_MAIN01.json'
CUDA_PREPARE_SHA='2e680c6ff62e3a93b9c8cb7eef13d96510fe6d0b699561aedd84b616490bbdf9'
CUDA_REPORT_SHA='8067a743e3893b120ac8d57582c8552cb5b1dec182f7ca4bd0c6f70ffacaeeb5'
FIXED_SHA='7cb1cb3a89895755d2bb92cc9dd62de617e4a48471aa388f6fff4eab48a3aae9'
HELPER_SHA='c0842f63685718e3d1d562fbbe5e2e3fdf9103954804f4276a1186d8db91efe4'
AUTH='MAIN_P3_FIXED5_PHYSICS_ONLY'
RECORDED='fixed5_physical_recorded_not_admitted'
ROOT=allmode.ROOT
ORDER=[(s,m) for s in (0,1) for m in ('fixed','adaptive')]
FLAGS=dict(fixed.FLAGS)
SOURCE_SHA=allmode.SOURCE_SHA
CAP,FINAL,FLOOR,WALL,PREP_CAP=32*1024**2,4*1024**2,200*1024**2,900,64*1024**2
ARGS={'producer_output','producer_report_sha256','producer_source_sha256','producer_prepared','producer_prepare_sha256'}
core=fixed.core
base,moving,source,paired=allmode.base,allmode.moving,allmode.source,allmode.paired
require,sha,raw,read,decode=allmode.require,allmode.sha,allmode.raw,allmode.read,allmode.decode
actual_hash,absolute,merge=allmode.actual_hash,allmode.absolute,allmode.merge
place,bind_goal,invariant=allmode.place,allmode.bind_goal,allmode.invariant
observe,native_hooks,validate_env=allmode.observe,allmode.native_hooks,allmode.validate_env
source_material,saved_artifacts=allmode.source_material,allmode.saved_artifacts
load_producer=importlib.import_module
read_export,validate_export,load_blind=codec.read,codec.task_interfaces,codec.load_blind


def contract():
    return dict(schema=NAME+'.contract',fixed_denominator=4,order=[list(x) for x in ORDER],known_roots=2,fresh_roots=0,
        requested_steps=5,requested_packets=1,warmup=10,checkpoints=[0,5],planning_horizons=[1,3],
        action_source='literal float32 winner first5 only; H3 remaining10 NOT dispatched',
        no_padding=True,no_child_replanning=True,no_root_signal_for_child=True,no_outcome_selection=True,
        raw_action_scale=100,relative=True,control_with_target=False,no_goal_pose_edit=True,
        success='native jointXY<20 AND min(abs(angle),2pi-abs(angle))<pi/9',reward='negative native7 Euclidean distance INCLUDING velocities',
        wall_seconds=WALL,output_cap_bytes=CAP,final_reserve_bytes=FINAL,disk_floor_bytes=FLOOR,prepare_cap_bytes=PREP_CAP,
        wall_scope='execute only; complete saved verify BEFORE timer, NOT whole job',
        caps=dict(constructor=4,close=4,reset=8,public_step=60,physics_step=616,set_state=16,
            goal_setter=12,goal_render=8,goal_bind=4,explicit_render=204),
        resource_scope='real dispatch intent/entry/return; nested axes NOT added; constructor failure internal work unknown',
        equal_requested_execution_horizon=True,equal_FLOPs=False,equal_total_compute=False,neural_calls=0,training=False,
        **FLAGS)


def commands(value,n=5):
    require(type(value) is list and len(value)==n and all(type(r) is list and len(r)==2 and
        all(type(x) in (int,float) for x in r) for r in value),'literal raw action count/shape/type, never bool')
    a=np.asarray(value,np.float64)
    require(np.isfinite(a).all() and np.max(np.abs(a))<=1 and np.array_equal(a,a.astype(np.float32).astype(np.float64)),
        'exact lossless float32 raw, no tanh/clip/scale/cast invention')
    return np.ascontiguousarray(a,np.float32)


def metrics(goal,state):
    g,s=source.vector(goal),source.vector(state);require(g.shape==s.shape==(7,),'native7 task states')
    delta=s-g;native=base.previous.metric(goal,state)
    return dict(native=native,native_success=native['native_success'],joint_xy_error=float(np.linalg.norm(delta[:4])),
        native_angle_term=native['native_angle_term'],wrapped_angle_error=float(abs(np.arctan2(np.sin(delta[4]),np.cos(delta[4])))),
        velocity_error=float(np.linalg.norm(delta[5:])),native7_distance=float(np.linalg.norm(delta)),native7_reward=-float(np.linalg.norm(delta)))


def validate_requests(receipt,tasks,expected):
    actual_hash(expected)
    require(receipt['schema']==PRODUCER+'.consume' and receipt['status']=='saved_prefix_verified_not_admitted' and
        receipt['report_sha256']==expected and len(tasks)==2 and len(receipt['prefixes'])==len(receipt['winners'])==4 and
        all(receipt[k] is False for k in FLAGS) and receipt['equal_requested_execution_horizon'] is True and
        receipt['equal_actual_physical_execution_horizon'] is None and receipt['actual_physics_status']=='unknown_not_run',
        'complete real producer.consume, not contract/prepare/RAM/partial release')
    fields={'schema','slot','mode','binding','root_context_sha256','planning_horizon_packets','planned_action_steps',
        'requested_execution_steps','commands','commands_sha256','full_plan_actions_sha256','nonexecuted_suffix_steps',
        'predicted_execution_endpoint_sha256','predicted_execution_modelQ','physical_root_binding','physical_goal_binding',
        'physical_result','status'}|set(FLAGS)
    plans=[]
    for i,(prefix,winner) in enumerate(zip(receipt['prefixes'],receipt['winners'],strict=True)):
        slot,mode=ORDER[i];q=prefix['request'];task=core.validate_task(tasks[slot]);h=q['planning_horizon_packets']
        require(type(prefix['slot']) is type(winner['slot']) is type(q['slot']) is int and
            (prefix['slot'],prefix['mode'])==(winner['slot'],winner['mode'])==(q['slot'],q['mode'])==(slot,mode) and
            prefix['status']=='model_verified_physical_blocked' and prefix['error'] is None,'all4 literal task/method order')
        require(set(q)==fields and q['schema']==fixed.NAME+'.execution_request' and
            q['binding']==task['binding'] and q['root_context_sha256']==fixed.digest(task['root_context']) and
            winner['task_binding_sha256']==fixed.digest(task['binding']) and winner['root_context_sha256']==q['root_context_sha256'],
            'exact original model/task/normalizer/Q/context identity, no extra outcome fields')
        require(type(h) is int and h in (1,3) and type(winner['horizon']) is int and winner['horizon']==h and
            q['planned_action_steps']==5*h and q['nonexecuted_suffix_steps']==5*h-5 and q['requested_execution_steps']==5 and
            all(q[k] is False for k in FLAGS) and q['physical_root_binding'] is q['physical_goal_binding'] is q['physical_result'] is None and
            q['status']=='prefix_model_verified_physical_blocked','planning H is NOT execution length or physical release')
        a=commands(q['commands']);full=commands(winner['actions'],5*h)
        require(a.tobytes()==full[:5].tobytes() and q['commands_sha256']==fixed.digest(q['commands']) and
            q['full_plan_actions_sha256']==fixed.digest(winner['actions']),'same actual selected first packet, no H3 suffix or H1 extension')
        actual_hash(q['predicted_execution_endpoint_sha256'])
        require(type(q['predicted_execution_modelQ']) in (int,float) and np.isfinite(q['predicted_execution_modelQ']) and
            q['predicted_execution_modelQ']>=0,'bound model-only prefix diagnostic, not task success')
        plans.append(dict(index=i,slot=slot,root_slot=slot,mode=mode,condition='off',arm=mode,
            task_id=q['binding']['task_id'],selection_digest=fixed.digest(q),commands=deepcopy(q['commands']),request=deepcopy(q)))
    return plans


def bridge(tasks,coldslots,exports,packets,data):
    """No name/nearby-state inference: exact encoded task and RGB/raw ancestry."""
    require(len(tasks)==len(coldslots)==len(exports)==len(packets)==len(data['roots'])==2,'fixed two-task source bridge')
    for s,(t,c,e,packet,root) in enumerate(zip(tasks,coldslots,exports,packets,data['roots'],strict=True)):
        core.validate_task(t);b=t['binding'];original=e['task'];off=root['packets']['off']
        require(type(c['source_slot']) is type(e['source_slot']) is int and c['source_slot']==e['source_slot']==s and
            all(c[k]==b[k] for k in ('root_sha256','task_id','task_sha256','task_contract_sha256','blind_sha256')) and
            e['task_id']==b['task_id'] and e['blind_sha256']==b['blind_sha256'] and e['with_target'] is False and
            fixed.digest(e['task_contract'])==b['task_contract_sha256'],'literal original cold task/contract/blind slot identity')
        require(all(original[k]==t[k] for k in ('root_context','goal','projection','action_mean','action_std')) and
            b['root_sha256']==root['root_sha256'],'same complete encoded task and actual native7 source root, not latent coincidence')
        require(all(moving.exact(packet[k],off[k]) for k in ('rgb_history','goal_rgb','raw_action_history')) and
            np.asarray(packet['raw_action_history']).dtype==np.float32 and
            np.asarray(packet['raw_action_history']).tobytes()==moving.warmup_commands().tobytes(),
            'exact original off RGB history3/goal/raw10 to physical source packets')
    return allmode.root_metadata(data)


def own_files():return [Path(__file__).resolve(),ROOT/f'tests/{NAME}_test.py',ROOT/f'innerdocs/{NAME}.md']


def material(arguments):
    require(type(arguments) is dict and set(arguments)==ARGS,'actual external producer source/prepare/report authorities required')
    for k in ('producer_report_sha256','producer_source_sha256','producer_prepare_sha256'):actual_hash(arguments[k])
    code=ROOT/'experiments/spectral_reachability'/(PRODUCER+'.py');output=Path(absolute(arguments['producer_output']))
    rp=output/'report.json';pp=Path(absolute(arguments['producer_prepared']))
    require(pp==ROOT/CUDA_PREPARE and arguments['producer_prepare_sha256']==CUDA_PREPARE_SHA and
        arguments['producer_report_sha256']==CUDA_REPORT_SHA,'actual MAIN fixed5 batch seals, not another/future batch')
    require(arguments['producer_source_sha256']==SUPPORTED_SOURCE_SHA==sha(code) and sha(fixed.__file__)==FIXED_SHA and
        sha(allmode.__file__)==HELPER_SHA,'reviewed source ABI BEFORE import; source changes require new MAIN review')
    require(sha(rp)==arguments['producer_report_sha256'] and sha(pp)==arguments['producer_prepare_sha256'],'actual existing terminal report/prepare, never future SHA')
    report=read(rp,256*1024**2);prepared=read(pp,16*1024**2)
    require(report['schema']==PRODUCER+'.report' and report['status']=='saved_model_prefix_not_admitted' and report['errors']==[] and
        report['prepared']==str(pp) and report['prepare_sha256']==arguments['producer_prepare_sha256'],'complete real CUDA report and its actual prepare')
    producer=load_producer('experiments.spectral_reachability.'+PRODUCER)
    require(Path(producer.__file__).resolve()==code,'fixed real producer module, not an injected runner')
    consumed=producer.consume(output,arguments['producer_report_sha256'])
    tasks=prepared['panel']['base']['tasks'];plans=validate_requests(consumed,tasks,arguments['producer_report_sha256'])
    bindings={};merge(bindings,prepared['sources'])
    coldpath=ROOT/'results/agent_p3_resume_20261027_next_prepare01.json'
    require(str(coldpath) in bindings and sha(coldpath)==bindings[str(coldpath)],'bound original cold task manifest, not directory discovery')
    cold=read(coldpath,16*1024**2);exports=[];packets=[]
    for slot,t in zip(cold['slots'],tasks,strict=True):
        path=absolute(slot['task_export']);expected=t['binding']['task_sha256']
        require(bindings[path]==expected==sha(path),'actual bound original encoded task export SHA')
        e=read_export(path);validate_export(e);exports.append(e)
        paths=[f for f,h in bindings.items() if h==t['binding']['blind_sha256']]
        require(len(paths)==1,'unambiguous explicit SHA-bound original blind input edge')
        packet=load_blind(paths[0],t['binding']['blind_sha256']);packets.append(packet)
    data=source_material();roots=bridge(tasks,cold['slots'],exports,packets,data);merge(bindings,data['bindings'])
    merge(bindings,{str(code):arguments['producer_source_sha256'],str(rp):arguments['producer_report_sha256'],str(pp):arguments['producer_prepare_sha256']})
    for f in own_files():merge(bindings,{str(f.resolve()):sha(f)})
    for f in output.iterdir():
        require(f.is_file() and not f.is_symlink(),'regular complete producer artifacts');merge(bindings,{str(f):sha(f)})
    # Explicitly seal every newly reused helper, including source recipe/render hooks.
    for f in [Path(fixed.__file__).resolve(),Path(allmode.__file__).resolve(),Path(codec.__file__).resolve(),*allmode.own_files()]:merge(bindings,{str(f):sha(f)})
    for f,h in bindings.items():require(sha(f)==h,'complete final input/source/model/task hash closure')
    p=dict(schema=NAME+'.prepare',contract=contract(),arguments=deepcopy(arguments),producer_consume=consumed,
        producer_report_sha256=arguments['producer_report_sha256'],producer_prepare_sha256=arguments['producer_prepare_sha256'],
        source_report_sha256=SOURCE_SHA,tasks=deepcopy(tasks),plans=plans,source_roots=roots,
        source_slots=[deepcopy(r['slot']) for r in data['roots']],warm_traces=[deepcopy(r['warm_trace']) for r in data['roots']],bindings=bindings)
    require(len(raw(p))<=PREP_CAP,'bounded complete own prepare, do not drop bindings');return p,data


def write_new(output,value):
    path=Path(absolute(str(output)));body=raw(value)
    require(path.name.startswith(NAME+'_') and not path.exists() and not path.is_symlink() and len(body)<=PREP_CAP and
        shutil.disk_usage(path.parent).free>=FLOOR+len(body),'NEW exclusive bounded own metadata +200MiB floor')
    with path.open('xb') as stream:
        require(stream.write(body)==len(body),'complete write');stream.flush();os.fsync(stream.fileno())
    paired.fsync_dir(path.parent);return dict(prepared=str(path),prepare_sha256=sha(path),bytes=len(body))


def prepare(output,arguments):
    path=Path(absolute(output));require(not path.exists() and not path.is_symlink(),'NEW prepare BEFORE complete saved consumption')
    p,_=material(arguments);return write_new(path,p)


def verify(prepared,expected):
    actual_hash(expected);absolute(str(prepared));require(sha(prepared)==expected,'external own prepare SHA')
    p=read(prepared,PREP_CAP);now,data=material(p['arguments'])
    require(p==now and sha(prepared)==expected,'complete real producer.consume/task/source bridge reconstruction');return p,data


class Store(allmode.Store):
    def __init__(self,path):
        self.path=Path(absolute(str(path)))
        require(self.path.name.startswith(NAME+'_') and not self.path.exists() and not self.path.is_symlink() and
            self.path.parent.is_dir() and shutil.disk_usage(self.path.parent).free>=FLOOR+CAP,'NEW fixed5 output32MiB +200MiB floor')
        self.path.mkdir();paired.fsync_dir(self.path.parent);self.previous=None;self.index=0


class Meter:
    def __init__(self,store,started=None):
        self.store=store;self.started=time.monotonic() if started is None else started;self.where={};self.receipts=[];self.secondary=[];self.fault=False
        self.counts={k:dict(attempted=0,completed=0) for k in contract()['caps']};self.active={k:0 for k in self.counts}
    def call(self,kind,fn,*,cleanup=False,**fields):
        require(not cleanup or kind=='close','only close cleanup may bypass fault/wall')
        if not cleanup:require(not self.fault and time.monotonic()-self.started<WALL,'wall/fault before dispatch');self.store.check()
        count=self.counts[kind];require(count['attempted']<contract()['caps'][kind],'cap before actual '+kind)
        item=dict(index=len(self.receipts),resource=kind,**deepcopy(self.where),**deepcopy(fields),status='pending')
        try:self.store.emit('operation_dispatch_intent',receipt=item)
        except BaseException as exc:
            self.fault=True
            if not cleanup:raise
            self.secondary.append(base.error('cleanup_intent',exc))
        self.receipts.append(item);count['attempted']+=1;self.active[kind]+=1
        try:value=fn();count['completed']+=1;item['status']='returned'
        except BaseException as exc:
            item.update(status='failed',error=base.error(kind,exc))
            try:self.store.emit('operation_failed',receipt=item)
            except BaseException as extra:self.fault=True;self.secondary.append(base.error('failure_journal',extra))
            raise
        finally:self.active[kind]-=1
        try:self.store.emit('operation_returned',receipt=item)
        except BaseException as exc:
            self.fault=True
            if not cleanup:raise
            self.secondary.append(base.error('cleanup_return',exc))
        return value


def validate_execution(p):
    require(p['schema']==NAME+'.prepare' and p['contract']==contract() and p['source_report_sha256']==SOURCE_SHA and
        len(p['source_slots'])==len(p['source_roots'])==len(p['warm_traces'])==2 and
        validate_requests(p['producer_consume'],p['tasks'],p['producer_report_sha256'])==p['plans'],'own fixed5 schema/actual four consumed requests')


def dispatch(env,command,phase,step,row,meter,store,goal):
    class Sink:
        def __getattr__(self,key):return getattr(store,key)
        def emit(self,kind,**fields):
            store.emit(kind,**fields)
            if kind=='returned_transition' and fields['phase']=='plan':row['actual_returns'].append(deepcopy(fields))
    result=allmode.dispatch(env,command,phase,step,row,meter,Sink(),goal);actual=result[-1]
    require(actual.dtype==np.float32 and actual.shape==(2,) and actual.tobytes()==command.tobytes(),
        'native returned actual raw float32 bits, not merely numerically equal')
    return result


def episode(env,root,source_plan,plan,row,meter,store):
    body=env.unwrapped;packet=root['packets']['off'];trace=root['warm_trace'];a=np.asarray(packet['raw_action_history'])
    require(a.shape==(10,2) and a.dtype==np.float32 and a.tobytes()==moving.warmup_commands().tobytes(),'original nonzero float32 warm10, no zero padding')
    obs,goal0=place(env,source_plan,root['slot'],meter,store)
    history=[observe(env,obs,'warmup',0,row,meter,store,trace)]
    for step,command in enumerate(moving.warmup_commands(),1):
        if body.eval_state(goal0,source.vector(obs['state']))[0]:raise allmode.NativeStop('warmup native terminal, no authenticated task root')
        obs,info,t,tr,reward,actual=dispatch(env,command,'warmup',step,row,meter,store,goal0)
        image=observe(env,obs,'warmup',step,row,meter,store,trace);paired.check_return(env,command,actual,info,goal0,reward,t)
        if t or tr:raise allmode.NativeStop('warmup native stop; task result unknown')
        if step in (5,10):history.append(image)
    require(moving.exact(np.asarray(history),packet['rgb_history']),'actual off history3')
    initial=source.vector(obs['state']);goal=source.vector(root['goal'])
    require(source.root_hash(initial,goal)==plan['request']['binding']['root_sha256']==root['root_sha256'],'original P3 native7 root/goal identity')
    bind_goal(body,goal,packet['goal_rgb'],meter,store);bound=invariant(body)
    row.update(initial_exact=True,root_state=initial.tolist(),root_sha256=root['root_sha256'],goal=goal.tolist(),
        goal_pose=bound['goal_pose'],goal_pixel_sha256=moving.image_sha(packet['goal_rgb']))
    observe(env,obs,'plan',0,row,meter,store,trace)
    if body.eval_state(goal,initial)[0]:
        calc=metrics(goal,initial);row.update(final_state=initial.tolist(),final_metrics=calc['native'],native_success=True,terminated=True,truncated=False)
        store.emit('task_terminal_at_root',**meter.where,state=initial.tolist(),metrics=calc['native'])
        raise allmode.NativeStop('terminated at authenticated task root')
    for step,command in enumerate(commands(plan['commands']),1):
        obs,info,t,tr,reward,actual=dispatch(env,command,'plan',step,row,meter,store,goal)
        observe(env,obs,'plan',step,row,meter,store,trace);success=paired.check_return(env,command,actual,info,goal,reward,t)
        require(moving.exact(np.asarray(info['goal']),packet['goal_rgb']) and
            {k:v for k,v in invariant(body).items() if k!='snapshot'}=={k:v for k,v in bound.items() if k!='snapshot'},'original true goalRGB/pose/renderer')
        state=source.vector(obs['state']);delta=state-initial;calc=metrics(goal,state)
        require(calc['native_success'] is bool(success) and reward==calc['native7_reward'],'native task success versus native7 reward')
        response=dict(step=step,state=state.tolist(),delta_native7=delta.tolist(),delta_angle_wrapped=float(np.arctan2(np.sin(delta[4]),np.cos(delta[4]))),
            reward=reward,native_success=bool(success),terminated=t,truncated=tr,native_metrics=calc['native'])
        row['responses'].append(response);row.update(final_state=state.tolist(),final_metrics=calc['native'],native_success=bool(success),terminated=t,truncated=tr)
        store.emit('own_response',**meter.where,response=response)
        if t or tr:row['status']='native_stopped';return
    row['status']='five_steps_completed'


def physical_receipt(row):
    if not row['initial_exact']:return None
    n=row['own_returned'];trans=row['actual_returns'];failure=row['status']=='engineering_failed'
    if len(trans)!=n or any(t['actual_dtype']!=np.dtype(np.float32).str or
        t['actual_bits']!=t['requested_bits'] or t['requested_bits']!=commands(row['commands'])[i].tobytes().hex()
        for i,t in enumerate(trans)):return None
    stop='engineering_failure' if failure else ('terminated' if row['terminated'] else ('truncated' if row['truncated'] else 'completed'))
    return dict(request_sha256=row['selection_digest'],commands=deepcopy(row['commands'][:n]),
        states=[deepcopy(row['root_state'])]+[deepcopy(r['state']) for r in trans],returned_steps=n,stop_reason=stop,
        native_success=row['native_success'] if n==len(row['responses']) else None,error=None if not failure else str(row['first_error']))


def execute(p,data,store,factory,*,authorization,prepared,prepare_sha256,started=None):
    require(authorization==AUTH,'MAIN explicit physical authority');actual_hash(prepare_sha256);validate_execution(p)
    require(p['source_slots']==[r['slot'] for r in data['roots']] and p['source_roots']==allmode.root_metadata(data) and
        p['warm_traces']==[r['warm_trace'] for r in data['roots']],'actual prepared physical source/context')
    meter=Meter(store,started);errors=[];seen=[]
    binding=dict(prepared=absolute(str(prepared)),prepare_sha256=prepare_sha256,producer_report_sha256=p['producer_report_sha256'],source_report_sha256=SOURCE_SHA)
    rows=[dict(index=i,slot=plan['slot'],mode=plan['mode'],root_slot=plan['slot'],condition='off',arm=plan['mode'],task_id=plan['task_id'],
        selection_digest=plan['selection_digest'],request=deepcopy(plan['request']),commands=deepcopy(plan['commands']),status='not_started',
        first_error=None,secondary_errors=[],warmup_attempted=0,warmup_returned=0,own_attempted=0,own_returned=0,PNG=0,
        initial_exact=False,trajectory=[],checkpoints=[],missing_checkpoints=[0,5],last_observation=None,responses=[],actual_returns=[],
        final_state=None,final_metrics=None,native_success=None,terminated=None,truncated=None,prefix51=dict(complete=False,frames=[]),
        source_unpaired_on_steps=[],physical_receipt=None) for i,plan in enumerate(p['plans'])]
    def fail(stage,exc,row=None):
        item=base.error(stage,exc);errors.append(item)
        if row is not None:
            if row['first_error'] is None:row['first_error']=deepcopy(item)
            else:row['secondary_errors'].append(deepcopy(item))
            row['status']='engineering_failed'
    try:
        store.put('execution_binding.json',binding);store.emit('run_started',prepare_sha256=prepare_sha256,contract=contract())
        for row,plan in zip(rows,p['plans'],strict=True):
            meter.where=dict(branch=row['index'],**{k:row[k] for k in ('mode','root_slot','condition','arm','selection_digest')})
            row['status']='running';env=None
            try:
                env=meter.call('constructor',factory)
                require(all(env is not x and env.unwrapped is not x.unwrapped for x in seen),'fresh env per fixed task/method');seen.append(env)
                validate_env(env.unwrapped)
                require(env.unwrapped.render_action is False and env.unwrapped.render_size==224 and env.unwrapped.render_mode=='rgb_array','original native renderer')
                with native_hooks(env,meter,row['secondary_errors']):episode(env,data['roots'][plan['slot']],data['source_plan'],plan,row,meter,store)
                require(not row['secondary_errors'],'native hook restoration')
            except allmode.NativeStop as exc:row.update(status='native_stopped',stop_reason=str(exc))
            except BaseException as exc:fail('branch',exc,row)
            finally:
                if env is not None:
                    try:meter.call('close',env.close,cleanup=True)
                    except BaseException as exc:fail('close',exc,row)
                if row['secondary_errors'] and row['first_error'] is None:fail('cleanup',ValueError('see secondary cleanup errors'),row)
                for phase,label in (('warmup','warmup'),('plan','own')):
                    ops=[x for x in meter.receipts if x['resource']=='public_step' and x['branch']==row['index'] and x['phase']==phase]
                    row[label+'_attempted']=len(ops);row[label+'_returned']=sum(x['status']=='returned' for x in ops)
            row['physical_receipt']=physical_receipt(row);store.emit('branch_closed',receipt=row)
    except BaseException as exc:fail('execution',exc)
    source_ok=False
    try:
        for f,h in p['bindings'].items():require(sha(f)==h,'final full input/source seals')
        source_ok=True
    except BaseException as exc:fail('final_inputs',exc)
    errors.extend(meter.secondary)
    if time.monotonic()-meter.started>=WALL:fail('final_wall',ValueError('900s wall cap'))
    good=not errors and source_ok and all(r['status'] in ('five_steps_completed','native_stopped') for r in rows)
    status=RECORDED if good else 'partial_or_failed'
    try:store.emit('run_finished',status=status,counts=meter.counts,errors=errors)
    except BaseException as exc:fail('final_journal',exc);status='partial_or_failed'
    report=dict(schema=NAME+'.report',contract=contract(),status=status,rows=rows,errors=errors,counts=deepcopy(meter.counts),
        actual_operation_receipts=deepcopy(meter.receipts),execution_binding=binding,inputs_final_verified=source_ok,
        fixed_denominator=4,known_roots=2,fresh_roots=0,new_neural_calls=0,new_independent_samples=0,
        equal_requested_execution_horizon=True,equal_actual_physical_execution_horizon=actual_equal(rows),
        all_requested_steps_completed=all(r['status']=='five_steps_completed' and r['own_returned']==5 for r in rows),
        total_internal_physics_scope='metered_native_calls' if meter.counts['constructor']['attempted']==meter.counts['constructor']['completed'] else 'unknown_constructor_work',
        journal_sha256=None,execution_binding_sha256=None,journal_chain_tail=store.previous,wall_seconds=time.monotonic()-meter.started,**FLAGS)
    for name,key in (('journal.jsonl','journal_sha256'),('execution_binding.json','execution_binding_sha256')):
        try:report[key]=sha(store.path/name)
        except BaseException as exc:fail('final_'+key,exc);report['status']='partial_or_failed'
    try:store.put('report.json',report,final=True)
    except BaseException as exc:fail('final_report',exc);report['status']='partial_or_failed'
    return report


def actual_equal(rows):
    if any(r['physical_receipt'] is None or r['status'] not in ('five_steps_completed','native_stopped') for r in rows):return None
    return len({r['physical_receipt']['returned_steps'] for r in rows})==1


def validate_saved(report,events,p):
    """Own schema only. Rebuild actual dispatches, source bridge and task labels."""
    validate_execution(p)
    require(report['schema']==NAME+'.report' and report['contract']==contract() and
        report['status'] in (RECORDED,'partial_or_failed') and report['fixed_denominator']==len(report['rows'])==4 and
        report['known_roots']==2 and report['fresh_roots']==0 and
        report['new_neural_calls']==report['new_independent_samples']==0 and all(report[k] is False for k in FLAGS),
        'own full4 schema and closed scientific gates')
    binding=report['execution_binding']
    require(binding['producer_report_sha256']==p['producer_report_sha256'] and binding['source_report_sha256']==SOURCE_SHA,
        'actual producer and original physical source ancestry')
    require(events and events[0]['kind']=='run_started' and events[0]['contract']==contract() and
        events[0]['prepare_sha256']==binding['prepare_sha256'] and events[-1]['kind']=='run_finished' and
        all(events[-1][k]==report[k] for k in ('status','counts','errors')),'durable run boundaries')
    tail=None;intents={};terminals={};stack=[]
    allowed={'run_started','run_finished','operation_dispatch_intent','operation_returned','operation_failed',
        'placement_recipe','returned_transition','fresh_observation','invalid_observation','goal_attached',
        'own_response','task_terminal_at_root','branch_closed'}
    for i,e in enumerate(events):
        require(e['kind'] in allowed and type(e['index']) is int and e['index']==i and e['previous_sha256']==tail and
            e['event_sha256']==source.digest({k:v for k,v in e.items() if k!='event_sha256'}),'journal chain/type')
        tail=e['event_sha256']
        if 'branch' in e:
            b=e['branch'];require(type(b) is int and 0<=b<4 and all(e[k]==report['rows'][b][k]
                for k in ('mode','root_slot','condition','arm','selection_digest')),'event exact branch identity')
        if e['kind'] not in ('operation_dispatch_intent','operation_returned','operation_failed'):continue
        op=e['receipt'];n=op['index'];b=op['branch']
        require(type(n) is int and type(b) is int and 0<=b<4 and op['resource'] in contract()['caps'] and
            all(op[k]==report['rows'][b][k] for k in ('mode','root_slot','condition','arm','selection_digest')),'operation scope')
        if e['kind']=='operation_dispatch_intent':
            require(n==len(intents) and op['status']=='pending','ordered actual intent');intents[n]=op;stack.append(n)
        else:
            require(n in intents and n not in terminals and stack and stack.pop()==n and
                op['status']==('returned' if e['kind']=='operation_returned' else 'failed') and
                dict({k:v for k,v in op.items() if k!='error'},status='pending')==intents[n],'nested actual callback receipt')
            terminals[n]=op
    require(not stack and len(intents)==len(terminals) and tail==report['journal_chain_tail'] and
        report['actual_operation_receipts']==[terminals[i] for i in range(len(terminals))],'complete operation closure')
    counts={k:dict(attempted=0,completed=0) for k in contract()['caps']}
    for op in report['actual_operation_receipts']:
        counts[op['resource']]['attempted']+=1;counts[op['resource']]['completed']+=op['status']=='returned'
    require(counts==report['counts'] and all(type(report['counts'][k][q]) is int for k in counts for q in ('attempted','completed')) and
        all(v['attempted']<=contract()['caps'][k] for k,v in counts.items()),'actual counts and caps')
    require([e['receipt'] for e in events if e['kind']=='branch_closed']==[r for r in report['rows'] if r['status']!='not_started'],
        'fixed4 branch closure including failures')
    for i,row in enumerate(report['rows']):
        plan=p['plans'][i];slot=plan['slot'];task=p['source_roots'][slot];scoped=[e for e in events if e.get('branch')==i]
        require(type(row['index']) is type(row['slot']) is type(row['root_slot']) is int and row['index']==i and
            all(row[k]==plan[k] for k in ('slot','mode','root_slot','condition','arm','task_id','selection_digest','commands','request')) and
            row['status'] in ('five_steps_completed','native_stopped','engineering_failed','not_started'),'all4 exact original requests')
        ops=[op for op in report['actual_operation_receipts'] if op['branch']==i and op['resource']=='public_step']
        trans=[e for e in scoped if e['kind']=='returned_transition']
        resets=[op for op in report['actual_operation_receipts'] if op['branch']==i and op['resource']=='reset']
        recipe=[e for e in scoped if e['kind']=='placement_recipe']
        require(len(resets)<=2 and all(op['role']==('seed','placement')[j] and
            op['seed']==(p['source_slots'][slot]['reset_seed'] if j==0 else 0) for j,op in enumerate(resets)),
            'original reset seed then literal placement reset, never task reset padding')
        require(len(recipe)<=1 and (not recipe or recipe[0]['slot']==p['source_slots'][slot]),'original literal seed/placement recipe')
        for phase,label,actions in (('warmup','warmup',moving.warmup_commands()),('plan','own',commands(plan['commands']))):
            current=[op for op in ops if op['phase']==phase];ts=[t for t in trans if t['phase']==phase]
            require(len(current)<=len(actions) and [op['step'] for op in current]==list(range(1,len(current)+1)) and
                all(op['raw_bits']==actions[j].tobytes().hex() for j,op in enumerate(current)) and
                row[label+'_attempted']==len(current) and row[label+'_returned']==sum(op['status']=='returned' for op in current),
                'exact contiguous actual raw actions, no H3 tail/padding')
            require([t['step'] for t in ts]==list(range(1,len(ts)+1)) and len(ts)<=row[label+'_returned'],'actual return chronology')
            for t in ts:
                require(t['requested_bits']==actions[t['step']-1].tobytes().hex(),'returned request bits')
                if row['status']!='engineering_failed':require(t['actual_bits']==t['requested_bits'] and
                    t['actual_dtype']==np.dtype(np.float32).str,'no raw substitution')
            require(all(not t['terminated'] and not t['truncated'] for t in ts[:-1]) and
                (not ts or not (ts[-1]['terminated'] or ts[-1]['truncated']) or len(current)==ts[-1]['step']),
                'no dispatch after any actual native stop, even before label validation')
        obs=[e for e in scoped if e['kind']=='fresh_observation'];frames=[{k:deepcopy(e[k]) for k in ('receipt','pre','post')} for e in obs]
        require(row['PNG']==len(obs) and row['prefix51']==dict(complete=len(frames)==51,frames=frames),'actual51 no padding')
        unknown=[]
        for e in obs:
            f=e['receipt'];require(f['phase'] in ('warmup','plan') and type(f['step']) is int and
                0<=f['step']<=(10 if f['phase']=='warmup' else 5) and type(f['overlay_index']) is int and
                f['overlay_index'] in (0,1,2) and f['condition']==('on' if f['overlay_index']==1 else 'off') and
                f['png']==f'b{i}_{f["phase"]}_{f["step"]:02}_{f["overlay_index"]}.png','frame horizon and overlays')
            require(e['pre']==e['post'] and f['snapshot']==e['post']['snapshot'] and f['state']==f['snapshot']['state'] and
                e['pre']['renderer']==task['renderer'] and e['pre']['goal_pose']==task['goal_pose'],'original renderer/body invariant')
            if f['step']:
                ts=[t for t in trans if t['phase']==f['phase'] and t['step']==f['step']]
                require(len(ts)==1 and f['state']==ts[0]['state'] and f['snapshot']==ts[0]['snapshot'],
                    'render is actual returned transition, not substituted old frame')
            elif f['phase']=='warmup':require(len(recipe)==1 and f['state']==recipe[0]['requested_state'],'actual placement state')
            else:require(f['state']==task['root_state'],'actual requested native root')
            if f['phase']=='warmup':
                require(len(recipe)==1 and e['pre']['goal_state']==recipe[0]['native_goal'],'actual original native warmup goal')
                ref=allmode.reference_for(p['warm_traces'][slot],f['step'],f['overlay_index'])
                if ref is not None:allmode.check_warm_frame(e,ref)
                else:unknown.append(f['step'])
            else:require(e['pre']['goal_state']==task['goal'] and e['pre']['goal_rgb_sha256']==task['goal_pixels']['off'],
                'actual original off goal, not constructed goal_pose')
        require(unknown==row['source_unpaired_on_steps'],'unavailable historical on frames remain unknown')
        for e in obs:
            f=e['receipt']
            if f['overlay_index']==2:
                first=[x['receipt'] for x in obs if x['receipt']['phase']==f['phase'] and x['receipt']['step']==f['step'] and
                    x['receipt']['overlay_index']==0]
                require(len(first)==1 and all(first[0][k]==f[k] for k in ('pixel_sha256','state','snapshot')),'actual off/on/off replay')
        selected=[e['receipt'] for e in obs if e['receipt']['overlay_index']==0 and
            any(x['receipt']['phase']==e['receipt']['phase'] and x['receipt']['step']==e['receipt']['step'] and
                x['receipt']['overlay_index']==2 for x in obs)]
        require(row['trajectory']==selected and row['last_observation']==(obs[-1]['receipt'] if obs else None),
            'actual selected off trajectory and last rendered identity')
        actual=[{k:deepcopy(v) for k,v in e.items() if k not in ('kind','index','previous_sha256','event_sha256')}
            for e in trans if e['phase']=='plan']
        require(actual==row['actual_returns'],'durable actual returns distinct from validated labels')
        responses=row['responses'];require([e['response'] for e in scoped if e['kind']=='own_response']==responses and
            [x['step'] for x in responses]==list(range(1,len(responses)+1)) and len(responses)<=len(actual),'accepted response chronology')
        if row['initial_exact']:
            require(all(row[k]==task[k] for k in ('root_state','root_sha256','goal','goal_pose')) and
                row['goal_pixel_sha256']==task['goal_pixels']['off'] and
                source.root_hash(source.vector(row['root_state']),source.vector(row['goal']))==plan['request']['binding']['root_sha256'],
                'native task root and model request source binding')
            attached=[e for e in scoped if e['kind']=='goal_attached']
            require(len(attached)==1 and attached[0]['goal']==row['goal'] and attached[0]['goal_pose']==row['goal_pose'] and
                attached[0]['goal_rgb_sha256']==row['goal_pixel_sha256'],'actual true goal attach')
            warmroot=[e['receipt'] for e in obs if e['receipt']['phase']=='warmup' and e['receipt']['step']==10]
            require(warmroot and attached[0]['snapshot']==warmroot[-1]['snapshot'],'goal attach preserves complete native body state')
            require(all(e['receipt']['snapshot']==attached[0]['snapshot'] for e in obs if
                e['receipt']['phase']=='plan' and e['receipt']['step']==0),'plan0 is original body after pure goal attach')
        for j,r in enumerate(responses):
            t=actual[j];calc=metrics(row['goal'],r['state']);delta=source.vector(r['state'])-source.vector(row['root_state'])
            require(r['native_metrics']==calc['native'] and r['reward']==calc['native7_reward'] and
                r['native_success'] is r['terminated'] is calc['native_success'] and r['delta_native7']==delta.tolist() and
                r['delta_angle_wrapped']==float(np.arctan2(np.sin(delta[4]),np.cos(delta[4]))) and
                all(r[k]==t[k] for k in ('step','state','reward','terminated','truncated')) and t['goal_state']==row['goal'] and
                t['actual_bits']==t['requested_bits'] and t['actual_dtype']==np.dtype(np.float32).str,'independent native metrics/return/raw')
        require(all(not r['terminated'] and not r['truncated'] for r in responses[:-1]),'no execution after native stop')
        if responses:
            last=responses[-1];require(row['final_state']==last['state'] and row['final_metrics']==last['native_metrics'] and
                all(row[k] is last[k] for k in ('native_success','terminated','truncated')),'last validated endpoint')
        elif row['final_state'] is not None:
            terminal=[e for e in scoped if e['kind']=='task_terminal_at_root'];calc=metrics(row['goal'],row['root_state'])
            require(row['initial_exact'] and len(terminal)==1 and terminal[0]['state']==row['root_state']==row['final_state'] and
                terminal[0]['metrics']==row['final_metrics']==calc['native'] and row['native_success'] is row['terminated'] is True and
                row['truncated'] is False and calc['native_success'] is True,'native terminal at bound root')
        else:require(row['final_metrics'] is row['native_success'] is row['terminated'] is row['truncated'] is None,'unknown never zero')
        cps=[]
        for e in obs:
            f=e['receipt']
            if f['phase']=='plan' and f['step'] in (0,5) and f['overlay_index']==0:
                cps.append({k:deepcopy(f[k]) for k in ('step','png','file_sha256','pixel_sha256','state','snapshot')})
        require(row['checkpoints']==cps and row['missing_checkpoints']==[n for n in (0,5) if n not in [c['step'] for c in cps]],'actual checkpoint identities')
        require(row['physical_receipt']==physical_receipt(row),'actual physical receipt, no invented missing return')
        if row['physical_receipt'] is not None:fixed.audit_physical_receipt(row['request'],row['physical_receipt'])
        if row['status'] in ('five_steps_completed','native_stopped'):
            require(row['first_error'] is None and row['secondary_errors']==[] and len(responses)==row['own_returned'],'clean engineering closure')
            expected=[('warmup',n,k) for n in range(row['warmup_returned']+1) for k in (0,1,2)]
            if row['initial_exact']:expected += [('plan',n,k) for n in range(row['own_returned']+1) for k in (0,1,2)]
            require([(e['receipt']['phase'],e['receipt']['step'],e['receipt']['overlay_index']) for e in obs]==expected,'complete actual frame chronology')
            if row['status']=='five_steps_completed':require(row['initial_exact'] and row['own_returned']==5 and
                not row['terminated'] and not row['truncated'],'five actual returns only')
            elif row['initial_exact']:require(row['terminated'] or row['truncated'],'native stop label')
    require(report['equal_requested_execution_horizon'] is True and
        report['equal_actual_physical_execution_horizon'] is actual_equal(report['rows']) and
        report['all_requested_steps_completed'] is all(r['status']=='five_steps_completed' and r['own_returned']==5 for r in report['rows']),
        'actual length summary distinct from equal requested length')
    scope='metered_native_calls' if counts['constructor']['attempted']==counts['constructor']['completed'] else 'unknown_constructor_work'
    require(report['total_internal_physics_scope']==scope,'unknown constructor internal work retained')
    if report['status']==RECORDED:
        require(not report['errors'] and report['inputs_final_verified'] is True and 0<=report['wall_seconds']<WALL and
            all(r['status'] in ('five_steps_completed','native_stopped') for r in report['rows']),'engineering complete not admission')
        c={k:v['completed'] for k,v in counts.items()}
        require(c['constructor']==c['close']==4 and c['reset']==8 and c['set_state']==2*c['reset'] and
            c['physics_step']==2*c['reset']+10*c['public_step'] and c['goal_render']==c['reset'] and
            c['goal_setter']==c['reset']+c['goal_bind'] and c['goal_bind']==sum(r['initial_exact'] for r in report['rows']) and
            c['public_step']==sum(r['warmup_returned']+r['own_returned'] for r in report['rows']) and
            c['explicit_render']==sum(r['PNG'] for r in report['rows']) and
            all(v['attempted']==v['completed'] for v in counts.values()),'nested native census, not added axes')
    return deepcopy(report['rows'])


def consume(output,expected):
    actual_hash(expected);directory=Path(absolute(str(output)));files=saved_artifacts(directory)
    require(hashlib.sha256(files['report.json']).hexdigest()==expected,'external actual physical report SHA')
    r=decode(files['report.json']);binding=r['execution_binding']
    require(hashlib.sha256(files['execution_binding.json']).hexdigest()==r['execution_binding_sha256'] and
        decode(files['execution_binding.json'])==binding,'actual external execution binding')
    p,_=verify(binding['prepared'],binding['prepare_sha256']);body=files['journal.jsonl']
    require(body.endswith(b'\n') and hashlib.sha256(body).hexdigest()==r['journal_sha256'],'complete actual journal SHA')
    events=[decode(line) for line in body.splitlines()];rows=validate_saved(r,events,p)
    names={'report.json','journal.jsonl','execution_binding.json'}
    from PIL import Image
    for e in events:
        if e['kind']=='invalid_observation':
            name=e['file'];require(Path(name).name==name and name not in ('','.','..') and
                hashlib.sha256(files[name]).hexdigest()==e['file_sha256'],'sealed invalid incident');names.add(name)
        if e['kind']!='fresh_observation':continue
        f=e['receipt'];name=f['png'];require(Path(name).name==name and name not in ('','.','..') and name not in names and
            len(files[name])<=1024**2 and hashlib.sha256(files[name]).hexdigest()==f['file_sha256'],'actual unique flat PNG SHA');names.add(name)
        with Image.open(io.BytesIO(files[name])) as image:
            require(image.mode=='RGB' and image.size==(224,224) and moving.image_sha(np.asarray(image))==f['pixel_sha256'],'actual RGB224 pixels')
    require(set(files)==names,'no unaccounted artifacts')
    bindings=deepcopy(p['bindings']);merge(bindings,{binding['prepared']:binding['prepare_sha256']})
    merge(bindings,{str(directory/name):hashlib.sha256(value).hexdigest() for name,value in files.items()})
    for path,h in p['bindings'].items():require(sha(path)==h,'final full source/input closure')
    require(saved_artifacts(directory)==files,'final actual artifact drift')
    outcomes=[]
    for row in rows:
        valid=row['initial_exact'] and row['own_returned']==len(row['responses']) and row['physical_receipt'] is not None
        state=row['final_state'] if valid else None
        outcomes.append(dict(slot=row['slot'],mode=row['mode'],returned_steps=row['own_returned'],
            validated_steps=len(row['responses']),metrics=None if state is None else metrics(row['goal'],state),
            contract_audit=None if row['physical_receipt'] is None else fixed.audit_physical_receipt(row['request'],row['physical_receipt'])))
    return dict(schema=NAME+'.consume',status=r['status'],rows=rows,outcomes=outcomes,fixed_denominator=4,known_roots=2,fresh_roots=0,
        errors=deepcopy(r['errors']),counts=r['counts'],bindings=bindings,producer_consume=deepcopy(p['producer_consume']),
        physical_report_sha256=expected,producer_report_sha256=p['producer_report_sha256'],artifact_authenticity_verified=True,
        equal_requested_execution_horizon=True,equal_actual_physical_execution_horizon=r['equal_actual_physical_execution_horizon'],
        all_requested_steps_completed=r['all_requested_steps_completed'],new_neural_calls=0,new_physics_calls=0,**FLAGS)


def run(prepared,expected,output,factory,*,authorization=None):
    require(authorization==AUTH,'MAIN authority BEFORE reads/output/env')
    p,data=verify(prepared,expected);store=Store(output)
    return execute(p,data,store,factory,authorization=authorization,prepared=prepared,prepare_sha256=expected)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('plan','prepare','verify','run','consume'))
    for name in ('output','prepared','prepare-sha','report-sha','producer-output','producer-report-sha','producer-source-sha','producer-prepared','producer-prepare-sha'):
        parser.add_argument('--'+name)
    parser.add_argument('--authorization');a=parser.parse_args(argv)
    if a.command=='plan':r=dict(contract=contract(),supported_producer_source_sha256=SUPPORTED_SOURCE_SHA,
        actual_producer_report_sha256=CUDA_REPORT_SHA,actual_producer_prepare_sha256=CUDA_PREPARE_SHA,
        status='actual_terminal_bound_MAIN_strict_consume_required',physics_run=False)
    elif a.command=='prepare':r=prepare(a.output,dict(producer_output=a.producer_output,producer_report_sha256=a.producer_report_sha,
        producer_source_sha256=a.producer_source_sha,producer_prepared=a.producer_prepared,producer_prepare_sha256=a.producer_prepare_sha))
    elif a.command=='verify':
        p,_=verify(a.prepared,a.prepare_sha);r=dict(status='verified_no_execution',fixed_denominator=4,**FLAGS)
    elif a.command=='consume':r=consume(a.output,a.report_sha)
    else:
        require(a.authorization==AUTH,'MAIN CLI explicit physical authorization BEFORE native import')
        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401
            return gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224,with_target=False)
        r=run(a.prepared,a.prepare_sha,a.output,factory,authorization=a.authorization)
        path=Path(absolute(a.output))/'report.json'
        r=dict(schema=NAME+'.run_summary',status=r['status'],report=str(path),report_sha256=sha(path) if path.is_file() else None,
            fixed_denominator=r['fixed_denominator'],errors=r['errors'],counts=r['counts'],wall_seconds=r['wall_seconds'],
            wall_scope=contract()['wall_scope'],all_requested_steps_completed=r['all_requested_steps_completed'],**FLAGS)
    print(source.canonical(r));return 2 if r.get('status')=='partial_or_failed' else 0


if __name__=='__main__':raise SystemExit(main())
