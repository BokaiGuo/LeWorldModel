"""Mixed222 fixed4 native adapter. Only MAIN authorizes physics, never neural.

No old schema projection, globals patch, outcome/root selection or partial release.
One branch is either an entire strictly inherited old row or an entire new run.
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

from experiments.spectral_reachability import agent_allmode_physical_20261028_v1 as allmode

NAME='agent_mixedmode_physical_20261028_v1'
PRODUCER='agent_mixedmode_prefix_20261028_v1'
OLD='agent_allmode_physical_20261028_v1'
ARM='mixedmode_full10_conditional_D2P'
AUTH='MAIN_MIXEDMODE4_RAW10_PHYSICS_ONLY'
RECORDED='mixedmode4_physical_recorded_not_admitted'
PRODUCER_SHA='ec912669c678960aac0d004a2e764b4bf036194673da5d1c63d0b535bc199396'
REPORT_SHA='d6166b98958af70cecb8529135396c77467b7c803bfe12a1a5de29c422c22ad2'
PREP_SHA='8f6d027116b76334e80f9a761688ffd33825bc25d825484012d37230f083ed6b'
OLD_SHA='c0842f63685718e3d1d562fbbe5e2e3fdf9103954804f4276a1186d8db91efe4'
OLD_REPORT_SHA='3ca4891a5f0823bede431e85442513a095b1e9bc8d49c87feefbb56663dca3b4'
ROOT,ORDER,SOURCE_SHA=allmode.ROOT,allmode.ORDER,allmode.SOURCE_SHA
PREPARED=ROOT/'results/agent_mixedmode_prefix_20261028_v1_prepare_MAIN01.json'
OLD_OUTPUT=ROOT/'results/agent_allmode_physical_20261028_v1_MAIN01'
CAP,FLOOR,FINAL,WALL=allmode.CAP,allmode.FLOOR,allmode.FINAL,allmode.WALL
PREP_CAP=64*1024**2  # Complete mixed artifact hash closure, never strip bindings.
ARGUMENTS=allmode.ARGUMENTS|{'old_output','old_report_sha256','old_source_sha256'}
require,sha,raw,read=allmode.require,allmode.sha,allmode.raw,allmode.read
absolute,actual_hash,merge,decode=allmode.absolute,allmode.actual_hash,allmode.merge,allmode.decode
core,base,moving,source,paired=allmode.core,allmode.base,allmode.moving,allmode.source,allmode.paired
raw10=allmode.raw10
place,bind_goal,invariant=allmode.place,allmode.bind_goal,allmode.invariant
observe,dispatch=allmode.observe,allmode.dispatch
native_hooks,validate_env=allmode.native_hooks,allmode.validate_env
Meter=allmode.Meter  # Same physical caps/wall; no schema checks inside Meter.
consume_old,source_material,saved_artifacts=allmode.consume,allmode.source_material,allmode.saved_artifacts


def contract():
    c=deepcopy(allmode.contract())
    c.update(schema=NAME+'.contract',arm=ARM,producer=PRODUCER,
        inheritance='whole old allmode row only after both full consumes and exact raw/task/context/source match',
        native_success='joint ||state[:4]-goal[:4]||<20 AND min(abs(angle),2pi-abs(angle))<pi/9',
        native_reward='negative Euclidean native7 distance INCLUDING velocities, NOT native success threshold',
        prepare_cap_bytes=PREP_CAP,no_partial_trajectory_splicing=True,freshness=False)
    return c


def metrics(goal,state):
    g,s=source.vector(goal),source.vector(state);require(g.shape==s.shape==(7,),'true native7 metrics')
    native=base.previous.metric(goal,state);delta=s-g
    return dict(native=native,native_success=native['native_success'],joint_xy_error=float(np.linalg.norm(delta[:4])),
        angle_native_term=native['native_angle_term'],angle_wrapped_error=float(abs(np.arctan2(np.sin(delta[4]),np.cos(delta[4])))),
        velocity_error=float(np.linalg.norm(delta[5:])),native7_distance=float(np.linalg.norm(delta)),
        native7_reward=-float(np.linalg.norm(delta)))


def validate_plans(report,compiled,verified,data,expected):
    actual_hash(expected)
    require(report['schema']==PRODUCER+'.report' and report['status']=='all4_mixedmode_plans_sealed_not_admitted' and
        report['errors']==[] and report['first_error'] is None and
        report['commands_released'] is report['model_unchanged'] is report['source_verified'] is True and
        report['P1_admission'] is report['P3_admission'] is False,'complete mixed222 terminal release only')
    require(compiled['schema']==PRODUCER+'.prepare' and compiled['source_report_sha256']==SOURCE_SHA and
        len(compiled['slots'])==len(report['rows'])==4,'same four original encoded tasks')
    require(verified['schema']==PRODUCER+'.consume' and verified['status']=='all4_mixedmode_commands_verified_not_admitted' and
        verified['errors']==[] and len(verified['plans'])==4 and
        all(verified[k] is False for k in ('P1_admission','P3_admission','equal_total_compute')) and
        verified['new_neural_calls']==verified['new_physics_calls']==0,'complete actual producer.consume')
    require(report['resources']=={k:dict(attempted=5352,completed=5352) for k in ('predict','action_encoder','pred_proj')},
        'all5352 actual forwards on each nested module axis')
    for i,(slot,row,plan) in enumerate(zip(compiled['slots'],report['rows'],verified['plans'],strict=True)):
        key=ORDER[i];export=data['exports'][key]
        require(type(slot['index']) is type(row['index']) is type(plan['root_slot']) is type(plan['source_slot']) is int and
            slot['index']==row['index']==i and (slot['root_slot'],slot['condition'])==key and
            (row['root_slot'],row['condition'])==key and (plan['root_slot'],plan['condition'])==key and
            plan['source_slot']==key[0] and plan['arm']==ARM and row['status']=='plan_sealed','literal fixed4 mixed keys/order')
        require(plan['schema']==PRODUCER+'.model_plan' and plan['source_report_sha256']==SOURCE_SHA and
            all(plan[k]==slot[k] for k in ('task_id','blind_sha256','task_sha256')) and
            plan['blind_sha256']==export['sha256'],'exact bound original condition task/blind identity')
        body={k:v for k,v in plan.items() if k not in ('selection_digest','whole_run_report_sha256')}
        require(plan['selection_digest']==core.digest(body) and plan['whole_run_report_sha256']==expected and
            row['plan']=={k:v for k,v in plan.items() if k!='whole_run_report_sha256'},'actual complete mixed selection/dispatch seal')
        require(type(plan['selected_index']) is int and 0<=plan['selected_index']<222 and plan['horizon']==2 and plan['raw_steps']==10 and
            all(plan[k] is False for k in ('selection_uses_physics','P1_admission','P3_admission','equal_total_compute')),'frozen222 model-only selection')
        a=raw10(plan['commands']);b=raw10(plan['prefix_commands']+plan['suffix_commands'])
        require(len(plan['prefix_commands'])==len(plan['suffix_commands'])==5 and a.tobytes()==b.tobytes(),'literal raw10, no regenerated tanh')
    return deepcopy(verified['plans'])


def routes(plans,compiled,old,oldcompiled,receipt,new_source):
    require(len(plans)==len(compiled['slots'])==len(old['plans'])==len(oldcompiled['slots'])==len(receipt['rows'])==4,
        'full four-task inheritance comparison denominator')
    result=[]
    for i,(slot,condition) in enumerate(ORDER):
        new,prior=plans[i],old['plans'][i]
        require((new['root_slot'],new['condition'])==(prior['root_slot'],prior['condition'])==(slot,condition),
            'exact old/new four-task order, never join by outcomes')
        same_task=(old['source_report_sha256']==SOURCE_SHA and
            all(new[k]==prior[k] for k in ('task_id','blind_sha256','task_sha256','source_report_sha256')) and
            raw(compiled['slots'][i])==raw(oldcompiled['slots'][i]))
        same_source=all(old[k][slot]==new_source[k][slot] for k in ('source_slots','source_roots','warm_traces'))
        same_raw=raw10(new['commands']).tobytes()==raw10(prior['commands']).tobytes()
        complete=receipt['rows'][i]['status'] in ('horizon_completed','native_stopped')
        inherit=same_task and same_source and same_raw and complete
        result.append(dict(index=i,old_index=i,mode='inherit' if inherit else 'execute',
            reason='exact_task_context_source_raw10_and_complete_old_consume' if inherit else 'requires_whole_new_branch_no_splicing'))
    return result


def own_files():
    return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def material(arguments):
    require(type(arguments) is dict and set(arguments)==ARGUMENTS,'all actual producer/old physical external authorities')
    for k in ('producer_report_sha256','producer_source_sha256','producer_prepare_sha256','old_report_sha256','old_source_sha256'):actual_hash(arguments[k])
    code=ROOT/'experiments/spectral_reachability'/(PRODUCER+'.py')
    output=Path(absolute(arguments['producer_output']));rp=output/'report.json'
    pp=Path(absolute(arguments['producer_prepared']));oldout=Path(absolute(arguments['old_output']))
    require(arguments['producer_source_sha256']==PRODUCER_SHA==sha(code) and
        arguments['old_source_sha256']==OLD_SHA==sha(allmode.__file__),'frozen actual producer and public adapter source BEFORE consume')
    require(pp==PREPARED and arguments['producer_prepare_sha256']==PREP_SHA==sha(pp) and
        oldout==OLD_OUTPUT and arguments['old_report_sha256']==OLD_REPORT_SHA==sha(oldout/'report.json') and
        sha(rp)==arguments['producer_report_sha256']==REPORT_SHA,'actual mixed prepare/report and fixed allmode MAIN01, no future SHA')
    report=read(rp,1024**3);compiled=read(pp,2*1024**2)
    require(report['schema']==PRODUCER+'.report' and report['status']=='all4_mixedmode_plans_sealed_not_admitted' and
        report['errors']==[] and report['prepared']==str(pp) and report['prepare_sha256']==PREP_SHA,'terminal full4 mixed, never running/partial')
    producer=importlib.import_module('experiments.spectral_reachability.'+PRODUCER)
    require(Path(producer.__file__).resolve()==code and report['contract']==compiled['contract']==producer.contract(),'real mixed producer ABI')
    verified=producer.consume(output,arguments['producer_report_sha256'])
    data=source_material();plans=validate_plans(report,compiled,verified,data,arguments['producer_report_sha256'])
    oldreceipt=consume_old(oldout,arguments['old_report_sha256'])
    require(oldreceipt['schema']==OLD+'.consume' and oldreceipt['status']==allmode.RECORDED and oldreceipt['errors']==[] and
        oldreceipt['fixed_denominator']==len(oldreceipt['rows'])==4 and oldreceipt['inputs_final_verified'] is True,
        'complete strict old allmode physical MAIN01 consume, not raw-equality certification')
    oldreport=read(oldout/'report.json',CAP);binding=oldreport['execution_binding'];oldpp=absolute(binding['prepared'])
    require(oldreceipt['physical_report_sha256']==arguments['old_report_sha256'] and oldreceipt['rows']==oldreport['rows'] and
        sha(oldpp)==binding['prepare_sha256']==oldreceipt['bindings'][oldpp],'actual old report/prepare/full rows')
    oldprepared=read(oldpp,allmode.PREPARE_CAP)
    oldcompiler=oldprepared['arguments']['producer_prepared']
    require(sha(oldcompiler)==oldprepared['arguments']['producer_prepare_sha256']==oldreceipt['bindings'][oldcompiler],'old encoded task context closure')
    oldcompiled=read(oldcompiler,2*1024**2)
    newroots=allmode.root_metadata(data)
    newmeta=dict(source_slots=[deepcopy(r['slot']) for r in data['roots']],
        source_roots=newroots,warm_traces=[deepcopy(r['warm_trace']) for r in data['roots']])
    route=routes(plans,compiled,oldprepared,oldcompiled,oldreceipt,newmeta)
    bindings={};merge(bindings,compiled['bindings']);merge(bindings,data['bindings']);merge(bindings,oldreceipt['bindings'])
    merge(bindings,{str(code):PRODUCER_SHA,str(pp):PREP_SHA,str(rp):arguments['producer_report_sha256'],str(Path(allmode.__file__).resolve()):OLD_SHA})
    for slot in compiled['slots']:merge(bindings,{absolute(slot['task_file']):slot['task_sha256']})
    for f in output.iterdir():
        require(f.is_file() and not f.is_symlink(),'regular complete mixed artifacts');merge(bindings,{str(f):sha(f)})
    for f in own_files():merge(bindings,{str(f):sha(f)})
    for f,h in bindings.items():require(sha(f)==h,'complete final input hash closure')
    p=dict(schema=NAME+'.prepare',contract=contract(),arguments=deepcopy(arguments),plans=plans,routes=route,
        producer_consume=verified,old_physical_consume=oldreceipt,bindings=bindings,source_report_sha256=SOURCE_SHA,
        producer_report_sha256=arguments['producer_report_sha256'],producer_prepare_sha256=PREP_SHA,old_report_sha256=OLD_REPORT_SHA,
        source_slots=[deepcopy(r['slot']) for r in data['roots']],warm_traces=[deepcopy(r['warm_trace']) for r in data['roots']],source_roots=newroots)
    require(len(raw(p))<=PREP_CAP,'bounded complete metadata, never truncate input hashes');return p,data


def write_new(path,value,cap):
    path=Path(absolute(str(path)));body=raw(value)
    require(path.name.startswith(NAME+'_') and not path.exists() and not path.is_symlink() and len(body)<=cap and
        shutil.disk_usage(path.parent).free>=FLOOR+len(body),'NEW own bounded metadata +200MiB floor')
    with path.open('xb') as stream:
        require(stream.write(body)==len(body),'complete write');stream.flush();os.fsync(stream.fileno())
    paired.fsync_dir(path.parent);return dict(output=str(path),sha256=sha(path),bytes=len(body))


def prepare(output,arguments):
    path=Path(absolute(output));require(not path.exists() and not path.is_symlink(),'NEW prepare BEFORE consuming')
    p,_=material(arguments);return write_new(path,p,PREP_CAP)


def verify(prepared,expected):
    actual_hash(expected);absolute(str(prepared));require(sha(prepared)==expected,'actual own prepare external SHA')
    p=read(prepared,PREP_CAP);now,data=material(p['arguments'])
    require(p==now and sha(prepared)==expected,'complete new+old consumes and own prepare reconstruction');return p,data


class Store(allmode.Store):
    def __init__(self,path):
        self.path=Path(absolute(str(path)))
        require(self.path.name.startswith(NAME+'_') and not self.path.exists() and not self.path.is_symlink() and
            self.path.parent.is_dir() and shutil.disk_usage(self.path.parent).free>=FLOOR+CAP,'NEW own output32MiB +200MiB floor')
        self.path.mkdir();paired.fsync_dir(self.path.parent);self.previous=None;self.index=0


def validate_execution(p):
    require(p['schema']==NAME+'.prepare' and p['contract']==contract() and p['source_report_sha256']==SOURCE_SHA and
        len(p['plans'])==len(p['routes'])==4 and len(p['source_roots'])==len(p['source_slots'])==len(p['warm_traces'])==2 and
        p['plans']==p['producer_consume']['plans'],'own mixed schema/full4 prepare, not old schema projection')
    for i,plan in enumerate(p['plans']):
        require(type(plan['root_slot']) is int and (plan['root_slot'],plan['condition'])==ORDER[i] and plan['arm']==ARM and
            plan['whole_run_report_sha256']==p['producer_report_sha256'] and p['routes'][i]['index']==p['routes'][i]['old_index']==i and
            p['routes'][i]['mode'] in ('inherit','execute'),'fixed4 plan/route identity');raw10(plan['commands'])


def episode(env,root,source_plan,plan,row,meter,store):
    body=env.unwrapped;packet=root['packets'][plan['condition']];trace=root['warm_trace']
    a=np.asarray(packet['raw_action_history'])
    require(a.shape==(10,2) and a.dtype==np.float32 and a.tobytes()==moving.warmup_commands().tobytes(),'original fixed nonzero warmup float32 raw10')
    obs,goal0=place(env,source_plan,root['slot'],meter,store)
    history=[observe(env,obs,'warmup',0,row,meter,store,trace)]
    for step,command in enumerate(moving.warmup_commands(),1):
        if body.eval_state(goal0,source.vector(obs['state']))[0]:raise allmode.NativeStop('native warmup terminal')
        obs,info,t,tr,reward,actual=dispatch(env,command,'warmup',step,row,meter,store,goal0)
        image=observe(env,obs,'warmup',step,row,meter,store,trace);paired.check_return(env,command,actual,info,goal0,reward,t)
        if t or tr:raise allmode.NativeStop('warmup native stop, no constructed root padding')
        if step in (5,10):history.append(image)
    require(moving.exact(np.asarray(history),packet['rgb_history']),'same-condition history3 identity')
    state=source.vector(obs['state']);goal=source.vector(root['goal'])
    require(source.root_hash(state,goal)==root['root_sha256'],'original native7 root/true goal')
    bind_goal(body,goal,packet['goal_rgb'],meter,store);bound=invariant(body)
    row.update(initial_exact=True,root_state=state.tolist(),root_sha256=root['root_sha256'],goal=goal.tolist(),
        goal_pose=bound['goal_pose'],goal_pixel_sha256=moving.image_sha(packet['goal_rgb']))
    observe(env,obs,'plan',0,row,meter,store,trace)
    if body.eval_state(goal,state)[0]:raise allmode.NativeStop('native task terminal at original root')
    for step,command in enumerate(raw10(plan['commands']),1):
        obs,info,t,tr,reward,actual=dispatch(env,command,'plan',step,row,meter,store,goal)
        observe(env,obs,'plan',step,row,meter,store,trace);success=paired.check_return(env,command,actual,info,goal,reward,t)
        require(moving.exact(np.asarray(info['goal']),packet['goal_rgb']) and
            {k:v for k,v in invariant(body).items() if k!='snapshot'}=={k:v for k,v in bound.items() if k!='snapshot'},'original goalRGB/pose/renderer unchanged')
        s=source.vector(obs['state']);delta=s-state;calc=metrics(goal,s)
        require(calc['native_success'] is bool(success) and reward==calc['native7_reward'],'native jointXY/angle success versus seven-dimensional reward')
        response=dict(step=step,state=s.tolist(),delta_native7=delta.tolist(),delta_angle_wrapped=float(np.arctan2(np.sin(delta[4]),np.cos(delta[4]))),
            reward=reward,native_success=bool(success),terminated=t,truncated=tr,native_metrics=calc['native'])
        row['responses'].append(response);row.update(final_state=s.tolist(),final_metrics=calc['native'],native_success=bool(success),terminated=t,truncated=tr)
        store.emit('own_response',**meter.where,response=response)
        if t or tr:row['status']='native_stopped';return
    row['status']='horizon_completed'


def execute(p,data,store,factory,*,authorization,prepared,prepare_sha256,started=None):
    require(authorization==AUTH,'MAIN authorization before execution');actual_hash(prepare_sha256);validate_execution(p)
    require(p['source_slots']==[r['slot'] for r in data['roots']] and p['source_roots']==allmode.root_metadata(data) and
        p['warm_traces']==[r['warm_trace'] for r in data['roots']],'actual prepared source task/warmup identities')
    meter=Meter(store,started);errors=[];seen=[]
    binding=dict(prepared=absolute(str(prepared)),prepare_sha256=prepare_sha256,producer_report_sha256=p['producer_report_sha256'],
        old_report_sha256=p['old_report_sha256'],source_report_sha256=SOURCE_SHA)
    rows=[dict(index=i,root_slot=k[0],condition=k[1],arm=ARM,task_id=plan['task_id'],selection_digest=plan['selection_digest'],
        plan=deepcopy(plan),route=p['routes'][i]['mode'],status='not_started',first_error=None,secondary_errors=[],
        warmup_attempted=0,warmup_returned=0,own_attempted=0,own_returned=0,PNG=0,initial_exact=False,trajectory=[],
        checkpoints=[],missing_checkpoints=[0,5,10],last_observation=None,responses=[],final_state=None,final_metrics=None,
        native_success=None,terminated=None,truncated=None,prefix51=dict(complete=False,frames=[]),source_unpaired_on_steps=[],
        inherited_evidence=None,evidence_origin=None) for i,(k,plan) in enumerate(zip(ORDER,p['plans'],strict=True))]
    def fail(stage,exc,row=None):
        item=base.error(stage,exc);errors.append(item)
        if row is not None:
            if row['first_error'] is None:row['first_error']=deepcopy(item)
            else:row['secondary_errors'].append(deepcopy(item))
            row['status']='engineering_failed'
    try:
        store.put('execution_binding.json',binding);store.emit('run_started',prepare_sha256=prepare_sha256,contract=contract())
        for row,plan in zip(rows,p['plans'],strict=True):
            meter.where=dict(branch=row['index'],**{k:row[k] for k in ('root_slot','condition','arm','selection_digest')})
            if row['route']=='inherit':
                prior=deepcopy(p['old_physical_consume']['rows'][row['index']])
                row.update(status='inherited_saved_physical',inherited_evidence=prior,evidence_origin='allmode_MAIN01_historical_not_fresh')
                for k in ('final_state','final_metrics','native_success','terminated','truncated'):row[k]=deepcopy(prior[k])
                store.emit('branch_closed',receipt=row);continue
            env=None;row['status']='running';row['evidence_origin']='new_native_execution_known_root'
            try:
                env=meter.call('constructor',factory)
                require(all(env is not x and env.unwrapped is not x.unwrapped for x in seen),'fresh native env per executed branch');seen.append(env)
                validate_env(env.unwrapped)
                require(env.unwrapped.render_action is False and env.unwrapped.render_size==224 and env.unwrapped.render_mode=='rgb_array','original renderer/action semantics')
                with native_hooks(env,meter,row['secondary_errors']):episode(env,data['roots'][plan['root_slot']],data['source_plan'],plan,row,meter,store)
                require(not row['secondary_errors'],'native hooks restore')
            except allmode.NativeStop as exc:row.update(status='native_stopped',stop_reason=str(exc))
            except BaseException as exc:fail('branch',exc,row)
            finally:
                if env is not None:
                    try:meter.call('close',env.close,cleanup=True)
                    except BaseException as exc:fail('close',exc,row)
                if row['secondary_errors'] and row['first_error'] is None:fail('cleanup',ValueError('secondary native cleanup error'),row)
                for phase,label in (('warmup','warmup'),('plan','own')):
                    ops=[x for x in meter.receipts if x['resource']=='public_step' and x.get('branch')==row['index'] and x['phase']==phase]
                    row[label+'_attempted']=len(ops);row[label+'_returned']=sum(x['status']=='returned' for x in ops)
            store.emit('branch_closed',receipt=row)
    except BaseException as exc:fail('execution',exc)
    source_ok=False
    try:
        for f,h in p['bindings'].items():require(sha(f)==h,'final full source/input seals')
        source_ok=True
    except BaseException as exc:fail('final_inputs',exc)
    errors.extend(meter.secondary)
    if time.monotonic()-meter.started>=WALL:fail('final_wall',ValueError('900s wall'))
    good=not errors and source_ok and all(r['status'] in ('horizon_completed','native_stopped','inherited_saved_physical') for r in rows)
    status=RECORDED if good else 'partial_or_failed'
    try:store.emit('run_finished',status=status,counts=meter.counts,errors=errors)
    except BaseException as exc:fail('final_journal',exc);status='partial_or_failed'
    report=dict(schema=NAME+'.report',status=status,contract=contract(),execution_binding=binding,rows=rows,errors=errors,
        counts=deepcopy(meter.counts),actual_operation_receipts=deepcopy(meter.receipts),fixed_denominator=4,known_roots=2,fresh_roots=0,
        inherited=sum(r['status']=='inherited_saved_physical' for r in rows),inputs_final_verified=source_ok,
        new_neural_calls=0,new_independent_samples=0,P1_admission=False,P3_admission=False,scientific_admission=False,
        freshness=False,equal_total_compute=False,equal_FLOPs=False,training=False,independent_confirmation=False,
        total_internal_physics_scope='metered_native_calls' if meter.counts['constructor']['attempted']==meter.counts['constructor']['completed'] else 'unknown_constructor_work',
        journal_sha256=None,execution_binding_sha256=None,journal_chain_tail=store.previous,wall_seconds=time.monotonic()-meter.started)
    for name,key in (('journal.jsonl','journal_sha256'),('execution_binding.json','execution_binding_sha256')):
        try:report[key]=sha(store.path/name)
        except BaseException as exc:fail('final_'+key,exc);report['status']='partial_or_failed'
    try:store.put('report.json',report,final=True)
    except BaseException as exc:fail('final_report',exc);report['status']='partial_or_failed'
    return report


def run(prepared,expected,output,factory,*,authorization=None):
    require(authorization==AUTH,'explicit MAIN authority BEFORE reads/output/env')
    p,data=verify(prepared,expected);store=Store(output)
    return execute(p,data,store,factory,authorization=authorization,prepared=prepared,prepare_sha256=expected)


def validate_saved(r,events,p):
    """Own mixed schema, not an old-schema view of a new scientific arm."""
    validate_execution(p)
    require(r['schema']==NAME+'.report' and r['status'] in (RECORDED,'partial_or_failed') and r['contract']==contract() and
        r['fixed_denominator']==len(r['rows'])==4 and r['known_roots']==2 and r['fresh_roots']==0 and
        r['execution_binding']['producer_report_sha256']==p['producer_report_sha256'] and
        r['execution_binding']['old_report_sha256']==p['old_report_sha256'] and r['execution_binding']['source_report_sha256']==SOURCE_SHA,
        'fixed4 own mixed report/source/new+old actual report seals')
    require(all(r[k] is False for k in ('P1_admission','P3_admission','scientific_admission','freshness','equal_total_compute',
        'equal_FLOPs','training','independent_confirmation')) and r['new_neural_calls']==r['new_independent_samples']==0,'no scientific/freshness upgrade')
    require(events and events[0]['kind']=='run_started' and events[0]['contract']==contract() and
        events[0]['prepare_sha256']==r['execution_binding']['prepare_sha256'] and events[-1]['kind']=='run_finished' and
        events[-1]['status']==r['status'] and events[-1]['counts']==r['counts'] and events[-1]['errors']==r['errors'],'durable mixed begin/end')
    tail=None;intents={};terminals={};stack=[]
    for i,e in enumerate(events):
        require(e['index']==i and e['previous_sha256']==tail and e['event_sha256']==source.digest({k:v for k,v in e.items() if k!='event_sha256'}),'actual journal hashchain')
        tail=e['event_sha256']
        if e['kind'] not in ('operation_dispatch_intent','operation_returned','operation_failed'):continue
        op=e['receipt'];b=op['branch'];n=op['index']
        require(type(b) is type(n) is int and 0<=b<4 and op['resource'] in contract()['caps'] and
            p['routes'][b]['mode']=='execute' and all(op[k]==r['rows'][b][k] for k in ('root_slot','condition','arm','selection_digest')),
            'actual operations only on authorized new branches with exact identity')
        if e['kind']=='operation_dispatch_intent':
            require(n==len(intents) and op['status']=='pending','ordered operation intent');intents[n]=op;stack.append(n)
        else:
            require(n in intents and n not in terminals and stack and stack.pop()==n and
                op['status']==('returned' if e['kind']=='operation_returned' else 'failed') and
                dict({k:v for k,v in op.items() if k!='error'},status='pending')==intents[n],'nested exact actual return/failure')
            terminals[n]=op
    require(not stack and len(intents)==len(terminals) and tail==r['journal_chain_tail'] and
        r['actual_operation_receipts']==[terminals[i] for i in range(len(terminals))],'complete operation closure, no refund')
    counts={k:dict(attempted=0,completed=0) for k in contract()['caps']}
    for op in r['actual_operation_receipts']:
        counts[op['resource']]['attempted']+=1;counts[op['resource']]['completed']+=op['status']=='returned'
    require(counts==r['counts'] and all(v['attempted']<=contract()['caps'][k] for k,v in counts.items()),'real counts/caps, nested axes not added')
    require([e['receipt'] for e in events if e['kind']=='branch_closed']==[row for row in r['rows'] if row['status']!='not_started'],'full4 failure closure')
    for i,row in enumerate(r['rows']):
        plan=p['plans'][i];slot=plan['root_slot'];task=p['source_roots'][slot]
        require(type(row['index']) is type(row['root_slot']) is int and row['index']==i and (row['root_slot'],row['condition'])==ORDER[i] and
            row['arm']==ARM and row['plan']==plan and row['route']==p['routes'][i]['mode'] and
            all(row[k]==plan[k] for k in ('task_id','selection_digest')),'fixed4 mixed row/route/task/selection')
        scoped=[e for e in events if e.get('branch')==i]
        require(all(all(e[k]==row[k] for k in ('root_slot','condition','arm','selection_digest')) for e in scoped),'event branch identity')
        if row['route']=='inherit' and row['status']=='inherited_saved_physical':
            prior=p['old_physical_consume']['rows'][i]
            require(row['inherited_evidence']==prior and row['evidence_origin']=='allmode_MAIN01_historical_not_fresh' and
                all(row[k]==prior[k] for k in ('final_state','final_metrics','native_success','terminated','truncated')) and
                not scoped and all(row[k]==0 for k in ('warmup_attempted','warmup_returned','own_attempted','own_returned','PNG')) and
                row['trajectory']==row['responses']==row['checkpoints']==[] and row['prefix51']==dict(complete=False,frames=[]) and
                row['first_error'] is None and row['secondary_errors']==[],'entire inherited old row only, no new trajectory splicing/charge')
            continue
        require(row['status'] in ('not_started','engineering_failed','native_stopped','horizon_completed') and
            row['inherited_evidence'] is None,'new-only trajectory or explicit failure/unknown')
        ops=[op for op in r['actual_operation_receipts'] if op['branch']==i and op['resource']=='public_step']
        for phase,label,commands in (('warmup','warmup',moving.warmup_commands()),('plan','own',raw10(plan['commands']))):
            current=[op for op in ops if op['phase']==phase]
            require([op['step'] for op in current]==list(range(1,len(current)+1)) and len(current)<=10 and
                all(op['raw_bits']==commands[j].tobytes().hex() for j,op in enumerate(current)) and
                row[label+'_attempted']==len(current) and row[label+'_returned']==sum(op['status']=='returned' for op in current),'literal contiguous raw authority and actual attempts/returns')
        observations=[e for e in scoped if e['kind']=='fresh_observation'];trans=[e for e in scoped if e['kind']=='returned_transition']
        require(row['PNG']==len(observations),'actual PNG count')
        bridge=[{k:deepcopy(e[k]) for k in ('receipt','pre','post')} for e in observations if e['receipt']['phase']=='warmup' or e['receipt']['step']<=5]
        require(row['prefix51']==dict(complete=len(bridge)==51,frames=bridge),'real prefix51 only, no inherited/new concatenation')
        unknown=[]
        for e in observations:
            f=e['receipt'];require(e['pre']==e['post'] and f['snapshot']==e['post']['snapshot'] and f['state']==f['snapshot']['state'] and
                e['pre']['renderer']==task['renderer'] and e['pre']['goal_pose']==task['goal_pose'],'bound native render/pose/body invariants')
            if f['phase']=='warmup':
                ref=allmode.reference_for(p['warm_traces'][slot],f['step'],f['overlay_index'])
                if ref is not None:allmode.check_warm_frame(e,ref)
                else:unknown.append(f['step'])
            else:
                require(f['phase']=='plan' and e['pre']['goal_state']==task['goal'] and
                    e['pre']['goal_rgb_sha256']==task['goal_pixels'][row['condition']], 'source condition-specific actual goalRGB, not default pose')
        require(row['source_unpaired_on_steps']==unknown,'missing historical intermediate on frames remain unknown')
        for phase,label in (('warmup','warmup'),('plan','own')):
            current=[t for t in trans if t['phase']==phase]
            require([t['step'] for t in current]==list(range(1,len(current)+1)) and len(current)<=row[label+'_returned'] and
                all(t['actual_dtype']==np.dtype(np.float32).str and t['actual_bits']==t['requested_bits'] for t in current),'exact returned raw, no substitution')
        if row['initial_exact']:
            require(all(row[k]==task[k] for k in ('root_state','root_sha256','goal','goal_pose')) and
                row['goal_pixel_sha256']==task['goal_pixels'][row['condition']] and
                source.root_hash(source.vector(row['root_state']),source.vector(row['goal']))==row['root_sha256'],'original native7 root/task SHA')
            attached=[e for e in scoped if e['kind']=='goal_attached']
            require(len(attached)==1 and attached[0]['goal']==row['goal'] and attached[0]['goal_pose']==row['goal_pose'] and
                attached[0]['goal_rgb_sha256']==row['goal_pixel_sha256'],'actual task attachment identity')
        responses=row['responses']
        require([x['step'] for x in responses]==list(range(1,len(responses)+1)) and len(responses)<=row['own_returned'] and
            [e['response'] for e in scoped if e['kind']=='own_response']==responses,'durable exact response chronology')
        for response in responses:
            calc=metrics(row['goal'],response['state']);delta=source.vector(response['state'])-source.vector(row['root_state'])
            require(response['native_metrics']==calc['native'] and response['reward']==calc['native7_reward'] and
                response['native_success'] is response['terminated'] is calc['native_success'] and type(response['truncated']) is bool and
                np.array_equal(delta,np.asarray(response['delta_native7'])) and
                response['delta_angle_wrapped']==float(np.arctan2(np.sin(delta[4]),np.cos(delta[4]))),'independent actual native7 reward and jointXY/angle success/delta')
            t=next(e for e in trans if e['phase']=='plan' and e['step']==response['step'])
            require(t['state']==response['state'] and t['goal_state']==row['goal'] and t['reward']==response['reward'] and
                t['terminated'] is response['terminated'] and t['truncated'] is response['truncated'],'actual returned transition identity')
        if responses:
            last=responses[-1];require(row['final_state']==last['state'] and row['final_metrics']==last['native_metrics'] and
                all(row[k] is last[k] for k in ('native_success','terminated','truncated')) and
                all(not x['terminated'] and not x['truncated'] for x in responses[:-1]),'actual endpoint/no post-native-stop calls')
        else:require(row['final_state'] is row['final_metrics'] is row['native_success'] is None,'missing task outcome unknown, not zero')
        require(len({cp['step'] for cp in row['checkpoints']})==len(row['checkpoints']),'unique actual checkpoints')
        for cp in row['checkpoints']:
            found=[e['receipt'] for e in observations if e['receipt']['phase']=='plan' and e['receipt']['step']==cp['step'] and
                e['receipt']['overlay_index']==(1 if row['condition']=='on' else 0)]
            require(cp['step'] in (0,5,10) and len(found)==1 and
                all(cp[k]==found[0][k] for k in ('step','png','file_sha256','pixel_sha256','state','snapshot')),'checkpoint from actual selected-condition frame')
        if row['status'] in ('horizon_completed','native_stopped'):
            require(row['first_error'] is None and row['secondary_errors']==[],'no engineering errors in complete row')
            if responses:require(row['warmup_returned']==10 and row['initial_exact'] and row['own_returned']==len(responses) and
                ((row['status']=='native_stopped')==(responses[-1]['terminated'] or responses[-1]['truncated'])),'native earlystop no padding')
            if row['status']=='horizon_completed':require(row['own_returned']==10,'actual complete raw10')
            expected=[n for n in (0,5,10) if row['initial_exact'] and n<=row['own_returned']]
            require([cp['step'] for cp in row['checkpoints']]==expected and row['missing_checkpoints']==[n for n in (0,5,10) if n not in expected],'only reached checkpoints')
            framekeys=[('warmup',n,k) for n in range(row['warmup_returned']+1) for k in (0,1,2)]
            if row['initial_exact']:framekeys += [('plan',n,k) for n in range(row['own_returned']+1) for k in (0,1,2)]
            require([(e['receipt']['phase'],e['receipt']['step'],e['receipt']['overlay_index']) for e in observations]==framekeys,'complete unpadded actual triplets')
    require(r['inherited']==sum(x['status']=='inherited_saved_physical' for x in r['rows']),'inherited denominator summary')
    scope='metered_native_calls' if counts['constructor']['attempted']==counts['constructor']['completed'] else 'unknown_constructor_work'
    require(r['total_internal_physics_scope']==scope,'failed partial constructor internal work unknown')
    if r['status']==RECORDED:
        require(r['errors']==[] and r['inputs_final_verified'] is True and r['wall_seconds']<WALL and
            all(x['status'] in ('inherited_saved_physical','horizon_completed','native_stopped') for x in r['rows']),'complete engineering, never science pass')
        c={k:v['completed'] for k,v in counts.items()};n=4-r['inherited']
        require(all(v['attempted']==v['completed'] for v in counts.values()) and c['constructor']==c['close']==n and c['reset']==2*n and
            c['goal_bind']==sum(x['initial_exact'] for x in r['rows'] if x['route']=='execute') and
            c['physics_step']==2*c['reset']+10*c['public_step'] and c['set_state']==2*c['reset'] and
            c['goal_render']==c['reset'] and c['goal_setter']==c['reset']+c['goal_bind'] and
            c['public_step']==sum(x['warmup_returned']+x['own_returned'] for x in r['rows'] if x['route']=='execute') and
            c['explicit_render']==sum(x['PNG'] for x in r['rows']),'actual new-only census, old resources not charged again')
    return deepcopy(r['rows'])


def consume(output,expected):
    actual_hash(expected);directory=Path(absolute(str(output)));files=saved_artifacts(directory)
    require(hashlib.sha256(files['report.json']).hexdigest()==expected,'external actual mixed physical report SHA')
    r=decode(files['report.json']);binding=r['execution_binding']
    require(hashlib.sha256(files['execution_binding.json']).hexdigest()==r['execution_binding_sha256'] and
        decode(files['execution_binding.json'])==binding,'actual execution binding')
    p,_=verify(binding['prepared'],binding['prepare_sha256']);content=files['journal.jsonl']
    require(content.endswith(b'\n') and hashlib.sha256(content).hexdigest()==r['journal_sha256'],'complete actual journal SHA')
    events=[decode(line) for line in content.splitlines()];rows=validate_saved(r,events,p)
    names={'report.json','execution_binding.json','journal.jsonl'}
    from PIL import Image
    for e in events:
        if e['kind']=='invalid_observation':
            name=e['file'];require(Path(name).name==name and name not in ('','.','..') and
                hashlib.sha256(files[name]).hexdigest()==e['file_sha256'],'sealed invalid observation incident');names.add(name)
        if e['kind']!='fresh_observation':continue
        f=e['receipt'];name=f['png'];require(Path(name).name==name and name not in ('','.','..') and len(files[name])<=1024**2 and
            hashlib.sha256(files[name]).hexdigest()==f['file_sha256'],'actual flat PNG bytes SHA');names.add(name)
        with Image.open(io.BytesIO(files[name])) as image:
            require(image.mode=='RGB' and image.size==(224,224) and moving.image_sha(np.asarray(image))==f['pixel_sha256'],'actual native RGB224 pixels')
    require(set(files)==names,'no unaccounted output artifacts')
    bindings=deepcopy(p['bindings']);merge(bindings,{binding['prepared']:binding['prepare_sha256']})
    merge(bindings,{str(directory/name):hashlib.sha256(body).hexdigest() for name,body in files.items()})
    for f,h in p['bindings'].items():require(sha(f)==h,'final input seals')
    require(saved_artifacts(directory)==files,'final saved artifact drift')
    breakdown=[]
    for row in rows:
        evidence=row['inherited_evidence'] if row['status']=='inherited_saved_physical' else row
        value=None if evidence['final_state'] is None else metrics(evidence['goal'],evidence['final_state'])
        breakdown.append(dict(root_slot=row['root_slot'],condition=row['condition'],origin=row['evidence_origin'],metrics=value))
    return dict(schema=NAME+'.consume',status=r['status'],rows=rows,metric_breakdown=breakdown,counts=r['counts'],
        errors=r['errors'],fixed_denominator=4,known_roots=2,fresh_roots=0,inherited=r['inherited'],bindings=bindings,
        physical_report_sha256=expected,producer_report_sha256=p['producer_report_sha256'],old_report_sha256=p['old_report_sha256'],
        P1_admission=False,P3_admission=False,scientific_admission=False,freshness=False,equal_total_compute=False,equal_FLOPs=False,
        independent_confirmation=False,new_neural_calls=0,new_physics_calls=0)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('output','prepared','prepare-sha','report-sha','producer-output','producer-report-sha','producer-source-sha',
              'producer-prepared','producer-prepare-sha','old-output','old-report-sha','old-source-sha'):parser.add_argument('--'+k)
    parser.add_argument('--authorization');a=parser.parse_args(argv)
    if a.command=='prepare':
        cli=dict(producer_output=a.producer_output,producer_report_sha256=a.producer_report_sha,producer_source_sha256=a.producer_source_sha,
            producer_prepared=a.producer_prepared,producer_prepare_sha256=a.producer_prepare_sha,old_output=a.old_output,
            old_report_sha256=a.old_report_sha,old_source_sha256=a.old_source_sha)
        result=prepare(a.output,cli)
    elif a.command=='verify':
        p,_=verify(a.prepared,a.prepare_sha);result=dict(status='verified_no_execution',routes=p['routes'],fixed_denominator=4,P1_admission=False)
    elif a.command=='consume':result=consume(a.output,a.report_sha)
    else:
        require(a.authorization==AUTH,'MAIN CLI authorization before importing native environment')
        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401
            return gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224,with_target=False)
        result=run(a.prepared,a.prepare_sha,a.output,factory,authorization=a.authorization)
    print(source.canonical(result));return 2 if result.get('status')=='partial_or_failed' else 0


if __name__=='__main__':raise SystemExit(main())
