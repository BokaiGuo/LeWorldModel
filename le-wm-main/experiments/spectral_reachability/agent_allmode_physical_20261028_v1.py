"""MAIN-only four known-root allmode raw10 physical plans, own schema/kernel.

Only complete producer.consume releases commands. No model or old executor call,
global patching, outcome/action-change filtering, padding, or P1 admission.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from experiments.spectral_reachability import p1_moving_overlay_recompile_physical_20261027_v1 as base
from experiments.spectral_reachability import p1_conditional_task_admission_20261027_v1 as core

NAME='agent_allmode_physical_20261028_v1'
# Only this actual module/schema is accepted. MAIN supplies its final source,
# prepare and report seals explicitly; none has a future/default report SHA.
PRODUCER='agent_allmode_prefix_20261028_v2'
ROOT=base.ROOT
PRODUCER_SHA='2cc7201377c24533aaa8f4c2c968ae7959a3c55df97b40964116e26a944380f5'
PRODUCER_PREP=ROOT/'results/agent_allmode_prefix_20261028_v2_prepare_MAIN01.json'
PRODUCER_PREP_SHA='c11543eaa251903ff6d0429d2447f03c7427e1fe65be3d0aabfb0d9c4ef79d0d'
BASE_SHA='d6ee57d4d902c2d8c865c8372718a3bbbea3964a23973da535f106e415229570'
OLD_PHYSICAL='capability_suffix_factorial_physical_20261027_v1'
OLD_PHYSICAL_SHA='15125f4d86e5a6b3344f750205101a0e8a6f4a8dfba82ce46ce3d925f4ea8bd6'
ARM='allmode_full10_conditional_D2P'
ORDER=[(s,c) for s in (0,1) for c in ('off','on')]
AUTH='MAIN_ALLMODE4_RAW10_PHYSICS_ONLY'
RECORDED='allmode4_physical_recorded_not_admitted'
CAP,FLOOR,FINAL,PREPARE_CAP,WALL=32*1024**2,200*1024**2,4*1024**2,16*1024**2,900
ARGUMENTS={'producer_output','producer_report_sha256','producer_source_sha256','producer_prepared','producer_prepare_sha256'}
moving,source,paired=base.moving,base.source,base.paired
require,sha,raw=base.require,base.sha,base.raw
place,bind_goal,invariant=base.place,base.bind_goal,base.invariant
native_hooks,validate_env=base.native_hooks,base.validate_env
SOURCE_SHA=base.SOURCE_SHA
FRAME_FIELDS=('phase','step','condition','overlay_index','file_sha256','pixel_sha256','state','snapshot')


def contract():
    return dict(fixed_denominator=4,order=[list(k) for k in ORDER],arm=ARM,known_roots=2,fresh_roots=0,
        warmup=10,horizon=10,checkpoints=[0,5,10],state_dimension=7,flags=[False,True,False],
        control_with_target=False,no_goal_pose_edit=True,action_scale=100,relative=True,
        raw_authority='complete actual producer.consume literal float32 raw10, no tanh/clip/scale',
        prefix51='fresh warmup0..10 + own allmode plan0..5, each off/on/off; incomplete when early stopped',
        source_matching='original saved warmup off0..10 and on0/5/10; intermediate on identities unknown',
        wall_seconds=WALL,output_cap_bytes=CAP,final_reserve_bytes=FINAL,disk_floor_bytes=FLOOR,
        caps=dict(constructor=4,close=4,reset=8,public_step=80,physics_step=816,set_state=16,
                  goal_setter=12,goal_render=8,goal_bind=4,explicit_render=264),
        count_scope='durable dispatch intent then actual callback entry/return; nested resources NOT added',
        producer=PRODUCER,no_neural=True,training=False,Qfit=False,P1_admission=False,P3_admission=False,
        scientific_admission=False,independent_confirmation=False,equal_total_compute=False,equal_FLOPs=False,
        interpretation='fixed4 root-condition known-root calibration, not four fresh independent roots or physical equivalence')


def actual_hash(h):
    base.hash_value(h);require(h!='0'*64,'actual nonzero SHA required, no future seal')


def absolute(value):
    require(type(value) is str and Path(value).is_absolute() and str(Path(value).resolve())==value,'canonical absolute path')
    return value


def merge(bindings,more):
    for path,h in more.items():
        absolute(path);actual_hash(h)
        require(path not in bindings or bindings[path]==h,'nonconflicting complete input seals')
        bindings[path]=h


def read(path,cap=PREPARE_CAP):
    p=Path(path);require(p.is_file() and not p.is_symlink() and p.stat().st_size<=cap,'regular bounded metadata')
    body=p.read_bytes();require(len(body)<=cap,'bounded metadata bytes');return decode(body)


def decode(body):
    def pairs(items):
        out={}
        for k,v in items:require(k not in out,'duplicate JSON key');out[k]=v
        return out
    def constant(_):raise ValueError('nonfinite JSON')
    value=json.loads(body,object_pairs_hook=pairs,parse_constant=constant);raw(value);return value


def raw10(value):
    require(type(value) is list and len(value)==10 and all(type(row) is list and len(row)==2 and
        all(type(x) in (int,float) for x in row) for row in value),'literal raw10 numeric values, not bool')
    return base.previous.commands(value)


def validate_verified(report,compiled,consumed,data,report_sha):
    """Pure ABI seam; production material ALWAYS calls complete real consume first."""
    actual_hash(report_sha)
    require(report['schema']==PRODUCER+'.report' and report['status']=='all4_allmode_plans_sealed_not_admitted' and
        report['errors']==[] and report['first_error'] is None and
        report['commands_released'] is report['model_unchanged'] is report['source_verified'] is True and
        report['P1_admission'] is report['P3_admission'] is False,'no partial producer release, including completed3/4')
    require(compiled['schema']==PRODUCER+'.prepare' and compiled['source_report_sha256']==SOURCE_SHA and
        len(compiled['slots'])==len(report['rows'])==4,'fixed four actual source tasks')
    require(consumed['schema']==PRODUCER+'.consume' and consumed['status']=='all4_allmode_commands_verified_not_admitted' and
        consumed['errors']==[] and len(consumed['plans'])==4 and
        consumed['P1_admission'] is consumed['P3_admission'] is consumed['equal_total_compute'] is False and
        consumed['new_neural_calls']==consumed['new_physics_calls']==0,'complete producer own-consume, not prepare/test')
    require(report['resources']=={k:dict(attempted=1032,completed=1032) for k in ('predict','action_encoder','pred_proj')},'full actual1032 resource axes')
    for i,(slot,row,plan) in enumerate(zip(compiled['slots'],report['rows'],consumed['plans'],strict=True)):
        key=ORDER[i];export=data['exports'][key]
        require(type(slot['index']) is type(row['index']) is type(plan['root_slot']) is type(plan['source_slot']) is int and
            slot['index']==row['index']==i and (slot['root_slot'],slot['condition'])==key and
            (row['root_slot'],row['condition'])==key and (plan['root_slot'],plan['condition'])==key and
            plan['source_slot']==key[0] and plan['arm']==ARM and row['status']=='plan_sealed','fixed literal4 keys/order/new arm')
        require(plan['schema']==PRODUCER+'.model_plan' and plan['source_report_sha256']==SOURCE_SHA and
            all(plan[k]==slot[k] for k in ('task_id','blind_sha256','task_sha256')) and
            plan['blind_sha256']==export['sha256'] and Path(export['path']).is_absolute(),'same original paired source/task/blind condition')
        body={k:v for k,v in plan.items() if k not in ('selection_digest','whole_run_report_sha256')}
        require(plan['selection_digest']==core.digest(body) and plan['whole_run_report_sha256']==report_sha and
            row['plan']=={k:v for k,v in plan.items() if k!='whole_run_report_sha256'},'complete own selection/dispatch seal')
        require(plan['horizon']==2 and plan['raw_steps']==10 and type(plan['selected_index']) is int and 0<=plan['selected_index']<42 and
            all(plan[k] is False for k in ('selection_uses_physics','P1_admission','P3_admission','equal_total_compute')),'unchanged fixed42 model-only objective')
        actions=raw10(plan['commands']);split=raw10(plan['prefix_commands']+plan['suffix_commands'])
        require(len(plan['prefix_commands'])==len(plan['suffix_commands'])==5 and actions.tobytes()==split.tobytes(),'literal selected raw10 bytes, not regenerated params')
    return deepcopy(consumed['plans'])


def source_material():
    """Read old saved source only. Source has17 warmup frames, NOT33 old matches."""
    data=base.source_material();report=base.read(base.SOURCE_REPORT)
    events=base.previous.journal(base.SOURCE_REPORT.parent/'journal.jsonl',report['journal_chain_tail'])
    for slot,root in enumerate(data['roots']):
        found=[e['receipt'] for e in events if e['kind']=='fresh_overlay_frame' and
               e.get('key')==f's{slot}_replay0' and e['receipt']['reference_key'] in {str(n) for n in range(11)}]
        keys=[(int(f['reference_key']),f['overlay_index']) for f in found]
        expected=[(n,k) for n in range(11) for k in ((0,1,2) if n in (0,5,10) else (0,))]
        require(keys==expected,'original17 saved warmup off/on receipts, not invented missing on frames')
        trace=[]
        for f in found:
            require(f['pre']==f['post'] and f['with_target'] is (f['overlay_index']==1),'source warmup render identity')
            path=str(paired.flat(base.SOURCE_REPORT.parent,f['png']).resolve())
            require(sha(path)==f['file_sha256'],'actual source warmup file SHA');merge(data['bindings'],{path:f['file_sha256']})
            trace.append(dict(receipt=dict(phase='warmup',step=int(f['reference_key']),condition='on' if f['with_target'] else 'off',
                overlay_index=f['overlay_index'],file_sha256=f['file_sha256'],pixel_sha256=f['pixel_sha256'],
                state=deepcopy(f['post']['snapshot']['state']),snapshot=deepcopy(f['post']['snapshot'])),
                pre=deepcopy(f['pre']),post=deepcopy(f['post'])))
        root['warm_trace']=trace
    return data


def root_metadata(data):
    """Task authority comes from bound source packets, never a physical row."""
    return [dict(root_slot=i,root_state=deepcopy(root['warm_trace'][-1]['receipt']['state']),
        root_sha256=root['root_sha256'],goal=source.vector(root['goal']).tolist(),
        goal_pose=deepcopy(root['warm_trace'][0]['pre']['goal_pose']),
        renderer=deepcopy(root['warm_trace'][0]['pre']['renderer']),
        goal_pixels={c:moving.image_sha(root['packets'][c]['goal_rgb']) for c in ('off','on')})
        for i,root in enumerate(data['roots'])]


def own_files():
    return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md'),
            Path(core.__file__).resolve(),*base.own_files()]


def constructor_contract():
    path=Path(moving.ENV)
    require(sha(path)==moving.ENV_SHA,'frozen native source before any constructor')
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PushT')
    init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    require(not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in
        {'reset','step','render','_set_state','_set_goal_state','_setup'} for n in ast.walk(init)),
        'constructor inert before hooks; partial constructor failure work still unknown')
    return str(path.resolve()),moving.ENV_SHA


def material(arguments):
    require(type(arguments) is dict and set(arguments)==ARGUMENTS,'all external producer authorities required')
    for k in ('producer_report_sha256','producer_source_sha256','producer_prepare_sha256'):actual_hash(arguments[k])
    path=ROOT/'experiments/spectral_reachability'/(PRODUCER+'.py')
    require(arguments['producer_source_sha256']==PRODUCER_SHA==sha(path) and sha(base.__file__)==BASE_SHA,'external actual v2 producer/helper SHA BEFORE import')
    output=Path(absolute(arguments['producer_output']));rp=output/'report.json'
    pp=Path(absolute(arguments['producer_prepared']));prepare_sha=arguments['producer_prepare_sha256']
    require(pp==PRODUCER_PREP and prepare_sha==PRODUCER_PREP_SHA,'only MAIN actual fixed v2 prepare, never failed v1')
    require(sha(rp)==arguments['producer_report_sha256'] and sha(pp)==prepare_sha,'external actual producer report/prepare SHA')
    report=read(rp,512*1024**2);compiled=read(pp,2*1024**2)
    require(report['schema']==PRODUCER+'.report' and report['status']=='all4_allmode_plans_sealed_not_admitted' and
        report['errors']==[] and report['commands_released'] is True,'terminal complete v2 only BEFORE consume import')
    require(report['prepared']==str(pp) and report['prepare_sha256']==prepare_sha,'exact MAIN fixed4 v2 prepare')
    producer=importlib.import_module('experiments.spectral_reachability.'+PRODUCER)
    require(Path(producer.__file__).resolve()==path and compiled['contract']==report['contract']==producer.contract(),'actual producer module/contract')
    consumed=producer.consume(output,arguments['producer_report_sha256'])
    data=source_material();plans=validate_verified(report,compiled,consumed,data,arguments['producer_report_sha256'])
    bindings={};merge(bindings,compiled['bindings']);merge(bindings,data['bindings'])
    native,native_sha=constructor_contract();merge(bindings,{native:native_sha})
    merge(bindings,{str(path):arguments['producer_source_sha256'],str(rp):arguments['producer_report_sha256'],str(pp):prepare_sha})
    # The producer verifies this inherited prepare; retain its complete input seals.
    parent=ROOT/'results/p1_moving_overlay_recompile_cuda_20261027_v1_prepare_MAIN01.json'
    inherited=read(parent,2*1024**2);require(sha(parent)==bindings[str(parent)],'bound inherited original task prepare')
    merge(bindings,inherited['bindings'])
    for slot in compiled['slots']:
        merge(bindings,{absolute(slot['task_file']):slot['task_sha256']})
    names=list(output.iterdir());require(all(f.is_file() and not f.is_symlink() for f in names),'regular complete producer artifacts')
    for f in names:merge(bindings,{str(f):sha(f)})
    for f in own_files():merge(bindings,{str(f.resolve()):sha(f)})
    for f,h in bindings.items():require(sha(f)==h,'final complete source/task/producer seal')
    p=dict(schema=NAME+'.prepare',arguments=deepcopy(arguments),contract=contract(),plans=plans,
        source_slots=[deepcopy(root['slot']) for root in data['roots']],warm_traces=[deepcopy(root['warm_trace']) for root in data['roots']],
        source_roots=root_metadata(data),
        source_report_sha256=SOURCE_SHA,producer_report_sha256=arguments['producer_report_sha256'],
        producer_prepare_sha256=prepare_sha,producer_consume=deepcopy(consumed),bindings=bindings)
    require(len(raw(p))<=PREPARE_CAP,'bounded complete own prepare');return p,data


def prepare(output,arguments):
    p,_=material(arguments);payload=raw(p);path=Path(absolute(output))
    require(path.name.startswith(NAME+'_') and not path.exists() and not path.is_symlink() and
        shutil.disk_usage(path.parent).free>=FLOOR+len(payload),'NEW own prepare +200MiB floor')
    with path.open('xb') as stream:
        require(stream.write(payload)==len(payload),'complete prepare write');stream.flush();os.fsync(stream.fileno())
    paired.fsync_dir(path.parent);return dict(prepared=str(path),prepare_sha256=sha(path),bytes=len(payload))


def verify(prepared,expected):
    actual_hash(expected);absolute(str(prepared));require(sha(prepared)==expected,'external own prepare SHA')
    p=read(prepared);now,data=material(p['arguments'])
    require(p==now and sha(prepared)==expected,'complete own prepare/real producer.consume reconstruction');return p,data


class Store(base.Store):
    def __init__(self,path):
        self.path=Path(absolute(str(path)))
        require(self.path.name.startswith(NAME+'_') and not self.path.exists() and not self.path.is_symlink() and
            self.path.parent.is_dir() and shutil.disk_usage(self.path.parent).free>=FLOOR+CAP,'NEW32MiB own output +200MiB floor')
        self.path.mkdir();paired.fsync_dir(self.path.parent);self.previous=None;self.index=0
    def check(self,size=4096,final=False):
        files=list(self.path.iterdir())
        require(all(f.is_file() and not f.is_symlink() for f in files) and
            sum(f.stat().st_size for f in files)+size<=CAP-(0 if final else FINAL) and
            shutil.disk_usage(self.path).free>=FLOOR+size,'own output cap/reserve/floor including partial files')


class Meter:
    def __init__(self,store,started=None):
        self.store=store;self.started=time.monotonic() if started is None else started
        self.where={};self.receipts=[];self.secondary=[];self.fault=False
        self.counts={k:dict(attempted=0,completed=0) for k in contract()['caps']};self.active={k:0 for k in self.counts}
    def call(self,kind,fn,*,cleanup=False,**fields):
        require(not cleanup or kind=='close','only nonphysical close may bypass predispatch floor/wall fault')
        if not cleanup:
            require(not self.fault and time.monotonic()-self.started<WALL,'wall/fault predispatch');self.store.check()
        count=self.counts[kind];require(count['attempted']<contract()['caps'][kind],'cap before callback: '+kind)
        receipt=dict(index=len(self.receipts),resource=kind,**deepcopy(self.where),**deepcopy(fields),status='pending')
        try:self.store.emit('operation_dispatch_intent',receipt=receipt)
        except BaseException as exc:
            self.fault=True
            if not cleanup:raise
            self.secondary.append(base.error('cleanup_intent',exc))
        self.receipts.append(receipt);count['attempted']+=1;self.active[kind]+=1
        try:
            value=fn();count['completed']+=1;receipt['status']='returned'
        except BaseException as exc:
            receipt.update(status='failed',error=base.error(kind,exc))
            try:self.store.emit('operation_failed',receipt=receipt)
            except BaseException as extra:self.fault=True;self.secondary.append(base.error('failure_journal',extra))
            raise
        finally:self.active[kind]-=1
        try:self.store.emit('operation_returned',receipt=receipt)
        except BaseException as exc:
            self.fault=True
            if not cleanup:raise
            self.secondary.append(base.error('cleanup_return',exc))
        return value


def validate_execution(p):
    require(p['schema']==NAME+'.prepare' and p['contract']==contract() and
        p['source_report_sha256']==SOURCE_SHA and p['producer_prepare_sha256']==p['arguments']['producer_prepare_sha256'] and
        len(p['plans'])==4 and len(p['source_slots'])==len(p['warm_traces'])==len(p['source_roots'])==2 and
        p['plans']==p['producer_consume']['plans'],'own fixed4 strict-consumed prepare, not borrowed schema')
    for i,plan in enumerate(p['plans']):
        require(type(plan['root_slot']) is int and (plan['root_slot'],plan['condition'])==ORDER[i] and plan['arm']==ARM and
            plan['whole_run_report_sha256']==p['producer_report_sha256'],'all4 keys/newarm/producer identity');raw10(plan['commands'])
    expected=[(n,k) for n in range(11) for k in ((0,1,2) if n in (0,5,10) else (0,))]
    for trace in p['warm_traces']:
        require([(x['receipt']['step'],x['receipt']['overlay_index']) for x in trace]==expected,'only original17 warmup references')
    require([r['root_slot'] for r in p['source_roots']]==[0,1],'two original source root identities')


def reference_for(trace,step,overlay):
    exact=next((x for x in trace if (x['receipt']['step'],x['receipt']['overlay_index'])==(step,overlay)),None)
    if exact is None and overlay==2:
        exact=next(x for x in trace if (x['receipt']['step'],x['receipt']['overlay_index'])==(step,0))
    return exact


def check_warm_frame(actual,reference):
    require(actual['pre']==actual['post']==reference['pre']==reference['post'] and
        all(actual['receipt'][k]==reference['receipt'][k] for k in ('phase','step','condition','file_sha256','pixel_sha256','state','snapshot')),
        'original saved warmup goal/body/RGB identity, including source on context')


def observe(env,obs,phase,step,row,meter,store,trace):
    class Sink:
        def __getattr__(self,name):return getattr(store,name)
        def emit(self,kind,**fields):
            store.emit(kind,**fields)
            if kind=='fresh_observation':
                item={k:deepcopy(fields[k]) for k in ('receipt','pre','post')}
                if phase=='warmup' or step<=5:
                    row['prefix51']['frames'].append(item)
    image=base.capture(env,obs,phase,step,row,meter,Sink())
    if phase=='warmup':
        frames=row['prefix51']['frames'][-3:]
        for item in frames:
            overlay=item['receipt']['overlay_index'];ref=reference_for(trace,step,overlay)
            if ref is not None:check_warm_frame(item,ref)
            else:row['source_unpaired_on_steps'].append(step)
    row['prefix51']['complete']=len(row['prefix51']['frames'])==51
    return image


def dispatch(env,command,phase,step,row,meter,store,goal):
    require(command.dtype==np.float32 and command.shape==(2,) and env.unwrapped.with_target is False,'literal raw2/native-off dispatch')
    obs,reward,t,tr,info=meter.call('public_step',lambda:env.step(command.copy()),phase=phase,step=step,raw_bits=command.tobytes().hex())
    actual=np.asarray(env.unwrapped.latest_action).copy();state=source.vector(obs['state'])
    store.emit('returned_transition',**meter.where,phase=phase,step=step,requested_bits=command.tobytes().hex(),
        actual_bits=actual.tobytes().hex(),actual_dtype=actual.dtype.str,state=state.tolist(),snapshot=moving.snapshot(env.unwrapped),
        reward=float(reward),terminated=bool(t),truncated=bool(tr),goal_state=np.asarray(info['goal_state']).tolist())
    return obs,info,bool(t),bool(tr),float(reward),actual


class NativeStop(Exception):
    """Legal stop: never pad history or dispatch beyond native terminal/truncation."""


def episode(env,root,source_plan,plan,row,meter,store):
    body=env.unwrapped;packet=root['packets'][plan['condition']];trace=root['warm_trace']
    require(np.asarray(packet['raw_action_history']).shape==(10,2) and np.asarray(packet['raw_action_history']).dtype==np.float32 and
        np.asarray(packet['raw_action_history']).tobytes()==moving.warmup_commands().tobytes(),'original nonzero warmup raw10 exact')
    obs,native_goal=place(env,source_plan,root['slot'],meter,store)
    history=[observe(env,obs,'warmup',0,row,meter,store,trace)]
    for step,command in enumerate(moving.warmup_commands(),1):
        if body.eval_state(native_goal,source.vector(obs['state']))[0]:raise NativeStop('warmup already native terminal')
        obs,info,t,tr,reward,actual=dispatch(env,command,'warmup',step,row,meter,store,native_goal)
        image=observe(env,obs,'warmup',step,row,meter,store,trace)
        paired.check_return(env,command,actual,info,native_goal,reward,t)
        if t or tr:raise NativeStop('warmup native stop; no padded root context')
        if step in (5,10):history.append(image)
    require(moving.exact(np.asarray(history),packet['rgb_history']),'exact same-condition original RGB history3')
    state=source.vector(obs['state']);goal=source.vector(root['goal'])
    require(source.root_hash(state,goal)==root['root_sha256'],'exact original native7 root/true goal')
    bind_goal(body,goal,packet['goal_rgb'],meter,store);bound=invariant(body)
    row.update(initial_exact=True,root_state=state.tolist(),root_sha256=root['root_sha256'],goal=goal.tolist(),
        goal_pose=bound['goal_pose'],goal_pixel_sha256=moving.image_sha(packet['goal_rgb']))
    observe(env,obs,'plan',0,row,meter,store,trace)
    if body.eval_state(goal,state)[0]:raise NativeStop('constructed task already terminal at root; no own dispatch')
    for step,command in enumerate(raw10(plan['commands']),1):
        obs,info,t,tr,reward,actual=dispatch(env,command,'plan',step,row,meter,store,goal)
        observe(env,obs,'plan',step,row,meter,store,trace)
        success=paired.check_return(env,command,actual,info,goal,reward,t)
        now=invariant(body)
        require(moving.exact(np.asarray(info['goal']),packet['goal_rgb']) and
            {k:v for k,v in now.items() if k!='snapshot'}=={k:v for k,v in bound.items() if k!='snapshot'},'true goalRGB/goal_pose/renderer invariant')
        actual_state=source.vector(obs['state']);delta=actual_state-state;metrics=base.previous.metric(goal,actual_state)
        require(metrics['native_success'] is bool(success) and reward==-metrics['native_distance'],'independent native7 labels')
        response=dict(step=step,state=actual_state.tolist(),delta_native7=delta.tolist(),
            delta_angle_wrapped=float(np.arctan2(np.sin(delta[4]),np.cos(delta[4]))),reward=reward,native_success=bool(success),
            terminated=t,truncated=tr,native_metrics=metrics)
        row['responses'].append(response);row.update(final_state=actual_state.tolist(),final_metrics=metrics,
            native_success=bool(success),terminated=t,truncated=tr)
        store.emit('own_response',**meter.where,response=response)
        if t or tr:row['status']='native_stopped';return
    row['status']='horizon_completed'


def census(rows,counts):
    c={k:v['completed'] for k,v in counts.items()}
    require(all(v['attempted']==v['completed'] for v in counts.values()) and
        c['constructor']==c['close']==4 and c['reset']==8 and
        c['physics_step']==2*c['reset']+10*c['public_step'] and c['set_state']==2*c['reset'] and
        c['goal_render']==c['reset'] and c['goal_setter']==c['reset']+c['goal_bind'] and
        c['public_step']==sum(r['warmup_returned']+r['own_returned'] for r in rows) and
        c['explicit_render']==sum(r['PNG'] for r in rows),'actual4 native counts, no nested summation')


def execute(p,data,store,factory,*,authorization,prepared,prepare_sha256,started=None):
    require(authorization==AUTH,'explicit MAIN physical authority');actual_hash(prepare_sha256);validate_execution(p)
    require(p['source_slots']==[r['slot'] for r in data['roots']] and p['warm_traces']==[r['warm_trace'] for r in data['roots']],
        'same prepared source recipes and exact available warmup references')
    require(p['source_roots']==root_metadata(data),'bound original native task/goal/render metadata')
    meter=Meter(store,started);errors=[];source_ok=False;seen=[]
    binding=dict(prepared=absolute(str(prepared)),prepare_sha256=prepare_sha256,
        producer_report_sha256=p['producer_report_sha256'],producer_prepare_sha256=p['producer_prepare_sha256'],source_report_sha256=SOURCE_SHA)
    rows=[dict(index=i,root_slot=k[0],condition=k[1],arm=ARM,task_id=plan['task_id'],selection_digest=plan['selection_digest'],
        plan=deepcopy(plan),status='not_started',first_error=None,secondary_errors=[],warmup_attempted=0,warmup_returned=0,
        own_attempted=0,own_returned=0,PNG=0,initial_exact=False,trajectory=[],checkpoints=[],missing_checkpoints=[0,5,10],
        last_observation=None,responses=[],final_state=None,final_metrics=None,native_success=None,terminated=None,truncated=None,
        prefix51=dict(complete=False,frames=[]),source_unpaired_on_steps=[]) for i,(k,plan) in enumerate(zip(ORDER,p['plans'],strict=True))]
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
            env=None;row['status']='running'
            try:
                env=meter.call('constructor',factory)
                require(all(env is not old and env.unwrapped is not old.unwrapped for old in seen),'fresh environment per fixed condition');seen.append(env)
                validate_env(env.unwrapped)
                require(env.unwrapped.render_action is False and env.unwrapped.render_size==224 and env.unwrapped.render_mode=='rgb_array','native renderer/action semantics')
                with native_hooks(env,meter,row['secondary_errors']):episode(env,data['roots'][plan['root_slot']],data['source_plan'],plan,row,meter,store)
                require(not row['secondary_errors'],'native hook restoration failed')
            except NativeStop as exc:row.update(status='native_stopped',stop_reason=str(exc))
            except BaseException as exc:fail('branch',exc,row)
            finally:
                if env is not None:
                    try:meter.call('close',env.close,cleanup=True)
                    except BaseException as exc:fail('close',exc,row)
                if row['secondary_errors'] and row['first_error'] is None:fail('cleanup',ValueError('see secondary hook restoration errors'),row)
                for phase,label in (('warmup','warmup'),('plan','own')):
                    ops=[r for r in meter.receipts if r['resource']=='public_step' and r.get('branch')==row['index'] and r['phase']==phase]
                    row[label+'_attempted']=len(ops);row[label+'_returned']=sum(r['status']=='returned' for r in ops)
            store.emit('branch_closed',receipt=row)
    except BaseException as exc:fail('execution',exc)
    try:
        for path,h in p['bindings'].items():require(sha(path)==h,'final input/source seal')
        source_ok=True
    except BaseException as exc:fail('final_inputs',exc)
    errors.extend(meter.secondary)
    if time.monotonic()-meter.started>=WALL:fail('final_wall',ValueError('900s wall cap; cleanup retained'))
    if not errors:
        try:census(rows,meter.counts)
        except BaseException as exc:fail('census',exc)
    complete=not errors and source_ok and all(r['status'] in ('horizon_completed','native_stopped') for r in rows)
    status=RECORDED if complete else 'partial_or_failed'
    try:store.emit('run_finished',status=status,counts=meter.counts,errors=errors)
    except BaseException as exc:fail('final_journal',exc);status='partial_or_failed'
    report=dict(schema=NAME+'.report',status=status,rows=rows,errors=errors,execution_binding=binding,execution_binding_sha256=None,
        fixed_denominator=4,known_roots=2,fresh_roots=0,contract=contract(),counts=deepcopy(meter.counts),actual_operation_receipts=deepcopy(meter.receipts),
        inputs_final_verified=source_ok,full_horizons=sum(r['status']=='horizon_completed' for r in rows),
        early_stops=sum(r['status']=='native_stopped' for r in rows),engineering_failures=sum(r['status']=='engineering_failed' for r in rows),
        not_started=sum(r['status']=='not_started' for r in rows),total_internal_physics_scope='metered_native_calls' if meter.counts['constructor']['attempted']==meter.counts['constructor']['completed'] else 'unknown_constructor_work',
        new_neural_calls=0,new_independent_samples=0,training=False,P1_admission=False,P3_admission=False,scientific_admission=False,
        independent_confirmation=False,equal_total_compute=False,equal_FLOPs=False,freshness=False,
        wall_seconds=time.monotonic()-meter.started,journal_sha256=None,journal_chain_tail=store.previous)
    for name,key in (('journal.jsonl','journal_sha256'),('execution_binding.json','execution_binding_sha256')):
        try:report[key]=sha(store.path/name)
        except BaseException as exc:fail('final_'+key,exc);report['status']='partial_or_failed'
    report['wall_seconds']=time.monotonic()-meter.started
    if report['wall_seconds']>=WALL:report['status']='partial_or_failed'
    try:store.put('report.json',report,final=True)
    except BaseException as exc:fail('final_report',exc);report['status']='partial_or_failed'
    return report


def run(prepared,expected,output,factory,*,authorization):
    require(authorization==AUTH,'MAIN authorization before reads/output/factory')
    started=time.monotonic();p,data=verify(prepared,expected)
    return execute(p,data,Store(output),factory,authorization=authorization,prepared=prepared,prepare_sha256=expected,started=started)


def validate_saved(report,events,p):
    validate_execution(p)
    require(report['schema']==NAME+'.report' and report['status'] in (RECORDED,'partial_or_failed') and
        report['fixed_denominator']==len(report['rows'])==4 and report['known_roots']==2 and report['fresh_roots']==0 and
        report['contract']==contract() and report['execution_binding']['producer_report_sha256']==p['producer_report_sha256'] and
        report['execution_binding']['producer_prepare_sha256']==p['producer_prepare_sha256'] and report['execution_binding']['source_report_sha256']==SOURCE_SHA,
        'own fixed4 report/source/producer identity')
    require(all(report[k] is False for k in ('training','P1_admission','P3_admission','scientific_admission','independent_confirmation','equal_total_compute','equal_FLOPs','freshness')) and
        report['new_neural_calls']==report['new_independent_samples']==0,'no scientific/freshness upgrade')
    require(events and events[0]['kind']=='run_started' and events[0]['contract']==contract() and
        events[0]['prepare_sha256']==report['execution_binding']['prepare_sha256'] and events[-1]['kind']=='run_finished' and
        events[-1]['status']==report['status'] and events[-1]['counts']==report['counts'] and events[-1]['errors']==report['errors'],'durable begin/end')
    tail=None;intents={};terminals={};stack=[]
    for i,e in enumerate(events):
        require(e['index']==i and e['previous_sha256']==tail and e['event_sha256']==source.digest({k:v for k,v in e.items() if k!='event_sha256'}),'journal hashchain')
        tail=e['event_sha256']
        if e['kind'] not in ('operation_dispatch_intent','operation_returned','operation_failed'):continue
        op=e['receipt'];n=op['index'];require(type(n) is int and op['resource'] in contract()['caps'],'literal operation index/resource')
        b=op['branch'];require(type(b) is int and 0<=b<4 and
            all(op[k]==report['rows'][b][k] for k in ('root_slot','condition','arm','selection_digest')),
            'actual operation scope matches fixed branch identity')
        if e['kind']=='operation_dispatch_intent':
            require(n==len(intents) and op['status']=='pending','ordered durable dispatch');intents[n]=op;stack.append(n)
        else:
            require(n in intents and n not in terminals and stack and stack.pop()==n and
                op['status']==('returned' if e['kind']=='operation_returned' else 'failed') and
                dict({k:v for k,v in op.items() if k!='error'},status='pending')==intents[n],'nested actual return/failure identity')
            terminals[n]=op
    require(not stack and len(intents)==len(terminals) and tail==report['journal_chain_tail'] and
        report['actual_operation_receipts']==[terminals[i] for i in range(len(terminals))],'complete actual operation receipts')
    counts={k:dict(attempted=0,completed=0) for k in contract()['caps']}
    for op in report['actual_operation_receipts']:
        counts[op['resource']]['attempted']+=1;counts[op['resource']]['completed']+=op['status']=='returned'
    require(counts==report['counts'] and all(v['attempted']<=contract()['caps'][k] for k,v in counts.items()),'actual counts/caps, not fixed fill/refund')
    require([e['receipt'] for e in events if e['kind']=='branch_closed']==[r for r in report['rows'] if r['status']!='not_started'],'fixed4 failure closure')
    for i,row in enumerate(report['rows']):
        plan=p['plans'][i];slot=plan['root_slot'];scoped=[e for e in events if e.get('branch')==i]
        task=p['source_roots'][slot]
        require(type(row['index']) is type(row['root_slot']) is int and row['index']==i and
            (row['root_slot'],row['condition'])==ORDER[i] and row['arm']==ARM and row['plan']==plan and
            all(row[k]==plan[k] for k in ('selection_digest','task_id')),'exact fixed4 row/source/raw plan')
        require(row['status'] in ('horizon_completed','native_stopped','engineering_failed','not_started'),'terminal row status')
        for e in scoped:
            require(all(e[k]==row[k] for k in ('root_slot','condition','arm','selection_digest')),'branch-scoped physical identity')
        ops=[r for r in report['actual_operation_receipts'] if r.get('branch')==i and r['resource']=='public_step']
        for phase,label,commands in (('warmup','warmup',moving.warmup_commands()),('plan','own',raw10(plan['commands']))):
            current=[op for op in ops if op['phase']==phase]
            require([op['step'] for op in current]==list(range(1,len(current)+1)) and len(current)<=10 and
                all(op['raw_bits']==commands[j].tobytes().hex() for j,op in enumerate(current)) and
                row[label+'_attempted']==len(current) and row[label+'_returned']==sum(op['status']=='returned' for op in current),'contiguous actual raw bits/attempts/returns')
        observations=[e for e in scoped if e['kind']=='fresh_observation']
        require(row['PNG']==len(observations),'actual PNG count')
        bridge=[{k:deepcopy(e[k]) for k in ('receipt','pre','post')} for e in observations if e['receipt']['phase']=='warmup' or e['receipt']['step']<=5]
        require(row['prefix51']['frames']==bridge and row['prefix51']['complete'] is (len(bridge)==51),'actual51 bridge, no padded source identity')
        unknown=[]
        for e in observations:
            f=e['receipt'];require(e['pre']==e['post'] and f['snapshot']==e['post']['snapshot'] and f['state']==f['snapshot']['state'],'render/native7/body invariant')
            require(e['pre']['renderer']==task['renderer'] and e['pre']['goal_pose']==task['goal_pose'],
                'original bound renderer/goal pose, not self-certified physical row')
            if f['phase']=='plan':
                require(e['pre']['goal_state']==task['goal'] and
                    e['pre']['goal_rgb_sha256']==task['goal_pixels'][row['condition']], 'bound condition-specific true task goal')
            if f['phase']=='warmup':
                ref=reference_for(p['warm_traces'][slot],f['step'],f['overlay_index'])
                if ref is not None:check_warm_frame(e,ref)
                else:unknown.append(f['step'])
        require(row['source_unpaired_on_steps']==unknown,'unavailable intermediate old on identities remain explicit')
        responses=row['responses'];trans=[e for e in scoped if e['kind']=='returned_transition']
        require([r['step'] for r in responses]==list(range(1,len(responses)+1)) and len(responses)<=row['own_returned'] and
            [e['response'] for e in scoped if e['kind']=='own_response']==responses,'durable actual response chronology')
        for phase,label in (('warmup','warmup'),('plan','own')):
            transitions=[e for e in trans if e['phase']==phase]
            require(len(transitions)<=row[label+'_returned'],'actual transitions at most returned callbacks')
            for t in transitions:
                require(t['actual_dtype']==np.dtype(np.float32).str and t['actual_bits']==t['requested_bits'],'literal dispatched raw return, no substitution')
        if row['initial_exact']:
            require(all(row[k]==task[k] for k in ('root_state','root_sha256','goal','goal_pose')) and
                row['goal_pixel_sha256']==task['goal_pixels'][row['condition']], 'original source root/goal/condition identity')
            require(source.root_hash(source.vector(row['root_state']),source.vector(row['goal']))==row['root_sha256'],'true native7 root/goal SHA')
            attached=[e for e in scoped if e['kind']=='goal_attached']
            require(len(attached)==1 and attached[0]['goal']==row['goal'] and attached[0]['goal_pose']==row['goal_pose'] and
                attached[0]['goal_rgb_sha256']==row['goal_pixel_sha256'],'actual same-condition goal attach, not default pose construction')
        for response in responses:
            delta=source.vector(response['state'])-source.vector(row['root_state']);metrics=base.previous.metric(row['goal'],response['state'])
            require(response['native_metrics']==metrics and response['reward']==-metrics['native_distance'] and
                response['native_success'] is response['terminated'] is metrics['native_success'] and
                np.array_equal(delta,np.asarray(response['delta_native7'])) and response['delta_angle_wrapped']==float(np.arctan2(np.sin(delta[4]),np.cos(delta[4]))),'independent actual native7/reward/success/delta')
            t=next(e for e in trans if e['phase']=='plan' and e['step']==response['step'])
            require(t['state']==response['state'] and t['reward']==response['reward'] and t['goal_state']==row['goal'] and
                t['terminated'] is response['terminated'] and t['truncated'] is response['truncated'],'response/actual return identity')
        if responses:
            last=responses[-1];require(row['final_state']==last['state'] and row['final_metrics']==last['native_metrics'] and
                all(row[k] is last[k] for k in ('native_success','terminated','truncated')),'actual endpoint only')
            require(all(not x['terminated'] and not x['truncated'] for x in responses[:-1]),'no post-native-stop calls')
        else:require(row['final_state'] is row['final_metrics'] is row['native_success'] is None,'unknown task outcome not zero')
        for cp in row['checkpoints']:
            matches=[e['receipt'] for e in observations if e['receipt']['phase']=='plan' and e['receipt']['step']==cp['step'] and
                e['receipt']['overlay_index']==(1 if row['condition']=='on' else 0)]
            require(len(matches)==1 and all(cp[k]==matches[0][k] for k in ('step','png','file_sha256','pixel_sha256','state','snapshot')),'checkpoint from actual own chosen-condition frame')
        if row['status'] in ('horizon_completed','native_stopped'):
            require(row['first_error'] is None and row['secondary_errors']==[],'engineering complete row')
            if responses:
                require(row['warmup_returned']==10 and row['initial_exact'] and row['own_returned']==len(responses) and
                    ((row['status']=='native_stopped')==(responses[-1]['terminated'] or responses[-1]['truncated'])),'native earlystop, not padded horizon')
            if row['status']=='horizon_completed':require(row['own_returned']==10,'full10 actual returns')
            expected_cp=[n for n in (0,5,10) if row['initial_exact'] and n<=row['own_returned']]
            require([cp['step'] for cp in row['checkpoints']]==expected_cp and row['missing_checkpoints']==[n for n in (0,5,10) if n not in expected_cp],'reached checkpoints only')
            expected_frames=[('warmup',n,k) for n in range(row['warmup_returned']+1) for k in (0,1,2)]
            if row['initial_exact']:expected_frames += [('plan',n,k) for n in range(row['own_returned']+1) for k in (0,1,2)]
            require([(e['receipt']['phase'],e['receipt']['step'],e['receipt']['overlay_index']) for e in observations]==expected_frames,'complete actual off/on/off chronology, no substituted frames')
    for field,status in (('full_horizons','horizon_completed'),('early_stops','native_stopped'),('engineering_failures','engineering_failed'),('not_started','not_started')):
        require(report[field]==sum(r['status']==status for r in report['rows']),'fixed4 terminal summary')
    expected_scope='metered_native_calls' if counts['constructor']['attempted']==counts['constructor']['completed'] else 'unknown_constructor_work'
    require(report['total_internal_physics_scope']==expected_scope,'partial constructor work unknown')
    if report['status']==RECORDED:
        require(report['errors']==[] and report['inputs_final_verified'] is True and report['wall_seconds']<WALL and
            all(r['status'] in ('horizon_completed','native_stopped') for r in report['rows']),'engineering complete, NEVER P1 pass');census(report['rows'],counts)
    return deepcopy(report['rows'])


def saved_artifacts(directory):
    files=list(Path(directory).iterdir());require(all(f.is_file() and not f.is_symlink() for f in files) and
        sum(f.stat().st_size for f in files)<=CAP,'bounded regular own artifacts')
    values={f.name:f.read_bytes() for f in files};require(sum(map(len,values.values()))<=CAP,'bounded actual bytes');return values


def consume(output,expected):
    actual_hash(expected);directory=Path(absolute(str(output)));files=saved_artifacts(directory)
    require(hashlib.sha256(files['report.json']).hexdigest()==expected,'external actual own physical report SHA')
    r=decode(files['report.json']);binding=r['execution_binding']
    require(hashlib.sha256(files['execution_binding.json']).hexdigest()==r['execution_binding_sha256'] and
        decode(files['execution_binding.json'])==binding,'actual execution binding')
    p,_=verify(binding['prepared'],binding['prepare_sha256']);content=files['journal.jsonl']
    require(content.endswith(b'\n') and hashlib.sha256(content).hexdigest()==r['journal_sha256'],'complete actual journal')
    events=[decode(line) for line in content.splitlines()];rows=validate_saved(r,events,p)
    names={'report.json','execution_binding.json','journal.jsonl'}
    from PIL import Image
    for e in events:
        if e['kind']=='invalid_observation':
            name=e['file'];require(Path(name).name==name and name not in ('','.','..') and
                hashlib.sha256(files[name]).hexdigest()==e['file_sha256'],'sealed invalid incident');names.add(name)
        if e['kind']!='fresh_observation':continue
        f=e['receipt'];name=f['png'];require(Path(name).name==name and name not in ('','.','..') and len(files[name])<=1024**2 and
            hashlib.sha256(files[name]).hexdigest()==f['file_sha256'],'actual flat PNG SHA');names.add(name)
        with Image.open(io.BytesIO(files[name])) as image:
            require(image.mode=='RGB' and image.size==(224,224) and moving.image_sha(np.asarray(image))==f['pixel_sha256'],'actual own RGB224 pixels')
    require(set(files)==names,'no unaccounted artifacts/partial writes')
    bindings=deepcopy(p['bindings']);merge(bindings,{binding['prepared']:binding['prepare_sha256']})
    merge(bindings,{str(directory/name):hashlib.sha256(body).hexdigest() for name,body in files.items()})
    for path,h in p['bindings'].items():require(sha(path)==h,'final full input/source closure')
    require(saved_artifacts(directory)==files,'final actual artifact drift')
    return dict(schema=NAME+'.consume',status=r['status'],rows=rows,fixed_denominator=4,known_roots=2,fresh_roots=0,
        errors=deepcopy(r['errors']),inputs_final_verified=r['inputs_final_verified'],
        counts=r['counts'],bindings=bindings,physical_report_sha256=expected,producer_report_sha256=p['producer_report_sha256'],
        P1_admission=False,P3_admission=False,scientific_admission=False,independent_confirmation=False,
        equal_total_compute=False,equal_FLOPs=False,freshness=False,new_neural_calls=0,new_physics_calls=0)


def reuse_rows(p,old,prepared):
    """Only fixed teacher_full D2P, never search outcomes or another cohort.

    Original full8 has no consume API. It is NOT authenticated here by its
    prefix-only probe.material. A fully consumed spectral24 suffix execution
    is usable only when its entire literal raw10 also equals the original10.
    """
    require(old['schema']==OLD_PHYSICAL+'.consume' and old['fixed_denominator']==len(old['rows'])==24 and
        all(old[k] is False for k in ('P1_admission','scientific_admission','independent_confirmation')) and
        old['new_neural_calls']==0 and len(prepared['plans'])==8,'complete oldphysical.consume24 only, never observed4')
    keys=[(s,c,a,v) for s in (0,1) for c in ('off','on') for a in ('line_search_GN','full10_conditional_D2P')
          for v in ('open_full','teacher_full','teacher_small')]
    require([(r['root_slot'],r['condition'],r['arm'],r['variant']) for r in old['rows']]==keys and
        all(type(r['index']) is int and r['index']==i for i,r in enumerate(old['rows'])), 'complete original24 order/unique denominator')
    combined=deepcopy(p['bindings']);merge(combined,old['bindings'])
    source_path=str(base.SOURCE_REPORT)
    require(p['bindings'][source_path]==old['bindings'][source_path]==SOURCE_SHA,'same actual paired source report ancestry')
    rows=[]
    for i,(slot,condition) in enumerate(ORDER):
        new=p['plans'][i];ordinal=2*i+1;old_index=3*ordinal+1
        original=prepared['plans'][ordinal];r=old['rows'][old_index];selected=r['selected_suffix']['plan']
        require((new['root_slot'],new['condition'])==(slot,condition) and new['arm']==ARM and
            (original['root_slot'],original['condition'],original['arm'])==(slot,condition,'full10_conditional_D2P') and
            r['task_id']==original['task_id']==new['task_id'] and
            r['selection_digest']==original['selection_digest'] and
            all(selected['original_plan'][k]==v for k,v in original.items()) and
            original['source_report_sha256']==new['source_report_sha256']==SOURCE_SHA and
            original['blind_sha256']==new['blind_sha256'],'exact original task/blind/source/prefix ordinal ancestry')
        task_paths=[f for f,h in p['bindings'].items() if h==new['task_sha256']]
        require(len(task_paths)==1 and old['bindings'].get(task_paths[0])==new['task_sha256'],
            'same full encoded task artifact SHA/context/goal/Q/model; raw equality alone insufficient')
        item=dict(root_slot=slot,condition=condition,arm=ARM,selection_digest=new['selection_digest'],
            old_index=old_index,old_selection_digest=original['selection_digest'],status='unknown',reason=None,
            evidence=None,native_success=None,freshness=False,P1_admission=False)
        same=raw10(new['commands']).tobytes()==raw10(original['commands']).tobytes()==raw10(selected['commands']).tobytes()
        if not same:item['reason']='literal_full10_mismatch_no_reuse_no_automatic_physics'
        elif r['status'] not in ('suffix_completed','native_stopped'):
            item['reason']='old_physical_failed_or_unstarted_outcome_unknown'
        else:
            task=p['source_roots'][slot]
            require(r['first_error'] is None and r['secondary_errors']==[] and r['start_verified'] is True and
                all(r[k]==task[k] for k in ('root_state','root_sha256','goal','goal_pose')) and
                r['goal_pixel_sha256']==task['goal_pixels'][condition],'bound true native task and condition-specific goal image')
            item.update(status='inherited_saved_physical_not_fresh',reason='both_complete_consumers_and_exact_ancestry',
                evidence=deepcopy(r),native_success=r['native_success'])
        rows.append(item)
    return rows


def reuse_material(arguments,old_output,old_report_sha256,old_source_sha256):
    """Saved only. Exceptions from either complete consume propagate unchanged."""
    actual_hash(old_report_sha256);actual_hash(old_source_sha256)
    directory=Path(absolute(old_output));rp=directory/'report.json'
    path=ROOT/'experiments/spectral_reachability'/(OLD_PHYSICAL+'.py')
    require(old_source_sha256==OLD_PHYSICAL_SHA==sha(path) and sha(rp)==old_report_sha256,
        'external fixed oldphysical source/report seals BEFORE either consume')
    report=read(rp,192*1024**2)
    require(report['schema']==OLD_PHYSICAL+'.report' and len(report['rows'])==24,
        'terminal old24 report required, no original8 prefix shortcut or observed4')
    binding=report['execution_binding'];pp=absolute(binding['prepared'])
    actual_hash(binding['prepare_sha256'])
    require(sha(pp)==binding['prepare_sha256'],'external old prepare SHA BEFORE either consume')
    p,_=material(arguments)  # Mandatory complete real newproducer.consume.
    module=importlib.import_module('experiments.spectral_reachability.'+OLD_PHYSICAL)
    require(Path(module.__file__).resolve()==path and callable(getattr(module,'consume',None)), 'real fixed oldphysical.consume API')
    old=module.consume(directory,old_report_sha256)  # Mandatory complete oldphysical.consume.
    require(old['physical_report_sha256']==old_report_sha256 and old['rows']==report['rows'], 'full old consumer/report row identity')
    require(sha(pp)==binding['prepare_sha256'] and old['bindings'][pp]==binding['prepare_sha256'],'actual consumed old prepare')
    prepared=read(pp,PREPARE_CAP);rows=reuse_rows(p,old,prepared)
    bindings=deepcopy(p['bindings']);merge(bindings,old['bindings']);merge(bindings,{str(path):old_source_sha256,str(rp):old_report_sha256})
    for f,h in bindings.items():require(sha(f)==h,'final complete both-consumer closure')
    return dict(schema=NAME+'.reuse',status='saved_reuse_checked_not_admitted',
        arguments=deepcopy(arguments),old_output=old_output,old_report_sha256=old_report_sha256,old_source_sha256=old_source_sha256,
        producer_consume=p['producer_consume'],old_physical_consume=old,
        old_report_metadata={k:deepcopy(v) for k,v in report.items() if k not in ('rows','actual_operation_receipts')},rows=rows,bindings=bindings,
        fixed_denominator=4,known_roots=2,independent_root_count=2,fresh_roots=0,
        reused=sum(r['status']=='inherited_saved_physical_not_fresh' for r in rows),
        unknown=sum(r['status']=='unknown' for r in rows),new_neural_calls=0,new_physics_calls=0,
        freshness=False,P1_admission=False,P3_admission=False,equal_total_compute=False,equal_FLOPs=False,
        scientific_admission=False,independent_confirmation=False,
        budget_note='historical physical evidence only; allmode and old planner budgets differ; nested resource axes not added',
        original_full8_limit='no consume API in original full8; prefix-only material is not full10 evidence release')


def reuse(output,arguments,old_output,old_report_sha256,old_source_sha256):
    path=Path(absolute(output))
    require(path.name.startswith(NAME+'_') and not path.exists() and not path.is_symlink(),'NEW exclusive reuse receipt BEFORE validation')
    result=reuse_material(arguments,old_output,old_report_sha256,old_source_sha256);body=raw(result)
    require(len(body)<=CAP and shutil.disk_usage(path.parent).free>=FLOOR+len(body),'bounded reuse receipt +200MiB floor')
    with path.open('xb') as stream:
        require(stream.write(body)==len(body),'complete reuse write');stream.flush();os.fsync(stream.fileno())
    paired.fsync_dir(path.parent)
    return dict(output=str(path),sha256=sha(path),bytes=len(body),reused=result['reused'],unknown=result['unknown'],P1_admission=False,freshness=False)


def verify_reuse(output,expected):
    actual_hash(expected);path=Path(absolute(output));require(sha(path)==expected,'external reuse receipt SHA')
    saved=read(path,CAP)
    now=reuse_material(saved['arguments'],saved['old_output'],saved['old_report_sha256'],saved['old_source_sha256'])
    require(saved==now and sha(path)==expected,'complete both-consumer reuse reconstruction')
    return dict(status='saved_reuse_verified_not_admitted',fixed_denominator=4,reused=now['reused'],unknown=now['unknown'],
        P1_admission=False,freshness=False,new_neural_calls=0,new_physics_calls=0)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','run','consume','reuse','verify-reuse'))
    for name in ('output','prepared','prepare-sha','report-sha','producer-output','producer-report-sha','producer-source-sha',
                 'producer-prepared','producer-prepare-sha','old-output','old-report-sha','old-source-sha'):parser.add_argument('--'+name)
    parser.add_argument('--authorize-physics',action='store_true');args=parser.parse_args(argv)
    if args.command in ('prepare','reuse'):
        arguments=dict(producer_output=args.producer_output,producer_report_sha256=args.producer_report_sha,
            producer_source_sha256=args.producer_source_sha,producer_prepared=args.producer_prepared,producer_prepare_sha256=args.producer_prepare_sha)
        result=prepare(args.output,arguments) if args.command=='prepare' else reuse(args.output,arguments,args.old_output,args.old_report_sha,args.old_source_sha)
    elif args.command=='verify':
        p,_=verify(args.prepared,args.prepare_sha);result=dict(status='verified_no_execution',contract=p['contract'])
    elif args.command=='consume':result=consume(args.output,args.report_sha)
    elif args.command=='verify-reuse':result=verify_reuse(args.output,args.report_sha)
    else:
        require(args.authorize_physics,'MAIN explicit --authorize-physics')
        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401
            return gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224,with_target=False)
        result=run(args.prepared,args.prepare_sha,args.output,factory,authorization=AUTH)
    print(source.canonical(result));return 1 if result.get('status')=='partial_or_failed' else 0


if __name__=='__main__':raise SystemExit(main())
