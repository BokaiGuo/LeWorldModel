"""Own full30 four-slot native replay. MAIN only; zero neural execution.

Own full30 schema/loop/validator, frozen native placement/hooks/capture reused.
No model, imagined-context restoration, outcome selection or scientific admission.
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
import time
import numpy as np
from experiments.spectral_reachability import agent_mixedmode_physical_20261028_v1 as native
from experiments.spectral_reachability import p3_goal_patch_consistent_compiler_20261027_v2 as normalization
from experiments.spectral_reachability import agent_p2_two_program_postcontext_20261029_v1_kernel as K

NAME='agent_p2_two_program_full30_physical_20261029_v1'
PRODUCER='agent_p2_two_program_postcontext_20261029_v1'
AUTH='MAIN_TWO_PROGRAM_FULL30_PHYSICS_ONLY'
RECORDED='full30_four_native_recorded_NOT_admitted'
ROOT=native.ROOT
ARMS=('two_program_transfer','joint_two_round')
ORDER=[(r,'off',a) for r in (0,1) for a in ARMS]
CP=(0,5,10,15,20,25,30)
CAP,FINAL,FLOOR,WALL,META=256*1024**2,8*1024**2,200*1024**2,900,16*1024**2
PNG_CAP,JOURNAL_CAP,OTHER_CAP=256*1024,64*1024**2,16*1024**2
require,sha,raw,read=native.require,native.sha,native.raw,native.read
source,moving,paired=native.source,native.moving,native.paired
# Own physical schema uses spaced JSON; planner uses compact JSON + newline.
digest=native.core.digest
ARGS={'planner_prepared','planner_prepare_sha','planner_output','planner_report_sha','planner_receipt','planner_receipt_sha'}
PINS=dict(planner_prepared=str(ROOT/'results/agent_p2_two_program_postcontext_20261029_v1_prepare_MAIN01.json'),
    planner_prepare_sha='da1803815f14a1e3ee254e7915dc781d66367c7f96b011547c10f0736ba3f9c1',
    planner_output=str(ROOT/'results/agent_p2_two_program_postcontext_20261029_v1_cuda_MAIN01'),
    planner_report_sha='99f2b5edceae839bc3b0387ac93ef1d866a3fb74ec869ec7a30ceeaf4211ae00',
    planner_receipt=str(ROOT/'results/agent_p2_two_program_postcontext_20261029_v1_consume_MAIN01.json'),
    planner_receipt_sha='af2890ae4d330f69e8b01c7a144786ee454b08091e6d79ef63fa1b42e0bf6c5e')
PROTOCOL=ROOT/'innerdocs/agent_p2_two_program_full30_physical_protocol_20261029_v1.md'
PROTOCOL_SHA='dcfc98c43431d09a74227ffc5d1ebaaf9b3edab144ac3387eef46d6170da35dd'
CODE_PINS={
    str(ROOT/'experiments/spectral_reachability'/('agent_p2_two_program_postcontext_20261029_v1.py')):
        'a438e13844c3ce613f22ece0aef104e2a167b6ff3557f39c10feb5a8a52981f6',
    str(Path(K.__file__).resolve()):'3c961b5ae736cfa887a8f81f05838155066523a6d1e139711691ef13a748614d'}

place,bind_goal,dispatch=native.place,native.bind_goal,native.dispatch
validate_env,native_hooks,invariant=native.validate_env,native.native_hooks,native.invariant


def planner_digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n').hexdigest()


def raw30(value):
    require(type(value) is list and len(value)==30 and all(type(r) is list and len(r)==2 and
        all(type(x) in (int,float) for x in r) for r in value),'literal30x2 numeric raw commands, no bool')
    a=np.asarray(value,dtype=np.float32)
    require(np.isfinite(a).all() and np.all(np.abs(a)<=1) and
        all(float(x)==float(y) for x,y in zip(np.asarray(value,dtype=np.float64).flat,a.flat)),
        'literal lossless float32 raw30 in box, no clipping or reconstruction')
    return a


def contract():
    c=deepcopy(native.allmode.contract())
    c.pop('arm')  # Old single-arm identifier is not this own two-arm H30 schema.
    c.update(schema=NAME+'.contract',order=[list(x) for x in ORDER],arms=list(ARMS),condition='off',
        warmup=10,horizon=30,checkpoints=list(CP),producer=PRODUCER,
        raw_authority='pinned MAIN complete full30.consume literal raw30; no native reselection',
        missing_export='invalid authority blocked before run; runtime failures retain all4 rows',
        scientific_authority='physical_release false is not scientific authorization; strict + explicit MAIN engineering AUTH required',
        execution='original reset/place/warm10 then literal selectedraw30, no replanning or imagined-state restoration',
        caps=dict(constructor=4,close=4,reset=8,public_step=160,physics_step=1616,set_state=16,
            goal_setter=12,goal_render=8,goal_bind=4,explicit_render=504),
        wall_seconds=WALL,output_cap_bytes=CAP,final_reserve_bytes=FINAL,disk_floor_bytes=FLOOR,
        prepare_cap_bytes=META,inheritance=False,full_upstream_rebuilt_here=False,
        file_caps=dict(PNG=PNG_CAP,journal=JOURNAL_CAP,other=OTHER_CAP,max_PNG=504),
        budget_proof=dict(PNG_bytes=504*PNG_CAP,journal_bytes=JOURNAL_CAP,other_bytes=OTHER_CAP,
            reserve_bytes=FINAL,total_reserved_bytes=504*PNG_CAP+JOURNAL_CAP+OTHER_CAP+FINAL),selection_uses_physics=False,
        P1_admission=False,P2_admission=False,P2_mechanism=False,physical_release=False,
        interpretation='four paired arm branches at two known original off roots, not independent fresh roots; not equal FLOPs')
    return c


def own_files():
    return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def check_planner_plan(chosen,slot,report_sha,model_hash):
    require(type(chosen) is dict and chosen.get('schema')==PRODUCER+'.model_plan','own full30 model_plan schema')
    require(chosen['witness_sha256']==K.digest({k:v for k,v in chosen.items() if k!='witness_sha256'}),
        'original full30 plan witness')
    a=raw30(chosen['commands']);K.context(chosen['root_context'])
    require(type(chosen['root_slot']) is int and chosen['root_slot']==slot['root_slot']
        and chosen['task_id']==slot['task_id'] and chosen['task_sha256']==slot['task_sha256']
        and chosen['model_hash']==model_hash and chosen['root_context']==slot['root_context']
        and chosen['root_warm_raw10']==slot['root_warm_raw10'] and chosen['goal']==slot['goal'],
        'original task/model/root/full history/goal join')
    require(chosen['arm'] in ARMS and chosen['new_raw_steps']==30
        and chosen['program_count']==(2 if chosen['arm']==ARMS[0] else 1)
        and chosen['program_start_controls']==([0,15] if chosen['arm']==ARMS[0] else [0])
        and chosen['new_domain']==dict(logit_dim=60,logit_cap=K.C30,raw_bounds=[-1,1]),
        'H6 full30 identity, no full15 schema impersonation')
    K.legal(chosen['logits'])
    require(a.tobytes().hex()==chosen['raw_float32_bits']
        and a.tobytes()==K.chart(chosen['logits']).tobytes(),'literal tanh64 cast32 raw30, no repair')
    require(chosen['source_programs']==[dict(id=x['id'],sha256=x['program_sha256']) for x in slot['programs']]
        and chosen['normalization']==slot['programs'][0]['scope']['normalization']
        ==dict(mean=normalization.ACTION_MEAN.tolist(),std=normalization.ACTION_STD.tolist()),
        'two complete source programs and unchanged native normalizer')
    require(type(chosen['contexts']) is list and len(chosen['contexts'])==6,'six full postcontexts')
    previous=chosen['root_context']
    for n,c in enumerate(chosen['contexts']):
        K.context(c)
        # Context motion and literal normalized action-token insertion, not just terminal equality.
        expect=normalization.normalized_history(a[n*5:n*5+5])[0]
        require(np.asarray(c['latent_history'][:2],np.float32).tobytes()
                ==np.asarray(previous['latent_history'][1:],np.float32).tobytes()
            and np.asarray(c['action_history'][0],np.float32).tobytes()
                ==np.asarray(previous['action_history'][1],np.float32).tobytes()
            and np.asarray(c['action_history'][1],np.float32).tobytes()==expect.astype(np.float32).tobytes(),
            'every model context exact history/action seam')
        previous=c
    require(type(chosen['predicted_Q']) in (int,float) and np.isfinite(chosen['predicted_Q'])
        and chosen['predicted_Q']==K.objective(chosen['contexts'][-1],chosen['goal']),'independent terminal model Q')
    require(all(chosen[k] is False for k in K.FLAGS),'original science/release flags remain false')


def assemble(arguments,compiled,consumed,data):
    require(arguments==PINS and compiled['schema']==PRODUCER+'.prepare'
        and consumed['schema']==PRODUCER+'.consume'
        and consumed['status']=='full30_transfer_and_joint_verified_NOT_admitted'
        and consumed['report_sha256']==arguments['planner_report_sha']
        and consumed['prepare_sha256']==arguments['planner_prepare_sha']
        and consumed['errors']==[] and len(consumed['plans'])==4
        and len(compiled['slots'])==2 and all(consumed[k] is False for k in K.FLAGS),
        'pinned MAIN actual complete full30 model authority, all four')
    plans=[]
    for i,(chosen,key) in enumerate(zip(consumed['plans'],ORDER,strict=True)):
        slot=compiled['slots'][key[0]]
        require(type(slot['root_slot']) is int and slot['root_slot']==key[0] and slot['condition']=='off',
            'known two off roots')
        check_planner_plan(chosen,slot,arguments['planner_report_sha'],compiled['expected_model_hash'])
        root=data['roots'][key[0]];packet=root['packets']['off']
        require(chosen['arm']==key[2] and slot['blind_sha256']==data['exports'][(key[0],'off')]['sha256']
            and moving.exact(np.asarray(packet['raw_action_history']),moving.warmup_commands())
            and np.asarray(slot['root_warm_raw10'],np.float32).tobytes()==moving.warmup_commands().tobytes(),
            'all4 exact task blind/source/raw10 authority')
        warm=moving.warmup_commands()
        hist=normalization.normalized_history(warm)
        require(hist.tobytes()==np.asarray(slot['root_context']['action_history'],np.float32).tobytes(),
            'root full raw10 history normalized once')
        meta=native.allmode.root_metadata(data)[key[0]];a=raw30(chosen['commands'])
        plan=dict(schema=NAME+'.execution_plan',index=i,root_slot=key[0],condition='off',arm=key[2],
            task_id=chosen['task_id'],commands=deepcopy(chosen['commands']),raw_steps=30,
            raw_float32_sha256=hashlib.sha256(a.tobytes()).hexdigest(),planner_plan=deepcopy(chosen),
            native_goal=deepcopy(meta['goal']),native_goal_pose=deepcopy(meta['goal_pose']),
            native_goal_pixel_sha256=meta['goal_pixels']['off'],native_root_sha256=meta['root_sha256'],
            selection_uses_physics=False,P1_admission=False,P2_admission=False)
        plan['selection_digest']=digest(plan);plans.append(plan)
    return dict(schema=NAME+'.prepare',contract=contract(),arguments=deepcopy(arguments),plans=plans,bindings={},
        source_report_sha256=native.SOURCE_SHA,source_roots=native.allmode.root_metadata(data),
        source_slots=[deepcopy(r['slot']) for r in data['roots']],warm_traces=[deepcopy(r['warm_trace']) for r in data['roots']],
        model_slots=deepcopy(compiled['slots']),expected_model_hash=compiled['expected_model_hash'],
        planner_authority=dict(producer=PRODUCER,**arguments,full_upstream_rebuilt_here=False,
            evidence='pinned MAIN complete full30 receipt; native source/task saved recipe independently joined',
            plans_sha256=digest(plans),selection_uses_physics=False,missing_exports=[]))


def validate_execution(p):
    require(p['schema']==NAME+'.prepare' and p['contract']==contract() and p['arguments']==PINS
        and len(p['plans'])==4 and len(p['model_slots'])==2 and len(p['source_roots'])==len(p['source_slots'])==len(p['warm_traces'])==2,
        'own H30 full4 schema and strict pinned authority')
    require(p['source_report_sha256']==native.SOURCE_SHA and p['planner_authority']==dict(
        producer=PRODUCER,**PINS,full_upstream_rebuilt_here=False,
        evidence='pinned MAIN complete full30 receipt; native source/task saved recipe independently joined',
        plans_sha256=digest(p['plans']),selection_uses_physics=False,missing_exports=[]),'explicit inherited evidence boundary')
    for i,(plan,key) in enumerate(zip(p['plans'],ORDER,strict=True)):
        require(type(plan) is dict,'no missing sealed export or replacement')
        chosen=plan['planner_plan'];check_planner_plan(chosen,p['model_slots'][key[0]],PINS['planner_report_sha'],p['expected_model_hash'])
        root=p['source_roots'][key[0]];a=raw30(plan['commands'])
        require(plan['index']==i and type(plan['index']) is int
            and (plan['root_slot'],plan['condition'],plan['arm'])==key and chosen['arm']==key[2]
            and plan['schema']==NAME+'.execution_plan' and plan['raw_steps']==30
            and plan['selection_digest']==digest({k:v for k,v in plan.items() if k!='selection_digest'})
            and a.tobytes()==raw30(chosen['commands']).tobytes()
            and plan['raw_float32_sha256']==hashlib.sha256(a.tobytes()).hexdigest()
            and plan['task_id']==chosen['task_id'] and plan['native_goal']==root['goal']
            and plan['native_goal_pose']==root['goal_pose'] and plan['native_root_sha256']==root['root_sha256']
            and plan['native_goal_pixel_sha256']==root['goal_pixels']['off']
            and all(plan[k] is False for k in ('selection_uses_physics','P1_admission','P2_admission')),
            'exact ordered raw30 model plan and original native source')


def rehash(bindings):
    for f,h in bindings.items():
        path=Path(f)
        require(path.is_absolute() and path.is_file() and not path.is_symlink() and sha(path)==h,'immutable binding '+f)


def material(arguments):
    require(type(arguments) is dict and arguments==PINS,'only actual MAIN pinned three authorities BEFORE reads')
    rehash(CODE_PINS)
    require(sha(PROTOCOL)==PROTOCOL_SHA,'approved full30 physical protocol')
    for k in ('prepared','report','receipt'):
        path=arguments['planner_'+k] if k!='report' else str(Path(arguments['planner_output'])/'report.json')
        require(sha(path)==arguments['planner_'+('prepare' if k=='prepared' else k)+'_sha'],'actual pinned '+k)
    compiled=read(arguments['planner_prepared'],META)
    report=read(Path(arguments['planner_output'])/'report.json',META)
    consumed=read(arguments['planner_receipt'],META)
    require(report['schema']==PRODUCER+'.report' and report['errors']==[]
        and report['prepared']==arguments['planner_prepared']
        and report['prepare_sha256']==arguments['planner_prepare_sha']
        and report['model_before']==report['model_after']==compiled['expected_model_hash'],
        'pinned actual report/prepare/model unchanged join')
    data=native.source_material()
    p=assemble(arguments,compiled,consumed,data)
    bindings={}
    native.merge(bindings,compiled['bindings_sha256']);native.merge(bindings,data['bindings'])
    native.merge(bindings,CODE_PINS)
    native.merge(bindings,{arguments['planner_prepared']:arguments['planner_prepare_sha'],
        str(Path(arguments['planner_output'])/'report.json'):arguments['planner_report_sha'],
        arguments['planner_receipt']:arguments['planner_receipt_sha'],str(PROTOCOL):PROTOCOL_SHA})
    output=Path(arguments['planner_output'])
    # All producer artifacts are sealed, not only the pointers to paid evidence.
    require(set(f.name for f in output.iterdir())==set(report['artifact_sha256'])|{'report.json'},
        'complete immutable upstream artifact inventory')
    for name,h in report['artifact_sha256'].items():
        require(Path(name).name==name and name not in ('','.','..'),'flat model artifact')
        native.merge(bindings,{str(output/name):h})
    for f in set(own_files()+native.allmode.own_files()+native.base.own_files()+native.own_files()+[Path(normalization.__file__)]):
        native.merge(bindings,{str(Path(f).resolve()):sha(f)})
    env,h=native.allmode.constructor_contract();native.merge(bindings,{env:h})
    p['bindings']=bindings;rehash(bindings);validate_execution(p)
    require(len(raw(p))<=META,'bounded full prepare16MiB')
    return p,data

def write_new(path,value):
    p=Path(native.absolute(str(path)));b=raw(value)
    require(p.name.startswith(NAME+'_') and not p.exists() and not p.is_symlink() and len(b)<=META
        and shutil.disk_usage(p.parent).free>=FLOOR+len(b),'exclusive bounded own metadata')
    with p.open('xb') as f:require(f.write(b)==len(b),'complete write');f.flush();os.fsync(f.fileno())
    paired.fsync_dir(p.parent);return dict(path=str(p),sha256=sha(p),bytes=len(b))


def prepare(output,arguments):
    p,_=material(arguments);return write_new(output,p)


def verify(prepared,expected):
    native.actual_hash(expected);native.absolute(str(prepared));require(sha(prepared)==expected,'actual own prepare SHA')
    p=read(prepared,META);now,data=material(p['arguments'])
    require(raw(p)==raw(now) and sha(prepared)==expected,'complete saved planner.consume/source/own prepare reconstruction');return p,data


class Store(moving.Store):
    """Native raw/digest ABI; incremental bounded writes, no per-event directory scan."""
    def __init__(self,path):
        self.path=Path(native.absolute(str(path)))
        require(self.path.name.startswith(NAME+'_') and not self.path.exists() and not self.path.is_symlink()
            and self.path.parent.is_dir() and shutil.disk_usage(self.path.parent).free>=FLOOR+CAP,
            'new own256MiB output +200MiB floor')
        self.path.mkdir();paired.fsync_dir(self.path.parent)
        self.previous=None;self.index=0;self.sizes={};self.bytes=0;self.finalizing=False
        self.journal_hash=hashlib.sha256()
    def check(self,size=4096,final=False):
        require(type(size) is int and size>=0 and self.bytes+size<=CAP-(0 if final or self.finalizing else FINAL)
            and shutil.disk_usage(self.path).free>=FLOOR+size,'output cap/reserve/disk floor')
    def write(self,name,value,append=False,final=False):
        require(type(name) is str and Path(name).name==name and name not in ('','.','..')
            and not (self.path/name).is_symlink() and type(value) is bytes,'flat byte artifact')
        require(not append or name=='journal.jsonl','append only native journal')
        old=self.sizes.get(name,0)
        if not append:require(name not in self.sizes,'exclusive artifact, no overwrite')
        new=old+len(value)
        if name.endswith('.png'):
            require(new<=PNG_CAP and sum(n.endswith('.png') for n in self.sizes)+(name not in self.sizes)<=504,
                '504 PNG and256KiB per PNG caps')
        elif name=='journal.jsonl':require(new<=JOURNAL_CAP,'64MiB journal cap')
        else:
            require(sum(v for n,v in self.sizes.items() if n!='journal.jsonl' and not n.endswith('.png'))+len(value)<=OTHER_CAP,
                '16MiB total metadata/incident cap')
        self.check(len(value),final)
        path=self.path/name
        # Exclusive xb semantics preserved; append cannot introduce an external file.
        require(not append or path.exists()==(name in self.sizes),'owned append file identity')
        try:
            with path.open('ab' if append else 'xb') as f:
                require(f.write(value)==len(value),'short artifact write');f.flush();os.fsync(f.fileno())
        finally:
            if path.is_file():
                size=path.stat().st_size;self.bytes+=size-old;self.sizes[name]=size
        paired.fsync_dir(self.path)
        if name=='journal.jsonl':
            self.journal_hash.update(value)
            return self.journal_hash.hexdigest()
        return sha(path)
    def emit(self,kind,**fields):
        e=dict(index=self.index,previous_sha256=self.previous,kind=kind,**deepcopy(fields))
        e['event_sha256']=source.digest(e)
        self.write('journal.jsonl',moving.raw(e),append=True,final=self.finalizing)
        self.previous=e['event_sha256'];self.index+=1


class Meter:
    def __init__(self,store,started=None):
        self.store=store;self.started=time.monotonic() if started is None else started
        self.where={};self.receipts=[];self.secondary=[];self.fault=False
        self.counts={k:dict(attempted=0,completed=0) for k in contract()['caps']};self.active={k:0 for k in self.counts}
    def call(self,kind,fn,*,cleanup=False,**fields):
        require(not cleanup or kind=='close','only nonphysical close may bypass wall/fault/floor')
        if not cleanup:
            require(not self.fault and time.monotonic()-self.started<WALL,'predispatch900s/fault');self.store.check()
        c=self.counts[kind];require(c['attempted']<contract()['caps'][kind],'full30 cap before callback: '+kind)
        receipt=dict(index=len(self.receipts),resource=kind,**deepcopy(self.where),**deepcopy(fields),status='pending')
        try:self.store.emit('operation_dispatch_intent',receipt=receipt)
        except BaseException as e:
            self.fault=True
            if not cleanup:raise
            self.secondary.append(native.base.error('cleanup_intent',e))
        self.receipts.append(receipt);c['attempted']+=1;self.active[kind]+=1
        try:value=fn();c['completed']+=1;receipt['status']='returned'
        except BaseException as e:
            receipt.update(status='failed',error=native.base.error(kind,e))
            try:self.store.emit('operation_failed',receipt=receipt)
            except BaseException as extra:self.fault=True;self.secondary.append(native.base.error('failure_journal',extra))
            raise
        finally:self.active[kind]-=1
        try:self.store.emit('operation_returned',receipt=receipt)
        except BaseException as e:
            self.fault=True
            if not cleanup:raise
            self.secondary.append(native.base.error('cleanup_return',e))
        return value


def observe(env,obs,phase,step,row,meter,store,trace):
    image=native.allmode.observe(env,obs,phase,step,row,meter,store,trace)
    if phase=='plan' and step in (15,20,25,30):
        f=row['trajectory'][-1]
        row['checkpoints'].append({k:deepcopy(f[k]) for k in ('step','png','file_sha256','pixel_sha256','state','snapshot')})
        row['missing_checkpoints'].remove(step)
    return image


def episode(env,root,source_plan,plan,row,meter,store):
    body=env.unwrapped;packet=root['packets']['off'];trace=root['warm_trace']
    require(moving.exact(np.asarray(packet['raw_action_history']),moving.warmup_commands()),'original raw warm10')
    obs,goal0=place(env,source_plan,root['slot'],meter,store)
    history=[observe(env,obs,'warmup',0,row,meter,store,trace)]
    for step,command in enumerate(moving.warmup_commands(),1):
        if body.eval_state(goal0,source.vector(obs['state']))[0]:raise native.allmode.NativeStop('warmup already terminal')
        obs,info,t,tr,reward,actual=dispatch(env,command,'warmup',step,row,meter,store,goal0)
        image=observe(env,obs,'warmup',step,row,meter,store,trace);paired.check_return(env,command,actual,info,goal0,reward,t)
        if t or tr:raise native.allmode.NativeStop('native warmup early stop, no padded root')
        if step in (5,10):history.append(image)
    require(moving.exact(np.asarray(history),packet['rgb_history']),'original actual root history3 RGB')
    state=source.vector(obs['state']);goal=source.vector(root['goal'])
    require(source.root_hash(state,goal)==root['root_sha256'],'original native7 root/goal identity')
    bind_goal(body,goal,packet['goal_rgb'],meter,store);bound=invariant(body)
    row.update(initial_exact=True,root_state=state.tolist(),root_sha256=root['root_sha256'],goal=goal.tolist(),
        goal_pose=bound['goal_pose'],goal_pixel_sha256=moving.image_sha(packet['goal_rgb']))
    # Durable original native-history acceptance BEFORE the first plan action intent.
    # This certifies actual RGB/raw history; latent identity is inherited, not re-encoded here.
    store.emit('native_history_guard',**meter.where,guard=dict(
        task_id=plan['task_id'],planner_witness_sha256=plan['planner_plan']['witness_sha256'],
        root_context_sha256=K.digest(plan['planner_plan']['root_context']),
        warm_raw_float32_bits=moving.warmup_commands().tobytes().hex(),
        normalized_action_history_bits=normalization.normalized_history(moving.warmup_commands()).tobytes().hex(),
        observed_history_pixel_sha256=[moving.image_sha(x) for x in history],
        history_controls=[-10,-5,0],history_absolute_steps=[0,5,10],
        root_state=state.tolist(),root_sha256=root['root_sha256'],goal=goal.tolist(),snapshot=bound['snapshot']))
    observe(env,obs,'plan',0,row,meter,store,trace)
    if body.eval_state(goal,state)[0]:raise native.allmode.NativeStop('original root already task-terminal')
    for step,command in enumerate(raw30(plan['commands']),1):
        obs,info,t,tr,reward,actual=dispatch(env,command,'plan',step,row,meter,store,goal)
        observe(env,obs,'plan',step,row,meter,store,trace);success=paired.check_return(env,command,actual,info,goal,reward,t)
        require(moving.exact(np.asarray(info['goal']),packet['goal_rgb']) and
            {k:v for k,v in invariant(body).items() if k!='snapshot'}=={k:v for k,v in bound.items() if k!='snapshot'},
            'original task goalRGB/pose/renderer unchanged for all30')
        s=source.vector(obs['state']);delta=s-state;calc=native.metrics(goal,s)
        require(calc['native_success'] is bool(success) and calc['native7_reward']==reward,'exact original full7 native labels')
        response=dict(step=step,state=s.tolist(),delta_native7=delta.tolist(),
            delta_angle_wrapped=float(np.arctan2(np.sin(delta[4]),np.cos(delta[4]))),reward=reward,
            native_success=bool(success),terminated=t,truncated=tr,native_metrics=calc['native'])
        row['responses'].append(response);row.update(final_state=s.tolist(),final_metrics=calc['native'],
            native_success=bool(success),terminated=t,truncated=tr)
        store.emit('own_response',**meter.where,response=response)
        if t or tr:row['status']='native_stopped';return
    row['status']='horizon_completed'


def rows_for(p):
    rows=[]
    for i,key in enumerate(ORDER):
        x=p['plans'][i]
        row=dict(index=i,root_slot=key[0],condition=key[1],arm=key[2],
            task_id=p['model_slots'][key[0]]['task_id'],selection_digest=None if x is None else x['selection_digest'],
            plan=deepcopy(x),status='export_missing' if x is None else 'not_started',
            first_error=None,secondary_errors=[],warmup_attempted=0,warmup_returned=0,own_attempted=0,own_returned=0,
            PNG=0,initial_exact=False,trajectory=[],checkpoints=[],missing_checkpoints=list(CP),
            last_observation=None,responses=[],final_state=None,final_metrics=None,native_success=None,
            terminated=None,truncated=None,prefix51=dict(complete=False,frames=[]),source_unpaired_on_steps=[],
            evidence_origin='no_execution_missing_export' if x is None else 'new_native_execution_known_root')
        rows.append(row)
    return rows

def execute(p,data,store,factory,*,authorization,prepared,prepare_sha256,started=None):
    require(authorization==AUTH,'MAIN physical authorization before callbacks');validate_execution(p);native.actual_hash(prepare_sha256)
    require(p['source_slots']==[r['slot'] for r in data['roots']] and p['source_roots']==native.allmode.root_metadata(data)
        and p['warm_traces']==[r['warm_trace'] for r in data['roots']],'actual source/warmup/task identity')
    meter=Meter(store,started);rows=rows_for(p);errors=[];seen=[]
    binding=dict(prepared=native.absolute(str(prepared)),prepare_sha256=prepare_sha256,
        source_report_sha256=native.SOURCE_SHA,planner_authority=deepcopy(p['planner_authority']))
    def fail(stage,e,row=None):
        item=native.base.error(stage,e);errors.append(item)
        if row is not None:
            if row['first_error'] is None:row['first_error']=deepcopy(item)
            else:row['secondary_errors'].append(deepcopy(item))
            row['status']='engineering_failed'
    try:
        store.put('execution_binding.json',binding);store.emit('run_started',prepare_sha256=prepare_sha256,contract=contract())
        for row,plan in zip(rows,p['plans'],strict=True):
            meter.where=dict(branch=row['index'],**{k:row[k] for k in ('root_slot','condition','arm','selection_digest')})
            if plan is None:
                store.emit('branch_closed',receipt=row);continue
            env=None;row['status']='running'
            try:
                env=meter.call('constructor',factory)
                require(all(env is not x and env.unwrapped is not x.unwrapped for x in seen),'fresh native env per arm');seen.append(env)
                validate_env(env.unwrapped)
                require(env.unwrapped.render_action is False and env.unwrapped.render_size==224
                    and env.unwrapped.render_mode=='rgb_array','original native RGB224 no action marker')
                with native_hooks(env,meter,row['secondary_errors']):
                    episode(env,data['roots'][plan['root_slot']],data['source_plan'],plan,row,meter,store)
                require(not row['secondary_errors'],'native hook restoration')
            except native.allmode.NativeStop as e:row.update(status='native_stopped',stop_reason=str(e))
            except BaseException as e:fail('branch',e,row)
            finally:
                if env is not None:
                    try:meter.call('close',env.close,cleanup=True)
                    except BaseException as e:fail('close',e,row)
                if row['secondary_errors'] and row['first_error'] is None:fail('cleanup',ValueError('native cleanup errors'),row)
                for phase,label in (('warmup','warmup'),('plan','own')):
                    ops=[op for op in meter.receipts if op['resource']=='public_step' and op.get('branch')==row['index'] and op['phase']==phase]
                    row[label+'_attempted']=len(ops);row[label+'_returned']=sum(op['status']=='returned' for op in ops)
            store.emit('branch_closed',receipt=row)
    except BaseException as e:fail('execution',e)
    source_ok=False
    try:
        rehash(p['bindings'])
        source_ok=True
    except BaseException as e:fail('final_inputs',e)
    errors.extend(meter.secondary)
    if time.monotonic()-meter.started>=WALL:fail('final_wall',ValueError('900s whole-stage wall cap'))
    good=not errors and source_ok and all(x['status'] in ('horizon_completed','native_stopped','export_missing') for x in rows)
    status=RECORDED if good else 'partial_or_failed'
    store.finalizing=True
    try:store.emit('run_finished',status=status,counts=meter.counts,errors=errors)
    except BaseException as e:fail('final_journal',e);status='partial_or_failed'
    r=dict(schema=NAME+'.report',contract=contract(),status=status,execution_binding=binding,rows=rows,errors=errors,
        counts=deepcopy(meter.counts),actual_operation_receipts=deepcopy(meter.receipts),fixed_denominator=4,known_roots=2,
        fresh_roots=0,inherited=0,inputs_final_verified=source_ok,new_neural_calls=0,new_independent_samples=0,
        P1_admission=False,P2_admission=False,P2_mechanism=False,scientific_admission=False,freshness=False,
        equal_FLOPs=False,equal_total_compute=False,training=False,physical_release=False,independent_confirmation=False,
        journal_sha256=None,execution_binding_sha256=None,journal_chain_tail=store.previous,wall_seconds=time.monotonic()-meter.started)
    for n,k in (('journal.jsonl','journal_sha256'),('execution_binding.json','execution_binding_sha256')):
        try:r[k]=sha(store.path/n)
        except BaseException as e:fail('final_'+k,e);r['status']='partial_or_failed'
    try:store.put('report.json',r,final=True)
    except BaseException as e:fail('final_report',e);r['status']='partial_or_failed'
    # Completion is published only after durable report bytes and their actual hash.
    try:
        report_sha=sha(store.path/'report.json')
        elapsed=time.monotonic()-meter.started
        require(0<=elapsed<WALL,'whole stage including report fsync/hash900s')
        store.put('completion.json',dict(schema=NAME+'.completion',report_sha256=report_sha,
            journal_sha256=r['journal_sha256'],wall_seconds=elapsed,
            status=r['status'],boundary='verify start through report fsync and hash'),final=True)
        require(time.monotonic()-meter.started<WALL,'completion publication within900s')
    except BaseException as e:
        fail('completion',e);r['status']='partial_or_failed'
        try:store.put('terminal_failure.json',dict(stage='completion',errors=errors),final=True)
        except BaseException:pass  # Previous durable partial remains, never a successful completion.
    return r


def run(prepared,expected,output,factory,*,authorization):
    require(authorization==AUTH,'MAIN authorization before input reads/output/env')
    started=time.monotonic();p,data=verify(prepared,expected)
    require(time.monotonic()-started<WALL,'verify exhausted whole900s before output/env')
    store=Store(output)
    return execute(p,data,store,factory,authorization=authorization,prepared=prepared,prepare_sha256=expected,started=started)


def audit_chain(r,events):
    require(events and events[0]['kind']=='run_started' and events[0]['contract']==contract()
        and events[0]['prepare_sha256']==r['execution_binding']['prepare_sha256']
        and events[-1]['kind']=='run_finished' and all(events[-1][k]==r[k] for k in ('status','counts','errors')),
        'exact native lifecycle bookends')
    allowed={'run_started','run_finished','operation_dispatch_intent','operation_returned','operation_failed',
        'placement_recipe','goal_attached','native_history_guard','invalid_observation','fresh_observation','returned_transition','own_response','branch_closed'}
    tail=None;intents={};terminals={};stack=[];closed=0
    for i,e in enumerate(events):
        require(e['kind'] in allowed and type(e['index']) is int and e['index']==i
            and e['previous_sha256']==tail and e['event_sha256']==source.digest({k:v for k,v in e.items() if k!='event_sha256'}),
            'all event kinds/index/hash chain, unknown event rejected')
        tail=e['event_sha256']
        if e['kind']=='branch_closed':
            require(closed<4 and e['receipt']==r['rows'][closed],'ordered exact branch closure');closed+=1
        if 'branch' in e:
            require(type(e['branch']) is int and e['branch']==closed and closed<4
                and all(e[k]==r['rows'][closed][k] for k in ('root_slot','condition','arm','selection_digest')),
                'active exact full30 branch identity')
        if e['kind'] not in ('operation_dispatch_intent','operation_returned','operation_failed'):continue
        op=e['receipt'];n=op['index'];b=op['branch']
        require(type(n) is type(b) is int and b==closed and 0<=b<4 and op['resource'] in contract()['caps']
            and all(op[k]==r['rows'][b][k] for k in ('root_slot','condition','arm','selection_digest')),
            'native operation selected branch and allowed resource')
        if e['kind']=='operation_dispatch_intent':
            require(n==len(intents) and op['status']=='pending','ordered real operation intent');intents[n]=op;stack.append(n)
        else:
            require(stack and stack.pop()==n and n in intents and n not in terminals
                and op['status']==('returned' if e['kind']=='operation_returned' else 'failed')
                and dict({k:v for k,v in op.items() if k!='error'},status='pending')==intents[n],
                'nested operation return/failure exact closure');terminals[n]=op
    require(not stack and len(intents)==len(terminals) and tail==r['journal_chain_tail']
        and r['actual_operation_receipts']==[terminals[i] for i in range(len(terminals))]
        and [e['receipt'] for e in events if e['kind']=='branch_closed']==[x for x in r['rows'] if x['status']!='not_started'],
        'complete attempted resources and partial/unstarted denominator')
    counts={k:dict(attempted=0,completed=0) for k in contract()['caps']}
    for op in r['actual_operation_receipts']:
        counts[op['resource']]['attempted']+=1;counts[op['resource']]['completed']+=op['status']=='returned'
    require(counts==r['counts'] and all(v['attempted']<=contract()['caps'][k] for k,v in counts.items()),
        'all independent actual counts/caps')
    return closed,counts


def audit_labels(row,plan,scoped,p):
    root=p['source_roots'][plan['root_slot']];trace=p['warm_traces'][plan['root_slot']]
    placement=[e for e in scoped if e['kind']=='placement_recipe'];attached=[e for e in scoped if e['kind']=='goal_attached']
    warm_goal=trace[0]['pre']['goal_state']
    if placement:
        require(len(placement)==1 and placement[0]['native_goal']==warm_goal
            and placement[0]['slot']==p['source_slots'][plan['root_slot']], 'canonical seed placement/native warmup goal')
    require(len(attached)==int(row['initial_exact']),'single genuine original task binding after root')
    guards=[e for e in scoped if e['kind']=='native_history_guard']
    require(len(guards)==int(row['initial_exact']),'single durable actual native history guard')
    if guards:
        guard=guards[0]
        frames=[e for e in scoped if e['kind']=='fresh_observation' and e['receipt']['phase']=='warmup'
                and e['receipt']['overlay_index']==0 and e['receipt']['step'] in (0,5,10)]
        require(len(frames)==3 and [f['receipt']['step'] for f in frames]==[0,5,10]
            and all(f['index']<guard['index'] for f in frames)
            and attached and attached[0]['index']<guard['index'],'actual full history and bind precede acceptance')
        require(guard['guard']==dict(task_id=plan['task_id'],
            planner_witness_sha256=plan['planner_plan']['witness_sha256'],
            root_context_sha256=K.digest(plan['planner_plan']['root_context']),
            warm_raw_float32_bits=moving.warmup_commands().tobytes().hex(),
            normalized_action_history_bits=normalization.normalized_history(moving.warmup_commands()).tobytes().hex(),
            observed_history_pixel_sha256=[f['receipt']['pixel_sha256'] for f in frames],
            history_controls=[-10,-5,0],history_absolute_steps=[0,5,10],
            root_state=root['root_state'],root_sha256=root['root_sha256'],goal=root['goal'],
            snapshot=frames[-1]['receipt']['snapshot']), 'complete original native RGB/raw/context history guard')
        require(all(guard['index']<e['index'] for e in scoped if
            (e['kind']=='operation_dispatch_intent' and e['receipt']['resource']=='public_step'
             and e['receipt'].get('phase')=='plan') or
            (e['kind']=='fresh_observation' and e['receipt']['phase']=='plan')),
            'durable native history guard before first plan action intent and every plan observation')
    if attached:
        g=attached[0]
        require(g['goal']==root['goal'] and g['goal_pose']==root['goal_pose']
            and g['goal_rgb_sha256']==root['goal_pixels']['off'],'canonical target, not arbitrary phase goal')
        intent=[e for e in scoped if e['kind']=='operation_dispatch_intent' and e['receipt']['resource']=='public_step'
            and e['receipt'].get('phase')=='plan']
        require(all(g['index']<e['index'] for e in intent),'canonical task bound before every plan dispatch intent')
    for e in scoped:
        if e['kind']=='fresh_observation' and e['receipt']['phase']=='plan':
            require(attached and attached[0]['index']<e['index'],'goal bind before plan0 and every RGB')
        if e['kind']!='returned_transition':continue
        phase=e['phase'];require(phase in ('warmup','plan') and len(placement)==1 and placement[0]['index']<e['index'],
            'all public returns after original placement')
        if phase=='warmup':
            require(not attached or e['index']<attached[0]['index'],'warmup must precede task binding');goal=warm_goal
        else:
            require(attached and attached[0]['index']<e['index'],'task attached before every raw30 return');goal=root['goal']
        require(e['goal_state']==goal and e['state']==e['snapshot']['state'],'all full7 return goal/state identities')
        calc=native.metrics(goal,e['state'])
        require(type(e['reward']) is float and e['reward']==calc['native7_reward']
            and type(e['terminated']) is bool and e['terminated'] is calc['native_success'] and type(e['truncated']) is bool,
            'all warmup/plan native7 reward and original success/termination')


def validate_saved(r,events,p):
    validate_execution(p)
    require(r['schema']==NAME+'.report' and r['contract']==contract() and r['status'] in (RECORDED,'partial_or_failed')
        and len(r['rows'])==r['fixed_denominator']==4 and r['known_roots']==2 and r['fresh_roots']==r['inherited']==0
        and r['execution_binding']['planner_authority']==p['planner_authority']
        and r['execution_binding']['source_report_sha256']==native.SOURCE_SHA,'own full30 native schema/authority/all4')
    require(all(r[k] is False for k in ('P1_admission','P2_admission','P2_mechanism','scientific_admission','freshness',
        'equal_FLOPs','equal_total_compute','training','physical_release','independent_confirmation'))
        and r['new_neural_calls']==r['new_independent_samples']==0,'engineering only, no science/fresh/equalcompute')
    closed,counts=audit_chain(r,events)
    for i,row in enumerate(r['rows']):
        plan=p['plans'][i]
        if plan is None:
            require(row==rows_for(p)[i] and not any(e.get('branch')==i for e in events)
                and not any(o['branch']==i for o in r['actual_operation_receipts']),
                'missing export retained unchanged, no constructor/actions/replacement')
            continue
        root=p['source_roots'][plan['root_slot']]
        require(type(row['index']) is int and row['index']==i and row['plan']==plan
            and (row['root_slot'],row['condition'],row['arm'])==ORDER[i]
            and all(row[k]==plan[k] for k in ('task_id','selection_digest'))
            and row['status'] in ('not_started','engineering_failed','native_stopped','horizon_completed')
            and row['evidence_origin']=='new_native_execution_known_root','fixed whole original raw30 branches, no inheritance')
        scoped=[e for e in events if e.get('branch')==i];ops=[o for o in r['actual_operation_receipts'] if o['branch']==i]
        public=[o for o in ops if o['resource']=='public_step'];transitions=[e for e in scoped if e['kind']=='returned_transition']
        observations=[e for e in scoped if e['kind']=='fresh_observation'];audit_labels(row,plan,scoped,p)
        for phase,label,commands,limit in (('warmup','warmup',moving.warmup_commands(),10),('plan','own',raw30(plan['commands']),30)):
            current=[o for o in public if o['phase']==phase]
            require(len(current)<=limit and [o['step'] for o in current]==list(range(1,len(current)+1))
                and all(o['raw_bits']==commands[j].tobytes().hex() for j,o in enumerate(current))
                and row[label+'_attempted']==len(current) and row[label+'_returned']==sum(o['status']=='returned' for o in current),
                'complete contiguous warm10/raw30 authority, no padding/change/duplicate')
            returned=[e for e in transitions if e['phase']==phase]
            require([e['step'] for e in returned]==list(range(1,len(returned)+1)) and len(returned)<=row[label+'_returned'],
                'actual return chronology and missing incidence')
            for e in returned:
                require(e['actual_dtype']==np.dtype(np.float32).str and e['actual_bits']==e['requested_bits']==commands[e['step']-1].tobytes().hex(),
                    'all actual returned literal float32 action bits')
            if row['status'] in ('horizon_completed','native_stopped'):
                require(len(returned)==row[label+'_returned'],'every completed native public return has full labels')
        require(row['PNG']==len(observations),'all rendered observations counted')
        bridge=[{k:deepcopy(e[k]) for k in ('receipt','pre','post')} for e in observations
            if e['receipt']['phase']=='warmup' or e['receipt']['step']<=5]
        require(row['prefix51']==dict(complete=len(bridge)==51,frames=bridge),'real source warmup and own prefix, not imagined context restore')
        unknown=[];selected=[]
        for e in observations:
            f=e['receipt'];phase,step,overlay=f['phase'],f['step'],f['overlay_index']
            require(phase in ('warmup','plan') and type(step) is type(overlay) is int
                and 0<=step<=(10 if phase=='warmup' else 30) and overlay in (0,1,2)
                and f['condition']==('on' if overlay==1 else 'off') and e['pre']==e['post']
                and f['snapshot']==e['post']['snapshot'] and f['state']==f['snapshot']['state'] and len(f['state'])==7
                and e['pre']['renderer']==root['renderer'] and e['pre']['goal_pose']==root['goal_pose'],
                'all actual RGB native7 snapshot/renderer/pose exact')
            if overlay==0:selected.append(f)
            if phase=='warmup':
                trace=p['warm_traces'][plan['root_slot']]
                require(e['pre']['goal_state']==trace[0]['pre']['goal_state']
                    and e['pre']['goal_rgb_sha256']==trace[0]['pre']['goal_rgb_sha256'],
                    'all warmup renderer goals precede final task binding')
                ref=native.allmode.reference_for(p['warm_traces'][plan['root_slot']],step,overlay)
                if ref is None:unknown.append(step)
                else:native.allmode.check_warm_frame(e,ref)
            else:
                require(e['pre']['goal_state']==root['goal'] and e['pre']['goal_rgb_sha256']==root['goal_pixels']['off'],
                    'unchanged original off goalRGB and native goal')
                if step:
                    t=next(t for t in transitions if t['phase']=='plan' and t['step']==step)
                    require(f['state']==t['state'] and f['snapshot']==t['snapshot'],'same time RGB/native return')
        require(row['trajectory']==selected and row['source_unpaired_on_steps']==unknown,'all selected off frames, no fabricated old on references')
        if row['initial_exact']:
            require(all(row[k]==root[k] for k in ('root_state','root_sha256','goal','goal_pose'))
                and row['goal_pixel_sha256']==root['goal_pixels']['off']
                and source.root_hash(source.vector(row['root_state']),source.vector(row['goal']))==row['root_sha256'],
                'exact original root/source/goal identity after warm10')
        responses=row['responses']
        require([x['step'] for x in responses]==list(range(1,len(responses)+1)) and len(responses)<=row['own_returned']
            and [e['response'] for e in scoped if e['kind']=='own_response']==responses,'all actual full30 response labels retained')
        for response in responses:
            calc=native.metrics(row['goal'],response['state']);delta=source.vector(response['state'])-source.vector(row['root_state'])
            require(response['native_metrics']==calc['native'] and response['reward']==calc['native7_reward']
                and response['native_success'] is response['terminated'] is calc['native_success']
                and type(response['truncated']) is bool and np.array_equal(delta,response['delta_native7'])
                and response['delta_angle_wrapped']==float(np.arctan2(np.sin(delta[4]),np.cos(delta[4]))),
                'original full7 reward/native success/metrics, diagnostic angle only')
            t=next(t for t in transitions if t['phase']=='plan' and t['step']==response['step'])
            require(all(t[k]==response[k] for k in ('state','reward','terminated','truncated')),'response joins exact public return')
        if responses:
            last=responses[-1]
            require(row['final_state']==last['state'] and row['final_metrics']==last['native_metrics']
                and all(row[k] is last[k] for k in ('native_success','terminated','truncated'))
                and all(not x['terminated'] and not x['truncated'] for x in responses[:-1]),'actual endpoint/stop, no post-terminal padding')
        else:require(row['final_state'] is row['final_metrics'] is row['native_success'] is None,'missing endpoints unknown, not zero benefit')
        cps=[f for f in selected if f['phase']=='plan' and f['step'] in CP]
        require(row['checkpoints']==[{k:f[k] for k in ('step','png','file_sha256','pixel_sha256','state','snapshot')} for f in cps]
            and row['missing_checkpoints']==[b for b in CP if b not in [f['step'] for f in cps]],'cp0/5/10/15/20/25/30 exact and explicit missing')
        if row['status'] in ('horizon_completed','native_stopped'):
            require(row['first_error'] is None and row['secondary_errors']==[],'no engineering failure relabelled as native stop')
            keys=[('warmup',b,k) for b in range(row['warmup_returned']+1) for k in (0,1,2)]
            if row['initial_exact']:keys += [('plan',b,k) for b in range(row['own_returned']+1) for k in (0,1,2)]
            require([(e['receipt']['phase'],e['receipt']['step'],e['receipt']['overlay_index']) for e in observations]==keys,
                'all real unpadded full30 off/on/off triplets')
            for j in range(0,len(observations),3):
                a,b,c=observations[j:j+3]
                require(a['receipt']['file_sha256']==c['receipt']['file_sha256'] and a['receipt']['pixel_sha256']==c['receipt']['pixel_sha256']
                    and a['receipt']['state']==b['receipt']['state']==c['receipt']['state'],'off/on/off closes without dynamics change')
            if responses:
                require(row['warmup_returned']==10 and row['initial_exact'] and row['own_returned']==len(responses)
                    and ((row['status']=='native_stopped')==(last['terminated'] or last['truncated'])),'native stop retained without padding')
            require(len(responses)==row['own_returned'],'all successful raw30 returns have own responses')
            if row['status']=='native_stopped' and not responses:
                goal=root['goal'] if row['initial_exact'] else p['warm_traces'][plan['root_slot']][0]['pre']['goal_state']
                final_observation=observations[-1]['receipt'] if observations else None
                require(final_observation is not None and (native.metrics(goal,final_observation['state'])['native_success']
                    or bool(transitions and (transitions[-1]['terminated'] or transitions[-1]['truncated']))),
                    'zero-own native stop requires authenticated actual terminal/truncation, not arbitrary flag')
            if row['status']=='horizon_completed':require(row['own_returned']==30,'actual full30, not full10')
            kinds={k:sum(o['resource']==k and o['status']=='returned' for o in ops) for k in contract()['caps']}
            require(kinds['constructor']==kinds['close']==1 and kinds['reset']==2
                and kinds['set_state']==4 and kinds['goal_render']==2 and kinds['goal_bind']==int(row['initial_exact'])
                and kinds['goal_setter']==2+int(row['initial_exact'])
                and kinds['physics_step']==4+10*kinds['public_step'] and kinds['explicit_render']==row['PNG'],
                'per-branch actual full native lifecycle census')
    if r['status']==RECORDED:
        require(closed==4 and r['errors']==[] and r['inputs_final_verified'] is True and type(r['wall_seconds']) in (int,float) and np.isfinite(r['wall_seconds']) and 0<=r['wall_seconds']<WALL
            and all(x['status'] in ('horizon_completed','native_stopped','export_missing') for x in r['rows']),'complete engineering four only')
        c={k:v['completed'] for k,v in counts.items()}
        require(all(v['attempted']==v['completed'] for v in counts.values()) and c['constructor']==c['close']==sum(x is not None for x in p['plans']) and c['reset']==2*c['constructor']
            and c['goal_bind']==sum(x['initial_exact'] for x in r['rows']) and c['set_state']==2*c['reset']
            and c['goal_render']==c['reset'] and c['goal_setter']==c['reset']+c['goal_bind']
            and c['physics_step']==2*c['reset']+10*c['public_step']
            and c['public_step']==sum(x['warmup_returned']+x['own_returned'] for x in r['rows'])
            and c['explicit_render']==sum(x['PNG'] for x in r['rows']),'complete actual native lifecycle resource census')
    return deepcopy(r['rows'])


def byte_census(files):
    require(type(files) is dict and all(type(v) is bytes for v in files.values()),'saved bytes')
    png=[v for n,v in files.items() if n.endswith('.png')]
    journal=files.get('journal.jsonl',b'')
    other=sum(len(v) for n,v in files.items() if n!='journal.jsonl' and not n.endswith('.png'))
    require(len(png)<=504 and all(len(v)<=PNG_CAP for v in png)
        and len(journal)<=JOURNAL_CAP and other<=OTHER_CAP and sum(map(len,files.values()))<=CAP,
        'full metadata/PNG/journal budget proof')
    return dict(PNG=len(png),PNG_bytes=sum(map(len,png)),journal_bytes=len(journal),other_bytes=other,
                total_bytes=sum(map(len,files.values())))


def saved_artifacts(directory):
    files=list(Path(directory).iterdir())
    require(all(f.is_file() and not f.is_symlink() for f in files) and sum(f.stat().st_size for f in files)<=CAP,
        'own256MiB bounded regular artifact census')
    result={f.name:f.read_bytes() for f in files}
    byte_census(result);return result


def consume(output,expected):
    native.actual_hash(expected);directory=Path(native.absolute(str(output)))
    files=saved_artifacts(directory)
    require(sum(map(len,files.values()))<=CAP and hashlib.sha256(files['report.json']).hexdigest()==expected,'actual own report SHA/256MiB')
    r=native.decode(files['report.json']);b=r['execution_binding'];p,_=verify(b['prepared'],b['prepare_sha256'])
    require(hashlib.sha256(files['execution_binding.json']).hexdigest()==r['execution_binding_sha256']
        and native.decode(files['execution_binding.json'])==b,'actual execution binding bytes')
    journal=files['journal.jsonl']
    require(journal.endswith(b'\n') and hashlib.sha256(journal).hexdigest()==r['journal_sha256'],'complete saved journal SHA')
    require(b'\n\n' not in journal and b'\r' not in journal,'native singleLF records no empty lines')
    events=[native.decode(line) for line in journal.splitlines()];rows=validate_saved(r,events,p)
    completion=native.decode(files['completion.json'])
    require(set(completion)=={'schema','report_sha256','journal_sha256','wall_seconds','status','boundary'}
        and completion['schema']==NAME+'.completion' and completion['report_sha256']==expected
        and completion['journal_sha256']==r['journal_sha256'] and completion['status']==r['status']
        and completion['boundary']=='verify start through report fsync and hash'
        and type(completion['wall_seconds']) in (int,float) and np.isfinite(completion['wall_seconds'])
        and r['wall_seconds']<=completion['wall_seconds']<WALL,'completion-bound whole wall/report hash')
    byte_census(files)
    from PIL import Image
    names={'report.json','execution_binding.json','journal.jsonl','completion.json'};n=0
    for e in events:
        if e['kind']=='invalid_observation':
            name=e['file'];require(Path(name).name==name and name not in names
                and hashlib.sha256(files[name]).hexdigest()==e['file_sha256'],'retained invalid-render incident');names.add(name)
        if e['kind']!='fresh_observation':continue
        f=e['receipt'];name=f['png']
        require(Path(name).name==name and name not in ('','.','..') and name not in names
            and len(files[name])<=PNG_CAP and hashlib.sha256(files[name]).hexdigest()==f['file_sha256'],'unique native PNG byte seal')
        with Image.open(io.BytesIO(files[name])) as im:
            require(im.format=='PNG' and im.mode=='RGB' and im.size==(224,224),'native RGB224 PNG')
            x=np.asarray(im).copy()
        require(moving.image_sha(x)==f['pixel_sha256'],'exact saved native RGB pixels');moving.guard(x,source.vector(f['state']))
        names.add(name);n+=1
    require(set(files)==names,'all saved artifacts, no unknown extra files')
    bindings=deepcopy(p['bindings']);native.merge(bindings,{b['prepared']:b['prepare_sha256']})
    native.merge(bindings,{str(directory/name):hashlib.sha256(body).hexdigest() for name,body in files.items()})
    for f,h in p['bindings'].items():require(sha(f)==h,'final complete source/producer/task seals')
    require(saved_artifacts(directory)==files,'saved bytes unchanged after strict consume')
    return dict(schema=NAME+'.consume',status=r['status'],rows=rows,counts=r['counts'],errors=r['errors'],
        fixed_denominator=4,known_roots=2,fresh_roots=0,inherited=0,saved_PNG_verified=n,
        full_upstream_rebuilt_here=False,own_native_saved_reconstructed_here=True,byte_census=byte_census(files),
        prepare_sha256=b['prepare_sha256'],completion=completion,
        physical_report_sha256=expected,planner_authority=p['planner_authority'],bindings=bindings,
        plans=deepcopy(p['plans']),missing_exports=deepcopy(p['planner_authority']['missing_exports']),
        all_four_physically_executed=all(x['own_returned']==30 for x in rows),
        metric_breakdown=[dict(root_slot=x['root_slot'],arm=x['arm'],
            metrics=None if x['final_state'] is None else native.metrics(x['goal'],x['final_state'])) for x in rows],
        P1_admission=False,P2_admission=False,P2_mechanism=False,scientific_admission=False,freshness=False,
        equal_FLOPs=False,equal_total_compute=False,independent_confirmation=False,physical_release=False,new_neural_calls=0,new_physics_calls=0)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('output','prepared','prepare-sha','report-sha','receipt','authorization',
              'planner-prepared','planner-prepare-sha','planner-output','planner-report-sha','planner-receipt','planner-receipt-sha'):
        ap.add_argument('--'+k)
    a=ap.parse_args()
    if a.command=='prepare':
        args={k:(getattr(a,k) or PINS[k]) for k in ARGS};result=prepare(a.output,args)
    elif a.command=='verify':p,_=verify(a.prepared,a.prepare_sha);result=dict(verified=True,plans=len(p['plans']))
    elif a.command=='run':
        require(a.authorization==AUTH,'MAIN explicit native authority before gym import')
        def factory():
            import gymnasium as gym
            import stable_worldmodel.envs  # registration, only MAIN production run
            return gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224,with_target=False,render_action=False)
        r=run(a.prepared,a.prepare_sha,a.output,factory,authorization=a.authorization)
        result=dict(output=a.output,status=r['status'],errors=r['errors'],report_sha256=sha(Path(a.output)/'report.json'))
    else:
        r=consume(a.output,a.report_sha)
        result=write_new(a.receipt,r) if a.receipt else {k:v for k,v in r.items() if k not in ('rows','bindings')}
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()


