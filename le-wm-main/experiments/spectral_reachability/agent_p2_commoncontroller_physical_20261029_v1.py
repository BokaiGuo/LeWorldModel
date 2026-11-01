"""Fixed six-slot commoncontroller native full15 replay; MAIN physics only.

Own schema and causal first-plan-step guard. Reuses native kernel interfaces,
never fabricates a three-program plan, loads a model, or selects by outcomes.
"""
from __future__ import annotations
import argparse
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
from experiments.spectral_reachability import agent_p2_three_program_physical_20261029_v1 as kernel

NAME='agent_p2_commoncontroller_physical_20261029_v1'
AUTH='MAIN_COMMONCONTROLLER4_FULL15_PHYSICS_ONLY'
RECORDED='commoncontroller4_full15_native_recorded_not_admitted'
PAID='agent_p2_delivered_mean_paid_replay_cuda_20261029_v1'
PAID_SOURCE_SHA='19b8b36337de68233e504a647b66b468e0fde95608a655e8e57803886b938481'
PROTOCOL_SHA='42e082498484f8bb399f819d31531c2a77f73b161e0bd2e15bb8ec900f5e8de6'
PROTOCOL=kernel.ROOT/'innerdocs/agent_p2_commoncontroller_physical_protocol_20261029_v1.md'
ORIGINAL=kernel.ROOT/'results/agent_p2_three_program_physical_20261029_v1_prepare_MAIN01.json'
ORIGINAL_SHA='a4fb18b5aedd53e1820717f827d4703774b8c41dd08b0fd3974cca58593ade93'
KERNEL_SHA='b0e7ae15d4810963a64916b1da16e34ae1017b189156a93b635d73520e2666e5'
ROOT=kernel.ROOT
native=kernel.native
source,moving,paired=native.source,native.moving,native.paired
require,sha,raw,read=native.require,native.sha,native.raw,native.read
digest=native.core.digest  # Native own-schema spaced JSON, NOT paid compact codec.
raw15=kernel.raw15
CP=kernel.CP
CAP,FINAL,FLOOR,WALL,META=kernel.CAP,kernel.FINAL,kernel.FLOOR,kernel.WALL,kernel.META
ACTIVE=(0,1,3,4)
ARMS=('prior_FIRST','program_first_postcondition','arbitrary')
ORDER=[(i//3,'off',ARMS[i%3]) for i in range(6)]
ARGS={'replay_manifest','replay_manifest_sha','replay_report','replay_report_sha',
      'replay_receipt','replay_receipt_sha'}
place,bind_goal,dispatch=native.place,native.bind_goal,native.dispatch
validate_env,native_hooks,invariant=native.validate_env,native.native_hooks,native.invariant
observe=kernel.observe
audit_labels=kernel.audit_labels


def byte_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def same32(value,expected,label):
    a=np.asarray(value,dtype=np.float32)
    require(a.shape==expected.shape and a.tobytes()==expected.tobytes(),label)
    return a


def contract():
    c=deepcopy(kernel.contract())
    c.update(schema=NAME+'.contract',producer=PAID,protocol_sha256=PROTOCOL_SHA,
        fixed_denominator=6,active_slots=list(ACTIVE),missing_slots=[2,5],
        order=[list(x) for x in ORDER],arms=list(ARMS),arm='commoncontroller_returned_mean_raw15',
        raw_authority='complete actual-token paid replay, sealed original raw15 bytes',
        scientific_authority='explicit MAIN engineering AUTH only, upstream raw_deployment_authorized stays false',
        first_step_guard='durable original warm10/history3/current RGB/task goal identity before plan step1 intent',
        prefix51='original warmup0..10 plus actual own plan0..5; not an imagined restore',
        missing_export='two arbitrary missing; all four paid replays required before ANY env',
        interpretation='two known off roots, six target slots/four native branches; not fresh/equal budget or P2 science')
    return c


def own_files():
    return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def paid_complete(receipt,args):
    require(type(receipt) is dict and receipt['status']=='all_saved_paid_replays_verified_with_explicit_blocked_slots'
        and receipt['manifest_sha256']==args['replay_manifest_sha']
        and receipt['report_sha256']==args['replay_report_sha']
        and receipt['denominator']==dict(total=6,returned=4,blocked=0,missing=2)
        and receipt['predict_API']==receipt['predictor_samples']==24
        and receipt['module_resources']==dict(action_encoder=24,low_projection=24,AE_tokens=96,
            predictor_context_tokens=72,projection_rows=72)
        and all(receipt[k] is False for k in ('original_Hi_scaler_certified','raw_deployment_authorized','scientific_admission')),
        'actual complete paid replay ABI, no invented schema/flags or deployment authorization')
    require([x['slot'] for x in receipt['local_checks']]==list(ACTIVE)
        and all(x['blocked'] is None and x['predict_API']==x['predictor_samples']==6
            and x['action_encoder']==x['low_projection']==6 and x['AE_tokens']==24
            and x['predictor_context_tokens']==x['projection_rows']==18 for x in receipt['local_checks']),
        'all four complete actual-token replay checks, never partial subset')


def check_plan(plan,i,p):
    key=ORDER[i];root=p['source_roots'][key[0]]
    require(plan['schema']==NAME+'.execution_plan' and type(plan['index']) is int and plan['index']==i
        and (plan['root_slot'],plan['condition'],plan['arm'])==key
        and plan['selection_digest']==digest({k:v for k,v in plan.items() if k!='selection_digest'})
        and all(plan[k] is False for k in ('selection_uses_physics','P1_admission','P2_admission')),
        'own fixed six-slot task and distinct native spaced seal')
    if i not in ACTIVE:
        require(plan['status']=='missing' and plan['commands'] is None and plan['raw_steps']==0
            and plan['raw_float32_sha256'] is None and plan['controller_payload'] is None,
            'arbitrary remains missing, never padded or assigned target')
        return
    a=raw15(plan['commands'])
    payload=plan['controller_payload']
    require(plan['status']=='sealed_raw15' and plan['raw_steps']==15 and byte_sha(a)==plan['raw_float32_sha256']
        and a.tobytes().hex()==plan['raw_float32_bits'] and set(payload)=={'z','a','target'}
        and plan['native_goal']==root['goal'] and plan['native_goal_pose']==root['goal_pose']
        and plan['native_goal_pixel_sha256']==root['goal_pixels']['off']
        and plan['native_root_sha256']==root['root_sha256'],
        'original sealed raw15/native task, target never becomes native goal')
    for k,shape in (('z',(1,3,192)),('a',(1,2,10)),('target',(1,192))):
        value=np.asarray(payload[k],np.float32)
        require(value.shape==shape and np.isfinite(value).all()
            and byte_sha(value)==plan['payload_sha256'][k],'exact source-bound input '+k)
    mean=np.frombuffer(bytes.fromhex(plan['mean_bits']),'<f4')
    std=np.frombuffer(bytes.fromhex(plan['std_bits']),'<f4')
    require(mean.shape==std.shape==(2,) and np.isfinite(mean).all() and np.isfinite(std).all() and (std>0).all(),
        'original finite positive public float32 normalizer')
    history=moving.warmup_commands()
    token=np.divide(np.subtract(history,mean,dtype=np.float32),std,dtype=np.float32).reshape(1,2,10)
    same32(payload['a'],token,'original once-normalized two-token warm history; no /100')
    require(history.tobytes().hex()==plan['warm_raw10_bits']
        and len(plan['history_pixel_sha256'])==3 and plan['goal_pixel_sha256']==root['goal_pixels']['off'],
        'original real history/goal pixel bindings')


def assemble(args,manifest,report,receipt,original,data,arrays,raw_descriptors):
    paid_complete(receipt,args)
    require(original['schema']==kernel.NAME+'.prepare'
        and original['source_report_sha256']==native.SOURCE_SHA
        and original['source_roots']==native.allmode.root_metadata(data)
        and original['source_slots']==[r['slot'] for r in data['roots']]
        and original['warm_traces']==[r['warm_trace'] for r in data['roots']],
        'whole original physical task/source metadata, no new root placement')
    require(report['schema']==PAID+'.report' and report['completed'] is True and report['errors']==[]
        and report['manifest_sha256']==args['replay_manifest_sha']
        and manifest['schema']==PAID+'.prepare' and len(report['rows'])==6,
        'actual paid report/prepare schema and complete six-slot denominator')
    delivery=manifest['controller_prepare']['delivery_artifact']
    require(len(delivery['rows'])==6 and delivery['denominator']==dict(roots=2,arms=3,total=6,assembled=4,missing=2,dispatched=0),
        'actual delivery has six rows, positional identity not invented slot field')
    plans=[]
    for i,key in enumerate(ORDER):
        d=delivery['rows'][i];r=report['rows'][i];root=data['roots'][key[0]]
        require((d['root'],d['condition'],d['arm'])==key
            and (r['slot'],r['root'],r['arm'])==(i,key[0],key[2]),'exact delivery/report slot order')
        s=original['plans'][2*key[0]+1]['planner_plan']
        require(all(d[k]==s[k] for k in ('task_id','task_sha256','blind_sha256')),
            'original program task/source/blind identity, never outcome join')
        plan=dict(schema=NAME+'.execution_plan',index=i,root_slot=key[0],condition='off',arm=key[2],
            task_id=d['task_id'],task_sha256=d['task_sha256'],blind_sha256=d['blind_sha256'],
            status='missing',commands=None,raw_steps=0,raw_float32_sha256=None,controller_payload=None,
            selection_uses_physics=False,P1_admission=False,P2_admission=False)
        if i in ACTIVE:
            require(r['status']=='returned' and d['status']=='assembled_not_dispatched',
                'all four returned means, no subset or new selection')
            a,z,past,target,mean,std=arrays[i]
            raw15(a.tolist())
            require(a.dtype==np.float32 and a.shape==(15,2)
                and a.tobytes().hex()==manifest['raw_artifact']['rows'][i]['checks']['array_bits']['raw'],
                'actual CUDA raw identical to original sealed raw 120 bytes')
            same32(d['controller_payload']['z'],z,'paid z matches delivered z')
            same32(d['controller_payload']['a'],past,'paid past a matches delivered a')
            same32(d['controller_payload']['target'],target,'paid target matches delivered target')
            same32(s['root_context']['latent_history'],z[0],'original program context same coordinate')
            same32(s['root_context']['action_history'],past[0],'original program past tokens')
            packet=root['packets']['off']
            meta=original['source_roots'][key[0]]
            plan.update(status='sealed_raw15',commands=a.tolist(),raw_steps=15,raw_float32_sha256=byte_sha(a),
                raw_float32_bits=a.tobytes().hex(),controller_payload=deepcopy(d['controller_payload']),
                payload_sha256={k:byte_sha(v) for k,v in (('z',z),('a',past),('target',target))},
                mean_bits=mean.tobytes().hex(),std_bits=std.tobytes().hex(),
                warm_raw10_bits=moving.warmup_commands().tobytes().hex(),
                history_pixel_sha256=[moving.image_sha(f) for f in packet['rgb_history']],
                goal_pixel_sha256=moving.image_sha(packet['goal_rgb']),
                native_goal=deepcopy(meta['goal']),native_goal_pose=deepcopy(meta['goal_pose']),
                native_goal_pixel_sha256=meta['goal_pixels']['off'],native_root_sha256=meta['root_sha256'],
                paid_authority=dict(manifest_sha256=args['replay_manifest_sha'],report_sha256=args['replay_report_sha'],
                    receipt_sha256=args['replay_receipt_sha'],slot=i,raw_descriptor=deepcopy(raw_descriptors[i])),
                target_origin=deepcopy(d['origin']))
        else:
            require(r['status']==d['status']=='missing' and d['controller_payload'] is None and i not in arrays,
                'arbitrary absent through all authorities')
        plan['selection_digest']=digest(plan);plans.append(plan)
    p=dict(schema=NAME+'.prepare',arguments=deepcopy(args),contract=contract(),plans=plans,bindings={},
        source_report_sha256=native.SOURCE_SHA,source_roots=deepcopy(original['source_roots']),
        source_slots=deepcopy(original['source_slots']),warm_traces=deepcopy(original['warm_traces']),
        planner_authority=dict(producer=PAID,**args,protocol_sha256=PROTOCOL_SHA,
            complete_paid_consume_rebuilt=True,persisted_receipt_equal=True,receipt=deepcopy(receipt),
            original_physical_prepare=dict(path=str(ORIGINAL),sha256=ORIGINAL_SHA),
            original_prediction_cost_reuse_authorized=False,selection_uses_physics=False))
    validate_execution(p)
    return p


def validate_execution(p):
    require(p['schema']==NAME+'.prepare' and p['contract']==contract() and len(p['plans'])==6
        and p['source_report_sha256']==native.SOURCE_SHA
        and len(p['source_roots'])==len(p['source_slots'])==len(p['warm_traces'])==2 and set(p['arguments'])==ARGS,
        'own complete six-slot prepare/contract')
    authority=p['planner_authority'];args=p['arguments']
    require(authority==dict(producer=PAID,**args,protocol_sha256=PROTOCOL_SHA,
        complete_paid_consume_rebuilt=True,persisted_receipt_equal=True,receipt=authority['receipt'],
        original_physical_prepare=dict(path=str(ORIGINAL),sha256=ORIGINAL_SHA),
        original_prediction_cost_reuse_authorized=False,selection_uses_physics=False),'whole actual paid replay authority')
    paid_complete(authority['receipt'],args)
    for i,plan in enumerate(p['plans']):check_plan(plan,i,p)


def material(args):
    require(type(args) is dict and set(args)==ARGS,'explicit actual paid paths/SHA')
    for k,v in args.items():
        if k.endswith('_sha'):native.actual_hash(v)
        else:native.absolute(v)
    code=ROOT/'experiments/spectral_reachability'/(PAID+'.py')
    require(sha(code)==PAID_SOURCE_SHA and sha(kernel.__file__)==KERNEL_SHA and sha(PROTOCOL)==PROTOCOL_SHA,
        'fixed approved protocol/native/paid source before saved consume')
    paid=importlib.import_module('experiments.spectral_reachability.'+PAID)
    manifest=paid.verify(args['replay_manifest'],args['replay_manifest_sha'])
    report=paid.read(args['replay_report'],args['replay_report_sha']);folder=Path(args['replay_report']).parent
    require(Path(report['manifest']).resolve()==Path(args['replay_manifest'])
        and report['manifest_sha256']==args['replay_manifest_sha'],'actual report points at explicit manifest')
    paid.validate_files(report,folder)
    receipt=paid.analyze_saved(manifest,report,folder)
    receipt.update(manifest_sha256=args['replay_manifest_sha'],report_sha256=args['replay_report_sha'])
    require(raw(paid.read(args['replay_receipt'],args['replay_receipt_sha']))==raw(receipt),
        'complete paid strict rebuilt without writing or inventing receipt flags')
    paid_complete(receipt,args)
    original=read(ORIGINAL,META);require(sha(ORIGINAL)==ORIGINAL_SHA,'original whole physical prepare SHA')
    data=native.source_material()
    arrays={};raw_descriptors={}
    for i in ACTIVE:
        row=report['rows'][i];trace=paid.read(folder/row['trace']['file'],row['trace']['file_sha256'])
        # Actual CUDA add output, not original normalized mean or regenerated raw.
        desc=row['arrays'][trace['arithmetic'][1]['output']]
        raw_descriptors[i]=deepcopy(desc)
        load=lambda d:paid.controller.load_array(folder,d)
        arrays[i]=(load(desc),*(load(row['inputs'][k]) for k in ('z','a','target','mean_xy','std')))
    p=assemble(args,manifest,report,receipt,original,data,arrays,raw_descriptors)
    bindings=deepcopy(data['bindings']);native.merge(bindings,manifest['sources']);native.merge(bindings,original['bindings'])
    native.merge(bindings,{str(ORIGINAL):ORIGINAL_SHA,str(PROTOCOL):PROTOCOL_SHA,str(code):PAID_SOURCE_SHA,
        str(Path(kernel.__file__).resolve()):KERNEL_SHA,**{args[k]:args[k+'_sha'] for k in ('replay_manifest','replay_report','replay_receipt')}})
    for f in folder.iterdir():native.merge(bindings,{str(f.resolve()):sha(f)})
    for f in own_files():native.merge(bindings,{str(f):sha(f)})
    p['bindings']=bindings
    for f,h in bindings.items():require(sha(f)==h,'complete final input/model/source seal')
    require(len(raw(p))<=META,'whole128MiB prepare cap')
    return p,data


def write_new(path,value):
    p=Path(native.absolute(str(path)));b=raw(value)
    require(p.name.startswith(NAME+'_') and not p.exists() and not p.is_symlink()
        and len(b)<=META and shutil.disk_usage(p.parent).free>=FLOOR+len(b),'exclusive bounded metadata +floor')
    with p.open('xb') as f:require(f.write(b)==len(b),'complete write');f.flush();os.fsync(f.fileno())
    paired.fsync_dir(p.parent)
    return dict(path=str(p),sha256=sha(p),bytes=len(b))


def prepare(output,args):
    p,_=material(args);return write_new(output,p)


def verify(prepared,expected):
    native.absolute(str(prepared));native.actual_hash(expected)
    require(sha(prepared)==expected,'explicit actual own prepare SHA')
    p=read(prepared,META);now,data=material(p['arguments'])
    require(raw(p)==raw(now) and sha(prepared)==expected,'whole own prepare rebuilt from real authorities')
    return p,data


class Store(kernel.Store):
    def __init__(self,path):
        self.path=Path(native.absolute(str(path)))
        require(self.path.name.startswith(NAME+'_') and not self.path.exists() and not self.path.is_symlink()
            and self.path.parent.is_dir() and shutil.disk_usage(self.path.parent).free>=CAP+FLOOR,'new64MiB output +200MiB floor')
        self.path.mkdir();paired.fsync_dir(self.path.parent);self.previous=None;self.index=0
    # check/write/put/frame/emit are frozen native persistence primitives.


class RecordingSink:
    """Tee the real durable journal prefix; no alternate producer schema."""
    def __init__(self,store):
        self.store=store;self.prefix={}
    def __getattr__(self,key):return getattr(self.store,key)
    def emit(self,kind,**fields):
        result=self.store.emit(kind,**fields)  # fsync completes BEFORE in-RAM proof inventory.
        branch=fields.get('branch')
        if branch is not None and (kind in ('placement_recipe','goal_attached')
            or kind=='returned_transition' and fields['phase']=='warmup'
            or kind=='fresh_observation' and (fields['receipt']['phase']=='warmup' or fields['receipt']['step']==0)):
            self.prefix.setdefault(branch,[]).append(dict(kind=kind,**deepcopy(fields)))
        return result


def initial_proof(plan,p,events):
    """Pure saved-prefix proof, recomputed both before dispatch and by consume."""
    i=plan['index'];root=p['source_roots'][plan['root_slot']]
    scoped=[e for e in events if e.get('branch')==i]
    obs=[e for e in scoped if e['kind']=='fresh_observation']
    require([(e['receipt']['phase'],e['receipt']['step'],e['receipt']['overlay_index']) for e in obs]
        ==[('warmup',s,k) for s in range(11) for k in (0,1,2)]+[('plan',0,k) for k in (0,1,2)],
        'exact real warmup33 plus plan0 triplet before first plan action')
    returned=[e for e in scoped if e['kind']=='returned_transition']
    require(len(returned)==10 and [e['step'] for e in returned]==list(range(1,11))
        and all(e['phase']=='warmup' and e['actual_dtype']==np.dtype(np.float32).str
            and e['actual_bits']==e['requested_bits']==moving.warmup_commands()[n].tobytes().hex()
            and e['terminated'] is False and e['truncated'] is False for n,e in enumerate(returned)),
        'all actual warm10 returns before control1; no stop/padding')
    for e in obs:
        f=e['receipt'];phase,step,overlay=f['phase'],f['step'],f['overlay_index']
        require(e['pre']==e['post'] and f['snapshot']==e['post']['snapshot']
            and f['state']==f['snapshot']['state'] and e['pre']['renderer']==root['renderer']
            and e['pre']['goal_pose']==root['goal_pose'],'all pre-guard renderer/state/snapshot identities')
        if phase=='warmup':
            ref=native.allmode.reference_for(p['warm_traces'][plan['root_slot']],step,overlay)
            if ref is not None:native.allmode.check_warm_frame(e,ref)
        else:
            require(f['state']==root['root_state'] and e['pre']['goal_state']==root['goal']
                and e['pre']['goal_rgb_sha256']==root['goal_pixels']['off']
                and f['snapshot']==obs[30]['receipt']['snapshot'],
                'true current state/snapshot and original task goal after warm10')
    history=[e['receipt']['pixel_sha256'] for e in obs if e['receipt']['phase']=='warmup'
        and e['receipt']['step'] in (0,5,10) and e['receipt']['overlay_index']==0]
    require(history==plan['history_pixel_sha256']
        and obs[33]['receipt']['pixel_sha256']==history[-1],'actual history3/current pixels match encoded source')
    for n in range(0,36,3):
        a,b,c=obs[n:n+3]
        require(a['receipt']['file_sha256']==c['receipt']['file_sha256']
            and a['receipt']['pixel_sha256']==c['receipt']['pixel_sha256'],'all off/on/off before control1 exact')
    attached=[e for e in scoped if e['kind']=='goal_attached']
    require(len(attached)==1 and attached[0]['goal']==root['goal']
        and attached[0]['goal_pose']==root['goal_pose'] and attached[0]['goal_rgb_sha256']==root['goal_pixels']['off'],
        'canonical goal attached, not latent controller target')
    # Same once-normalized real history is checked against original controller input.
    warm=np.frombuffer(bytes.fromhex(plan['warm_raw10_bits']),'<f4').reshape(10,2)
    mean=np.frombuffer(bytes.fromhex(plan['mean_bits']),'<f4');std=np.frombuffer(bytes.fromhex(plan['std_bits']),'<f4')
    a=np.divide(np.subtract(warm,mean,dtype=np.float32),std,dtype=np.float32).reshape(1,2,10)
    same32(plan['controller_payload']['a'],a,'actual history once-normalization to exact delivered action2')
    return dict(schema=NAME+'.initial_history_proof',slot=i,selection_digest=plan['selection_digest'],
        raw_float32_sha256=plan['raw_float32_sha256'],history_pixel_sha256=history,
        action_history_sha256=byte_sha(a),goal_pixel_sha256=plan['goal_pixel_sha256'],
        root_sha256=root['root_sha256'],prefix_sha256=digest(scoped))


class Meter(kernel.Meter):
    def __init__(self,store,started=None):
        super().__init__(store,started);self.context=None;self.accepted=set()
    def call(self,kind,fn,*,cleanup=False,**fields):
        if kind=='public_step' and fields.get('phase')=='plan':
            row,plan,p=self.context;i=plan['index']
            if fields['step']==1:
                require(i not in self.accepted,'single causal initial guard')
                proof=initial_proof(plan,p,self.store.prefix.get(i,[]))
                self.store.emit('initial_history_guard',**self.where,proof=proof)
                row['initial_history_guard']=deepcopy(proof);self.accepted.add(i)
            require(i in self.accepted,'no plan callback without accepted durable initial guard')
        return super().call(kind,fn,cleanup=cleanup,**fields)


def missing_row(plan):
    return dict(index=plan['index'],root_slot=plan['root_slot'],condition='off',arm='arbitrary',
        task_id=plan['task_id'],selection_digest=plan['selection_digest'],plan=deepcopy(plan),status='missing',
        reason='no authenticated arbitrary target',checkpoints=[],missing_checkpoints=list(CP),
        final_state=None,final_metrics=None,native_success=None,terminated=None,truncated=None,
        initial_history_guard=None,own_attempted=0,own_returned=0,warmup_attempted=0,warmup_returned=0,PNG=0)


def rows_for(p):
    rows=kernel.rows_for(p)
    for i,row in enumerate(rows):
        row['initial_history_guard']=None
        if i not in ACTIVE:rows[i]=missing_row(p['plans'][i])
    return rows


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
    observe(env,obs,'plan',0,row,meter,store,trace)
    if body.eval_state(goal,state)[0]:raise native.allmode.NativeStop('original root already task-terminal')
    for step,command in enumerate(raw15(plan['commands']),1):
        obs,info,t,tr,reward,actual=dispatch(env,command,'plan',step,row,meter,store,goal)
        observe(env,obs,'plan',step,row,meter,store,trace);success=paired.check_return(env,command,actual,info,goal,reward,t)
        require(moving.exact(np.asarray(info['goal']),packet['goal_rgb']) and
            {k:v for k,v in invariant(body).items() if k!='snapshot'}=={k:v for k,v in bound.items() if k!='snapshot'},
            'original task goalRGB/pose/renderer unchanged for all15')
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


def execute(p,data,store,factory,*,authorization,prepared,prepare_sha256,started=None):
    require(authorization==AUTH,'MAIN physical authorization before callbacks');validate_execution(p);native.actual_hash(prepare_sha256)
    require(p['source_slots']==[r['slot'] for r in data['roots']] and p['source_roots']==native.allmode.root_metadata(data)
        and p['warm_traces']==[r['warm_trace'] for r in data['roots']],'actual source/warmup/task identity')
    store=RecordingSink(store);meter=Meter(store,started);rows=rows_for(p);errors=[];seen=[]
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
        for i in ACTIVE:
            row,plan=rows[i],p['plans'][i]
            meter.context=(row,plan,p)
            meter.where=dict(branch=row['index'],**{k:row[k] for k in ('root_slot','condition','arm','selection_digest')})
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
        for f,h in p['bindings'].items():require(sha(f)==h,'final full input/source closure')
        source_ok=True
    except BaseException as e:fail('final_inputs',e)
    errors.extend(meter.secondary)
    if time.monotonic()-meter.started>=WALL:fail('final_wall',ValueError('900s whole-stage wall cap'))
    good=not errors and source_ok and all(rows[i]['status'] in ('horizon_completed','native_stopped') for i in ACTIVE)
    status=RECORDED if good else 'partial_or_failed'
    try:store.emit('run_finished',status=status,counts=meter.counts,errors=errors)
    except BaseException as e:fail('final_journal',e);status='partial_or_failed'
    r=dict(schema=NAME+'.report',contract=contract(),status=status,execution_binding=binding,rows=rows,errors=errors,
        counts=deepcopy(meter.counts),actual_operation_receipts=deepcopy(meter.receipts),fixed_denominator=6,executed_slots=list(ACTIVE),missing_slots=[2,5],known_roots=2,
        fresh_roots=0,inherited=0,inputs_final_verified=source_ok,new_neural_calls=0,new_independent_samples=0,
        P1_admission=False,P2_admission=False,P2_mechanism=False,scientific_admission=False,freshness=False,
        equal_FLOPs=False,equal_total_compute=False,training=False,physical_release=False,independent_confirmation=False,
        journal_sha256=None,execution_binding_sha256=None,journal_chain_tail=store.previous,wall_seconds=time.monotonic()-meter.started)
    for n,k in (('journal.jsonl','journal_sha256'),('execution_binding.json','execution_binding_sha256')):
        try:r[k]=sha(store.path/n)
        except BaseException as e:fail('final_'+k,e);r['status']='partial_or_failed'
    try:store.put('report.json',r,final=True)
    except BaseException as e:fail('final_report',e);r['status']='partial_or_failed'
    return r


def run(prepared,expected,output,factory,*,authorization):
    require(authorization==AUTH,'MAIN authorization before input reads/output/env')
    started=time.monotonic();p,data=verify(prepared,expected);store=Store(output)
    return execute(p,data,store,factory,authorization=authorization,prepared=prepared,prepare_sha256=expected,started=started)


def audit_chain(r,events):
    require(events and events[0]['kind']=='run_started' and events[0]['contract']==contract()
        and events[0]['prepare_sha256']==r['execution_binding']['prepare_sha256']
        and events[-1]['kind']=='run_finished' and all(events[-1][k]==r[k] for k in ('status','counts','errors')),
        'exact native lifecycle bookends')
    allowed={'run_started','run_finished','operation_dispatch_intent','operation_returned','operation_failed',
        'placement_recipe','goal_attached','invalid_observation','fresh_observation','returned_transition','own_response','branch_closed','initial_history_guard'}
    tail=None;intents={};terminals={};stack=[];closed=0
    for i,e in enumerate(events):
        require(e['kind'] in allowed and type(e['index']) is int and e['index']==i
            and e['previous_sha256']==tail and e['event_sha256']==source.digest({k:v for k,v in e.items() if k!='event_sha256'}),
            'all event kinds/index/hash chain, unknown event rejected')
        tail=e['event_sha256']
        if e['kind']=='branch_closed':
            require(closed<4 and e['receipt']==r['rows'][ACTIVE[closed]],'ordered exact branch closure');closed+=1
        if 'branch' in e:
            require(type(e['branch']) is int and closed<4 and e['branch']==ACTIVE[closed]
                and all(e[k]==r['rows'][ACTIVE[closed]][k] for k in ('root_slot','condition','arm','selection_digest')),
                'active exact full15 branch identity')
        if e['kind'] not in ('operation_dispatch_intent','operation_returned','operation_failed'):continue
        op=e['receipt'];n=op['index'];b=op['branch']
        require(type(n) is type(b) is int and closed<4 and b==ACTIVE[closed] and op['resource'] in contract()['caps']
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
        and [e['receipt'] for e in events if e['kind']=='branch_closed']==[x for x in r['rows'] if x['status'] not in ('not_started','missing')],
        'complete attempted resources and partial/unstarted denominator')
    counts={k:dict(attempted=0,completed=0) for k in contract()['caps']}
    for op in r['actual_operation_receipts']:
        counts[op['resource']]['attempted']+=1;counts[op['resource']]['completed']+=op['status']=='returned'
    require(counts==r['counts'] and all(v['attempted']<=contract()['caps'][k] for k,v in counts.items()),
        'all independent actual counts/caps')
    return closed,counts



def prefix_payload(events):
    """Only the authenticated journal envelope is projected away, no payload keys."""
    return [{k:v for k,v in e.items() if k not in ('index','previous_sha256','event_sha256')}
        for e in events if e['kind'] in ('placement_recipe','goal_attached')
        or e['kind']=='returned_transition' and e['phase']=='warmup'
        or e['kind']=='fresh_observation' and (e['receipt']['phase']=='warmup' or e['receipt']['step']==0)]


def audit_initial_guard(row,plan,events,p):
    guards=[e for e in events if e['kind']=='initial_history_guard']
    intents=[e for e in events if e['kind']=='operation_dispatch_intent'
        and e['receipt']['resource']=='public_step' and e['receipt'].get('phase')=='plan']
    if guards:
        require(len(guards)==1,'one initial guard per actual plan branch')
        g=guards[0];before=[e for e in events if e['index']<g['index']]
        proof=initial_proof(plan,p,prefix_payload(before))
        require(raw(proof)==raw(g['proof'])==raw(row['initial_history_guard']),'recomputed exact initial guard')
        require(all(g['index']<e['index'] for e in intents),'causal guard BEFORE first public intent')
        require(all(e['index']<g['index'] for e in prefix_payload_indexed(events)),
            'all actual warm10 and plan0 evidence BEFORE accepted guard')
    else:
        require(row['initial_history_guard'] is None and not intents,'no plan intent/callback without accepted guard')


def prefix_payload_indexed(events):
    return [e for e in events if e['kind'] in ('placement_recipe','goal_attached')
        or e['kind']=='returned_transition' and e['phase']=='warmup'
        or e['kind']=='fresh_observation' and (e['receipt']['phase']=='warmup' or e['receipt']['step']==0)]

def validate_saved(r,events,p):
    validate_execution(p)
    require(r['schema']==NAME+'.report' and r['contract']==contract() and r['status'] in (RECORDED,'partial_or_failed')
        and len(r['rows'])==r['fixed_denominator']==6 and r['known_roots']==2 and r['fresh_roots']==r['inherited']==0
        and r['execution_binding']['planner_authority']==p['planner_authority']
        and r['execution_binding']['source_report_sha256']==native.SOURCE_SHA,'own commoncontroller full15 six-slot native schema/authority')
    require(all(r[k] is False for k in ('P1_admission','P2_admission','P2_mechanism','scientific_admission','freshness',
        'equal_FLOPs','equal_total_compute','training','physical_release','independent_confirmation'))
        and r['new_neural_calls']==r['new_independent_samples']==0,'engineering only, no science/fresh/equalcompute')
    closed,counts=audit_chain(r,events)
    require(r['executed_slots']==list(ACTIVE) and r['missing_slots']==[2,5], 'six-slot denominator and fixed active identities')
    for i in (2,5):
        require(r['rows'][i]==missing_row(p['plans'][i]) and not any(e.get('branch')==i for e in events), 'missing arbitrary has no callbacks or endpoint')
    for i in ACTIVE:
        row=r['rows'][i]
        plan=p['plans'][i];root=p['source_roots'][plan['root_slot']]
        require(type(row['index']) is int and row['index']==i and row['plan']==plan
            and (row['root_slot'],row['condition'],row['arm'])==ORDER[i]
            and all(row[k]==plan[k] for k in ('task_id','selection_digest'))
            and row['status'] in ('not_started','engineering_failed','native_stopped','horizon_completed')
            and row['evidence_origin']=='new_native_execution_known_root','fixed whole original raw15 branches, no inheritance')
        scoped=[e for e in events if e.get('branch')==i];ops=[o for o in r['actual_operation_receipts'] if o['branch']==i]
        public=[o for o in ops if o['resource']=='public_step'];transitions=[e for e in scoped if e['kind']=='returned_transition']
        observations=[e for e in scoped if e['kind']=='fresh_observation'];audit_labels(row,plan,scoped,p)
        audit_initial_guard(row,plan,scoped,p)
        for phase,label,commands,limit in (('warmup','warmup',moving.warmup_commands(),10),('plan','own',raw15(plan['commands']),15)):
            current=[o for o in public if o['phase']==phase]
            require(len(current)<=limit and [o['step'] for o in current]==list(range(1,len(current)+1))
                and all(o['raw_bits']==commands[j].tobytes().hex() for j,o in enumerate(current))
                and row[label+'_attempted']==len(current) and row[label+'_returned']==sum(o['status']=='returned' for o in current),
                'complete contiguous warm10/raw15 authority, no padding/change/duplicate')
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
                and 0<=step<=(10 if phase=='warmup' else 15) and overlay in (0,1,2)
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
            and [e['response'] for e in scoped if e['kind']=='own_response']==responses,'all actual full15 response labels retained')
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
            and row['missing_checkpoints']==[b for b in CP if b not in [f['step'] for f in cps]],'cp0/5/10/15 exact and explicit missing')
        if row['status'] in ('horizon_completed','native_stopped'):
            require(row['first_error'] is None and row['secondary_errors']==[],'no engineering failure relabelled as native stop')
            keys=[('warmup',b,k) for b in range(row['warmup_returned']+1) for k in (0,1,2)]
            if row['initial_exact']:keys += [('plan',b,k) for b in range(row['own_returned']+1) for k in (0,1,2)]
            require([(e['receipt']['phase'],e['receipt']['step'],e['receipt']['overlay_index']) for e in observations]==keys,
                'all real unpadded full15 off/on/off triplets')
            for j in range(0,len(observations),3):
                a,b,c=observations[j:j+3]
                require(a['receipt']['file_sha256']==c['receipt']['file_sha256'] and a['receipt']['pixel_sha256']==c['receipt']['pixel_sha256']
                    and a['receipt']['state']==b['receipt']['state']==c['receipt']['state'],'off/on/off closes without dynamics change')
            if responses:
                require(row['warmup_returned']==10 and row['initial_exact'] and row['own_returned']==len(responses)
                    and ((row['status']=='native_stopped')==(last['terminated'] or last['truncated'])),'native stop retained without padding')
            require(len(responses)==row['own_returned'],'all successful raw15 returns have own responses')
            if row['status']=='native_stopped' and not responses:
                goal=root['goal'] if row['initial_exact'] else p['warm_traces'][plan['root_slot']][0]['pre']['goal_state']
                final_observation=observations[-1]['receipt'] if observations else None
                require(final_observation is not None and (native.metrics(goal,final_observation['state'])['native_success']
                    or bool(transitions and (transitions[-1]['terminated'] or transitions[-1]['truncated']))),
                    'zero-own native stop requires authenticated actual terminal/truncation, not arbitrary flag')
            if row['status']=='horizon_completed':require(row['own_returned']==15,'actual full15, not full10')
            kinds={k:sum(o['resource']==k and o['status']=='returned' for o in ops) for k in contract()['caps']}
            require(kinds['constructor']==kinds['close']==1 and kinds['reset']==2
                and kinds['set_state']==4 and kinds['goal_render']==2 and kinds['goal_bind']==int(row['initial_exact'])
                and kinds['goal_setter']==2+int(row['initial_exact'])
                and kinds['physics_step']==4+10*kinds['public_step'] and kinds['explicit_render']==row['PNG'],
                'per-branch actual full native lifecycle census')
    if r['status']==RECORDED:
        require(closed==4 and r['errors']==[] and r['inputs_final_verified'] is True and 0<=r['wall_seconds']<WALL
            and all(x['status'] in ('horizon_completed','native_stopped') for x in [r['rows'][i] for i in ACTIVE]),'complete engineering four only')
        c={k:v['completed'] for k,v in counts.items()}
        require(all(v['attempted']==v['completed'] for v in counts.values()) and c['constructor']==c['close']==4 and c['reset']==8
            and c['goal_bind']==sum(r['rows'][i]['initial_exact'] for i in ACTIVE) and c['set_state']==2*c['reset']
            and c['goal_render']==c['reset'] and c['goal_setter']==c['reset']+c['goal_bind']
            and c['physics_step']==2*c['reset']+10*c['public_step']
            and c['public_step']==sum(x['warmup_returned']+x['own_returned'] for x in r['rows'])
            and c['explicit_render']==sum(x['PNG'] for x in r['rows']),'complete actual native lifecycle resource census')
    return deepcopy(r['rows'])


def saved_artifacts(directory):
    files=list(Path(directory).iterdir())
    require(all(f.is_file() and not f.is_symlink() for f in files) and sum(f.stat().st_size for f in files)<=CAP,
        'own64MiB bounded regular artifact census')
    result={f.name:f.read_bytes() for f in files}
    require(sum(map(len,result.values()))<=CAP,'own64MiB actual saved bytes');return result


def consume(output,expected):
    native.actual_hash(expected);directory=Path(native.absolute(str(output)))
    files=saved_artifacts(directory)
    require(sum(map(len,files.values()))<=CAP and hashlib.sha256(files['report.json']).hexdigest()==expected,'actual own report SHA/64MiB')
    r=native.decode(files['report.json']);b=r['execution_binding'];p,_=verify(b['prepared'],b['prepare_sha256'])
    require(hashlib.sha256(files['execution_binding.json']).hexdigest()==r['execution_binding_sha256']
        and native.decode(files['execution_binding.json'])==b,'actual execution binding bytes')
    journal=files['journal.jsonl']
    require(journal.endswith(b'\n') and hashlib.sha256(journal).hexdigest()==r['journal_sha256'],'complete saved journal SHA')
    events=[native.decode(line) for line in journal.splitlines()];rows=validate_saved(r,events,p)
    from PIL import Image
    names={'report.json','execution_binding.json','journal.jsonl'};n=0
    for e in events:
        if e['kind']=='invalid_observation':
            name=e['file'];require(Path(name).name==name and name not in names
                and hashlib.sha256(files[name]).hexdigest()==e['file_sha256'],'retained invalid-render incident');names.add(name)
        if e['kind']!='fresh_observation':continue
        f=e['receipt'];name=f['png']
        require(Path(name).name==name and name not in ('','.','..') and name not in names
            and len(files[name])<=1024**2 and hashlib.sha256(files[name]).hexdigest()==f['file_sha256'],'unique native PNG byte seal')
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
        fixed_denominator=6,executed_slots=list(ACTIVE),missing_slots=[2,5],known_roots=2,fresh_roots=0,inherited=0,saved_PNG_verified=n,
        physical_report_sha256=expected,planner_authority=p['planner_authority'],bindings=bindings,
        metric_breakdown=[dict(root_slot=x['root_slot'],arm=x['arm'],
            metrics=None if x['final_state'] is None else native.metrics(x['goal'],x['final_state'])) for x in rows],
        P1_admission=False,P2_admission=False,P2_mechanism=False,scientific_admission=False,freshness=False,
        equal_FLOPs=False,equal_total_compute=False,independent_confirmation=False,physical_release=False,new_neural_calls=0,new_physics_calls=0)



def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('output','prepared','prepare-sha','report-sha','receipt','authorization',*(k.replace('_','-') for k in ARGS)):
        ap.add_argument('--'+k)
    a=ap.parse_args(argv)
    if a.command=='prepare':result=prepare(a.output,{k:getattr(a,k) for k in ARGS})
    elif a.command=='verify':
        p,_=verify(a.prepared,a.prepare_sha);result=dict(verified=True,denominator=6,active_slots=list(ACTIVE))
    elif a.command=='run':
        require(a.authorization==AUTH,'MAIN explicit authority BEFORE gym import')
        def factory():
            import gymnasium as gym
            import stable_worldmodel.envs
            return gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224,with_target=False)
        r=run(a.prepared,a.prepare_sha,a.output,factory,authorization=a.authorization)
        result=dict(output=a.output,status=r['status'],errors=r['errors'],report_sha256=sha(Path(a.output)/'report.json'))
    else:
        r=consume(a.output,a.report_sha)
        result=write_new(a.receipt,r) if a.receipt else {k:v for k,v in r.items() if k not in ('rows','bindings')}
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
