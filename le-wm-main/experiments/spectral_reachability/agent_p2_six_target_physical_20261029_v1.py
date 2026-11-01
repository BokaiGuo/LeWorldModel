"""Own-six new native branches from actual paid raw NPYs; MAIN engineering only."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import io
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import agent_p2_six_target_paid_replay_20261029_v1 as paid
from experiments.spectral_reachability import agent_p2_commoncontroller_physical_20261029_v1 as prior
from experiments.spectral_reachability import agent_p2_common_resource_physical_20261029_v1 as lifecycle
from experiments.spectral_reachability import agent_p2_common_resource_physical_summary_20261029_v1 as independent

NAME='agent_p2_six_target_physical_20261029_v1'
AUTH='MAIN_SIX_TARGET6_FULL15_PHYSICS_ONLY'
RECORDED='six_target6_native_recorded_not_admitted'
ROOT=prior.ROOT
native,source,moving,paired=prior.native,prior.source,prior.moving,prior.paired
require,sha,raw,read,digest=prior.require,prior.sha,prior.raw,prior.read,prior.digest
CAP,FINAL,FLOOR,WALL,META=128*1024**2,16*1024**2,200*1024**2,900,128*1024**2
CP=(0,5,10,15)
ACTIVE=tuple(range(6));ARMS=prior.ARMS;ORDER=[(i//3,'off',ARMS[i%3]) for i in ACTIVE]
ARGS=prior.ARGS
ACTUAL_PAID={
 'replay_manifest':str(ROOT/'results/agent_p2_six_target_paid_replay_20261029_v1_prepare_MAIN01.json'),
 'replay_manifest_sha':'9f367dbfdf91dc69ff4ca79562868a8bf16074bff55b91699f5894356a1dc5df',
 'replay_report':str(ROOT/'results/agent_p2_six_target_paid_replay_20261029_v1_cuda_MAIN01/report.json'),
 'replay_report_sha':'c3733a752d1e9b80929edf9726bc0eba2ad1c2ae39b6f05550652ff7b3b74dbd',
 'replay_receipt':str(ROOT/'results/agent_p2_six_target_paid_replay_20261029_v1_consume_MAIN01.json'),
 'replay_receipt_sha':'6e5fa7c57721cd4765430c7379d501d19978bdb44eebe96dce94f7bc68d9a3db'}
PROTOCOL=ROOT/'innerdocs/agent_p2_six_target_physical_comparison_protocol_20261029_v1.md'
PROTOCOL_SHA='6e45721309e9a1fe1107918c988c170daf65ebe032163d0beace73d2bcb118f0'
PAID_SHA='84ebc96da0f7e351132bf697a90a7eaab2e2aa296646fe62c0013db65d9157d2'
PINS={Path(prior.__file__):'7e715a9005a4408fbc7899184f338aa4ae6ee6b20dd414356744aa830ceff5f0',
      Path(lifecycle.__file__):'fb660748af42fa98bcb124fcb67b5ee21059f7e10b457496521afc0a1bbb19ff',
      Path(independent.__file__):'95eb0fa6da34b7cabe1b1ea64bcba6eba98185081a5767651f973f8d1d6c1562',
      Path(paid.__file__):PAID_SHA,PROTOCOL:PROTOCOL_SHA,
      Path(prior.kernel.__file__):prior.KERNEL_SHA}
BOUNDARY='Before prepare verification through native report fsync/hash; completion acknowledgement excluded'
place,bind_goal,dispatch=prior.place,prior.bind_goal,prior.dispatch
validate_env,native_hooks,invariant=prior.validate_env,prior.native_hooks,prior.invariant
observe=prior.observe
raw15,byte_sha,same32=prior.raw15,prior.byte_sha,prior.same32


def own_files():return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def contract():
    c=deepcopy(lifecycle.contract())
    c.update(schema=NAME+'.contract',producer=paid.TAG,protocol_sha256=PROTOCOL_SHA,
        order=[list(x) for x in ORDER],arms=list(ARMS),active_slots=list(ACTIVE),missing_slots=[],fixed_denominator=6,
        raw_authority='six new actual CUDA add outputs authenticated by entire own-six paid receipt',
        first_step_guard='durable exact real warm10/history/current/task before any plan step1 intent',
        missing_export='none permitted at prepare; failures retain all six rows',
        scientific_authority='MAIN explicit engineering authorization; upstream false flags preserved',
        whole_wall_boundary=BOUNDARY,interpretation='six new envs, two known off roots, not fresh or equal total compute')
    require(c['caps']==dict(constructor=6,close=6,reset=12,public_step=150,physics_step=1524,set_state=24,
        goal_setter=18,goal_render=12,goal_bind=6,explicit_render=486),'approved native recipe exact caps; never extend')
    return c


def paid_complete(c,args):
    require(c['schema']==paid.TAG+'.receipt' and c['status']=='all6_actual_deployment_token_replays_verified_not_physically_released'
        and c['manifest_sha256']==args['replay_manifest_sha'] and c['report_sha256']==args['replay_report_sha']
        and c['denominator']==dict(total=6,returned=6,blocked=0,missing=0)
        and c['predict_API']==c['predictor_samples']==36
        and c['module_resources']==dict(action_encoder=36,low_projection=36,AE_tokens=144,predictor_context_tokens=108,projection_rows=108)
        and c['evidence_backend']=='HOST_CUDA' and c['physical_approval_required'] is True
        and all(c[k] is False for k in ('raw_deployment_authorized','scientific_admission','original_Hi_scaler_certified')),
        'actual full six paid authority; false science flags are not refusal of explicit MAIN engineering authority')
    require([(x['slot'],x['root'],x['arm']) for x in c['local_checks']]==[(i,i//3,ARMS[i%3]) for i in ACTIVE]
        and all(x['blocked'] is None and x['predict_API']==x['predictor_samples']==6
            and x['action_encoder']==x['low_projection']==6 and x['AE_tokens']==24
            and x['predictor_context_tokens']==x['projection_rows']==18 for x in c['local_checks']), 'six paid slots, no old-four or partial subset')


def authority(args,c):
    return dict(producer=paid.TAG,arguments=deepcopy(args),protocol_sha256=PROTOCOL_SHA,
        complete_paid_consume_rebuilt=True,persisted_receipt_equal=True,receipt=deepcopy(c),
        original_physical_prepare=dict(path=str(prior.ORIGINAL),sha256=prior.ORIGINAL_SHA),
        original_prediction_cost_reuse_authorized=False,selection_uses_physics=False)


def check_plan(plan,i,p):
    root=p['source_roots'][i//3]
    require(plan['schema']==NAME+'.execution_plan' and type(plan['index']) is int and plan['index']==i
        and (plan['root_slot'],plan['condition'],plan['arm'])==ORDER[i]
        and plan['selection_digest']==digest({k:v for k,v in plan.items() if k!='selection_digest'})
        and all(plan[k] is False for k in ('selection_uses_physics','P1_admission','P2_admission')),'own six native spaced plan codec')
    a=raw15(plan['commands']);payload=plan['controller_payload']
    require(plan['status']=='sealed_raw15' and plan['raw_steps']==15 and a.tobytes().hex()==plan['raw_float32_bits']
        and byte_sha(a)==plan['raw_float32_sha256'] and set(payload)=={'z','a','target'}
        and plan['native_goal']==root['goal'] and plan['native_goal_pose']==root['goal_pose']
        and plan['native_goal_pixel_sha256']==root['goal_pixels']['off'] and plan['native_root_sha256']==root['root_sha256'], 'actual raw15 and original native goal, not target')
    for k,shape in (('z',(1,3,192)),('a',(1,2,10)),('target',(1,192))):
        v=np.asarray(payload[k],np.float32);require(v.shape==shape and np.isfinite(v).all() and byte_sha(v)==plan['payload_sha256'][k],'six input bits '+k)
    mean=np.frombuffer(bytes.fromhex(plan['mean_bits']),'<f4');std=np.frombuffer(bytes.fromhex(plan['std_bits']),'<f4')
    require(mean.shape==std.shape==(2,) and np.isfinite(mean).all() and np.isfinite(std).all() and (std>0).all(),'finite frozen normalization values')
    warm=moving.warmup_commands();past=np.divide(np.subtract(warm,mean,dtype=np.float32),std,dtype=np.float32).reshape(1,2,10)
    same32(payload['a'],past,'real warm10 once-normalized chronology')
    require(plan['warm_raw10_bits']==warm.tobytes().hex() and len(plan['history_pixel_sha256'])==3
        and plan['goal_pixel_sha256']==root['goal_pixels']['off'],'original history and goal pixels')
    ap=plan['paid_authority'];require(ap['slot']==i and all(ap[k]==p['arguments']['replay_'+key+'_sha'] for k,key in
        (('manifest_sha256','manifest'),('report_sha256','report'),('receipt_sha256','receipt')))
        and ap['raw_descriptor']['file'].startswith(f'slot{i}_') and ap['raw_descriptor']['root']==i//3
        and ap['raw_descriptor']['data_sha256']==byte_sha(a),'explicit source-bound actual raw NPY')


def validate_execution(p):
    require(p['schema']==NAME+'.prepare' and p['contract']==contract() and len(p['plans'])==6
        and p['source_report_sha256']==native.SOURCE_SHA and len(p['source_roots'])==len(p['source_slots'])==len(p['warm_traces'])==2
        and set(p['arguments'])==ARGS,'complete own-six execution prepare')
    c=p['planner_authority']['receipt'];paid_complete(c,p['arguments'])
    require(p['planner_authority']==authority(p['arguments'],c),'entire source-bound paid authority')
    for i,plan in enumerate(p['plans']):check_plan(plan,i,p)


def assemble(args,manifest,domain,report,c,original,data,arrays,descs):
    paid_complete(c,args)
    require(manifest['schema']==paid.TAG+'.prepare' and domain['schema']==paid.TAG+'.domain'
        and report['schema']==paid.TAG+'.report' and report['completed'] is True and report['errors']==[]
        and report['manifest_sha256']==args['replay_manifest_sha'] and c['domain']==manifest['domain']
        and c['controller_authorities']==domain['authorities'] and domain['blocked_slot_ids']==[]
        and domain['denominator']==dict(total=6,checked=6,missing=0),'own-six real domain/prepare/report/receipt chain')
    require(original['schema']==prior.kernel.NAME+'.prepare' and original['source_report_sha256']==native.SOURCE_SHA
        and original['source_roots']==native.allmode.root_metadata(data) and original['source_slots']==[r['slot'] for r in data['roots']]
        and original['warm_traces']==[r['warm_trace'] for r in data['roots']],'original exact recipe roots/renderer/warm source')
    slots=domain['controller_prepare']['slots'];require(len(slots)==len(report['rows'])==len(domain['rows'])==6 and set(arrays)==set(descs)==set(ACTIVE),'no borrowed/missing branch')
    plans=[]
    for i,key in enumerate(ORDER):
        d,r,dr=slots[i],report['rows'][i],domain['rows'][i];root=data['roots'][i//3];s=original['plans'][2*(i//3)+1]['planner_plan']
        require((d['slot'],d['root'],d['condition'],d['arm'])==(i,*key)
            and (r['slot'],r['root'],r['condition'],r['arm'],r['status'])==(i,*key,'returned')
            and (dr['slot'],dr['root'],dr['condition'],dr['arm'])==(i,*key),'six fixed actual slots')
        require(all(d[k]==s[k]==dr['task_identity'][k] for k in ('task_id','task_sha256','blind_sha256')),'whole original task/blind identity')
        a,z,past,target,mean,std=arrays[i]
        require(a.dtype==np.float32 and a.shape==(15,2) and a.tobytes().hex()==dr['checks']['array_bits']['raw'],'actual CUDA add output equals domain sealed120 bytes')
        raw15(a.tolist())
        for k,v in (('z',z),('a',past),('target',target)):
            same32(d['controller_payload'][k],v,'exact new paid payload '+k)
            require(byte_sha(v)==d['input_sha256'][k],'producer compact raw-data SHA, not own JSON codec')
        same32(s['root_context']['latent_history'],z[0],'original history3 coordinate')
        same32(s['root_context']['action_history'],past[0],'original true past action2')
        require(mean.tobytes().hex()==domain['normalizer']['mean_bits'] and std.tobytes().hex()==domain['normalizer']['std_bits'],'frozen public stats bits')
        packet=root['packets']['off'];meta=original['source_roots'][i//3]
        plan=dict(schema=NAME+'.execution_plan',index=i,root_slot=i//3,condition='off',arm=key[2],
            **{k:d[k] for k in ('task_id','task_sha256','blind_sha256')},status='sealed_raw15',commands=a.tolist(),raw_steps=15,
            raw_float32_sha256=byte_sha(a),raw_float32_bits=a.tobytes().hex(),controller_payload=deepcopy(d['controller_payload']),
            payload_sha256={k:byte_sha(v) for k,v in (('z',z),('a',past),('target',target))},
            mean_bits=mean.tobytes().hex(),std_bits=std.tobytes().hex(),warm_raw10_bits=moving.warmup_commands().tobytes().hex(),
            history_pixel_sha256=[moving.image_sha(f) for f in packet['rgb_history']],goal_pixel_sha256=moving.image_sha(packet['goal_rgb']),
            native_goal=deepcopy(meta['goal']),native_goal_pose=deepcopy(meta['goal_pose']),native_goal_pixel_sha256=meta['goal_pixels']['off'],native_root_sha256=meta['root_sha256'],
            paid_authority=dict(manifest_sha256=args['replay_manifest_sha'],report_sha256=args['replay_report_sha'],receipt_sha256=args['replay_receipt_sha'],slot=i,raw_descriptor=deepcopy(descs[i])),
            target_origin=deepcopy(d['source']),selection_uses_physics=False,P1_admission=False,P2_admission=False)
        plan['selection_digest']=digest(plan);plans.append(plan)
    p=dict(schema=NAME+'.prepare',arguments=deepcopy(args),contract=contract(),plans=plans,bindings={},
        source_report_sha256=native.SOURCE_SHA,source_roots=deepcopy(original['source_roots']),source_slots=deepcopy(original['source_slots']),warm_traces=deepcopy(original['warm_traces']),
        planner_authority=authority(args,c));validate_execution(p);return p


def material(args):
    require(type(args) is dict and set(args)==ARGS,'explicit final real paid paths/SHA, never future hashes')
    for k,v in args.items():native.actual_hash(v) if k.endswith('_sha') else native.absolute(v)
    require(args==ACTUAL_PAID,'fixed actual six paid production authority, never old-four')
    require(all(sha(p)==h for p,h in PINS.items()),'frozen sources/protocol before complete saved authentication')
    manifest,domain=paid.verify(args['replay_manifest'],args['replay_manifest_sha'])
    report=paid.read(args['replay_report'],args['replay_report_sha']);folder=Path(args['replay_report']).parent
    require(Path(report['manifest']).resolve()==Path(args['replay_manifest']) and report['manifest_sha256']==args['replay_manifest_sha'],'actual paid manifest/report path identity')
    ch=sha(folder/'completion.json');done=paid.read(folder/'completion.json',ch);paid.check_completion(report,done,args['replay_report_sha'])
    paid.validate_files(report,folder);c=paid.validate_saved(manifest,domain,report,folder)
    c.update(manifest_sha256=args['replay_manifest_sha'],report_sha256=args['replay_report_sha'],completion_sha256=ch,domain=manifest['domain'],
        controller_authorities=domain['authorities'],whole_wall_through_report_fsync_hash=done['whole_wall_through_report_fsync_hash'])
    require(paid.encode(paid.read(args['replay_receipt'],args['replay_receipt_sha']))==paid.encode(c),'entire real paid receipt rebuilt without overwriting it')
    original=read(prior.ORIGINAL,META);require(sha(prior.ORIGINAL)==prior.ORIGINAL_SHA,'original recipe authority')
    data=native.source_material();arrays={};descs={}
    for i in ACTIVE:
        row=report['rows'][i];tr=paid.read(folder/row['trace']['file'],row['trace']['file_sha256'])
        require([e['kind'] for e in tr['arithmetic']]==['mul','add','sub','div'] and tr['arithmetic'][1]['returned'] is True,'actual raw add step, not inferred mean')
        desc=row['arrays'][tr['arithmetic'][1]['output']];descs[i]=deepcopy(desc)
        load=lambda d:paid.six.load_array(folder,d)
        arrays[i]=(load(desc),*(load(row['inputs'][k]) for k in ('z','a','target','mean_xy','std')))
    p=assemble(args,manifest,domain,report,c,original,data,arrays,descs)
    bindings=deepcopy(data['bindings']);native.merge(bindings,manifest['sources']);native.merge(bindings,original['bindings'])
    native.merge(bindings,{str(prior.ORIGINAL):prior.ORIGINAL_SHA,**{str(p):h for p,h in PINS.items()},
        **{args[k]:args[k+'_sha'] for k in ('replay_manifest','replay_report','replay_receipt')}})
    for f in folder.iterdir():native.merge(bindings,{str(f.resolve()):sha(f)})
    for f in own_files():native.merge(bindings,{str(f):sha(f)})
    p['bindings']=bindings;require(all(sha(f)==h for f,h in bindings.items()),'entire native/model/input/source seal')
    require(len(raw(p))<=META,'whole128MiB metadata cap, never expand');return p,data


def write_new(path,value):
    p=Path(native.absolute(str(path)));b=raw(value)
    require(p.name.startswith(NAME+'_') and not p.exists() and not p.is_symlink() and len(b)<=META
        and shutil.disk_usage(p.parent).free>=FLOOR+len(b),'exclusive bounded metadata/floor')
    with p.open('xb') as f:require(f.write(b)==len(b),'complete metadata write');f.flush();os.fsync(f.fileno())
    paired.fsync_dir(p.parent);return dict(path=str(p),sha256=sha(p),bytes=len(b))


def prepare(output,args):p,_=material(args);return write_new(output,p)


def verify(prepared,expected):
    native.absolute(str(prepared));native.actual_hash(expected);require(sha(prepared)==expected,'actual prepare SHA')
    p=read(prepared,META);now,data=material(p['arguments'])
    require(raw(p)==raw(now) and sha(prepared)==expected,'whole own-six prepare rebuilt');return p,data


class Store(lifecycle.Store):
    def __init__(self,path):
        self.path=Path(native.absolute(str(path)))
        require(self.path.name.startswith(NAME+'_') and not self.path.exists() and not self.path.is_symlink() and self.path.parent.is_dir()
            and shutil.disk_usage(self.path.parent).free>=CAP+FLOOR,'new128MiB output +200MiB floor')
        self.path.mkdir();paired.fsync_dir(self.path.parent);self.previous=None;self.index=0


RecordingSink=prior.RecordingSink


def initial_proof(plan,p,events):
    # Source-bound nested old low-level proof, NOT a forged new old-schema report.
    return dict(schema=NAME+'.initial_history_proof',algorithm_source_sha256=PINS[Path(prior.__file__)],
        proof=prior.initial_proof(plan,p,events))


class Meter(lifecycle.Meter):
    def __init__(self,store,started=None):
        require(contract()['caps']==lifecycle.contract()['caps'],'identical source-reviewed six native resource policy')
        super().__init__(store,started);self.context=None;self.accepted=set()
    def call(self,kind,fn,*,cleanup=False,**fields):
        if kind=='public_step' and fields.get('phase')=='plan':
            row,plan,p=self.context;i=plan['index']
            if fields['step']==1:
                require(i not in self.accepted,'one initial causal guard')
                proof=initial_proof(plan,p,self.store.prefix.get(i,[]))
                self.store.emit('initial_history_guard',**self.where,proof=proof)
                row['initial_history_guard']=deepcopy(proof);self.accepted.add(i)
            require(i in self.accepted,'no plan intent/callback before durable guard')
        return super().call(kind,fn,cleanup=cleanup,**fields)


def rows_for(p):
    return [dict(index=i,root_slot=i//3,condition='off',arm=ARMS[i%3],task_id=x['task_id'],selection_digest=x['selection_digest'],plan=deepcopy(x),
        status='not_started',first_error=None,secondary_errors=[],warmup_attempted=0,warmup_returned=0,own_attempted=0,own_returned=0,PNG=0,
        initial_exact=False,initial_history_guard=None,trajectory=[],checkpoints=[],missing_checkpoints=list(CP),last_observation=None,responses=[],
        final_state=None,final_metrics=None,native_success=None,terminated=None,truncated=None,prefix51=dict(complete=False,frames=[]),
        source_unpaired_on_steps=[],evidence_origin='new_native_execution_known_root') for i,x in enumerate(p['plans'])]


def episode(env,root,source_plan,plan,row,meter,store):
    body=env.unwrapped;packet=root['packets']['off'];trace=root['warm_trace']
    require(moving.exact(np.asarray(packet['raw_action_history']),moving.warmup_commands()),'original literal warm10')
    obs,goal0=place(env,source_plan,root['slot'],meter,store);history=[observe(env,obs,'warmup',0,row,meter,store,trace)]
    for step,command in enumerate(moving.warmup_commands(),1):
        if body.eval_state(goal0,source.vector(obs['state']))[0]:raise native.allmode.NativeStop('warmup already terminal')
        obs,info,t,tr,reward,actual=dispatch(env,command,'warmup',step,row,meter,store,goal0)
        image=observe(env,obs,'warmup',step,row,meter,store,trace);paired.check_return(env,command,actual,info,goal0,reward,t)
        if t or tr:raise native.allmode.NativeStop('warmup native stop, no padding')
        if step in (5,10):history.append(image)
    require(moving.exact(np.asarray(history),packet['rgb_history']),'real same history3 RGB')
    state=source.vector(obs['state']);goal=source.vector(root['goal'])
    require(source.root_hash(state,goal)==root['root_sha256'],'actual root7 / original goal')
    bind_goal(body,goal,packet['goal_rgb'],meter,store);bound=invariant(body)
    row.update(initial_exact=True,root_state=state.tolist(),root_sha256=root['root_sha256'],goal=goal.tolist(),goal_pose=bound['goal_pose'],goal_pixel_sha256=moving.image_sha(packet['goal_rgb']))
    observe(env,obs,'plan',0,row,meter,store,trace)
    if body.eval_state(goal,state)[0]:raise native.allmode.NativeStop('root already task terminal')
    for step,command in enumerate(raw15(plan['commands']),1):
        obs,info,t,tr,reward,actual=dispatch(env,command,'plan',step,row,meter,store,goal)
        observe(env,obs,'plan',step,row,meter,store,trace);success=paired.check_return(env,command,actual,info,goal,reward,t)
        require(moving.exact(np.asarray(info['goal']),packet['goal_rgb']) and {k:v for k,v in invariant(body).items() if k!='snapshot'}==
            {k:v for k,v in bound.items() if k!='snapshot'},'native goal and renderer unchanged')
        s=source.vector(obs['state']);delta=s-state;calc=native.metrics(goal,s)
        require(calc['native_success'] is bool(success) and calc['native7_reward']==reward,'original native labels')
        response=dict(step=step,state=s.tolist(),delta_native7=delta.tolist(),delta_angle_wrapped=float(np.arctan2(np.sin(delta[4]),np.cos(delta[4]))),
            reward=reward,native_success=bool(success),terminated=t,truncated=tr,native_metrics=calc['native'])
        row['responses'].append(response);row.update(final_state=s.tolist(),final_metrics=calc['native'],native_success=bool(success),terminated=t,truncated=tr)
        store.emit('own_response',**meter.where,response=response)
        if t or tr:row['status']='native_stopped';return
    row['status']='horizon_completed'


def execute(p,data,store,factory,*,authorization,prepared,prepare_sha256,started=None):
    require(authorization==AUTH,'explicit MAIN engineering authority before callbacks');validate_execution(p);native.actual_hash(prepare_sha256)
    require(p['source_slots']==[r['slot'] for r in data['roots']] and p['source_roots']==native.allmode.root_metadata(data)
        and p['warm_traces']==[r['warm_trace'] for r in data['roots']],'same original roots/recipe')
    store=RecordingSink(store);meter=Meter(store,started);rows=rows_for(p);errors=[];seen=[]
    binding=dict(prepared=native.absolute(str(prepared)),prepare_sha256=prepare_sha256,source_report_sha256=native.SOURCE_SHA,planner_authority=deepcopy(p['planner_authority']))
    def fail(stage,e,row=None):
        item=native.base.error(stage,e);errors.append(item)
        if row is not None:
            if row['first_error'] is None:row['first_error']=deepcopy(item)
            else:row['secondary_errors'].append(deepcopy(item))
            row['status']='engineering_failed'
    try:
        store.put('execution_binding.json',binding);store.emit('run_started',prepare_sha256=prepare_sha256,contract=contract())
        for i,row in enumerate(rows):
            plan=p['plans'][i];meter.context=(row,plan,p)
            meter.where=dict(branch=i,**{k:row[k] for k in ('root_slot','condition','arm','selection_digest')})
            env=None;row['status']='running'
            try:
                env=meter.call('constructor',factory)
                require(all(env is not x and env.unwrapped is not x.unwrapped for x in seen),'six distinct new envs, no inheritance');seen.append(env)
                validate_env(env.unwrapped)
                require(env.unwrapped.render_action is False and env.unwrapped.render_size==224 and env.unwrapped.render_mode=='rgb_array','original RGB224 renderer')
                with native_hooks(env,meter,row['secondary_errors']):episode(env,data['roots'][i//3],data['source_plan'],plan,row,meter,store)
                require(not row['secondary_errors'],'native hooks restored')
            except native.allmode.NativeStop as e:row.update(status='native_stopped',stop_reason=str(e))
            except BaseException as e:fail('branch',e,row)
            finally:
                if env is not None:
                    try:meter.call('close',env.close,cleanup=True)
                    except BaseException as e:fail('close',e,row)
                if row['secondary_errors'] and row['first_error'] is None:fail('cleanup',ValueError('native cleanup errors'),row)
                for phase,label in (('warmup','warmup'),('plan','own')):
                    ops=[o for o in meter.receipts if o['resource']=='public_step' and o.get('branch')==i and o['phase']==phase]
                    row[label+'_attempted']=len(ops);row[label+'_returned']=sum(o['status']=='returned' for o in ops)
            store.emit('branch_closed',receipt=row)
    except BaseException as e:fail('execution',e)
    source_ok=False
    try:
        for f,h in p['bindings'].items():require(sha(f)==h,'final entire input/source closure')
        source_ok=True
    except BaseException as e:fail('final_inputs',e)
    errors.extend(meter.secondary)
    if time.monotonic()-meter.started>=WALL:fail('final_wall',ValueError('900s whole-stage cap'))
    status=RECORDED if not errors and source_ok and all(x['status'] in ('horizon_completed','native_stopped') for x in rows) else 'partial_or_failed'
    try:store.emit('run_finished',status=status,counts=meter.counts,errors=errors)
    except BaseException as e:fail('final_journal',e);status='partial_or_failed'
    r=dict(schema=NAME+'.report',contract=contract(),status=status,execution_binding=binding,rows=rows,errors=errors,counts=deepcopy(meter.counts),
        actual_operation_receipts=deepcopy(meter.receipts),fixed_denominator=6,executed_slots=list(ACTIVE),missing_slots=[],known_roots=2,fresh_roots=0,inherited=0,
        inputs_final_verified=source_ok,new_neural_calls=0,new_independent_samples=0,P1_admission=False,P2_admission=False,P2_mechanism=False,
        scientific_admission=False,freshness=False,equal_FLOPs=False,equal_total_compute=False,training=False,physical_release=False,independent_confirmation=False,
        journal_sha256=None,execution_binding_sha256=None,journal_chain_tail=store.previous,wall_seconds=time.monotonic()-meter.started)
    for name,key in (('journal.jsonl','journal_sha256'),('execution_binding.json','execution_binding_sha256')):
        try:r[key]=sha(store.path/name)
        except BaseException as e:fail('final_'+key,e);r['status']='partial_or_failed'
    store.put('report.json',r,final=True);rh=sha(store.path/'report.json');elapsed=time.monotonic()-meter.started
    store.put('completion.json',dict(schema=NAME+'.completion',report_sha256=rh,complete=r['status']==RECORDED and elapsed<WALL,
        wall_through_report_fsync_hash=elapsed,boundary=BOUNDARY),final=True)
    return r


def run(prepared,expected,output,factory,*,authorization):
    require(authorization==AUTH,'MAIN authority before reads/output/factory')
    started=time.monotonic();p,data=verify(prepared,expected);store=Store(output)
    return execute(p,data,store,factory,authorization=authorization,prepared=prepared,prepare_sha256=expected,started=started)


def audit_chain(r,events):
    require(events and events[0]['kind']=='run_started' and events[0]['contract']==contract() and events[0]['prepare_sha256']==r['execution_binding']['prepare_sha256']
        and events[-1]['kind']=='run_finished' and all(events[-1][k]==r[k] for k in ('status','counts','errors')),'exact lifecycle bookends')
    allowed={'run_started','run_finished','operation_dispatch_intent','operation_returned','operation_failed','placement_recipe','goal_attached',
        'invalid_observation','fresh_observation','returned_transition','own_response','branch_closed','initial_history_guard'}
    tail=None;intents={};terminals={};stack=[];closed=0
    for i,e in enumerate(events):
        require(e['kind'] in allowed and type(e['index']) is int and e['index']==i and e['previous_sha256']==tail
            and e['event_sha256']==source.digest({k:v for k,v in e.items() if k!='event_sha256'}),'exact allowed event/hashchain')
        tail=e['event_sha256']
        if e['kind']=='branch_closed':require(closed<6 and e['receipt']==r['rows'][closed],'ordered six new closures');closed+=1
        if 'branch' in e:require(type(e['branch']) is int and closed<6 and e['branch']==closed and
            all(e[k]==r['rows'][closed][k] for k in ('root_slot','condition','arm','selection_digest')),'same active branch identity')
        if e['kind'] not in ('operation_dispatch_intent','operation_returned','operation_failed'):continue
        op=e['receipt'];n=op['index'];b=op['branch']
        require(type(n) is type(b) is int and b==closed and 0<=b<6 and op['resource'] in contract()['caps'] and
            all(op[k]==r['rows'][b][k] for k in ('root_slot','condition','arm','selection_digest')),'literal native resource scope')
        if e['kind']=='operation_dispatch_intent':require(n==len(intents) and op['status']=='pending','ordered debit');intents[n]=op;stack.append(n)
        else:
            require(stack and stack.pop()==n and n in intents and n not in terminals and op['status']==('returned' if e['kind']=='operation_returned' else 'failed')
                and dict({k:v for k,v in op.items() if k!='error'},status='pending')==intents[n],'nested resource return/failure closure');terminals[n]=op
    require(not stack and len(intents)==len(terminals) and tail==r['journal_chain_tail'] and r['actual_operation_receipts']==[terminals[i] for i in range(len(terminals))]
        and [e['receipt'] for e in events if e['kind']=='branch_closed']==[x for x in r['rows'] if x['status']!='not_started'],'all operations/closed partial branches exact')
    counts={k:dict(attempted=0,completed=0) for k in contract()['caps']}
    for op in r['actual_operation_receipts']:counts[op['resource']]['attempted']+=1;counts[op['resource']]['completed']+=op['status']=='returned'
    require(counts==r['counts'] and all(v['attempted']<=contract()['caps'][k] for k,v in counts.items()),'six exact actual resource census')
    return closed,counts


def audit_initial_guard(row,plan,events,p):
    gs=[e for e in events if e['kind']=='initial_history_guard']
    intents=[e for e in events if e['kind']=='operation_dispatch_intent' and e['receipt']['resource']=='public_step' and e['receipt'].get('phase')=='plan']
    if gs:
        require(len(gs)==1,'one own guard');g=gs[0]
        proof=initial_proof(plan,p,prior.prefix_payload([e for e in events if e['index']<g['index']]))
        require(raw(proof)==raw(g['proof'])==raw(row['initial_history_guard']) and all(g['index']<e['index'] for e in intents)
            and all(e['index']<g['index'] for e in prior.prefix_payload_indexed(events)),'exact causal prefix proof before ALL plan intents')
    else:require(row['initial_history_guard'] is None and not intents,'no accepted guard, absolutely no plan intent')


def validate_row(row,plan,events,p,ops):
    """Per-branch checks, independent of old fixed-four outer validator."""
    root=p['source_roots'][plan['root_slot']]
    require(row['plan']==plan and row['index']==plan['index'] and type(row['index']) is int and
        (row['root_slot'],row['condition'],row['arm'])==ORDER[plan['index']] and all(row[k]==plan[k] for k in ('task_id','selection_digest'))
        and row['status'] in ('not_started','engineering_failed','native_stopped','horizon_completed') and
        row['evidence_origin']=='new_native_execution_known_root','own six new branch, no inherited old row')
    public=[o for o in ops if o['resource']=='public_step'];transitions=[e for e in events if e['kind']=='returned_transition']
    observations=[e for e in events if e['kind']=='fresh_observation']
    prior.audit_labels(row,plan,events,p);audit_initial_guard(row,plan,events,p)
    for phase,label,commands,limit in (('warmup','warmup',moving.warmup_commands(),10),('plan','own',raw15(plan['commands']),15)):
        current=[o for o in public if o['phase']==phase];returns=[e for e in transitions if e['phase']==phase]
        require(len(current)<=limit and [o['step'] for o in current]==list(range(1,len(current)+1)) and
            all(o['raw_bits']==commands[j].tobytes().hex() for j,o in enumerate(current)) and row[label+'_attempted']==len(current)
            and row[label+'_returned']==sum(o['status']=='returned' for o in current),'contiguous actual warm10/raw15 attempts')
        require([e['step'] for e in returns]==list(range(1,len(returns)+1)) and len(returns)<=row[label+'_returned'],'all actual return chronology')
        for e in returns:require(e['actual_dtype']==np.dtype(np.float32).str and e['actual_bits']==e['requested_bits']==commands[e['step']-1].tobytes().hex(),'exact requested/actual signedzero bytes')
        if row['status'] in ('horizon_completed','native_stopped'):require(len(returns)==row[label+'_returned'],'no missing successful public labels')
    require(row['PNG']==len(observations),'all render count')
    bridge=[{k:deepcopy(e[k]) for k in ('receipt','pre','post')} for e in observations if e['receipt']['phase']=='warmup' or e['receipt']['step']<=5]
    require(row['prefix51']==dict(complete=len(bridge)==51,frames=bridge),'actual history/prefix51')
    selected=[];unknown=[]
    for e in observations:
        f=e['receipt'];phase,step,overlay=f['phase'],f['step'],f['overlay_index']
        require(phase in ('warmup','plan') and type(step) is type(overlay) is int and 0<=step<=(10 if phase=='warmup' else 15)
            and overlay in (0,1,2) and f['condition']==('on' if overlay==1 else 'off') and e['pre']==e['post']
            and f['snapshot']==e['post']['snapshot'] and f['state']==f['snapshot']['state'] and len(f['state'])==7
            and e['pre']['renderer']==root['renderer'] and e['pre']['goal_pose']==root['goal_pose'],'full real RGB snapshot/renderer invariants')
        if overlay==0:selected.append(f)
        if phase=='warmup':
            trace=p['warm_traces'][plan['root_slot']]
            require(e['pre']['goal_state']==trace[0]['pre']['goal_state'] and e['pre']['goal_rgb_sha256']==trace[0]['pre']['goal_rgb_sha256'],'warm goal before binding')
            ref=native.allmode.reference_for(trace,step,overlay)
            if ref is None:unknown.append(step)
            else:native.allmode.check_warm_frame(e,ref)
        else:
            require(e['pre']['goal_state']==root['goal'] and e['pre']['goal_rgb_sha256']==root['goal_pixels']['off'],'original native task goal after binding')
            if step:
                t=next(t for t in transitions if t['phase']=='plan' and t['step']==step)
                require(f['state']==t['state'] and f['snapshot']==t['snapshot'],'same time state/snapshot/RGB')
    require(row['trajectory']==selected and row['source_unpaired_on_steps']==unknown,'all off frames and source incidence')
    if row['initial_exact']:require(all(row[k]==root[k] for k in ('root_state','root_sha256','goal','goal_pose')) and row['goal_pixel_sha256']==root['goal_pixels']['off']
        and source.root_hash(source.vector(row['root_state']),source.vector(row['goal']))==row['root_sha256'],'actual root/goal exact')
    responses=row['responses']
    require([x['step'] for x in responses]==list(range(1,len(responses)+1)) and len(responses)<=row['own_returned'] and
        [e['response'] for e in events if e['kind']=='own_response']==responses,'complete own response list')
    for resp in responses:
        calc=native.metrics(row['goal'],resp['state']);delta=source.vector(resp['state'])-source.vector(row['root_state']);sc=independent.scalars(resp['state'],row['goal'])
        require(resp['native_metrics']==calc['native'] and resp['reward']==calc['native7_reward']==sc['native7_reward'] and
            resp['native_success'] is resp['terminated'] is calc['native_success'] is sc['native_success'] and type(resp['truncated']) is bool
            and np.array_equal(delta,resp['delta_native7']) and resp['delta_angle_wrapped']==float(np.arctan2(np.sin(delta[4]),np.cos(delta[4]))),'independent original full7 / native success / angle diagnostic')
        t=next(t for t in transitions if t['phase']=='plan' and t['step']==resp['step']);require(all(t[k]==resp[k] for k in ('state','reward','terminated','truncated')),'response/public time join')
    if responses:
        last=responses[-1];require(row['final_state']==last['state'] and row['final_metrics']==last['native_metrics'] and
            all(row[k] is last[k] for k in ('native_success','terminated','truncated')) and
            all(not x['terminated'] and not x['truncated'] for x in responses[:-1]),'native stop without padding')
    else:require(row['final_state'] is row['final_metrics'] is row['native_success'] is None,'missing true terminal remains null')
    cps=[f for f in selected if f['phase']=='plan' and f['step'] in CP]
    require(row['checkpoints']==[{k:f[k] for k in ('step','png','file_sha256','pixel_sha256','state','snapshot')} for f in cps]
        and row['missing_checkpoints']==[b for b in CP if b not in [f['step'] for f in cps]],'fixed four cp slots, no substituted terminal')
    if row['status'] in ('horizon_completed','native_stopped'):
        require(row['first_error'] is None and row['secondary_errors']==[],'no engineering exception relabeled as native stop')
        keys=[('warmup',b,k) for b in range(row['warmup_returned']+1) for k in (0,1,2)]
        if row['initial_exact']:keys += [('plan',b,k) for b in range(row['own_returned']+1) for k in (0,1,2)]
        require([(e['receipt']['phase'],e['receipt']['step'],e['receipt']['overlay_index']) for e in observations]==keys,'all actual triplets')
        for j in range(0,len(observations),3):
            a,b,c=observations[j:j+3];require(a['receipt']['file_sha256']==c['receipt']['file_sha256'] and a['receipt']['pixel_sha256']==c['receipt']['pixel_sha256']
                and a['receipt']['state']==b['receipt']['state']==c['receipt']['state'],'off/on/off does not move dynamics')
        if responses:require(row['warmup_returned']==10 and row['initial_exact'] and row['own_returned']==len(responses)
            and ((row['status']=='native_stopped')==(last['terminated'] or last['truncated'])),'true native stop labels')
        require(len(responses)==row['own_returned'],'all actual response returns')
        if row['status']=='native_stopped' and not responses:
            goal=root['goal'] if row['initial_exact'] else p['warm_traces'][plan['root_slot']][0]['pre']['goal_state']
            f=observations[-1]['receipt'] if observations else None
            require(f is not None and (native.metrics(goal,f['state'])['native_success'] or bool(transitions and
                (transitions[-1]['terminated'] or transitions[-1]['truncated']))),'zero-own stop has actual native terminal evidence')
        if row['status']=='horizon_completed':require(row['own_returned']==15,'actual full15')
        c={k:sum(o['resource']==k and o['status']=='returned' for o in ops) for k in contract()['caps']}
        require(c['constructor']==c['close']==1 and c['reset']==2 and c['set_state']==4 and c['goal_render']==2
            and c['goal_bind']==int(row['initial_exact']) and c['goal_setter']==2+int(row['initial_exact'])
            and c['physics_step']==4+10*c['public_step'] and c['explicit_render']==row['PNG'],'per-env native resource recipe exact')


def validate_saved(r,events,p):
    validate_execution(p)
    require(r['schema']==NAME+'.report' and r['contract']==contract() and r['status'] in (RECORDED,'partial_or_failed')
        and len(r['rows'])==r['fixed_denominator']==6 and r['executed_slots']==list(ACTIVE) and r['missing_slots']==[]
        and r['known_roots']==2 and r['fresh_roots']==r['inherited']==0 and r['execution_binding']['planner_authority']==p['planner_authority']
        and r['execution_binding']['source_report_sha256']==native.SOURCE_SHA,'own-six schema/denominator/authority, never concatenate old rows')
    require(all(r[k] is False for k in ('P1_admission','P2_admission','P2_mechanism','scientific_admission','freshness','equal_FLOPs','equal_total_compute','training','physical_release','independent_confirmation'))
        and r['new_neural_calls']==r['new_independent_samples']==0,'engineering only not science admission')
    closed,counts=audit_chain(r,events)
    for i,row in enumerate(r['rows']):validate_row(row,p['plans'][i],[e for e in events if e.get('branch')==i],p,[o for o in r['actual_operation_receipts'] if o['branch']==i])
    if r['status']==RECORDED:
        require(closed==6 and r['errors']==[] and r['inputs_final_verified'] is True and 0<=r['wall_seconds']<WALL
            and all(x['status'] in ('horizon_completed','native_stopped') for x in r['rows']),'complete new six engineering only')
        c={k:v['completed'] for k,v in counts.items()}
        require(all(v['attempted']==v['completed'] for v in counts.values()) and c['constructor']==c['close']==6 and c['reset']==12
            and c['goal_bind']==sum(x['initial_exact'] for x in r['rows']) and c['set_state']==2*c['reset'] and c['goal_render']==c['reset']
            and c['goal_setter']==c['reset']+c['goal_bind'] and c['physics_step']==2*c['reset']+10*c['public_step']
            and c['public_step']==sum(x['warmup_returned']+x['own_returned'] for x in r['rows'])
            and c['explicit_render']==sum(x['PNG'] for x in r['rows']),'all six real lifecycle, not old reset8')
    return deepcopy(r['rows'])


def saved_artifacts(directory):
    fs=list(Path(directory).iterdir());require(all(f.is_file() and not f.is_symlink() for f in fs) and sum(f.stat().st_size for f in fs)<=CAP,'regular complete128MiB census')
    out={f.name:f.read_bytes() for f in fs};require(sum(map(len,out.values()))<=CAP,'actual saved cap');return out


def checkpoint_panel(rows):
    panel=[]
    for i,row in enumerate(rows):
        cp={x['step']:x for x in row['checkpoints']}
        base=None if 0 not in cp else independent.scalars(cp[0]['state'],row['goal'])
        for step in CP:
            f=cp.get(step);s=None if f is None else independent.scalars(f['state'],row['goal'])
            panel.append(dict(slot=i,root=i//3,arm=ARMS[i%3],step=step,status='missing' if f is None else 'observed',
                reason='checkpoint_not_reached' if f is None else None,checkpoint=deepcopy(f),metrics=s,
                cp0_error_gain=None if s is None or base is None else {k:base[k]-s[k] for k in independent.ERRORS},
                cp0_reward_gain=None if s is None or base is None else s['native7_reward']-base['native7_reward'],
                cp0_angle_change_wrapped=None if s is None or base is None else float(np.arctan2(np.sin(s['state'][4]-base['state'][4]),np.cos(s['state'][4]-base['state'][4]))),
                native7_squared_components=None if s is None else dict(native7_squared=s['native7_squared'],joint_xy_squared=s['joint_xy_squared'],
                    raw_angle_squared=s['raw_angle_squared'],agent_velocity_squared=s['agent_velocity_squared'])))
    pairs=[]
    for root in (0,1):
        for step in CP:
            for left,right in ((0,1),(0,2),(1,2)):
                a=panel[(3*root+left)*4+CP.index(step)]['metrics'];b=panel[(3*root+right)*4+CP.index(step)]['metrics']
                pairs.append(dict(root=root,step=step,left_slot=3*root+left,right_slot=3*root+right,direction='right minus left',
                    comparable=a is not None and b is not None,delta=None if a is None or b is None else
                    {k:float(b[k])-float(a[k]) for k in (*independent.SCALARS,'native_success')}))
    return dict(rows=panel,pairs=pairs,denominator=dict(rows=24,pairs=24),comparable_pairs=sum(x['comparable'] for x in pairs))


def consume(output,expected):
    native.actual_hash(expected);directory=Path(native.absolute(str(output)));files=saved_artifacts(directory)
    require(hashlib.sha256(files['report.json']).hexdigest()==expected,'actual own report SHA')
    r=native.decode(files['report.json']);b=r['execution_binding'];p,_=verify(b['prepared'],b['prepare_sha256'])
    require(hashlib.sha256(files['execution_binding.json']).hexdigest()==r['execution_binding_sha256'] and native.decode(files['execution_binding.json'])==b,'execution binding exact')
    journal=files['journal.jsonl'];require(journal.endswith(b'\n') and hashlib.sha256(journal).hexdigest()==r['journal_sha256'],'complete durable journal SHA')
    done=native.decode(files['completion.json'])
    require(done['schema']==NAME+'.completion' and done['report_sha256']==expected and done['boundary']==BOUNDARY
        and 0<=r['wall_seconds']<=done['wall_through_report_fsync_hash'],'report real fsync/hash wall')
    if r['status']==RECORDED:require(done['complete'] is True and done['wall_through_report_fsync_hash']<WALL,'whole900 inclusive footer, not pre-footer only')
    else:require(done['complete'] is False,'partial never claims complete')
    events=[native.decode(line) for line in journal.splitlines()];rows=validate_saved(r,events,p)
    from PIL import Image
    names={'report.json','execution_binding.json','journal.jsonl','completion.json'};n=0
    for e in events:
        if e['kind']=='invalid_observation':
            name=e['file'];require(Path(name).name==name and name not in names and hashlib.sha256(files[name]).hexdigest()==e['file_sha256'],'retained invalid-render incident');names.add(name)
        if e['kind']!='fresh_observation':continue
        f=e['receipt'];name=f['png'];require(Path(name).name==name and name not in ('','.','..') and name not in names
            and len(files[name])<=1024**2 and hashlib.sha256(files[name]).hexdigest()==f['file_sha256'],'unique actual native PNG byte seal')
        with Image.open(io.BytesIO(files[name])) as im:
            require(im.format=='PNG' and im.mode=='RGB' and im.size==(224,224),'actual RGB224 PNG');x=np.asarray(im).copy()
        require(moving.image_sha(x)==f['pixel_sha256'],'actual PNG pixels');moving.guard(x,source.vector(f['state']));names.add(name);n+=1
    require(set(files)==names,'complete file inventory, no unknown extra work')
    bindings=deepcopy(p['bindings']);native.merge(bindings,{b['prepared']:b['prepare_sha256']})
    native.merge(bindings,{str(directory/name):hashlib.sha256(body).hexdigest() for name,body in files.items()})
    require(all(sha(f)==h for f,h in p['bindings'].items()) and saved_artifacts(directory)==files,'all saved/input/source bytes unchanged')
    return dict(schema=NAME+'.consume',status=r['status'],rows=rows,counts=r['counts'],errors=r['errors'],fixed_denominator=6,
        executed_slots=list(ACTIVE),missing_slots=[],known_roots=2,fresh_roots=0,inherited=0,saved_PNG_verified=n,physical_report_sha256=expected,
        planner_authority=p['planner_authority'],bindings=bindings,checkpoint_panel=checkpoint_panel(rows),
        metric_breakdown=[dict(root_slot=x['root_slot'],arm=x['arm'],metrics=None if x['final_state'] is None else native.metrics(x['goal'],x['final_state'])) for x in rows],
        P1_admission=False,P2_admission=False,P2_mechanism=False,scientific_admission=False,freshness=False,equal_FLOPs=False,equal_total_compute=False,
        independent_confirmation=False,physical_release=False,new_neural_calls=0,new_physics_calls=0,FLOPs=None,
        wall_through_report_fsync_hash=done['wall_through_report_fsync_hash'])


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('output','prepared','prepare-sha','report-sha','receipt','authorization',*(k.replace('_','-') for k in ARGS)):ap.add_argument('--'+k)
    a=ap.parse_args(argv)
    if a.command=='prepare':result=prepare(a.output,{k:getattr(a,k) for k in ARGS})
    elif a.command=='verify':verify(a.prepared,a.prepare_sha);result=dict(verified=True,denominator=6)
    elif a.command=='run':
        require(a.authorization==AUTH,'MAIN explicit engineering AUTH before gym import')
        def factory():
            import gymnasium as gym
            import stable_worldmodel.envs
            return gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224,with_target=False)
        r=run(a.prepared,a.prepare_sha,a.output,factory,authorization=a.authorization)
        result=dict(report=str(Path(a.output)/'report.json'),sha256=sha(Path(a.output)/'report.json'),status=r['status'])
    else:result=write_new(a.receipt,consume(a.output,a.report_sha))
    print(raw(result).decode())


if __name__=='__main__':main()
