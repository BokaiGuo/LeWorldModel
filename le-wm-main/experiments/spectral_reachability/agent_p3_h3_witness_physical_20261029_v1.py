"""MAIN-only same-action H3 witness replay: known2 x banks, prefix5 + raw15.

Own physical schema, loop and strict; frozen native IO primitives only.
No planner calls, reselection, schema impersonation or scientific admission.
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
from experiments.spectral_reachability import agent_mixedmode_physical_20261028_v1 as native
from experiments.spectral_reachability import p3_goal_patch_consistent_compiler_20261027_v2 as normalization

NAME='agent_p3_h3_witness_physical_20261029_v1'
PRODUCER='agent_p3_shared_panel_20261029_v1'
AUTH='MAIN_P3_H3_WITNESS4_CONTROL20_PHYSICS_ONLY'
RECORDED='h3_witness4_control20_native_recorded_not_admitted'
ROOT=native.ROOT
ARMS=('endpoints','midpoints')
ORDER=[(s,'off',b) for s in (0,1) for b in ARMS]
CP=(0,5,10,15,20)
CAP,FINAL,FLOOR,WALL,META=64*1024**2,8*1024**2,200*1024**2,900,128*1024**2
require,sha,raw,read=native.require,native.sha,native.raw,native.read
source,moving,paired=native.source,native.moving,native.paired
# Own physical schema uses spaced JSON; planner uses compact JSON + newline.
digest=native.core.digest
ARGS={'planner_output','planner_report_sha','planner_receipt','planner_receipt_sha','planner_source_sha'}
REPORT_SHA='668db35d79eb650419e4efa42dfc21a15cdf91725700b2f202e2a2ec975bcd6a'
RECEIPT_SHA='d23ffa48bc954306bc27a15f924da8daafffc9b4bc76a91e59aa949ab1abbafd'
PREPARE_SHA='7ca85050eeab4e47ed35cf84c284ed7e08fc64bce8064d7db9450074045b643a'

place,bind_goal,dispatch=native.place,native.bind_goal,native.dispatch
validate_env,native_hooks,invariant=native.validate_env,native.native_hooks,native.invariant


def planner_digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n').hexdigest()


def literal(value,n):
    require(type(value) is list and len(value)==n and all(type(r) is list and len(r)==2 and
        all(type(x) in (int,float) for x in r) for r in value),'literal Nx2 numeric actions, no bool')
    x=np.asarray(value,dtype=np.float32)
    require(np.isfinite(x).all() and np.all(abs(x)<=1) and
        all(float(v)==float(w) for v,w in zip(np.asarray(value,np.float64).flat,x.flat)),
        'lossless float32 raw box, no repair')
    return x


def raw15(value):return literal(value,15)
def raw20(value):return literal(value,20)


def contract():
    c=deepcopy(native.allmode.contract())
    c.update(schema=NAME+'.contract',order=[list(x) for x in ORDER],arms=list(ARMS),condition='off',
        warmup=10,prefix=5,suffix=15,horizon=20,checkpoints=list(CP),producer=PRODUCER,
        raw_authority='fixed adaptive b5 selected H3 winner full15, not MPC later packets; no reselection',
        missing_export='fail prepare on missing upstream witness; runtime failures retain all4',
        execution='independent reset/place/warm10 then original prefix5 plus literal H3 raw15',
        cp5_gate='exact original actual state7/snapshot/RGB before suffix dispatch',
        caps=dict(constructor=4,close=4,reset=8,public_step=120,physics_step=1216,set_state=16,
            goal_setter=12,goal_render=8,goal_bind=4,explicit_render=384),
        wall_seconds=WALL,output_cap_bytes=CAP,final_reserve_bytes=FINAL,disk_floor_bytes=FLOOR,
        prepare_cap_bytes=META,inheritance=False,selection_uses_physics=False,
        new_predict=0,new_encoder=0,new_projector=0,new_AD=0,
        P1_admission=False,P2_admission=False,P3_admission=False,C3=False,
        P2_mechanism=False,physical_release=False,
        interpretation='same-action H3 fidelity evidence only; known2 not fresh, no C3/P1 pass')
    return c


def own_files():
    return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def select_witnesses(compiled,report):
    """Extract already authenticated panel winner, never choose via labels/cost."""
    from experiments.spectral_reachability import agent_p3_shared_panel_20261029_v1 as upstream
    require(compiled['schema']==PRODUCER+'.prepare' and report['schema']==PRODUCER+'.report'
        and report['contract']==compiled['contract']==upstream.contract()
        and report['prepared'] and report['prepare_sha256']==PREPARE_SHA
        and len(compiled['tasks'])==len(compiled['layers'])==2
        and len(report['nodes'])==24 and len(report['rows'])==8,'original shared8/24 and known2')
    witnesses=[]
    for i,(slot,condition,bank) in enumerate(ORDER):
        mode='adaptive_'+bank;eid=f's{slot}_{mode}'
        ns=[n for n in report['nodes'] if n['episode_id']==eid and n['boundary']==5]
        rs=[r for r in report['rows'] if r['episode_id']==eid]
        require(len(ns)==len(rs)==1,'one exact original adaptive b5 branch')
        node,row=ns[0],rs[0];req,res=node['request'],node['receipt']
        require(node['status']=='sealed' and node['source_slot']==row['source_slot']==slot
            and node['mode']==row['mode']==mode and req['boundary']==5
            and req['episode_id']==eid and req['mode']==mode and req['source_slot']==slot
            and row['prefix_exact'] is True and row['initial_exact'] is True,
            'actual independent task prefix and b5 request')
        upstream.validate_request(req,compiled['tasks'][slot])
        require(res['status']=='sealed' and res['selected_horizon']==3
            and res['request_sha256']==upstream.digest(req) and len(res['panels'])==2
            and [p['horizon'] for p in res['panels']]==[1,3],
            'already selected H3, no forced H1 replacement')
        panel=res['panels'][1];winner=panel['winner'];w=winner['index']
        require(panel['status']=='returned' and type(w) is int and 0<=w<4
            and len(panel['candidates'])==4 and winner==panel['candidates'][w]
            and winner['status']=='returned' and panel['selected_path']==winner['whole'],
            'literal authenticated full winner not outcome reselection')
        suffix=raw15(winner['actions'])
        require(literal(res['commands'],5).tobytes()==suffix[:5].tobytes(),
            'H3 winner FIRST5 equals original selected dispatch')
        prefix=upstream.bridge.prefix_binding(compiled['layers'][slot])
        require(row['prefix']==prefix,'original selected prefix binding')
        cps=[c for c in row['checkpoints'] if c['phase']=='plan' and c['step']==5]
        require(len(cps)==1 and cps[0]['pixel_sha256']==req['observation']['pixel_sha256']
            and cps[0]['file_sha256']==req['observation']['png_sha256'],
            'actual cp5 tied to predicted-context encoding')
        witness=dict(episode_id=eid,boundary=5,mode=mode,root_slot=slot,bank=bank,
            task_id=compiled['layers'][slot]['task_id'],task=deepcopy(compiled['tasks'][slot]),
            prefix=prefix,request=deepcopy(req),result=deepcopy(res),node_sha256=upstream.digest(node),
            winner_index=w,winner=deepcopy(winner),actual_cp5=deepcopy(cps[0]),
            suffix_commands=suffix.tolist(),commands=literal(prefix['commands'],5).tolist()+suffix.tolist())
        witness['witness_sha256']=digest(witness);witnesses.append(witness)
    return witnesses


def assemble(arguments,compiled,report,data):
    witnesses=select_witnesses(compiled,report);meta=native.allmode.root_metadata(data);plans=[]
    for i,((slot,condition,bank),w) in enumerate(zip(ORDER,witnesses,strict=True)):
        root=meta[slot];task=w['task'];cp=w['actual_cp5']
        require(task['binding']['root_sha256']==root['root_sha256']
            and cp['goal']==root['goal'] and cp['goal_pose']==root['goal_pose']
            and cp['goal_rgb_sha256']==root['goal_pixels']['off'],'original source root and goal, no retarget')
        require(moving.exact(np.asarray(data['roots'][slot]['packets']['off']['raw_action_history']),
            moving.warmup_commands()),'original source warm10 raw')
        commands=raw20(w['commands'])
        plan=dict(schema=NAME+'.execution_plan',index=i,root_slot=slot,condition=condition,arm=bank,
            task_id=w['task_id'],commands=commands.tolist(),raw_steps=20,
            raw_float32_sha256=hashlib.sha256(commands.tobytes()).hexdigest(),
            prefix_commands=w['prefix']['commands'],suffix_commands=w['suffix_commands'],
            source_witness=w,native_goal=root['goal'],native_goal_pose=root['goal_pose'],
            native_goal_pixel_sha256=root['goal_pixels']['off'],native_root_sha256=root['root_sha256'],
            selection_uses_physics=False,P1_admission=False,P3_admission=False,C3=False)
        plan['selection_digest']=digest(plan);plans.append(plan)
    p=dict(schema=NAME+'.prepare',contract=contract(),arguments=deepcopy(arguments),plans=plans,
        planner_authority=dict(report_sha256=arguments['planner_report_sha'],
            receipt_sha256=arguments['planner_receipt_sha'],prepare_sha256=PREPARE_SHA,
            source_sha256=arguments['planner_source_sha'],missing_exports=[],
            full_upstream_reconstructed_here=True,selection='sealed adaptive b5 H3 winner; no new scoring'),
        source_report_sha256=native.SOURCE_SHA,
        source_roots=meta,source_slots=[deepcopy(r['slot']) for r in data['roots']],
        warm_traces=[deepcopy(r['warm_trace']) for r in data['roots']],bindings={})
    validate_execution(p);return p


def validate_execution(p):
    require(p['schema']==NAME+'.prepare' and p['contract']==contract()
        and p['source_report_sha256']==native.SOURCE_SHA and len(p['plans'])==4
        and len(p['source_roots'])==len(p['source_slots'])==len(p['warm_traces'])==2,
        'own four branches and source closure')
    for i,(x,key) in enumerate(zip(p['plans'],ORDER,strict=True)):
        require(x is not None and (x['root_slot'],x['condition'],x['arm'])==key
            and type(x['index']) is int and x['index']==i and x['raw_steps']==20
            and x['schema']==NAME+'.execution_plan'
            and x['selection_digest']==digest({k:v for k,v in x.items() if k!='selection_digest'}),
            'fixed full20 plan identity/digest')
        a=raw20(x['commands']);w=x['source_witness'];root=p['source_roots'][key[0]]
        require(w['witness_sha256']==digest({k:v for k,v in w.items() if k!='witness_sha256'})
            and x['raw_float32_sha256']==hashlib.sha256(a.tobytes()).hexdigest()
            and a.tobytes()==np.concatenate((literal(x['prefix_commands'],5),raw15(x['suffix_commands']))).tobytes()
            and x['prefix_commands']==w['prefix']['commands'] and x['suffix_commands']==w['winner']['actions']
            and x['commands']==w['commands'] and x['task_id']==w['task_id'],
            'unchanged prefix5 + all15 selected raw bits')
        require(x['native_goal']==root['goal'] and x['native_goal_pose']==root['goal_pose']
            and x['native_root_sha256']==root['root_sha256']
            and x['native_goal_pixel_sha256']==root['goal_pixels']['off']
            and all(x[k] is False for k in ('selection_uses_physics','P1_admission','P3_admission','C3')),
            'original task/no scientific admission')


def material(arguments):
    require(type(arguments) is dict and set(arguments)==ARGS,'explicit actual shared-panel authorities')
    for k,v in arguments.items():
        if k.endswith('_sha'):native.actual_hash(v)
        else:native.absolute(v)
    require(arguments['planner_report_sha']==REPORT_SHA and arguments['planner_receipt_sha']==RECEIPT_SHA,
        'only fixed MAIN shared-panel actual report/strict')
    code=ROOT/'experiments/spectral_reachability'/(PRODUCER+'.py')
    require(sha(code)==arguments['planner_source_sha'],'frozen upstream source before consume')
    upstream=importlib.import_module('experiments.spectral_reachability.'+PRODUCER)
    output=Path(arguments['planner_output']);rp=output/'report.json'
    require(sha(rp)==REPORT_SHA and sha(arguments['planner_receipt'])==RECEIPT_SHA,'actual report/receipt bytes')
    report=read(rp,upstream.contract()['final_reserve_bytes'])
    pp=report['prepared'];require(report['prepare_sha256']==PREPARE_SHA and sha(pp)==PREPARE_SHA,'actual original prepare')
    consumed=upstream.consume(output,REPORT_SHA,pp,PREPARE_SHA)
    require(consumed==read(arguments['planner_receipt'],META) and consumed['complete'] is True,
        'full real shared-panel strict reconstructed, not status-only')
    compiled=upstream.verify(pp,PREPARE_SHA)
    data=native.source_material();p=assemble(arguments,compiled,report,data)
    # Sources include upstream installed-model inventory without loading a model.
    bindings=deepcopy(data['bindings']);native.merge(bindings,compiled['sources'])
    native.merge(bindings,{str(code):arguments['planner_source_sha'],str(rp):REPORT_SHA,
        arguments['planner_receipt']:RECEIPT_SHA,pp:PREPARE_SHA})
    for f in output.rglob('*'):
        require(not f.is_symlink(),'no symlink upstream artifacts')
        if f.is_file():native.merge(bindings,{str(f):sha(f)})
    for module in (native,native.allmode,native.base,moving,source,paired,normalization):
        native.merge(bindings,{str(Path(module.__file__).resolve()):sha(module.__file__)})
    for f in own_files():native.merge(bindings,{str(f):sha(f)})
    p['bindings']=bindings
    for f,h in bindings.items():require(sha(f)==h,'final full source/task/report/model inventory seal')
    require(len(raw(p))<=META,'complete metadata cap');return p,data


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


class Store(native.allmode.Store):
    def __init__(self,path):
        self.path=Path(native.absolute(str(path)))
        require(self.path.name.startswith(NAME+'_') and not self.path.exists() and not self.path.is_symlink()
            and self.path.parent.is_dir() and shutil.disk_usage(self.path.parent).free>=FLOOR+CAP,'new own64MiB output +200MiB floor')
        self.path.mkdir();paired.fsync_dir(self.path.parent);self.previous=None;self.index=0
    def check(self,size=4096,final=False):
        files=list(self.path.iterdir())
        require(all(x.is_file() and not x.is_symlink() for x in files)
            and sum(x.stat().st_size for x in files)+size<=CAP-(0 if final else FINAL)
            and shutil.disk_usage(self.path).free>=FLOOR+size,'whole output cap/final reserve/disk floor')


class Meter:
    def __init__(self,store,started=None):
        self.store=store;self.started=time.monotonic() if started is None else started
        self.where={};self.receipts=[];self.secondary=[];self.fault=False
        self.counts={k:dict(attempted=0,completed=0) for k in contract()['caps']};self.active={k:0 for k in self.counts}
    def call(self,kind,fn,*,cleanup=False,**fields):
        require(not cleanup or kind=='close','only nonphysical close may bypass wall/fault/floor')
        if not cleanup:
            require(not self.fault and time.monotonic()-self.started<WALL,'predispatch900s/fault');self.store.check()
        c=self.counts[kind];require(c['attempted']<contract()['caps'][kind],'control20 cap before callback: '+kind)
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
    if phase=='plan' and step in (15,20):
        f=row['trajectory'][-1]
        row['checkpoints'].append({k:deepcopy(f[k]) for k in ('step','png','file_sha256','pixel_sha256','state','snapshot')})
        row['missing_checkpoints'].remove(step)
    return image


def check_cp5(row,plan):
    ref=plan['source_witness']['actual_cp5']
    frames=[f for f in row['trajectory'] if f['phase']=='plan' and f['step']==5]
    require(len(frames)==1 and all(raw(frames[0][k])==raw(ref[k]) for k in
        ('state','snapshot','pixel_sha256')),'exact actual original cp5 before suffix; no substituted context')
    return True


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
    for step,command in enumerate(raw20(plan['commands']),1):
        obs,info,t,tr,reward,actual=dispatch(env,command,'plan',step,row,meter,store,goal)
        observe(env,obs,'plan',step,row,meter,store,trace);success=paired.check_return(env,command,actual,info,goal,reward,t)
        require(moving.exact(np.asarray(info['goal']),packet['goal_rgb']) and
            {k:v for k,v in invariant(body).items() if k!='snapshot'}=={k:v for k,v in bound.items() if k!='snapshot'},
            'original task goalRGB/pose/renderer unchanged for all20')
        if step==5:check_cp5(row,plan)
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
            task_id=x['task_id'],selection_digest=x['selection_digest'],
            plan=deepcopy(x),status='not_started',
            first_error=None,secondary_errors=[],warmup_attempted=0,warmup_returned=0,own_attempted=0,own_returned=0,
            PNG=0,initial_exact=False,trajectory=[],checkpoints=[],missing_checkpoints=list(CP),
            last_observation=None,responses=[],final_state=None,final_metrics=None,native_success=None,
            terminated=None,truncated=None,prefix51=dict(complete=False,frames=[]),source_unpaired_on_steps=[],
            evidence_origin='new_native_execution_known_root')
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
        for f,h in p['bindings'].items():require(sha(f)==h,'final full input/source closure')
        source_ok=True
    except BaseException as e:fail('final_inputs',e)
    errors.extend(meter.secondary)
    if time.monotonic()-meter.started>=WALL:fail('final_wall',ValueError('900s whole-stage wall cap'))
    good=not errors and source_ok and all(x['status'] in ('horizon_completed','native_stopped','export_missing') for x in rows)
    status=RECORDED if good else 'partial_or_failed'
    try:store.emit('run_finished',status=status,counts=meter.counts,errors=errors)
    except BaseException as e:fail('final_journal',e);status='partial_or_failed'
    r=dict(schema=NAME+'.report',contract=contract(),status=status,execution_binding=binding,rows=rows,errors=errors,
        counts=deepcopy(meter.counts),actual_operation_receipts=deepcopy(meter.receipts),fixed_denominator=4,known_roots=2,
        fresh_roots=0,inherited=0,inputs_final_verified=source_ok,new_neural_calls=0,new_independent_samples=0,
        P1_admission=False,P2_admission=False,P3_admission=False,C3=False,P2_mechanism=False,scientific_admission=False,freshness=False,
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
        'placement_recipe','goal_attached','invalid_observation','fresh_observation','returned_transition','own_response','branch_closed'}
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
                'active exact control20 branch identity')
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
            require(attached and attached[0]['index']<e['index'],'task attached before every raw20 return');goal=root['goal']
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
        and r['execution_binding']['source_report_sha256']==native.SOURCE_SHA,'own control20 native schema/authority/all4')
    require(all(r[k] is False for k in ('P1_admission','P2_admission','P3_admission','C3','P2_mechanism','scientific_admission','freshness',
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
            and row['evidence_origin']=='new_native_execution_known_root','fixed whole original raw20 branches, no inheritance')
        scoped=[e for e in events if e.get('branch')==i];ops=[o for o in r['actual_operation_receipts'] if o['branch']==i]
        public=[o for o in ops if o['resource']=='public_step'];transitions=[e for e in scoped if e['kind']=='returned_transition']
        observations=[e for e in scoped if e['kind']=='fresh_observation'];audit_labels(row,plan,scoped,p)
        for phase,label,commands,limit in (('warmup','warmup',moving.warmup_commands(),10),('plan','own',raw20(plan['commands']),20)):
            current=[o for o in public if o['phase']==phase]
            require(len(current)<=limit and [o['step'] for o in current]==list(range(1,len(current)+1))
                and all(o['raw_bits']==commands[j].tobytes().hex() for j,o in enumerate(current))
                and row[label+'_attempted']==len(current) and row[label+'_returned']==sum(o['status']=='returned' for o in current),
                'complete contiguous warm10/raw20 authority, no padding/change/duplicate')
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
                and 0<=step<=(10 if phase=='warmup' else 20) and overlay in (0,1,2)
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
        if row['own_returned']>=5:check_cp5(row,plan)
        responses=row['responses']
        require([x['step'] for x in responses]==list(range(1,len(responses)+1)) and len(responses)<=row['own_returned']
            and [e['response'] for e in scoped if e['kind']=='own_response']==responses,'all actual control20 response labels retained')
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
                'all real unpadded control20 off/on/off triplets')
            for j in range(0,len(observations),3):
                a,b,c=observations[j:j+3]
                require(a['receipt']['file_sha256']==c['receipt']['file_sha256'] and a['receipt']['pixel_sha256']==c['receipt']['pixel_sha256']
                    and a['receipt']['state']==b['receipt']['state']==c['receipt']['state'],'off/on/off closes without dynamics change')
            if responses:
                require(row['warmup_returned']==10 and row['initial_exact'] and row['own_returned']==len(responses)
                    and ((row['status']=='native_stopped')==(last['terminated'] or last['truncated'])),'native stop retained without padding')
            require(len(responses)==row['own_returned'],'all successful raw20 returns have own responses')
            if row['status']=='native_stopped' and not responses:
                goal=root['goal'] if row['initial_exact'] else p['warm_traces'][plan['root_slot']][0]['pre']['goal_state']
                final_observation=observations[-1]['receipt'] if observations else None
                require(final_observation is not None and (native.metrics(goal,final_observation['state'])['native_success']
                    or bool(transitions and (transitions[-1]['terminated'] or transitions[-1]['truncated']))),
                    'zero-own native stop requires authenticated actual terminal/truncation, not arbitrary flag')
            if row['status']=='horizon_completed':require(row['own_returned']==20,'actual control20, not full10')
            kinds={k:sum(o['resource']==k and o['status']=='returned' for o in ops) for k in contract()['caps']}
            require(kinds['constructor']==kinds['close']==1 and kinds['reset']==2
                and kinds['set_state']==4 and kinds['goal_render']==2 and kinds['goal_bind']==int(row['initial_exact'])
                and kinds['goal_setter']==2+int(row['initial_exact'])
                and kinds['physics_step']==4+10*kinds['public_step'] and kinds['explicit_render']==row['PNG'],
                'per-branch actual full native lifecycle census')
    if r['status']==RECORDED:
        require(closed==4 and r['errors']==[] and r['inputs_final_verified'] is True and 0<=r['wall_seconds']<WALL
            and all(x['status'] in ('horizon_completed','native_stopped','export_missing') for x in r['rows']),'complete engineering four only')
        c={k:v['completed'] for k,v in counts.items()}
        require(all(v['attempted']==v['completed'] for v in counts.values()) and c['constructor']==c['close']==sum(x is not None for x in p['plans']) and c['reset']==2*c['constructor']
            and c['goal_bind']==sum(x['initial_exact'] for x in r['rows']) and c['set_state']==2*c['reset']
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
        fixed_denominator=4,known_roots=2,fresh_roots=0,inherited=0,saved_PNG_verified=n,
        physical_report_sha256=expected,planner_authority=p['planner_authority'],bindings=bindings,
        plans=deepcopy(p['plans']),missing_exports=deepcopy(p['planner_authority']['missing_exports']),
        all_four_physically_executed=all(x['own_returned']==20 for x in rows),
        same_action_H3_fidelity_available=[dict(index=x['index'],prefix5_exact=x['own_returned']>=5,
            actual_control_returned=x['own_returned'],suffix_returned=max(0,x['own_returned']-5),
            missing_checkpoints=x['missing_checkpoints']) for x in rows],
        metric_breakdown=[dict(root_slot=x['root_slot'],arm=x['arm'],
            metrics=None if x['final_state'] is None else native.metrics(x['goal'],x['final_state'])) for x in rows],
        P1_admission=False,P2_admission=False,P3_admission=False,C3=False,P2_mechanism=False,scientific_admission=False,freshness=False,
        equal_FLOPs=False,equal_total_compute=False,independent_confirmation=False,physical_release=False,new_neural_calls=0,new_physics_calls=0)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('output','prepared','prepare-sha','report-sha','receipt','authorization',
              'planner-output','planner-report-sha','planner-receipt','planner-receipt-sha','planner-source-sha'):
        ap.add_argument('--'+k)
    a=ap.parse_args()
    if a.command=='prepare':
        args={k:getattr(a,k) for k in ARGS};result=prepare(a.output,args)
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


