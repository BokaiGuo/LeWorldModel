"""Bounded CEM H3 actual delivered mean two-root raw15 native replay. MAIN only.

Own full15 schema/loop/validator, frozen native placement/hooks/capture reused.
No model, imagined-context restoration, outcome selection or scientific admission.
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

NAME='agent_lewm_bounded_cem_h3_physical_20261029_v1'
PRODUCER='agent_lewm_bounded_native_cem_h3_20261029_v1'
AUTH='MAIN_BOUNDED_CEM_H3_KNOWN2_FULL15_PHYSICS_ONLY'
RECORDED='bounded_CEM_H3_known2_native_recorded_not_admitted'
ROOT=native.ROOT
ARMS=('bounded_native_CEM_H3_actualmean',)
ORDER=[(r,'off',ARMS[0]) for r in (0,1)]
CP=(0,5,10,15)
CAP,FINAL,FLOOR,WALL,META=64*1024**2,8*1024**2,200*1024**2,900,128*1024**2
require,sha,raw,read=native.require,native.sha,native.raw,native.read
source,moving,paired=native.source,native.moving,native.paired
# Own physical schema uses spaced JSON; planner uses compact JSON + newline.
digest=native.core.digest
ARGS=set()
place,bind_goal,dispatch=native.place,native.bind_goal,native.dispatch
validate_env,native_hooks,invariant=native.validate_env,native.native_hooks,native.invariant


def raw15(value):
    require(type(value) is list and len(value)==15 and all(type(r) is list and len(r)==2 and
        all(type(x) in (int,float) for x in r) for r in value),'literal15x2 numeric raw commands, no bool')
    a=np.asarray(value,dtype=np.float32)
    require(np.isfinite(a).all() and np.all(np.abs(a)<=1) and
        all(float(x)==float(y) for x,y in zip(np.asarray(value,dtype=np.float64).flat,a.flat)),
        'literal lossless float32 raw15 in box, no clipping or reconstruction')
    return a


def contract():
    c=deepcopy(native.allmode.contract())
    c.update(schema=NAME+'.contract',fixed_denominator=2,arm=ARMS[0],order=[list(x) for x in ORDER],arms=list(ARMS),condition='off',
        warmup=10,horizon=15,checkpoints=list(CP),producer=PRODUCER,
        prefix51='fresh original warmup0..10 + selected bounded CEM H3 plan0..5 off/on/off; not forced to equal another action prefix',
        raw_authority='pinned original bounded CEM H3.consume actual_raw15 literal + saved raw NPY; no native reselection',
        missing_export='reject incomplete upstream before execution; native stops retain both slots',
        scientific_authority='physical_release false is not scientific authorization; strict + explicit MAIN engineering AUTH required',
        execution='original reset/place/warm10 then literal selectedraw15, no replanning or imagined-state restoration',
        caps=dict(constructor=2,close=2,reset=4,public_step=50,physics_step=508,set_state=8,
            goal_setter=6,goal_render=4,goal_bind=2,explicit_render=162),
        wall_seconds=WALL,output_cap_bytes=CAP,final_reserve_bytes=FINAL,disk_floor_bytes=FLOOR,
        prepare_cap_bytes=META,inheritance=False,selection_uses_physics=False,
        P1_admission=False,P2_admission=False,P2_mechanism=False,physical_release=False,
        official_reproduction=False,equal_budget_to_1902=False,fair_ranking=False,
        same_support_as_P2=False,same_planning_reduction_as_P2=False,
        interpretation='two known off roots; bounded CEM H3 tanh support/f32sum/N300I30; not official or equal budget')
    return c


def own_files():
    return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]



PLANNER_PREP=ROOT/'results'/ (PRODUCER+'_prepare_MAIN01.json')
PLANNER_OUTPUT=ROOT/'results'/ (PRODUCER+'_cuda_MAIN01')
PLANNER_RECEIPT=ROOT/'results'/ (PRODUCER+'_consume_MAIN01.json')
PREP_SHA='89b091af85bb36aa36a684a438d2179e7ab0c74c85de8303d3ac3c37e10d642b'
REPORT_SHA='b3e239367a482a9ff561e69f7a09657540cac690eecd2adcba08d948c210d6a1'
RECEIPT_SHA='d0e215618c367049c09eed7e93accb90615973f4f8cb322fd19359fef1f7e7a4'
PRODUCER_SHA='4a2f7bf18729f569854610d2f0b5f54c12a52d7941a82acd779a0cbdb4889e7b'
META_WORK=16*1024**2
META_TOTAL=META_WORK+FINAL


def shape_census():
    # Native placement: two resets/root; each reset has two set-state steps.
    # Observations: warm0..10 plus plan0..15, each off/on/off, no goal bodies reused.
    scan=224*(1+224*3)
    # Pessimistic generic deflateBound (not compressBound's default-parameter
    # tight formula): accommodate Pillow's filtered streaming deflate setup.
    compressed=scan+((scan+7)>>3)+((scan+63)>>6)+11
    png=8+25+12+compressed+12*((compressed+65535)//65536)
    caps=dict(constructor=2,close=2,reset=4,public_step=2*(10+15),
        physics_step=2*(4+10*(10+15)),set_state=8,goal_setter=6,goal_render=4,goal_bind=2,
        explicit_render=2*((10+1)+(15+1))*3)
    return dict(caps=caps,PNG_count=162,PNG_scanline_bytes=scan,PNG_file_upper_bytes=png,
        PNG_total_upper_bytes=162*png,operation_receipts=sum(caps.values()),operation_events=2*sum(caps.values()),
        fresh_events=162,transition_events=50,response_events=30,branch_closures=2,
        selected_frames=54,prefix_copies=102,checkpoint_copies=8,last_observation_copies=2,
        metadata_work_cap=META_WORK,metadata_total_cap=META_TOTAL,final_reserve_bytes=FINAL,
        working_upper_bytes=162*png+META_WORK,total_upper_bytes=162*png+META_TOTAL,
        output_cap_bytes=CAP,
        metadata_proof='aggregate enforced quotas, not assumed arbitrary JSON copy factor; journal/rows/report counted on every write')


def assemble(arguments,compiled,consumed,data):
    require(arguments=={} and compiled['schema']==PRODUCER+'.prepare'
        and consumed['schema']==PRODUCER+'.consume'
        and consumed['status']=='known2_H3_bounded_CEM_actual_raw_saved_verified'
        and consumed['report_sha256']==REPORT_SHA and len(consumed['rows'])==2,
        'fixed complete actual CEM ownreceipt and prepare')
    require(all(consumed[k] is False for k in ('fair_ranking','official_reproduction','physical_release','scientific_admission')),
        'no scientific/fair-ranking authorization')
    require([(s['root_slot'],s['condition']) for s in compiled['slots']]==[(0,'off'),(1,'off')],
        'both original off tasks, no deployable-root filtering')
    roots=native.allmode.root_metadata(data);plans=[]
    for i,(chosen,slot,root) in enumerate(zip(consumed['rows'],compiled['slots'],roots,strict=True)):
        a=raw15(chosen['actual_raw15']);packet=data['roots'][i]['packets']['off']
        require(type(chosen['root']) is int and chosen['root']==i
            and all(chosen[k]==slot[k] for k in ('task_id','task_sha256'))
            and chosen['actual_raw15_bytes_sha256']==hashlib.sha256(a.tobytes()).hexdigest()
            and chosen['delivery']['returned_mean_evaluated_in_new_job'] is True
            and chosen['physical_release'] is False,'CEM actual delivered raw15/root/task byte witness')
        require(np.asarray(slot['warm_raw10'],np.float32).tobytes()==moving.warmup_commands().tobytes()
            and moving.exact(np.asarray(packet['raw_action_history']),moving.warmup_commands())
            and slot['pixel_sha256']==[moving.image_sha(x) for x in packet['rgb_history']]+[moving.image_sha(packet['goal_rgb'])]
            and slot['blind_sha256']==data['exports'][i,'off']['sha256']
            and slot['blind_path']==str(data['exports'][i,'off']['path']),
            'original native warm10/history3/goal pixels and blind task')
        plan=dict(schema=NAME+'.execution_plan',index=i,root_slot=i,condition='off',arm=ARMS[0],
            task_id=slot['task_id'],task_sha256=slot['task_sha256'],commands=deepcopy(chosen['actual_raw15']),
            raw_steps=15,raw_float32_sha256=chosen['actual_raw15_bytes_sha256'],
            planner_row=deepcopy(chosen),native_goal=deepcopy(root['goal']),native_goal_pose=deepcopy(root['goal_pose']),
            native_goal_pixel_sha256=root['goal_pixels']['off'],native_root_sha256=root['root_sha256'],
            selection_uses_physics=False,P1_admission=False,P2_admission=False)
        plan['selection_digest']=digest(plan);plans.append(plan)
    p=dict(schema=NAME+'.prepare',contract=contract(),arguments={},plans=plans,bindings={},
        source_report_sha256=native.SOURCE_SHA,source_roots=roots,
        source_slots=[deepcopy(r['slot']) for r in data['roots']],warm_traces=[deepcopy(r['warm_trace']) for r in data['roots']],
        model_slots=deepcopy(compiled['slots']),expected_model_hash=compiled['expected_model_hash'],
        planner_authority=dict(producer=PRODUCER,prepare=str(PLANNER_PREP),prepare_sha256=PREP_SHA,
            report=str(PLANNER_OUTPUT/'report.json'),report_sha256=REPORT_SHA,
            receipt=str(PLANNER_RECEIPT),receipt_sha256=RECEIPT_SHA,
            source_sha256=PRODUCER_SHA,resources=deepcopy(consumed['resources']),inherited=deepcopy(consumed['inherited']),
            missing_exports=[],trust='pinned completed ownreceipt + producer.verify/source/artifact SHA; no recursive CEM numerical fullstrict',
            official_reproduction=False,fair_ranking=False,physical_release=False),
        shape_census=shape_census())
    validate_execution(p);return p


def validate_execution(p):
    require(p['schema']==NAME+'.prepare' and p['contract']==contract() and p['arguments']=={}
        and p['shape_census']==shape_census() and p['source_report_sha256']==native.SOURCE_SHA
        and len(p['plans'])==len(p['source_slots'])==len(p['source_roots'])==len(p['warm_traces'])==len(p['model_slots'])==2,
        'own bounded CEM full15 schema and complete2 roster')
    a=p['planner_authority']
    require(a['producer']==PRODUCER and a['prepare']==str(PLANNER_PREP) and a['prepare_sha256']==PREP_SHA
        and a['report']==str(PLANNER_OUTPUT/'report.json') and a['report_sha256']==REPORT_SHA
        and a['receipt']==str(PLANNER_RECEIPT) and a['receipt_sha256']==RECEIPT_SHA and a['source_sha256']==PRODUCER_SHA
        and a['missing_exports']==[] and a['fair_ranking'] is a['physical_release'] is a['official_reproduction'] is False,
        'explicit pinned completed CEM authority')
    require(shape_census()['working_upper_bytes']<=CAP-FINAL and shape_census()['total_upper_bytes']<=CAP,
        'integer PNG+enforced aggregate metadata transport proof')
    for i,(plan,key) in enumerate(zip(p['plans'],ORDER,strict=True)):
        x=plan['planner_row'];slot=p['model_slots'][i];root=p['source_roots'][i];commands=raw15(plan['commands'])
        require(plan['schema']==NAME+'.execution_plan' and type(plan['index']) is int and plan['index']==i
            and type(plan['root_slot']) is int and (plan['root_slot'],plan['condition'],plan['arm'])==key
            and plan['raw_steps']==15 and plan['selection_digest']==digest({k:v for k,v in plan.items() if k!='selection_digest'})
            and type(x['root']) is int and x['root']==i and slot['root_slot']==i and slot['condition']=='off'
            and all(x[k]==plan[k]==slot[k] for k in ('task_id','task_sha256'))
            and raw15(x['actual_raw15']).tobytes()==commands.tobytes()
            and plan['raw_float32_sha256']==x['actual_raw15_bytes_sha256']==hashlib.sha256(commands.tobytes()).hexdigest()
            and x['delivery']['returned_mean_evaluated_in_new_job'] is True
            and x['physical_release'] is False
            and plan['native_goal']==root['goal'] and plan['native_goal_pose']==root['goal_pose']
            and plan['native_root_sha256']==root['root_sha256'] and plan['native_goal_pixel_sha256']==root['goal_pixels']['off']
            and all(plan[k] is False for k in ('selection_uses_physics','P1_admission','P2_admission')),
            'own exact execution witness literal raw15 task/root/goal identity')


def material(arguments=None):
    require(arguments in (None,{}),'no user overrides of actual cohort/authority')
    code=ROOT/'experiments/spectral_reachability'/(PRODUCER+'.py')
    require(sha(code)==PRODUCER_SHA,'CEM source seal BEFORE import')
    producer=importlib.import_module('experiments.spectral_reachability.'+PRODUCER)
    require(Path(producer.__file__).resolve()==code,'exact installed CEM module')
    require(sha(PLANNER_PREP)==PREP_SHA and sha(PLANNER_OUTPUT/'report.json')==REPORT_SHA
        and sha(PLANNER_RECEIPT)==RECEIPT_SHA,'actual CEM prepare/report/complete ownreceipt SHA')
    compiled=producer.verify(PLANNER_PREP,PREP_SHA)  # saved source reconstruction, zero models
    consumed=read(PLANNER_RECEIPT,META);report=read(PLANNER_OUTPUT/'report.json',META)
    require(report['prepared']==str(PLANNER_PREP) and report['prepare_sha256']==PREP_SHA
        and report['schema']==PRODUCER+'.report' and report['errors']==[]
        and report['status']=='known2_H3_bounded_CEM_actual_raw_evaluated'
        and report['source_verified'] is report['all_final_predictions_sealed'] is True
        and report['model_before']==report['model_after']==compiled['expected_model_hash']
        and report['actual_modules']==consumed['actual_modules'] and len(report['rows'])==2,
        'actual completed CEM report lineage')
    bindings=deepcopy(compiled['sources'])
    for name,rec in report['artifacts'].items():
        require(Path(name).name==name and not (PLANNER_OUTPUT/name).is_symlink()
            and (PLANNER_OUTPUT/name).stat().st_size==rec['file_bytes']
            and sha(PLANNER_OUTPUT/name)==rec['sha256'],'original CEM artifact bytes')
        native.merge(bindings,{str(PLANNER_OUTPUT/name):rec['sha256']})
    require(set(report['artifacts'])|{'report.json'}=={f.name for f in PLANNER_OUTPUT.iterdir()},'complete CEM artifact roster')
    for i,row in enumerate(report['rows']):
        rec=row['delivery']['raw15']
        require(report['artifacts'][rec['file']]=={k:rec[k] for k in ('sha256','file_bytes')}
            and rec['dtype']=='<f4' and rec['shape']==[15,2] and rec['nbytes']==120,
            'paid final actualraw NPY reference')
        with (PLANNER_OUTPUT/rec['file']).open('rb') as f:
            a=np.load(f,allow_pickle=False);require(not f.read(),'no raw NPY trailing bytes')
        require(a.dtype==np.float32 and a.shape==(15,2)
            and a.tobytes()==raw15(consumed['rows'][i]['actual_raw15']).tobytes(),
            'paid actual raw file matches complete receipt literal; no inverse/tanh/repair')
    data=native.source_material();p=assemble({},compiled,consumed,data)
    native.merge(bindings,data['bindings'])
    for f,h in {str(PLANNER_PREP):PREP_SHA,str(PLANNER_OUTPUT/'report.json'):REPORT_SHA,
                str(PLANNER_RECEIPT):RECEIPT_SHA,str(code):PRODUCER_SHA}.items():native.merge(bindings,{f:h})
    for f in [*own_files(),Path(native.__file__).resolve(),Path(native.allmode.__file__).resolve(),Path(native.base.__file__).resolve()]:
        native.merge(bindings,{str(f):sha(f)})
    require(all(sha(f)==h for f,h in bindings.items()),'complete original source/task/scaler/model and native helpers closure')
    p['bindings']=bindings;require(len(raw(p))<=META,'bounded complete prepare');return p,data

def write_new(path,value):
    p=Path(native.absolute(str(path)));b=raw(value)
    require(p.name.startswith(NAME+'_') and not p.exists() and not p.is_symlink() and len(b)<=META
        and shutil.disk_usage(p.parent).free>=FLOOR+len(b),'exclusive bounded own metadata')
    with p.open('xb') as f:require(f.write(b)==len(b),'complete write');f.flush();os.fsync(f.fileno())
    paired.fsync_dir(p.parent);return dict(path=str(p),sha256=sha(p),bytes=len(b))


def prepare(output,arguments=None):
    p,_=material(arguments);return write_new(output,p)


def verify(prepared,expected):
    native.actual_hash(expected);native.absolute(str(prepared));require(sha(prepared)==expected,'actual own prepare SHA')
    p=read(prepared,META);now,data=material(p['arguments'])
    require(raw(p)==raw(now) and sha(prepared)==expected,'complete pinned CEM receipt/source/own prepare reconstruction');return p,data


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

    def write(self,name,value,append=False,final=False):
        require(type(value) is bytes and Path(name).name==name and name not in ('','.','..'),'flat exact serialized bytes')
        files=list(self.path.iterdir());c=shape_census()
        if name.endswith('.png'):
            require(not append and len(value)<=c['PNG_file_upper_bytes']
                and sum(f.suffix=='.png' for f in files)<c['PNG_count'],'actual native PNG bound/count')
        elif name.endswith('_invalid.npz'):
            require(not append and len(value)<=1024**2 and sum(f.suffix=='.npz' for f in files)<2,
                'at most one bounded invalid native observation per branch')
            require(sum(f.stat().st_size for f in files if f.suffix!='.png')+len(value)<=META_WORK,
                'invalid observations charged to aggregate metadata quota')
        else:
            require(name in {'execution_binding.json','journal.jsonl','report.json','completion.json','completion_failure.json'},
                'own fixed metadata roster; no unbudgeted artifacts')
            require(value==raw(native.decode(value)),'canonical exact one-newline JSON record')
            require(append==(name=='journal.jsonl'),'only canonical journal may append')
            if append:require(len(value)<=2*1024**2,'bounded event including full branch closure')
            require(sum(f.stat().st_size for f in files if f.suffix!='.png')+len(value)
                <=(META_TOTAL if final else META_WORK),'aggregate actual metadata copies quota')
        return super().write(name,value,append=append,final=final)


def journal_events(body):
    require(body and body.endswith(b'\n') and not body.endswith(b'\n\n'),'single newline journal framing')
    result=[]
    for line in body.splitlines(keepends=True):
        require(len(line)<=2*1024**2,'bounded actual journal event')
        e=native.decode(line);require(raw(e)==line,'strict canonical journal bytes, no blank lines/alternate codec')
        result.append(e)
    return result


def publish(store,report,started,clock=None):
    clock=time.monotonic if clock is None else clock
    store.put('report.json',report,final=True)
    hashes={f.name:sha(f) for f in store.path.iterdir()}
    elapsed=clock()-started
    c=dict(schema=NAME+'.completion',report_sha256=hashes['report.json'],
        prepared=report['execution_binding']['prepared'],prepare_sha256=report['execution_binding']['prepare_sha256'],
        artifact_sha256=hashes,bytes_before_completion=sum(f.stat().st_size for f in store.path.iterdir()),
        inclusive_wall_seconds=elapsed,wall_seconds_cap=WALL,
        wall_scope='through report fsync and all artifact hashes; completion self-publication excluded',
        status='published_within_cap' if elapsed<WALL else 'publication_wall_failed')
    store.put('completion.json',c,final=True);h=sha(store.path/'completion.json')
    end=clock()-started
    if end>=WALL:
        store.put('completion_failure.json',dict(schema=NAME+'.completion_failure',report_sha256=hashes['report.json'],
            completion_sha256=h,wall_seconds=end),final=True)
    require(end<WALL,'inclusive completion publication wall failed; diagnostic retained')
    return h


def audit_completion(files,r,report_sha,expected):
    native.actual_hash(expected)
    require('completion_failure.json' not in files and 'completion.json' in files
        and hashlib.sha256(files['completion.json']).hexdigest()==expected,'actual completion SHA, no late publication failure')
    c=native.decode(files['completion.json'])
    require(c['schema']==NAME+'.completion' and raw(c)==files['completion.json']
        and c['report_sha256']==report_sha and c['status']=='published_within_cap'
        and c['wall_seconds_cap']==WALL and 0<=r['wall_seconds']<=c['inclusive_wall_seconds']<WALL
        and c['wall_scope']=='through report fsync and all artifact hashes; completion self-publication excluded'
        and all(c[k]==r['execution_binding'][k] for k in ('prepared','prepare_sha256')),
        'inclusive report-fsync/hash wall and prepare authority')
    inventory={n:hashlib.sha256(v).hexdigest() for n,v in files.items() if n!='completion.json'}
    require(c['artifact_sha256']==inventory and c['bytes_before_completion']==sum(len(v) for n,v in files.items() if n!='completion.json'),
        'completion full inventory bytes and SHA')
    pngs=[v for n,v in files.items() if n.endswith('.png')];census=shape_census()
    meta=sum(len(v) for n,v in files.items() if not n.endswith('.png'))
    require(len(pngs)<=162 and all(len(v)<=census['PNG_file_upper_bytes'] for v in pngs)
        and meta<=META_TOTAL and sum(map(len,files.values()))<=CAP,'consumer actual whole transport census')
    nonfinal=sum(len(v) for n,v in files.items() if not n.endswith('.png') and n not in
        {'report.json','completion.json','completion_failure.json'})
    require(nonfinal<=META_WORK,'consumer working metadata quota')


class Meter:
    def __init__(self,store,started=None):
        self.store=store;self.started=time.monotonic() if started is None else started
        self.where={};self.receipts=[];self.secondary=[];self.fault=False
        self.counts={k:dict(attempted=0,completed=0) for k in contract()['caps']};self.active={k:0 for k in self.counts}
    def call(self,kind,fn,*,cleanup=False,**fields):
        require(not cleanup or kind=='close','only nonphysical close may bypass wall/fault/floor')
        if not cleanup:
            require(not self.fault and time.monotonic()-self.started<WALL,'predispatch900s/fault');self.store.check()
        c=self.counts[kind];require(c['attempted']<contract()['caps'][kind],'full15 cap before callback: '+kind)
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
    if phase=='plan' and step==15:
        f=row['trajectory'][-1]
        row['checkpoints'].append({k:deepcopy(f[k]) for k in ('step','png','file_sha256','pixel_sha256','state','snapshot')})
        row['missing_checkpoints'].remove(15)
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


def rows_for(p):
    rows=[]
    for i,key in enumerate(ORDER):
        x=p['plans'][i]
        row=dict(index=i,root_slot=key[0],condition=key[1],arm=key[2],
            task_id=p['model_slots'][key[0]]['task_id'],selection_digest=x['selection_digest'],
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
    good=not errors and source_ok and all(x['status'] in ('horizon_completed','native_stopped') for x in rows)
    status=RECORDED if good else 'partial_or_failed'
    try:store.emit('run_finished',status=status,counts=meter.counts,errors=errors)
    except BaseException as e:fail('final_journal',e);status='partial_or_failed'
    r=dict(schema=NAME+'.report',contract=contract(),status=status,execution_binding=binding,rows=rows,errors=errors,
        counts=deepcopy(meter.counts),actual_operation_receipts=deepcopy(meter.receipts),fixed_denominator=2,known_roots=2,
        fresh_roots=0,inherited=0,inputs_final_verified=source_ok,new_neural_calls=0,new_predict_calls=0,new_AD_calls=0,new_independent_samples=0,
        P1_admission=False,P2_admission=False,P2_mechanism=False,scientific_admission=False,freshness=False,
        equal_FLOPs=False,equal_total_compute=False,training=False,physical_release=False,independent_confirmation=False,
        official_reproduction=False,equal_budget_to_1902=False,fair_ranking=False,
        journal_sha256=None,execution_binding_sha256=None,journal_chain_tail=store.previous,wall_seconds=time.monotonic()-meter.started)
    for n,k in (('journal.jsonl','journal_sha256'),('execution_binding.json','execution_binding_sha256')):
        try:r[k]=sha(store.path/n)
        except BaseException as e:fail('final_'+k,e);r['status']='partial_or_failed'
    try:publish(store,r,meter.started)
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
            require(closed<2 and e['receipt']==r['rows'][closed],'ordered exact branch closure');closed+=1
        if 'branch' in e:
            require(type(e['branch']) is int and e['branch']==closed and closed<2
                and all(e[k]==r['rows'][closed][k] for k in ('root_slot','condition','arm','selection_digest')),
                'active exact full15 branch identity')
        if e['kind'] not in ('operation_dispatch_intent','operation_returned','operation_failed'):continue
        op=e['receipt'];n=op['index'];b=op['branch']
        require(type(n) is type(b) is int and b==closed and 0<=b<2 and op['resource'] in contract()['caps']
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
            require(attached and attached[0]['index']<e['index'],'task attached before every raw15 return');goal=root['goal']
        require(e['goal_state']==goal and e['state']==e['snapshot']['state'],'all full7 return goal/state identities')
        calc=native.metrics(goal,e['state'])
        require(type(e['reward']) is float and e['reward']==calc['native7_reward']
            and type(e['terminated']) is bool and e['terminated'] is calc['native_success'] and type(e['truncated']) is bool,
            'all warmup/plan native7 reward and original success/termination')


def validate_saved(r,events,p):
    validate_execution(p)
    require(r['schema']==NAME+'.report' and r['contract']==contract() and r['status'] in (RECORDED,'partial_or_failed')
        and len(r['rows'])==r['fixed_denominator']==2 and r['known_roots']==2 and r['fresh_roots']==r['inherited']==0
        and r['execution_binding']['planner_authority']==p['planner_authority']
        and r['execution_binding']['source_report_sha256']==native.SOURCE_SHA,'own full15 bounded CEM H3 native schema/authority/both roots')
    require(all(r[k] is False for k in ('P1_admission','P2_admission','P2_mechanism','scientific_admission','freshness',
        'equal_FLOPs','equal_total_compute','training','physical_release','independent_confirmation',
        'official_reproduction','equal_budget_to_1902','fair_ranking'))
        and r['new_neural_calls']==r['new_predict_calls']==r['new_AD_calls']==r['new_independent_samples']==0,
        'engineering only, no science/fresh/equalcompute')
    closed,counts=audit_chain(r,events)
    for i,row in enumerate(r['rows']):
        plan=p['plans'][i]
        root=p['source_roots'][plan['root_slot']]
        require(type(row['index']) is int and row['index']==i and row['plan']==plan
            and (row['root_slot'],row['condition'],row['arm'])==ORDER[i]
            and all(row[k]==plan[k] for k in ('task_id','selection_digest'))
            and row['status'] in ('not_started','engineering_failed','native_stopped','horizon_completed')
            and row['evidence_origin']=='new_native_execution_known_root','fixed whole original raw15 branches, no inheritance')
        scoped=[e for e in events if e.get('branch')==i];ops=[o for o in r['actual_operation_receipts'] if o['branch']==i]
        public=[o for o in ops if o['resource']=='public_step'];transitions=[e for e in scoped if e['kind']=='returned_transition']
        observations=[e for e in scoped if e['kind']=='fresh_observation'];audit_labels(row,plan,scoped,p)
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
        require(closed==2 and r['errors']==[] and r['inputs_final_verified'] is True and 0<=r['wall_seconds']<WALL
            and all(x['status'] in ('horizon_completed','native_stopped') for x in r['rows']),'complete engineering two only')
        c={k:v['completed'] for k,v in counts.items()}
        require(all(v['attempted']==v['completed'] for v in counts.values()) and c['constructor']==c['close']==2 and c['reset']==2*c['constructor']
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


def consume(output,expected,completion_sha):
    native.actual_hash(expected);directory=Path(native.absolute(str(output)))
    files=saved_artifacts(directory)
    require(sum(map(len,files.values()))<=CAP and hashlib.sha256(files['report.json']).hexdigest()==expected,'actual own report SHA/64MiB')
    r=native.decode(files['report.json']);require(raw(r)==files['report.json'],'canonical complete report bytes')
    b=r['execution_binding'];p,_=verify(b['prepared'],b['prepare_sha256'])
    audit_completion(files,r,expected,completion_sha)
    require(hashlib.sha256(files['execution_binding.json']).hexdigest()==r['execution_binding_sha256']
        and files['execution_binding.json']==raw(b),'actual canonical execution binding bytes')
    journal=files['journal.jsonl']
    require(journal.endswith(b'\n') and hashlib.sha256(journal).hexdigest()==r['journal_sha256'],'complete saved journal SHA')
    events=journal_events(journal);rows=validate_saved(r,events,p)
    from PIL import Image
    names={'report.json','execution_binding.json','journal.jsonl','completion.json'};n=0
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
        fixed_denominator=2,known_roots=2,fresh_roots=0,inherited=0,saved_PNG_verified=n,
        physical_report_sha256=expected,completion_sha256=completion_sha,completion=native.decode(files['completion.json']),
        shape_census=shape_census(),planner_authority=p['planner_authority'],bindings=bindings,
        plans=deepcopy(p['plans']),missing_exports=deepcopy(p['planner_authority']['missing_exports']),
        all_two_horizons_completed=all(x['own_returned']==15 for x in rows),
        metric_breakdown=[dict(root_slot=x['root_slot'],arm=x['arm'],
            metrics=None if x['final_state'] is None else native.metrics(x['goal'],x['final_state'])) for x in rows],
        P1_admission=False,P2_admission=False,P2_mechanism=False,scientific_admission=False,freshness=False,
        equal_FLOPs=False,equal_total_compute=False,official_reproduction=False,equal_budget_to_1902=False,
        independent_confirmation=False,physical_release=False,fair_ranking=False,
        new_neural_calls=0,new_predict_calls=0,new_AD_calls=0,new_physics_calls=0)



def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('output','prepared','prepare-sha','report-sha','completion-sha','receipt','authorization'):ap.add_argument('--'+k)
    a=ap.parse_args()
    if a.command=='prepare':result=prepare(a.output)
    elif a.command=='verify':
        p,_=verify(a.prepared,a.prepare_sha);result=dict(status='prepared_verified',plans=len(p['plans']),shape_census=p['shape_census'])
    elif a.command=='run':
        require(a.authorization==AUTH,'MAIN explicit native authority before gym import')
        def factory():
            import gymnasium as gym
            import stable_worldmodel.envs
            return gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224,with_target=False,render_action=False)
        r=run(a.prepared,a.prepare_sha,a.output,factory,authorization=a.authorization)
        result=dict(output=a.output,status=r['status'],errors=r['errors'],report_sha256=sha(Path(a.output)/'report.json'),
            completion_sha256=sha(Path(a.output)/'completion.json') if (Path(a.output)/'completion.json').is_file() else None)
    else:
        require(a.receipt is not None,'exclusive persistent receipt required')
        r=consume(a.output,a.report_sha,a.completion_sha);result=write_new(a.receipt,r)
        result['status']=r['status']
    print(json.dumps(result,sort_keys=True,allow_nan=False))
    if result.get('status')=='partial_or_failed':raise SystemExit(2)


if __name__=='__main__':main()

