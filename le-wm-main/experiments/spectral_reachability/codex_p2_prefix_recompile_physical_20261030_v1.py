"""Full25 selected versus fixed candidate, replayed after exact feedback guard.

Own full25 schema/loop/validator, frozen native placement/hooks/capture reused.
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

from experiments.spectral_reachability import agent_p2_two_program_full30_physical_20261029_v1 as parent
from experiments.spectral_reachability import agent_p2_full30_control15_feedback_20261029_v2 as planner
from experiments.spectral_reachability import agent_p2_full30_control15_feedback_physical_20261029_v1 as original
from experiments.spectral_reachability import codex_p2_feedback_shared_candidate_20261030_v1 as candidates
from experiments.spectral_reachability import codex_p2_cp15_execution_plan_20261030_v1 as cpbridge
from experiments.spectral_reachability import codex_p2_cp15_physical_binding_20261030_v1 as cpbinding
from experiments.spectral_reachability import codex_p2_prefix_recompile_physical_binding_20261030_v1 as program_binding

NAME='codex_p2_prefix_recompile_physical_20261030_v1'
PRODUCER=planner.NAME
AUTH='P2_PREFIX_RECOMPILE_H25_EXPLORATORY_PHYSICS_ONLY'
RECORDED='prefix_recompile_h25_native_recorded_NOT_admitted'
ROOT=native.ROOT
CP=(0,5,10,15,20,25)
CAP,FINAL,FLOOR,WALL,META=128*1024**2,8*1024**2,200*1024**2,900,16*1024**2
PNG_CAP,JOURNAL_CAP,OTHER_CAP=256*1024,32*1024**2,16*1024**2
require,sha,raw,read=native.require,native.sha,native.raw,native.read
source,moving,paired=native.source,native.moving,native.paired
digest=native.core.digest
place,bind_goal,dispatch=native.place,native.bind_goal,native.dispatch
validate_env,native_hooks,invariant=native.validate_env,native.native_hooks,native.invariant
PINS={
    'prepared':dict(path=str(ROOT/'results/agent_p2_full30_control15_feedback_20261029_v2_prepare_MAIN01.json'),
       sha256='498b561855d7859681a18ed47e0157908a146a796145b929b5436a1e73b79630'),
    'report':dict(path=str(ROOT/'results/agent_p2_full30_control15_feedback_20261029_v2_cuda_MAIN01/report.json'),
       sha256='6e2c82633140e071c880aed92c3cde025635cf7dd92c6687d9e0334bc30db3a8'),
    'completion':dict(path=str(ROOT/'results/agent_p2_full30_control15_feedback_20261029_v2_cuda_MAIN01/completion.json'),
       sha256='8622879f9c674d6acffcbc6646fbaccf25023486b0534626c3b63a5bab1d7329'),
    'receipt':dict(path=str(ROOT/'results/agent_p2_full30_control15_feedback_20261029_v2_consume_MAIN01.json'),
       sha256='4e7d5e1f6b4b03b28b68eb8a31016211f7d1e0ff3115671923b6df2e97f3d893')}
PROTOCOL=ROOT/'innerdocs/agent_p2_full30_control15_feedback_physical_protocol_20261029_v1.md'
PROTOCOL_SHA='de6fdc85a131d1a0a1eac3d27c2cf1450b2cbaf67563f56aa346169e50d4f0f8'
CODE_PINS={str(Path(planner.__file__).resolve()):'2c01821b10a21dd9f9fad43234b71fb76a4ef3169d7993786ab0f006a4983362',
           str(Path(parent.__file__).resolve()):'df65ccb93a800dea8b8e810b488b4818a70f058ee721b685dd49ef10d680a4a9'}
def raw25(value):
    a=np.asarray(value,dtype='<f4')
    require(a.shape==(25,2) and np.isfinite(a).all() and np.all(np.abs(a)<=1)
            and a.astype(float).tolist()==value, 'exact finite float32 H25 without padding')
    return a
PARENT_PINS=deepcopy(planner.E.ACTUAL)
read_pin=planner.read


def source_data():
    return native.source_material()


def contract():
    c=deepcopy(parent.contract())
    c.update(schema=NAME+'.contract',producer=program_binding.NAME,order=['joint','prefix_chain'],arms=['joint','prefix_chain'],
        horizon=25,checkpoints=list(CP),
        fixed_denominator=2,
        raw_authority='independently reconstructed saved CUDA winner float32 bytes and original native raw15 prefix',
        execution='independent natural warm10/prefix15; durable exact prefix guard before suffix10, no hydrate',
        caps=dict(constructor=2,close=2,reset=4,public_step=70,physics_step=708,set_state=8,
            goal_setter=6,goal_render=4,goal_bind=2,explicit_render=222),
        global_policy_slots=8,new_trajectories=8,job_count=4,new_trajectories_per_job=2,
        output_cap_bytes=CAP,final_reserve_bytes=FINAL,
        file_caps=dict(PNG=PNG_CAP,journal=JOURNAL_CAP,other=OTHER_CAP,max_PNG=222),
        budget_proof=dict(PNG_bytes=222*PNG_CAP,journal_bytes=JOURNAL_CAP,other_bytes=OTHER_CAP,
                         reserve_bytes=FINAL,total_reserved_bytes=222*PNG_CAP+JOURNAL_CAP+OTHER_CAP+FINAL),
        inheritance='saved CUDA winner actions; original task/prefix acquisition and CUDA costs inherited separately',
        preregistered=False,after_existing_results_known=True,equal_total_compute=False)
    return c


def own_files():
    return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def normalized_prefix_event(e):
    require(e['kind'] in ('fresh_observation','returned_transition'),'prefix event ABI')
    if e['kind']=='fresh_observation':
        return dict(kind=e['kind'],receipt={k:deepcopy(v) for k,v in e['receipt'].items() if k!='png'},
                    pre=deepcopy(e['pre']),post=deepcopy(e['post']))
    return {k:deepcopy(v) for k,v in e.items() if k not in
        ('index','previous_sha256','event_sha256','branch','root_slot','condition','arm','selection_digest','policy','base_index')}


def is_prefix_event(e):
    if e['kind'] not in ('fresh_observation','returned_transition'):return False
    x=e['receipt'] if e['kind']=='fresh_observation' else e
    return x['phase']=='warmup' or (x['phase']=='plan' and x['step']<=15)


def feedback_guard_value(plan,p):
    return dict(base_index=p['job_index'],root_slot=plan['root_slot'],policy=plan['policy'],task_id=plan['task_id'],
        parent_witness_sha256=plan['planner_plan']['witness_sha256'],
        new_witness_sha256=plan['new_model_plan']['new_witness_sha256'],
        prefix_reference_sha256=digest(p['prefix_reference']),prefix_raw15_bits=raw25(plan['commands'])[:15].tobytes().hex(),
        compile_input_context_sha256=planner.digest(plan['new_model_plan']['input_context']),
        actual_observed_context_sha256=planner.digest(p['observed_context']),
        history_controls=[5,10,15],history_absolute_steps=[15,20,25],no_state_restore=True)


def check_new_plan(plan,job,expected_model):
    planner.K.validate(job)
    require(plan['schema']==PRODUCER+'.model_plan' and plan['new_witness_sha256']==planner.digest(
        {k:v for k,v in plan.items() if k!='new_witness_sha256'}),'new feedback plan witness/schema')
    for k in ('base_index','root_slot','arm','policy','task_id','task_sha256','parent_witness_sha256','parent_raw_bits','goal'):
        require(plan[k]==job[k],'new plan fixed job '+k)
    require(plan['input_context']==job['input_context'] and plan['suffix_cap']==job['suffix_cap']
        and plan['model_hash']==expected_model and plan['feedback_control']==15
        and plan['selection_uses_future_labels'] is False,'new plan context/model/domain scope')
    a=raw25(plan['commands'])
    require(a.tobytes().hex()==plan['raw_float32_bits'] and
        a[:15].tobytes()==np.asarray(job['prefix_raw'],'<f4').tobytes() and
        a[15:].tobytes()==K.chart(plan['suffix_logits']).tobytes(),'fixed prefix and selected suffix literal chart')
    if plan['suffix_logits']!=job['incumbent_logits']:planner.K.legal_new(plan['suffix_logits'],job)
    require(len(plan['new_contexts'])==3,'three full suffix contexts')
    previous=job['input_context']
    for n,c in enumerate(plan['new_contexts']):
        K.context(c);token=normalization.normalized_history(a[15+5*n:20+5*n])[0]
        require(c['latent_history'][:2]==previous['latent_history'][1:] and c['action_history'][0]==previous['action_history'][1]
            and np.asarray(c['action_history'][1],'<f4').tobytes()==token.astype('<f4').tobytes(),'new context causal history')
        previous=c
    require(type(plan['new_suffix_Q']) in (int,float) and np.isfinite(plan['new_suffix_Q']) and
        plan['new_suffix_Q']==K.objective(previous,job['goal']) and
        all(plan[k] is False for k in planner.K.FLAGS),'new modeled Q and science flags')


def assemble(job_index,compiled,receipt,op,oldrows,prefix_reference):
    require(type(job_index) is int and job_index in range(4),'fixed job0..3')
    require(compiled['schema']==PRODUCER+'.prepare' and receipt['schema']==PRODUCER+'.consume' and
        receipt['status']=='control15_feedback_verified_NOT_admitted' and receipt['errors']==[] and
        receipt['report_sha256']==PINS['report']['sha256'] and receipt['prepare_sha256']==PINS['prepared']['sha256'] and
        receipt['completion_sha256']==PINS['completion']['sha256'] and len(receipt['plans'])==8 and
        len(receipt['policy_slots'])==12 and len(compiled['jobs'])==8,'all8 model exports +12 fixed slots')
    plans=[]
    original=op['plans'][job_index]
    for i,policy in enumerate(('I','F')):
        j=2*job_index+i;new=receipt['plans'][j];job=compiled['jobs'][j]
        require(job['base_index']==job_index and job['policy']==policy,'fixed base I/F job')
        check_new_plan(new,job,compiled['expected_model_hash'])
        require(new['parent_witness_sha256']==original['planner_plan']['witness_sha256'] and
            raw25(new['commands'])[:15].tobytes()==raw25(original['commands'])[:15].tobytes(),'exact parent task prefix')
        plan=deepcopy(original)
        plan.update(schema=NAME+'.execution_plan',index=i,base_index=job_index,policy=policy,
                    commands=deepcopy(new['commands']),new_model_plan=deepcopy(new),
                    raw_float32_sha256=hashlib.sha256(raw25(new['commands']).tobytes()).hexdigest())
        plan.pop('selection_digest');plan['selection_digest']=digest(plan);plans.append(plan)
    p=dict(schema=NAME+'.prepare',contract=contract(),arguments=deepcopy(PINS),job_index=job_index,plans=plans,bindings={},
        planner_authority=dict(producer=PRODUCER,authorities=deepcopy(PINS),full_upstream_rebuilt_here=False,
            evidence='pinned MAIN complete model + parent native + RGB receipts; no upstream fullstrict',
            plans_sha256=digest(plans),missing_exports=[]),
        model_jobs=deepcopy(compiled['jobs'][2*job_index:2*job_index+2]),bases=deepcopy(compiled['bases']),
        original_O=deepcopy(oldrows),prefix_reference=deepcopy(prefix_reference),
        observed_context=deepcopy(compiled['jobs'][2*job_index+1]['input_context']),
        inherited_resources=deepcopy(compiled['inherited_resources']),new_model_resources=deepcopy(receipt['actual_new']))
    for k in ('source_report_sha256','source_roots','source_slots','warm_traces','model_slots','expected_model_hash'):
        p[k]=deepcopy(op[k])
    validate_execution(p)
    return p


def validate_execution(p):
    require(p['schema']==NAME+'.prepare' and p['contract']==contract() and p['arguments']==PINS and
            type(p['job_index']) is int and p['job_index'] in range(4) and len(p['plans'])==len(p['model_jobs'])==2 and
            len(p['original_O'])==len(p['bases'])==4 and len(p['source_roots'])==len(p['source_slots'])==len(p['warm_traces'])==2,
            'own physical4jobs/12policies/2new schema')
    require(p['planner_authority']['plans_sha256']==digest(p['plans']) and
        p['planner_authority']['full_upstream_rebuilt_here'] is False and
        p['planner_authority']['authorities']==PINS,'inherited authority/policy pins')
    require(sum(e['kind']=='fresh_observation' for e in p['prefix_reference'])==81 and
        sum(e['kind']=='returned_transition' for e in p['prefix_reference'])==25,'complete warm10+prefix15 reference')
    for i,(plan,job) in enumerate(zip(p['plans'],p['model_jobs'],strict=True)):
        require(plan['index']==i and type(plan['index']) is int and plan['schema']==NAME+'.execution_plan' and
            plan['base_index']==p['job_index'] and plan['policy']==('I','F')[i] and
            plan['selection_digest']==digest({k:v for k,v in plan.items() if k!='selection_digest'}),'own fixed pair plan')
        check_new_plan(plan['new_model_plan'],job,p['expected_model_hash'])
        original=p['original_O'][p['job_index']]['plan']
        require(plan['planner_plan']==original['planner_plan'] and
            plan['task_id']==original['task_id'] and plan['root_slot']==original['root_slot'] and
            plan['arm']==original['arm'] and plan['commands']==plan['new_model_plan']['commands'] and
            plan['native_goal']==original['native_goal'] and plan['native_root_sha256']==original['native_root_sha256'],
            'physical recipe inherited parent only; new own selected commands')


def rehash(bindings):
    parent.rehash(bindings)


def material(job_index):
    require(type(job_index) is int and job_index in range(4),'fixed job index before input reads')
    rehash(CODE_PINS);require(sha(PROTOCOL)==PROTOCOL_SHA,'approved physical protocol')
    c,r,completion,u=[read_pin(PINS[k]['path'],PINS[k]['sha256']) for k in ('prepared','report','completion','receipt')]
    require(r['schema']==PRODUCER+'.report' and r['status']=='all8_feedback_sealed_NOT_admitted' and
        r['prepare_sha256']==PINS['prepared']['sha256'] and r['prepared']==PINS['prepared']['path'] and
        completion['report_sha256']==PINS['report']['sha256'] and completion['elapsed_seconds']<=900*2 and
        r['errors']==[] and r['model_before']==r['model_after']==c['expected_model_hash'],'actual model four authorities')
    pins=PARENT_PINS
    op=read(pins['physical_prepared'],META);oldreport=read(Path(pins['physical_output'])/'report.json',META)
    oldreceipt=read(pins['physical_receipt'],META)
    require(sha(pins['physical_prepared'])==pins['physical_prepare_sha'] and
        sha(Path(pins['physical_output'])/'report.json')==pins['physical_report_sha'] and
        sha(pins['physical_receipt'])==pins['physical_receipt_sha'] and
        oldreceipt['physical_report_sha256']==pins['physical_report_sha'] and oldreport['rows']==oldreceipt['rows'],
        'actual pinned parent native complete receipt')
    jp=Path(pins['physical_output'])/'journal.jsonl'
    require(jp.is_file() and not jp.is_symlink() and jp.stat().st_size<=parent.JOURNAL_CAP,
        'bounded original journal before allocation')
    content=jp.read_bytes()
    require(len(content)<=parent.JOURNAL_CAP and sha(Path(pins['physical_output'])/'journal.jsonl')==oldreport['journal_sha256'],
        'actual parent journal')
    events=[native.decode(line) for line in content.splitlines()]
    ref=[normalized_prefix_event(e) for e in events if e.get('branch')==job_index and is_prefix_event(e)]
    p=assemble(job_index,c,u,op,oldreceipt['rows'],ref)
    bindings=deepcopy(c['bindings_sha256'])
    native.merge(bindings,op['bindings']);native.merge(bindings,oldreceipt['bindings'])
    native.merge(bindings,{d['path']:d['sha256'] for d in PINS.values()})
    directory=Path(PINS['report']['path']).parent
    require({f.name for f in directory.iterdir()}==set(r['artifact_sha256'])|{'report.json','completion.json'},
        'complete actual model artifact inventory')
    for n,h in r['artifact_sha256'].items():
        require(Path(n).name==n,'flat original model artifacts');native.merge(bindings,{str(directory/n):h})
    native.merge(bindings,CODE_PINS);native.merge(bindings,{str(PROTOCOL):PROTOCOL_SHA})
    data=source_data()  # Saved-only original task arrays, no fullstrict or neural calls.
    native.merge(bindings,data['bindings'])
    require(p['source_slots']==[x['slot'] for x in data['roots']] and p['source_roots']==native.allmode.root_metadata(data),
        'source task native metadata not guessed from latents')
    for f in own_files()+parent.own_files()+native.own_files()+native.base.own_files()+[Path(normalization.__file__)]:
        native.merge(bindings,{str(Path(f).resolve()):sha(f)})
    env,h=native.allmode.constructor_contract();native.merge(bindings,{env:h})
    p['bindings']=bindings;rehash(bindings);require(len(raw(p))<=META,'bounded own prepare16MiB')
    return p,data


def write_new(path,value):
    p=Path(native.absolute(str(path)));b=raw(value)
    require(p.name.startswith(NAME+'_') and not p.exists() and not p.is_symlink() and len(b)<=META
        and shutil.disk_usage(p.parent).free>=FLOOR+len(b),'exclusive bounded own metadata')
    with p.open('xb') as f:require(f.write(b)==len(b),'complete write');f.flush();os.fsync(f.fileno())
    paired.fsync_dir(p.parent);return dict(path=str(p),sha256=sha(p),bytes=len(b))


def prepare(output,job_index):
    p,_=material(job_index);return write_new(output,p)


def verify(prepared,expected):
    native.actual_hash(expected);native.absolute(str(prepared));require(sha(prepared)==expected,'actual own prepare SHA')
    p=read(prepared,META);now,data=material(p['job_index'])
    require(raw(p)==raw(now) and sha(prepared)==expected,'own prepare reconstruction from pinned MAIN receipts/source closure');return p,data


def policy_roster(p,rows):
    require(len(p['bases'])==4 and len(rows)==2,'all8 denominator')
    out=[]
    for base in range(4):
        for j,pol in enumerate(('joint','prefix_chain')):
            out.append(dict(base_index=base,policy=pol,job_index=base,
                status=rows[j]['status'] if base==p['job_index'] else 'external_fixed_job'))
    return out


def audit_feedback(row,plan,scoped,p):
    require(row['policy']==plan['policy'] and row['base_index']==p['job_index'],'fixed I/F deployment scope')
    actual=[normalized_prefix_event(e) for e in scoped if is_prefix_event(e)]
    guards=[e for e in scoped if e['kind']=='feedback_prefix_guard']
    suffix=[e for e in scoped if e['kind']=='operation_dispatch_intent' and
        e['receipt']['resource']=='public_step' and e['receipt'].get('phase')=='plan' and e['receipt']['step']>=16]
    expected=p['prefix_reference'][:len(actual)]
    # A real terminal/truncation before cp15 may differ solely in the last
    # transition's stop flags. It never authorizes a suffix.
    differences=[(a,b) for a,b in zip(actual,expected,strict=True) if a!=b]
    if differences:
        require(len(differences)==1 and row['status']=='native_stopped' and
            row['feedback_guard'] is None and not suffix and
            differences[0][0]['kind']=='returned_transition' and
            differences[0][0]['phase']=='plan' and
            differences[0][0]['step']==row['own_returned'] and
            (differences[0][0]['terminated'] or differences[0][0]['truncated']) and
            {k:v for k,v in differences[0][0].items() if k not in ('terminated','truncated')}==
            {k:v for k,v in differences[0][1].items() if k not in ('terminated','truncated')},
            'only real terminal flags may differ before cp15')
    if guards:
        require(len(guards)==1 and actual==p['prefix_reference'] and
            guards[0]['guard']==row['feedback_guard']==feedback_guard_value(plan,p),'full durable feedback guard')
        prefix=[e for e in scoped if is_prefix_event(e)]
        require(prefix and prefix[-1]['index']<guards[0]['index'] and
            all(guards[0]['index']<e['index'] for e in suffix),'guard after prefix return/images before suffix intent')
    else:require(not suffix and row['feedback_guard'] is None,'no suffix without durable exact feedback guard')


class Store(moving.Store):
    """Native raw/digest ABI; incremental bounded writes, no per-event directory scan."""
    def __init__(self,path):
        self.path=Path(native.absolute(str(path)))
        require(self.path.name.startswith(NAME+'_') and not self.path.exists() and not self.path.is_symlink()
            and self.path.parent.is_dir() and shutil.disk_usage(self.path.parent).free>=FLOOR+CAP,
            'new own128MiB output +200MiB floor')
        self.path.mkdir();paired.fsync_dir(self.path.parent)
        self.previous=None;self.index=0;self.sizes={};self.bytes=0;self.finalizing=False
        self.journal_hash=hashlib.sha256();self.prefix_events={}
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
            require(new<=PNG_CAP and sum(n.endswith('.png') for n in self.sizes)+(name not in self.sizes)<=222,
                '222 PNG and256KiB per PNG caps')
        elif name=='journal.jsonl':require(new<=JOURNAL_CAP,'32MiB journal cap')
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
        if is_prefix_event(e):self.prefix_events.setdefault(e['branch'],[]).append(normalized_prefix_event(e))


class Meter:
    def __init__(self,store,started=None):
        self.store=store;self.started=time.monotonic() if started is None else started
        self.where={};self.receipts=[];self.secondary=[];self.fault=False
        self.counts={k:dict(attempted=0,completed=0) for k in contract()['caps']};self.active={k:0 for k in self.counts}
    def call(self,kind,fn,*,cleanup=False,**fields):
        require(not cleanup or kind=='close','only nonphysical close may bypass wall/fault/floor')
        if not cleanup:
            require(not self.fault and time.monotonic()-self.started<WALL,'predispatch900s/fault');self.store.check()
        c=self.counts[kind];require(c['attempted']<contract()['caps'][kind],'full25 cap before callback: '+kind)
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
    if phase=='plan' and step in (15,20,25):
        f=row['trajectory'][-1]
        row['checkpoints'].append({k:deepcopy(f[k]) for k in ('step','png','file_sha256','pixel_sha256','state','snapshot')})
        row['missing_checkpoints'].remove(step)
    return image


def episode(env,root,source_plan,plan,row,meter,store,p):
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
    for step,command in enumerate(raw25(plan['commands']),1):
        if step==16:
            require(store.prefix_events.get(row['index'],[])==p['prefix_reference'],
                'exact full warm10/prefix15 native/RGB source before any suffix dispatch')
            guard=feedback_guard_value(plan,p)
            store.emit('feedback_prefix_guard',**meter.where,guard=guard)
            row['feedback_guard']=guard
        obs,info,t,tr,reward,actual=dispatch(env,command,'plan',step,row,meter,store,goal)
        observe(env,obs,'plan',step,row,meter,store,trace);success=paired.check_return(env,command,actual,info,goal,reward,t)
        require(moving.exact(np.asarray(info['goal']),packet['goal_rgb']) and
            {k:v for k,v in invariant(body).items() if k!='snapshot'}=={k:v for k,v in bound.items() if k!='snapshot'},
            'original task goalRGB/pose/renderer unchanged for all25')
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
    for i,x in enumerate(p['plans']):
        key=(x['root_slot'],'off',x['arm'])
        row=dict(index=i,root_slot=key[0],condition=key[1],arm=key[2],
            task_id=p['model_slots'][key[0]]['task_id'],selection_digest=None if x is None else x['selection_digest'],
            plan=deepcopy(x),status='export_missing' if x is None else 'not_started',
            first_error=None,secondary_errors=[],warmup_attempted=0,warmup_returned=0,own_attempted=0,own_returned=0,
            PNG=0,initial_exact=False,trajectory=[],checkpoints=[],missing_checkpoints=list(CP),
            last_observation=None,responses=[],final_state=None,final_metrics=None,native_success=None,
            terminated=None,truncated=None,prefix51=dict(complete=False,frames=[]),source_unpaired_on_steps=[],
            evidence_origin='no_execution_missing_export' if x is None else 'new_native_execution_known_root',
            policy=x['policy'],base_index=p['job_index'],feedback_guard=None)
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
                    episode(env,data['roots'][plan['root_slot']],data['source_plan'],plan,row,meter,store,p)
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
        counts=deepcopy(meter.counts),actual_operation_receipts=deepcopy(meter.receipts),fixed_denominator=2,global_policy_slots=8,job_index=p['job_index'],policy_roster=policy_roster(p,rows),known_roots=2,
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
        'placement_recipe','goal_attached','native_history_guard','feedback_prefix_guard','invalid_observation','fresh_observation','returned_transition','own_response','branch_closed'}
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
                'active exact full25 branch identity')
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
            require(attached and attached[0]['index']<e['index'],'task attached before every raw25 return');goal=root['goal']
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
        and r['execution_binding']['source_report_sha256']==native.SOURCE_SHA,'own full25 native schema/authority/all4')
    require(all(r[k] is False for k in ('P1_admission','P2_admission','P2_mechanism','scientific_admission','freshness',
        'equal_FLOPs','equal_total_compute','training','physical_release','independent_confirmation'))
        and r['new_neural_calls']==r['new_independent_samples']==0,'engineering only, no science/fresh/equalcompute')
    require(r['job_index']==p['job_index'] and r['global_policy_slots']==8 and r['policy_roster']==policy_roster(p,r['rows']),
        'whole8 fixed external/new roster')
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
            and (row['root_slot'],row['condition'],row['arm'])==(plan['root_slot'],'off',plan['arm'])
            and all(row[k]==plan[k] for k in ('task_id','selection_digest'))
            and row['status'] in ('not_started','engineering_failed','native_stopped','horizon_completed')
            and row['evidence_origin']=='new_native_execution_known_root','fixed whole original raw25 branches, no inheritance')
        # Native Meter operation scopes live inside receipt, not on the event.
        # Include them so history/feedback guard chronology actually sees intents.
        scoped=[e for e in events if e.get('branch',e.get('receipt',{}).get('branch'))==i]
        ops=[o for o in r['actual_operation_receipts'] if o['branch']==i]
        public=[o for o in ops if o['resource']=='public_step'];transitions=[e for e in scoped if e['kind']=='returned_transition']
        observations=[e for e in scoped if e['kind']=='fresh_observation'];audit_labels(row,plan,scoped,p)
        audit_feedback(row,plan,scoped,p)
        for phase,label,commands,limit in (('warmup','warmup',moving.warmup_commands(),10),('plan','own',raw25(plan['commands']),25)):
            current=[o for o in public if o['phase']==phase]
            require(len(current)<=limit and [o['step'] for o in current]==list(range(1,len(current)+1))
                and all(o['raw_bits']==commands[j].tobytes().hex() for j,o in enumerate(current))
                and row[label+'_attempted']==len(current) and row[label+'_returned']==sum(o['status']=='returned' for o in current),
                'complete contiguous warm10/raw25 authority, no padding/change/duplicate')
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
                and 0<=step<=(10 if phase=='warmup' else 25) and overlay in (0,1,2)
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
            and [e['response'] for e in scoped if e['kind']=='own_response']==responses,'all actual full25 response labels retained')
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
            and row['missing_checkpoints']==[b for b in CP if b not in [f['step'] for f in cps]],'cp0/5/10/15/20/25 exact and explicit missing')
        if row['status'] in ('horizon_completed','native_stopped'):
            require(row['first_error'] is None and row['secondary_errors']==[],'no engineering failure relabelled as native stop')
            keys=[('warmup',b,k) for b in range(row['warmup_returned']+1) for k in (0,1,2)]
            if row['initial_exact']:keys += [('plan',b,k) for b in range(row['own_returned']+1) for k in (0,1,2)]
            require([(e['receipt']['phase'],e['receipt']['step'],e['receipt']['overlay_index']) for e in observations]==keys,
                'all real unpadded full25 off/on/off triplets')
            for j in range(0,len(observations),3):
                a,b,c=observations[j:j+3]
                require(a['receipt']['file_sha256']==c['receipt']['file_sha256'] and a['receipt']['pixel_sha256']==c['receipt']['pixel_sha256']
                    and a['receipt']['state']==b['receipt']['state']==c['receipt']['state'],'off/on/off closes without dynamics change')
            if responses:
                require(row['warmup_returned']==10 and row['initial_exact'] and row['own_returned']==len(responses)
                    and ((row['status']=='native_stopped')==(last['terminated'] or last['truncated'])),'native stop retained without padding')
            require(len(responses)==row['own_returned'],'all successful raw25 returns have own responses')
            if row['status']=='native_stopped' and not responses:
                goal=root['goal'] if row['initial_exact'] else p['warm_traces'][plan['root_slot']][0]['pre']['goal_state']
                final_observation=observations[-1]['receipt'] if observations else None
                require(final_observation is not None and (native.metrics(goal,final_observation['state'])['native_success']
                    or bool(transitions and (transitions[-1]['terminated'] or transitions[-1]['truncated']))),
                    'zero-own native stop requires authenticated actual terminal/truncation, not arbitrary flag')
            if row['status']=='horizon_completed':require(row['own_returned']==25,'actual full25, not full10')
            kinds={k:sum(o['resource']==k and o['status']=='returned' for o in ops) for k in contract()['caps']}
            require(kinds['constructor']==kinds['close']==1 and kinds['reset']==2
                and kinds['set_state']==4 and kinds['goal_render']==2 and kinds['goal_bind']==int(row['initial_exact'])
                and kinds['goal_setter']==2+int(row['initial_exact'])
                and kinds['physics_step']==4+10*kinds['public_step'] and kinds['explicit_render']==row['PNG'],
                'per-branch actual full native lifecycle census')
    if r['status']==RECORDED:
        require(closed==2 and r['errors']==[] and r['inputs_final_verified'] is True and type(r['wall_seconds']) in (int,float) and np.isfinite(r['wall_seconds']) and 0<=r['wall_seconds']<WALL
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
    require(len(png)<=222 and all(len(v)<=PNG_CAP for v in png)
        and len(journal)<=JOURNAL_CAP and other<=OTHER_CAP and sum(map(len,files.values()))<=CAP,
        'full metadata/PNG/journal budget proof')
    return dict(PNG=len(png),PNG_bytes=sum(map(len,png)),journal_bytes=len(journal),other_bytes=other,
                total_bytes=sum(map(len,files.values())))


def shape_budget(files):
    """Full-field RAM bound, floats32 chars, all PNGs at their individual hard cap."""
    byte_census(files)
    def bound(x):
        if type(x) is float:
            require(np.isfinite(x) and len(json.dumps(x))<=32,'finite JSON float width')
            return 32
        if type(x) is dict:
            return 2+max(0,len(x)-1)*2+sum(len(json.dumps(k))+2+bound(v) for k,v in x.items())
        if type(x) is list:return 2+max(0,len(x)-1)*2+sum(bound(v) for v in x)
        return len(json.dumps(x,allow_nan=False))
    journal=sum(bound(native.decode(line))+1 for line in files['journal.jsonl'].splitlines())
    other=sum(bound(native.decode(b))+1 for n,b in files.items() if n!='journal.jsonl' and not n.endswith('.png'))
    png=sum(n.endswith('.png') for n in files)*PNG_CAP
    require(journal<=JOURNAL_CAP and other<=OTHER_CAP and png+journal+other+FINAL<=CAP,
            'complete fields shape budget exceeds approved groups')
    return dict(PNG_hard_bound=png,journal_float_width_bound=journal,metadata_float_width_bound=other,
                final_reserve=FINAL,total_with_reserve=png+journal+other+FINAL)


def saved_artifacts(directory):
    files=list(Path(directory).iterdir())
    require(all(f.is_file() and not f.is_symlink() for f in files) and sum(f.stat().st_size for f in files)<=CAP,
        'own128MiB bounded regular artifact census')
    result={f.name:f.read_bytes() for f in files}
    byte_census(result);return result


def consume(output,expected,completion_sha):
    native.actual_hash(expected);directory=Path(native.absolute(str(output)))
    files=saved_artifacts(directory)
    require(sum(map(len,files.values()))<=CAP and hashlib.sha256(files['report.json']).hexdigest()==expected,'actual own report SHA/128MiB')
    r=native.decode(files['report.json']);b=r['execution_binding'];p,_=verify(b['prepared'],b['prepare_sha256'])
    require(hashlib.sha256(files['execution_binding.json']).hexdigest()==r['execution_binding_sha256']
        and native.decode(files['execution_binding.json'])==b,'actual execution binding bytes')
    journal=files['journal.jsonl']
    require(journal.endswith(b'\n') and hashlib.sha256(journal).hexdigest()==r['journal_sha256'],'complete saved journal SHA')
    require(b'\n\n' not in journal and b'\r' not in journal,'native singleLF records no empty lines')
    events=[native.decode(line) for line in journal.splitlines()];rows=validate_saved(r,events,p)
    require(hashlib.sha256(files['completion.json']).hexdigest()==completion_sha,'external completion SHA')
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
        fixed_denominator=2,known_roots=2,fresh_roots=0,inherited=0,saved_PNG_verified=n,
        full_upstream_rebuilt_here=False,own_native_saved_reconstructed_here=True,byte_census=byte_census(files),
        prepare_sha256=b['prepare_sha256'],completion=completion,completion_sha256=completion_sha,
        job_index=p['job_index'],policy_roster=policy_roster(p,rows),global_policy_slots=8,
        full_stage_wall_seconds=completion['wall_seconds'],
        physical_report_sha256=expected,planner_authority=p['planner_authority'],bindings=bindings,
        plans=deepcopy(p['plans']),missing_exports=deepcopy(p['planner_authority']['missing_exports']),
        both_job_trajectories_full25=all(x['own_returned']==25 for x in rows),
        metric_breakdown=[dict(root_slot=x['root_slot'],arm=x['arm'],
            metrics=None if x['final_state'] is None else native.metrics(x['goal'],x['final_state'])) for x in rows],
        P1_admission=False,P2_admission=False,P2_mechanism=False,scientific_admission=False,freshness=False,
        equal_FLOPs=False,equal_total_compute=False,independent_confirmation=False,physical_release=False,new_neural_calls=0,new_physics_calls=0)


def join(paths,shas):
    require(type(paths) is list and type(shas) is list and len(paths)==len(shas)==4,'fixed all4 job receipts')
    receipts=[]
    for i,(path,h) in enumerate(zip(paths,shas,strict=True)):
        require(sha(path)==h,'actual fixed-job receipt SHA')
        r=read(path,META)
        require(r['schema']==NAME+'.consume' and type(r['job_index']) is int and r['job_index']==i and
            r['planner_authority']['authorities']==PINS and len(r['rows'])==2 and r['global_policy_slots']==8,
            'fixed same model authority/all4 jobs')
        rehash(r['bindings']);receipts.append(r)
    rows=[]
    for i,r in enumerate(receipts):
        for pol,row in zip(('joint','prefix_chain'),r['rows'],strict=True):
            require(row['policy']==pol,'fixed policy row')
            rows.append(dict(base_index=i,policy=pol,status=row['status'],row=row))
    return dict(schema=NAME+'.joined_jobs',policy_slots=rows,fixed_denominator=8,new_trajectories=8,inherited_O=0,
        job_authorities=[dict(path=p,sha256=h) for p,h in zip(paths,shas,strict=True)],
        sum_job_whole_wall_seconds=sum(r['full_stage_wall_seconds'] for r in receipts),
        wall_scope='sum of four inclusive job stages; excludes between-job idle gaps, not orchestration wall',
        counts={k:{v:sum(r['counts'][k][v] for r in receipts) for v in ('attempted','completed')} for k in contract()['caps']},
        known_roots=2,fresh_roots=0,full_upstream_rebuilt_here=False,**planner.K.FLAGS)


# This version reuses the original physical implementation but builds a distinct
# execution authority. Candidate 1 is not a producer-selected model plan.
CANDIDATE_PREP = ROOT/'results/codex_p2_feedback_shared_candidate_20261030_v1_PREP01.json'
CANDIDATE_SHA = 'bda456029c4fd7d5f22cabb57662240d0f906652d52fd05b170024d0e7c8ca13'


def assemble_candidates(upstream, recipe, slot):
    p=deepcopy(upstream);i=p['job_index']
    require(recipe['base']==i and slot['base_index']==i and slot['policy']=='F', 'same base F authority')
    observed_job=deepcopy(p['model_jobs'][1])
    selected_plan=deepcopy(p['plans'][1])
    require(observed_job['policy']=='F' and selected_plan['new_model_plan']['policy']=='F',
            'genuine observed feedback input')
    require(recipe['feedback_context_sha256']==planner.digest(observed_job['input_context']),
            'candidate observed full context')
    plans=[]
    for index,entry in enumerate(recipe['arms']):
        require(entry['arm']==('selected','fixed_index_1')[index], 'fixed arms')
        candidate_index=entry['candidate_index']
        require(type(candidate_index) is int and 0<=candidate_index<5 and
                (index==0 or candidate_index==1), 'fixed candidate index')
        ev=slot['result']['evaluations'][candidate_index]
        require(ev['index']==candidate_index and ev['Q']==entry['model_Q'], 'candidate exact source')
        model=deepcopy(selected_plan['new_model_plan'])
        model.update(commands=deepcopy(entry['raw25']),raw_float32_bits=entry['raw_float32_hex'],
                     suffix_logits=deepcopy(ev['logits']),new_contexts=deepcopy(ev['contexts']),
                     new_suffix_Q=ev['Q'],old_fields_scope='derived candidate for physical replay; not a new producer selection')
        model.pop('new_witness_sha256');model['new_witness_sha256']=planner.digest(model)
        check_new_plan(model,observed_job,p['expected_model_hash'])
        if index==0:
            require(entry['raw_float32_hex']==selected_plan['new_model_plan']['raw_float32_bits'],
                    'selected action source exact')
        plan=deepcopy(selected_plan)
        plan.update(schema=NAME+'.execution_plan',index=index,policy=entry['arm'],
                    commands=deepcopy(entry['raw25']),new_model_plan=model,
                    candidate_index=candidate_index,candidate_source='saved_F_candidate',
                    candidate_model_witness_scope='derived from authenticated candidate; not producer selection',
                    raw_float32_sha256=entry['raw_float32_sha256'])
        plan.pop('selection_digest');plan['selection_digest']=digest(plan)
        plans.append(plan)
    p.update(schema=NAME+'.prepare',contract=contract(),plans=plans,
             model_jobs=[deepcopy(observed_job),deepcopy(observed_job)],
             candidate_recipe=deepcopy(recipe))
    p['planner_authority']['plans_sha256']=digest(plans)
    p['planner_authority']['candidate_prepare_sha256']=CANDIDATE_SHA
    p['planner_authority']['evidence']='pinned prior F receipt and independently rebuilt saved candidate recipe; no new model'
    validate_execution(p)
    return p


def validate_execution(p):
    require(p['schema']==NAME+'.prepare' and p['contract']==contract() and p['arguments']==PINS and
            type(p['job_index']) is int and p['job_index'] in range(4) and
            len(p['plans'])==len(p['model_jobs'])==2 and len(p['bases'])==4,
            'fixed four jobs and two candidate arms')
    require(p['planner_authority']['plans_sha256']==digest(p['plans']) and
            p['planner_authority']['candidate_prepare_sha256']==CANDIDATE_SHA and
            p['planner_authority']['authorities']==PINS and
            p['planner_authority']['full_upstream_rebuilt_here'] is False, 'candidate and parent authority')
    require(len(p['prefix_reference'])==106 and
            sum(e['kind']=='fresh_observation' for e in p['prefix_reference'])==81 and
            sum(e['kind']=='returned_transition' for e in p['prefix_reference'])==25,
            'complete native/RGB prefix')
    recipe=p['candidate_recipe']
    require(recipe['base']==p['job_index'] and p['model_jobs'][0]==p['model_jobs'][1] and
            recipe['feedback_context_sha256']==planner.digest(p['observed_context'])==
            planner.digest(p['model_jobs'][0]['input_context']), 'same observed context')
    for i,(plan,job,entry) in enumerate(zip(p['plans'],p['model_jobs'],recipe['arms'],strict=True)):
        require(plan['index']==i and plan['schema']==NAME+'.execution_plan' and
                plan['policy']==entry['arm']==('selected','fixed_index_1')[i] and
                plan['candidate_index']==entry['candidate_index'] and
                (i==0 or plan['candidate_index']==1) and
                plan['selection_digest']==digest({k:v for k,v in plan.items() if k!='selection_digest'}),
                'fixed selected/candidate1 identities')
        check_new_plan(plan['new_model_plan'],job,p['expected_model_hash'])
        require(plan['commands']==entry['raw25']==plan['new_model_plan']['commands'] and
                raw25(plan['commands']).tobytes().hex()==entry['raw_float32_hex'] and
                hashlib.sha256(raw25(plan['commands']).tobytes()).hexdigest()==entry['raw_float32_sha256']==plan['raw_float32_sha256'] and
                plan['task_id']==recipe['task_id'] and plan['planner_plan']['witness_sha256']==recipe['parent_witness_sha256'],
                'exact candidate/task/raw bits')


def material(job_index):
    require(type(job_index) is int and job_index in range(4), 'fixed job before reads')
    upstream,data=original.material(job_index)
    require(sha(CANDIDATE_PREP)==CANDIDATE_SHA, 'fixed candidate recipe SHA')
    prepared=read(CANDIDATE_PREP,META)
    require(prepared==candidates.build(), 'saved candidates fully reconstructed')
    report=candidates.source.read(str(candidates.PATHS['report']),candidates.source.PINS['report'])
    slotname=f'slot{2*job_index+1}.json.gz'
    slot=candidates.source.read(str(candidates.PATHS['report'].parent/slotname),
                                report['artifact_sha256'][slotname],True)
    p=assemble_candidates(upstream,prepared['pairs'][job_index],slot)
    native.merge(p['bindings'],{str(CANDIDATE_PREP):CANDIDATE_SHA,
        str(Path(__file__).resolve()):sha(__file__),
        str(Path(candidates.__file__).resolve()):sha(candidates.__file__),
        str(ROOT/'tests'/('test_'+NAME+'.py')):sha(ROOT/'tests'/('test_'+NAME+'.py')),
        str(ROOT/'innerdocs'/(NAME+'.md')):sha(ROOT/'innerdocs'/(NAME+'.md'))})
    rehash(p['bindings'])
    require(len(raw(p))<=META, 'bounded prepare')
    return p,data


def cp15_assemble(upstream, source_binding):
    """Distinct CUDA authority; do not reinterpret the old model-plan witness."""
    p=deepcopy(upstream);base=p['job_index']
    joined=cpbridge.assemble(base, upstream, source_binding)
    plans=joined['plans']
    for plan in plans:
        plan.pop('selection_digest')
        plan['selection_digest']=digest(plan)
    p.update(schema=NAME+'.prepare',contract=contract(),plans=plans,
             cp15_binding=deepcopy(source_binding),cp15_bridge_schema=joined['schema'])
    p['planner_authority'].update(producer=cpbridge.NAME,authorities=deepcopy(PINS),
        plans_sha256=digest(plans),cuda_report_sha256=cpbinding.REPORT_SHA,
        cuda_prepare_sha256=cpbinding.PREP_SHA,cuda_receipt_sha256=cpbinding.RECEIPT_SHA,
        full_upstream_rebuilt_here=False,missing_exports=[],
        evidence='CUDA saved consumer reconstructed; original native provenance retained; no model reexecution')
    validate_execution(p)
    return p


def validate_execution(p):
    require(p['schema']==NAME+'.prepare' and p['contract']==contract() and
            type(p['job_index']) is int and p['job_index'] in range(4) and
            len(p['plans'])==2 and len(p['bases'])==4, 'fixed cp15 four jobs/two arms')
    authority=p['planner_authority']
    require(authority['authorities']==PINS and authority['plans_sha256']==digest(p['plans']) and
            authority['full_upstream_rebuilt_here'] is False and authority['missing_exports']==[] and
            authority['cuda_report_sha256']==cpbinding.REPORT_SHA and
            authority['cuda_prepare_sha256']==cpbinding.PREP_SHA and
            authority['cuda_receipt_sha256']==cpbinding.RECEIPT_SHA, 'CUDA authority')
    b=p['cp15_binding']
    require(b['base']==p['job_index'] and b['prefix_reference']==p['prefix_reference'] and
            b['observed_context']==p['observed_context'] and len(p['prefix_reference'])==106 and
            sum(e['kind']=='fresh_observation' for e in p['prefix_reference'])==81 and
            sum(e['kind']=='returned_transition' for e in p['prefix_reference'])==25,
            'actual cp15 complete reference and observed context')
    for i,(plan,entry) in enumerate(zip(p['plans'],b['plans'],strict=True)):
        model=plan['new_model_plan']
        require(plan['index']==i and plan['base_index']==p['job_index'] and
                plan['schema']==cpbridge.NAME+'.execution_plan' and
                plan['policy']==entry['arm']==('joint','prefix_chain')[i] and
                plan['selection_digest']==digest({k:v for k,v in plan.items() if k!='selection_digest'}),
                'fixed new spectral plan')
        require(model['schema']==cpbridge.NAME+'.cuda_winner_reference' and
                model['new_witness_sha256']==cpbridge.digest(
                    {k:v for k,v in model.items() if k!='new_witness_sha256'}) and
                model['input_context']==p['observed_context'] and model['arm']==plan['policy'] and
                model['base']==p['job_index'] and
                model['cuda_report_sha256']==cpbinding.REPORT_SHA and
                model['cuda_prepare_sha256']==cpbinding.PREP_SHA and
                model['cuda_receipt_sha256']==cpbinding.RECEIPT_SHA and
                model['suffix_raw_sha256']==entry['suffix_raw_sha256'] and
                model['feedback_context_sha256']==entry['feedback_context_sha256'],
                'new CUDA witness and context identity')
        rawbytes=raw25(plan['commands']).tobytes()
        require(rawbytes.hex()==entry['full_raw_float32_hex']==plan['raw_float32_hex'] and
                hashlib.sha256(rawbytes).hexdigest()==plan['raw_float32_sha256'] and
                rawbytes[120:].hex()==entry['suffix_raw_float32_hex'] and
                plan['task_id']==entry['task_id'] and plan['root_slot']==entry['root'],
                'actual CUDA raw bits, task and root')


def material(job_index):
    require(type(job_index) is int and job_index in range(4), 'fixed job before reads')
    upstream,data=original.material(job_index)
    source_path=ROOT/'results'/(cpbinding.NAME+f'_B{job_index}_PREP_01.json')
    require(source_path.is_file() and not source_path.is_symlink(), 'saved binding')
    b=read(source_path,META)
    require(b==cpbinding.build(job_index), 'independently reconstructed saved CUDA and binding')
    p=cp15_assemble(upstream,b)
    paths=[source_path,cpbinding.PREP,cpbinding.RECEIPT,cpbinding.RUN/'report.json',
           cpbinding.RUN/'events.jsonl',cpbinding.RUN/'prepare.json',
           Path(__file__).resolve(),Path(cpbridge.__file__).resolve(),
           Path(cpbinding.__file__).resolve(),
           ROOT/'experiments/spectral_reachability/codex_p2_cp15_physical_saved_20261030_v1.py',
           ROOT/'experiments/spectral_reachability/codex_p2_cp15_physical_partial_20261030_v1.py',
           ROOT/'tests'/('test_'+NAME+'.py'),
           ROOT/'innerdocs'/(NAME+'.md')]
    native.merge(p['bindings'],{str(path):sha(path) for path in paths})
    rehash(p['bindings'])
    require(len(raw(p))<=META, 'bounded full cp15 prepare')
    return p,data


def h25_assemble(upstream,b):
    p=deepcopy(upstream); base=p['job_index']
    require(b['base']==base and b['schema']==program_binding.NAME+'.binding'
            and b['physical_ready'] is False and b['scientific_pass'] is False
            and b['observed_context']==p['observed_context']
            and b['prefix_reference']==p['prefix_reference']
            and b['original_bindings']==p['bindings'] and len(b['plans'])==2,
            'H25 original identity and exact feedback reference')
    parent_plan=p['plans'][1]
    prefix=np.asarray(parent_plan['commands'],dtype='<f4')[:15].tobytes()
    plans=[]
    for i,(arm,entry) in enumerate(zip(('joint','prefix_chain'),b['plans'],strict=True)):
        bits=bytes.fromhex(entry['full25_raw_float32_hex'])
        suffix=bytes.fromhex(entry['winner_suffix_float32_hex'])
        require(entry['base']==base and entry['arm']==arm and entry['horizon']==25
                and entry['task_id']==parent_plan['task_id'] and entry['root']==parent_plan['root_slot']
                and len(bits)==200 and len(suffix)==80 and bits[:120]==prefix and bits[120:]==suffix
                and hashlib.sha256(prefix).hexdigest()==entry['prefix_sha256']
                and hashlib.sha256(suffix).hexdigest()==entry['winner_suffix_sha256'],
                'fixed winner literal task prefix and suffix')
        a=np.frombuffer(bits,dtype='<f4').reshape(25,2)
        require(np.isfinite(a).all() and np.all(np.abs(a)<=1),'H25 action domain')
        witness=dict(schema=NAME+'.cuda_winner_reference',base=base,arm=arm,
            input_context=deepcopy(p['observed_context']),winner_index=entry['winner_index'],
            program_ids=deepcopy(entry['program_ids']),suffix_sha256=entry['winner_suffix_sha256'],
            cuda_report_sha256=program_binding.REPORT_SHA,cuda_receipt_sha256=program_binding.RECEIPT_SHA,
            scope='saved CUDA winner reference; not parent model witness')
        witness['new_witness_sha256']=planner.digest(witness)
        plan=deepcopy(parent_plan)
        plan.update(schema=NAME+'.execution_plan',index=i,base_index=base,policy=arm,
            commands=a.astype(float).tolist(),raw_float32_hex=bits.hex(),
            raw_float32_sha256=hashlib.sha256(bits).hexdigest(),new_model_plan=witness)
        plan.pop('selection_digest',None);plan['selection_digest']=digest(plan)
        plans.append(plan)
    p.update(schema=NAME+'.prepare',contract=contract(),plans=plans,program_binding=deepcopy(b))
    p['planner_authority'].update(producer=program_binding.NAME,authorities=deepcopy(PINS),
        plans_sha256=digest(plans),cuda_report_sha256=program_binding.REPORT_SHA,
        cuda_receipt_sha256=program_binding.RECEIPT_SHA,full_upstream_rebuilt_here=False,
        missing_exports=[],evidence='saved CUDA reconstructed; native source inherited')
    return p


def validate_execution(p):
    require(p['schema']==NAME+'.prepare' and p['contract']==contract()
        and type(p['job_index']) is int and p['job_index'] in range(4)
        and len(p['plans'])==2 and len(p['bases'])==4,'fixed H25 batch')
    b=p['program_binding']; authority=p['planner_authority']
    require(authority['authorities']==PINS and authority['plans_sha256']==digest(p['plans'])
        and authority['full_upstream_rebuilt_here'] is False and authority['missing_exports']==[]
        and authority['cuda_report_sha256']==program_binding.REPORT_SHA
        and authority['cuda_receipt_sha256']==program_binding.RECEIPT_SHA
        and b['base']==p['job_index'] and b['prefix_reference']==p['prefix_reference']
        and b['observed_context']==p['observed_context'] and len(p['prefix_reference'])==106,
        'H25 source and authority')
    for i,(plan,entry) in enumerate(zip(p['plans'],b['plans'],strict=True)):
        w=plan['new_model_plan']; bits=raw25(plan['commands']).tobytes()
        require(plan['index']==i and plan['base_index']==p['job_index']
            and plan['policy']==entry['arm']==('joint','prefix_chain')[i]
            and plan['selection_digest']==digest({k:v for k,v in plan.items() if k!='selection_digest'})
            and bits.hex()==entry['full25_raw_float32_hex']==plan['raw_float32_hex']
            and hashlib.sha256(bits).hexdigest()==plan['raw_float32_sha256']
            and plan['task_id']==entry['task_id'] and plan['root_slot']==entry['root']
            and w['input_context']==p['observed_context'] and w['base']==p['job_index']
            and w['arm']==plan['policy'] and w['winner_index']==entry['winner_index']
            and w['program_ids']==entry['program_ids'] and w['suffix_sha256']==entry['winner_suffix_sha256']
            and w['cuda_report_sha256']==program_binding.REPORT_SHA
            and w['cuda_receipt_sha256']==program_binding.RECEIPT_SHA
            and w['new_witness_sha256']==planner.digest({k:v for k,v in w.items() if k!='new_witness_sha256'}),
            'fixed winner, context and action identity')


def material(job_index):
    require(type(job_index) is int and job_index in range(4),'fixed H25 job')
    upstream,data=original.material(job_index)
    b=program_binding.build(job_index)
    p=h25_assemble(upstream,b)
    paths=[program_binding.PREP,program_binding.RECEIPT,program_binding.RUN/'report.json',
        program_binding.RUN/'supervisor_events.jsonl',program_binding.RUN/'child_events.jsonl',
        Path(program_binding.__file__).resolve(),Path(__file__).resolve(),
        ROOT/'experiments/spectral_reachability/codex_p2_prefix_recompile_physical_saved_20261030_v1.py',
        ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]
    paths.extend(program_binding.RUN/f'branch_{i}.json' for i in range(8))
    cuda_prepare=program_binding.saved.host.verify(program_binding.PREP,program_binding.PREP_SHA)
    paths.extend(Path(row['path']) for row in cuda_prepare['own_sources'])
    paths.extend(Path(row['path']) for row in cuda_prepare['parent']['sources'])
    native.merge(p['bindings'],{str(x):sha(x) for x in paths})
    rehash(p['bindings']);validate_execution(p)
    require(len(raw(p))<=META,'bounded H25 prepare')
    return p,data


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=('prepare','verify','run','consume','join'))
    for k in ('output','prepared','prepare-sha','report-sha','completion-sha','receipt','authorization'):
        ap.add_argument('--'+k)
    ap.add_argument('--job',type=int)
    ap.add_argument('--job-receipts',nargs=4)
    ap.add_argument('--job-receipt-shas',nargs=4)
    a=ap.parse_args()
    if a.command=='prepare':result=prepare(a.output,a.job)
    elif a.command=='verify':
        p,_=verify(a.prepared,a.prepare_sha);result=dict(verified=True,job_index=p['job_index'],plans=2)
    elif a.command=='run':
        require(a.authorization==AUTH,'MAIN explicit authority before gym import')
        def factory():
            import gymnasium as gym
            import stable_worldmodel.envs
            return gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224,with_target=False,render_action=False)
        r=run(a.prepared,a.prepare_sha,a.output,factory,authorization=a.authorization)
        result=dict(output=a.output,status=r['status'],errors=r['errors'],
            report_sha256=sha(Path(a.output)/'report.json'),completion_sha256=sha(Path(a.output)/'completion.json'))
    elif a.command=='join':result=write_new(a.receipt,join(a.job_receipts,a.job_receipt_shas))
    else:result=write_new(a.receipt,consume(a.output,a.report_sha,a.completion_sha))
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
