"""Ten fresh-env settling tails, own journal and independent saved reconstruction.

No environment is imported/constructed until the explicit run entry point.
"""
from copy import deepcopy
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import struct
import time

import numpy as np
from PIL import Image

from experiments.spectral_reachability import agent_p3_same_time_capability_probe_physical_20261029_v2 as old
from experiments.spectral_reachability import agent_p3_settling_cp15_guard_20261029_v1 as gate

NAME = 'agent_p3_settling_tail_physical_20261029_v1'
ROOT = Path(__file__).resolve().parents[2]
GUARD_SHA = 'f2cc64780931a76a86cc5859370cab515882b31f55a86c8983e12326bbb4aa5c'
AUTH = 'MAIN_P3_SETTLING_10_FRESH_ENVS_V1'
COUNTS = gate.CAPS
require, same, digest, raw, sha, read = old.require, old.same, old.digest, old.raw, old.sha, old.read
FLAGS = dict(old.FLAGS)


def limits(rows):
    # Upper bound the full source-observation structure, not a guessed small schema.
    observation = max(old.numeric_bound(e) for r in rows for e in r['old_row']['observations'])
    op, guard = 8192, 512*1024
    journal = 888*op+96*observation+3*guard+128*1024
    report = 444*op+96*observation+3*guard+256*1024
    total = 96*old.png_budget()['png_upper_bytes']+sum(old.QUOTAS.values())+old.RESERVE
    require(journal<=old.QUOTAS['journal.jsonl'] and report<=old.QUOTAS['report.json']
            and total<=old.CAP, 'complete 96-PNG metadata bound exceeds sealed budget')
    return dict(PNG=96,operations=444,operation_events=888,observation_events=96,guards=3,
        lifecycle=3,total_events=990,png_upper_bytes=old.png_budget()['png_upper_bytes'],
        observation_upper_bytes=observation,operation_event_upper_bytes=op,guard_upper_bytes=guard,
        journal_upper_bytes=journal,report_upper_bytes=report,quotas=old.QUOTAS,
        conservative_total_bytes=total,cap_bytes=old.CAP,reserve_bytes=old.RESERVE,
        disk_floor_bytes=old.FLOOR,wall_seconds=old.WALL,counts=COUNTS)


def material():
    require(sha(str(Path(gate.__file__).resolve()))==GUARD_SHA, 'frozen independent cp15 guard source')
    refs,rows=gate.load_references()
    source_prepare=read(refs['prepare']['path'])
    fwref=source_prepare['authority']['full_witness']
    require(sha(fwref['path'])==fwref['sha256'], 'full witness source')
    witness=read(fwref['path'])
    require(witness['schema']==old.fw.NAME+'.witness' and witness['actual_MAIN_authority'] is True
            and len(witness['rows'])==5, 'full five actual cp10 witnesses')
    source_bindings=deepcopy(source_prepare['bindings'])
    for ref in refs.values(): source_bindings[ref['path']]=ref['sha256']
    source_bindings[fwref['path']]=fwref['sha256']
    for f in (Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):
        source_bindings[str(f)]=sha(str(f))
    source_bindings[str(Path(gate.__file__).resolve())]=GUARD_SHA
    require(all(sha(f)==h for f,h in source_bindings.items()), 'full source closure before prepare')
    p=dict(schema=NAME+'.prepare',authority=refs,old_prepare_sha256=refs['prepare']['sha256'],
        jobs=deepcopy(source_prepare['jobs']),goal_payloads=deepcopy(source_prepare['goal_payloads']),
        full_witness=fwref,bindings=source_bindings,limits=limits(rows),tail_raw5=deepcopy(gate.TAIL),
        fixture_only=False,known_roots=2,fresh_roots=0,**FLAGS)
    return p,witness,rows


def prepare(output):
    require(not Path(output).exists(),'exclusive prepare')
    p,_,_=material();return old.write_new(output,p)


def verify(prepared,prepare_sha):
    require(sha(prepared)==prepare_sha,'explicit prepare SHA')
    p=read(prepared);expected,w,rows=material()
    require(same(p,expected) and sha(prepared)==prepare_sha,'whole prepare/source reconstruction')
    return p,w,rows


class Store(old.Store):
    def write(self,name,body,*,append=False,final=False):
        require(Path(name).name==name and name not in ('','.','..'),'flat owned artifact')
        p=self.path/name;require(not p.is_symlink(),'no symlink output')
        if name.endswith('.png'):
            require(not append and len(body)<=self.limits['png_upper_bytes']
                    and len(list(self.path.glob('*.png')))<96,'96 bounded PNG')
        else:
            require(name in old.QUOTAS and (not append or name=='journal.jsonl'),'owned metadata names')
            require((p.stat().st_size if p.exists() else 0)+len(body)<=old.QUOTAS[name], 'metadata quota')
        self.check(len(body),final)
        with p.open('ab' if append else 'xb') as f:
            require(f.write(body)==len(body),'complete byte write');f.flush();os.fsync(f.fileno())
        old.fsync_directory(self.path)
        return sha(str(p))


def guard81_proof(job,original,observations,operations,folder=None):
    proof=gate.guard81(job,original,observations,operations)
    frames=[]
    for event in observations:
        f=event['receipt'];name=f['png'];require(Path(name).name==name,'owned pre-tail PNG filename')
        if folder is not None:
            path=Path(folder)/name
            require(path.is_file() and not path.is_symlink() and path.stat().st_size<=old.png_budget()['png_upper_bytes'], 'bounded regular pre-tail PNG')
            body=path.read_bytes();require(sha(str(path))==f['file_sha256'],'pre-tail PNG actual byte identity')
            old.check_png_stream(body)
            with Image.open(io.BytesIO(body)) as image:
                require(image.mode=='RGB' and image.size==(224,224) and image.format=='PNG','pre-tail PNG RGB224')
                pixels=np.asarray(image).copy()
            require(hashlib.sha256(pixels.tobytes()).hexdigest()==f['pixel_sha256'],'pre-tail actual pixel identity')
        frames.append(dict(png=name,file_sha256=f['file_sha256'],pixel_sha256=f['pixel_sha256']))
    return dict(full_guard=proof,pre_tail_all81_files=frames)


class Meter:
    def __init__(self,store,job,reference,original,public_encoder=lambda x:x,reset_encoder=lambda x:x):
        self.store,self.job,self.reference,self.original=store,job,reference,original
        self.public_encoder,self.reset_encoder=public_encoder,reset_encoder
        self.where=dict(branch=job['index'],root_slot=job['root_slot'],arm=job['probe'],condition='off',selection_digest=job['selection_digest'])
        self.counts={k:dict(attempted=0,completed=0) for k in COUNTS};self.active={k:0 for k in COUNTS}
        self.receipts=[];self.guarded=set();self.secondary=[]

    def call(self,resource,fn,*,cleanup=False,**fields):
        require(resource in COUNTS and (not cleanup or resource=='close'),'known resource and cleanup')
        if not cleanup:self.store.check()
        if resource=='public_step' and fields.get('phase')=='plan' and fields['step'] in (6,11,16):
            n={6:51,11:66,16:81}[fields['step']]
            try:
                proof=(guard81_proof(self.job,self.original,self.store.observations,self.receipts,self.store.path) if n==81
                       else old.guard_proof(self.job,self.reference,self.store.observations,self.receipts,n))
            except BaseException as e:
                self.store.emit('guard',n=n,accepted=False,proof=None,error=old.error(e));raise
            self.store.emit('guard',n=n,accepted=True,proof=proof,error=None)
            self.guarded.add(n)  # Store.emit writes+fsyncs first.
        if resource=='public_step' and fields.get('phase')=='plan':
            s=fields['step'];require(s<=5 or 51 in self.guarded,'causal cp5 guard')
            require(s<=10 or 66 in self.guarded,'causal cp10 guard')
            require(s<=15 or 81 in self.guarded,'causal cp15 guard')
        require(self.counts[resource]['attempted']<COUNTS[resource],'resource attempt cap')
        op=dict(index=len(self.receipts),resource=resource,**self.where,**fields,status='pending',result=None,error=None)
        self.store.emit('operation_dispatch_intent',receipt=op)
        self.receipts.append(op);self.counts[resource]['attempted']+=1;self.active[resource]+=1
        try:
            value=fn();op['result']=self.public_encoder(value) if resource=='public_step' else self.reset_encoder(value) if resource=='reset' else None
            op['status']='returned';self.counts[resource]['completed']+=1
        except BaseException as e:
            op.update(status='failed',error=old.error(e));self.store.emit('operation_failed',receipt=op);raise
        finally:self.active[resource]-=1
        self.store.emit('operation_returned',receipt=op)
        return value


def read_events(output):
    return old.read_events(output)


def reconstruct(binding,reference,original,events,*,folder=None):
    job=binding['job'];counts={k:dict(attempted=0,completed=0) for k in COUNTS}
    ops=[];stack=[];observations=[];guards=[];closed=None;finished=None
    require(events and set(events[0])=={'index','previous_sha256','event_sha256','kind','binding_sha256'}
        and events[0]['kind']=='run_started' and events[0]['binding_sha256']==digest(binding),'binding event')
    for e in events[1:]:
        k=e['kind'];base={'index','previous_sha256','event_sha256','kind'}
        extra={'operation_dispatch_intent':{'receipt'},'operation_returned':{'receipt'},'operation_failed':{'receipt'},
               'fresh_observation':{'branch','root_slot','arm','condition','selection_digest','receipt','pre','post'},
               'guard':{'n','accepted','proof','error'},'branch_closed':{'result'},'run_finished':{'result'}}
        require(k in extra and set(e)==base|extra[k], 'owned event schema')
        if k not in ('branch_closed','run_finished'):require(closed is None,'no operation or observation after closure')
        if k=='operation_dispatch_intent':
            op=e['receipt'];r=op['resource'];require(r in counts and op['index']==len(ops) and op['status']=='pending'
                and op['result'] is None and op['error'] is None,'contiguous operation intent')
            require((op['branch'],op['root_slot'],op['arm'],op['condition'],op['selection_digest'])==
                (job['index'],job['root_slot'],job['probe'],'off',job['selection_digest']),'whole job scope')
            if r=='reset':
                resets=[x for x in ops if x['resource']=='reset'];recipe=reference['placement_events'][0]
                if len(resets)==0:
                    require(op['role']=='seed' and op['seed']==reference['source_slot_metadata']['reset_seed'],'original reset seed')
                else:
                    require(len(resets)==1 and op['role']=='placement' and op['seed']==0
                            and same(op['requested_state'],recipe['requested_state']),'original placement reset, no cp15 restore')
            if r=='public_step':
                prior=[x for x in ops if x['resource']=='public_step'];i=len(prior)
                commands=job['warm10']+job['raw15']+gate.TAIL
                require(i<30 and (op['phase'],op['step'])==(('warmup',i+1) if i<10 else ('plan',i-9))
                        and op['raw_bits']==old.bits(commands,30)[i], 'exact full30 action order')
                require(not any(x['status']=='returned' and (x['result']['terminated'] or x['result']['truncated']) for x in prior), 'no actions after native stop')
                require(i<15 or 51 in guards,'cp5 before control6');require(i<20 or 66 in guards,'cp10 before control11')
                require(i<25 or 81 in guards,'cp15 before control16')
            counts[r]['attempted']+=1;require(counts[r]['attempted']<=COUNTS[r], 'no resource refunds')
            active=[ops[i]['resource'] for i in stack]
            if r=='set_state':require('reset' in active,'no restore')
            if r=='goal_setter':require('reset' in active or 'goal_bind' in active,'native goal setter scope')
            if r=='physics_step':require(not set(active)&{'explicit_render','goal_render','goal_bind'},'no hidden render physics')
            stack.append(op['index']);ops.append(deepcopy(op))
        elif k in ('operation_returned','operation_failed'):
            op=e['receipt'];require(stack and stack.pop()==op['index'],'nested operation terminal')
            require(same(dict(op,status='pending',result=None,error=None),ops[op['index']])
                and op['status']==('returned' if k=='operation_returned' else 'failed'), 'intent/terminal identity')
            if op['resource']=='public_step' and op['status']=='returned':
                require(op['result']['actual_bits']==op['raw_bits'] and op['result']['actual_dtype']=='<f4','native raw bit return')
                goal=reference['placement_events'][0]['native_goal'] if op['phase']=='warmup' else reference['source_row']['goal']
                require(same(op['result']['goal_state'],goal),'native returned entire original goal7')
                old.response(goal,reference['source_row']['root_state'],op['step'],op['result'])
            if op['resource']=='reset' and op['status']=='returned':
                require(same(op['result']['goal_state'],reference['placement_events'][0]['native_goal']), 'reset native goal7')
            ops[op['index']]=deepcopy(op);counts[op['resource']]['completed']+=op['status']=='returned'
        elif k=='fresh_observation':
            f=e['receipt'];i=len(observations)
            order=[(ph,s,j) for ph,last in (('warmup',10),('plan',20)) for s in range(last+1) for j in range(3)]
            require(i<96 and (f['phase'],f['step'],f['overlay_index'])==order[i], 'fixed full96 sequence')
            require((e['branch'],e['root_slot'],e['arm'],e['condition'],e['selection_digest'])==
                (job['index'],job['root_slot'],job['probe'],'off',job['selection_digest']),'observation scope')
            require(same(e['pre'],e['post']) and same(f['snapshot'],e['post']['snapshot'])
                and same(f['state'],f['snapshot']['state']) and struct.pack('<7d',*f['state']).hex()==f['snapshot']['state_float64_bits']
                and f['condition']==('on' if f['overlay_index']==1 else 'off'),'inert full snapshot')
            goal=reference['placement_events'][0]['native_goal'] if f['phase']=='warmup' else reference['source_row']['goal']
            require(same(e['pre']['goal_state'],goal),'full observation original goal7, including agent velocity')
            if f['overlay_index']==2:
                require(same(e['pre'],observations[-2]['pre']) and same(e['pre'],observations[-1]['pre'])
                    and f['file_sha256']==observations[-2]['receipt']['file_sha256']
                    and f['pixel_sha256']==observations[-2]['receipt']['pixel_sha256'], 'off/on/off equality')
            if f['step']:
                a=[x for x in ops if x['resource']=='public_step' and x.get('phase')==f['phase'] and x.get('step')==f['step']]
                require(len(a)==1 and a[0]['status']=='returned' and all(same(f[key],a[0]['result'][key]) for key in ('state','snapshot')), 'fresh image bound native return')
            elif f['phase']=='warmup':
                resets=[x for x in ops if x['resource']=='reset']
                require(len(resets)==2 and resets[-1]['status']=='returned' and
                    all(same(f[key],resets[-1]['result'][key]) for key in ('state','snapshot')), 'warm0 from real placement reset')
            else:
                warm=[x for x in ops if x['resource']=='public_step' and x.get('phase')=='warmup']
                require(len(warm)==10 and warm[-1]['status']=='returned' and
                    all(same(f[key],warm[-1]['result'][key]) for key in ('state','snapshot')), 'plan0 from actual warm10')
            observations.append(e)
        elif k=='guard':
            n=e['n'];require(type(n) is int and n in (51,66,81) and n not in guards
                             and type(e['accepted']) is bool,'unique literal causal guard')
            def proof():
                return guard81_proof(job,original,observations,ops,folder) if n==81 else old.guard_proof(job,reference,observations,ops,n)
            if e['accepted']:
                require(e['error'] is None and same(e['proof'],proof()),'reconstruct full guard before dispatch')
                guards.append(n)
            else:
                require(e['proof'] is None and e['error'] is not None,'explicit rejected guard')
                rejected=False
                try:proof()
                except (ValueError,KeyError):rejected=True
                require(rejected,'guard failure reproducible')
        elif k=='branch_closed':
            require(closed is None and set(e['result'])=={'status','error','cleanup_errors'},'branch terminal')
            closed=e['result']
        else:
            require(k=='run_finished' and e is events[-1] and closed is not None,'final order');finished=e['result']
    require(closed is not None and same(closed,finished),'whole terminal closure')
    public=[x for x in ops if x['resource']=='public_step' and x['status']=='returned']
    responses=[old.response(reference['source_row']['goal'],reference['source_row']['root_state'],x['step'],x['result'])
               for x in public if x['phase']=='plan']
    checkpoints=[deepcopy(e['receipt']) for e in observations if e['receipt']['phase']=='plan'
                 and e['receipt']['step'] in (0,5,10,15,20) and e['receipt']['overlay_index']==0]
    if closed['status']=='horizon_completed':
        require(not stack and not closed['error'] and closed['cleanup_errors']==[] and len(observations)==96 and guards==[51,66,81]
            and all(c['attempted']==c['completed']==COUNTS[k] for k,c in counts.items()),'all444/96 complete')
    elif closed['status']=='native_stopped':
        require(not stack and not closed['cleanup_errors'] and public and
            (public[-1]['result']['terminated'] or public[-1]['result']['truncated']) and closed['error'] is None,'actual native early stop')
    else:require(closed['status']=='engineering_failed' and closed['error'] is not None,'explicit engineering failure')
    return dict(job=job,status=closed['status'],error=closed['error'],cleanup_errors=closed['cleanup_errors'],counts=counts,
        operations=ops,pending_operation_indices=stack,observations=observations,guards=guards,responses=responses,
        checkpoints=checkpoints,missing_checkpoints=[n for n in (0,5,10,15,20) if n not in [x['step'] for x in checkpoints]],
        endpoint15=next((x for x in checkpoints if x['step']==15),None),endpoint20=next((x for x in checkpoints if x['step']==20),None),
        unknowns=dict(**reference['unknown_semantics'],FLOPs=None,CUDA_ns=None),new_neural_calls=0,
        fixture_only=binding['fixture_only'],**FLAGS)


def publish(store,reference,original):
    events,tail=read_events(store.path);row=reconstruct(store.binding,reference,original,events,folder=store.path)
    for event in row['observations']:
        f=event['receipt'];path=store.path/f['png']
        require(Path(f['png']).name==f['png'] and path.is_file() and not path.is_symlink()
            and path.stat().st_size<=old.png_budget()['png_upper_bytes'], 'publication bounded PNG')
        body=path.read_bytes();require(sha(str(path))==f['file_sha256'], 'publication PNG byte identity')
        old.check_png_stream(body)
        with Image.open(io.BytesIO(body)) as image:
            require(image.mode=='RGB' and image.size==(224,224) and image.format=='PNG','publication RGB224')
            pixels=np.asarray(image).copy()
        require(hashlib.sha256(pixels.tobytes()).hexdigest()==f['pixel_sha256'], 'publication PNG pixel identity')
    report=dict(schema=NAME+'.report',binding=store.binding,row=row,journal_tail=tail,
        journal_sha256=sha(str(store.path/'journal.jsonl')),elapsed_seconds=store.clock()-store.started)
    store.put('report.json',report,final=True)
    files=list(store.path.iterdir())
    completion=dict(schema=NAME+'.completion',report_sha256=sha(str(store.path/'report.json')),
        artifact_sha256={f.name:sha(str(f)) for f in files},bytes_before_completion=sum(f.stat().st_size for f in files),
        inclusive_wall_seconds=store.clock()-store.started,wall_scope='through report hash; completion self-write excluded')
    store.put('completion.json',completion,final=True)
    if store.clock()-store.started>old.WALL:
        raise ValueError('completion publication exceeded per-job wall')
    return dict(output=str(store.path),report_sha256=sha(str(store.path/'report.json')),
                completion_sha256=sha(str(store.path/'completion.json')))


def failed_publication(store,meter,closed,exc):
    files=list(store.path.iterdir())
    value=dict(schema=NAME+'.publication_failure',binding_sha256=digest(store.binding),
        error=old.error(exc),closed=closed,observed_meter_counts=deepcopy(meter.counts),
        journal_sha256=sha(str(store.path/'journal.jsonl')) if (store.path/'journal.jsonl').exists() else None,
        artifact_sha256={f.name:sha(str(f)) for f in files},status='transport_failed_NOT_verified',
        unknown_final_callbacks=True,endpoint20=None)
    store.put('failure.json',value,final=True)
    return dict(output=str(store.path),status='transport_failed_NOT_verified',failure_sha256=sha(str(store.path/'failure.json')))


def consume_job(output,report_sha,completion_sha,binding,reference,original,*,permit_fixture=False):
    require(not binding['fixture_only'] or permit_fixture,'RAM fixture not production')
    out=Path(output);require(out.is_absolute() and not out.is_symlink(),'absolute job output')
    require(sha(str(out/'report.json'))==report_sha and sha(str(out/'completion.json'))==completion_sha,'actual saved report/completion')
    files=list(out.iterdir());require(all(f.is_file() and not f.is_symlink() for f in files)
        and not (out/'failure.json').exists() and sum(f.stat().st_size for f in files)<=old.CAP,'bounded file inventory')
    report=read(str(out/'report.json'));c=read(str(out/'completion.json'))
    require(set(report)=={'schema','binding','row','journal_tail','journal_sha256','elapsed_seconds'}
        and set(c)=={'schema','report_sha256','artifact_sha256','bytes_before_completion','inclusive_wall_seconds','wall_scope'},'exact own report/completion fields')
    require(c['schema']==NAME+'.completion' and report['schema']==NAME+'.report' and c['report_sha256']==report_sha
        and 0<=report['elapsed_seconds']<=c['inclusive_wall_seconds']<=old.WALL
        and c['wall_scope']=='through report hash; completion self-write excluded','own publication closure')
    require(c['artifact_sha256']=={f.name:sha(str(f)) for f in files if f.name!='completion.json'}
        and c['bytes_before_completion']==sum(f.stat().st_size for f in files if f.name!='completion.json'),'all file SHA and sizes')
    require(same(read(str(out/'binding.json')),binding) and same(report['binding'],binding),'whole binding')
    for name,quota in old.QUOTAS.items():
        if (out/name).exists():require((out/name).stat().st_size<=quota,'metadata quota')
    events,tail=read_events(out)
    require(tail==report['journal_tail'] and report['journal_sha256']==sha(str(out/'journal.jsonl')),'full hashchain')
    row=reconstruct(binding,reference,original,events,folder=out);require(same(row,report['row']),'independent complete row reconstruction')
    png=set()
    for e in row['observations']:
        f=e['receipt'];name=f['png'];require(Path(name).name==name and name not in png,'unique owned PNG')
        png.add(name);p=out/name
        require(p.is_file() and not p.is_symlink() and p.stat().st_size<=old.png_budget()['png_upper_bytes'],'bounded regular PNG')
        body=p.read_bytes();require(sha(str(p))==f['file_sha256'],'PNG byte SHA')
        old.check_png_stream(body)
        with Image.open(io.BytesIO(body)) as image:
            require(image.mode=='RGB' and image.size==(224,224) and image.format=='PNG','RGB224')
            pixels=np.asarray(image).copy()
        require(hashlib.sha256(pixels.tobytes()).hexdigest()==f['pixel_sha256'],'decoded pixel SHA')
    require({f.name for f in files}==png|{'binding.json','journal.jsonl','report.json','completion.json'},'no unconsumed files')
    require(sha(str(out/'completion.json'))==completion_sha and all(sha(str(out/n))==h for n,h in c['artifact_sha256'].items()),'stable bytes after consume')
    return dict(schema=NAME+'.job_consume',output=str(out),report_sha256=report_sha,completion_sha256=completion_sha,
        row=row,event_count=len(events),fixture_only=binding['fixture_only'],**FLAGS)


def consume_failure(output,expected,binding):
    out=Path(output);f=out/'failure.json';require(sha(str(f))==expected and f.stat().st_size<=old.QUOTAS['failure.json'],'failure bytes')
    v=read(str(f));require(v['schema']==NAME+'.publication_failure' and v['status']=='transport_failed_NOT_verified'
        and v['binding_sha256']==digest(binding) and v['unknown_final_callbacks'] is True and v['endpoint20'] is None,'partial is not success')
    files=list(out.iterdir());require(all(p.is_file() and not p.is_symlink() for p in files)
        and sum(p.stat().st_size for p in files)<=old.CAP and same(read(str(out/'binding.json')),binding),'bounded failure inventory')
    require(v['artifact_sha256']=={p.name:sha(str(p)) for p in files if p.name!='failure.json'}
        and v['journal_sha256']==v['artifact_sha256'].get('journal.jsonl'),'all partial bytes')
    return dict(schema=NAME+'.failure_consume',status=v['status'],output=str(out),declarations=v,
                counts=None,endpoint20=None,unknown_final_callbacks=True,**FLAGS)


def binding_for(p,job,prepared,prepare_sha,fixture=False):
    return dict(schema=NAME+'.job_binding',prepared=str(prepared),prepare_sha256=prepare_sha,
        job=job,fixture_only=fixture,authority=p['authority'],tail_raw5=deepcopy(gate.TAIL))


def native_job(p,w,original,job,output,prepared,prepare_sha,helper):
    reference=w['rows'][job['endpoint']];binding=binding_for(p,job,prepared,prepare_sha)
    store=Store(output,binding,p['limits']);store.emit('run_started',binding_sha256=digest(binding))
    meter=Meter(store,job,reference,original);env=None
    scratch=dict(index=job['index'],condition='off',PNG=0,trajectory=[],checkpoints=[],
                 missing_checkpoints=[0,5,10],last_observation=None)
    closed=dict(status='engineering_failed',error=None,cleanup_errors=[])
    def public(ret):
        obs,reward,t,tr,info=ret;actual=np.asarray(env.unwrapped.latest_action)
        return dict(state=np.asarray(obs['state'],np.float64).tolist(),snapshot=helper.moving.snapshot(env.unwrapped),
            reward=float(reward),terminated=bool(t),truncated=bool(tr),goal_state=np.asarray(info['goal_state']).tolist(),
            actual_bits=actual.tobytes().hex(),actual_dtype=actual.dtype.str)
    def reset(ret):
        obs,info=ret
        return dict(state=np.asarray(obs['state'],np.float64).tolist(),goal_state=np.asarray(info['goal_state']).tolist(),
                    snapshot=helper.moving.snapshot(env.unwrapped))
    meter.public_encoder=public;meter.reset_encoder=reset
    try:
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401
        env=meter.call('constructor',lambda:gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224,with_target=False))
        helper.validate_env(env.unwrapped)
        require(env.unwrapped.render_action is False and env.unwrapped.render_size==224,'original render contract')
        payload=p['goal_payloads'][job['root_slot']];require(sha(payload['path'])==payload['sha256'],'goal NPZ SHA')
        with np.load(payload['path'],allow_pickle=False) as z:
            require(set(z.files)=={'rgb_history','raw_action_history','goal_rgb'},'blind NPZ keys');goal_rgb=z['goal_rgb'].copy()
        require(goal_rgb.shape==(224,224,3) and goal_rgb.dtype==np.uint8 and
            hashlib.sha256(goal_rgb.tobytes()).hexdigest()==reference['source_root']['goal_pixels']['off'],'goal pixels')
        with helper.native_hooks(env,meter,closed['cleanup_errors']):
            slot=reference['source_slot_metadata'];recipe=reference['placement_events'][0]
            _,info=meter.call('reset',lambda:env.reset(seed=slot['reset_seed']),role='seed',seed=slot['reset_seed'])
            require(same(np.asarray(info['goal_state']).tolist(),recipe['native_goal']),'source reset goal')
            obs,info=meter.call('reset',lambda:env.reset(seed=0,options=dict(
                state=np.asarray(recipe['requested_state'],np.float64),goal_state=np.asarray(recipe['native_goal'],np.float64))),
                role='placement',seed=0,requested_state=deepcopy(recipe['requested_state']))
            for phase,commands in (('warmup',job['warm10']),('plan',job['raw15']+gate.TAIL)):
                if phase=='plan':
                    body=env.unwrapped;before=helper.invariant(body);goal=np.asarray(reference['source_root']['goal'],np.float64)
                    meter.call('goal_bind',lambda:helper.moving.attach_goal(body,goal,goal_rgb));after=helper.invariant(body)
                    require(all(same(before[k],after[k]) for k in ('snapshot','renderer','goal_pose')),'inert original goal bind')
                helper.capture(env,obs,phase,0,scratch,meter,store)
                for step,command in enumerate(commands,1):
                    a=np.asarray(command,dtype='<f4')
                    obs,reward,t,tr,info=meter.call('public_step',lambda:env.step(a.copy()),phase=phase,step=step,raw_bits=a.tobytes().hex())
                    helper.capture(env,obs,phase,step,scratch,meter,store)
                    expected=np.asarray(recipe['native_goal'] if phase=='warmup' else reference['source_root']['goal'])
                    helper.paired.check_return(env,a,np.asarray(env.unwrapped.latest_action),info,expected,float(reward),bool(t))
                    if t or tr:closed['status']='native_stopped';raise old.NativeStop()
            closed['status']='horizon_completed'
    except old.NativeStop:closed['error']=None
    except BaseException as e:closed.update(status='engineering_failed',error=old.error(e))
    finally:
        if env is not None:
            try:meter.call('close',env.close,cleanup=True)
            except BaseException as e:closed['cleanup_errors'].append(old.error(e))
        if closed['cleanup_errors']:closed.update(status='engineering_failed',error=closed['error'] or dict(type='CleanupFailure',message='cleanup'))
        try:
            require(all(sha(f)==h for f,h in p['bindings'].items()) and sha(str(prepared))==prepare_sha,'final source closure')
        except BaseException as e:closed.update(status='engineering_failed',error=old.error(e))
        try:
            store.emit('branch_closed',result=closed);store.emit('run_finished',result=closed)
            return publish(store,reference,original)
        except BaseException as e:return failed_publication(store,meter,closed,e)


def run(prepared,prepare_sha,output,authorization):
    require(authorization==AUTH,'MAIN explicit engineering authorization')
    p,w,rows=verify(prepared,prepare_sha);require(p['fixture_only'] is False,'no fixtures')
    root=Path(output);require(root.is_absolute() and not root.exists() and root.parent.is_dir(),'exclusive batch directory')
    require(shutil.disk_usage(root.parent).free>=10*old.CAP+old.FLOOR, 'full ten-job maximum plus disk floor before any environment')
    helper=old.native_helpers(p);root.mkdir();old.fsync_directory(root.parent);results=[]
    for job,ref in zip(p['jobs'],rows,strict=True):
        out=root/f'job{job["index"]:02}'
        try:results.append(native_job(p,w,ref,job,str(out),prepared,prepare_sha,helper))
        except BaseException as e:
            # Initialization or even reserve publication may fail. Preserve the
            # actual flat remnants and explicitly refuse to infer callback counts.
            inventory={}
            if out.exists():
                require(out.is_dir() and not out.is_symlink(),'startup partial owned directory')
                files=list(out.iterdir());require(all(f.is_file() and not f.is_symlink() for f in files), 'startup partial flat files')
                inventory={f.name:sha(str(f)) for f in files}
            failure=root/f'failure{job["index"]:02}.json'
            record=dict(schema=NAME+'.unpublished_failure',index=job['index'],job_sha256=digest(job),
                output=str(out),directory_present=out.exists(),artifact_sha256=inventory,
                error=old.error(e),counts=None,endpoint20=None,unknown_final_callbacks=True)
            old.write_new(str(failure),record)
            results.append(dict(output=str(out),status='unpublished_failure_NOT_verified',
                                failure=str(failure),failure_sha256=sha(str(failure))))
    return old.write_new(str(root/'batch.json'),dict(schema=NAME+'.batch',prepared=str(prepared),prepare_sha256=prepare_sha,jobs=results,**FLAGS))


def consume(output,prepared,prepare_sha,batch_sha):
    p,w,rows=verify(prepared,prepare_sha);root=Path(output);batch=read(str(root/'batch.json'))
    require(sha(str(root/'batch.json'))==batch_sha and batch['schema']==NAME+'.batch'
        and batch['prepared']==str(prepared) and batch['prepare_sha256']==prepare_sha and len(batch['jobs'])==10,'whole batch authority')
    require(set(batch)=={'schema','prepared','prepare_sha256','jobs'}|set(FLAGS)
        and all(batch[k] is v for k,v in FLAGS.items()), 'whole original false science flags')
    expected={'batch.json'}
    for i,seal in enumerate(batch['jobs']):
        if seal.get('status')=='unpublished_failure_NOT_verified':
            expected.add(f'failure{i:02}.json')
            if (root/f'job{i:02}').exists():expected.add(f'job{i:02}')
        else:expected.add(f'job{i:02}')
    require({f.name for f in root.iterdir()}==expected,'all ten fixed branch inventories')
    results=[]
    for job,original,seal in zip(p['jobs'],rows,batch['jobs'],strict=True):
        out=root/f'job{job["index"]:02}';binding=binding_for(p,job,prepared,prepare_sha)
        require(seal['output']==str(out),'per-job output identity')
        if seal.get('status')=='unpublished_failure_NOT_verified':
            failure=root/f'failure{job["index"]:02}.json'
            require(seal['failure']==str(failure) and sha(str(failure))==seal['failure_sha256'],'own failure SHA')
            record=read(str(failure));require(record['schema']==NAME+'.unpublished_failure'
                and record['index']==job['index'] and record['job_sha256']==digest(job) and record['output']==str(out)
                and record['directory_present']==out.exists() and record['counts'] is None
                and record['endpoint20'] is None and record['unknown_final_callbacks'] is True,'unpublished failure cannot authorize')
            files=list(out.iterdir()) if out.exists() else []
            require(all(f.is_file() and not f.is_symlink() for f in files) and
                    record['artifact_sha256']=={f.name:sha(str(f)) for f in files},'all unpublished partial bytes')
            results.append(dict(schema=NAME+'.unpublished_failure_consume',status=seal['status'],declarations=record,
                                counts=None,endpoint20=None,unknown_final_callbacks=True,**FLAGS))
        elif seal.get('status')=='transport_failed_NOT_verified':results.append(consume_failure(str(out),seal['failure_sha256'],binding))
        else:results.append(consume_job(str(out),seal['report_sha256'],seal['completion_sha256'],binding,w['rows'][job['endpoint']],original))
    require(sha(str(root/'batch.json'))==batch_sha and sha(str(prepared))==prepare_sha
        and all(sha(f)==h for f,h in p['bindings'].items()),'stable batch and source closure')
    partial=any(x.get('status') in ('transport_failed_NOT_verified','unpublished_failure_NOT_verified') for x in results)
    return dict(schema=NAME+'.consume',status='saved_partial_transport_NOT_verified' if partial else 'all10_saved_native_jobs_reconstructed_NOT_admitted',
        prepared=str(prepared),prepare_sha256=prepare_sha,fixed_denominator=10,jobs=results,
        source_authorities=p['authority'],new_neural_calls=0,FLOPs=None,**FLAGS)


def main(argv=None):
    q=argparse.ArgumentParser(description=__doc__);sub=q.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--output',required=True)
    for command in ('verify','run','consume'):
        a=sub.add_parser(command);a.add_argument('--prepared',required=True);a.add_argument('--prepare-sha',required=True)
        if command in ('run','consume'):a.add_argument('--output',required=True)
        if command=='run':a.add_argument('--authorization',required=True)
        if command=='consume':a.add_argument('--receipt',required=True);a.add_argument('--batch-sha',required=True)
    a=q.parse_args(argv)
    if a.command=='prepare':value=prepare(a.output)
    elif a.command=='verify':p,_,_=verify(a.prepared,a.prepare_sha);value=dict(verified=True,jobs=10,limits=p['limits'])
    elif a.command=='run':value=run(a.prepared,a.prepare_sha,a.output,a.authorization)
    else:value=old.write_new(a.receipt,consume(a.output,a.prepared,a.prepare_sha,a.batch_sha))
    print(json.dumps(value,sort_keys=True))


if __name__=='__main__':main()
