"""Resume the sealed rank-reserve P3 front end; saved operations only.

Reuses existing prepare verifier and actual CUDA consumer. Never invokes run,
loads a neural model, reads downstream physical outcomes or changes allocation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np

NAME = 'agent_p3_resume_20261027_frontend'
PREFIX = 'agent_p3_resume_20261027_'
ROOT = Path(__file__).resolve().parents[2]
CUDA_PREPARE = ROOT/'results/chat_p3_20261027_cuda_prepare01.json'
CUDA_PREPARE_SHA = '72a02d66c99c9037cd55828bf67342514ec110a3310ced194477ef6e34c432f1'
CUDA_OUTPUT = ROOT/'results/chat_p3_20261027_cuda_run01'
CUDA_REPORT_SHA = '6df856e36b145162e6a8d5ce5f141880a49afe88f43d3cc88f8b98ad790b6d08'
ARMS = ('U','D_rank','S_rank','N')
FLAGS = dict(P1_admission=False,P3_admission=False,freshness=False,
             independent_confirmation=False,compute_advantage=False,walltime_advantage=False)
ID_KEYS = ('source_slot','root_sha256','task_id','task_sha256','task_contract_sha256','blind_sha256','bank_sha256')
CAP, FLOOR = 2*1024**2, 200*1024**2


def require(ok, message):
    if not ok: raise ValueError(message)


def canonical(v):
    return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def digest(v): return hashlib.sha256(canonical(v)).hexdigest()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2),b''): h.update(chunk)
    return h.hexdigest()


def hash_value(v):
    require(type(v) is str and len(v)==64 and all(c in '0123456789abcdef' for c in v),'literal external SHA256')


def read(path):
    path=Path(path); require(path.stat().st_size<=CAP,'bounded own metadata')
    def pairs(items):
        out={}
        for k,v in items:
            require(k not in out,'duplicate JSON key'); out[k]=v
        return out
    value=json.loads(path.read_bytes(),object_pairs_hook=pairs)
    canonical(value); return value


def dependency():
    # Imports constants/saved validators, not torch/gym or any runtime factory.
    from experiments.spectral_reachability import chat_p3_20261027_cuda as module
    return module


def own_files():
    return [Path(__file__).resolve(),ROOT/'tests/agent_p3_resume_20261027_test_frontend.py',
            ROOT/'innerdocs/agent_p3_resume_20261027_delivery.md']


def raw_commands(value):
    require(type(value) is list and len(value)==10 and all(type(row) is list and len(row)==2 and
            all(type(x) in (int,float) and math.isfinite(x) for x in row) for row in value),'literal raw10x2')
    a=np.asarray(value,dtype='<f4')
    require(np.isfinite(a).all() and np.array_equal(a.astype(float),np.asarray(value)) and
            np.all(np.abs(a)<=1),'lossless bounded float32 commands')
    return a


def compile_queue(p,banks,sealed,rule):
    """Signal-only allocation; score/outcome fields never enter this function."""
    require(len(p['slots'])==len(banks)==2 and tuple(rule.ARMS)==ARMS,'original fixed two-root/four-arm identities')
    w=sealed['weights']
    require(type(w) is list and len(w)==2 and all(type(x) in (int,float) and math.isfinite(x) and x>=0 for x in w),'model-only finite weights')
    queues={}
    for s,slot in enumerate(p['slots']):
        require(type(slot['source_slot']) is int and slot['source_slot']==s and len(banks[s])==45,'ordered complete frozen bank')
        for k in ID_KEYS[1:]: hash_value(slot[k])
    for arm in ARMS:
        # U/N have no signal argument; exact equality is a real collision.
        blocks=rule.allocation(arm) if arm in ('U','N') else rule.allocation(arm,w)
        require(len(blocks)==2 and all(type(x) is int and 5<=x<=15 for x in blocks) and sum(blocks)==20,'fixed total20/minimum5')
        jobs=rule.queue(blocks)
        frozen=sealed['plans'][arm]
        require(blocks==frozen['blocks'] and jobs==frozen['jobs'] and len(jobs)==60,'exact sealed discrete treatment')
        requests=[]
        for n,job in enumerate(jobs):
            s,index=job['slot'],job['index']; bank=banks[s][index]
            require(type(s) is int and s in (0,1) and type(index) is int and bank['index']==index,'literal stable bank index')
            require(digest(bank)==frozen['candidate_sha256'][n],'all candidate seals, no winner-only projection')
            raw=raw_commands(bank['commands'])
            requests.append(dict(ordinal=n,arm=arm,slot=s,index=index,
                identity={k:deepcopy(p['slots'][s][k]) for k in ID_KEYS},candidate_sha256=digest(bank),
                raw_float32_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),commands=raw.tolist(),
                packet_steps=5,packets=2,predict_API=2,independent_full_H2=True))
        queues[arm]=dict(blocks=blocks,candidates_per_root=[3*x for x in blocks],requests=requests,
            search_predict_API=120,winner_replay_predict_API=8,
            replay_order=[dict(slot=s,mode=mode,predict_API=2,depends_on='stable(cost,index) of this arm/root')
                          for s in (0,1) for mode in ('whole_replay','segmented_replay')])
    return dict(arms=list(ARMS),weights=deepcopy(w),queues=queues,
        allocation_tie_break='exact equal weights ->10/10; strict rank; no epsilon/gain/temperature',
        candidate_tie_break='(cost,index), original stable bank indices',
        dispatch_order='minimum5 alternating slot0/1 blocks, then original slot0/1 rounds',
        budget=dict(shared_predict_API=8,search_predict_API=480,replay_predict_API=32,total_predict_API=520,
            per_arm_predict_API=128,shared_J=2,shared_VJP_directions=384,shared_grad_API=384,
            shared_raw_backward_callbacks=384,shared_model_SVD=2,new_solve=0,
            predictor_samples='must come from actual module tensor shapes, not inferred from API'),
        allocation_separated=len({tuple(queues[a]['blocks']) for a in ('U','D_rank','S_rank')})==3,
        allocation_collision_pairs=[[a,b] for i,a in enumerate(ARMS) for b in ARMS[i+1:] if queues[a]['blocks']==queues[b]['blocks']],
        no_outcome_selection=True,no_retries=True,no_dedup=True,**FLAGS)


def material():
    m=dependency()
    p,_,banks=m.verify(CUDA_PREPARE,CUDA_PREPARE_SHA)
    sealed=m.read(m.SEAL)
    queue=compile_queue(p,banks,sealed,m.rule)
    bindings=deepcopy(p['bindings_sha256'])
    for path in [CUDA_PREPARE,m.SEAL,Path(m.__file__),Path(m.rule.__file__),*own_files()]:
        resolved=str(Path(path).resolve()); h=sha(path)
        require(resolved not in bindings or bindings[resolved]==h,'conflicting closure SHA'); bindings[resolved]=h
    require(bindings[str(CUDA_PREPARE)]==CUDA_PREPARE_SHA and bindings[str(m.SEAL)]==m.SEAL_SHA,'frozen source prepare/queue')
    value=dict(schema=NAME+'.prepare',execution_adapter='chat_p3_20261027_cuda; use existing sealed executor, not a new algorithm',
        predecessor_prepare=str(CUDA_PREPARE),predecessor_prepare_sha256=CUDA_PREPARE_SHA,
        rule_seal_sha256=m.SEAL_SHA,model_hash=p['expected_model_hash'],queue=queue,bindings_sha256=bindings,
        historical_encoder=deepcopy(p['historical_encoder']),historical_anchor_receipts=deepcopy(p['historical_anchor_receipts']),
        fixed_denominator=dict(roots=2,arms=4,branches=8,search_candidates=240),
        admission='engineering known-root replay only; P1 and freshness unavailable',**FLAGS)
    canonical(value); return value


def write_new(path,value):
    path=Path(path).absolute(); data=canonical(value)
    require(path.parent==ROOT/'results' and path.name.startswith(PREFIX) and not path.exists() and not path.is_symlink(),
            'exclusive new agent_p3_resume result only')
    require(len(data)<=CAP and shutil.disk_usage(path.parent).free>=FLOOR+len(data),'2MiB output/200MiB floor')
    with path.open('xb') as stream:
        require(stream.write(data)==len(data),'short metadata write'); stream.flush(); os.fsync(stream.fileno())
    fd=os.open(path.parent,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return dict(path=str(path),sha256=sha(path),bytes=len(data))


def prepare(path): return write_new(path,material())


def verify(path,expected):
    hash_value(expected); require(sha(path)==expected,'external own prepare SHA')
    saved=read(path); now=material()
    require(canonical(saved)==canonical(now) and sha(path)==expected,'complete own prepare/source/queue drift')
    return saved


def queue(path,expected,arm):
    require(arm in ARMS,'explicit existing arm')
    p=verify(path,expected)
    return dict(schema=NAME+'.queue',prepare_sha256=expected,model_hash=p['model_hash'],
        arm=arm,**deepcopy(p['queue']['queues'][arm]),
        execution_adapter=p['execution_adapter'],neural_execution_started=False,**FLAGS)


def summarize(p,verified,result,events,report):
    """All gains are model-only; allocation already frozen before this call."""
    require(verified['schema']=='chat_p3_20261027_cuda.consume' and
            verified['status']=='actual_rank_queues_verified_not_admitted' and all(verified[k] is False for k in FLAGS),
            'existing complete own consumer required')
    require(len(verified['plans'])==8 and all(type(x['source_slot']) is int for x in verified['plans']) and
            [(x['arm'],x['source_slot']) for x in verified['plans']]==[(a,s) for a in ARMS for s in (0,1)],
            'full eight-plan denominator')
    require([x['signal']['w'] for x in result['signals']]==p['queue']['weights'],'fresh execution signal/queue exact equality')
    for arm in ARMS:
        q=p['queue']['queues'][arm]
        require(result['allocations'][arm]==q['blocks'],'actual discrete allocation')
        frozen=[e for e in events if e['kind']=='queue_frozen' and e['arm']==arm]
        require(len(frozen)==1 and frozen[0]['blocks']==q['blocks'] and
                frozen[0]['jobs']==[dict(slot=j['slot'],index=j['index']) for j in q['requests']], 'actual queue release order')
        calls=[e for e in events if e['kind']=='candidate_request' and e['scope'][0]==arm]
        require(len(calls)==60,'complete sixty independent paid requests')
        for n,(e,j) in enumerate(zip(calls,q['requests'],strict=True)):
            candidate=deepcopy(e['candidate']); require(candidate.pop('status')=='attempted','attempt before candidate dispatch')
            require(e['scope']==[arm,j['slot']] and type(e['paid']) is int and e['paid']==n+1 and digest(candidate)==j['candidate_sha256'],
                    'every actual candidate bound to exact executable queue')
        for s in (0,1):
            plan=next(x for x in verified['plans'] if x['arm']==arm and x['source_slot']==s)
            require(all(plan[k]==q['requests'][next(i for i,j in enumerate(q['requests']) if j['slot']==s)]['identity'][k] for k in ID_KEYS),
                    'same root/goal/task/blind/model source across controls')
    samples={}
    for group in ('shared',)+ARMS:
        samples[group]={}
        for resource in ('action_encoder','predictor','pred_proj'):
            shapes=[x['shape'] for x in report['actual_module_shapes'] if x['resource']==resource and x['scope'][0]==group]
            rank=2 if resource=='pred_proj' else 3
            require(all(len(s)==rank and all(type(x) is int and x>0 for x in s) for s in shapes),'actual module shapes')
            samples[group][resource]=dict(observed_calls=len(shapes),input_rows=sum(s[0] for s in shapes),
                context_tokens=None if rank==2 else sum(s[0]*s[1] for s in shapes))
    for resource in ('action_encoder','predictor','pred_proj'):
        observed={k:None if k=='context_tokens' and resource=='pred_proj' else sum(samples[g][resource][k] for g in samples)
                  for k in ('observed_calls','input_rows','context_tokens')}
        require(observed==verified['module_sample_axes'][resource],'separate actual sample/token axes')
    collisions=[]
    for s in (0,1):
        plans=[next(x for x in verified['plans'] if x['arm']==a and x['source_slot']==s) for a in ARMS]
        commands=[raw_commands(x['commands']).tobytes() for x in plans]
        collisions.append(dict(source_slot=s,winner_indices={a:x['candidate_index'] for a,x in zip(ARMS,plans,strict=True)},
            winner_collision_pairs=[[a,b] for i,a in enumerate(ARMS) for j,b in enumerate(ARMS) if j>i and plans[i]['candidate_index']==plans[j]['candidate_index']],
            commands_collision_pairs=[[a,b] for i,a in enumerate(ARMS) for j,b in enumerate(ARMS) if j>i and commands[i]==commands[j]]))
    times={g:{} for g in ('shared',)+ARMS}
    for e in events:
        if e['kind']!='resource_return': continue
        r=e['entry']; elapsed=r['ended_seconds']-r['started_seconds']; require(math.isfinite(elapsed) and elapsed>=0,'actual resource timing')
        g=r['scope'][0]; name=r['resource']; times[g][name]=times[g].get(name,0.)+elapsed
    return dict(schema=NAME+'.consume',status='completed_saved_execution_do_not_rerun_not_admitted',
        queue=p['queue'],fixed_denominator=p['fixed_denominator'],collisions=collisions,
        paired_model_only=deepcopy(verified['paired_summary']),plans=deepcopy(verified['plans']),
        resources=dict(actual_execution=deepcopy(verified['resources']),scope_resources=deepcopy(verified['scope_resources']),
            observed_samples_by_group=samples,observed_module_axes=deepcopy(verified['module_sample_axes']),
            execution_wall_seconds=verified['new_wall_seconds'],inclusive_resource_wall_seconds=times,
            wall_semantics='nested inclusive; never sum API/module times as total wall or FLOPs',
            backward_engine_calls=report['backward_engine_calls'],
            shared_spectral_cost_paid_once=True,logical_deployment_predict_API=dict(U=136,D_rank=136,S_rank=136,N=128),
            historical_encoder=p['historical_encoder'],historical_anchor_receipts=p['historical_anchor_receipts']),
        new_neural_calls=0,new_physics_calls=0,new_training=False,physical_outcomes_read=False,
        missing_scientific_dependencies=['P1 mechanism admission','certified independent fresh ordered roots/goals',
            'approved fresh-panel source/encoder/anchor receipts and full acquisition/resource budget'],
        limitations=['rank-only rule amplifies arbitrarily small weight differences; not the original proportional rule',
            'allocation separation does not imply winner/action/physical benefit separation',
            'known-root model-only replay; no equal FLOPs/total compute or causal/generalization claim'],**FLAGS)


def consume(path,expected,output,report_sha):
    start=time.monotonic(); p=verify(path,expected); hash_value(report_sha)
    m=dependency(); verified=m.consume(output,report_sha)
    result=m.read(Path(output)/'panel.json.gz',32*1024**2); report=m.read(Path(output)/'report.json')
    plain,eof=m.old.blind.inflate((Path(output)/'events.jsonl.gz').read_bytes(),128*1024**2)
    require(eof and plain.endswith(b'\n'),'complete saved execution journal')
    events=[m.old.blind.decode(line) for line in plain.splitlines()]
    out=summarize(p,verified,result,events,report)
    for filename,h in p['bindings_sha256'].items(): require(sha(filename)==h,'final own input drift')
    require(sha(Path(output)/'report.json')==report_sha and sha(path)==expected,'final external seals')
    out.update(prepare_sha256=expected,actual_execution_report_sha256=report_sha,
        saved_validation_wall_seconds=time.monotonic()-start,saved_validation_array_SVD_calls=None,
        saved_validation_array_SVD_note='inherited full validators recompute saved spectra; not instrumented here, not zero',
        new_model_SVD_calls=0)
    return out


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','verify','queue','consume'))
    parser.add_argument('--prepared'); parser.add_argument('--prepare-sha'); parser.add_argument('--output')
    parser.add_argument('--arm',choices=ARMS); parser.add_argument('--cuda-output',default=str(CUDA_OUTPUT))
    parser.add_argument('--cuda-report-sha',default=CUDA_REPORT_SHA)
    a=parser.parse_args(argv)
    if a.action=='prepare': value=prepare(a.output)
    elif a.action=='verify':
        p=verify(a.prepared,a.prepare_sha); value=dict(status='verified_no_execution',allocations={k:v['blocks'] for k,v in p['queue']['queues'].items()},**FLAGS)
    elif a.action=='queue': value=queue(a.prepared,a.prepare_sha,a.arm)
    else: value=consume(a.prepared,a.prepare_sha,a.cuda_output,a.cuda_report_sha)
    if a.output and a.action!='prepare': value=write_new(a.output,value)
    print(canonical(value).decode())
    return 0


if __name__=='__main__': raise SystemExit(main())
