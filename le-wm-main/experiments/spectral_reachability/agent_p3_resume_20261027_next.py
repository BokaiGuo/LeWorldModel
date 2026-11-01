"""P3 cold-proposal control: saved diagnosis and MAIN-only host CUDA entry.

No execution on import/prepare/verify/diagnose. No physical outcome inputs.
The new bank is fixed independently of scores; historical negatives survive.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np

from experiments.spectral_reachability import agent_p3_resume_20261027_frontend as prior

ROOT, ARMS, FLAGS = prior.ROOT, prior.ARMS, prior.FLAGS
NAME = 'agent_p3_resume_20261027_next'
PREVIOUS = ROOT/'results/agent_p3_resume_20261027_prepare01.json'
PREVIOUS_SHA = 'f6b827cc50f1daf4070a67058fbddb548e24a223c62645f6788654352602723b'
require, sha, canonical, digest = prior.require, prior.sha, prior.canonical, prior.digest


def dependency(): return prior.dependency()


def maxima():
    m = dependency()
    return {r: 512 if r in m.old.NEURAL else 0 for r in m.old.MAXIMA}


def protocol():
    return dict(schema=NAME+'.protocol', arms=list(ARMS), roots=2, branches=8,
        total_blocks=20, minimum_blocks=5, maximum_blocks=15, candidates_per_block=3,
        candidates_per_bank=45, horizon=2, search_predict_per_arm=120, replay_predict_per_arm=8,
        new_resource_maxima=maxima(), signal='immutable saved model-only zero-nominal J signal',
        candidate_family='cold PCG64 ball proposals only; original 42 plus next 3 draws',
        seed='20261027+101*slot; original sequential stream; no reseed/redraw',
        selection='stable(cost,index); all paid candidates eligible; no early stop',
        equal_incremental_resource_vector=True, equal_wall=False, equal_FLOPs=False,
        shared_historical_predict_API=8, shared_historical_J=2,
        shared_historical_VJP=384, shared_historical_grad_API=384,
        shared_historical_raw_backward_callbacks=384, shared_historical_model_SVD=2,
        standalone_common_spectral_charge='all four policies pay same historical spectral acquisition, including U/N unused signal',
        standalone_predict_including_spectral=136, backward_engine_calls=None,
        no_retry=True, no_dedup=True, CPU_neural=False, SDP='MATH', TF32=False,
        wall_seconds=600, output_cap_bytes=16*1024**2, disk_floor_bytes=200*1024**2,
        allocated_bytes=3*1024**3, reserved_bytes=4*1024**3, **FLAGS)


def cold_bank(slot, original):
    """No scores/goal/outcomes accepted. Preserve RNG/domain, not incumbents."""
    m=dependency().old
    require(type(slot) is int and slot in (0,1) and len(original)==45, 'ordered original bank')
    rng=np.random.Generator(np.random.PCG64(20261027+101*slot)); out=[]
    for index in range(45):
        packets=[]
        for _ in range(2):
            v=rng.standard_normal(10); norm=np.linalg.norm(v)
            require(norm>0 and np.isfinite(norm), 'reject invalid direction without redraw')
            packets.append(v/norm*m.BALL*rng.random()**(1/10))
        logits=np.asarray(packets,np.float32); raw=np.tanh(logits).reshape(10,2)
        m.legal(raw,logits)
        if index<42:
            old=original[index+3]
            require(old['index']==index+3 and old['kind']=='PCG64_ball' and
                    old['commands']==raw.astype(float).tolist() and old['logits']==logits.astype(float).tolist(),
                    'exact original proposal stream prefix')
        out.append(dict(index=index,kind='PCG64_ball',old_stream_index=index+3,
            logits=logits.astype(float).tolist(),commands=raw.astype(float).tolist()))
    require(len({digest(c['commands']) for c in out})==45, '45 distinct literal proposals; no silent dedup')
    return out


def queues(slots,banks,weights):
    m=dependency(); out={}
    require(len(slots)==len(banks)==2, 'two ordered roots')
    require(type(weights) is list and len(weights)==2 and all(type(w) in (int,float) and math.isfinite(w) and w>=0 for w in weights), 'frozen weights')
    for s,slot in enumerate(slots):
        require(type(slot['source_slot']) is int and slot['source_slot']==s and len(banks[s])==45, 'root/bank census')
        for k in prior.ID_KEYS[1:]: prior.hash_value(slot[k])
        for i,c in enumerate(banks[s]):
            require(type(c['index']) is int and c['index']==i and c['kind']=='PCG64_ball' and c['old_stream_index']==i+3, 'stable cold bank identity')
            prior.raw_commands(c['commands']); m.old.legal(c['commands'],c['logits'])
    for arm in ARMS:
        blocks=m.rule.allocation(arm) if arm in ('U','N') else m.rule.allocation(arm,weights)
        requests=[]
        for ordinal,j in enumerate(m.rule.queue(blocks)):
            s,i=j['slot'],j['index']; c=banks[s][i]
            requests.append(dict(ordinal=ordinal,slot=s,index=i,arm=arm,
                identity={k:deepcopy(slots[s][k]) for k in prior.ID_KEYS},
                search_bank_sha256=digest(banks[s]),candidate_sha256=digest(c),
                raw_float32_sha256=hashlib.sha256(prior.raw_commands(c['commands']).tobytes()).hexdigest(),
                commands=deepcopy(c['commands']),packets=2,packet_steps=5,predict_API=2))
        out[arm]=dict(blocks=blocks,requests=requests,search_predict_API=120,replay_predict_API=8)
    return out


def diagnose_rows(result,banks):
    """Only called after the complete historical consumer succeeded."""
    require([(r['arm'],r['source_slot']) for r in result['rows']]==[(a,s) for a in ARMS for s in (0,1)], 'eight historical branches')
    rows=[]
    for r in result['rows']:
        cs=r['candidates']; s=r['source_slot']; winner=min(cs,key=lambda c:(c['cost'],c['index']))
        require(winner['index']==r['winner_index'] and winner['cost']==r['plan']['model_cost'], 'historical actual winner')
        proposals=[c for c in cs if c['index']>=3]
        require(proposals and all({k:c[k] for k in banks[s][c['index']]}==banks[s][c['index']] for c in cs), 'all literal scored bank entries')
        costs=[float(c['cost']) for c in cs]
        require(all(math.isfinite(v) for v in costs), 'finite actual costs')
        commands={digest(c['commands']) for c in proposals}
        terminals={digest(c['contexts'][-1]['latent_history'][-1]) for c in proposals}
        curve=[dict(paid_candidates=k,winner_index=min(cs[:k],key=lambda c:(c['cost'],c['index']))['index'],
                    best_cost=min(c['cost'] for c in cs[:k])) for k in (3,15,len(cs))]
        rows.append(dict(arm=r['arm'],source_slot=s,paid_candidates=len(cs),winner_index=winner['index'],
            winner_kind=banks[s][winner['index']]['kind'],best_so_far=curve,
            best_proposal_minus_winner=min(c['cost'] for c in proposals)-winner['cost'],
            distinct_proposal_commands=len(commands),distinct_proposal_predicted_endpoints=len(terminals),
            model_cost_min=min(costs),model_cost_max=max(costs),
            winning_literal_commands=deepcopy(winner['commands'])))
    # Compare only identical candidates/identities, not physical endpoints.
    repeated={}
    for r in result['rows']:
        for c in r['candidates']:
            key=(r['source_slot'],c['index']); value=(c['cost'],digest(c['contexts']))
            require(key not in repeated or repeated[key]==value, 'repeated same-root candidate cost/full context consistency')
            repeated[key]=value
    return dict(rows=rows,actual_allocations=deepcopy(result['allocations']),
        quota_separated=result['allocations']['D_rank']!=result['allocations']['U'],
        all_winners_in_common_first_three=all(r['winner_index']<3 for r in rows),
        wiring='full predecessor consumer checked actual queue/request/module arrays; repeated contexts agree',
        interpretation='known-root model-Q anchor saturation; not proof of physical unreachability or general causal diagnosis',
        physical_reachability=None,physical_outcomes_read=False,new_neural_calls=0,new_physics_calls=0,**FLAGS)


def historical():
    # Old frozen source is never patched. Read only model-only saved artifacts.
    verified=prior.consume(PREVIOUS,PREVIOUS_SHA,prior.CUDA_OUTPUT,prior.CUDA_REPORT_SHA)
    m=dependency(); p,t,b=m.verify(prior.CUDA_PREPARE,prior.CUDA_PREPARE_SHA)
    result=m.read(prior.CUDA_OUTPUT/'panel.json.gz',32*1024**2)
    diagnosis=diagnose_rows(result,b)
    diagnosis.update(schema=NAME+'.diagnosis',source_sha256=sha(__file__),
        predecessor_report_sha256=prior.CUDA_REPORT_SHA,predecessor_prepare_sha256=PREVIOUS_SHA,
        verification_status=verified['status'],historical_execution_resources=verified['resources'],
        historical_execution_wall_seconds=verified['resources']['execution_wall_seconds'] if 'execution_wall_seconds' in verified['resources'] else None)
    return verified,p,t,b,diagnosis


def own_files():
    return [Path(__file__).resolve(),ROOT/'tests/agent_p3_resume_20261027_next_test.py',
            ROOT/'innerdocs/agent_p3_resume_20261027_next.md']


def material():
    verified,p,t,original,diagnosis=historical()
    banks=[cold_bank(s,original[s]) for s in (0,1)]
    weights=prior.read(PREVIOUS)['queue']['weights']
    bindings=deepcopy(prior.read(PREVIOUS)['bindings_sha256'])
    for f in [PREVIOUS,Path(prior.__file__),prior.CUDA_OUTPUT/'report.json',*own_files()]: bindings[str(f.resolve())]=sha(f)
    bindings.update({str((prior.CUDA_OUTPUT/n).resolve()):h for n,h in dependency().read(prior.CUDA_OUTPUT/'report.json')['artifact_sha256'].items()})
    value=dict(schema=NAME+'.prepare',protocol=protocol(),policy=p['policy'],expected_model_hash=p['expected_model_hash'],
        slots=deepcopy(p['slots']),bindings_sha256=bindings,banks=banks,weights=deepcopy(weights),
        queues=queues(p['slots'],banks,weights),diagnosis=diagnosis,
        unscored_original_stream_indices=[45,46,47],
        predecessor_prepare_sha256=PREVIOUS_SHA,predecessor_report_sha256=prior.CUDA_REPORT_SHA,
        historical_resources=deepcopy(verified['resources']),historical_encoder=deepcopy(p['historical_encoder']),
        historical_anchor_receipts=deepcopy(p['historical_anchor_receipts']),
        execution_scope='known-root engineering cold-start only; not a fresh experiment or physical release',
        missing_dependencies=['MAIN source audit and explicit host CUDA scheduling',
            'new cold-bank CUDA report and complete saved consumer',
            'P1 independent mechanism admission','certified independent fresh ordered roots/goals and full acquisition budget'],**FLAGS)
    return value,t,banks


def exclusive(path):
    p=Path(path).absolute()
    require(p.parent==ROOT/'results' and p.name.startswith(NAME) and not p.exists() and not p.is_symlink(), 'new own next prefix only')
    return p


def write_new(path,value):
    p=exclusive(path); raw=canonical(value)
    require(len(raw)<=prior.CAP and shutil.disk_usage(p.parent).free>=prior.FLOOR+len(raw), '2MiB metadata/200MiB floor')
    with p.open('xb') as f: require(f.write(raw)==len(raw),'short write'); f.flush(); os.fsync(f.fileno())
    fd=os.open(p.parent,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return dict(path=str(p),sha256=sha(p),bytes=len(raw))


def verify(path,expected):
    prior.hash_value(expected); require(sha(path)==expected,'external prepare SHA')
    p,t,b=material()
    require(canonical(prior.read(path))==canonical(p) and sha(path)==expected,'full prepare/source closure equality')
    return p,t,b


def queue(path,expected,arm):
    require(arm in ARMS,'explicit arm'); p,_,_=verify(path,expected)
    return dict(schema=NAME+'.queue',prepare_sha256=expected,model_hash=p['expected_model_hash'],
        arm=arm,**deepcopy(p['queues'][arm]),replays=[dict(slot=s,mode=mode,predict_API=2)
        for s in (0,1) for mode in ('whole_replay','segmented_replay')],neural_execution_started=False,**FLAGS)


class Meter(dependency().Meter):
    def begin(self,resource):
        require(resource in dependency().old.NEURAL and self.scope is not None and self.scope[0] in ARMS,
                'cold search only; no shared/model AD/SVD/visual acquisition')
        require(sum(e['resource']==resource for e in self.events)<512,'512 global cap')
        return super().begin(resource)


def plan(slot,arm,winner,bank_sha):
    v=dict(schema=NAME+'.model_plan',**deepcopy(slot),arm=arm,commands=deepcopy(winner['commands']),
        candidate_index=winner['index'],model_cost=winner['cost'],search_bank_sha256=bank_sha,
        physical_execution=False,selection_uses_physics=False,**FLAGS)
    v['selection_sha256']=digest(v); return v


def census(p):
    return dict(allocations={},rows=[dict(arm=a,source_slot=s,task_id=p['slots'][s]['task_id'],
        status='not_started',candidates=[]) for a in ARMS for s in (0,1)])


def paired(result): return dependency().paired(result)


def panel(p,tasks,banks,backends,meter,result):
    m=dependency().old
    for arm in ARMS:
        q=p['queues'][arm]; result['allocations'][arm]=deepcopy(q['blocks'])
        meter.scope=[arm,None]; meter.emit('queue_frozen',arm=arm,blocks=q['blocks'],requests=q['requests'])
        rows=[r for r in result['rows'] if r['arm']==arm]
        for paid,j in enumerate(q['requests'],1):
            s,i=j['slot'],j['index']; meter.scope=[arm,s]; row=rows[s]; row['status']='attempted'
            c=dict(deepcopy(banks[s][i]),status='attempted'); row['candidates'].append(c)
            meter.emit('candidate_request',paid=paid,candidate=deepcopy(c))
            with backends[s].guard(meter): path=backends[s].rollout(tasks[s],c['commands'],meter,'search')
            m.check_path(tasks[s],c['commands'],path)
            c.update(contexts=path,cost=m.cost(tasks[s],path),status='complete'); meter.emit('candidate_complete',candidate=deepcopy(c))
        for s,row in enumerate(rows):
            meter.scope=[arm,s]; winner=min(row['candidates'],key=lambda c:(c['cost'],c['index']))
            row.update(winner_index=winner['index'],replays={})
            meter.emit('winner_frozen',index=winner['index'],cost=winner['cost'],commands=winner['commands'])
            with backends[s].guard(meter):
                for mode in ('whole_replay','segmented_replay'):
                    meter.emit('replay_request',mode=mode,commands=winner['commands'])
                    path=backends[s].rollout(tasks[s],winner['commands'],meter,mode)
                    row['replays'][mode]=path; m.check_path(tasks[s],winner['commands'],path)
                    require(path==winner['contexts'],'independent full replay exact')
                    meter.emit('replay_complete',mode=mode,contexts=path)
            row['plan']=plan(p['slots'][s],arm,winner,digest(banks[s])); row['status']='model_plan_sealed'
            meter.emit('selection_sealed',plan=row['plan'])
    result['paired_summary']=paired(result)


def validate_saved(p,tasks,banks,result,events):
    m=dependency().old
    require([(r['arm'],r['source_slot']) for r in result['rows']]==[(a,s) for a in ARMS for s in (0,1)], 'full8 denominator')
    require(p['queues']==queues(p['slots'],banks,p['weights']), 'queue reconstruction without costs')
    attempts=[e['entry'] for e in events if e['kind']=='resource_attempt']
    returns=[e['entry'] for e in events if e['kind']=='resource_return']; returned={e['index']:e for e in returns}
    require(all(type(e['index']) is int for e in attempts+returns) and
        [e['index'] for e in attempts]==list(range(len(attempts))) and len(returns)==len(attempts)==len(returned),'ordered unique complete ledger')
    for a in attempts:
        b=returned[a['index']]
        require(b==dict(a,completed=True,ended_seconds=b['ended_seconds']) and a['completed'] is False and
            all(type(a[k]) in (int,float) and math.isfinite(a[k]) for k in ('started_seconds',)) and
            type(b['ended_seconds']) in (int,float) and math.isfinite(b['ended_seconds']) and b['ended_seconds']>=a['started_seconds'], 'actual attempt-return timing/identity')
    require({k:sum(e['resource']==k for e in attempts) for k in maxima()}==maxima(),'full512 actual vector')
    semantics=[]; contexts=[]; successors=[]; packets=[]
    def emit(kind,scope,**fields): semantics.append(dict(kind=kind,scope=scope,**deepcopy(fields)))
    def pathcheck(s,raw,path,arm,mode):
        m.check_path(tasks[s],raw,path); before=tasks[s]['root_context']
        for k,after in enumerate(path):
            contexts.append(([arm,s],m.moving.incoming(tasks[s],before,np.asarray(raw,np.float32)[5*k:5*k+5])))
            successors.append(([arm,s],after['latent_history'][-1]))
            packets.append(dict(kind='native_packet',scope=[arm,s],mode=mode,packet=k,context=after))
            emit('native_packet',[arm,s],mode=mode,packet=k,context=after); before=after
    for arm in ARMS:
        q=p['queues'][arm]; require(result['allocations'][arm]==q['blocks'],'actual allocation')
        emit('queue_frozen',[arm,None],arm=arm,blocks=q['blocks'],requests=q['requests'])
        rows=[r for r in result['rows'] if r['arm']==arm]; seen=[0,0]
        for paid,j in enumerate(q['requests'],1):
            s,i=j['slot'],j['index']; c=rows[s]['candidates'][seen[s]]; seen[s]+=1
            require(c==dict(banks[s][i],status='complete',contexts=c['contexts'],cost=c['cost']), 'complete exact cold candidate')
            require(type(c['cost']) in (int,float) and c['cost']==m.cost(tasks[s],c['contexts']), 'all costs recomputed')
            emit('candidate_request',[arm,s],paid=paid,candidate=dict(banks[s][i],status='attempted'))
            pathcheck(s,c['commands'],c['contexts'],arm,'search')
            emit('candidate_complete',[arm,s],candidate=c)
        for s,r in enumerate(rows):
            require(r['task_id']==p['slots'][s]['task_id'] and r['status']=='model_plan_sealed' and len(r['candidates'])==seen[s], 'branch identity/census')
            w=min(r['candidates'],key=lambda c:(c['cost'],c['index']))
            require(type(r['winner_index']) is int and r['winner_index']==w['index'] and canonical(r['plan'])==canonical(plan(p['slots'][s],arm,w,digest(banks[s]))), 'stable winner/selection seal')
            emit('winner_frozen',[arm,s],index=w['index'],cost=w['cost'],commands=w['commands'])
            require(set(r['replays'])=={'whole_replay','segmented_replay'},'two replay denominator')
            for mode in ('whole_replay','segmented_replay'):
                path=r['replays'][mode]; require(path==w['contexts'],'all replay contexts exact')
                emit('replay_request',[arm,s],mode=mode,commands=w['commands']); pathcheck(s,w['commands'],path,arm,mode)
                emit('replay_complete',[arm,s],mode=mode,contexts=path)
            emit('selection_sealed',[arm,s],plan=r['plan'])
            expected=6*q['blocks'][s]+4
            require(all(sum(e['scope']==[arm,s] and e['resource']==k for e in attempts)==expected for k in m.NEURAL),'per-root actual API/module vector')
        require(all(sum(e['scope'][0]==arm and e['resource']==k for e in attempts)==128 for k in m.NEURAL),'per-arm equal128 actual')
    kinds={e['kind'] for e in semantics}
    require(canonical([e for e in events if e['kind'] in kinds])==canonical(semantics),'complete semantic execution order')
    require(canonical([e for e in events if e['kind']=='native_packet'])==canonical(packets),'all actual packet receipts')
    require([(e['scope'],e['context']) for e in events if e['kind']=='context_input']==contexts,'every literal history/action context')
    outputs=[e for e in events if e['kind']=='predict_output']
    require([(e['scope'],m.unblob(e['array'])[0,-1].tolist()) for e in outputs]==successors,'all predictor successors')
    modules={}
    for e in events:
        if e['kind'] not in ('module_input','module_output'): continue
        key=(e['entry'],e['kind']); require(type(e['entry']) is int and 0<=e['entry']<len(attempts) and key not in modules,'unique bounded module receipt')
        require(attempts[e['entry']]['resource']==e['resource'] and attempts[e['entry']]['scope']==e['scope'],'module scope/entry')
        modules[key]=[m.unblob(a) for a in e['arrays']] if isinstance(e['arrays'],list) else m.unblob(e['arrays'])
    wanted={(e['index'],k) for e in attempts if e['resource'] in ('action_encoder','predictor','pred_proj') for k in ('module_input','module_output')}
    require(set(modules)==wanted,'all native arrays')
    require([e['resource'] for e in attempts]==['action_encoder','predict','predictor','pred_proj']*512,'native forward order; no hidden math')
    for n in range(512):
        ae,api,pred,proj=attempts[4*n:4*n+4]
        def get(e,k): return modules[e['index'],'module_'+k]
        ai,ao,pi,po,qi,qo=get(ae,'input'),get(ae,'output'),get(pred,'input'),get(pred,'output'),get(proj,'input'),get(proj,'output')
        native=m.unblob(outputs[n]['array']); scope,c=contexts[n]
        require(ae['scope']==api['scope']==pred['scope']==proj['scope']==outputs[n]['scope']==scope,'native exact request scope')
        require(len(ai)==1 and len(pi)==2 and len(qi)==1,'native module arity')
        require(ai[0].shape==(1,3,10) and pi[0].shape==ao.shape==po.shape==native.shape==(1,3,192) and qo.shape==(3,192),'actual full batch/token shape')
        require(all(a.dtype==np.float32 for a in ai+[ao]+pi+[po]+qi+[qo,native]),'actual float32')
        require(np.array_equal(ai[0][0],c['action_history']) and np.array_equal(pi[0][0],c['latent_history']) and
            np.array_equal(pi[1],ao) and np.array_equal(qi[0],po.reshape(3,192)) and np.array_equal(qo,native.reshape(3,192)), 'full native module bridge')
    require(result['paired_summary']==paired(result),'paired model-Q reconstruction')
    return [r['plan'] for r in result['rows']]


def run(prepared,expected,output,authorization):
    require(authorization=='MAIN_APPROVED_HOST_CUDA', 'MAIN audit/scheduling acknowledgement required')
    output=exclusive(output); m=dependency().old
    limits=m.fidelity.Limits(free=lambda:shutil.disk_usage(output.parent).free)
    p,tasks,banks=verify(prepared,expected); limits.check()
    store=m.Store(output); meter=Meter(store.event,limits); result=census(p)
    errors=[]; model=None; before=after=host=None; source_ok=False; complete=False
    try:
        store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
        store.put('banks.json.gz',banks); store.event(dict(kind='panel_census',rows=deepcopy(result['rows'])))
        os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0'); device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        limits.memory=lambda:(torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device))
        limits.check(); model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        m.enc.frozen(model); before=m.enc.tensor_hash(model); require(before==p['expected_model_hash'],'original frozen model')
        meter.sync=lambda:torch.cuda.synchronize(device)
        with sdpa_kernel(SDPBackend.MATH): panel(p,tasks,banks,[m.Native(model,device,t) for t in tasks],meter,result)
        require(meter.counts()=={r:dict(attempted=v,completed=v) for r,v in maxima().items()},'complete actual resource vector')
    except BaseException as exc:
        errors.append(m.enc.error('execution',exc))
        try: store.event(dict(kind='execution_error',error=errors[0]))
        except BaseException as err: errors.append(m.enc.error('error_journal',err))
    finally:
        for r in result['rows']:
            if r['status']=='attempted': r['status']='failed_partial'
            elif errors and r['status']=='not_started': r['status']='blocked_not_dispatched'
            for c in r['candidates']:
                if c['status']=='attempted': c['status']='failed_partial'
        if model is not None:
            try: m.enc.frozen(model); after=m.enc.tensor_hash(model); require(before==after,'model unchanged')
            except BaseException as exc: errors.append(m.enc.error('final_model',exc))
        try: verify(prepared,expected); source_ok=True
        except BaseException as exc: errors.append(m.enc.error('final_source',exc))
        try: limits.check()
        except BaseException as exc: errors.append(m.enc.error('final_limits',exc))
        try: store.put('panel.json.gz',result,final=True)
        except BaseException as exc: errors.append(m.enc.error('partial_panel',exc))
        try: store.close()
        except BaseException as exc: errors.append(m.enc.error('journal_finalize',exc))
        hashes={}
        try: hashes={f.name:sha(f) for f in store.output.iterdir()}
        except BaseException as exc: errors.append(m.enc.error('artifact_hash',exc))
        complete=not errors and source_ok and before is not None and before==after and all(r['status']=='model_plan_sealed' for r in result['rows'])
        report=dict(schema=NAME+'.report',status='cold_queues_executed_not_admitted' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).absolute()),prepare_sha256=expected,artifact_sha256=hashes,errors=errors,
            resources=meter.counts(),actual_module_shapes=meter.module_shapes,source_verified=source_ok,
            model_hash_before=before,model_hash_after=after,forbidden_visual=meter.forbidden_visual,
            new_encoder=0,new_projector=0,new_physics=0,host=host,limits=limits.receipt(),
            protocol=protocol(),paired_summary=result.get('paired_summary'),commands_released=complete,
            physical_release=False,backward_engine_calls=None,**FLAGS)
        try: store.put('report.json',report,final=True)
        except BaseException as exc: complete=False; print(repr(exc),flush=True)
    require(complete,'partial preserved; no retry: '+repr(errors))
    return dict(output=str(output),report_sha256=sha(output/'report.json'))


def consume(output,expected):
    start=time.monotonic(); prior.hash_value(expected); output=Path(output).absolute(); m=dependency().old
    require(sha(output/'report.json')==expected,'external actual report SHA'); r=m.read(output/'report.json')
    require(r['schema']==NAME+'.report' and r['status']=='cold_queues_executed_not_admitted' and r['errors']==[],'complete cold producer')
    require(all(r[k] is False for k in FLAGS) and r['commands_released'] is True and r['source_verified'] is True and
        r['physical_release'] is False and r['forbidden_visual']==[] and r['new_encoder']==r['new_projector']==r['new_physics']==0,'no hidden acquisition/scientific admission')
    p,t,b=verify(r['prepared'],r['prepare_sha256'])
    require(r['model_hash_before']==r['model_hash_after']==p['expected_model_hash'] and r['protocol']==protocol(),'frozen model/protocol')
    host=r['host']; limits=r['limits']
    require(isinstance(host,dict) and host['device']=='cuda:0' and host['allocation_sum']==4. and
        bool(host['visible_nvidia_nodes']) and host['torch_cuda'] is not None,'actual host CUDA preflight receipt')
    require(type(limits['elapsed_seconds']) in (int,float) and math.isfinite(limits['elapsed_seconds']) and
        0<=limits['elapsed_seconds']<600 and 0<=limits['peak_allocated_bytes']<=3*1024**3 and
        0<=limits['peak_reserved_bytes']<=4*1024**3 and limits['minimum_observed_free_bytes']>=prior.FLOOR,'actual limits receipt')
    names={'binding.json','banks.json.gz','panel.json.gz','events.jsonl.gz'}
    require(set(r['artifact_sha256'])==names and {f.name for f in output.iterdir()}==names|{'report.json'},'exact artifacts')
    for n,h in r['artifact_sha256'].items(): require(sha(output/n)==h,'artifact SHA '+n)
    require(m.read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']) and m.read(output/'banks.json.gz')==b,'binding/bank equality')
    plain,eof=m.blind.inflate((output/'events.jsonl.gz').read_bytes(),128*1024**2)
    require(eof and plain.endswith(b'\n'),'complete journal'); events=[m.blind.decode(line) for line in plain.splitlines()]
    result=m.read(output/'panel.json.gz',32*1024**2); plans=validate_saved(p,t,b,result,events)
    require(r['resources']=={k:dict(attempted=v,completed=v) for k,v in maxima().items()} and r['paired_summary']==result['paired_summary'],'report resources/paired')
    shapes=[dict(entry=e['entry'],resource=e['resource'],shape=list(m.unblob(e['arrays'][0]).shape),scope=e['scope']) for e in events if e['kind']=='module_input']
    require(r['actual_module_shapes']==shapes,'actual shape receipt')
    collisions=[]
    for s in (0,1):
        ps=[next(v for v in plans if v['arm']==a and v['source_slot']==s) for a in ARMS]
        collisions.append(dict(source_slot=s,winner_indices=[v['candidate_index'] for v in ps],
            command_collision_pairs=[[ARMS[i],ARMS[j]] for i in range(4) for j in range(i+1,4) if ps[i]['commands']==ps[j]['commands']]))
    for f,h in p['bindings_sha256'].items(): require(sha(f)==h,'final input seal')
    require(sha(output/'report.json')==expected,'final report SHA')
    inclusive={a:{} for a in ARMS}
    for e in events:
        if e['kind']=='resource_return':
            v=e['entry']; a=v['scope'][0]; k=v['resource']
            inclusive[a][k]=inclusive[a].get(k,0.)+v['ended_seconds']-v['started_seconds']
    return dict(schema=NAME+'.consume',status='cold_saved_verified_not_admitted',report_sha256=expected,
        plans=plans,paired_model_only=result['paired_summary'],collisions=collisions,
        resources=r['resources'],module_sample_axes=dependency().sample_axes(events),
        per_arm_sample_axes={a:dependency().sample_axes([e for e in events if e.get('scope',[None])[0]==a]) for a in ARMS},
        execution_wall_seconds=r['limits']['elapsed_seconds'],saved_validation_wall_seconds=time.monotonic()-start,
        historical_previous_panel_resources=p['historical_resources'],historical_encoder=p['historical_encoder'],
        historical_anchor_receipts=p['historical_anchor_receipts'],backward_engine_calls=None,
        inclusive_resource_wall_seconds=inclusive,wall_semantics='nested inclusive; not additive total wall or FLOPs',
        reused_signal_acquisition={k:v for k,v in protocol().items() if k.startswith('shared_historical_')},
        saved_validation_array_SVD_calls=None,new_neural_calls=0,new_physics_calls=0,physical_release=False,**FLAGS)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('diagnose','prepare','verify','queue','run','consume'))
    parser.add_argument('--prepared'); parser.add_argument('--sha256'); parser.add_argument('--output')
    parser.add_argument('--arm',choices=ARMS); parser.add_argument('--cuda-output'); parser.add_argument('--authorization')
    a=parser.parse_args(argv)
    if a.action=='diagnose': value=historical()[-1]
    elif a.action=='prepare': value=write_new(a.output,material()[0])
    elif a.action=='verify':
        p,_,_=verify(a.prepared,a.sha256); value=dict(status='verified_no_execution',allocations={k:v['blocks'] for k,v in p['queues'].items()},**FLAGS)
    elif a.action=='queue': value=queue(a.prepared,a.sha256,a.arm)
    elif a.action=='run': value=run(a.prepared,a.sha256,a.output,a.authorization)
    else: value=consume(a.cuda_output,a.sha256)
    if a.output and a.action not in ('prepare','run'): value=write_new(a.output,value)
    print(canonical(value).decode()); return 0


if __name__=='__main__': raise SystemExit(main())
