"""Fixed rank-reserve experiment. Only run loads frozen host-CUDA model.

Reuse immutable normalization/transition/blob/durable-store primitives, never
replace old scheduler globals. Own queue, scope caps, schemas and saved check.
"""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import time

import numpy as np

from experiments.spectral_reachability import chat_p3_20261027_scheduler as rule
from experiments.spectral_reachability import moving_history_budget_scheduler_cuda_20261027_v1 as old

ROOT = rule.ROOT
NAME = 'chat_p3_20261027_cuda'
ARMS = rule.ARMS
SEAL = ROOT/'results/chat_p3_20261027_queue_seal.json'
SEAL_SHA = '2f39483bceb87e638f74b2da170db8cf7ddbf6a1050df4137bbbcf9dde37ed78'
OLD_PREPARE = ROOT/'results/moving_history_budget_scheduler_cuda_20261027_v1_prepare_main01.json'
OLD_PREPARE_SHA = '330bb32be9cc3ccf330835b94e336fff9aad057a6b85ce468678d65397369aaf'
FLAGS = dict(P1_admission=False, P3_admission=False, freshness=False,
             independent_confirmation=False, compute_advantage=False, walltime_advantage=False)
require, read, sha = old.require, old.read, old.sha


def exclusive(path):
    path = Path(path).absolute()
    require(path.parent == ROOT/'results' and path.name.startswith('chat_p3_20261027_') and not path.exists(),
            'exclusive new result prefix')
    return path


def protocol():
    return dict(schema=NAME+'.protocol', arms=list(ARMS), rule_seal_sha256=SEAL_SHA,
                total_blocks=20, minimum_blocks=5, candidates_per_block=3, horizon=2,
                search_predict_per_arm=120, replay_predict_per_arm=8,
                shared_predict=8, maxima=old.MAXIMA, SDP='MATH', TF32=False,
                CPU_neural=False, no_retries=True, no_dedup=True,
                output_cap_bytes=old.CAP, disk_floor_bytes=old.FLOOR, wall_seconds=600,
                allocated_bytes=3*1024**3, reserved_bytes=4*1024**3,
                gain_primary='D_rank-U', gain_secondary=['D_rank-S_rank','D_rank-N'],
                scientific_endpoint_missing=True, **FLAGS)


def material():
    require(sha(SEAL)==SEAL_SHA, 'fixed external queue seal')
    # Complete original consumer certifies shared inputs; no physics outcomes.
    require(read(SEAL)==rule.build(), 'original source/queue seal reconstruction')
    p,tasks,banks = old.verify(OLD_PREPARE, OLD_PREPARE_SHA)
    files = [Path(__file__), ROOT/'tests/chat_p3_20261027_test_cuda.py',
             ROOT/'innerdocs/chat_p3_20261027_cuda_prereg.md',
             ROOT.parent/'docs/chat_p3_20261027_prereg.md', Path(rule.__file__), SEAL,
             rule.OLD/'report.json', rule.OLD/'panel.json.gz', rule.OLD/'banks.json.gz',
             rule.OLD/'binding.json', rule.OLD/'events.jsonl.gz', OLD_PREPARE]
    bindings = dict(p['bindings_sha256'])
    bindings.update({str(f.resolve()):sha(f) for f in files})
    return dict(schema=NAME+'.prepare', protocol=protocol(), rule_seal=str(SEAL),
                bindings_sha256=bindings, policy=p['policy'], expected_model_hash=p['expected_model_hash'],
                slots=p['slots'], historical_encoder=p['encoder_receipt'],
                historical_anchor_receipts=p['historical_anchor_receipts']),tasks,banks


def prepare(output):
    p,_,_=material(); path=exclusive(output); raw=rule.canonical(p)
    require(len(raw)<=256*1024 and shutil.disk_usage(path.parent).free>=old.FLOOR+len(raw), 'prepare cap/floor')
    with path.open('xb') as f:
        require(f.write(raw)==len(raw), 'prepare short write'); f.flush(); os.fsync(f.fileno())
    return dict(path=str(path), sha256=sha(path))


def verify(path, expected):
    require(expected is not None and sha(path)==expected, 'external prepare SHA')
    p,t,b=material(); require(read(path)==p, 'prepare/source/input equality')
    return p,t,b


class Meter(old.Meter):
    """Own rank-arm scope caps. Shape samples and times are separate axes."""
    def __init__(self, sink, limits, sync=lambda:None):
        super().__init__(sink,limits,sync)
        self.module_shapes=[]

    def emit(self,kind,**fields):
        if kind=='module_input':
            arrays=fields['arrays']
            require(isinstance(arrays,list) and len(arrays)>0, 'actual module tensor inputs')
            a=old.unblob(arrays[0]); resource=fields['resource']
            require(a.dtype==np.float32 and a.ndim in (2,3), 'native module shape')
            self.module_shapes.append(dict(entry=fields['entry'],resource=resource,
                                           shape=list(a.shape), scope=deepcopy(self.scope)))
        super().emit(kind,**fields)

    def begin(self,resource):
        self.limits.check()
        require(not self.failed and resource in old.MAXIMA and self.scope is not None, 'scoped no-retry dispatch')
        group,slot=self.scope
        require(group in ('shared',)+ARMS and type(slot) is int and slot in (0,1), 'own scope')
        require(sum(e['resource']==resource for e in self.events)<old.MAXIMA[resource], 'global predispatch cap')
        scoped=[e for e in self.events if e['resource']==resource and e['scope']==self.scope]
        if resource in old.NEURAL:
            cap=4 if group=='shared' else (64 if group in ('U','N') else 94)
            require(len(scoped)<cap, 'root dispatch cap')
            if group!='shared':
                require(sum(e['resource']==resource and e['scope'][0]==group for e in self.events)<128, 'arm dispatch128')
        else:
            require(group=='shared', 'search never accesses AD/SVD')
            cap=192 if resource in ('grad_api','vjp_directions','raw_backward_hook') else 1
            require(len(scoped)<cap, 'shared mathematical cap')
        e=dict(index=len(self.events),resource=resource,scope=deepcopy(self.scope),samples=1,
               completed=False,started_seconds=self.limits.clock()-self.limits.start)
        self.events.append(e); self.emit('resource_attempt',entry=deepcopy(e)); return e

    def end(self,e):
        require(not e['completed'], 'single completion'); self.sync(); self.limits.check()
        e['completed']=True; e['ended_seconds']=self.limits.clock()-self.limits.start
        self.emit('resource_return',entry=deepcopy(e))


def plan(slot,arm,winner):
    value=dict(schema=NAME+'.model_plan', **deepcopy(slot), arm=arm,
               commands=deepcopy(winner['commands']),candidate_index=winner['index'],model_cost=winner['cost'],
               physical_execution=False,selection_uses_physics=False,**FLAGS)
    value['selection_sha256']=old.core.digest(value)
    return value


def census(p):
    return dict(signals=[dict(source_slot=s,status='not_started') for s in (0,1)],allocations={},curves={},
                rows=[dict(source_slot=s,task_id=p['slots'][s]['task_id'],arm=a,status='not_started',candidates=[])
                      for a in ARMS for s in (0,1)])


def paired(result):
    costs={a:[next(r['plan']['model_cost'] for r in result['rows'] if r['arm']==a and r['source_slot']==s)
              for s in (0,1)] for a in ARMS}
    differences={a:[costs['D_rank'][s]-costs[a][s] for s in (0,1)] for a in ('U','S_rank','N')}
    return dict(status='two_known_roots_model_only',costs=costs,D_minus_control=differences,
                primary_mean=float(np.mean(differences['U'])),
                secondary_means={a:float(np.mean(differences[a])) for a in ('S_rank','N')},
                sign='negative model-Q difference favors D; not physical benefit',**FLAGS)


def panel(p,tasks,banks,backends,meter,result,sealed):
    for s in (0,1):
        meter.scope=['shared',s]; backend,task=backends[s],tasks[s]; entry=result['signals'][s]
        entry['status']='attempted'; zero=banks[s][0]
        meter.emit('nominal_request',candidate=zero)
        with backend.guard(meter):
            path=backend.rollout(task,zero['commands'],meter,'nominal')
            entry['nominal_contexts']=path; old.check_path(task,zero['commands'],path)
            meter.emit('nominal_complete',contexts=path)
            terminal,j=meter.call('jacobian',lambda:backend.jacobian(task,dict(zero,contexts=path),meter))
        entry.update(terminal=terminal.tolist(),jacobian=old.blob(j))
        meter.emit('jacobian_complete',terminal=entry['terminal'],jacobian=entry['jacobian'])
        require(np.array_equal(terminal,path[-1]['latent_history'][-1]), 'AD nominal terminal exact')
        entry['signal']=meter.call('svd',lambda:old.signal(task,terminal,j)); entry['status']='complete'
        meter.emit('signal_complete',signal=entry['signal'])
    weights=[e['signal']['w'] for e in result['signals']]
    # Already-known-root engineering replay: never alter a sealed queue if fresh
    # floating-point signals disagree. Stop and preserve the attempt instead.
    require(weights==sealed['weights'], 'fresh signal must reproduce sealed engineering weights')
    for arm in ARMS:
        blocks=rule.allocation(arm) if arm in ('U','N') else rule.allocation(arm,weights)
        jobs=rule.queue(blocks)
        require(blocks==sealed['plans'][arm]['blocks'] and jobs==sealed['plans'][arm]['jobs'], 'queue fixed before score')
        result['allocations'][arm]=blocks
        meter.scope=[arm,None]; meter.emit('queue_frozen',arm=arm,blocks=blocks,jobs=jobs)
        rows=[r for r in result['rows'] if r['arm']==arm]
        for paid,job in enumerate(jobs,1):
            s,index=job['slot'],job['index']; meter.scope=[arm,s]; row=rows[s]; row['status']='attempted'
            c=deepcopy(banks[s][index]); c['status']='attempted'; row['candidates'].append(c)
            meter.emit('candidate_request',paid=paid,candidate=c)
            with backends[s].guard(meter):
                path=backends[s].rollout(tasks[s],c['commands'],meter,'search')
            c['contexts']=path; old.check_path(tasks[s],c['commands'],path)
            c.update(cost=old.cost(tasks[s],path),status='complete'); meter.emit('candidate_complete',candidate=c)
            if paid in (30,60):
                curve=dict(paid_candidates=paid,best_costs=[min(x['cost'] for x in r['candidates']) for r in rows])
                result['curves'].setdefault(arm,[]).append(curve); meter.emit('paid_curve',curve=curve)
        for s,row in enumerate(rows):
            meter.scope=[arm,s]; winner=min(row['candidates'],key=lambda c:(c['cost'],c['index']))
            row['winner_index']=winner['index']; row['replays']={}
            meter.emit('winner_frozen',index=winner['index'],cost=winner['cost'],commands=winner['commands'])
            with backends[s].guard(meter):
                for mode in ('whole_replay','segmented_replay'):
                    meter.emit('replay_request',mode=mode,commands=winner['commands'])
                    path=backends[s].rollout(tasks[s],winner['commands'],meter,mode)
                    row['replays'][mode]=path; old.check_path(tasks[s],winner['commands'],path)
                    require(path==winner['contexts'], 'exact independent replay, no tolerance')
                    meter.emit('replay_complete',mode=mode,contexts=path)
            row['plan']=plan(p['slots'][s],arm,winner); row['status']='model_plan_sealed'
            meter.emit('selection_sealed',plan=row['plan'])
    result['paired_summary']=paired(result)


def sample_axes(events):
    """Independently read actual tensor shapes, do not infer samples from API."""
    inputs=[e for e in events if e['kind']=='module_input']
    out={}
    for resource in ('action_encoder','predictor','pred_proj'):
        es=[e for e in inputs if e['resource']==resource]
        shapes=[old.unblob(e['arrays'][0]).shape for e in es]
        rank=2 if resource=='pred_proj' else 3
        require(all(len(sh)==rank for sh in shapes), 'actual module batch/token shape')
        out[resource]=dict(observed_calls=len(shapes),input_rows=sum(sh[0] for sh in shapes),
                           context_tokens=None if rank==2 else sum(sh[0]*sh[1] for sh in shapes))
    return out


def validate_saved(p,tasks,banks,result,events,sealed):
    """Full own-schema reconstruction. Never load model/score missing tails."""
    require([(r['arm'],r['source_slot']) for r in result['rows']]==[(a,s) for a in ARMS for s in (0,1)], 'all8 branches')
    attempts=[e['entry'] for e in events if e['kind']=='resource_attempt']
    returns=[e['entry'] for e in events if e['kind']=='resource_return']
    require([e['index'] for e in attempts]==list(range(len(attempts))), 'ordered attempt identity')
    returned={e['index']:e for e in returns}
    require(len(returns)==len(attempts)==len(returned), 'full returns no duplicate')
    for a in attempts:
        b=returned[a['index']]
        require(a=={k:v for k,v in b.items() if k!='ended_seconds'}|{'completed':False}, 'attempt/return identity')
        require(b['completed'] and b['ended_seconds']>=a['started_seconds'], 'actual completed timing')
    def counts(group=None,slot=None):
        es=[e for e in attempts if (group is None or e['scope'][0]==group) and (slot is None or e['scope'][1]==slot)]
        return {r:sum(e['resource']==r for e in es) for r in old.MAXIMA}
    require(counts()==old.MAXIMA, 'complete 520predict and AD/SVD vector')
    expected_contexts=[]; expected_outputs=[]; semantic=[]
    def expect(kind,scope,**fields):semantic.append(dict(kind=kind,scope=scope,**deepcopy(fields)))
    def pathcheck(s,raw,path,group):
        old.check_path(tasks[s],raw,path); before=tasks[s]['root_context']
        for k,after in enumerate(path):
            expected_contexts.append(( [group,s],old.moving.incoming(tasks[s],before,np.asarray(raw,np.float32)[5*k:5*k+5]) ))
            expected_outputs.append(([group,s],after['latent_history'][-1])); before=after
    for s,e in enumerate(result['signals']):
        require(e['source_slot']==s and e['status']=='complete', 'shared signal census')
        j=old.unblob(e['jacobian'])
        require(old.signal(tasks[s],e['terminal'],j)==e['signal'], 'SVD signal reconstruction')
        require(e['terminal']==e['nominal_contexts'][-1]['latent_history'][-1], 'AD value/nominal equality')
        for _ in range(2):pathcheck(s,banks[s][0]['commands'],e['nominal_contexts'],'shared')
        expect('nominal_request',['shared',s],candidate=banks[s][0])
        expect('nominal_complete',['shared',s],contexts=e['nominal_contexts'])
        expect('jacobian_complete',['shared',s],terminal=e['terminal'],jacobian=e['jacobian'])
        expect('signal_complete',['shared',s],signal=e['signal'])
        jr=[v for v in events if v['kind']=='jacobian_row' and v['scope']==['shared',s]]
        br=[v for v in events if v['kind']=='raw_backward' and v['scope']==['shared',s]]
        require([v['row'] for v in jr]==list(range(192)) and np.array_equal(np.stack([old.unblob(v['array']) for v in jr]),j), 'complete J rows')
        require(len(br)==192 and np.array_equal(np.stack([old.unblob(v['array']).reshape(20) for v in br]),j), 'backward callback arrays')
        c=counts('shared',s)
        require(all(c[r]==4 for r in old.NEURAL) and all(c[r]==192 for r in ('grad_api','vjp_directions','raw_backward_hook')) and c['svd']==c['jacobian']==1, 'shared per root')
    weights=[e['signal']['w'] for e in result['signals']]
    require(weights==sealed['weights'], 'sealed engineering signal reproduction')
    for arm in ARMS:
        blocks=rule.allocation(arm) if arm in ('U','N') else rule.allocation(arm,weights)
        jobs=rule.queue(blocks); rows=[r for r in result['rows'] if r['arm']==arm]
        require(blocks==result['allocations'][arm]==sealed['plans'][arm]['blocks'] and jobs==sealed['plans'][arm]['jobs'], 'exact prespecified treatment')
        expect('queue_frozen',[arm,None],arm=arm,blocks=blocks,jobs=jobs)
        seen=[0,0]; curves=[]
        for paid,job in enumerate(jobs,1):
            s,index=job['slot'],job['index']; c=rows[s]['candidates'][index]
            require(c['status']=='complete' and {k:c[k] for k in banks[s][index]}==banks[s][index], 'entire exact candidate prefix')
            expect('candidate_request',[arm,s],paid=paid,candidate=dict(banks[s][index],status='attempted'))
            pathcheck(s,c['commands'],c['contexts'],arm)
            require(c['cost']==old.cost(tasks[s],c['contexts']), 'all candidate costs reconstructed')
            expect('candidate_complete',[arm,s],candidate=c); seen[s]+=1
            if paid in (30,60):
                curve=dict(paid_candidates=paid,best_costs=[min(x['cost'] for x in r['candidates'][:seen[q]]) for q,r in enumerate(rows)])
                curves.append(curve); expect('paid_curve',[arm,s],curve=curve)
        require(curves==result['curves'][arm], 'only paid best-so-far')
        for s,r in enumerate(rows):
            require(r['status']=='model_plan_sealed' and r['task_id']==p['slots'][s]['task_id'] and len(r['candidates'])==3*blocks[s], 'full stable branch identity/count')
            w=min(r['candidates'],key=lambda c:(c['cost'],c['index']))
            require(r['winner_index']==w['index'] and r['plan']==plan(p['slots'][s],arm,w), 'stable winner and own seal')
            expect('winner_frozen',[arm,s],index=w['index'],cost=w['cost'],commands=w['commands'])
            require(set(r['replays'])=={'whole_replay','segmented_replay'}, 'both independent replays')
            for mode in ('whole_replay','segmented_replay'):
                path=r['replays'][mode]
                require(path==w['contexts'], 'all replay context exact')
                pathcheck(s,w['commands'],path,arm)
                expect('replay_request',[arm,s],mode=mode,commands=w['commands'])
                expect('replay_complete',[arm,s],mode=mode,contexts=path)
            expect('selection_sealed',[arm,s],plan=r['plan'])
            require(all(counts(arm,s)[r]==6*blocks[s]+4 for r in old.NEURAL), 'per-root budget actually dispatched')
        require(all(counts(arm)[r]==128 for r in old.NEURAL) and all(counts(arm)[r]==0 for r in old.MAXIMA if r not in old.NEURAL), 'arm128; no search AD')
    kinds={e['kind'] for e in semantic}
    require([e for e in events if e['kind'] in kinds]==semantic, 'all semantic requests/returns/selection ordered')
    actual_contexts=[(e['scope'],e['context']) for e in events if e['kind']=='context_input']
    require(actual_contexts==expected_contexts, 'independently reconstructed every float32 history')
    outputs=[e for e in events if e['kind']=='predict_output']
    require([(e['scope'],old.unblob(e['array'])[0,-1].tolist()) for e in outputs]==expected_outputs, 'all actual predictor successor arrays')
    modules={}; items=[]
    for e in events:
        if e['kind'] not in ('module_input','module_output'):continue
        key=(e['entry'],e['kind']); require(key not in modules, 'unique actual module input/output')
        require(attempts[e['entry']]['resource']==e['resource'] and attempts[e['entry']]['scope']==e['scope'], 'module ledger scope')
        modules[key]=[old.unblob(a) for a in e['arrays']] if isinstance(e['arrays'],list) else old.unblob(e['arrays'])
        items.append(key)
    expected_keys={(e['index'],k) for e in attempts if e['resource'] in ('action_encoder','predictor','pred_proj') for k in ('module_input','module_output')}
    require(set(items)==expected_keys and len(items)==len(expected_keys), 'full native module arrays')
    neural=[e for e in attempts if e['resource'] in old.NEURAL]
    require([e['resource'] for e in neural]==['action_encoder','predict','predictor','pred_proj']*520, 'actual native entry order')
    for n in range(520):
        ae,api,pred,proj=neural[4*n:4*n+4]
        get=lambda e,k:modules[e['index'],'module_'+k]
        ai,ao,pi,po,qi,qo=get(ae,'input'),get(ae,'output'),get(pred,'input'),get(pred,'output'),get(proj,'input'),get(proj,'output')
        native=old.unblob(outputs[n]['array']); scope,context=expected_contexts[n]
        require(ae['scope']==api['scope']==pred['scope']==proj['scope']==outputs[n]['scope']==scope, 'request scope identity')
        require(len(ai)==1 and len(pi)==2 and len(qi)==1, 'actual native input arity')
        require(np.array_equal(ai[0][0],context['action_history']) and np.array_equal(pi[0][0],context['latent_history']), 'actual AE/predictor full input context')
        require(ao.shape==po.shape==native.shape==(1,3,192) and qo.shape==(3,192), 'batch/context dimensions')
        require(all(a.dtype==np.float32 for a in ai+[ao]+pi+[po]+qi+[qo,native]), 'no precision downgrade')
        require(np.array_equal(pi[1],ao) and np.array_equal(qi[0],po.reshape(3,192)) and np.array_equal(qo,native.reshape(3,192)), 'actual module input-output chain')
    require(result['paired_summary']==paired(result), 'paired model-only result reconstruction')
    return [r['plan'] for r in result['rows']]


def run(prepared, expected, output):
    output=exclusive(output)
    limits=old.fidelity.Limits(free=lambda:shutil.disk_usage(output.parent).free)
    p,tasks,banks=verify(prepared,expected); sealed=read(SEAL); limits.check()
    store=old.Store(output); meter=Meter(store.event,limits); result=census(p)
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
        old.enc.frozen(model); before=old.enc.tensor_hash(model)
        require(before==p['expected_model_hash'], 'original frozen model')
        meter.sync=lambda:torch.cuda.synchronize(device)
        print('Frozen rank-reserve engineering comparison: fixed queues, no physical benefit or admission claim.',flush=True)
        with sdpa_kernel(SDPBackend.MATH):
            panel(p,tasks,banks,[old.Native(model,device,t) for t in tasks],meter,result,sealed)
        require(meter.counts()=={r:dict(attempted=v,completed=v) for r,v in old.MAXIMA.items()}, 'actual complete vector')
    except BaseException as exc:
        errors.append(old.enc.error('execution',exc))
        try:store.event(dict(kind='execution_error',error=errors[0]))
        except BaseException as exc:errors.append(old.enc.error('error_journal',exc))
    finally:
        for row in result['signals']+result['rows']:
            if row['status']=='attempted':row['status']='failed_partial'
            elif errors and row['status']=='not_started':row['status']='blocked_not_dispatched'
            for candidate in row.get('candidates',[]):
                if candidate['status']=='attempted':candidate['status']='failed_partial'
        if model is not None:
            try:
                old.enc.frozen(model); after=old.enc.tensor_hash(model); require(before==after,'final model unchanged')
            except BaseException as exc:errors.append(old.enc.error('final_model',exc))
        try:verify(prepared,expected); source_ok=True
        except BaseException as exc:errors.append(old.enc.error('final_source',exc))
        try:limits.check()
        except BaseException as exc:errors.append(old.enc.error('final_limits',exc))
        try:store.put('panel.json.gz',result,final=True)
        except BaseException as exc:errors.append(old.enc.error('partial_panel',exc))
        try:store.close()
        except BaseException as exc:errors.append(old.enc.error('journal_finalize',exc))
        hashes={}
        try:hashes={f.name:sha(f) for f in store.output.iterdir()}
        except BaseException as exc:errors.append(old.enc.error('artifact_hash',exc))
        complete=not errors and source_ok and before is not None and before==after and all(r['status']=='model_plan_sealed' for r in result['rows'])
        report=dict(schema=NAME+'.report',status='rank_queues_executed_not_admitted' if complete else 'partial_or_failed',
                    prepared=str(Path(prepared).absolute()),prepare_sha256=expected,artifact_sha256=hashes,
                    errors=errors,first_error=errors[0] if errors else None,resources=meter.counts(),
                    scope_resources={g:{str(s):meter.counts(g,s) for s in (0,1)} for g in ('shared',)+ARMS},
                    actual_module_shapes=meter.module_shapes, source_verified=source_ok,
                    model_hash_before=before,model_hash_after=after,model_unchanged=before==after if before is not None else None,
                    forbidden_visual=meter.forbidden_visual,new_encoder=0,new_projector=0,new_physics=0,
                    limits=limits.receipt(),host=host,protocol=protocol(),
                    paired_summary=result.get('paired_summary'),commands_released=complete,
                    resource_time_semantics='nested inclusive module/API wall timestamps; never sum as FLOPs',
                    backward_engine_calls=None,**FLAGS)
        try:store.put('report.json',report,final=True)
        except BaseException as exc:
            errors.append(old.enc.error('final_report',exc)); complete=False; print(repr(errors),flush=True)
    require(complete, 'partial preserved; no automatic retry: '+repr(errors))
    return dict(output=str(output),report_sha256=sha(output/'report.json'),status=report['status'])


def consume(output, report_sha):
    output=Path(output).absolute(); require(sha(output/'report.json')==report_sha,'external actual report SHA')
    r=read(output/'report.json')
    require(r['schema']==NAME+'.report' and r['status']=='rank_queues_executed_not_admitted' and r['errors']==[] and r['first_error'] is None, 'complete new producer only')
    require(all(r[k] is False for k in FLAGS) and r['new_encoder']==r['new_projector']==r['new_physics']==0 and r['forbidden_visual']==[], 'no admission or hidden acquisition')
    require(r['commands_released'] is True and r['source_verified'] is True and r['model_unchanged'] is True, 'complete model/source/selection')
    p,tasks,banks=verify(r['prepared'],r['prepare_sha256'])
    require(r['model_hash_before']==r['model_hash_after']==p['expected_model_hash'] and r['protocol']==protocol(),'frozen model/protocol')
    names={'binding.json','banks.json.gz','panel.json.gz','events.jsonl.gz'}
    require(set(r['artifact_sha256'])==names and {f.name for f in output.iterdir()}==names|{'report.json'}, 'exact artifact census')
    for n,h in r['artifact_sha256'].items():require(sha(output/n)==h, 'artifact SHA '+n)
    require(read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']) and read(output/'banks.json.gz')==banks, 'prepare/bank equality')
    plain,eof=old.blind.inflate((output/'events.jsonl.gz').read_bytes(),128*1024**2)
    require(eof and plain.endswith(b'\n'), 'complete bounded journal')
    events=[old.blind.decode(line) for line in plain.splitlines()]
    result=read(output/'panel.json.gz',32*1024**2)
    plans=validate_saved(p,tasks,banks,result,events,read(SEAL))
    require(r['resources']=={k:dict(attempted=v,completed=v) for k,v in old.MAXIMA.items()} and r['paired_summary']==result['paired_summary'], 'report resources/summary')
    attempts=[e['entry'] for e in events if e['kind']=='resource_attempt']
    scopes={g:{str(s):{k:dict(attempted=sum(e['resource']==k and e['scope']==[g,s] for e in attempts),
                             completed=sum(e['resource']==k and e['scope']==[g,s] for e in attempts))
                      for k in old.MAXIMA} for s in (0,1)} for g in ('shared',)+ARMS}
    require(r['scope_resources']==scopes, 'actual scope ledger reconstructed')
    shapes=[dict(entry=e['entry'],resource=e['resource'],shape=list(old.unblob(e['arrays'][0]).shape),scope=e['scope']) for e in events if e['kind']=='module_input']
    require(r['actual_module_shapes']==shapes, 'every sample axis shape bound to actual input')
    for path,h in p['bindings_sha256'].items():require(sha(path)==h, 'final input/source SHA')
    require(sha(output/'report.json')==report_sha, 'final report SHA')
    return dict(schema=NAME+'.consume',status='actual_rank_queues_verified_not_admitted',report_sha256=report_sha,
                plans=[dict(plan,whole_run_report_sha256=report_sha) for plan in plans],
                module_sample_axes=sample_axes(events),resources=r['resources'],scope_resources=scopes,
                paired_summary=r['paired_summary'],new_wall_seconds=r['limits']['elapsed_seconds'],
                equal_predict_API_budget=True,equal_FLOPs=False,**FLAGS)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['prepare','verify','run','consume'])
    parser.add_argument('--output',type=Path); parser.add_argument('--prepared',type=Path); parser.add_argument('--sha256')
    parser.add_argument('--receipt',type=Path)
    args=parser.parse_args()
    if args.action=='prepare': value=prepare(args.output)
    elif args.action=='verify':
        verify(args.prepared,args.sha256); value={'status':'verified','prepare_sha256':args.sha256}
    elif args.action=='run': value=run(args.prepared,args.sha256,args.output)
    else:
        value=consume(args.output,args.sha256)
        if args.receipt:
            path=exclusive(args.receipt)
            with path.open('xb') as f:
                raw=rule.canonical(value);require(f.write(raw)==len(raw),'short receipt'); f.flush(); os.fsync(f.fileno())
            value={'status':value['status'],'receipt':str(path),'sha256':sha(path),'module_sample_axes':value['module_sample_axes'],'paired_summary':value['paired_summary']}
    print(rule.canonical(value).decode(),flush=True)


if __name__=='__main__':
    main()
