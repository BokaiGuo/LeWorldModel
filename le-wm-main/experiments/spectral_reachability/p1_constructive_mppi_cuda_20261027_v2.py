"""Project-kernel-faithful MPPI mathematics with FINAL NOMINAL dispatch.

Box-parameter proposals are mapped into original legal packet logits for costs.
Not an official or density-corrected MPPI reproduction. MAIN CUDA only.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import p1_constructive_sampling_cuda_20261027_v1 as sampling
from experiments.spectral_reachability import d2p_sampling_baselines as kernels

NAME='p1_constructive_mppi_cuda_20261027_v2'
ARM='kernel_faithful_mppi_nominal'
ROOT=sampling.ROOT
base,gn,enc,core=sampling.base,sampling.gn,sampling.enc,sampling.core
require,sha,read,canonical=base.require,base.sha,base.read,base.canonical
SAMPLING_SHA='5fe2af6256071d4a056a9f40908ee616c284e7b72e61a5c1fc5a79e359b86862'
KERNEL_SHA='e1978044c81c6313b4b0a18fb7417cca4c46578860cc382909770641b4aafc04'
TEMPERATURE,NOISE_SCALE=1.,.5


def seed(slot):
    require(type(slot) is int and slot in (0,1),'two fixed blind slots')
    return 20261029+slot*101


def budget():return dict(predict=72,action_encoder=72)


def schedule():
    return [('initial',-1,1),('proposals',0,16),('updated_nominal',0,1),('proposals',1,16),('updated_nominal',1,1)]


def contract():
    return dict(arm=ARM,roots=2,seed=[seed(0),seed(1)],iterations=2,samples=16,temperature=1.,noise_scale=.5,
        initial='zero2x10, scored',kernel='d2p_sampling_baselines.mppi',kernel_source_sha256=KERNEL_SHA,
        reference_status='project NumPy reference, not pinned author/official MPPI source',official_MPPI=False,
        kernel_parameter_domain='float64 x in [-.75sqrt5,+.75sqrt5]^2x10; Gaussian before box clipping covariance .25 I20',
        model_action_map='logits=sealed packet-ball projection(x,.75sqrt5), commands=float32 tanh(logits).reshape10x2',
        update='UNMODIFIED kernel weights exp(-(J-minJ)/1); nominal=boxclip(sum weights*x), NOT average projected logits',
        cost='J(x)=original identity192 terminal mean-square after action_map(x); explicit control coefficient0, no intermediate/control/cross term',
        importance_correction=False,projection_density_correction=False,
        dispatch='last updated nominal x at evaluation index34; kernel best-ever return diagnostic ONLY; never substitute best sample',
        initial_evaluations=1,proposal_evaluations=32,updated_nominal_evaluations=2,kernel_H2_evaluations=35,
        fresh_nominal_H2_replays=1,total_H2_evaluations=36,candidate_predict=70,replay_predict=2,
        per_root_budget=budget(),total_budget={k:2*v for k,v in budget().items()},
        AD_API=0,mathematical_VJP=0,jacobian=0,svd=0,solve=0,image_encoder=0,image_projector=0,new_physics=0,
        replay='one fresh complete H2 of final nominal; no independent segmented/triple replay claim',
        numerical=dict(SDP='MATH',matmul_TF32=False,cudnn_TF32=False),training=False,new_Q=False,
        original_Q_normalizer_history_model=True,with_target=False,P1_admission=False,new_independent_confirmation=False,
        equal_D2P_compute_claim=False,equal_sampling64_compute_claim=False,paper_strong_MPPI_requirement_closed=False,
        v1_status='exploratory best-evaluated shooting, unchanged; not main-table MPPI evidence',
        inherited_encoding='separate historical cost, not free total pipeline',output_cap_bytes=base.CAP,reserve_bytes=base.FLOOR)


def build():
    require(sha(sampling.__file__)==SAMPLING_SHA and sha(kernels.__file__)==KERNEL_SHA,'sealed sampling and full kernel source')
    p=sampling.build();bindings=deepcopy(p['bindings_sha256'])
    for f in (Path(__file__),Path(kernels.__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):
        bindings[str(f.resolve())]=sha(f)
    for path,h in bindings.items():require(sha(path)==h,'prepare source/input TOCTOU')
    return dict(schema=NAME+'.prepare',slots=p['slots'],policy=p['policy'],expected_model_hash=p['expected_model_hash'],
        contract=contract(),bindings_sha256=bindings,global_freshness='same_two_calibration_tasks_only')


def prepare(output):
    p=build();raw=canonical(p);path=Path(output).absolute()
    require(len(raw)<256*1024 and not path.exists() and shutil.disk_usage(path.parent).free>=base.FLOOR+len(raw),'NEW prepare +200MiB floor')
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return dict(prepared=str(path),prepare_sha256=sha(path),bytes=len(raw),contract=p['contract'])


def verify(path,expected):
    require(sha(path)==expected,'external prepare SHA');p=read(path)
    require(p==build(),'complete own protocol/source/input reconstruction');return p


def run_kernel(cost_fn,slot,cap):
    # Use the ENTIRE sealed optimizer, not just softmax or a surrogate loop.
    return kernels.mppi(cost_fn,initial=np.zeros((2,10)),lower=-cap,upper=cap,
                        samples=16,iterations=2,temperature=TEMPERATURE,noise_scale=NOISE_SCALE,seed=seed(slot))


def check_single(task,p):
    require(set(p)=={'task_id','logits','commands','contexts','cost'} and p['task_id']==core.task_id(task),'single H2 task/schema')
    x=core.array(p['logits'],(2,10));raw=core.array(p['commands'],(10,2))
    require(np.array_equal(x,x.astype(np.float32)) and (np.linalg.norm(x,axis=1)<=task['packet_logit_cap']).all(),'original float32 packet balls')
    require(np.array_equal(raw,np.tanh(x.astype(np.float32)).reshape(10,2)),'exact rawcontrols map')
    gn.saved.check_path(task,raw,p['contexts'])
    require(p['cost']==core.model_cost(task,p['contexts'][-1]['latent_history'][-1]),'original-Q model cost')


class Meter(sampling.Meter):
    def __init__(self,sink,sync=lambda:None):
        gn.screen.Meter.__init__(self,budget(),sink);self.sync=sync


def update_record(it,chunk,parameters,cap):
    require(len(chunk)==16,'complete16 proposal costs before update')
    weights=kernels.softmax_weights(np.asarray([c['cost'] for c in chunk]),temperature=TEMPERATURE)
    expected=np.clip(np.tensordot(weights,np.asarray([c['parameters'] for c in chunk]),axes=(0,0)),-cap,cap)
    require(np.array_equal(expected,parameters),'actual kernel updated nominal uses PARAMETER coordinates')
    return dict(iteration=it,temperature=TEMPERATURE,weights=weights.tolist(),nominal_parameters=expected.tolist())


def optimize(task,slot,backend,meter):
    core.validate_task(task);cap=task['packet_logit_cap'];bank=[];updates=[];batches=[]
    def costs(batch):
        b=len(batches);require(b<5,'fixed five kernel callbacks');stage,it,n=schedule()[b]
        require(batch.shape==(n,2,10) and np.isfinite(batch).all(),'kernel callback schedule')
        if stage=='updated_nominal':
            update=update_record(it,bank[-16:],batch[0],cap);updates.append(update);meter.emit('kernel_update',**update)
        frozen=dict(batch=b,stage=stage,iteration=it,parameters=batch.tolist());batches.append(frozen)
        meter.emit('kernel_batch_frozen',**frozen)
        values=[]
        for parameters in batch:
            i=len(bank);meter.group=f'evaluation_{i}'
            logits=gn.screen.projected_logits(parameters,cap)
            p=gn.single(task,logits,backend,meter,'whole_replay');check_single(task,p)
            c=dict(index=i,stage=stage,iteration=it,parameters=parameters.tolist(),**sampling.compact(p))
            bank.append(c);values.append(c['cost']);meter.emit('kernel_evaluation_saved',evaluation=c)
        return np.asarray(values)
    try:best,receipt=run_kernel(costs,slot,cap)
    except BaseException as exc:
        try:meter.emit('kernel_failure',kernel_receipt=getattr(exc,'receipt',None),error_type=type(exc).__name__,message=str(exc))
        except BaseException as secondary:exc.add_note('kernel failure journal: '+repr(secondary))
        raise
    require(len(bank)==35 and len(batches)==5 and receipt['cost_evaluations']==35 and receipt['cost_calls']==5 and receipt['completed'] is True,'full initial/proposals/nominals denominator')
    best_i=min(range(35),key=lambda i:(bank[i]['cost'],i))
    require(best.tolist()==bank[best_i]['parameters'] and receipt['best_cost']==bank[best_i]['cost'],'kernel best diagnostics only')
    meter.emit('kernel_completed',receipt=receipt,best_parameters=best.tolist(),best_index=best_i)
    # Crucial: ignore the kernel's best-ever dispatch return. The actual MPPI
    # action is its last updated nominal, already scored by callback5.
    chosen=bank[34];require(chosen['stage']=='updated_nominal' and chosen['iteration']==1,'last nominal index34')
    meter.emit('nominal_dispatch_frozen',index=34,parameters=chosen['parameters'],commands=chosen['commands'],cost=chosen['cost'])
    meter.group='final_nominal_independent_H2';fresh=gn.single(task,chosen['logits'],backend,meter,'whole_replay');check_single(task,fresh)
    require(sampling.compact(fresh)=={k:chosen[k] for k in sampling.compact(fresh)},'MATH nominal search/fresh replay exact')
    base.exact_counts(meter.counts(),budget())
    return dict(task_id=core.task_id(task),source_slot=slot,arm=ARM,seed=seed(slot),evaluations=bank,batches=batches,updates=updates,
        kernel_receipt=receipt,kernel_best_parameters=best.tolist(),diagnostic_best_index=best_i,
        selected_index=34,selection='last_updated_nominal_not_best',final_prediction=fresh,resources=meter.counts(),replay_policy=contract()['replay'])


def choice(task,result,slot):
    require(result['task_id']==core.task_id(task) and result['source_slot']==slot and result['arm']==ARM and result['seed']==seed(slot) and
        result['selection']=='last_updated_nominal_not_best' and result['replay_policy']==contract()['replay'],'own nominal task/dispatch contract')
    base.exact_counts(result['resources'],budget());bank=result['evaluations'];require(len(bank)==35,'ALL35 kernel evaluations')
    offset=0;batch_records=[];updates=[]
    def saved_cost(batch):
        nonlocal offset
        b=len(batch_records);require(b<5,'five complete callbacks');stage,it,n=schedule()[b]
        require(batch.shape==(n,2,10),'reference batch schedule')
        if stage=='updated_nominal':updates.append(update_record(it,bank[offset-16:offset],batch[0],task['packet_logit_cap']))
        batch_records.append(dict(batch=b,stage=stage,iteration=it,parameters=batch.tolist()))
        values=[]
        for j,params in enumerate(batch):
            c=bank[offset+j];logits=gn.screen.projected_logits(params,task['packet_logit_cap'])
            terminal=core.array(c['terminal'],(192,))
            require(c==dict(index=offset+j,stage=stage,iteration=it,parameters=params.tolist(),logits=logits.tolist(),
                commands=np.tanh(logits.astype(np.float32)).reshape(10,2).astype(float).tolist(),terminal=terminal.tolist(),cost=core.model_cost(task,terminal)),'all parameters/RNG/projection/rawcommands/original-Q costs')
            values.append(c['cost'])
        offset+=n;return np.asarray(values)
    best,receipt=run_kernel(saved_cost,slot,task['packet_logit_cap'])
    best_i=min(range(35),key=lambda i:(bank[i]['cost'],i))
    require(offset==35 and batch_records==result['batches'] and updates==result['updates'] and receipt==result['kernel_receipt'] and
        best.tolist()==result['kernel_best_parameters']==bank[best_i]['parameters'] and result['diagnostic_best_index']==best_i,'exact entire reference kernel receipt/update replay, no model calls')
    p=result['final_prediction'];check_single(task,p);c=bank[34]
    require(result['selected_index']==34 and c['stage']=='updated_nominal' and c['iteration']==1 and
        sampling.compact(p)=={k:c[k] for k in sampling.compact(p)},'dispatch FINAL NOMINAL even if best-ever has lower cost')
    return dict(task_id=core.task_id(task),logits=p['logits'],commands=p['commands'],prediction=p['contexts'][-1],model_terminal_cost=p['cost'],
        selected_index=34,selected_parameters=c['parameters'],selection='last_updated_nominal_not_best',replay_policy=result['replay_policy'])


def journal_check(task,result,slot,events):
    pos=0;op=0
    def take(kind):
        nonlocal pos
        require(pos<len(events) and events[pos]['kind']==kind,f'causal event{pos}: expected {kind}')
        e=events[pos];pos+=1;return e
    require(take('arm_started')==dict(kind='arm_started',task_sha256=slot['task_sha256'],seed=seed(slot['source_slot']),temperature=TEMPERATURE),'fixed lifecycle')
    def evaluate(c,gid):
        nonlocal op
        require(take('evaluation_request')==dict(kind='evaluation_request',evaluation_id=gid,mode='whole_replay',logits=c['logits'],commands=c['commands']),'full H2 request/action binding')
        path=[]
        for k in (0,1):
            for resource in ('action_encoder','predict'):
                entry=dict(index=op,phase='direct',evaluation_id=gid,resource=resource,samples=1,completed=False)
                require(take('operation_attempted')==dict(kind='operation_attempted',evaluation_id=gid,event=entry),'actual predispatch entry')
                require(take('operation_completed')==dict(kind='operation_completed',evaluation_id=gid,event=dict(entry,completed=True)),'same actual return')
                op+=1
            e=take('native_packet');require(set(e)=={'kind','evaluation_id','mode','packet','context'} and e['evaluation_id']==gid and e['mode']=='whole_replay' and e['packet']==k,'both packet contexts in scope')
            path.append(e['context'])
        gn.saved.check_path(task,np.asarray(c['commands']),path)
        p=dict(logits=c['logits'],commands=c['commands'],terminal=path[-1]['latent_history'][-1],cost=core.model_cost(task,path[-1]['latent_history'][-1]))
        require(p=={k:c[k] for k in p} and take('evaluation_complete')==dict(kind='evaluation_complete',evaluation_id=gid,mode='whole_replay',prediction=p),'saved histories/terminal/own-Q arithmetic')
        return path
    offset=0
    for b,(stage,it,n) in enumerate(schedule()):
        prev=None if offset==0 else f'evaluation_{offset-1}'
        if stage=='updated_nominal':require(take('kernel_update')==dict(kind='kernel_update',evaluation_id=prev,**result['updates'][it]),'actual weighted update BEFORE nominal score')
        require(take('kernel_batch_frozen')==dict(kind='kernel_batch_frozen',evaluation_id=prev,**result['batches'][b]),'callback parameter bank frozen BEFORE model work')
        for c in result['evaluations'][offset:offset+n]:
            gid=f'evaluation_{c["index"]}';evaluate(c,gid)
            require(take('kernel_evaluation_saved')==dict(kind='kernel_evaluation_saved',evaluation_id=gid,evaluation=c),'every scored initial/proposal/nominal saved')
        offset+=n
    require(take('kernel_completed')==dict(kind='kernel_completed',evaluation_id='evaluation_34',receipt=result['kernel_receipt'],best_parameters=result['kernel_best_parameters'],best_index=result['diagnostic_best_index']),'complete unchanged kernel receipt')
    c=result['evaluations'][34]
    require(take('nominal_dispatch_frozen')==dict(kind='nominal_dispatch_frozen',evaluation_id='evaluation_34',index=34,parameters=c['parameters'],commands=c['commands'],cost=c['cost']),'nominal, not best-sample seal BEFORE replay')
    path=evaluate(c,'final_nominal_independent_H2')
    require(path==result['final_prediction']['contexts'] and op==144,'36 actual H2 evaluations/72 transitions')
    require(pos==len(events)-1 and events[pos]['kind']=='model_selection_sealed','no hidden model work or extra replays')


def execute(task,slot,model,device,torch,sink,row):
    meter=Meter(sink,lambda:torch.cuda.synchronize(device))
    try:
        backend=gn.screen.NativeCuda(model,device,task)
        with torch.inference_mode(),backend.guard(meter):return optimize(task,slot,backend,meter)
    finally:row['resources']=meter.counts()


def plan_all(p,store,runner,rows):
    for slot,row in zip(p['slots'],rows,strict=True):
        s=slot['source_slot'];task=base.load_export(slot['task_export'],slot['task_sha256'])['gn_task'];row['status']='attempted'
        def sink(e):store.event(dict(source_slot=s,arm=ARM,task_id=slot['task_id'],event=e))
        sink(dict(kind='arm_started',task_sha256=slot['task_sha256'],seed=seed(s),temperature=TEMPERATURE))
        result=runner(task,s,sink,row);selected=choice(task,result,s);base.exact_counts(row['resources'],budget())
        stem=f'root_{s}_{ARM}';store.put(stem+'.result.json.gz',result)
        plan=dict(schema=NAME+'.model_plan',source_slot=s,arm=ARM,blind_sha256=slot['blind_sha256'],task_export=slot['task_export'],task_sha256=slot['task_sha256'],
            task_contract_sha256=task['task_contract_sha256'],with_target=False,physical_execution=False,selection_uses_physics=False,P1_admission=False,official_MPPI=False,**selected)
        plan['selection_sha256']=core.digest(plan);store.put(stem+'.plan.json',plan)
        sink(dict(kind='model_selection_sealed',selection_sha256=plan['selection_sha256'],commands=plan['commands']))
        row.update(status='model_plan_sealed',commands=plan['commands'],result_file=stem+'.result.json.gz',plan_file=stem+'.plan.json',selection_sha256=plan['selection_sha256'])


def run(prepared,expected,output):
    p=verify(prepared,expected);store=base.Store(output);errors=[];model=None;before=after=host=None;source_ok=False
    rows=[dict(source_slot=s['source_slot'],arm=ARM,blind_sha256=s['blind_sha256'],task_id=s['task_id'],status='not_started',commands=None,resources={}) for s in p['slots']]
    try:
        print('Project-kernel MPPI math, FINAL NOMINAL dispatch;35+1 H2/root,72predict, NOT official MPPI or equal compute.',flush=True)
        store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        enc.frozen(model);before=enc.tensor_hash(model);require(before==p['expected_model_hash'],'original frozen model tensor hash')
        with sdpa_kernel(SDPBackend.MATH):plan_all(p,store,lambda t,s,k,r:execute(t,s,model,device,torch,k,r),rows)
    except BaseException as exc:
        errors.append(enc.error('execution',exc))
        try:store.event(dict(kind='execution_error',error=errors[0]))
        except BaseException as e:errors.append(enc.error('error_journal',e))
    finally:
        for row in rows:
            if row['status']=='attempted':row['status']='failed_partial'
        if model is not None:
            try:enc.frozen(model);after=enc.tensor_hash(model)
            except BaseException as e:errors.append(enc.error('final_model',e))
        try:verify(prepared,expected);source_ok=True
        except BaseException as e:errors.append(enc.error('final_source',e))
        try:store.close()
        except BaseException as e:errors.append(enc.error('journal_finalize',e))
        complete=not errors and source_ok and before is not None and before==after and all(r['status']=='model_plan_sealed' for r in rows)
        hashes={}
        try:files=list(store.output.iterdir())
        except BaseException as e:errors.append(enc.error('artifact_listing',e));complete=False;files=[]
        for f in files:
            try:
                if f.is_file():hashes[f.name]=sha(f)
            except BaseException as e:errors.append(enc.error('artifact_hash',e));complete=False
        resources={k:dict(attempted=sum(r['resources'].get(k,{}).get('attempted',0) for r in rows),completed=sum(r['resources'].get(k,{}).get('completed',0) for r in rows)) for k in budget()}
        report=dict(schema=NAME+'.report',status='kernel_nominal_plans_sealed_not_admitted' if complete else 'partial_or_failed',prepared=str(Path(prepared).absolute()),prepare_sha256=expected,
            rows=rows,resources=resources,errors=errors,first_error=errors[0] if errors else None,artifact_sha256=hashes,source_verified=source_ok,
            model_hash_before=before,model_hash_after=after,model_unchanged=before==after if before else None,host=host,contract=p['contract'],commands_released=complete,
            new_images=0,new_physics=0,training=False,official_MPPI=False,P1_admission=False,equal_D2P_compute_claim=False,paper_strong_MPPI_requirement_closed=False)
        try:store.put('report.json',report,final=True)
        except BaseException as e:complete=False;errors.append(enc.error('final_report',e));print(canonical(dict(errors=errors)).decode(),flush=True)
    require(complete,'partial saved, no retries: '+str(errors));return report


def consume(output,expected):
    output=Path(output).absolute();require(sha(output/'report.json')==expected,'external actual report SHA');r=read(output/'report.json')
    require(r['schema']==NAME+'.report' and r['status']=='kernel_nominal_plans_sealed_not_admitted' and r['commands_released'] is True and
        r['errors']==[] and r['first_error'] is None and r['source_verified'] is True and r['model_unchanged'] is True and r['model_hash_before']==r['model_hash_after']==enc.MODEL_TENSOR_SHA,'complete actual frozen-model closure')
    p=verify(r['prepared'],r['prepare_sha256']);require(r['contract']==p['contract'] and len(r['rows'])==2,'two fixed nominal plans')
    names={'binding.json','events.jsonl.gz'}|{f'root_{s}_{ARM}.{suffix}' for s in (0,1) for suffix in ('result.json.gz','plan.json')}
    require(set(r['artifact_sha256'])==names and {f.name for f in output.iterdir()}==names|{'report.json'},'exact full artifacts')
    for n,h in r['artifact_sha256'].items():require(sha(output/n)==h,'artifact SHA')
    require(read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']),'prepare binding file')
    raw,eof=base.inflate((output/'events.jsonl.gz').read_bytes());require(eof and raw.endswith(b'\n'),'complete bounded journal')
    groups=[]
    for line in raw.splitlines():
        e=base.decode(line);require(set(e)=={'source_slot','arm','task_id','event'} and e['arm']==ARM,'only own nonerror scoped events')
        if not groups or groups[-1][0]!=e['source_slot']:groups.append((e['source_slot'],[]))
        groups[-1][1].append(e)
    require([s for s,_ in groups]==[0,1],'source order, no resumed arms');plans=[]
    for slot,row,(s,group) in zip(p['slots'],r['rows'],groups,strict=True):
        stem=f'root_{s}_{ARM}'
        require(slot['source_slot']==row['source_slot']==s and row['arm']==ARM and row['blind_sha256']==slot['blind_sha256'] and row['task_id']==slot['task_id'] and
            row['status']=='model_plan_sealed' and row['result_file']==stem+'.result.json.gz' and row['plan_file']==stem+'.plan.json' and all(e['task_id']==slot['task_id'] for e in group),'stable blind/task/slot joins')
        task=base.load_export(slot['task_export'],slot['task_sha256'])['gn_task'];result=read(output/row['result_file']);selected=choice(task,result,s)
        plan=read(output/row['plan_file']);ident=plan.pop('selection_sha256')
        want=dict(schema=NAME+'.model_plan',source_slot=s,arm=ARM,blind_sha256=slot['blind_sha256'],task_export=slot['task_export'],task_sha256=slot['task_sha256'],
            task_contract_sha256=task['task_contract_sha256'],with_target=False,physical_execution=False,selection_uses_physics=False,P1_admission=False,official_MPPI=False,**selected)
        require(plan==want and core.digest(plan)==ident==row['selection_sha256'] and row['commands']==selected['commands'],'FINAL NOMINAL commands and seal')
        base.exact_counts(row['resources'],budget());events=[e['event'] for e in group];journal_check(task,result,slot,events)
        require(events[-1]==dict(kind='model_selection_sealed',selection_sha256=ident,commands=selected['commands']),'final nominal plan durable')
        plans.append(dict(plan,selection_sha256=ident,whole_run_report_sha256=expected))
    base.exact_counts(r['resources'],p['contract']['total_budget'])
    require(sha(output/'report.json')==expected,'final report TOCTOU')
    for path,h in p['bindings_sha256'].items():require(sha(path)==h,'final source/input TOCTOU')
    for n,h in r['artifact_sha256'].items():require(sha(output/n)==h,'final artifact TOCTOU')
    return dict(schema=NAME+'.consume',rows=plans,official_MPPI=False,paper_strong_MPPI_requirement_closed=False,P1_admission=False,new_physics=0)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','verify','run','consume'))
    for key in ('prepared','prepare-sha','output','report-sha'):p.add_argument('--'+key)
    a=p.parse_args()
    if a.command=='prepare':result=prepare(a.output)
    elif a.command=='verify':result=dict(status='verified_no_execution',roots=len(verify(a.prepared,a.prepare_sha)['slots']))
    elif a.command=='consume':result=consume(a.output,a.report_sha)
    else:result=run(a.prepared,a.prepare_sha,a.output)
    print(canonical(result).decode().strip())


if __name__=='__main__':main()
