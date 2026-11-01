"""Blind projected-logit MPPI-inspired shooting, MAIN CUDA only, not official MPPI."""
from __future__ import annotations
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import p1_constructive_sampling_cuda_20261027_v1 as sampling
from experiments.spectral_reachability import d2p_sampling_baselines as kernels

NAME='p1_constructive_mppi_cuda_20261027_v1'
ARM='mppi_inspired_shooting'
ROOT=sampling.ROOT
base,gn,enc,core=sampling.base,sampling.gn,sampling.enc,sampling.core
require,sha,read,canonical=base.require,base.sha,base.read,base.canonical
SAMPLING_SHA='5fe2af6256071d4a056a9f40908ee616c284e7b72e61a5c1fc5a79e359b86862'
TEMPERATURE=1.0
NOISE_SCALE=.5
ITERATIONS,SAMPLES=2,16


def budget():return dict(predict=66,action_encoder=66)


def seed(slot):
    require(type(slot) is int and slot in (0,1),'two fixed source slots')
    return 20261027+101*slot+2


def contract():
    return dict(arm=ARM,roots=2,iterations=2,samples=16,candidates=32,temperature=1.0,noise_scale=.5,
        seeds=[seed(0),seed(1)],rng='NumPy default_rng PCG64, normal(mean,.5)',initial_nominal='zero2x10_unscored',
        proposals='float64 Gaussian then sealed float32 packet logit ball projection .75sqrt5',
        update='stable exp(-(cost-mincost)/1) normalized weights; weighted PROJECTED logits, projected again; nominal unscored',
        selection='stable global argmin(cost,index) over32 evaluated proposals, not unscored weighted nominal',
        replay='ONE independent fullH2 whole_replay; no triple/segmented replay claim',
        commands='float32 NumPy tanh logits then reshape10x2, no extra clip',
        per_root_budget=budget(),total_budget={k:2*v for k,v in budget().items()},candidate_predict=64,winner_predict=2,
        AD_API=0,mathematical_VJP=0,jacobian=0,svd=0,solve=0,image_encoder=0,image_projector=0,new_physics=0,
        metric='same approved identity192 original Q/goal/normalizer/history/checkpoint',training=False,new_Q=False,
        official_MPPI=False,importance_correction=False,automatic_control_cross_term=False,
        algorithm_boundary='budget-constrained projected-logit MPPI-inspired shooting, not exact existing mppi() nor paper reproduction',
        equal_D2P_compute_claim=False,equal_sampling64_compute_claim=False,P1_admission=False,new_independent_confirmation=False,
        inherited_encoding='historically paid and separate, not free total pipeline',
        numerical=dict(SDP='MATH',matmul_TF32=False,cudnn_TF32=False),
        input_access='only exact original approved blind task exports, no witness/sourcepack/outcomes',
        output_cap_bytes=base.CAP,reserve_bytes=base.FLOOR)


def build():
    require(sha(sampling.__file__)==SAMPLING_SHA,'sealed sampling interface source')
    p=sampling.build()  # Pure original blind export/source/model hashes, not sampling results.
    bindings=deepcopy(p['bindings_sha256'])
    for f in (Path(__file__),Path(kernels.__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):
        bindings[str(f.resolve())]=sha(f)
    for path,h in bindings.items():require(sha(path)==h,'prepare source/input TOCTOU')
    return dict(schema=NAME+'.prepare',slots=p['slots'],policy=p['policy'],expected_model_hash=p['expected_model_hash'],
        contract=contract(),bindings_sha256=bindings,global_freshness='existing_two_calibration_tasks_only')


def prepare(output):
    p=build();raw=canonical(p);path=Path(output).absolute()
    require(len(raw)<256*1024 and not path.exists() and shutil.disk_usage(path.parent).free>=base.FLOOR+len(raw),'NEW small prepare +200MiB reserve')
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return dict(prepared=str(path),prepare_sha256=sha(path),bytes=len(raw),contract=p['contract'])


def verify(path,expected):
    require(sha(path)==expected,'external prepare SHA');p=read(path)
    require(p==build(),'complete own source/input/protocol reconstruction');return p


def weight_update(bank,cap):
    require(len(bank)==16,'exact16 proposals per iteration')
    costs=np.asarray([c['cost'] for c in bank],float)
    require(costs.shape==(16,) and np.isfinite(costs).all() and (costs>=0).all(),'finite nonnegative fullH2 costs')
    weights=kernels.softmax_weights(costs,temperature=TEMPERATURE)
    require(np.isfinite(weights).all() and (weights>=0).all() and np.isclose(weights.sum(),1.,rtol=0,atol=1e-15),'stable nonnegative unit weights')
    nominal=gn.screen.projected_logits(np.tensordot(weights,np.asarray([c['logits'] for c in bank]),axes=(0,0)),cap)
    return weights,nominal


class Meter(sampling.Meter):
    def __init__(self,sink,sync=lambda:None):
        gn.screen.Meter.__init__(self,budget(),sink);self.sync=sync


def check_single(task,p):
    """No forged triple-replay fields: this budget supports ONE fresh H2 only."""
    require(set(p)=={'task_id','logits','commands','contexts','cost'} and p['task_id']==core.task_id(task),'single-H2 schema/task')
    x=core.array(p['logits'],(2,10));raw=core.array(p['commands'],(10,2))
    require(np.array_equal(x,x.astype(np.float32)) and (np.linalg.norm(x,axis=1)<=task['packet_logit_cap']).all(),'feasible float32 logits')
    require(np.array_equal(raw,np.tanh(x.astype(np.float32)).reshape(10,2)),'exact rawcommands from logits')
    gn.saved.check_path(task,raw,p['contexts'])
    require(p['cost']==core.model_cost(task,p['contexts'][-1]['latent_history'][-1]),'same-Q exact fullH2 cost')


def optimize(task,slot,backend,meter):
    core.validate_task(task);rng=np.random.default_rng(seed(slot));nominal=np.zeros((2,10));bank=[];updates=[]
    cap=task['packet_logit_cap']
    for it in range(ITERATIONS):
        xs=[gn.screen.projected_logits(x,cap) for x in rng.normal(nominal,NOISE_SCALE,size=(SAMPLES,2,10))]
        meter.emit('MPPI_bank_frozen',iteration=it,nominal=nominal.tolist(),noise_scale=NOISE_SCALE,temperature=TEMPERATURE,logits=[x.tolist() for x in xs])
        chunk=[]
        for x in xs:
            i=len(bank);meter.group=f'candidate_{i}'
            p=gn.single(task,x,backend,meter,'whole_replay');check_single(task,p)
            c=dict(index=i,iteration=it,**sampling.compact(p));bank.append(c);chunk.append(c)
            meter.emit('candidate_saved',candidate=c)
        weights,nominal=weight_update(chunk,cap)
        update=dict(iteration=it,weights=weights.tolist(),nominal=nominal.tolist(),temperature=TEMPERATURE)
        updates.append(update);meter.emit('MPPI_update',**update)
    require(len(bank)==32,'complete32 proposal costs')
    index=min(range(32),key=lambda k:(bank[k]['cost'],k));chosen=bank[index]
    meter.emit('winner_frozen_before_replay',index=index,cost=chosen['cost'],commands=chosen['commands'])
    meter.group='winner_independent_H2';fresh=gn.single(task,chosen['logits'],backend,meter,'whole_replay');check_single(task,fresh)
    require(sampling.compact(fresh)=={k:chosen[k] for k in sampling.compact(fresh)},'MATH search/independent replay exact')
    base.exact_counts(meter.counts(),budget())
    return dict(task_id=core.task_id(task),source_slot=slot,arm=ARM,seed=seed(slot),candidates=bank,updates=updates,
        winner_index=index,final_prediction=fresh,replay_policy=contract()['replay'],resources=meter.counts())


def choice(task,result,slot):
    require(result['task_id']==core.task_id(task) and result['source_slot']==slot and result['arm']==ARM and result['seed']==seed(slot) and
        result['replay_policy']==contract()['replay'],'task/arm/seed/single replay contract')
    base.exact_counts(result['resources'],budget());bank=result['candidates'];require(len(bank)==32,'ALL32')
    rng=np.random.default_rng(seed(slot));nominal=np.zeros((2,10));updates=[]
    for it in range(2):
        xs=[gn.screen.projected_logits(x,task['packet_logit_cap']) for x in rng.normal(nominal,NOISE_SCALE,size=(16,2,10))]
        chunk=bank[it*16:(it+1)*16]
        for j,(c,x) in enumerate(zip(chunk,xs,strict=True)):
            terminal=core.array(c['terminal'],(192,))
            require(c['index']==it*16+j and c['iteration']==it and c['logits']==x.tolist() and
                c['commands']==np.tanh(x.astype(np.float32)).reshape(10,2).astype(float).tolist() and
                c['cost']==core.model_cost(task,terminal),'deterministic all32 proposals/action/own-Q costs')
        weights,nominal=weight_update(chunk,task['packet_logit_cap'])
        updates.append(dict(iteration=it,weights=weights.tolist(),nominal=nominal.tolist(),temperature=TEMPERATURE))
    require(result['updates']==updates,'stable weights and both causal nominal updates')
    index=min(range(32),key=lambda k:(bank[k]['cost'],k));p=result['final_prediction'];check_single(task,p)
    require(result['winner_index']==index and sampling.compact(p)=={k:bank[index][k] for k in sampling.compact(p)},'global stable evaluated winner, never unscored nominal')
    return dict(task_id=core.task_id(task),logits=p['logits'],commands=p['commands'],prediction=p['contexts'][-1],model_terminal_cost=p['cost'],replay_policy=result['replay_policy'])


def journal_check(task,result,slot,group):
    """Exact saved grammar binds every history array/actual call and predispatch seal."""
    pos=0;op=0
    def take(kind):
        nonlocal pos
        require(pos<len(group) and group[pos]['kind']==kind,f'causal event {pos} expected {kind}')
        e=group[pos];pos+=1;return e
    require(take('arm_started')==dict(kind='arm_started',task_sha256=slot['task_sha256'],seed=seed(slot['source_slot']),temperature=TEMPERATURE),'fixed lifecycle')
    def evaluation(c,gid):
        nonlocal op
        require(take('evaluation_request')==dict(kind='evaluation_request',evaluation_id=gid,mode='whole_replay',logits=c['logits'],commands=c['commands']),'saved actual evaluation request')
        path=[]
        for k in (0,1):
            for resource in ('action_encoder','predict'):
                e=dict(index=op,phase='direct',evaluation_id=gid,resource=resource,samples=1,completed=False)
                require(take('operation_attempted')==dict(kind='operation_attempted',evaluation_id=gid,event=e),'unique predispatch actual attempt')
                require(take('operation_completed')==dict(kind='operation_completed',evaluation_id=gid,event=dict(e,completed=True)),'unique native return')
                op+=1
            e=take('native_packet')
            require(set(e)=={'kind','evaluation_id','mode','packet','context'} and e['evaluation_id']==gid and e['mode']=='whole_replay' and e['packet']==k,'actual H2 packet scope')
            path.append(e['context'])
        gn.saved.check_path(task,np.asarray(c['commands']),path)
        p=dict(logits=c['logits'],commands=c['commands'],cost=core.model_cost(task,path[-1]['latent_history'][-1]),terminal=path[-1]['latent_history'][-1])
        require(p=={k:c[k] for k in p} and take('evaluation_complete')==dict(kind='evaluation_complete',evaluation_id=gid,mode='whole_replay',prediction=p),'real packet histories/terminal/ownQ bind saved cost')
        return path
    nominal=np.zeros((2,10));bank=result['candidates']
    for it in (0,1):
        chunk=bank[16*it:16*(it+1)];previous=None if it==0 else 'candidate_15'
        require(take('MPPI_bank_frozen')==dict(kind='MPPI_bank_frozen',evaluation_id=previous,iteration=it,nominal=nominal.tolist(),noise_scale=NOISE_SCALE,temperature=TEMPERATURE,logits=[c['logits'] for c in chunk]),'bank frozen BEFORE costs')
        for c in chunk:
            gid=f'candidate_{c["index"]}';evaluation(c,gid)
            require(take('candidate_saved')==dict(kind='candidate_saved',evaluation_id=gid,candidate=c),'candidate immediately saved')
        update=result['updates'][it];nominal=np.asarray(update['nominal'])
        require(take('MPPI_update')==dict(kind='MPPI_update',evaluation_id=f'candidate_{16*(it+1)-1}',**update),'update after all16 costs, before next bank')
    w=bank[result['winner_index']]
    require(take('winner_frozen_before_replay')==dict(kind='winner_frozen_before_replay',evaluation_id='candidate_31',index=w['index'],cost=w['cost'],commands=w['commands']),'winner after32 before independent H2')
    path=evaluation(w,'winner_independent_H2');require(path==result['final_prediction']['contexts'] and op==132,'one independent H2,66 actual transitions')
    # Caller checks the final model_selection_sealed against the plan seal.
    require(pos==len(group)-1 and group[pos]['kind']=='model_selection_sealed','no hidden work or extra replays')


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
        plan=dict(schema=NAME+'.model_plan',source_slot=s,arm=ARM,blind_sha256=slot['blind_sha256'],
            task_export=slot['task_export'],task_sha256=slot['task_sha256'],task_contract_sha256=task['task_contract_sha256'],
            with_target=False,physical_execution=False,selection_uses_physics=False,P1_admission=False,official_MPPI=False,**selected)
        plan['selection_sha256']=core.digest(plan);store.put(stem+'.plan.json',plan)
        sink(dict(kind='model_selection_sealed',selection_sha256=plan['selection_sha256'],commands=plan['commands']))
        row.update(status='model_plan_sealed',commands=plan['commands'],result_file=stem+'.result.json.gz',plan_file=stem+'.plan.json',selection_sha256=plan['selection_sha256'])


def run(prepared,expected,output):
    p=verify(prepared,expected);store=base.Store(output);errors=[];model=None;before=after=host=None;source_ok=False
    rows=[dict(source_slot=s['source_slot'],arm=ARM,blind_sha256=s['blind_sha256'],task_id=s['task_id'],status='not_started',commands=None,resources={}) for s in p['slots']]
    try:
        print('Fixed blind MPPI-inspired32 proposals+oneH2 replay;66 predict/root, not official MPPI or equal D2P compute.',flush=True)
        store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        enc.frozen(model);before=enc.tensor_hash(model);require(before==p['expected_model_hash'],'original frozen model tensor SHA')
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
        report=dict(schema=NAME+'.report',status='mppi_model_plans_sealed_not_admitted' if complete else 'partial_or_failed',prepared=str(Path(prepared).absolute()),prepare_sha256=expected,
            rows=rows,resources=resources,errors=errors,first_error=errors[0] if errors else None,artifact_sha256=hashes,source_verified=source_ok,
            model_hash_before=before,model_hash_after=after,model_unchanged=before==after if before else None,host=host,contract=p['contract'],commands_released=complete,
            new_images=0,new_physics=0,training=False,official_MPPI=False,P1_admission=False,equal_D2P_compute_claim=False)
        try:store.put('report.json',report,final=True)
        except BaseException as e:complete=False;errors.append(enc.error('final_report',e));print(canonical(dict(errors=errors)).decode(),flush=True)
    require(complete,'partial preserved, no retries: '+str(errors));return report


def consume(output,expected):
    output=Path(output).absolute();require(sha(output/'report.json')==expected,'external complete report SHA');r=read(output/'report.json')
    require(r['schema']==NAME+'.report' and r['status']=='mppi_model_plans_sealed_not_admitted' and r['commands_released'] is True and
        r['errors']==[] and r['first_error'] is None and r['source_verified'] is True and r['model_unchanged'] is True and
        r['model_hash_before']==r['model_hash_after']==enc.MODEL_TENSOR_SHA,'complete frozen-model closure')
    p=verify(r['prepared'],r['prepare_sha256']);require(r['contract']==p['contract'] and len(r['rows'])==2,'two fixed MPPI rows')
    names={'binding.json','events.jsonl.gz'}|{f'root_{s}_{ARM}.{suffix}' for s in (0,1) for suffix in ('result.json.gz','plan.json')}
    require(set(r['artifact_sha256'])==names and {f.name for f in output.iterdir()}==names|{'report.json'},'complete exact output census')
    for n,h in r['artifact_sha256'].items():require(sha(output/n)==h,'artifact hash')
    require(read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']),'prepare binding artifact')
    raw,eof=base.inflate((output/'events.jsonl.gz').read_bytes());require(eof and raw.endswith(b'\n'),'complete bounded journal')
    wrapped=[base.decode(x) for x in raw.splitlines()];groups=[]
    for e in wrapped:
        require(set(e)=={'source_slot','arm','task_id','event'} and e['arm']==ARM,'only own scoped events, no errors')
        if not groups or groups[-1][0]!=e['source_slot']:groups.append((e['source_slot'],[]))
        groups[-1][1].append(e)
    require([s for s,_ in groups]==[0,1],'fixed nonresumed source order');plans=[]
    for slot,row,(s,group) in zip(p['slots'],r['rows'],groups,strict=True):
        stem=f'root_{s}_{ARM}';require(slot['source_slot']==row['source_slot']==s and row['arm']==ARM and
            row['blind_sha256']==slot['blind_sha256'] and row['task_id']==slot['task_id'] and row['status']=='model_plan_sealed' and
            row['result_file']==stem+'.result.json.gz' and row['plan_file']==stem+'.plan.json' and all(e['task_id']==slot['task_id'] for e in group),'direct stable source/blind/task identity')
        task=base.load_export(slot['task_export'],slot['task_sha256'])['gn_task'];result=read(output/row['result_file']);selected=choice(task,result,s)
        plan=read(output/row['plan_file']);ident=plan.pop('selection_sha256')
        want=dict(schema=NAME+'.model_plan',source_slot=s,arm=ARM,blind_sha256=slot['blind_sha256'],task_export=slot['task_export'],task_sha256=slot['task_sha256'],
            task_contract_sha256=task['task_contract_sha256'],with_target=False,physical_execution=False,selection_uses_physics=False,P1_admission=False,official_MPPI=False,**selected)
        require(plan==want and core.digest(plan)==ident==row['selection_sha256'] and row['commands']==selected['commands'],'actual selected command seal')
        base.exact_counts(row['resources'],budget());events=[e['event'] for e in group];journal_check(task,result,slot,events)
        require(events[-1]==dict(kind='model_selection_sealed',selection_sha256=ident,commands=selected['commands']),'final durable plan seal')
        plans.append(dict(plan,selection_sha256=ident,whole_run_report_sha256=expected))
    base.exact_counts(r['resources'],p['contract']['total_budget'])
    require(sha(output/'report.json')==expected,'report final TOCTOU')
    for path,h in p['bindings_sha256'].items():require(sha(path)==h,'source/input final TOCTOU')
    for n,h in r['artifact_sha256'].items():require(sha(output/n)==h,'artifact final TOCTOU')
    return dict(schema=NAME+'.consume',rows=plans,official_MPPI=False,P1_admission=False,new_physics=0)


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
