"""Explicit mean-ridge GN variant, MAIN-only HOST CUDA. No sealed globals patched.

Matching the D2P ridge coefficient does not match its point, dimension, objective,
candidate mechanism, budget or physical efficacy. Calibration only.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import p1_constructive_blind_planners_cuda_20261027_v1 as base

NAME='p1_constructive_matched_ridge_gn_cuda_20261027_v1'
ARM='matched_ridge_GN'
ROOT=base.ROOT
gn,enc=base.gn,base.encode
core=enc.core
require,sha,read,canonical=base.require,base.sha,base.read,base.canonical
LAMBDA_UNNORMALIZED=.002
LAMBDA_MEAN=LAMBDA_UNNORMALIZED/192
SCALES=(1.,.5,.25,.125,.0625)
UPDATES=4
BASE_SHA='e56bee610f4eb412cc03fbe49b3df91054d0defe23d24b6fd9c77e6a7f1e9f27'
GN_SHA='825c148cf515ad42d59b187d0d8c83984826db67f58d66b2e2d4f289dcecf70a'
EXPORTS=[str(ROOT/f'results/p1_constructive_blind_encode_cuda_20261027_v1_cuda_main01/task_{i:02d}.json.gz') for i in range(2)]
EXPORT_SHAS=['714f471761278fa654542d4bb0878ce18e3c304df6158005af46d2129533265a',
             '70865d56f9b66f0072b2ad7de01cc09d9a9275047925e188104a13d5182ce1f2']


def algebra(a,r,lambda_mean):
    """min mean(||A delta+r||²) + lambda_mean ||delta||² before projection.

    The unnormalized coefficient is 192*lambda_mean, not lambda_mean itself.
    SVD is metered by optimize, not hidden inside this arithmetic helper.
    """
    a=core.array(a,(192,20));r=core.array(r,(192,))
    require(type(lambda_mean) is float and np.isfinite(lambda_mean) and lambda_mean>0,'explicit positive mean ridge')
    left,s,right=np.linalg.svd(a/np.sqrt(192),full_matrices=False)
    return -right.T@(s*(left.T@(r/np.sqrt(192)))/(s*s+lambda_mean))


def optimize(task,backend,meter,*,lambda_mean,updates,scales):
    """Fresh GN, not old gn.optimize and not mutation of gn.DAMPING/SCALES."""
    core.validate_task(task)
    require(lambda_mean==LAMBDA_MEAN and updates==4 and tuple(scales)==SCALES,'frozen explicit variant parameters')
    require(tuple(gn.SCALES)==tuple(scales),'sealed reusable line_trials must use the same fixed scales')
    meter.group='initial'; incumbent=gn.verified(task,np.zeros((2,10)),backend,meter)
    meter.emit('initial_verified',incumbent=incumbent,lambda_mean=lambda_mean,lambda_unnormalized=192*lambda_mean)
    steps=[]
    for iteration in range(1,updates+1):
        meter.group=f'iter_{iteration}_AD'
        terminal,j=meter.call('jacobian',lambda:backend.jacobian(task,incumbent,meter))
        terminal,j=core.array(terminal,(192,)),core.array(j,(192,20))
        require(np.array_equal(terminal,incumbent['contexts'][-1]['latent_history'][-1]),'fresh incumbent AD point')
        q=np.asarray(task['projection']);residual=q@(terminal-np.asarray(task['goal']));a=q@j
        derivative=dict(task_id=core.task_id(task),at_logits=incumbent['logits'],at_terminal=terminal.tolist(),
                        raw_terminal_jacobian=j.tolist(),objective_gradient=(2*a.T@residual/192).tolist())
        core.check_derivative(task,derivative,incumbent);meter.emit('derivative_complete',derivative=derivative)
        left,s,right=meter.call('svd',lambda:np.linalg.svd(a/np.sqrt(192),full_matrices=False))
        delta=meter.call('linear_solve',lambda:-right.T@(s*(left.T@(residual/np.sqrt(192)))/(s*s+lambda_mean)))
        endpoint=meter.call('proposal',lambda:gn.screen.projected_logits(np.asarray(incumbent['logits'])+delta.reshape(2,10),task['packet_logit_cap']))
        trials=gn.line_trials(incumbent['logits'],endpoint,task['packet_logit_cap'])
        meter.emit('line_proposals_frozen',iteration=iteration,incumbent_logits=incumbent['logits'],delta=delta.tolist(),
                   endpoint=endpoint.tolist(),scales=list(scales),logits=[x.tolist() for x in trials],
                   lambda_mean=lambda_mean,lambda_unnormalized=192*lambda_mean)
        candidates=[]
        for index,x in enumerate(trials):
            meter.group=f'iter_{iteration}_scale_{index}'
            candidates.append(gn.single(task,x,backend,meter,'whole_replay'))
        pool=[incumbent]+candidates
        index=meter.call('selection',lambda:min(range(6),key=lambda k:(pool[k]['cost'],k)))
        chosen=pool[index]
        row=dict(iteration=iteration,incumbent_cost=incumbent['cost'],trial_costs=[c['cost'] for c in candidates],
                 selected_index=index,selected_scale=None if index==0 else scales[index-1],selected_cost=chosen['cost'])
        meter.emit('selection_complete',**row);meter.group=f'iter_{iteration}_selected_replay'
        fresh=gn.verified(task,chosen['logits'],backend,meter)
        require(fresh['contexts']==chosen['contexts'] and fresh['cost']==chosen['cost'],'selected exact independent triple replay')
        incumbent=fresh;steps.append(row);meter.emit('update_verified',iteration=iteration,prediction=incumbent)
    base.exact_counts(meter.counts(),gn.budget())
    return dict(task_id=core.task_id(task),steps=steps,final_prediction=incumbent,resources=meter.counts(),
                lambda_mean=lambda_mean,lambda_unnormalized=192*lambda_mean,updates=updates,scales=list(scales))


def contract():
    return dict(arm=ARM,roots=2,lambda_mean=LAMBDA_MEAN,lambda_unnormalized=LAMBDA_UNNORMALIZED,
        equivalent_D2P_penalty='.002*||u||²/192; coefficient only, not same direction or optimization problem',
        updates=4,scales=list(SCALES),initial_logits='zero2x10',per_root_budget=gn.budget(),
        total_budget={k:2*v for k,v in gn.budget().items()},AD_API_per_root=dict(grad=768,jacobian=0),
        mathematical_vjp_per_root=768,ADforward_in_predict_per_root=8,new_images=0,new_physics=0,training=False,
        with_target=False,new_Q=False,equal_budget_claim=False,P1_admission=False,new_independent_confirmation=False,
        inherited_encoding='eight image encoder/projector calls historically paid, not zero total pipeline cost',
        selection='incumbent + ALL five fresh fullH2 costs, stable minimum; selected fresh triple replay',
        output_cap_bytes=base.CAP,reserve_bytes=base.FLOOR,numerical=dict(SDP='MATH',matmul_TF32=False,cudnn_TF32=False))


def build(task_exports,task_shas):
    require(sha(base.__file__)==BASE_SHA and sha(gn.__file__)==GN_SHA,'approved sealed planner and GN helpers')
    paths=[str(Path(p).absolute()) for p in task_exports]
    require(paths==EXPORTS and list(task_shas)==EXPORT_SHAS,'exact two approved ordered exports; no outcome inputs')
    p=base.build(paths,task_shas)  # export/model/source hashes only; no NPZ/sourcepack reader.
    slots=[]
    for i,s in enumerate(p['slots']):
        e=read(s['task_export'])
        require(e['source_slot']==i and e['blind_source_sha256']==e['blind_sha256']==e['task_contract']['blind_source_sha256']==list(enc.APPROVED_INPUTS.values())[i],'approved blind metadata, not native root')
        slots.append(dict(s,source_slot=i,blind_sha256=e['blind_sha256']))
    bindings=deepcopy(p['bindings_sha256'])
    for f in (Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')): bindings[str(f.resolve())]=sha(f)
    for path,h in bindings.items():require(sha(path)==h,'prepare source/input TOCTOU')
    return dict(schema=NAME+'.prepare',arguments=dict(task_exports=paths,task_shas=list(task_shas)),slots=slots,
                policy=p['policy'],expected_model_hash=p['expected_model_hash'],contract=contract(),bindings_sha256=bindings,
                global_freshness='unknown_calibration_only')


def prepare(output,**arguments):
    p=build(**arguments);raw=canonical(p);path=Path(output).absolute()
    require(len(raw)<256*1024 and not path.exists() and shutil.disk_usage(path.parent).free>=base.FLOOR+len(raw),'NEW small prepare +200MiB reserve')
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return dict(prepared=str(path),prepare_sha256=sha(path),bytes=len(raw),contract=p['contract'])


def verify(path,expected):
    require(sha(path)==expected,'external prepare SHA');p=read(path)
    require(p==build(**p['arguments']),'complete own variant source/input/protocol reconstruction');return p


def choice(task,result):
    require(result['task_id']==core.task_id(task) and result['lambda_mean']==LAMBDA_MEAN and
            result['lambda_unnormalized']==LAMBDA_UNNORMALIZED and result['updates']==4 and result['scales']==list(SCALES),'own variant result parameters')
    base.exact_counts(result['resources'],gn.budget());require(len(result['steps'])==4,'all four updates')
    previous=None
    for it,s in enumerate(result['steps'],1):
        pool=[s['incumbent_cost']]+s['trial_costs'];require(len(pool)==6 and np.isfinite(pool).all() and min(pool)>=0,'all five trial costs')
        k=min(range(6),key=lambda k:(pool[k],k))
        require(s['iteration']==it and s['selected_index']==k and s['selected_scale']==(None if k==0 else SCALES[k-1]) and
                s['selected_cost']==pool[k] and (previous is None or s['incumbent_cost']==previous),'stable full-scale selection chain')
        previous=s['selected_cost']
    p=result['final_prediction'];core.check_prediction(task,p);require(p['cost']==previous,'final selected cost')
    return dict(task_id=core.task_id(task),logits=p['logits'],commands=p['commands'],prediction=p['contexts'][-1],model_terminal_cost=p['cost'])


def plan_all(p,store,runner,rows):
    for slot,row in zip(p['slots'],rows,strict=True):
        t=base.load_export(slot['task_export'],slot['task_sha256'])['gn_task'];row['status']='attempted'
        def sink(e):store.event(dict(source_slot=slot['source_slot'],arm=ARM,task_id=slot['task_id'],event=e))
        sink(dict(kind='arm_started',task_sha256=slot['task_sha256'],lambda_mean=LAMBDA_MEAN))
        result=runner(t,sink,row);selected=choice(t,result)
        base.exact_counts(row['resources'],gn.budget());base.exact_counts(row['AD_API_resources'],dict(grad=768,jacobian=0))
        stem=f'root_{slot["source_slot"]}_{ARM}';store.put(stem+'.result.json.gz',result)
        plan=dict(schema=NAME+'.model_plan',status='selected_pending_global_closure',source_slot=slot['source_slot'],arm=ARM,
            blind_sha256=slot['blind_sha256'],task_export=slot['task_export'],task_sha256=slot['task_sha256'],
            task_contract_sha256=t['task_contract_sha256'],with_target=False,physical_execution=False,selection_uses_physics=False,
            P1_admission=False,lambda_mean=LAMBDA_MEAN,lambda_unnormalized=LAMBDA_UNNORMALIZED,**selected)
        plan['selection_sha256']=core.digest(plan);store.put(stem+'.plan.json',plan)
        sink(dict(kind='model_selection_sealed',selection_sha256=plan['selection_sha256'],commands=plan['commands']))
        row.update(status='model_plan_sealed',commands=plan['commands'],result_file=stem+'.result.json.gz',plan_file=stem+'.plan.json',selection_sha256=plan['selection_sha256'])


def execute(task,model,device,torch,sink,row):
    sync=lambda:torch.cuda.synchronize(device)
    meter=gn.Meter(sink,sync);ad=base.ADTrace(torch,base.ARMS[0],sink,sync) # same GN API caps, not old optimize.
    try:
        backend=gn.screen.NativeCuda(model,device,task)
        with ad,backend.guard(meter):
            return optimize(task,backend,meter,lambda_mean=LAMBDA_MEAN,updates=4,scales=SCALES)
    finally:row['resources']=meter.counts();row['AD_API_resources']=ad.counts()


def run(prepared,expected,output):
    p=verify(prepared,expected);store=base.Store(output);errors=[];model=None;before=after=host=None;source_ok=False
    rows=[dict(source_slot=s['source_slot'],arm=ARM,blind_sha256=s['blind_sha256'],task_id=s['task_id'],
               status='not_started',commands=None,resources={},AD_API_resources={}) for s in p['slots']]
    try:
        store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        enc.frozen(model);before=enc.tensor_hash(model);require(before==p['expected_model_hash'],'original model tensor SHA')
        with sdpa_kernel(SDPBackend.MATH):plan_all(p,store,lambda t,sink,row:execute(t,model,device,torch,sink,row),rows)
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
        for f in store.output.iterdir():
            if f.is_file():
                try:hashes[f.name]=sha(f)
                except BaseException as e:errors.append(enc.error('artifact_hash',e));complete=False
        resources={k:dict(attempted=sum(r['resources'].get(k,{}).get('attempted',0) for r in rows),
                          completed=sum(r['resources'].get(k,{}).get('completed',0) for r in rows)) for k in gn.budget()}
        report=dict(schema=NAME+'.report',status='matched_ridge_model_plans_sealed_not_admitted' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).absolute()),prepare_sha256=expected,rows=rows,resources=resources,errors=errors,first_error=errors[0] if errors else None,
            artifact_sha256=hashes,source_verified=source_ok,model_hash_before=before,model_hash_after=after,
            model_unchanged=before==after if before is not None else None,host=host,contract=p['contract'],commands_released=complete,
            new_images=0,new_physics=0,training=False,new_Q=False,P1_admission=False,equal_budget_claim=False,global_freshness=p['global_freshness'])
        try:store.put('report.json',report,final=True)
        except BaseException as e:complete=False;errors.append(enc.error('final_report',e));print(canonical(dict(errors=errors)).decode(),flush=True)
    require(complete,'partial saved, no retry: '+str(errors));return report


def consume(output,report_sha):
    output=Path(output).absolute();require(sha(output/'report.json')==report_sha,'MAIN external report SHA')
    r=read(output/'report.json')
    require(r['schema']==NAME+'.report' and r['status']=='matched_ridge_model_plans_sealed_not_admitted' and r['commands_released'] is True and
            r['errors']==[] and r['first_error'] is None and r['source_verified'] is True and r['model_unchanged'] is True and
            r['model_hash_before']==r['model_hash_after']==enc.MODEL_TENSOR_SHA,'whole own variant closure, never partial')
    p=verify(r['prepared'],r['prepare_sha256']);require(r['contract']==p['contract'] and len(r['rows'])==2,'own protocol/two rows')
    expected={'binding.json','events.jsonl.gz'}|{f'root_{i}_{ARM}.{suffix}' for i in range(2) for suffix in ('result.json.gz','plan.json')}
    require(set(r['artifact_sha256'])==expected and {f.name for f in output.iterdir()}==expected|{'report.json'},'complete own artifacts')
    for name,h in r['artifact_sha256'].items():require(sha(output/name)==h,'all artifact hashes')
    plain,eof=base.inflate((output/'events.jsonl.gz').read_bytes());require(eof and plain.endswith(b'\n'),'complete single gzip journal')
    events=[base.decode(line) for line in plain.splitlines()]
    require(not any(e.get('kind')=='execution_error' for e in events),'no execution error hidden')
    require(all(set(e)=={'source_slot','arm','task_id','event'} and e['source_slot'] in (0,1) and e['arm']==ARM for e in events),'only own journal scopes')
    require([e['source_slot'] for k,e in enumerate(events) if k==0 or e['source_slot']!=events[k-1]['source_slot']]==[0,1],'fixed causal source order')
    plans=[]
    for slot,row in zip(p['slots'],r['rows'],strict=True):
        i=slot['source_slot'];stem=f'root_{i}_{ARM}'
        require(row['source_slot']==i and row['arm']==ARM and row['blind_sha256']==slot['blind_sha256'] and row['task_id']==slot['task_id'] and
                row['status']=='model_plan_sealed' and row['result_file']==stem+'.result.json.gz' and row['plan_file']==stem+'.plan.json','own row/source identity')
        base.exact_counts(row['resources'],gn.budget());base.exact_counts(row['AD_API_resources'],dict(grad=768,jacobian=0))
        task=base.load_export(slot['task_export'],slot['task_sha256'])['gn_task'];selected=choice(task,read(output/row['result_file']))
        plan=read(output/row['plan_file']);ident=plan.pop('selection_sha256')
        want=dict(schema=NAME+'.model_plan',status='selected_pending_global_closure',source_slot=i,arm=ARM,blind_sha256=slot['blind_sha256'],
            task_export=slot['task_export'],task_sha256=slot['task_sha256'],task_contract_sha256=task['task_contract_sha256'],with_target=False,
            physical_execution=False,selection_uses_physics=False,P1_admission=False,lambda_mean=LAMBDA_MEAN,lambda_unnormalized=LAMBDA_UNNORMALIZED,**selected)
        require(plan==want and core.digest(plan)==ident==row['selection_sha256'] and row['commands']==selected['commands'],'own plan/commands seal')
        group=[e['event'] for e in events if e.get('source_slot')==i and e.get('arm')==ARM and e.get('task_id')==slot['task_id']]
        require(group and group[0]==dict(kind='arm_started',task_sha256=slot['task_sha256'],lambda_mean=LAMBDA_MEAN) and
                group[-1]==dict(kind='model_selection_sealed',selection_sha256=ident,commands=plan['commands']),'durable own lifecycle')
        journal_counts(group,'operation_attempted','operation_completed','event',gn.budget())
        journal_counts(group,'AD_API_attempt','AD_API_completed','entry',dict(grad=768,jacobian=0))
        proposals=[e for e in group if e['kind']=='line_proposals_frozen']
        require(len(proposals)==4 and all(e['lambda_mean']==LAMBDA_MEAN and e['lambda_unnormalized']==LAMBDA_UNNORMALIZED and e['scales']==list(SCALES) for e in proposals),'own explicit lambda/scales in durable journal')
        plans.append(dict(plan,selection_sha256=ident,whole_run_report_sha256=report_sha))
    base.exact_counts(r['resources'],p['contract']['total_budget'])
    require(sha(output/'report.json')==report_sha,'final report TOCTOU')
    for path,h in p['bindings_sha256'].items():require(sha(path)==h,'final sources unchanged')
    for name,h in r['artifact_sha256'].items():require(sha(output/name)==h,'final artifacts unchanged')
    return dict(schema=NAME+'.consume',status='own_model_commands_verified_not_admitted',rows=plans,new_physics=0,P1_admission=False)


def journal_counts(events,attempt,returned,key,expected):
    """Release gate on actual journal attempts/returns, including nested J work."""
    entries=[];completed=[];stack=[]
    for e in events:
        require('forbidden' not in e['kind'] and 'error' not in e,'no forbidden/failed journal entry')
        if e['kind']==attempt:
            v=e[key];require(v['index']==len(entries) and v['completed'] is False,'unique ordered operation attempt')
            entries.append(v);stack.append(v['index'])
        elif e['kind']==returned:
            v=e[key];require(stack and stack.pop()==v['index'] and v==dict(entries[v['index']],completed=True),'matching nested completion')
            completed.append(v)
    require(not stack and len(entries)==len(completed)==sum(expected.values()),'complete raw ledger')
    counts={k:dict(attempted=sum(e['resource']==k for e in entries),completed=sum(e['resource']==k for e in completed)) for k in expected}
    base.exact_counts(counts,expected)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','run','consume'))
    parser.add_argument('--task-exports',nargs='+');parser.add_argument('--task-shas',nargs='+')
    for key in ('prepared','prepare-sha','output','report-sha'):parser.add_argument('--'+key)
    a=parser.parse_args()
    if a.command=='prepare':result=prepare(a.output,task_exports=a.task_exports,task_shas=a.task_shas)
    elif a.command=='verify':result=dict(status='verified_no_execution',roots=len(verify(a.prepared,a.prepare_sha)['slots']))
    elif a.command=='consume':result=consume(a.output,a.report_sha)
    else:result=run(a.prepared,a.prepare_sha,a.output)
    print(canonical(result).decode().strip())


if __name__=='__main__':main()
