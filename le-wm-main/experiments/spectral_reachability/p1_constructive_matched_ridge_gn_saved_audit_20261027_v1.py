"""Independent matched-ridge GN saved arithmetic, no optimizer/consume imports.

Only explicit latent exports, planner receipt/artifacts and source-file hashes.
No physical, sourcepack, blind NPZ, certificate, model execution or thresholds.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import p1_constructive_blind_planners_saved_audit_20261027_v1 as util

NAME='p1_constructive_matched_ridge_gn_saved_audit_20261027_v1'
RUNNER='p1_constructive_matched_ridge_gn_cuda_20261027_v1'
RUNNER_SHA='d72f6986070f2d75db4f0df711bda97a0ef636fabe566a271ff4b2ec337f0d26'
ARM='matched_ridge_GN'
LAMBDA_MEAN=.002/192
LAMBDA_RAW=.002
SCALES=(1.,.5,.25,.125,.0625)
BUDGET=dict(predict=78,action_encoder=78,jacobian=4,vjp_directions=768,svd=4,linear_solve=4,proposal=4,selection=4)
API=dict(grad=768,jacobian=0)
io=util.io
need=util.need


def external_sha(h):
    need(type(h) is str and len(h)==64 and all(c in '0123456789abcdef' for c in h),'full external lowercase SHA256 required')


def direction(j,residual):
    """Independent raw-scale normal equations, not producer's normalized SVD."""
    a=util.arr(j,(192,20));r=util.arr(residual,(192,))
    return np.linalg.solve(a.T@a+LAMBDA_RAW*np.eye(20),-a.T@r)


def audit_root(t,result,events):
    need(result['task_id']==io.task_hash(t) and result['resources']==util.counts(BUDGET),'own result task/resources')
    need(result['lambda_mean']==LAMBDA_MEAN and result['lambda_unnormalized']==LAMBDA_RAW and
         result['updates']==4 and result['scales']==list(SCALES),'explicit frozen mean/raw lambda and scales')
    need(not any('forbidden' in e['kind'] or 'error' in e for e in events),'no hidden forbidden/failure work')
    operations=util.lane(events,'operation_attempted','operation_completed','event',BUDGET)
    need(all(v['samples']==1 and v['phase']=='direct' for v in operations),'actual sample units: one per charged operation')
    api=util.lane(events,'AD_API_attempt','AD_API_completed','entry',API)
    need(all(e['is_grads_batched'] is False for e in api),'768 scalar grad API, not mathematical/API conflation')
    evaluations=[e for e in events if e['kind']=='evaluation_complete']
    requests=[e for e in events if e['kind']=='evaluation_request']
    packets=[e for e in events if e['kind']=='native_packet']
    groups=['initial']*3+[g for it in range(1,5) for g in [*[f'iter_{it}_scale_{k}' for k in range(5)],*[f'iter_{it}_selected_replay']*3]]
    modes=['nominal','whole_replay','segmented_replay']+[m for _ in range(4) for m in ['whole_replay']*5+['nominal','whole_replay','segmented_replay']]
    need(len(evaluations)==len(requests)==35 and len(packets)==70,'all35 H2 evaluations, all70 fresh packet contexts')
    stream=[e for e in events if e['kind'] in ('evaluation_request','native_packet','evaluation_complete')]
    need([e['kind'] for e in stream]==['evaluation_request','native_packet','native_packet','evaluation_complete']*35,'causal request/path/complete order')
    for n,(e,r,g,mode) in enumerate(zip(evaluations,requests,groups,modes,strict=True)):
        prediction=e['prediction'];util.prediction(t,prediction)
        need(e['evaluation_id']==r['evaluation_id']==g and e['mode']==r['mode']==mode and
             r['logits']==prediction['logits'] and r['commands']==prediction['commands'],'fresh request inputs/task/mode')
        need(packets[2*n:2*n+2]==[dict(kind='native_packet',evaluation_id=g,mode=mode,packet=k,context=prediction['contexts'][k]) for k in (0,1)],'native context event content')
        for resource in ('predict','action_encoder'):
            need(sum(v['resource']==resource and v['evaluation_id']==g for v in operations)==(6 if g=='initial' or g.endswith('_selected_replay') else 2),'charged fresh path neural forwards')
    values=[e['prediction'] for e in evaluations]
    need(values[0]==values[1]==values[2] and not util.arr(values[0]['logits'],(2,10)).any(),'fresh zero initialization and triple replay')
    starts=[e for e in events if e['kind']=='initial_verified']
    need(len(starts)==1 and starts[0]['lambda_mean']==LAMBDA_MEAN and starts[0]['lambda_unnormalized']==LAMBDA_RAW,'initial lambda receipt')
    util.prediction(t,starts[0]['incumbent'],True)
    need(all(starts[0]['incumbent'][k]==v for k,v in values[0].items()),'initial incumbent content')
    lifecycle=[e for e in events if e['kind'] in ('initial_verified','derivative_complete','line_proposals_frozen','selection_complete','update_verified')]
    need([e['kind'] for e in lifecycle]==['initial_verified']+['derivative_complete','line_proposals_frozen','selection_complete','update_verified']*4,'four causal GN updates')
    incumbent=values[0];steps=[];diagnostics=[]
    for it in range(1,5):
        derivative,proposal,selection,update=lifecycle[1+(it-1)*4:1+it*4]
        d=derivative['derivative'];j=util.arr(d['raw_terminal_jacobian'],(192,20));q=np.asarray(t['projection'])
        need(d['task_id']==io.task_hash(t) and d['at_logits']==incumbent['logits'] and
             d['at_terminal']==incumbent['contexts'][-1]['latent_history'][-1],'fresh J at own incumbent')
        a=q@j;r=q@(np.asarray(d['at_terminal'])-np.asarray(t['goal']))
        util.close(d['objective_gradient'],2*a.T@r/192,'unpenalized model MSE gradient')
        group=f'iter_{it}_AD'
        rows=[e for e in events if e['kind']=='jacobian_row' and e['evaluation_id']==group]
        need([e['row'] for e in rows]==list(range(192)),'all192 ordered raw VJP rows')
        rawj=util.arr([e['raw_control_derivative'] for e in rows],(192,20))
        need(np.array_equal(j,rawj*(1-np.asarray(incumbent['commands']).reshape(20)**2)[None]),'single raw-to-logit tanh chain')
        joint=[e for e in events if e['kind']=='joint_logit_jacobian' and e['evaluation_id']==group]
        forward=[e for e in events if e['kind']=='jacobian_forward_complete' and e['evaluation_id']==group]
        adpath=[e['context'] for e in events if e['kind']=='jacobian_forward_packet' and e['evaluation_id']==group]
        need(len(joint)==len(forward)==1 and joint[0]['jacobian']==j.tolist() and joint[0]['terminal']==d['at_terminal'] and
             forward[0]['terminal']==d['at_terminal'] and forward[0]['logits']==incumbent['logits'],'ADforward and saved chained J binding')
        util.path_check(t,incumbent['commands'],adpath);need(adpath==incumbent['contexts'],'fresh fullH2 AD path')
        for resource,count in dict(predict=2,action_encoder=2,jacobian=1,vjp_directions=192,svd=1,linear_solve=1,proposal=1).items():
            need(sum(v['resource']==resource and v['evaluation_id']==group for v in operations)==count,'ADforward/math scoped cost')
        need(proposal['iteration']==it and proposal['lambda_mean']==LAMBDA_MEAN and proposal['lambda_unnormalized']==LAMBDA_RAW and
             proposal['scales']==list(SCALES) and proposal['incumbent_logits']==incumbent['logits'],'explicit matched lambda/scale receipt')
        delta=direction(a,r);saved=util.arr(proposal['delta'],(20,))
        util.close(saved,delta,'raw-lambda .002 normal-equation direction')
        # Independent second formulation, matching the source convention but not its optimizer.
        left,s,right=np.linalg.svd(a/np.sqrt(192),full_matrices=False)
        spectral=-right.T@(s*(left.T@(r/np.sqrt(192)))/(s*s+LAMBDA_MEAN))
        util.close(saved,spectral,'mean-lambda .002/192 SVD direction')
        util.close(proposal['endpoint'],util.project(np.asarray(incumbent['logits'])+delta.reshape(2,10)),'projected endpoint')
        endpoint=util.arr(proposal['endpoint'],(2,10));x=np.asarray(incumbent['logits'])
        need((np.linalg.norm(endpoint,axis=1)<=util.CAP).all(),'feasible saved endpoint')
        trials=[(x+s*(endpoint-x)).astype(np.float32).astype(float).tolist() for s in SCALES]
        need(proposal['logits']==trials,'exact float32 line-segment candidates')
        offset=3+(it-1)*8;candidates=values[offset:offset+5]
        need([p['logits'] for p in candidates]==trials,'all five candidates freshly evaluated')
        pool=[incumbent]+candidates;k=min(range(6),key=lambda k:(pool[k]['cost'],k))
        row=dict(iteration=it,incumbent_cost=incumbent['cost'],trial_costs=[p['cost'] for p in candidates],
                 selected_index=k,selected_scale=None if k==0 else SCALES[k-1],selected_cost=pool[k]['cost'])
        need(all(selection[key]==v for key,v in row.items()),'stable incumbent-first full-candidate choice')
        need(values[offset+5:offset+8]==[pool[k]]*3,'independent selected triple replay')
        util.prediction(t,update['prediction'],True)
        need(update['iteration']==it and all(update['prediction'][key]==v for key,v in pool[k].items()),'updated incumbent content')
        # Bind freeze and selection to actual journal positions, not just list censuses.
        position=lambda e:next(n for n,v in enumerate(events) if v is e)
        need(position(proposal)<position(requests[offset]) and position(evaluations[offset+4])<position(selection)<position(requests[offset+5]) and
             position(evaluations[offset+7])<position(update),'freeze before trials; full candidates before selection; replay before update')
        diagnostics.append(dict(iteration=it,lambda_mean=LAMBDA_MEAN,lambda_unnormalized=LAMBDA_RAW,
            max_delta_difference=float(np.max(np.abs(saved-delta))),normal_equation_residual_norm=float(np.linalg.norm((a.T@a+LAMBDA_RAW*np.eye(20))@saved+a.T@r)),
            selected_index=k,selected_scale=row['selected_scale'],selected_cost=row['selected_cost'],jacobian_shape=[192,20],raw_VJP_rows=192))
        incumbent=pool[k];steps.append(row)
    need(result['steps']==steps,'full result step chain')
    util.prediction(t,result['final_prediction'],True)
    need(all(result['final_prediction'][k]==v for k,v in incumbent.items()),'final incumbent/result binding')
    choice=dict(task_id=io.task_hash(t),logits=incumbent['logits'],commands=incumbent['commands'],prediction=incumbent['contexts'][-1],model_terminal_cost=incumbent['cost'])
    return choice,diagnostics


def audit(report_path,report_sha,prepare_sha):
    external_sha(report_sha);external_sha(prepare_sha)
    report_path=Path(report_path).absolute();directory=report_path.parent;r=util.read(report_path,report_sha)
    need(r['schema']==RUNNER+'.report' and r['prepare_sha256']==prepare_sha,'own schema/external prepare SHA')
    need(r['status']=='matched_ridge_model_plans_sealed_not_admitted' and r['errors']==[] and r['first_error'] is None and
         r['commands_released'] is True,'partial/failure rejected for release')
    p=util.read(r['prepared'],prepare_sha);need(p['schema']==RUNNER+'.prepare' and len(p['slots'])==len(r['rows'])==2,'two own slots')
    source=io.ROOT/'experiments/spectral_reachability'/(RUNNER+'.py')
    need(p['bindings_sha256'].get(str(source))==RUNNER_SHA and io.sha(source)==RUNNER_SHA,'frozen matched runner')
    contract=p['contract']
    need(r['contract']==contract and contract['arm']==ARM and contract['lambda_mean']==LAMBDA_MEAN and contract['lambda_unnormalized']==LAMBDA_RAW and
         contract['updates']==4 and contract['scales']==list(SCALES) and contract['per_root_budget']==BUDGET and contract['total_budget']=={k:2*v for k,v in BUDGET.items()} and
         contract['AD_API_per_root']==API and contract['equal_budget_claim'] is False,'fixed matched-ridge protocol/resources')
    need(p['expected_model_hash']==io.TENSOR_SHA and r['source_verified'] is True and r['model_unchanged'] is True and
         r['model_hash_before']==r['model_hash_after']==io.TENSOR_SHA,'inherited frozen-model receipt')
    need(all(r[k] is False for k in ('training','new_Q','P1_admission','equal_budget_claim')) and r['new_images']==r['new_physics']==0,'no new physics/training/metric claim')
    exports=[str(io.ROOT/f'results/p1_constructive_blind_encode_cuda_20261027_v1_cuda_main01/task_{i:02d}.json.gz') for i in range(2)]
    export_shas=['714f471761278fa654542d4bb0878ce18e3c304df6158005af46d2129533265a','70865d56f9b66f0072b2ad7de01cc09d9a9275047925e188104a13d5182ce1f2']
    need(p['arguments']==dict(task_exports=exports,task_shas=export_shas),'exact approved ordered exports')
    for path,h in p['bindings_sha256'].items():
        f=Path(path);need(path in exports or path in io.MODELS or (f.suffix in ('.py','.md') and 'results' not in f.parts),'no privileged/unknown data paths')
        need(io.sha(f)==h,'source/model/export hash '+f.name)
    need(all(p['bindings_sha256'].get(path)==h for path,h in {**io.MODELS,**dict(zip(exports,export_shas))}.items()),'complete model/input bindings')
    expected={'binding.json','events.jsonl.gz'}|{f'root_{i}_{ARM}.{suffix}' for i in range(2) for suffix in ('result.json.gz','plan.json')}
    need(set(r['artifact_sha256'])==expected and {f.name for f in directory.iterdir()}==expected|{'report.json'},'all two-root artifacts')
    for name,h in r['artifact_sha256'].items():need(io.sha(directory/name)==h,'artifact hash '+name)
    need(util.read(directory/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=prepare_sha),'execution binding')
    raw=util.load(directory/'events.jsonl.gz');need(raw.endswith(b'\n'),'complete durable gzip lines')
    wrapped=[io.parse(line) for line in raw.splitlines()]
    need(all(set(e)=={'source_slot','arm','task_id','event'} and type(e['source_slot']) is int and e['source_slot'] in (0,1) and e['arm']==ARM for e in wrapped),'own journal scopes only')
    need([e['source_slot'] for k,e in enumerate(wrapped) if k==0 or e['source_slot']!=wrapped[k-1]['source_slot']]==[0,1],'causal complete source order')
    summaries=[]
    for i,(slot,row) in enumerate(zip(p['slots'],r['rows'],strict=True)):
        need(slot['order']==slot['source_slot']==i and slot['task_export']==exports[i] and slot['task_sha256']==export_shas[i] and slot['blind_sha256']==io.BLIND_SHA[i],'approved task/source metadata')
        t=util.export_task(exports[i],export_shas[i]);e=util.read(exports[i],export_shas[i]);ident=io.task_hash(t)
        need(e['source_slot']==i and e['blind_sha256']==slot['blind_sha256'] and slot['task_id']==ident==row['task_id'] and
             row['source_slot']==i and row['blind_sha256']==slot['blind_sha256'] and row['arm']==ARM and row['status']=='model_plan_sealed','same blind source/task row join')
        stem=f'root_{i}_{ARM}';need(row['result_file']==stem+'.result.json.gz' and row['plan_file']==stem+'.plan.json','canonical own result paths')
        need(row['resources']==util.counts(BUDGET) and row['AD_API_resources']==util.counts(API),'actual row counts')
        group=[v for v in wrapped if v['source_slot']==i];need(all(v['task_id']==ident for v in group),'journal task identity')
        events=[v['event'] for v in group]
        need(events[0]==dict(kind='arm_started',task_sha256=slot['task_sha256'],lambda_mean=LAMBDA_MEAN),'own start receipt')
        choice,iterations=audit_root(t,util.read(directory/row['result_file']),events)
        plan=util.read(directory/row['plan_file']);selection_sha=plan.pop('selection_sha256')
        expected_plan=dict(schema=RUNNER+'.model_plan',status='selected_pending_global_closure',source_slot=i,arm=ARM,
            blind_sha256=slot['blind_sha256'],task_export=slot['task_export'],task_sha256=slot['task_sha256'],task_contract_sha256=t['task_contract_sha256'],
            with_target=False,physical_execution=False,selection_uses_physics=False,P1_admission=False,lambda_mean=LAMBDA_MEAN,lambda_unnormalized=LAMBDA_RAW,**choice)
        need(plan==expected_plan and selection_sha==row['selection_sha256']==io.task_hash(plan) and row['commands']==choice['commands'],'independent own plan/selection/action seal')
        need(events[-1]==dict(kind='model_selection_sealed',selection_sha256=selection_sha,commands=choice['commands']),'durable command release')
        summaries.append(dict(source_slot=i,arm=ARM,task_id=ident,selection_sha256=selection_sha,model_terminal_cost=choice['model_terminal_cost'],iterations=iterations))
    totals=util.counts({k:2*v for k,v in BUDGET.items()});need(r['resources']==totals,'global full budget')
    size=sum(f.stat().st_size for f in directory.iterdir());need(size<=8*1024**2,'actual eightMiB cap')
    need(io.sha(report_path)==report_sha and io.sha(r['prepared'])==prepare_sha,'final report/prepare TOCTOU')
    for path,h in p['bindings_sha256'].items():need(io.sha(path)==h,'final source/input hash')
    for name,h in r['artifact_sha256'].items():need(io.sha(directory/name)==h,'final artifact hash')
    return dict(schema=NAME+'.report',status='matched_ridge_saved_arithmetic_consistent_not_admitted',report_sha256=report_sha,prepare_sha256=prepare_sha,
        runner_sha256=RUNNER_SHA,rows=summaries,resources=totals,event_count=len(wrapped),saved_bytes=size,jacobians_recomputed=8,raw_VJP_rows_checked=1536,
        AD_API_totals=util.counts(dict(grad=1536,jacobian=0)),lambda_mean=LAMBDA_MEAN,lambda_unnormalized=LAMBDA_RAW,
        new_neural_calls=0,new_physics=0,P1_admission=False,equal_budget_claim=False,
        limitations=['Saved arithmetic only, not independent derivative/neural truth or physical efficacy.',
                    'No optimizer/consume imported; only independent saved utility functions reused.',
                    'Lambda coefficient matches D2P; point, dimension, constraints, selection and budget remain different.',
                    'No physical/sourcepack/certificate or old outcome comparison inputs. Partial runs rejected, never repaired.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('report','report-sha','prepare-sha','output'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    try:result=audit(args.report,args.report_sha,args.prepare_sha);code=0
    except Exception as exc:
        result=dict(schema=NAME+'.report',status='saved_audit_rejected',report_sha256=args.report_sha,error=dict(type=type(exc).__name__,message=str(exc)),P1_admission=False);code=1
    result['auditor_sha256']=io.sha(__file__);result['utility_sha256']={str(Path(util.__file__)):io.sha(util.__file__),str(Path(io.__file__)):io.sha(io.__file__)}
    raw=io.canonical(result);out=Path(args.output)
    need(len(raw)<128*1024 and shutil.disk_usage(out.parent).free>=200*1024**2+len(raw),'exclusive small report with reserve')
    with out.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    print(raw.decode().strip());return code


if __name__=='__main__':raise SystemExit(main())
