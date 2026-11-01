"""Saved-only fixed5 cost/task comparison; no environment/model/run entrypoint."""
from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import agent_p3_fixed_execution_saved_20261028_v1 as saved

NAME='agent_p3_cost_comparison_20261028_v1'
ROOT=saved.ROOT
ORDER=[(s,m) for s in (0,1) for m in ('fixed','adaptive')]
FLAGS=dict(saved.FLAGS)
require,sha,raw,read=saved.require,saved.sha,saved.raw,saved.physical.read
digest=saved.fixed.digest
core=saved.fixed.core
AXES=tuple(core.AXES)
PRINCIPAL=('predict_API','predict_samples','action_encoder_calls','predictor_calls','pred_proj_calls','jacobian_API','grad_API','VJP_directions','raw_backward_callbacks')
METRICS=('native7_distance','joint_xy_error','object_xy_error','agent_xy_error','native_angle_term','wrapped_angle_error','velocity_error')
UPSTREAM_SHAS=('7bfc3a204e40be38b440f97249fdf2ee0330dfc8562baeef51662fff5701235f',
    '4a589ce79f16ea11ac6dc8205e5fe8d39be62394df9f4717e4f10991e5bd6dc1',
    '9112f2451b474f1af0631efac393caefb685822fdee117ca83a2b375a0b7c199')
CUDA_SHA='8067a743e3893b120ac8d57582c8552cb5b1dec182f7ca4bd0c6f70ffacaeeb5'
PHYSICAL_SHA='5fdfc8cfea7f6b4fe78d3e19b59fe946aad73a537850c52a86f2d395055b5e45'
CAP=8*1024**2
consume_saved=saved.consume


def own_files():
    return [Path(__file__).resolve(),ROOT/f'tests/{NAME}_test.py',ROOT/f'innerdocs/{NAME}.md']


def pin():
    paths=saved.own_files()
    require(all(sha(p)==h for p,h in zip(paths,UPSTREAM_SHAS,strict=True)),
        'frozen complete saved consumer source/tests/doc SHA before consumption')
    return {str(p):h for p,h in zip(paths,UPSTREAM_SHAS,strict=True)}


def ordered(rows,label):
    require(type(rows) is list and len(rows)==4 and all(type(r['slot']) is int for r in rows) and
        [(r['slot'],r['mode']) for r in rows]==ORDER,'complete exact all2roots fixed/adaptive '+label)
    return {(r['slot'],r['mode']):r for r in rows}


def counters(value):
    require(set(value)==set(AXES) and all(type(value[a]) is int and value[a]>=0 for a in AXES),
        'all original homogeneous counter axes, no missing-to-zero/float/bool')
    return deepcopy(value)


def sum_counts(values):
    require(bool(values),'nonempty measured count groups')
    for value in values:counters(value)
    return {a:sum(v[a] for v in values) for a in AXES}


def ledger_group(ledger,events):
    # Audit the full actual shape/API/sample/J/VJP ledger, never inferred calls.
    core.audit_ledger(ledger)
    require(events and all(e['status']=='returned' for e in events),'complete actual metered events')
    attempted=sum_counts([e['delta'] for e in events])
    inclusive={r:sum(e['ended_ns']-e['started_ns'] for e in events if e['resource']==r)
        for r in sorted({e['resource'] for e in events})}
    return dict(attempted=attempted,completed=deepcopy(attempted),inclusive_operation_wall_ns=inclusive,
        exclusive_branch_wall_seconds=None,wall_scope='nested inclusive per resource; NOT additive or exclusive branch time')


def native_errors(goal,state):
    values=saved.physical.metrics(goal,state);delta=saved.source.vector(state)-saved.source.vector(goal)
    return dict(native7_distance=values['native7_distance'],native7_reward=values['native7_reward'],
        native_success=values['native_success'],joint_xy_error=values['joint_xy_error'],
        object_xy_error=float(np.linalg.norm(delta[2:4])),agent_xy_error=float(np.linalg.norm(delta[:2])),
        native_angle_term=values['native_angle_term'],wrapped_angle_error=values['wrapped_angle_error'],
        velocity_error=values['velocity_error'],object_success=None)


def physical_account(row,operations):
    selected=[op for op in operations if op['branch']==row['index']]
    result={k:dict(attempted=0,completed=0) for k in saved.contract()['caps']}
    for op in selected:
        require(op['status'] in ('returned','failed') and all(op[k]==row[k] for k in ('root_slot','condition','arm','selection_digest')),
            'actual physical operation scoped to exact request')
        result[op['resource']]['attempted']+=1;result[op['resource']]['completed']+=op['status']=='returned'
    for phase,label in (('warmup','warmup'),('plan','own')):
        events=[e for e in selected if e['resource']=='public_step' and e['phase']==phase]
        require(row[label+'_attempted']==len(events) and row[label+'_returned']==sum(e['status']=='returned' for e in events),
            'actual attempt/return; no earlystop padding/refund')
    return result


def comparison(fixed,adaptive):
    return {a:adaptive[a]-fixed[a] for a in AXES}


def mean_known(values,denominator):
    known=[v for v in values if v is not None]
    return dict(denominator=denominator,known=len(known),unknown=denominator-len(known),
        full_denominator_mean=float(np.mean(known)) if len(known)==denominator else None,
        known_only_mean=float(np.mean(known)) if known else None)


def assemble(cuda,physical,verified,prepared,signals):
    """Pure saved arithmetic; caller MUST authenticate all objects via material."""
    require(cuda['schema']==saved.physical.PRODUCER+'.report' and cuda['status']=='saved_model_prefix_not_admitted' and
        cuda['errors']==[] and all(cuda[k] is False for k in FLAGS),'complete CUDA engineering report, gates closed')
    require(physical['schema']==saved.PHYSICAL+'.report' and physical['fixed_denominator']==4 and
        verified['schema']==saved.NAME+'.consume' and verified['physical_report_schema']==physical['schema'] and
        verified['physical_report_sha256']==PHYSICAL_SHA and verified['producer_report_sha256']==CUDA_SHA and
        verified['rows']==physical['rows'] and verified['counts']==physical['counts'] and verified['errors']==physical['errors'] and
        all(physical[k] is verified[k] is False for k in FLAGS),'complete real saved physical consumer and same report')
    producer=verified['producer_consume']
    require(producer['schema']==saved.physical.PRODUCER+'.consume' and producer['report_sha256']==CUDA_SHA and
        producer['status']=='saved_prefix_verified_not_admitted' and producer['per_branch']==cuda['per_branch'] and
        producer['prefixes']==cuda['prefixes'] and producer['winners']==cuda['winners'] and
        physical['execution_binding']['producer_report_sha256']==CUDA_SHA and all(producer[k] is False for k in FLAGS),
        'mandatory nested full producer.consume, not raw report trust')
    branches=ordered(cuda['per_branch'],'CUDA budgets');prefixes=ordered(cuda['prefixes'],'prefixes')
    winners=ordered(cuda['winners'],'winners');rows=ordered(physical['rows'],'physical rows')
    base=prepared['panel']['base'];plans=ordered(base['branches'],'original sealed budget allocation')
    contexts={x['slot']:x for x in producer['root_contexts']}
    require(len(producer['root_contexts'])==len(contexts)==2 and set(contexts)=={0,1} and all(type(x['slot']) is int for x in producer['root_contexts']),
        'two literal bound root contexts')
    ledger=cuda['resource_receipt'];core.audit_ledger(ledger)
    require(all(type(e['scope']['slot']) is int and (e['scope']['slot'],e['scope']['mode']) in ORDER and
        e['scope'].get('operation') in (None,'selected_prefix_verification') for e in ledger['events']), 'no unallocated CUDA event')
    require(set(signals)=={'fixed','adaptive'},'both original signal panels, no chosen subset')
    out=[]
    for key in ORDER:
        slot,mode=key;b=branches[key];plan=plans[key];row=rows[key];request=prefixes[key]['request'];winner=winners[key]
        require(all(b[k] is None for k in ('FLOPs','CUDA_ns','backward_engine_calls')),
            'unmeasured branch work stays unknown, never zero')
        require(type(row['index']) is type(row['root_slot']) is int and row['index']==ORDER.index(key) and row['root_slot']==slot and row['condition']=='off' and row['arm']==mode and
            row['request']==request and row['selection_digest']==digest(request) and
            row['commands']==request['commands']==winner['actions'][:5] and
            request['binding']==contexts[slot]['binding'] and request['root_context_sha256']==digest(contexts[slot]['context']),
            'exact source/task/context/goal/raw5/request ancestry')
        if row['initial_exact']:require(request['binding']['root_sha256']==row['root_sha256'],'actually authenticated native root')
        require(b['planning_horizon_packets']==plan['horizon']==request['planning_horizon_packets'] and
            b['requested_execution_steps']==request['requested_execution_steps']==5 and plan['candidates']==len(
                [r for r in cuda['rows'] if (r['slot'],r['mode'])==key]), 'original candidate quota/horizon, never reselect')
        signal=signals[mode];signal_ledger=signal['resource_receipt'];core.audit_ledger(signal_ledger)
        require(all(type(e['scope']['slot']) is int and e['scope']['slot'] in (0,1) for e in signal_ledger['events']), 'signal exact two-root scope')
        sig=ledger_group(signal_ledger,[e for e in signal_ledger['events'] if e['scope']['slot']==slot])
        new=[e for e in ledger['events'] if (e['scope']['slot'],e['scope']['mode'])==key]
        search=ledger_group(ledger,[e for e in new if e['scope'].get('operation') is None])
        prefix=ledger_group(ledger,[e for e in new if e['scope'].get('operation')=='selected_prefix_verification'])
        newgroup=ledger_group(ledger,new);total=sum_counts([sig['completed'],newgroup['completed']])
        require(sig['completed']==counters(plan['signal_resources'])==counters(b['signal_resources']) and
            newgroup['completed']==counters(b['new_model_resources']) and total==counters(b['total_resources']) and
            sig['inclusive_operation_wall_ns']==plan['inclusive_signal_wall_ns'] and
            signal['wall_seconds']==plan['signal_panel_wall_seconds'] and plan['signal_task_total_wall_seconds'] is None and
            plan['signal_report_sha256']==base['inputs'][mode]['report_sha256'],
            'full independently paid signal + search + prefix ledger; same units only')
        require(sig['completed']['predict_API']==sig['completed']['predict_samples']==8 and
            search['completed']['predict_API']==search['completed']['predict_samples']==24 and
            prefix['completed']['predict_API']==prefix['completed']['predict_samples']==1 and
            total['predict_API']==total['predict_samples']==33 and total['jacobian_API']==2 and total['VJP_directions']==384,
            'unchanged actual fixed5 budget including prefix and both Js')
        counts=physical_account(row,physical['actual_operation_receipts'])
        valid=(row['initial_exact'] and row['status'] in ('five_steps_completed','native_stopped') and row['physical_receipt'] is not None and
            row['own_returned']==len(row['responses']))
        initial=native_errors(row['goal'],row['root_state']) if row['initial_exact'] else None
        final=native_errors(row['goal'],row['final_state']) if valid and row['final_state'] is not None else None
        if final is not None:require(final['native_success'] is row['native_success'] and
            saved.physical.metrics(row['goal'],row['final_state'])['native']==row['final_metrics'],'independent native task metrics')
        out.append(dict(slot=slot,mode=mode,selection_digest=row['selection_digest'],request=deepcopy(request),
            planning_horizon_packets=plan['horizon'],candidate_quota=plan['candidates'],requested_steps=5,
            returned_steps=row['own_returned'],validated_steps=len(row['responses']),physical_status=row['status'],
            native_root_verified=row['initial_exact'],
            native_earlystop=(row['terminated'] or row['truncated']) if valid else None,
            engineering_error=deepcopy(row['first_error']),model=dict(full_horizon_modelQ=winner['cost'],
                full_horizon_packets=winner['horizon'],selected_prefix_modelQ=request['predicted_execution_modelQ'],
                physical_success_inferred=False),resources=dict(signal=sig,search=search,selected_prefix_verification=prefix,
                new_model=newgroup,total_attempted=sum_counts([sig['attempted'],newgroup['attempted']]),total_completed=total,
                physical=counts,FLOPs=b['FLOPs'],CUDA_ns=b['CUDA_ns'],backward_engine_calls=b['backward_engine_calls']),
            initial=initial,final=final,improvement=None if initial is None or final is None else
                {metric:initial[metric]-final[metric] for metric in METRICS}))
    for slot in (0,1):
        f,a=rows[slot,'fixed'],rows[slot,'adaptive']
        require(f['request']['binding']==a['request']['binding'] and f['request']['root_context_sha256']==a['request']['root_context_sha256'],
            'paired same complete task, not nearby/modelQ-equivalent roots')
        if f['initial_exact'] and a['initial_exact']:require(all(f[k]==a[k] for k in
            ('root_state','root_sha256','goal','goal_pose','goal_pixel_sha256')),'same physical paired native task and goal')
    require(sum_counts([r['resources']['new_model']['completed'] for r in out])==ledger['completed'],'no omitted CUDA budget')
    census={resource:{q:sum(r['resources']['physical'][resource][q] for r in out) for q in ('attempted','completed')}
        for resource in physical['counts']}
    require(census==physical['counts'],'fixed4 physical full denominator counts')
    modes={}
    for mode in ('fixed','adaptive'):
        group=[r for r in out if r['mode']==mode]
        modes[mode]=dict(denominator=2,known_roots=2,total_completed=sum_counts([r['resources']['total_completed'] for r in group]),
            physical={k:{q:sum(r['resources']['physical'][k][q] for r in group) for q in ('attempted','completed')} for k in census},
            native_success=mean_known([None if r['final'] is None else int(r['final']['native_success']) for r in group],2),
            final={metric:mean_known([None if r['final'] is None else r['final'][metric] for r in group],2) for metric in METRICS})
    pairs=[]
    for slot in (0,1):
        f,a=out[2*slot:2*slot+2]
        pairs.append(dict(slot=slot,cost_delta_adaptive_minus_fixed=comparison(f['resources']['total_completed'],a['resources']['total_completed']),
            final_error_delta_adaptive_minus_fixed=None if f['final'] is None or a['final'] is None else
                {metric:a['final'][metric]-f['final'][metric] for metric in METRICS},
            native_success_delta=None if f['final'] is None or a['final'] is None else int(a['final']['native_success'])-int(f['final']['native_success'])))
    delta=comparison(modes['fixed']['total_completed'],modes['adaptive']['total_completed'])
    success_values=[p['native_success_delta'] for p in pairs]
    return dict(schema=NAME+'.analysis',status='saved_comparison_not_admitted',fixed_denominator=4,paired_denominator=2,known_roots=2,
        rows=out,modes=modes,pairs=pairs,cost_evidence=dict(delta_adaptive_minus_fixed=delta,
            lower_count_axes=[k for k,v in delta.items() if v<0],equal_count_axes=[k for k,v in delta.items() if v==0],
            higher_count_axes=[k for k,v in delta.items() if v>0],
            forward_AD_count_saving=all(delta[k]<=0 for k in PRINCIPAL) and any(delta[k]<0 for k in PRINCIPAL),
            FLOPs_saving=None,whole_job_wall_saving=None,fixed_H1_cost_minimal_baseline=False,
            interpretation='per-axis operation counts only; replay/metric counts are not FLOPs, batching or per-call work'),
        benefit_evidence=dict(native_success_gain=mean_known(success_values,2),
            final_error_delta_adaptive_minus_fixed={metric:mean_known([None if p['final_error_delta_adaptive_minus_fixed'] is None else
                p['final_error_delta_adaptive_minus_fixed'][metric] for p in pairs],2) for metric in METRICS},
            native_success_gain_observed=(sum(success_values)>0) if all(v is not None for v in success_values) else None,
            interpretation='negative paired error delta means smaller error only on these two roots; not native success or generalization'),
        wall=dict(cuda_report_wall_seconds=cuda['wall_seconds'],cuda_new_ledger_wall_ns=ledger['wall_ns'],
            cuda_new_inclusive_operation_wall_ns=ledger['inclusive_operation_wall_ns'],
            signal_panels={mode:dict(wall_seconds=s['wall_seconds'],ledger_wall_ns=s['resource_receipt']['wall_ns'],
                report_sha256=base['inputs'][mode]['report_sha256']) for mode,s in signals.items()},
            physical_execute_wall_seconds=physical['wall_seconds'],physical_wall_scope=physical['contract']['wall_scope'],
            exclusive_branch_wall_seconds=None,whole_pipeline_wall_seconds=None,
            semantics='panel walls once per panel, not per root; nested inclusive axes never added; no per-method speedup inferred'),
        saved_verification=deepcopy(cuda['saved_verification']),
        historical_acquisition=dict(scope='inherited historical acquisition, different schema; not zero, not added to new controlled comparison',
            receipt=deepcopy(base['historical_acquisition']),signal_panels=deepcopy(base['historical_signal_panels'])),
        missing_for_C3=['P1/applicable P2 and P3 admission remain closed; no certified fresh independent cohort',
            'two known roots only; four arms are not four independent roots',
            'root-only joint H1/H3 lookahead is not a closed-loop C3 tree',
            'no newly observed child context, paid child-local signals or child replanning',
            'controlled fixed quota is not a minimal-cost H1 baseline',
            'FLOPs/CUDA timings/backward-engine work and exclusive end-to-end per-method wall unknown',
            'model-selected full-horizon Q is not common-step physical task success'],
        complete_C3_tree=False,new_neural_calls=0,new_physics_calls=0,**FLAGS)


def material(arguments):
    require(set(arguments)=={'cuda_output','cuda_report_sha256','physical_output','physical_report_sha256'},'exact saved input arguments')
    pins=pin()
    require(arguments['cuda_report_sha256']==CUDA_SHA and arguments['physical_report_sha256']==PHYSICAL_SHA,'fixed actual report authorities')
    cp=Path(saved.absolute(arguments['cuda_output']))/'report.json';pp=Path(saved.absolute(arguments['physical_output']))/'report.json'
    require(sha(cp)==CUDA_SHA and sha(pp)==PHYSICAL_SHA,'actual report SHA before any consume')
    verified=consume_saved(str(pp.parent),PHYSICAL_SHA)
    require(verified['producer_report_sha256']==CUDA_SHA and verified['bindings'][str(cp)]==CUDA_SHA,'exact nested CUDA path/whole report')
    cuda=read(cp,256*1024**2);physical=read(pp,32*1024**2)
    prepare_path=Path(saved.absolute(cuda['prepared']))
    require(sha(prepare_path)==cuda['prepare_sha256']==verified['bindings'][str(prepare_path)],'actual CUDA prepare closure')
    prepared=read(prepare_path,16*1024**2);signals={}
    for mode,value in prepared['panel']['base']['inputs'].items():
        path=Path(saved.absolute(value['output']))/'report.json'
        require(sha(path)==value['report_sha256']==prepared['sources'][str(path)]==verified['bindings'][str(path)],
            'explicit independently paid signal report edge')
        signals[mode]=read(path,256*1024**2)
    result=assemble(cuda,physical,verified,prepared,signals)
    bindings=deepcopy(verified['bindings']);saved.merge(bindings,pins)
    saved.merge(bindings,{str(p):sha(p) for p in own_files()})
    for path,h in bindings.items():require(sha(path)==h,'final full source/input/artifact closure')
    require(pin()==pins,'final frozen consumer seals')
    result.update(arguments=deepcopy(arguments),bindings=bindings,upstream_consume=dict(schema=verified['schema'],
        status=verified['status'],receipt_sha256=digest(verified),physical_report_sha256=PHYSICAL_SHA,producer_report_sha256=CUDA_SHA,
        errors=deepcopy(verified['errors']),artifact_authenticity_verified=verified['artifact_authenticity_verified']))
    require(len(raw(result))<=CAP,'bounded full comparison; never drop denominator or unknowns')
    return result


def write_new(output,arguments):
    path=Path(saved.absolute(output))
    require(path.parent==ROOT/'results' and path.name.startswith(NAME+'_') and not path.exists() and not path.is_symlink(),
        'NEW exclusive own analysis BEFORE any saved consumption')
    result=material(arguments);content=raw(result)
    require(len(content)<=CAP,'bounded output before exclusive write')
    with path.open('xb') as stream:
        require(stream.write(content)==len(content),'complete own analysis write');stream.flush();os.fsync(stream.fileno())
    saved.physical.paired.fsync_dir(path.parent)
    return dict(schema=NAME+'.written',output=str(path),sha256=sha(path),bytes=len(content),status=result['status'])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('analyze',))
    for name in ('cuda-output','cuda-report-sha','physical-output','physical-report-sha'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--output',help='optional NEW exclusive own result; otherwise stdout only')
    a=parser.parse_args(argv);arguments=dict(cuda_output=a.cuda_output,cuda_report_sha256=a.cuda_report_sha,
        physical_output=a.physical_output,physical_report_sha256=a.physical_report_sha)
    result=write_new(a.output,arguments) if a.output else material(arguments)
    print(saved.source.canonical(result));return 0


if __name__=='__main__':raise SystemExit(main())
