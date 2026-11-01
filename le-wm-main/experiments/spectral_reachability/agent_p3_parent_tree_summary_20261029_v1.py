"""Saved report + completed consume summary. Stdlib only; never invokes producer/strict.

External hashes authenticate the selected saved bytes, not a new certificate.
All four slots and their original144 expansion denominators remain visible.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
NAME='agent_p3_parent_tree_summary_20261029_v1'
PRODUCER='agent_p3_parent_native_cuda_20261029_v1'
PARENT='agent_p3_parent_native_adapter_20261029_v1'
REPORT=ROOT/'results/agent_p3_parent_native_cuda_20261029_v1_cuda_MAIN01/report.json'
REPORT_SHA='75f4c162aaa75223dfd8523381e7614c8a35637afd52278df8a80d6affa34e40'
CONSUME=ROOT/'results/agent_p3_parent_native_cuda_20261029_v1_consume_MAIN01.json'
CONSUME_SHA='622d1cdaf8d1ed69c4e78f8d1f5c7c6114285db6121f0021fbbaf6573a8a76ad'
PINS={ROOT/'experiments/spectral_reachability'/(PRODUCER+'.py'):'eeea5710d4171b7c43f99d1f2b5ad2412e03a25ba14e8fc5589ea5ef73f57b3b',
    ROOT/'tests'/('test_'+PRODUCER+'.py'):'31335c5f62936861f2af1a3f83c14eb862199f944626e0c49d077041b1042af5',
    ROOT/'innerdocs'/(PRODUCER+'.md'):'7ad5972d1efc9c2fec32454f756b7eff1e363fe5ece7750bbdad8305122e346b'}
ADAPTER_SHA='150a1c4189a68f9b7c71479e5aa6980cea490fdeed0d5524b7bfad0cc78c2558'
FLAGS=dict(P1_admission=False,P3_admission=False,scientific_admission=False,freshness=False,
    physical_release=False,full_policy_executed=False,full_C3_demonstrated=False,native_execution_certified=False,
    adaptive_width8_executed=False,reconfiguration_executed=False,all_program_types_covered=False,equal_FLOPs=False,
    P2_admission=False,complete_R=False)
AXES=('predict_API','predict_samples','predict_context_tokens','action_encoder_calls','action_encoder_samples',
    'action_encoder_context_tokens','predictor_calls','predictor_samples','predictor_context_tokens','pred_proj_calls',
    'pred_proj_rows','encoder_calls','encoder_samples','projector_calls','projector_samples','jacobian_API','VJP_directions',
    'grad_API','raw_backward_callbacks','model_SVD_calls','array_SVD_calls','solve_calls','signal_compile_calls',
    'program_compile_calls','replay_calls','metric_calls','schedule_decisions')
STATUSES={'callback_progress_complete_NOT_certified','refused_unsupported_original_type',
    'resource_cap_before_node_queue','expansion_cap_before_node_queue','failed_or_budget_exhausted'}
CAPS={'resource_cap_before_node_queue','expansion_cap_before_node_queue'}
DECISION_KEYS=('program_type','extension_type','horizon','candidates','program_depth_limit','continue_expanding',
    'decision_source','inherited_engineering_thresholds','no_optimality_claim')


def require(ok,message):
    if not ok:raise ValueError(message)


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def keys(value,names):require(type(value) is dict and set(value)==set(names),'exact fields: '+','.join(names))
def integer(value):require(type(value) is int and value>=0,'literal nonnegative integer')
def hash_value(value):require(type(value) is str and len(value)==64 and all(x in '0123456789abcdef' for x in value),'literal SHA256')
def pairs(items):
    value={}
    for k,v in items:require(k not in value,'duplicate JSON key');value[k]=v
    return value


def load(path,expected,cap):
    hash_value(expected);p=Path(path).absolute()
    require(p.resolve()==p and not p.is_symlink() and p.is_file() and p.stat().st_size<=cap,'bounded regular literal input path')
    require(sha(p)==expected,'external input SHA mismatch')
    value=json.loads(p.read_bytes(),object_pairs_hook=pairs);canonical(value)
    require(sha(p)==expected,'input changed while reading');return p,value


def roster():return [dict(slot=2*r+i,root=r,mode=m) for r in (0,1) for i,m in enumerate(('fixed','adaptive'))]


def counted(values):return dict(sorted(Counter(map(str,values)).items()))


def matrix(value,rows,cols):
    require(type(value) is list and len(value)==rows,'complete matrix rows')
    for row in value:
        require(type(row) is list and len(row)==cols,'complete matrix columns')
        require(all(type(x) in (float,int) and math.isfinite(x) for x in row),'literal finite matrix')


def signal_identities(runtime):
    result=[];root_binding=runtime['root_binding'];hash_value(runtime['root_task_sha256'])
    for i,m in enumerate(runtime['signal_mappings']):
        require(m['schema']==PARENT+'.signal_map' and m['mapping_sha256']==digest({k:v for k,v in m.items() if k!='mapping_sha256'}),'exact signal mapping digest')
        require(m['root_task_sha256']==runtime['root_task_sha256'] and m['root_binding']==root_binding,'original root authority in every parent map')
        outer,task,req,response,out=(m[k] for k in ('outer_request','parent_task','native_request','native_response','outward_response'))
        context=outer['parent_context'];csha=digest(context)
        matrix(context['latent_history'],3,192);matrix(context['action_history'],2,10)
        require(outer['binding']==root_binding and outer['binding_sha256']==digest(root_binding) and outer['parent_context_sha256']==csha,'outer root/full-parent identity')
        require(task['root_context']==context and task['binding']==dict(root_binding,context_sha256=csha),'only context binding rebased in parent view')
        h=outer['horizon'];require(type(h) is int and h in (1,3),'H1/H3 identity')
        require(req==dict(binding=task['binding'],binding_sha256=digest(task['binding']),parent_context=context,
            parent_context_sha256=csha,operation='compile_finite_horizon_J',horizon=h),'native same-parent request')
        require(response['binding_sha256']==req['binding_sha256'] and response['parent_context_sha256']==csha and response['horizon']==h,'native parent response identity')
        require(out==dict(response,binding_sha256=outer['binding_sha256'],producer_sha256=ADAPTER_SHA),'explicit outward adapter identity')
        matrix(response['jacobian'],192,10*h)
        result.append(dict(mapping_index=i,root_task_sha256=runtime['root_task_sha256'],parent_context_sha256=csha,
            is_original_root_context=csha==root_binding['context_sha256'],horizon=h,
            parent_task_sha256=digest(task),native_binding_sha256=req['binding_sha256'],outer_binding_sha256=outer['binding_sha256'],
            native_producer_sha256=response['producer_sha256'],mapping_sha256=m['mapping_sha256'],
            jacobian_shape=[192,10*h],jacobian_semantic_json_sha256=digest(response['jacobian']),
            nominal_semantic_json_sha256=digest(response['nominal']),native_array_bytes_rechecked=False))
    flat=[x for s in runtime['signals'] for x in s['responses']]
    require(flat==[m['outward_response'] for m in runtime['signal_mappings'][:len(flat)]],'returned local receipts match signal-map prefix')
    return result


def summarize_arm(row,accepted):
    runtime=row['result'];identity={k:row[k] for k in ('slot','root','mode')}
    base=dict(**identity,status=row['status'],original_expansion_denominator=144,observed_job_rows=0,
        unobserved_expansion_slots=144,job_status_counts=None,resources=None,plan_available=False,
        root_task_sha256=None,root_binding=None,queue_nodes=[],signal_parent_identities=[],
        realized_progress_depth=None,allocated_widths=None,returned_node_depth_counts=None,returned_extension_H_counts=None,
        queue_horizon_counts=None,mapped_signal_H_counts=None,mapped_parent_contexts=None,mapped_child_contexts=None,
        J_API_attempted=None,J_API_completed=None,signal_mappings_returned=None,refusal_nodes=None,cap_nodes=None,
        original_child_stops=None,plan=None,diagnostic_best_available=None,errors=None,
        native_calls_from_strict=None,complete_native_calls_from_strict=None,
        resource_unknowns=dict(scalar_backward_calls=None,AD_primal_predict_API=None,Jacobian_elements=None,total_FLOPs=None))
    if runtime is None:
        require(accepted is None and row['status']=='blocked_after_runtime_failure','only genuinely blocked arm has absent result')
        return base
    require(runtime['schema']==PARENT+'.runtime' and runtime['mode']==row['mode'] and runtime['status']==row['status'] and row['status'] in STATUSES,'native arm schema/status identity')
    require(all(runtime[k] is False for k in FLAGS if k not in ('P2_admission','complete_R')),'no admission in runtime')
    require(runtime['fixed_denominator']==dict(expansion_slots=144,max_original_expansions=144) and len(runtime['jobs'])==144,'original144 full denominator')
    require([x['slot'] for x in runtime['jobs']]==list(range(144)),'all ordered original expansion slots')
    require(accepted is not None and all(accepted[k]==row[k] for k in ('slot','root','mode','status')),'strict arm identity')
    keys(accepted,('slot','root','mode','status','native_calls','complete_native_calls','model_plan_available',
        'realized_progress_depth','realized_widths','resources','scalar_backward_calls','AD_primal_predict_API','Jacobian_elements'))
    integer(accepted['native_calls']);integer(accepted['complete_native_calls'])
    require(accepted['complete_native_calls']<=accepted['native_calls'],'completed native call count')
    resources=runtime['resource_receipt'];require(accepted['resources']==resources,'strict/report full resource receipt equality')
    for field in ('caps','attempted','completed'):
        keys(resources[field],AXES)
        for value in resources[field].values():integer(value)
    require(all(resources['completed'][k]<=resources['attempted'][k]<=resources['caps'][k] for k in AXES),'actual nonrefundable resource ordering')
    require(resources['caps']['predict_API']==(114 if row['mode']=='fixed' else 562),'original progress shell predict cap')
    require(all(runtime[k] is None and accepted[k] is None for k in ('scalar_backward_calls','AD_primal_predict_API','Jacobian_elements')),'unknown resources not zero')
    require(all(resources[k] is None for k in ('FLOPs','backward_engine_calls','cuda_ns')) and resources['physics_steps']==0,'no invented FLOPs/physical calls')
    jobs=runtime['jobs'];allowed={'not_requested','returned','refused_unsupported_original_type','failed','blocked_after_failure'}
    require(all(x['status'] in allowed for x in jobs),'only final expansion dispositions')
    cursor=0;queues=[]
    for i,q in enumerate(runtime['queues']):
        d=q['decision'];keys(d,DECISION_KEYS)
        require(type(d['candidates']) is int and d['candidates'] in (4,8) and type(d['horizon']) is int and d['horizon'] in (1,3),'original natural width/H')
        require(type(q['depth']) is int and 0<=q['depth']<3 and type(d['program_depth_limit']) is int and d['program_depth_limit'] in (1,3),'original parent/depth limit')
        require(q['status'] in {'queued','stopped_by_original_child_policy','refused_unsupported_original_type'}|CAPS,'known queue status')
        reqs=q['requests'];require(len(reqs)==(d['candidates'] if d['continue_expanding'] else 0),'complete4/8 queue or child stop')
        if reqs:
            require([x['candidate_index'] for x in reqs]==list(range(d['candidates'])),'atomic original candidate identities')
            require(all(x['parent_context_sha256']==q['parent_context_sha256'] and all(x[k]==d[k] for k in DECISION_KEYS) for x in reqs),'all requests retain full original decision')
        allocated=q['status'] in ('queued','refused_unsupported_original_type')
        selected=jobs[cursor:cursor+len(reqs)] if allocated else []
        if allocated:
            require([x['request'] for x in selected]==reqs,'queue/job sequential atomic assignment');cursor+=len(reqs)
        if q['status']=='refused_unsupported_original_type':
            require(d['extension_type']!='progress' and all(x['status']=='refused_unsupported_original_type' for x in selected),'R refused not remapped')
        if q['status']=='stopped_by_original_child_policy':require(d['continue_expanding'] is False,'original child stop')
        if q['status']=='queued':require(d['extension_type']=='progress' and d['program_type'] in ('progress','composition'),'no executed R fallback')
        require(i<len(runtime['signals']),'every scheduled node has paid signal receipt')
        signal=runtime['signals'][i]
        queues.append(dict(index=i,parent_context_sha256=q['parent_context_sha256'],parent_depth=q['depth'],status=q['status'],
            decision=deepcopy(d),requested_width=len(reqs),allocated_jobs=len(selected),job_status_counts=counted(x['status'] for x in selected),
            returned_candidates=sum(x['status']=='returned' for x in selected),actual_new_node_depths=counted(x['node']['depth'] for x in selected if x['node'] is not None),
            signed_horizon_coverage_gain=signal['signed_horizon_coverage_gain'],minimum=deepcopy(q['minimum']),
            verification_reserve=deepcopy(q['verification_reserve']),remaining_before=deepcopy(q['remaining_before'])))
    require(all(x['status']=='not_requested' and x['request'] is None for x in jobs[cursor:]),'unallocated144 tail explicit')
    mappings=signal_identities(runtime);plan=runtime['model_plan'];available=plan is not None
    require(type(accepted['model_plan_available']) is bool and accepted['model_plan_available']==available,'actual strict plan availability')
    require(available==(row['status']=='callback_progress_complete_NOT_certified'),'refusal/cap/failure never release diagnostic best')
    require(accepted['realized_progress_depth']==runtime['realized_progress_depth'] and accepted['realized_widths']==runtime['realized_widths'],'strict realized tree dimensions')
    returned=[x for x in jobs if x['status']=='returned']
    require(runtime['realized_progress_depth']==max([x['node']['depth'] for x in returned],default=0) and
        runtime['realized_widths']==[len(q['requests']) for q in runtime['queues'] if q['status']=='queued'],'observed depth and allocated widths')
    resource_summary={k:deepcopy(v) for k,v in resources.items() if k!='events'}
    resource_summary.update(operation_event_count=len(resources['events']),
        operation_status_counts=counted(x['status'] for x in resources['events']),operation_type_counts=counted(x['resource'] for x in resources['events']))
    base.update(observed_job_rows=144,unobserved_expansion_slots=0,job_status_counts=counted(x['status'] for x in jobs),
        resources=resource_summary,plan_available=available,root_task_sha256=runtime['root_task_sha256'],root_binding=deepcopy(runtime['root_binding']),
        queue_nodes=queues,signal_parent_identities=mappings,realized_progress_depth=runtime['realized_progress_depth'],
        allocated_widths=deepcopy(runtime['realized_widths']),returned_node_depth_counts=counted(x['node']['depth'] for x in returned),
        returned_extension_H_counts=counted(len(x['payload']['actions'])//5 for x in returned),
        queue_horizon_counts=counted(q['decision']['horizon'] for q in runtime['queues']),
        mapped_signal_H_counts=counted(x['horizon'] for x in mappings),mapped_parent_contexts=len({x['parent_context_sha256'] for x in mappings}),
        mapped_child_contexts=len({x['parent_context_sha256'] for x in mappings if not x['is_original_root_context']}),
        J_API_attempted=resources['attempted']['jacobian_API'],J_API_completed=resources['completed']['jacobian_API'],
        signal_mappings_returned=len(mappings),refusal_nodes=sum(q['status']=='refused_unsupported_original_type' for q in runtime['queues']),
        cap_nodes=sum(q['status'] in CAPS for q in runtime['queues']),original_child_stops=sum(q['status']=='stopped_by_original_child_policy' for q in runtime['queues']),
        plan=None if plan is None else dict(depth=plan['depth'],indices=plan['indices'],raw_steps=len(plan['actions']),
            action_sha256=digest(plan['actions']),terminal_context_sha256=digest(plan['context']),original_model_Q=plan['cost']),
        diagnostic_best_available=runtime['diagnostic_best'] is not None,errors=deepcopy(runtime['errors']),
        native_calls_from_strict=accepted['native_calls'],complete_native_calls_from_strict=accepted['complete_native_calls'])
    return base


def summarize(report,receipt):
    keys(report,('schema','status','prepared','prepare_sha256','rows','errors','model_before','model_after','classes_before','classes_after',
        'precision','host','sources_verified','source_manifest_sha256','wall_seconds','CUDA_peak_bytes','journal_tail','artifact_sha256',*FLAGS))
    keys(receipt,('schema','status','prepared','prepare_sha256','report_sha256','denominator','arms','slot_statuses','historical_acquisition',
        'saved_verification','total_FLOPs',*FLAGS))
    require(report['schema']==PRODUCER+'.report' and receipt['schema']==PRODUCER+'.consume','own original producer/consume schemas')
    require(all(report[k] is False and receipt[k] is False for k in FLAGS),'all admission flags remain false')
    require(not report['errors'] and report['sources_verified'] is True,'only completed source-bound strict inputs')
    require(receipt['total_FLOPs'] is None and receipt['saved_verification']==
        'original queue/core/ridge recomputed; full native IO algebra; no neural parity recomputation','complete upstream consume contract')
    require(report['prepared']==receipt['prepared'] and report['prepare_sha256']==receipt['prepare_sha256'],'same literal prepare identity')
    require(receipt['denominator']==4 and len(report['rows'])==4,'fixed four-slot denominator')
    for row,identity in zip(report['rows'],roster(),strict=True):require(all(row[k]==identity[k] for k in identity),'original slot-major known2 methods')
    require(receipt['slot_statuses']==[x['status'] for x in report['rows']],'strict/report full slot statuses')
    accepted=receipt['arms'];ids=[x['slot'] for x in accepted]
    require(ids==[x['slot'] for x in report['rows'] if x['result'] is not None],'strict includes all and only dispatched arms')
    settled=all(x['result'] is not None and not x['result']['errors'] for x in report['rows'])
    require(report['status']==('four_slots_settled_not_admitted' if settled else 'partial_or_failed') and receipt['status']==
        ('saved_four_slot_engineering_verified' if settled else 'saved_partial_engineering_verified'),'strict completion vs partial status')
    arms=[summarize_arm(row,next((a for a in accepted if a['slot']==row['slot']),None)) for row in report['rows']]
    for root in (0,1):
        known=[x for x in arms if x['root']==root and x['root_task_sha256'] is not None]
        require(len({x['root_task_sha256'] for x in known})<=1 and len({digest(x['root_binding']) for x in known})<=1,'same original root authority across fixed/adaptive')
    complete_resources=all(a['resources'] is not None for a in arms)
    totals={field:dict(known_sum={k:sum(a['resources'][field][k] for a in arms if a['resources'] is not None) for k in AXES},
        all_four_total=None,observed_arms=sum(a['resources'] is not None for a in arms)) for field in ('attempted','completed')}
    if complete_resources:
        for value in totals.values():value['all_four_total']=deepcopy(value['known_sum'])
    return dict(schema=NAME+'.summary',status='saved_descriptive_summary_not_admitted',denominator=dict(arms=4,expansion_slots_per_arm=144,total_expansion_slots=576),
        arms=arms,status_counts=counted(x['status'] for x in arms),plan_available_count=sum(x['plan_available'] for x in arms),
        unobserved_resource_arms=4-totals['attempted']['observed_arms'],resource_totals=totals,
        historical_acquisition=deepcopy(receipt['historical_acquisition']),total_FLOPs=None,
        limitation='Uses externally SHA-bound completed consume; no producer, journal, SVD/choose replay, model, physics or fullstrict rerun',**FLAGS)


def build(report,report_sha,consume,consume_sha):
    for f,h in PINS.items():require(sha(f)==h,'frozen producer triple changed')
    rp,r=load(report,report_sha,256*1024**2);cp,c=load(consume,consume_sha,128*1024**2)
    require(c['report_sha256']==report_sha,'complete persisted consume binds actual report SHA')
    result=summarize(r,c)
    result['inputs']=dict(report=dict(path=str(rp),sha256=report_sha),consume=dict(path=str(cp),sha256=consume_sha),
        original_prepare=dict(path=r['prepared'],sha256=r['prepare_sha256'],read_or_rebuilt_here=False),
        producer_files={str(f):h for f,h in PINS.items()})
    result['summary_source_sha256']=sha(__file__)
    require(sha(rp)==report_sha and sha(cp)==consume_sha and all(sha(f)==h for f,h in PINS.items()),'saved inputs/source pins unchanged')
    return result


def write_new(path,value):
    p=Path(path).absolute();require(p.parent==ROOT/'results' and p.name.startswith(NAME) and p.resolve()==p and not p.exists() and not p.is_symlink(),'new exclusive summary output')
    data=canonical(value);require(len(data)<=32*1024**2 and shutil.disk_usage(p.parent).free>=200*1024**2+len(data),'summary cap/disk floor')
    with p.open('xb') as f:require(f.write(data)==len(data),'complete saved summary');f.flush();os.fsync(f.fileno())
    return dict(path=str(p),sha256=sha(p))


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=('analyze','verify'))
    ap.add_argument('--report',default=str(REPORT));ap.add_argument('--report-sha',default=REPORT_SHA)
    ap.add_argument('--consume',default=str(CONSUME));ap.add_argument('--consume-sha',default=CONSUME_SHA)
    ap.add_argument('--output',required=True);ap.add_argument('--summary-sha')
    a=ap.parse_args(argv);value=build(a.report,a.report_sha,a.consume,a.consume_sha)
    if a.command=='analyze':out=write_new(a.output,value)
    else:
        require(a.summary_sha is not None,'verify requires actual summary SHA')
        p,saved=load(a.output,a.summary_sha,32*1024**2);require(saved==value,'exact saved-only summary rebuild')
        out=dict(path=str(p),sha256=a.summary_sha,verified=True)
    print(canonical(out).decode())


if __name__=='__main__':main()
