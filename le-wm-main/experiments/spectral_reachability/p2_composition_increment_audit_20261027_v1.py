"""Saved-data fail-closed P2 increment audit. Pure core is NumPy/stdlib.

Exit zero means an audit was emitted, NEVER P2 admission. No threshold tuning,
neural/physics execution or prospective evidence support exists in this version.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np

NAME='p2_composition_increment_audit_20261027_v1'
PROJECT=Path(__file__).resolve().parents[2]
ARMS=('joint_h2_no_composition','conditional_suffix_composition')
PHYSICAL='results/p2_composition_paired_physical_20261027_v2_main01'
PLANNING='results/p2_composition_paired_20261027_v2_prepared01'
P1='results/pusht_d2p_p1_outcome_physical_20261024_cuda/report.json'
METRICS=('predicted_q_mse','real_encoder_q_mse','object_pose_cost','official_state_distance')
COUNTERS=('predict_calls','predict_samples','action_encoder_calls','action_encoder_samples','predict_context_tokens','encoder_context_tokens','failed_entries')


def require(ok,message):
    if not ok:raise ValueError(message)


def keyed(rows,fields):
    require(isinstance(rows,list),'explicit list census');out={}
    for r in rows:
        k=tuple(r[f] for f in fields);require(k not in out,'duplicate identity: '+str(k));out[k]=r
    return out


def finite(x):
    require(type(x) in (int,float) and np.isfinite(x),'finite scalar, not bool');return float(x)


def canonical(x):return json.dumps(x,sort_keys=True,allow_nan=False,separators=(',',':'))


def direction(predicted,observed):
    if observed is None:return 'unavailable'
    if predicted==0 or observed==0:return 'tie_in_either'
    return 'strict_reversal' if (predicted<0)!=(observed<0) else 'same_direction'


def budget_pair(left,right,gaps):
    """Missing is unknown; inconsistent supplied arithmetic is an integrity error."""
    result=dict(complete_neural_receipts=False,equal_predict_budget=None,equal_action_encoder_budget=None)
    for arm,r in zip(ARMS,(left,right)):
        for scope in ('shared','exclusive','paired_unique','standalone_attributed'):
            for key in COUNTERS:
                value=r.get(scope,{}).get(key)
                if value is None:gaps.append('budget_missing:'+arm+':'+scope+':'+key)
                else:require(type(value) is int and value>=0,'nonnegative integer budget: '+key)
    complete=not any(g.startswith('budget_missing:') for g in gaps)
    if not complete:return result
    for key in COUNTERS:
        require(left['shared'][key]==right['shared'][key] and left['paired_unique'][key]==right['paired_unique'][key],'shared/unique receipt mismatch')
        require(left['paired_unique'][key]==left['shared'][key]+left['exclusive'][key]+right['exclusive'][key],'unique budget double-count/missing shared')
        for r in (left,right):require(r['standalone_attributed'][key]==r['shared'][key]+r['exclusive'][key],'standalone attribution arithmetic')
    for r in (left,right):
        for scope in ('shared','exclusive','paired_unique','standalone_attributed'):
            c=r[scope];require(c['predict_calls']==c['predict_samples']==c['action_encoder_calls']==c['action_encoder_samples'],'single-sample predict/action encoder accounting')
            require(c['predict_context_tokens']==3*c['predict_calls'] and c['encoder_context_tokens']==3*c['action_encoder_calls'],'full three-context token accounting')
    result.update(complete_neural_receipts=True,equal_predict_budget=left['standalone_attributed']['predict_samples']==right['standalone_attributed']['predict_samples'],
        equal_action_encoder_budget=left['standalone_attributed']['action_encoder_samples']==right['standalone_attributed']['action_encoder_samples'],
        standalone_predict=[left['standalone_attributed']['predict_samples'],right['standalone_attributed']['predict_samples']],
        shared=deepcopy(left['shared']),paired_unique=deepcopy(left['paired_unique']))
    return result


def audit(outcomes,planner,plans,p1=None):
    """Pure core; upstream loader independently rebuilds actions/Q/pose/journals."""
    rows=keyed(outcomes['rows'],('root_sha256','arm'));ps=keyed(plans,('root_sha256','method'));pr=keyed(planner['rows'],('root_sha256',))
    roots=sorted(k[0] for k in pr)
    require(len(roots)==4 and set(rows)==set(ps)=={(r,a) for r in roots for a in ARMS},'strict four-root eight-arm identity census')
    require({pr[r,]['root_index'] for r in roots}==set(range(4)),'unique stable root indices')
    for source in (outcomes,planner):
        require(source.get('P1_admission') is False and source.get('P2_admission') is False,'unsupported/missing scientific source admission boundary')
    gaps=['original_P1_independent_mechanism_not_admitted','prospective_independent_cohort_missing',
        'strict_total_resource_pairing_missing','original_controls_zero_prefix_greedy_CEM_missing',
        'all_candidate_physical_coverage_missing','independent_increment_uncertainty_multiplicity_missing',
        'same_proposal_distribution_not_established','saved_derivative_truth_not_independently_validated']
    pairs=[];totals={k:0 for k in COUNTERS};operations={k:0 for k in ('jacobian_calls','scalar_output_vjp_directions','svd_calls','bounded_solve_calls')};unknown_operations=set()
    for root in sorted(roots,key=lambda r:pr[r,]['root_index']):
        local=[];a,b=(rows[root,arm] for arm in ARMS);pa,pb=(ps[root,arm] for arm in ARMS);record=pr[root,]
        require(a['root_index']==b['root_index']==pa['root_index']==pb['root_index']==record['root_index'],'root index mismatch')
        for key in ('root','action_domain','goal_vector_sha256','projection_sha256'):
            require(pa[key]==pb[key],'different control input: '+key)
        require(a['goal']==b['goal'] and a['initial_state']==b['initial_state'],'actual reset/goal mismatch')
        horizons=[p['planned_horizon_tokens'] for p in (pa,pb)];commands=[p['planned_commands'] for p in (pa,pb)]
        for p,r in ((pa,a),(pb,b)):
            require(type(p['planned_horizon_tokens']) is int and p['planned_horizon_tokens'] in (2,3) and p['planned_commands']==5*p['planned_horizon_tokens'],'explicit horizon/action contract')
            raw=np.asarray(p['commands']);require(raw.shape==(p['planned_commands'],2) and raw.dtype.kind in 'fi' and np.isfinite(raw).all() and np.all(np.abs(raw)<=1),'all bounded commands')
            require(p['selected_candidate_id']==r['selected_candidate_id'] and p['receipt_sha256']==r['receipt_sha256'],'selected receipt identity')
            require(r.get('planning_accounting')==p.get('planning_accounting'),'outcome/plan budget drift')
            require(type(r['full_H2']) is bool,'explicit completion flag')
            if r['full_H2']:require(p['planned_horizon_tokens']==2 and r['returned_commands']==r['observed_commands']==10 and r['status']=='completed','full H2 cannot hide partial/mixed endpoint')
        rr=record.get('resource_receipt',{});arms=keyed(record['results'],('arm',))
        require(set(arms)=={(arm,) for arm in ARMS},'planner arm census')
        for arm,p,r in zip(ARMS,(pa,pb),(a,b)):
            candidates=arms[arm,]['attempts'];require(len(candidates)==40 and [c['index'] for c in candidates]==list(range(40)) and all(c['completed'] is True for c in candidates),'complete ordered40 candidate census')
            winner=min(candidates,key=lambda c:finite(c['cost']))
            require(winner['candidate_id']==p['selected_candidate_id'],'stable nonlinear winner')
            finite(r['predicted_q_mse']);require(np.isclose(r['predicted_q_mse'],winner['cost'],rtol=0,atol=1e-10),'predicted Q is selected nonlinear cost')
        ca,cb=(keyed(arms[arm,]['attempts'],('index',)) for arm in ARMS)
        same_prefix=all(ca[i,]['actions'][:5]==cb[i,]['actions'][:5] for i in range(40))
        require(same_prefix,'candidate-wise common prefix control violated')
        budget=budget_pair(a.get('planning_accounting',{}),b.get('planning_accounting',{}),local)
        if budget['complete_neural_receipts']:
            require(rr.get('unique')==budget['paired_unique'] and rr.get('shared')==budget['shared'],'planner resource receipt binding')
            for arm,r in zip(ARMS,(a,b)):
                for key in ('exclusive','standalone_attributed'):require(rr.get(key,{}).get(arm)==r['planning_accounting'][key],'arm resource binding')
            for k in COUNTERS:totals[k]+=budget['paired_unique'][k]
        for k in operations:
            value=rr.get('completed_operations',{}).get(k)
            if value is None:local.append('operation_budget_missing:'+k);unknown_operations.add(k)
            else:require(type(value) is int and value>=0,'operation count');operations[k]+=value
        if not budget['equal_predict_budget']:local.append('predict_budget_unmatched_or_unknown')
        full=a['full_H2'] and b['full_H2'] and horizons==[2,2] and commands==[10,10]
        delta=None
        if full:
            delta={k:None if a['full_H2_metrics'].get(k) is None or b['full_H2_metrics'].get(k) is None else finite(b['full_H2_metrics'][k])-finite(a['full_H2_metrics'][k]) for k in METRICS}
            if delta['real_encoder_q_mse'] is None:local.append('actual_terminal_Q_missing')
        else:local.append('full_same_H2_physical_pair_missing')
        gaps.extend(root+':'+g for g in local)
        pairs.append(dict(root_sha256=root,root_index=record['root_index'],selected_horizons=horizons,equal_horizon=horizons[0]==horizons[1],
            equal_selected_control_steps=commands[0]==commands[1],same_inputs=True,same_candidate_prefixes=same_prefix,full_same_H2=full,
            composition_minus_no_composition=delta,
            prediction_directions=None if delta is None else {k:direction(delta['predicted_q_mse'],delta[k]) for k in METRICS[1:]},
            observed_partials={arm:deepcopy(r['observed_endpoint_metrics']) for arm,r in zip(ARMS,(a,b)) if not r['full_H2']},budget=budget,missing=local))
    if all(p['budget']['complete_neural_receipts'] for p in pairs):
        for key,k in (('actual_unique_predict_calls','predict_calls'),('actual_unique_action_encoder_calls','action_encoder_calls')):
            if planner.get(key) is None or outcomes.get('planning_totals',{}).get(key) is None:gaps.append('global_neural_budget_missing:'+key)
            else:require(planner[key]==totals[k]==outcomes['planning_totals'][key],'unique neural totals')
    for report_key,k in (('completed_jacobian_calls','jacobian_calls'),('completed_scalar_output_vjp_directions','scalar_output_vjp_directions'),('completed_svd_calls','svd_calls')):
        if planner.get(report_key) is None or k in unknown_operations:gaps.append('global_operation_budget_missing:'+report_key)
        else:require(planner[report_key]==operations[k],'global operation accounting')
    budget_unknown=['FLOPs','backward_engine_API_or_kernel_calls','upstream_RGB_context_Q_cost','common_enforced_wall_time_cap','optimizer_and_expansion_strict_schema','all_candidate_physics_and_encoding']
    return dict(schema=NAME+'.report',status='saved_data_audited_P2_blocked',integrity_verified=True,pairs=pairs,
        known_unique_neural_cost=totals if all(p['budget']['complete_neural_receipts'] for p in pairs) else None,
        known_completed_operations={k:None if k in unknown_operations else v for k,v in operations.items()},unknown_or_unmatched_budget=budget_unknown,
        additional_verification_work=outcomes.get('additional_verification_work'),
        original_P1=dict(status=None if p1 is None else p1.get('status'),gates=None if p1 is None else p1.get('gates'),
            admission_revalidated=False,reason='historical failure/closed original mechanism; model calibration or new P2 execution cannot substitute independent original P1'),
        prerequisite_gaps=gaps,full_same_H2_pairs=sum(p['full_same_H2'] for p in pairs),
        equal_predict_budget=all(p['budget']['equal_predict_budget'] is True for p in pairs),equal_total_compute=False,
        scientific_increment_established=False,P1_admission=False,P2_admission=False,physical_efficacy_verified=False,
        cohort='four already-used calibration roots; no population inference',new_neural_calls=0,new_physical_steps=0)


def run(project_root=PROJECT):
    """Bound loader reuses sealed exporter arithmetic, never invokes its CUDA API."""
    root=Path(project_root).resolve()
    from experiments.spectral_reachability import p2_composition_paired_outcomes_20261027_v2 as saved
    reader=saved.evidence.EvidenceReader()
    try:
        outcomes=saved.load_and_compute(root/PHYSICAL,reader)
        execution=reader.load(root/PHYSICAL/'protocol.json');protocol=reader.load(Path(execution['prepared'])/'protocol.json')
        planner=reader.load(root/PLANNING/'report.json')
        require(Path(protocol['planning'])==root/PLANNING and protocol['bindings'].get(str(root/PLANNING/'report.json'))==saved.evidence.sha(root/PLANNING/'report.json'),'exact actual planner report')
        p1=reader.load(root/P1) if (root/P1).is_file() else None
        result=audit(outcomes,planner,protocol['plans'],p1)
        reader.unchanged()
    except (ValueError,KeyError,TypeError,OSError,IndexError) as exc:
        result=dict(schema=NAME+'.report',status='refused_missing_or_inconsistent_evidence',integrity_verified=False,
            failure=dict(type=type(exc).__name__,message=str(exc)),P1_admission=False,P2_admission=False,
            scientific_increment_established=False,new_neural_calls=0,new_physical_steps=0)
    result['input_bindings']=reader.hashes
    result['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['audit_source_bindings']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
        (Path(__file__).resolve(),PROJECT/'tests'/('test_'+NAME+'.py'),PROJECT/'docs'/(NAME+'.md'))}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--project-root',type=Path,default=PROJECT)
    parser.add_argument('--output',type=Path,help='optional NEW small JSON file; default stdout');a=parser.parse_args();r=run(a.project_root)
    text=json.dumps(r,sort_keys=True,indent=2,allow_nan=False)+'\n'
    if a.output:
        require(a.output.parent.is_dir(),'existing output parent required; no bulk preparations')
        with a.output.open('x') as stream:stream.write(text)
        print(canonical(dict(status=r['status'],output=str(a.output),P2_admission=False)))
    else:print(text,end='')


if __name__=='__main__':main()
