"""Decompose saved local reach scores; no neural work or causal claims."""
import argparse
import json
from pathlib import Path
import numpy as np


def analyze(directory):
    directory=Path(directory)
    report=json.loads((directory/'report.json').read_text())
    if report['status']!='calibration_completed' or not all(v is True for v in report['gates'].values()):
        raise ValueError('completed calibration required')
    rows=[]
    for physical in report['rows']:
        planpath=directory/f'plan_{physical["root_index"]}.json'
        from experiments.spectral_reachability.run_pusht_d2p_budget_guard_cuda import digest
        if digest(planpath)!=report['plan_hashes'][str(planpath.resolve())]:
            raise ValueError('plan binding changed')
        plan=json.loads(planpath.read_text()); c=plan['compiled']; s=c['scoring']
        if plan['root_sha256']!=physical['root_sha256']:
            raise ValueError('root mismatch')
        outcomes={b['candidate_id']:b for b in physical['branches']}
        if len(outcomes)!=10 or set(outcomes)!={r['candidate_id'] for r in c['receipts']}:
            raise ValueError('all candidates required')
        details=[]
        for score,receipt in zip(s['rows'],c['receipts'],strict=True):
            if score['candidate_id']!=receipt['candidate_id']:
                raise ValueError('scoring identity mismatch')
            residual=np.asarray(score['own_projected_residual']); nominal=float(residual@residual/len(residual))
            gain=nominal-score['own_reach_value']
            nonlinear=receipt['nonlinear_terminal_cost_diagnostic_only']
            physical_outcome=outcomes[score['candidate_id']]
            details.append(dict(candidate_id=score['candidate_id'],stage_cost=score['stage_cost'],
                nominal_suffix_goal_cost=nominal,regularized_reach_value=score['own_reach_value'],
                linear_action_improvement= gain,
                fraction_nominal_removed=gain/nominal if nominal else None,
                nonlinear_terminal_cost=nonlinear,
                nonlinear_minus_linear_error=nonlinear-score['linear_terminal_cost'],
                suffix_delta_norm=float(np.linalg.norm(score['suffix_delta_params'])),
                physical_status=physical_outcome['status'],
                physical_pose_cost=physical_outcome['pose']['pose_cost'],
                official_success=physical_outcome['official']['official_success']))
        zero=next(d for d in details if d['candidate_id']=='prefix_0')
        selected=next(d for d in details if d['candidate_id']==s['selected_candidate_id'])
        # Exact decomposition: V_zero - V_selected = baseline_residual_drop
        # + (action_improvement_selected - action_improvement_zero).
        baseline_drop=zero['nominal_suffix_goal_cost']-selected['nominal_suffix_goal_cost']
        action_gain=selected['linear_action_improvement']-zero['linear_action_improvement']
        total=zero['regularized_reach_value']-selected['regularized_reach_value']
        if not np.isclose(total,baseline_drop+action_gain,atol=1e-12,rtol=1e-12):
            raise ValueError('reach decomposition failed')
        rows.append(dict(root_index=physical['root_index'],root_sha256=physical['root_sha256'],
            selected_candidate_id=selected['candidate_id'],selected_equals_greedy=selected['candidate_id']==s['greedy_candidate_id'],
            reach_reduction=total,nominal_goal_residual_component=baseline_drop,
            incremental_linear_action_component=action_gain,
            selected_vs_zero_physical_pose_delta=selected['physical_pose_cost']-zero['physical_pose_cost'],
            details=details))
    return dict(schema='pusht_spectral_bank_decomposition.v1',scope='known calibration roots; arithmetic only',
        model_inference=False,physical_outcome_used_to_select=False,P1_admission=False,
        rows=rows,all_selected_greedy=all(r['selected_equals_greedy'] for r in rows),
        warnings=['reach reduction is not by itself increased controllability',
                  'nominal residual and response effects are jointly changed; this is arithmetic not causal identification',
                  'four known calibration roots do not establish generalization or theory failure'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=analyze(args.directory)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    for r in result['rows']:
        print(r['root_index'], 'V reduction',r['reach_reduction'],'nominal component',r['nominal_goal_residual_component'],'action component',r['incremental_linear_action_component'])
