"""Saved conditional-compiler mechanism audit, not physical P1 admission.

No torch or simulator import. A changed selected index is not by itself a
non-greedy witness: the selected prefix must have strictly larger stage cost.
Across-Q scalar costs are deliberately never subtracted.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import p3_goal_patch_consistent_compiler_20261027_v1 as compiler

NAME = 'p1_conditional_mechanism_arrays_main_20261027_v1'
TOL = 1e-12  # numerical comparison only; not a task success criterion


def describe_arm(packet, q, compilation):
    winner = compiler.validate_compilation(packet, q, compilation)
    candidates = compilation['candidates']
    greedy = min(candidates, key=lambda c: (c['stage_cost'], c['index']))
    zero = candidates[0]
    stage_gap = winner['stage_cost'] - greedy['stage_cost']
    reach_gap = winner['own_reach_value'] - greedy['own_reach_value']
    nonlinear_gap = winner['nonlinear_terminal_cost'] - greedy['nonlinear_terminal_cost']
    # A lower linearized value can be optimistic, despite an exact local solve.
    def optimism(c):
        return c['nonlinear_terminal_cost'] - c['suffix_solve']['linear_terminal_cost']
    rows = []
    for c in candidates:
        suffix_j = np.asarray(c['jacobian'], dtype=float)
        rows.append(dict(index=c['index'], proposal_kind=c['proposal_kind'],
            stage_cost=c['stage_cost'], own_reach_value=c['own_reach_value'],
            objective=c['objective'], nonlinear_terminal_cost=c['nonlinear_terminal_cost'],
            linearization_gap=optimism(c),
            projected_suffix_jacobian_frobenius=float(np.linalg.norm(np.asarray(q) @ suffix_j)),
            suffix_singular_values=c['suffix_solve']['singular_values'],
            commands_sha256=compiler.digest(c['commands'])))
    return dict(selected_index=winner['index'], greedy_index=greedy['index'],
        selected_commands_sha256=compiler.digest(winner['commands']),
        greedy_commands_sha256=compiler.digest(greedy['commands']),
        selected_minus_greedy_stage=stage_gap, selected_minus_greedy_reach=reach_gap,
        selected_minus_greedy_objective=winner['objective']-greedy['objective'],
        selected_minus_greedy_nonlinear_terminal=nonlinear_gap,
        selected_minus_zero_prefix_reach=winner['own_reach_value']-zero['own_reach_value'],
        strict_model_non_greedy_prefix=bool(stage_gap > TOL),
        model_non_greedy_with_lower_reach=bool(stage_gap > TOL and reach_gap < -TOL),
        selected_linearization_gap=optimism(winner), greedy_linearization_gap=optimism(greedy),
        exact_saved_model_replay=compilation['whole_and_segmented_exact'],
        candidates=rows, actual_capability_change_measured=False,
        actual_task_gain_measured=False, P1_admission=False)


def audit(cuda_output, expected_sha, output):
    cuda_output=Path(cuda_output).absolute(); output=Path(output).absolute()
    compiler.require(not output.exists(), 'exclusive new saved-data audit output')
    raw=(cuda_output/'report.json').read_bytes()
    compiler.require(compiler.io.sha(raw)==expected_sha, 'explicit CUDA report bytes')
    plans=compiler.selected_actions(cuda_output)  # full eight-arm witness verification
    binding=compiler.io.decode((cuda_output/'execution_protocol.json').read_bytes())
    body, identity=compiler.validate_prepared(binding['prepared'], binding['prepare_sha256'])
    report=compiler.io.decode(raw)
    index=compiler.io.index(report['rows'], ('root_sha256','arm'))
    rows=[]
    for packet in body['packets']:
        for arm in compiler.ARMS:
            row=index[packet['root_sha256'],arm]
            description=describe_arm(packet,packet['projections'][arm],row['compilation'])
            rows.append(dict(root_sha256=packet['root_sha256'], arm=arm,
                active_metric_id=packet['active_metric_ids'][arm], **description))
    plan_index=compiler.io.index(plans,('root_sha256','arm'))
    compiler.require(len(rows)==8 and all(r['selected_commands_sha256']==
        compiler.digest(plan_index[r['root_sha256'],r['arm']]['actions']) for r in rows),
        'mechanism rows match exact exported physical actions')
    pairs=[]
    for root in body['roots']:
        pair=[plan_index[root,arm] for arm in compiler.ARMS]
        pairs.append(dict(root_sha256=root,
            selected_raw_actions_exactly_equal=bool(np.array_equal(pair[0]['actions'],pair[1]['actions'])),
            same_horizon=pair[0]['horizon']==pair[1]['horizon']==2,
            cross_Q_cost_difference=None,
            reason='different task metrics: scalar costs are not a shared outcome'))
    result=dict(schema=NAME+'.report', status='saved_model_mechanism_recorded',
        source_report=str(cuda_output/'report.json'),source_report_sha256=expected_sha,
        source_protocol_id=identity, auditor_source_sha256=compiler.io.sha(Path(__file__).read_bytes()),
        rows=rows, pairs=pairs, numerical_tolerance=TOL,
        new_neural_calls=0,new_physical_steps=0,
        saved_array_svds=192, # selected_actions96 + describe_arm8*12; no neural derivatives
        description='known calibration roots only; predicted mechanism, not physical causal evidence',
        P1_admission=False,P2_admission=False,P3_admission=False)
    output.mkdir(exist_ok=False); compiler.write(output/'report.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda-output',type=Path,required=True)
    parser.add_argument('--report-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); r=audit(args.cuda_output,args.report_sha256,args.output)
    print(compiler.io.canonical({k:v for k,v in r.items() if k not in ('rows','pairs')}))


if __name__=='__main__': main()
