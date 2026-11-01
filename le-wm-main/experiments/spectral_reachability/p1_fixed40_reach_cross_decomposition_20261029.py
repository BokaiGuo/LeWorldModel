"""Cross fixed residuals and Jacobians to describe the source of reach-score changes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from experiments.spectral_reachability import p1_fixed40_conditional_reach_consume_20261029 as consumer
from experiments.spectral_reachability import agent_p1_joint_vreach_saved_diff_20261029_v1 as solver

ROOT = consumer.host.ROOT
SOURCE = ROOT/'results/p1_fixed40_conditional_reach_host_20261029_MAIN01.json'
SOURCE_SHA = '9281d3353c8a36bcf7d836614a3142afbc73321064efb68edd52ce2533e9439c'


def split(j0, r0, j1, r1):
    # Row is Jacobian (zero/S), column residual (zero/S).
    table = []
    for j in (j0, j1):
        row = []
        for r in (r0, r1):
            x, _ = solver.independent_solve(j, r, .002, np.sqrt(5)*.75)
            row.append(solver.compare(j, r, x, .002)['score_raw'])
        table.append(row)
    a, b = table
    jacobian_effect = .5*((b[0]-a[0])+(b[1]-a[1]))
    residual_effect = .5*((a[1]-a[0])+(b[1]-b[0]))
    return dict(cross_score=table, jacobian_effect=jacobian_effect, residual_effect=residual_effect,
                total_difference=b[1]-a[0], identity_error=(b[1]-a[0])-jacobian_effect-residual_effect)


def build():
    consumer.verify(SOURCE, SOURCE_SHA)
    report = json.loads(SOURCE.read_bytes())
    prep = consumer.host.prefix.history.rgb.read(consumer.host.prefix.history.TEACHER_PREP, consumer.host.prefix.history.TEACHER_PREP_SHA)
    pairs = []
    for i in range(0, 8, 2):
        a, b = report['rows'][i:i+2]
        if a['task_id'] != b['task_id'] or prep['anchors'][i]['goal'] != prep['anchors'][i+1]['goal']:
            raise ValueError('same-task/goal pair required')
        goal = np.asarray(prep['anchors'][i]['goal'], dtype=np.float64)
        cases = {}
        for name in ('predicted', 'observed'):
            x, y = a['cases'][name], b['cases'][name]
            cases[name] = split(np.asarray(x['jacobian']), goal-np.asarray(x['nominal']),
                                np.asarray(y['jacobian']), goal-np.asarray(y['nominal']))
        pairs.append(dict(anchor_indices=[i,i+1], task_id=a['task_id'], cases=cases))
    return dict(schema='p1_fixed40_reach_cross_decomposition_20261029', source_sha256=SOURCE_SHA,
                pairs=pairs, known_roots=2, physical_calls=0, neural_calls=0, P1_admission=False,
                scope='retrospective algebraic cross-substitution, not physical intervention or causal identification')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not args.output.is_absolute() or args.output.parent != ROOT/'results' or args.output.exists():
        raise ValueError('exclusive absolute results file required')
    print('Cross fixed Jacobians and residuals; this is algebraic, not a physical intervention.', flush=True)
    data = json.dumps(build(), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    with args.output.open('xb') as f:
        f.write(data)
    print(json.dumps(dict(path=str(args.output), sha256=hashlib.sha256(data).hexdigest())))


if __name__ == '__main__':
    main()
