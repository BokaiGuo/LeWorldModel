"""Post-execution diagnostic only: no selection, training, or P1 admission."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else None


def analyze(execution, compilation):
    if execution['status'] != 'completed' or compilation['status'] != 'compiled_calibration':
        raise ValueError('completed real execution and compilation required')
    for report in (execution, compilation):
        if not report['gates'] or not all(v is True for v in report['gates'].values()):
            raise ValueError('integrity gates required')
    def keyed(rows):
        result = {r['root_sha256']: r for r in rows}
        if len(result) != len(rows) or not result:
            raise ValueError('unique nonempty root identities required')
        return result
    physical, model = keyed(execution['rows']), keyed(compilation['rows'])
    if set(physical) != set(model):
        raise ValueError('root sets differ')
    result = []
    for root in sorted(physical):
        row, compiled = physical[root], model[root]['compiled']
        receipts = {r['candidate_id']: r for r in compiled['receipts']}
        branches = {r['candidate_id']: r for r in row['branches']}
        if len(branches) != 16 or len(receipts) != 16 or set(branches) != set(receipts):
            raise ValueError('complete unique paired candidate bank required')
        goal = np.asarray(compiled['goal_vectors'][0], float)
        projection = np.asarray(compiled['scoring']['metric']['projection'], float)
        values = []
        for name in receipts:
            branch, receipt = branches[name], receipts[name]
            if branch['status'] != 'executed' or len(branch['stages']) != 2:
                raise ValueError('all two-token executions required')
            stages = branch['stages']
            metric = []
            for stage in stages:
                residual = np.asarray(stage['vector'], float)-goal
                if residual.shape != goal.shape or not np.isfinite(residual).all():
                    raise ValueError('finite aligned vector required')
                raw = float(np.mean(residual**2))
                projected = float(np.mean((projection@residual)**2))
                if not np.isclose(projected, stage['metric_cost'], atol=1e-9, rtol=1e-7):
                    raise ValueError('saved metric cost mismatch')
                removed = float(np.mean(((np.eye(len(goal))-projection)@residual)**2))
                if not np.isclose(raw, projected+removed, atol=1e-8, rtol=1e-6):
                    raise ValueError('orthogonal energy decomposition failed')
                metric.append({'raw': raw, 'projected': projected, 'removed': removed,
                               'pose': stage['pose']['pose_cost']})
            if not np.allclose(branch['commands'], receipt['program_live_commands'], atol=0, rtol=0):
                raise ValueError('execution commands differ from sealed commands')
            values.append({'candidate_id': name, 'stages': metric})
        summary = {}
        for index, label in enumerate(('prefix', 'terminal')):
            pose = [v['stages'][index]['pose'] for v in values]
            summary[label] = {key+'_pose_correlation': correlation([v['stages'][index][key] for v in values], pose)
                              for key in ('raw', 'projected', 'removed')}
        result.append({'root_sha256': root, 'root_index': row['root_index'], 'correlations': summary, 'candidates': values})
    return {'schema': 'pusht_sealed_metric_alignment.v1', 'status': 'diagnostic_completed',
            'scope': 'known-root paired metric diagnostic; not causal attribution or efficacy',
            'selection_changed': False, 'P1_admission': False, 'rows': result}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execution', type=Path, required=True)
    p.add_argument('--compilation', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    execution = json.loads(args.execution.read_text())
    if digest(args.compilation) != execution['protocol']['preflight_sha256']:
        raise ValueError('compilation digest mismatch')
    result = analyze(execution, json.loads(args.compilation.read_text()))
    result['inputs'] = {str(path.resolve()): digest(path) for path in (args.execution, args.compilation)}
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps([{'root': r['root_index'], **r['correlations']} for r in result['rows']], indent=2))
