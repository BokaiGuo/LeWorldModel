"""Compare saved CUDA predictions to measured CUDA encodings; diagnostic only."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def residual(vector, actual, goal, projection):
    vector, actual, goal, projection = map(lambda x: np.asarray(x, float), (vector, actual, goal, projection))
    if vector.shape != goal.shape or actual.shape != goal.shape or projection.shape != (goal.size, goal.size):
        raise ValueError('aligned prediction, actual, goal and projection required')
    if not all(np.isfinite(x).all() for x in (vector, actual, goal, projection)):
        raise ValueError('finite inputs required')
    predicted_cost = float(np.mean((projection@(vector-goal))**2))
    actual_cost = float(np.mean((projection@(actual-goal))**2))
    error = float(np.mean((projection@(actual-vector))**2))
    # Exact quadratic decomposition. Cross term can be negative.
    cross = float(2*np.mean((projection@(vector-goal))*(projection@(actual-vector))))
    if not np.isclose(actual_cost, predicted_cost+error+cross, atol=1e-9, rtol=1e-7):
        raise ValueError('quadratic residual decomposition failed')
    return {'predicted_cost': predicted_cost, 'actual_cost': actual_cost,
            'projected_vector_error_mse': error, 'cross_term': cross,
            'cost_prediction_error': actual_cost-predicted_cost}


def analyze(execution, compilation):
    if execution['status'] != 'completed' or compilation['status'] != 'compiled_calibration':
        raise ValueError('completed evidence required')
    for r in (execution, compilation):
        if not r['gates'] or not all(v is True for v in r['gates'].values()):
            raise ValueError('integrity gates required')
    def index(rows, key):
        result = {r[key]: r for r in rows}
        if len(result) != len(rows) or not result:
            raise ValueError('unique nonempty identities required')
        return result
    e, c = index(execution['rows'], 'root_sha256'), index(compilation['rows'], 'root_sha256')
    if set(e) != set(c):
        raise ValueError('root set mismatch')
    rows = []
    for root in sorted(e):
        compiled = c[root]['compiled']
        receipts = index(compiled['receipts'], 'candidate_id')
        branches = index(e[root]['branches'], 'candidate_id')
        if len(receipts) != 16 or set(receipts) != set(branches):
            raise ValueError('complete paired bank required')
        projection, goal = compiled['scoring']['metric']['projection'], compiled['goal_vectors'][0]
        values = []
        for name, receipt in receipts.items():
            branch = branches[name]
            if branch['status'] != 'executed' or len(branch['stages']) != 2:
                raise ValueError('both physical stages required')
            stages = [residual(receipt[field], measured['vector'], goal, projection)
                      for field, measured in zip(('prefix_vector', 'predicted_terminal_vector'), branch['stages'], strict=True)]
            values.append({'candidate_id': name, 'prefix': stages[0], 'terminal': stages[1]})
        rows.append({'root_sha256': root, 'root_index': e[root]['root_index'], 'candidates': values,
                     'selected_candidate_id': compiled['selected_candidate_id']})
    return {'schema': 'pusht_sealed_prediction_residuals.v1', 'rows': rows,
            'scope': 'known calibration fixed-program prediction diagnostic; not a selector or causal proof',
            'P1_admission': False, 'selection_changed': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('execution', 'compilation', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    execution = json.loads(args.execution.read_text())
    if digest(args.compilation) != execution['protocol']['preflight_sha256']:
        raise ValueError('compilation digest mismatch')
    report = analyze(execution, json.loads(args.compilation.read_text()))
    report['inputs'] = {str(path.resolve()): digest(path) for path in (args.execution, args.compilation)}
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    for row in report['rows']:
        print(json.dumps({'root': row['root_index'], **{stage+'_median_vector_error_mse': float(np.median([v[stage]['projected_vector_error_mse'] for v in row['candidates']])) for stage in ('prefix', 'terminal')}}))
