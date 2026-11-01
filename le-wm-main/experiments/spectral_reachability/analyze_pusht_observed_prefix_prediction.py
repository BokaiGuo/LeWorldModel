"""Paired fixed-program forecast diagnostic; never a policy efficacy test."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.spectral_reachability.analyze_pusht_sealed_prediction_residuals import residual


def keyed(rows, key):
    result = {r[key]: r for r in rows}
    if not result or len(result) != len(rows):
        raise ValueError('unique nonempty identities required')
    return result


def analyze(observed, physical, compilation):
    for report, status in ((observed, 'completed'), (physical, 'completed'), (compilation, 'compiled_calibration')):
        if report['status'] != status or not report['gates'] or not all(v is True for v in report['gates'].values()):
            raise ValueError('complete passing evidence required')
    roots = [keyed(r['rows'], 'root_sha256') for r in (observed, physical, compilation)]
    if len(roots[0]) != 4 or not (set(roots[0]) == set(roots[1]) == set(roots[2])):
        raise ValueError('four identical root sets required')
    rows = []
    for root in sorted(roots[0]):
        actual = keyed(roots[1][root]['branches'], 'candidate_id')
        fresh = keyed(roots[0][root]['branches'], 'candidate_id')
        compiled = roots[2][root]['compiled']
        saved = keyed(compiled['receipts'], 'candidate_id')
        if len(actual) != 16 or not (set(actual) == set(fresh) == set(saved)):
            raise ValueError('complete identical program banks required')
        values = []
        for name in sorted(actual):
            branch = actual[name]
            if branch['status'] != 'executed' or len(branch['stages']) != 2 or fresh[name]['prefix_replay_exact'] is not True:
                raise ValueError('successful exact prefix replay and terminal execution required')
            target, projection = compiled['goal_vectors'][0], compiled['scoring']['metric']['projection']
            terminal = branch['stages'][-1]['vector']
            old = residual(saved[name]['predicted_terminal_vector'], terminal, target, projection)
            new = residual(fresh[name]['observed_prefix_terminal_prediction'], terminal, target, projection)
            values.append({'candidate_id': name, 'imagined_prefix': old, 'observed_prefix': new})
        old = [v['imagined_prefix']['projected_vector_error_mse'] for v in values]
        new = [v['observed_prefix']['projected_vector_error_mse'] for v in values]
        rows.append({'root_sha256': root, 'root_index': roots[0][root]['root_index'], 'candidates': values,
                     'imagined_median_error_mse': float(np.median(old)), 'observed_median_error_mse': float(np.median(new)),
                     'improved_candidates': int(np.sum(np.asarray(new) < old))})
    return {'schema': 'pusht_observed_prefix_analysis.v1', 'rows': rows,
            'improved_candidates': sum(r['improved_candidates'] for r in rows), 'total_candidates': 64,
            'scope': 'known-root fixed-suffix prediction diagnostic; physical outcomes unchanged',
            'policy_changed': False, 'P1_admission': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('observed', 'physical', 'compilation', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    observed, physical, compiled = (json.loads(path.read_text()) for path in (args.observed, args.physical, args.compilation))
    if digest(args.compilation) != physical['protocol']['preflight_sha256']:
        raise ValueError('physical/compilation binding changed')
    if observed['protocol']['bindings'].get(str(args.physical.resolve())) != digest(args.physical) or observed['protocol']['bindings'].get(str(args.compilation.resolve())) != digest(args.compilation):
        raise ValueError('observed input binding changed')
    result = analyze(observed, physical, compiled)
    result['inputs'] = {str(path.resolve()): digest(path) for path in (args.observed, args.physical, args.compilation)}
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(f'{result["improved_candidates"]}/64 fixed-program prediction errors decreased; no policy efficacy claim')
