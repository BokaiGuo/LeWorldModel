"""Independent saved FD arithmetic; no compiler, model or physics imports."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

NAME = 'p5_native_fd_array_audit_main_20261027_v1'
D = 98304


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    path = Path(path).absolute()
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'regular nonsymlink file required')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            require(key not in out, 'duplicate JSON key')
            out[key] = value
        return out
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique,
                      parse_constant=lambda s: require(False, 'nonfinite JSON'))


def load(record, shape, dtype):
    path = Path(record['path'])
    require(path.stat().st_size < 9 * 1024**2 and sha(path) == record['sha256'], 'array byte binding')
    a = np.load(path, allow_pickle=False)
    require(a.shape == shape and a.dtype == np.dtype(dtype) and np.isfinite(a).all(), 'array layout/finite')
    require(record['shape'] == list(shape) and record['dtype'] == dtype, 'array metadata')
    return a


def central(plus, minus, epsilon):
    plus, minus = np.asarray(plus), np.asarray(minus)
    require(epsilon in (.01, .005) and plus.shape == minus.shape and
            plus.ndim == 1 and plus.dtype == minus.dtype == np.float32 and
            np.isfinite(plus).all() and np.isfinite(minus).all(), 'finite aligned float32 successors')
    return (plus.astype(np.float64) - minus.astype(np.float64)) / (2. * epsilon)


def measures(j, f):
    j, f = np.asarray(j, np.float64), np.asarray(f, np.float64)
    nj, nf = float(np.linalg.norm(j)), float(np.linalg.norm(f))
    return dict(relative_error_to_jvp=float(np.linalg.norm(j-f))/nj if nj else None,
                cosine=float(np.sum(j*f)/(nj*nf)) if nj and nf else None,
                max_abs_error=float(np.max(np.abs(j-f))))


def audit(path, expected):
    path = Path(path).absolute()
    require(sha(path) == expected, 'explicit FD receipt SHA')
    r = read(path)
    require(r['status'] == 'native_JVP_FD_comparison_recorded' and r['error'] is None and
            r['finalization_errors'] == [] and r['counts_match'] is True and
            not r['partial_work_unknown'] and r['model_before'] == r['model_after'], 'complete fixed-model FD required')
    compiler_path = Path(r['compiler_receipt_path'])
    require(sha(compiler_path) == r['compiler_receipt_sha256'], 'compiler reference SHA')
    c = read(compiler_path)
    require(c['status'] == 'selected_packet_replayed' and c['model_after'] == r['model_before'] and
            c['compiler_inputs_sha256'] == r['compiler_inputs_sha256'], 'compiler/context identity')
    nominal = load(c['arrays']['normalized_nominal'], (10,), 'float32')
    j = load(c['arrays']['response_columns'], (D, 10), 'float32')
    checks = {}
    rebuilt = {}
    for eps in (.01, .005):
        columns = []
        for i in range(10):
            values = {}
            for sign, label in ((1, 'plus'), (-1, 'minus')):
                name = f'eps_{eps}_column_{i}_{label}'
                z = load(r['arrays'][name+'_normalized_input'], (10,), 'float32')
                expected_z = nominal.copy()
                expected_z[i] += np.float32(sign*eps)
                checks[name+'_input_exact'] = np.array_equal(z, expected_z)
                values[label] = load(r['arrays'][name+'_successor'], (D,), 'float32')
            columns.append(central(values['plus'], values['minus'], eps))
        matrix = np.column_stack(columns)
        saved = load(r['arrays'][f'FD_matrix_eps_{eps}'], (D, 10), 'float64')
        checks[f'eps_{eps}_matrix_exact'] = np.array_equal(matrix, saved)
        m = measures(j, matrix)
        reported = r['comparisons'][str(eps)]['matrix']
        for key, value in m.items():
            checks[f'eps_{eps}_{key}'] = bool(np.isclose(value, reported[key], rtol=1e-12, atol=1e-12))
        rebuilt[str(eps)] = dict(matrix=m, columns=[measures(j[:, i], matrix[:, i]) for i in range(10)])
    journal_path = path.parent/'journal.jsonl'
    require(sha(journal_path) == r['journal_sha256'], 'FD journal bytes')
    previous = None
    dispatches = []
    for i, line in enumerate(journal_path.read_bytes().splitlines()):
        e = json.loads(line)
        body = {k: v for k, v in e.items() if k != 'entry_sha256'}
        h = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        require(e['sequence'] == i and e['previous_sha256'] == previous and e['entry_sha256'] == h, 'FD journal chain')
        previous = h
        if e['event']['kind'] == 'FD_dispatch':
            dispatches.append(e['event'])
    checks['exact_40_dispatch_order'] = [(e['epsilon'], e['column'], e['sign']) for e in dispatches] == [
        (eps, i, sign) for eps in (.01, .005) for i in range(10) for sign in (1, -1)]
    for event in dispatches:
        eps, i, sign = event['epsilon'], event['column'], event['sign']
        z = load(r['arrays'][f'eps_{eps}_column_{i}_'+('plus' if sign == 1 else 'minus')+'_normalized_input'], (10,), 'float32')
        checks[f'displacement_{eps}_{i}_{sign}'] = np.array_equal(event['effective_displacement'], z.astype(float)-nominal.astype(float))
    checks['completed40_only'] = r['attempted'] == r['completed'] == dict(official_encoder=0, official_rgb_frames=0, proprio_encoder=0, predictor=40, jvp_columns=0, autograd_grad=0)
    checks['all_artifact_bindings'] = all(sha(p) == h for p, h in r['artifact_bindings'].items())
    require(sha(path) == expected and sha(compiler_path) == r['compiler_receipt_sha256'], 'receipt drift')
    return dict(schema=NAME, status='saved_fd_arithmetic_consistent' if all(checks.values()) else 'inconsistent',
                checks=checks, all_checks=all(checks.values()), check_count=len(checks), comparisons=rebuilt,
                fd_receipt_sha256=expected, source_sha256=sha(__file__), neural_calls=0, physical_calls=0,
                acceptance_threshold=None, P1_admission=False, P5_admission=False,
                limitation='Saved arithmetic and dispatch receipts only; not independent neural reproduction or proof of task efficacy.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--receipt-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = audit(a.receipt, a.receipt_sha256)
    with a.output.open('x') as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({k: result[k] for k in ('status', 'check_count', 'all_checks')}))
    require(result['all_checks'], 'FD saved arithmetic inconsistent')


if __name__ == '__main__':
    main()
