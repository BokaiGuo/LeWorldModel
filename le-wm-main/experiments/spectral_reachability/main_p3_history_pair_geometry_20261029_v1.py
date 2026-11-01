"""Independent scalar geometry of all eight saved A/R endpoints, not a gate."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

NAME = 'main_p3_history_pair_geometry_20261029_v1'
OWNER = 'agent_p3_same_action_actual_history_cuda_20261029_v1'
ROOT = Path(__file__).resolve().parents[2]
REPORT_SHA = 'cf038db843d2daa51b426e12aac46558d907d65762147ef465d8a5d82381b251'
RECEIPT_SHA = '0cea92f6ff26d5075e679a835a91e564d75830145e1cd109173a6d183f04d3a9'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def raw(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unique(items):
    result = {}
    for key, value in items:
        need(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def read(path, expected):
    path = Path(path).absolute()
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents))
         and path.stat().st_size <= 16*1024**2 and sha(path) == expected, 'bounded actual input SHA')
    value = json.loads(path.read_bytes(), object_pairs_hook=unique)
    raw(value)
    return value


def close(value, reference):
    # Float64 reduction-only bound, never used for ranking or scientific admission.
    need(abs(value-reference) <= 4096*math.ulp(1.)*max(1., abs(value), abs(reference)),
         'independent scalar arithmetic disagrees')


def sign(value):
    return (value > 0)-(value < 0)


def geometry(predicted, actual, goal):
    need(all(type(v) is list and len(v) == 192 and
             all(type(x) in (int, float) and math.isfinite(x) for x in v)
             for v in (predicted, actual, goal)), 'finite complete192 vectors')
    error = [p-a for p, a in zip(predicted, actual)]
    residual = [a-g for a, g in zip(actual, goal)]
    mse = math.fsum(e*e for e in error)/192
    actual_q = math.fsum(r*r for r in residual)/192
    model_q = math.fsum((p-g)**2 for p, g in zip(predicted, goal))/192
    cross = 2*math.fsum(e*r for e, r in zip(error, residual))/192
    close(model_q-actual_q, mse+cross)
    return dict(error_mse=mse, actual_encoded_Q=actual_q, model_Q=model_q,
                model_minus_actual_Q=model_q-actual_q, actual_residual_cross_term=cross,
                squared_error_cross_cancel=mse*cross < 0,
                identity_residual=(model_q-actual_q)-(mse+cross))


def describe(scores):
    need(len(scores) == 8, 'fixed eight endpoints')
    rows = []
    for i, row in enumerate(scores):
        branch, time = divmod(i, 2)
        need((row['index'], row['root_slot'], row['bank'], row['start'], row['end']) ==
             (i, branch//2, ('endpoints', 'midpoints')[branch % 2], (10, 15)[time], (15, 20)[time]),
             'complete fixed root/bank/time order')
        need(row['same_physical_label'] is True and row['physical_benefit_claim'] is False,
             'same-label diagnostic only')
        need(raw(row['A']['actual192']) == raw(row['R']['actual192']) and
             raw(row['A']['goal192']) == raw(row['R']['goal192']), 'same actual label and goal')
        value = {k: row[k] for k in ('index', 'root_slot', 'bank', 'start', 'end')}
        for cell in ('A', 'R'):
            old = row[cell]
            g = geometry(old['predicted192'], old['actual192'], old['goal192'])
            for key in ('error_mse', 'actual_encoded_Q', 'model_Q', 'model_minus_actual_Q', 'actual_residual_cross_term'):
                close(g[key], old[key])
            need(sign(g['model_minus_actual_Q']) == sign(old['model_minus_actual_Q']), 'no reduction sign ambiguity')
            value[cell] = g
        rows.append(value)
    pairs = []
    for root in (0, 1):
        for end in (15, 20):
            a, b = [next(r for r in rows if (r['root_slot'], r['end'], r['bank']) == (root, end, bank))
                    for bank in ('endpoints', 'midpoints')]
            original_a, original_b = scores[a['index']], scores[b['index']]
            need(raw(original_a['A']['goal192']) == raw(original_b['A']['goal192']), 'same paired task goal')
            margin = a['A']['actual_encoded_Q']-b['A']['actual_encoded_Q']
            need(sign(margin) == sign(original_a['A']['actual_encoded_Q']-original_b['A']['actual_encoded_Q']),
                 'actual pair reduction sign unambiguous')
            cells = {}
            for cell in ('A', 'R'):
                predicted = a[cell]['model_Q']-b[cell]['model_Q']
                need(sign(predicted) == sign(original_a[cell]['model_Q']-original_b[cell]['model_Q']),
                     'predicted pair reduction sign unambiguous')
                mse_delta = a[cell]['error_mse']-b[cell]['error_mse']
                cross_delta = a[cell]['actual_residual_cross_term']-b[cell]['actual_residual_cross_term']
                close(predicted-margin, mse_delta+cross_delta)
                cells[cell] = dict(model_margin=predicted, error_mse_delta=mse_delta,
                                   cross_term_delta=cross_delta, matches_actual_sign=sign(predicted) == sign(margin),
                                   margin_error=predicted-margin)
            pairs.append(dict(root_slot=root, end=end, first_bank='endpoints', second_bank='midpoints',
                              actual_margin=margin, actual_sign=sign(margin), cells=cells))
    return dict(rows=rows, pairs=pairs,
                counts=dict(endpoints=8, paired_comparisons=4,
                            A_sign_matches=sum(p['cells']['A']['matches_actual_sign'] for p in pairs),
                            R_sign_matches=sum(p['cells']['R']['matches_actual_sign'] for p in pairs),
                            A_cross_cancel=sum(r['A']['squared_error_cross_cancel'] for r in rows),
                            R_cross_cancel=sum(r['R']['squared_error_cross_cancel'] for r in rows)))


def analyze():
    directory = ROOT/'results'/(OWNER+'_cuda_MAIN01')
    rp = directory/'report.json'
    cp = ROOT/'results'/(OWNER+'_consume_MAIN01.json')
    report, receipt = read(rp, REPORT_SHA), read(cp, RECEIPT_SHA)
    need(report['schema'] == OWNER+'.report' and report['errors'] == [] and
         receipt['schema'] == OWNER+'.consume' and receipt['report_sha256'] == REPORT_SHA and
         receipt['status'] == 'all8_same_action_actual_history_verified_NOT_admitted' and
         receipt['fixed_denominator'] == 8, 'completed external MAIN strict authority')
    sp = directory/'scores.json'
    sh = report['artifact_sha256']['scores.json']
    scores = read(sp, sh)
    need(scores == receipt['rows'], 'receipt/scoring bytes join')
    for row, prediction in zip(scores, report['rows'], strict=True):
        need(row['A']['predicted192'] == prediction['after']['latent_history'][-1], 'actual CUDA output join')
    out = dict(schema=NAME+'.analysis', **describe(scores),
               inputs={str(rp): REPORT_SHA, str(cp): RECEIPT_SHA, str(sp): sh},
               source_sha256=sha(__file__), tests_sha256=sha(ROOT/'tests'/('test_'+NAME+'.py')),
               limitations=['post-outcome descriptive arithmetic, not preregistered confirmation',
                            'two known roots; four comparisons are not four independent roots',
                            'actual encoded Q is not physical task success',
                            'trusts prior strict for loading, input chronology and native IO'],
               P1_admission=False, P3_admission=False, new_neural_calls=0, new_physics_calls=0)
    need(all(sha(p) == h for p, h in out['inputs'].items()), 'inputs stable')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('analyze', 'verify'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--sha')
    args = parser.parse_args()
    path = Path(args.output).absolute()
    value = analyze()
    if args.command == 'verify':
        need(raw(read(path, args.sha)) == raw(value), 'whole deterministic analysis reproduction')
    else:
        need(path.parent == ROOT/'results' and path.name.startswith(NAME) and not path.exists(), 'exclusive own result')
        with path.open('xb') as handle:
            data = raw(value)
            need(handle.write(data) == len(data), 'complete write')
            handle.flush()
            os.fsync(handle.fileno())
    print(json.dumps(dict(path=str(path), sha256=sha(path), verified=args.command == 'verify')))


if __name__ == '__main__':
    main()
