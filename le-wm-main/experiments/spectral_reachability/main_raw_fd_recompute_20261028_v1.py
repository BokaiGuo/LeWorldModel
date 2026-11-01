"""Independent saved raw FD matrix arithmetic, requiring a pinned full consumer."""
import argparse
from io import BytesIO
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import main_midfeedback_physical_effect_20261028_v1 as base

NAME = 'main_raw_fd_recompute_20261028_v1'
OUTPUT = base.ROOT/'results/agent_p2_midfeedback_raw_jacobian_fd_20261028_v1_cuda_MAIN01'
PREPARE = base.ROOT/'results/agent_p2_midfeedback_raw_jacobian_fd_20261028_v1_prepare_MAIN01.json'
REPORT_SHA = 'f555758ec008cab87cdc375c8295a7940dc5c0354bd27049a1aa7787a5b945fa'
PREPARE_SHA = 'dd9127c1feb721b5c232482990f0bfbd19b3f6d8d0d2b737740687eeb3b98baa'
STEPS = (.01, .001, .0001)


def load_array(entry):
    name = entry['file']; path = OUTPUT/name
    base.require(Path(name).name == name and path.resolve() == path and not path.is_symlink() and
        base.sha(path) == entry['sha256'], 'exact saved native array file')
    stream = BytesIO(path.read_bytes()); a = np.load(stream, allow_pickle=False)
    base.require(stream.tell() == len(stream.getvalue()) and np.isfinite(a).all(), 'full finite array')
    return a


def matrix_stats(ad, fd):
    base.require(ad.shape == fd.shape == (192, 10), 'full matrix, not subset')
    da = fd-ad; na = float(np.linalg.norm(ad)); nf = float(np.linalg.norm(fd))
    base.require(na > 0 and nf > 0, 'actual cohort nonzero derivative matrices')
    return dict(absolute_l2=float(np.linalg.norm(da)), relative_l2_to_AD=float(np.linalg.norm(da))/na,
        direction_cosine=float(np.sum(ad*fd)/(na*nf)), max_abs=float(np.max(np.abs(da))))


def point_stats(point, schedule, row, strict):
    base.require(row['index'] == strict['index'] == point['index'] and len(row['evaluations']) == 60 and
        all(e['status'] == 'completed' for e in row['evaluations']), 'actual full60, missing cannot become zero')
    endpoints = []
    anchor = np.asarray(point['raw5'], np.float32).reshape(10)
    for request, e in zip(schedule, row['evaluations'], strict=True):
        base.require(e['request'] == request and request['legal'], 'same declared raw request')
        a = anchor.copy(); c = request['coordinate']
        a[c] = np.float32(float(anchor[c])+request['sign']*request['h'])
        base.require(a.astype('<f4').tobytes().hex() == request['raw_bits'] and
            float(a[c])-float(anchor[c]) == request['actual_delta'], 'independent raw coordinate and float32 spacing')
        z = load_array(e['latent_array']); base.require(z.shape == (3, 192) and z.dtype == np.float32, 'full3 native projected latent')
        endpoints.append(z[-1].astype(np.float64))
    ad = np.asarray(point['jacobian'], np.float64); groups = []
    for si, h in enumerate(STEPS):
        cols, actual_cols = [], []
        for c in range(10):
            lo = 20*si+2*c; hi = lo+1
            cols.append((endpoints[hi]-endpoints[lo])/(2*h))
            actual_cols.append((endpoints[hi]-endpoints[lo])/(schedule[hi]['actual_delta']-schedule[lo]['actual_delta']))
        nominal, actual = np.column_stack(cols), np.column_stack(actual_cols)
        saved = strict['steps'][si]
        base.require(saved['h'] == h and saved['full_matrix_available'] and
            np.array_equal(nominal, np.asarray(saved['nominal_FD_matrix'])) and
            np.array_equal(actual, np.asarray(saved['actual_spacing_FD_matrix'])), 'all192x10 FD values vs strict')
        stats, actual_stats = matrix_stats(ad, nominal), matrix_stats(ad, actual)
        base.require(all(stats[k] == saved['nominal_matrix_metrics'][k] and
            actual_stats[k] == saved['actual_spacing_matrix_metrics'][k] for k in stats), 'independent FD arithmetic vs strict')
        groups.append(dict(h=h, nominal=stats, actual_spacing=actual_stats))
    return dict(index=point['index'], steps=groups)


def build(receipt, expected):
    for p, h in ((OUTPUT/'report.json', REPORT_SHA), (PREPARE, PREPARE_SHA), (Path(receipt), expected)):
        base.require(base.sha(p) == h, 'fixed actual report/prepare/strict seal')
    r, p, strict = [json.loads(x.read_bytes()) for x in (OUTPUT/'report.json', PREPARE, Path(receipt))]
    base.require(strict['status'] == 'full8_raw_FD_saved_verified_diagnostic_only' and
        strict['report_sha256'] == REPORT_SHA and strict['predict_API_resources']['predict']['completed'] == 480 and
        len(strict['rows']) == len(r['rows']) == len(p['points']) == 8, 'actual full480 strict dependency')
    rows = []
    for point, schedule, entry, s in zip(p['points'], p['schedule'], r['rows'], strict['rows'], strict=True):
        path = OUTPUT/entry['file']; base.require(base.sha(path) == entry['sha256'], 'exact actual point record')
        rows.append(point_stats(point, schedule, json.loads(path.read_bytes()), s))
    ranges = []
    for si, h in enumerate(STEPS):
        ranges.append(dict(h=h, **{metric:dict(min=min(x['steps'][si]['nominal'][metric] for x in rows),
            max=max(x['steps'][si]['nominal'][metric] for x in rows))
            for metric in ('relative_l2_to_AD', 'direction_cosine', 'max_abs')}))
    return dict(schema=NAME, rows=rows, ranges=ranges, fixed_points=8, signed_probes=480, known_roots=2,
        independent_array_arithmetic=True, full_upstream_rebuilt_here=False, new_neural_calls=0, new_physics_calls=0,
        authority=dict(report_sha256=REPORT_SHA, prepare_sha256=PREPARE_SHA, strict_receipt_sha256=expected),
        AD_correctness_certified=False, scientific_threshold=None, P1_admission=False,
        interpretation='All fixed step sizes reported. Numerical agreement is local and does not certify learned action-response validity or physical benefit.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('receipt', 'receipt-sha', 'output'): p.add_argument('--'+k, required=True)
    a = p.parse_args(); output = Path(a.output).absolute()
    base.require(output.resolve() == output and output.parent == base.ROOT/'results' and
        output.name.startswith(NAME+'_') and output.suffix == '.json', 'exclusive own result')
    result = build(a.receipt, a.receipt_sha)
    with output.open('xb') as f: f.write(json.dumps(result, sort_keys=True, allow_nan=False).encode()+b'\n')
    print(json.dumps(dict(output=str(output), sha256=base.sha(output), ranges=result['ranges'])))


if __name__ == '__main__': main()
