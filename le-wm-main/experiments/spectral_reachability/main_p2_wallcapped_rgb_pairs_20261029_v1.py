"""Independent saved-array arithmetic, not a replacement for full strict consume."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np

NAME = 'main_p2_wallcapped_rgb_pairs_20261029_v1'
ROOT = Path(__file__).resolve().parents[2]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path, expected):
    body = Path(path).read_bytes()
    require(hashlib.sha256(body).hexdigest() == expected, 'external sealed SHA')
    return json.loads(body)


def mse(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    require(a.shape == b.shape == (192,) and np.isfinite(a).all() and np.isfinite(b).all(), '192 finite')
    return float(np.mean((a-b)**2))


def keyed(rows, physical=False):
    out = {}
    for row in rows:
        key = (row['root_slot'], row['arm']) + (() if physical else (row['checkpoint'],))
        require(key not in out, 'duplicate identity')
        out[key] = row
    return out


def pair(values):
    expected = {(r, a, s) for r in (0, 1) for a in ('joint', 'composition') for s in (5, 10, 15)}
    require(set(values) == expected, 'complete twelve-key roster')
    pairs = []
    for root in (0, 1):
        for step in (5, 10, 15):
            j, c = (values[root, arm, step] for arm in ('joint', 'composition'))
            delta = {k: None if c[k] is None or j[k] is None else c[k]-j[k] for k in j}
            def relation(a, b):
                if a is None or b is None:
                    return 'missing'
                if a == 0 or b == 0:
                    return 'tie'
                return 'reversed' if a*b < 0 else 'aligned'
            pairs.append(dict(root_slot=root, checkpoint=step, composition_minus_joint=delta,
                model_to_real=relation(delta['model_Q'], delta['real_Q']),
                real_to_joint=relation(delta['real_Q'], delta['joint_xy'])))
    return pairs


def analyze(report, receipt, physical, load):
    rr, cc, pp = keyed(report['rows']), keyed(receipt['rows']), keyed(physical['rows'], True)
    require(set(rr) == set(cc) and len(pp) == 4, 'same complete identities')
    require(report['errors'] == [] and report['model_unchanged'] is True, 'clean frozen report')
    require(receipt['status'] == 'all12_wallcapped_RGB_slots_verified_NOT_science'
            and receipt['missing_slots'] == [11] and receipt['real_pairs_available'] == 5,
            'complete strict receipt and explicit missing slot')
    values = {}
    for key, row in rr.items():
        accepted, branch = cc[key], pp[key[:2]]
        require(row['selection_digest'] == accepted['selection_digest'] == branch['selection_digest'], 'same literal plan')
        require(row['task_id'] == accepted['task_id'] == branch['task_id'], 'same task')
        q = mse(row['predicted192'], row['goal192'])
        require(q == row['model_Q'] == accepted['model_Q'], 'independent model Q')
        response = [x for x in branch['responses'] if x['step'] == key[2]]
        if row['latent'] is None:
            require(key == (1, 'joint', 15) and row['index'] == 11 and not response
                    and branch['status'] == 'native_stopped' and branch['own_returned'] == 14
                    and row['response'] is accepted['response'] is None
                    and accepted['real_Q'] is accepted['prediction_vs_encoded_MSE'] is None,
                    'native stop14, never replace15 with14')
            values[key] = dict(model_Q=q, real_Q=None, joint_xy=None, object_xy=None, wrapped_angle=None)
            continue
        z = load(row['latent'])
        require(z.dtype == np.float32 and z.shape == (192,) and np.isfinite(z).all(), 'native f32 latent')
        real = mse(z, row['goal192'])
        require(real == accepted['real_Q'] and mse(z, row['predicted192']) == accepted['prediction_vs_encoded_MSE'], 'independent real Q and residual')
        require(len(response) == 1 and response[0] == row['response'] == accepted['response'], 'same exact time response')
        d = np.asarray(response[0]['state'], np.float64) - np.asarray(branch['goal'], np.float64)
        joint, obj = float(np.linalg.norm(d[:4])), float(np.linalg.norm(d[2:4]))
        angle = float(abs(np.arctan2(np.sin(d[4]), np.cos(d[4]))))
        native = accepted['native_metrics']
        require(joint == native['joint_xy_error'] and obj == native['native']['object_position_error']
                and angle == native['angle_wrapped_error'], 'independent physical distances')
        values[key] = dict(model_Q=q, real_Q=real, joint_xy=joint, object_xy=obj, wrapped_angle=angle)
    pairs = pair(values)
    return dict(schema=NAME, fixed_slots=12, actual_encoded_slots=11, fixed_pairs=6, real_pairs=5,
        pairs=pairs, known_roots=2, fresh_roots=0, new_neural_calls=0, new_physics_calls=0,
        scientific_admission=False, interpretation='posthoc sealed-action diagnosis; not a gate or method confirmation')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('report', 'report-sha', 'receipt', 'receipt-sha', 'physical', 'physical-sha', 'output'):
        ap.add_argument('--'+name, required=True)
    a = ap.parse_args()
    report, receipt, physical = (read(getattr(a, k), getattr(a, k+'_sha')) for k in ('report', 'receipt', 'physical'))
    require(receipt['report_sha256'] == a.report_sha and receipt['physical_report_sha256'] == a.physical_sha, 'receipt source binding')
    def load(ref):
        require(Path(ref['file']).name == ref['file'], 'flat own array')
        body = Path(a.report).parent.joinpath(ref['file']).read_bytes()
        require(hashlib.sha256(body).hexdigest() == ref['sha256'], 'actual array SHA')
        return np.load(io.BytesIO(gzip.decompress(body)), allow_pickle=False)
    result = analyze(report, receipt, physical, load)
    result['sources'] = {k:dict(path=str(Path(getattr(a, k)).resolve()), sha256=getattr(a, k+'_sha')) for k in ('report', 'receipt', 'physical')}
    output = Path(a.output).resolve()
    require(output.parent == ROOT/'results' and output.name.startswith(NAME+'_'), 'exclusive new result')
    with output.open('x') as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
