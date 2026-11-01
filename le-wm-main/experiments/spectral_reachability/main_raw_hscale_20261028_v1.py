"""Compare both already fixed finite-difference scales, without choosing either."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NAME = 'main_raw_hscale_20261028_v1'
PINS = {
    'agent_observed_raw_coordinate_fd_20261028_v1_cuda_MAIN01/report.json':
        'df21e0cd0a235a6e6f43572cc02614f8c62b4704bac2aabb8ec4e2886104aa15',
    'agent_observed_raw_coordinate_fd_20261028_v1_consume_MAIN01.json':
        'c2c454cf51b11e5f28df12e2c2f9ef12c8c7cd1eba7bf99b9a9cf9c97977e2b0',
    'agent_observed_raw_coordinate_rgb_20261028_v1_consume_MAIN01.json':
        'd6b6ed08732d7595e740f98e0dc2450adf439e726aa3078854badb78f7d8b4b2',
    'agent_observed_raw_coordinate_physical_20261028_v1_MAIN01/report.json':
        'c4610c7f495137cbda7d70b147c74670db6f85339561da611ce5bee152d7bb10',
    'main_raw_coordinate_response_20261028_v1_MAIN01.json':
        '703142be3073336d76cd8cfb9cd8ef8fc41d0c3be0dd7141cc222a635e52c061',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def vector_comparison(large, small):
    large, small = np.asarray(large, float), np.asarray(small, float)
    require(large.shape == small.shape and large.ndim == 1
            and np.isfinite([large, small]).all(), 'same finite vectors')
    ln, sn = float(np.linalg.norm(large)), float(np.linalg.norm(small))
    return dict(large_norm=ln, small_norm=sn, large_exact_zero=ln == 0,
                small_exact_zero=sn == 0, cosine=float(np.dot(large, small)/(ln*sn)) if ln and sn else None,
                relative_L2_to_large=float(np.linalg.norm(small-large)/ln) if ln else None,
                norm_ratio_small_over_large=sn/ln if ln else None)


def scalar_relation(large, small):
    require(np.isfinite([large, small]).all(), 'finite scalars')
    if large == small == 0:
        return 'both_zero'
    if large == 0:
        return 'large_zero'
    if small == 0:
        return 'small_zero'
    return 'same_sign' if (large > 0) == (small > 0) else 'opposite_sign'


def summary(rows):
    out = {}
    for key in ('model192', 'actual192', 'physical_pose5'):
        cs = [row[key]['cosine'] for row in rows if row[key]['cosine'] is not None]
        out[key] = dict(pairs=len(rows), both_nonzero=len(cs),
                        small_exact_zero=sum(row[key]['small_exact_zero'] for row in rows),
                        large_exact_zero=sum(row[key]['large_exact_zero'] for row in rows),
                        cosine_min_median_max=[float(v) for v in np.quantile(cs, [0, .5, 1])] if cs else None)
    out['model_goal_signs'] = dict(Counter(row['model_goal_relation'] for row in rows))
    out['actual_goal_signs'] = dict(Counter(row['actual_goal_relation'] for row in rows))
    return out


def build():
    values = []
    for name, expected in PINS.items():
        path = ROOT/'results'/name
        body = path.read_bytes()
        require(not path.is_symlink() and hashlib.sha256(body).hexdigest() == expected,
                'pinned completed bytes '+name)
        values.append(json.loads(body))
    model, ms, real, physical, paired = values
    require(ms['complete'] is True and real['status'] == 'raw164_RGB_verified_not_admitted'
            and physical['errors'] == [] and len(real['rows']) == len(physical['rows']) == 164
            and paired['available_pairs'] == paired['fixed_pairs'] == 80, 'entire completed panel')
    model_values = [v for row in model['rows'] for v in row['values']]
    require(len(model_values) == 164, 'full164 model outputs')
    y = np.asarray([v['context']['latent_history'][-1] for v in model_values], float)
    z = np.asarray([r['actual192'] for r in real['rows']], float)
    states = np.asarray([r['final_state'] for r in physical['rows']], float)
    require(y.shape == z.shape == (164, 192) and states.shape == (164, 7), 'full endpoints')
    rows = []
    for slot, panel in enumerate(paired['panels']):
        require(panel['slot'] == slot and [s['h'] for s in panel['steps']] == [.01, .001], 'both fixed h')
        for coord in range(10):
            endpoints, slopes, comps = [], [], []
            for hi, step in enumerate(panel['steps']):
                c = step['comparisons'][coord]
                require(c['coord'] == coord and c['available'], 'same fixed coordinate')
                minus = slot*41 + 1 + hi*20 + 2*coord
                plus = minus+1
                span = c['actual_span']
                require(span > 0, 'actual float32 action span')
                pose = states[plus, :5]-states[minus, :5]
                pose[4] = np.arctan2(np.sin(pose[4]), np.cos(pose[4]))
                slopes.append(((y[plus]-y[minus])/span, (z[plus]-z[minus])/span, pose/span))
                endpoints.append(dict(h=step['h'], minus=minus, plus=plus, actual_span=span,
                                      RGB_identical=c['RGB_pixels_identical']))
                comps.append(c)
            rows.append(dict(slot=slot, coordinate=coord, endpoints=endpoints,
                             model192=vector_comparison(slopes[0][0], slopes[1][0]),
                             actual192=vector_comparison(slopes[0][1], slopes[1][1]),
                             physical_pose5=vector_comparison(slopes[0][2], slopes[1][2]),
                             model_goal_relation=scalar_relation(comps[0]['model_scalar_fd'], comps[1]['model_scalar_fd']),
                             actual_goal_relation=scalar_relation(comps[0]['actual_scalar_fd'], comps[1]['actual_scalar_fd'])))
    require(len(rows) == 40, 'all40 cross-scale comparisons from all80 original pairs')
    return dict(schema=NAME, bindings=PINS, rows=rows, summary=summary(rows),
                by_slot=[dict(slot=s, summary=summary([r for r in rows if r['slot'] == s])) for s in range(4)],
                fixed_original_pairs=80, cross_scale_comparisons=40, known_roots=2, fresh_roots=0,
                no_h_selected=True, new_neural_calls=0, new_physics_calls=0, P1_admission=False,
                scope='saved local scale sensitivity, not renderer root-cause proof or policy improvement')


if __name__ == '__main__':
    result = build()
    body = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    path = ROOT/'results'/(NAME+'_MAIN01.json')
    with path.open('xb') as f:
        f.write(body)
    print(json.dumps(dict(path=str(path), sha256=hashlib.sha256(body).hexdigest(), summary=result['summary'])))
