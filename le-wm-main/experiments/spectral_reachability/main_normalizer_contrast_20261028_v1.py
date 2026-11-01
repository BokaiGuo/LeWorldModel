"""Independent saved-only contrast of the MAIN completed fixed normalizer run."""
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NAME = 'main_normalizer_contrast_20261028_v1'
PINS = {
    'agent_raw_normalizer_ablation_20261028_v1_prepare_MAIN01.json':
        'e35710fbb588aadadf9c99bafa1872292ed6508604163450b753c03d0744468d',
    'agent_raw_normalizer_ablation_20261028_v1_cuda_MAIN01/report.json':
        'c676d40938695ecbb539bcc0bc66b4d3ecc370040c88e1c374650d2009861e26',
    'agent_raw_normalizer_ablation_20261028_v1_consume_MAIN01.json':
        '95d7d183dd98a4edd7df0bcd8750fc8cede2270939da04c32e16aa661813a8d0',
    'agent_observed_raw_coordinate_rgb_20261028_v1_consume_MAIN01.json':
        'd6b6ed08732d7595e740f98e0dc2450adf439e726aa3078854badb78f7d8b4b2',
    'agent_observed_raw_coordinate_fd_20261028_v1_cuda_MAIN01/report.json':
        'df21e0cd0a235a6e6f43572cc02614f8c62b4704bac2aabb8ec4e2886104aa15',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(body):
    return hashlib.sha256(body).hexdigest()


def loss(blob):
    body = base64.b64decode(blob['data'], validate=True)
    require(blob['dtype'] == '<f4' and blob['shape'] == [1]
            and len(body) == 4 and sha(body) == blob['sha256'], 'literal scalar blob')
    value = float(np.frombuffer(body, dtype='<f4')[0])
    require(np.isfinite(value), 'finite native loss')
    return value


def category(predicted, actual):
    require(np.isfinite([predicted, actual]).all(), 'finite signed differences')
    if actual == 0:
        return 'both_tie' if predicted == 0 else 'actual_tie'
    if predicted == 0:
        return 'predicted_tie'
    return 'same' if (predicted > 0) == (actual > 0) else 'opposite'


def pair_labels(model, actual):
    require(model.shape == actual.shape == (164,) and np.isfinite([model, actual]).all(),
            'whole fixed164 endpoints, no missing/subset')
    labels, panels = [], []
    for panel in range(4):
        for hi, h in enumerate((.01, .001)):
            local = []
            for coord in range(10):
                minus = panel*41 + 1 + hi*20 + coord*2
                local.append(category(model[minus+1]-model[minus], actual[minus+1]-actual[minus]))
            labels.extend(local)
            panels.append(dict(panel=panel, h=h, counts=dict(Counter(local))))
    require(len(labels) == 80, 'all80 pairs')
    return dict(labels=labels, counts=dict(Counter(labels)), by_panel_h=panels)


def build():
    values = []
    for name, expected in PINS.items():
        path = ROOT/'results'/name
        require(not path.is_symlink(), 'not symlink')
        body = path.read_bytes()
        require(sha(body) == expected, 'pinned completed bytes '+name)
        values.append(json.loads(body))
    p, r, strict, real, old = values
    require(strict['complete'] is True and strict['public_exact_controls'] == 164
            and strict['report_sha256'] == PINS[list(PINS)[1]] and not r['errors']
            and [row['arm'] for row in r['rows']] == [0]*4+[1]*4,
            'MAIN completed strict authority, not a new upstream certification')
    require(all(new['context'] == prior['context'] and new['objective'] == prior['objective']
                for row, orig in zip(r['rows'][:4], old['rows'], strict=True)
                for new, prior in zip(row['values'], orig['values'], strict=True)), 'all164 public exact')
    z = np.asarray([row['actual192'] for row in real['rows']], dtype=np.float64)
    goal = np.asarray([job['task']['goal'] for job in p['jobs']], dtype=np.float64)
    require(z.shape == goal.shape == (164, 192), 'same original real/goal shape')
    actual_q = ((z-goal)**2).mean(axis=-1)
    require(np.array_equal(actual_q, np.asarray([row['actual_Q'] for row in real['rows']])),
            'independent real float64 Q exact')
    arms = []
    for arm in range(2):
        v = [v for row in r['rows'] if row['arm'] == arm for v in row['values']]
        require([item['index'] for item in v] == list(range(164)), 'full arm order')
        y = np.asarray([item['context']['latent_history'][-1] for item in v], dtype=np.float64)
        native_q = np.asarray([loss(item['objective']['arrays']['loss']) for item in v])
        require(np.array_equal(native_q, [item['objective']['diagnostic']['native_float32_Q'] for item in v]),
                'native scalar independently decoded')
        native = pair_labels(native_q, actual_q)
        f64 = pair_labels(((y-goal)**2).mean(axis=-1), actual_q)
        mse = ((y-z)**2).mean(axis=-1)
        arms.append(dict(arm=p['arms'][arm]['name'], native_loss_vs_actual64=native,
                         independent_float64_vs_actual64=f64, mean_endpoint_MSE=float(mse.mean()),
                         baseline_endpoint_MSE=[float(mse[i]) for i in (0, 41, 82, 123)]))
    transition = dict(Counter(a+'->'+b for a, b in zip(
        arms[0]['native_loss_vs_actual64']['labels'], arms[1]['native_loss_vs_actual64']['labels'], strict=True)))
    return dict(schema=NAME, bindings=PINS, arms=arms, paired_transition=transition,
                fixed_pairs_per_arm=80, public_all164_exact=True, new_neural_calls=0,
                new_physics_calls=0, known_roots=2, fresh_roots=0, correction_deployed=False,
                P1_admission=False, training_provenance=False,
                scope='saved independent contrast; pinned already MAIN-completed strict, not recursive recertification')


if __name__ == '__main__':
    result = build()
    output = ROOT/'results'/(NAME+'_MAIN01.json')
    body = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    with output.open('xb') as f:
        f.write(body)
    print(json.dumps(dict(path=str(output), sha256=sha(body))))
