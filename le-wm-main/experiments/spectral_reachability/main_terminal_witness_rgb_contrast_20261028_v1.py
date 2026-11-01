"""Independent saved-array old/new cp10 decomposition; no neural/physics/fitting."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from experiments.spectral_reachability.main_terminal_witness_physical_contrast_20261028_v1 import metrics

ROOT = Path(__file__).resolve().parents[2]
NAME = 'main_terminal_witness_rgb_contrast_20261028_v1'
NEW = ROOT/'results/agent_terminal_witness_terminal_rgb_20261028_v1_MAIN01'
NEW_SHA = '8d933d9d9bd83e5b68eb71161a7edf9f6f60662d9e97dc183afc0a5cdd458cab'
NEW_RECEIPT = ROOT/'results/agent_terminal_witness_terminal_rgb_20261028_v1_consume_MAIN01.json'
NEW_RECEIPT_SHA = 'e44de69224bdcab89d6dd204099f8f725124db79043a0ae7370de221aef645dc'


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def read(path, expected):
    p = Path(path)
    require(p.is_file() and not p.is_symlink(), 'real artifact')
    b = p.read_bytes()
    require(sha(b) == expected, 'external artifact SHA')
    return json.loads(b)


def latent(directory, desc):
    require(set(desc) == {'file', 'sha256', 'array_sha256', 'shape', 'dtype'}, 'whole descriptor')
    require(Path(desc['file']).name == desc['file'], 'flat original array')
    p = Path(directory)/desc['file']
    require(p.is_file() and not p.is_symlink(), 'original real array')
    b = p.read_bytes()
    require(sha(b) == desc['sha256'], 'compressed original NPY SHA')
    a = np.load(io.BytesIO(gzip.decompress(b)), allow_pickle=False)
    require(a.shape == tuple(desc['shape']) == (192,) and a.dtype == np.dtype(desc['dtype']) == np.float32
            and np.isfinite(a).all() and sha(a.tobytes()) == desc['array_sha256'], 'original float32 latent bits')
    return a.astype(np.float64)


def vector(x):
    a = np.asarray(x, dtype=np.float64)
    require(a.shape == (192,) and np.isfinite(a).all(), 'finite original192')
    return a


def mse(a, b):
    return float(np.mean((a-b)**2))


def decompose(p0, p1, a0, a1, goal):
    p0, p1, a0, a1, goal = map(vector, (p0, p1, a0, a1, goal))
    e0 = p0-a0
    da = a1-a0
    de = (p1-p0)-da
    terms = dict(common_error_response_cross=float(2*np.mean(da*e0)),
                 actual_residual_response_error_cross=float(2*np.mean((a1-goal)*de)),
                 error_interaction=float(2*np.mean(e0*de)),
                 response_error_square=float(np.mean(de**2)))
    pred = mse(p1, goal)-mse(p0, goal)
    actual = mse(a1, goal)-mse(a0, goal)
    require(np.isclose(sum(terms.values()), pred-actual, atol=1e-12, rtol=1e-12), 'signed pair identity')
    return dict(predicted_delta=pred, actual_delta=actual, ranking_reversed=bool(pred*actual < 0),
                signed_error_terms=terms, gap=pred-actual,
                response_error_MSE=float(np.mean(de**2)),
                exact_action_independent_offset=bool(np.array_equal(p1-a1, e0)),
                opposing_signed_terms=bool(any(x < 0 for x in terms.values()) and any(x > 0 for x in terms.values())))


def build(old_output, old_sha, receipt, receipt_sha):
    old_output = Path(old_output).resolve()
    old = read(old_output/'report.json', old_sha)
    oldr = read(receipt, receipt_sha)
    new = read(NEW/'report.json', NEW_SHA)
    newr = read(NEW_RECEIPT, NEW_RECEIPT_SHA)
    require(old['schema'] == 'agent_terminal_witness_control_rgb_20261028_v1.report'
            and oldr['schema'] == 'agent_terminal_witness_control_rgb_20261028_v1.consume'
            and newr['schema'] == 'agent_terminal_witness_terminal_rgb_20261028_v1.consume'
            and oldr['report_sha256'] == old_sha and newr['report_sha256'] == NEW_SHA,
            'actual distinct producer/consumer authorities')
    require(old['status'] == 'old_mixed4_control_encoded_not_admitted'
            and oldr['status'] == 'old_mixed4_control_RGB_verified_not_admitted'
            and newr['status'] == 'terminal_witness4_RGB_verified_not_admitted'
            and all(x['errors'] == x['blocked'] == [] and x['model_unchanged'] is True
                    and x['source_verified'] is True for x in (old, new))
            and all(x['P1_admission'] is x['scientific_admission'] is x['freshness'] is False
                    for x in (oldr, newr)), 'complete fixed strict evidence, no admission')
    rows = []
    require(all(len(x['rows']) == 4 for x in (old, oldr, new, newr)), 'entire four slots')
    for i, (a, ar, b, br) in enumerate(zip(old['rows'], oldr['rows'], new['rows'], newr['rows'], strict=True)):
        require(a['index'] == ar['index'] == b['index'] == br['index'] == i, 'fixed order')
        for k in ('root_slot', 'condition', 'task_sha256', 'goal_sha256'):
            require(a[k] == ar[k] == b[k] == br[k], 'same task/condition/goal')
        require(all(a[k] == b[k] for k in ('context_sha256', 'q_sha256', 'goal192'))
                and a['source']['step'] == b['source']['step'] == 10, 'same context/Q/goal/time')
        z0, z1 = latent(old_output, a['latent']), latent(NEW, b['latent'])
        g, p0, p1 = map(vector, (a['goal192'], a['predicted192'], b['predicted192']))
        for spec, accepted, z, pred in ((a, ar, z0, p0), (b, br, z1, p1)):
            require(spec['predicted_Q'] == accepted['predicted_Q'] == mse(pred, g)
                    and accepted['actual_Q'] == mse(z, g) and np.array_equal(z, accepted['actual192'])
                    and accepted['endpoint_MSE'] == mse(pred, z)
                    and all(spec[k] == accepted[k] for k in ('raw_float32_sha256', 'selection_digest', 'evaluation')),
                    'independent original-NPY recomputation and receipt identity')
        same = a['raw_float32_sha256'] == b['raw_float32_sha256']
        if same:
            require(np.array_equal(p0, p1) and np.array_equal(z0, z1)
                    and a['source']['state'] == b['source']['state'], 'same task/raw action exact replay control')
        require(a['evaluation']['final_state'] == a['source']['state']
                and b['evaluation']['final_state'] == b['source']['state'], 'native actual final time')
        # Goal belongs to original native task, never estimated from latent coordinates.
        physical = read(ROOT/'results/agent_terminal_witness_physical_20261028_v1_MAIN01/report.json',
                        'd02868f33f5b4b348c53d3684f30d2681124fc391b4f8d0f9b1a9e8992a04850')['rows'][i]
        require(physical['task_id'] == b['task_id'] and physical['final_state'] == b['source']['state'], 'native task binding')
        ma, mb = metrics(a['source']['state'], physical['goal']), metrics(b['source']['state'], physical['goal'])
        rows.append(dict(index=i, root_slot=a['root_slot'], condition=a['condition'], same_raw10=same,
                         old_index=a['selected_index'], new_index=b['selected_index'],
                         old_actual_Q=ar['actual_Q'], new_actual_Q=br['actual_Q'],
                         old_physical=ma, new_physical=mb,
                         physical_delta={k:mb[k]-ma[k] for k in ma if k != 'native_success'},
                         **decompose(p0, p1, z0, z1, g)))
    return dict(schema=NAME, rows=rows, fixed_denominator=4, known_roots=2, fresh_roots=0,
                changed_actions=sum(not r['same_raw10'] for r in rows),
                reversals=sum(r['ranking_reversed'] for r in rows), new_neural_calls=0, new_physics_calls=0,
                full_upstream_rebuilt_here=False, P1_admission=False, scientific_admission=False,
                inputs=dict(old_report_sha256=old_sha, old_receipt_sha256=receipt_sha,
                            new_report_sha256=NEW_SHA, new_receipt_sha256=NEW_RECEIPT_SHA),
                caveat='All fixed slots; signed descriptive identity, not causal ablation, fitted correction, or unique-root-cause proof.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('old-output', 'old-sha', 'receipt', 'receipt-sha', 'output'):
        p.add_argument('--'+k, required=True)
    a = p.parse_args()
    r = build(a.old_output, a.old_sha, a.receipt, a.receipt_sha)
    out = Path(a.output).absolute()
    require(out.parent == ROOT/'results' and out.name.startswith(NAME+'_'), 'exclusive own output')
    data = json.dumps(r, sort_keys=True, allow_nan=False).encode()+b'\n'
    with out.open('xb') as f:
        f.write(data)
    print(json.dumps(dict(output=str(out), sha256=sha(data), rows=r['rows']), allow_nan=False))


if __name__ == '__main__':
    main()
