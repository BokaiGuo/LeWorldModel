"""Independent saved-only recomputation; imports no experiment/model runtime."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STEM = 'agent_p2_midpoint_teacher_suffix_20261028_v1'


def read(path, expected):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('input SHA mismatch: ' + str(path))
    return json.loads(data)


def main():
    directory = ROOT / ('results/' + STEM + '_cuda_MAIN01')
    run = read(directory / 'report.json', '08bec783a5791c15a03396131df4693e3d4c63c43c9523c38f3e499d65e9fe66')
    receipt = read(ROOT / ('results/' + STEM + '_consume_MAIN01.json'),
                   '36864d57aa6914f0d09ff4e947ef141ccf0037a2fc23f9d7e3f414fb217d04ac')
    protocol = read(ROOT / ('results/' + STEM + '_prepare_MAIN01/protocol.json'),
                    'e72c138505f5f80b856b95342b9d6034cd5a464cadafab0087fb2a62473dddc8')
    if not len(run['rows']) == len(receipt['rows']) == len(protocol['jobs']) == 8:
        raise ValueError('fixed eight-arm denominator')
    rows = []
    for job, saved, checked in zip(protocol['jobs'], run['rows'], receipt['rows'], strict=True):
        ref = saved['arrays']['terminal_z']
        path = directory / ref['file']
        if path.parent != directory or hashlib.sha256(path.read_bytes()).hexdigest() != ref['sha256']:
            raise ValueError('native array binding')
        raw = np.load(path, allow_pickle=False)
        if raw.shape != (192,) or raw.dtype != np.float32 or not np.isfinite(raw).all():
            raise ValueError('native output precision/shape')
        actual = np.asarray(job['actual_terminal_z'], np.float64)
        goal = np.asarray(job['goal'], np.float64)
        projection = np.asarray(job['projection'], np.float64)
        def scores(predicted):
            z = np.asarray(predicted, np.float64)
            return dict(teacher_Q=float(np.mean(((actual-goal) @ projection.T)**2)),
                        predicted_Q=float(np.mean(((z-goal) @ projection.T)**2)),
                        latent_prediction_mse=float(np.mean((actual-z)**2)),
                        projected_prediction_mse=float(np.mean(((actual-z) @ projection.T)**2)))
        old, new = scores(job['original_terminal_latent']), scores(raw)
        for label, values in [('original_open_loop', old), ('teacher_forced_suffix', new)]:
            for key, value in values.items():
                if not np.isclose(value, checked[label][key], rtol=1e-12, atol=1e-14):
                    raise ValueError('independent arithmetic mismatch')
        rows.append(dict(root_slot=job['root_slot'], arm=job['arm'], old=old, new=new,
                         latent_mse_change=new['latent_prediction_mse']-old['latent_prediction_mse']))
    pairs = []
    for i in range(4):
        joint, composed = rows[2*i:2*i+2]
        gaps = dict(actual=composed['new']['teacher_Q']-joint['new']['teacher_Q'],
                    old=composed['old']['predicted_Q']-joint['old']['predicted_Q'],
                    new=composed['new']['predicted_Q']-joint['new']['predicted_Q'])
        old_match = bool(np.sign(gaps['old']) == np.sign(gaps['actual']))
        new_match = bool(np.sign(gaps['new']) == np.sign(gaps['actual']))
        if old_match != receipt['pairs'][i]['original_order_matches_teacher'] or new_match != receipt['pairs'][i]['teacher_forced_order_matches_teacher']:
            raise ValueError('independent rank mismatch')
        pairs.append(dict(root_slot=i, gaps=gaps, old_match=old_match, new_match=new_match))
    result = dict(status='all8_saved_suffix_arithmetic_independently_verified', rows=rows, pairs=pairs,
                  all8_latent_mse_lower=all(row['latent_mse_change'] < 0 for row in rows),
                  old_rank_matches=sum(p['old_match'] for p in pairs),
                  new_rank_matches=sum(p['new_match'] for p in pairs), known_roots=4,
                  new_model_calls=0, new_physics_calls=0, scientific_admission=False)
    output = ROOT / 'results/audit_p2_suffix_MAIN_20261028_v1_MAIN01.json'
    payload = json.dumps(result, sort_keys=True, allow_nan=False).encode()
    with output.open('xb') as stream:
        if stream.write(payload) != len(payload):
            raise ValueError('short receipt write')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=result['status'], output=str(output), sha256=hashlib.sha256(payload).hexdigest(),
                          all8_latent_mse_lower=result['all8_latent_mse_lower'],
                          old_rank_matches=result['old_rank_matches'], new_rank_matches=result['new_rank_matches'])))


if __name__ == '__main__':
    main()
