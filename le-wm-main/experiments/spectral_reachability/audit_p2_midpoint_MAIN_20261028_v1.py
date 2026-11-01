"""Independent saved-array arithmetic; no producer/model/environment imports."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'results/agent_p2_midpoint_teacher_20261028_v1_cuda_MAIN01'


def read(path, expected=None):
    data = Path(path).read_bytes()
    if expected is not None and hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('saved input SHA mismatch')
    return json.loads(data)


def main():
    report = read(RUN / 'report.json', '7b0b5c00735a7326326a4619c86539fdb51b1bf99f384fe8794a71a6b4e6321c')
    strict = read(ROOT / 'results/agent_p2_midpoint_teacher_20261028_v1_consume_MAIN01.json',
                  '4e61d9251f86550d0830c33d5607cfe58d4b308b97865f7fc64de0013d25d135')
    protocol = read(ROOT / 'results/agent_p2_midpoint_teacher_20261028_v1_prepare_MAIN01/protocol.json',
                    '05f261374b147ce35c4b725b7d945215f0eb4ae9d10ae27f8223fae76519c1d4')
    compiled = read(protocol['compilation_path'], protocol['compilation_sha256'])
    projections = {r['root_sha256']: r['compiled']['scoring']['metric']['projection'] for r in compiled['rows']}
    rows = []
    if not (len(report['rows']) == len(strict['rows']) == len(protocol['frames']) == 8):
        raise ValueError('full eight-arm denominator')
    for frame, saved, checked in zip(protocol['frames'], report['rows'], strict['rows'], strict=True):
        ref = saved['arrays']['native192']
        path = RUN / ref['file']
        if path.parent != RUN or hashlib.sha256(path.read_bytes()).hexdigest() != ref['sha256']:
            raise ValueError('native array file identity')
        actual = np.load(path, allow_pickle=False)
        if actual.dtype != np.float32 or actual.shape != (192,):
            raise ValueError('native precision/shape')
        predicted = np.asarray(frame['predicted_vector'], np.float64)
        goal = np.asarray(frame['goal_vector'], np.float64)
        q = np.asarray(projections[frame['root_sha256']], np.float64)
        values = dict(teacher_Q=float(np.mean(((actual.astype(np.float64)-goal) @ q.T)**2)),
                      predicted_Q=float(np.mean(((predicted-goal) @ q.T)**2)),
                      latent_prediction_mse=float(np.mean((actual-predicted)**2)),
                      projected_prediction_mse=float(np.mean(((actual-predicted) @ q.T)**2)))
        if (frame['frame_id'] != saved['frame_id'] or frame['root_slot'] != checked['root_slot'] or
                frame['arm'] != checked['arm'] or values != checked['midpoint']):
            raise ValueError('independent recomputation mismatch')
        rows.append(dict(root_slot=frame['root_slot'], arm=frame['arm'], **values))
    pairs = []
    for slot in range(4):
        j, c = rows[2*slot:2*slot+2]
        delta = {k: c[k]-j[k] for k in ('predicted_Q', 'teacher_Q')}
        if any(delta[k] != strict['pairs'][slot]['midpoint_composition_minus_joint'][k] for k in delta):
            raise ValueError('paired arithmetic mismatch')
        old = strict['pairs'][slot]['terminal_composition_minus_joint']
        same_prefix = protocol['frames'][2*slot]['executed_prefix_actions'] == protocol['frames'][2*slot+1]['executed_prefix_actions']
        pairs.append(dict(root_slot=slot, same_literal_raw_prefix5=same_prefix, midpoint_delta=delta,
                          midpoint_rank_reversed=bool(delta['predicted_Q']*delta['teacher_Q'] < 0),
                          terminal_rank_reversed=bool(old['predicted_Q']*old['teacher_Q'] < 0),
                          terminal_scope='inherited strict receipt, not reencoded or independently recomputed here'))
    result = dict(schema='audit_p2_midpoint_MAIN_20261028_v1', rows=rows, pairs=pairs,
                  new_neural_calls=0, new_physics_steps=0, scientific_admission=False,
                  independent_roots=0, known_root_denominator=4,
                  limitations=['boundary observations do not locate a substep or establish causal mechanism',
                               'rank reversal is descriptive, not a theoretical impossibility result'])
    data = json.dumps(result, sort_keys=True, allow_nan=False).encode()
    output = ROOT / 'results/audit_p2_midpoint_MAIN_20261028_v1_MAIN01.json'
    with output.open('xb') as stream:
        if stream.write(data) != len(data):
            raise OSError('short audit write')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(output=str(output), sha256=hashlib.sha256(data).hexdigest(), pairs=pairs)))


if __name__ == '__main__':
    main()
