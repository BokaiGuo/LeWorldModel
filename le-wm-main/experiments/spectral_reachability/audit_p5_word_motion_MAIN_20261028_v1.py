"""Saved original public-body motion; no simulator/model/producer imports."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT/'results/agent_p5_action_words_physical_20261028_v1_physics_MAIN01/report.json'
REPORT_SHA = '796d757c6ffcf623f4e307f094dba9e39eda2a4810a140e5ab3cf0897b3c73a5'
RECEIPT = ROOT/'results/consume_p5_physical_MAIN_20261028_v1_MAIN01.json'
RECEIPT_SHA = '2248899fb03439bc4c52ee741dc0e2b497e15ce692d1d4d306f32fee7d476883'


def require(ok, label):
    if not ok:
        raise ValueError(label)


def bound(path, sha):
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == sha, 'external SHA: '+str(path))
    return json.loads(raw)


def block_pose(public):
    block = public['bodies']['block']
    return np.array(block['position']+[block['angle']], np.float64)


def main():
    report = bound(REPORT, REPORT_SHA)
    receipt = bound(RECEIPT, RECEIPT_SHA)
    require(receipt['status'] == 'all12_saved_native_replays_verified' and
            receipt['report_sha256'] == REPORT_SHA and len(report['rows']) == 12, 'strict full12')
    rows = []
    endpoints = {}
    for row in report['rows']:
        require(row['status'] == 'completed' and len(row['records']) == 21, 'full frame denominator')
        publics = []
        contacts = []
        for record in row['records'][10:]:
            frame = record['frame']
            path = REPORT.parent/row['case']/f'frame_{frame:02d}'
            public = bound(path/'full_public_state.json', record['files']['full_public_state.json']['sha256'])
            publics.append(public)
            if frame > 10:
                contacts.append(public['env']['n_contact_points'])
                transition = bound(path/'transition.json', record['files']['transition.json']['sha256'])
                require(transition['done'] is False, 'original native done always false')
                coverage = public['env']['coverage_arr'][-1]
                require(transition['reward'] == float(np.clip(coverage/public['env']['success_threshold'], 0, 1)),
                        'actual returned native reward')
        initial = block_pose(publics[0])
        poses = np.stack([block_pose(p) for p in publics])
        rows.append(dict(seed=row['seed'], word=row['word'], actual_suffix_frames=10,
                         block_position_max_change=float(np.max(np.linalg.norm(poses[:, :2]-initial[:2], axis=1))),
                         block_native_angle_max_change=float(np.max(np.abs(poses[:, 2]-initial[2]))),
                         block_pose_bitwise_unchanged=all(block_pose(p).tobytes() == initial.tobytes() for p in publics),
                         returned_step_contact_counts=contacts))
        endpoints[row['seed'], row['word']] = publics[-1]
    contrasts = []
    for seed in (2026102702, 2026102703, 2026102704):
        for first, second in (('xy', 'yx'), ('x_minus_x', 'zero_zero')):
            a, b = [endpoints[seed, word] for word in (first, second)]
            contrasts.append(dict(seed=seed, first=first, second=second,
                agent_position_distance=float(np.linalg.norm(np.array(a['bodies']['agent']['position'])-b['bodies']['agent']['position'])),
                agent_velocity_distance=float(np.linalg.norm(np.array(a['bodies']['agent']['velocity'])-b['bodies']['agent']['velocity'])),
                block_pose_bitwise_equal=block_pose(a).tobytes() == block_pose(b).tobytes()))
    result = dict(status='full_suffix_public_body_motion_described_not_admitted', rows=rows, contrasts=contrasts,
                  report_sha256=REPORT_SHA, strict_receipt_sha256=RECEIPT_SHA, new_neural_calls=0, new_physics_calls=0,
                  task_success=None, P5_admission=False,
                  limitation='Returned control-step contact counters and public poses; not all internal contact configurations or global controllability.')
    data = json.dumps(result, sort_keys=True, allow_nan=False).encode()
    output = ROOT/'results/audit_p5_word_motion_MAIN_20261028_v1_MAIN01.json'
    with output.open('xb') as stream:
        require(stream.write(data) == len(data), 'exclusive complete receipt')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(output=str(output), sha256=hashlib.sha256(data).hexdigest(),
                          all12_block_pose_unchanged=all(r['block_pose_bitwise_unchanged'] for r in rows),
                          all120_returned_contact_counts_zero=all(not any(r['returned_step_contact_counts']) for r in rows),
                          contrasts=contrasts)))


if __name__ == '__main__':
    main()
