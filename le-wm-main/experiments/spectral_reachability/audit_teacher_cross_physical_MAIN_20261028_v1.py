"""Saved-only independent all40 state/contact-label audit, no model/env imports."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / 'results/agent_p1_teacher_cross_library_physical_20261028_v1_MAIN02'


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def load(path, expected):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('external artifact SHA')
    return json.loads(data)


def main():
    report = load(DIRECTORY / 'report.json', '464656755bb83cec14b7acdd457641592b11be6d335f285166687df15368bc01')
    receipt = load(ROOT / 'results/agent_p1_teacher_cross_library_physical_20261028_v1_consume_MAIN02.json',
                   '44acd71c60c8edc7fbfa196c2f46f7d8692ef487ec0c3a01b6b8f7e5071e9212')
    if receipt['status'] != 'partial_teacher_cross_library40_preserved_not_admitted' or receipt['rows'] != report['rows']:
        raise ValueError('preserved partial authority, never full release')
    if len(report['rows']) != 40:
        raise ValueError('all40 denominator')
    raw = {r['job_digest']: [] for r in report['rows']}
    previous = None
    hasher = hashlib.sha256()
    with (DIRECTORY / 'journal.jsonl').open('rb') as stream:
        for index, line in enumerate(stream):
            hasher.update(line)
            event = json.loads(line)
            content = {k: v for k, v in event.items() if k != 'event_sha256'}
            if event['index'] != index or event['previous_sha256'] != previous or hashlib.sha256(encoded(content)).hexdigest() != event['event_sha256']:
                raise ValueError('complete independent event chain')
            previous = event['event_sha256']
            if event['kind'] == 'raw_return':
                raw[event['job_digest']].append(event)
    if hasher.hexdigest() != report['journal']['sha256'] or previous != report['journal_tail']:
        raise ValueError('journal final binding')
    rows = []
    for row in report['rows']:
        events = raw[row['job_digest']]
        if len(events) != 20 or [e['phase'] for e in events] != ['warmup']*10+['prefix']*5+['suffix']*5:
            raise ValueError('all20 actual returns required')
        for e, command in zip(events[10:], row['commands'], strict=True):
            bits = np.asarray(command, '<f4').tobytes().hex()
            if e['raw_bits'] != e['actual_bits'] or e['actual_bits'] != bits or e['terminated'] or e['truncated']:
                raise ValueError('same actual raw10, no native padding')
        state = np.asarray(events[-1]['state'], np.float64)
        goal = np.asarray(row['goal'], np.float64)
        allxy = float(np.linalg.norm(state[:4]-goal[:4]))
        angle = abs(float(state[4]-goal[4]))
        success = bool(allxy < 20 and min(angle, 2*np.pi-angle) < np.pi/9)
        if success != events[-1]['native_success'] or not np.isclose(allxy, row['last_return']['diagnostics']['independent']['all_xy_position_error'], rtol=1e-13, atol=1e-13):
            raise ValueError('native success/independent XY arithmetic')
        base = np.asarray(events[9]['state'][2:5], np.float64)
        changed = [not np.array_equal(np.asarray(e['state'][2:5]), base) for e in events[10:]]
        contacts = [e['info']['tree']['n_contacts'] for e in events]
        rows.append(dict(index=row['index'], task=row['task'], role=row['role'], library_label=row['library_label'],
                         status=row['status'], actual20_returns=True, native_success_at_last_return=success,
                         terminal_allxy=allxy, block_pose_changed_after_root=any(changed),
                         returned_contact_labels=contacts, RGB_completed=row['status']=='horizon_completed'))
    failed = report['rows'][12]
    png = DIRECTORY / failed['last_observation']['png']
    if hashlib.sha256(png.read_bytes()).hexdigest() != failed['last_observation']['file_sha256']:
        raise ValueError('failed final RGB binding')
    pixels = np.asarray(Image.open(png))
    blue = int(np.count_nonzero(np.all(pixels == [65, 105, 225], axis=-1)))
    failed_state = failed['last_return']['state']
    result = dict(status='all40_actual_raw_returns_audited_not_full_RGB_release', rows=rows,
                  native_successes_at_actual_last_return=sum(r['native_success_at_last_return'] for r in rows),
                  rows_with_block_pose_change_after_root=sum(r['block_pose_changed_after_root'] for r in rows),
                  nonzero_returned_contact_labels=sum(v != 0 for r in rows for v in r['returned_contact_labels']),
                  failed_observation=dict(index=12, canonical_blue_pixels=blue, agent_xy=failed_state[:2],
                                          canvas_violation=not all(0 <= v <= 512 for v in failed_state[:4])),
                  scientific_admission=False, independent_roots=0, known_roots=2,
                  full40_RGB_release=False, no_new_model_or_physics=True)
    output = ROOT / 'results/audit_teacher_cross_physical_MAIN_20261028_v1_MAIN02.json'
    payload = encoded(result)
    with output.open('xb') as stream:
        if stream.write(payload) != len(payload):
            raise ValueError('short receipt')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({**{k: v for k, v in result.items() if k != 'rows'}, 'output': str(output),
                      'sha256': hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == '__main__':
    main()
