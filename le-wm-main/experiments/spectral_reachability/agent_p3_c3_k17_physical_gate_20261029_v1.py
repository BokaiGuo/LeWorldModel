"""Saved-authority/live-capture K17 predispatch gate; deliberately no runner.

The fixed 64 MiB output proof is incomplete. This module cannot dispatch and
does not turn a caller-supplied capture into a native execution receipt.
"""
import gzip
import hashlib
import json
from pathlib import Path

from experiments.spectral_reachability import agent_p3_c3_k17_replay_guard_bridge_20261029_v1 as bridge
from experiments.spectral_reachability import agent_p3_c3_k17_physical_transport_20261029_v1 as transport


def need(ok, message):
    if not ok:
        raise ValueError(message)


def authorities(slot):
    """Read pinned MAIN receipts and actual paid native replay transcripts."""
    return _authorities_from_snapshot(slot, bridge.build()['rows'])


def _authorities_from_snapshot(slot, replay_rows):
    """Resolve one slot from an internally built snapshot; not an authority API."""
    rows = [r for r in replay_rows if r['slot'] == slot]
    need(len(rows) == 1, 'unknown K17 slot')
    item = rows[0]
    rec = bridge.handoff.pinned(bridge.handoff.RECEIPT, bridge.handoff.RECEIPT_SHA)
    task = bridge.handoff.pinned(rec['prepared'], rec['prepare_sha256'])['tasks'][item['root']]
    pp = bridge.handoff.pinned(bridge.PHYS_PREP, bridge.PHYS_PREP_SHA)
    report = bridge.handoff.pinned(bridge.handoff.PHYSICAL, bridge.handoff.PHYSICAL_SHA)
    receipt = bridge.handoff.pinned(bridge.handoff.PHYS_RECEIPT, bridge.handoff.PHYS_RECEIPT_SHA)
    path = bridge.ROOT / ('results/p1_constructive_moving_history_encode_cuda_20261027_v1_cuda_main01/'
                          f'task_{item["root"]:02}.json.gz')
    need(path.is_file() and not path.is_symlink() and bridge.sha(path) == task['binding']['task_sha256'],
         'task export bytes')
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        export = json.load(stream)
    ref = bridge.source_reference(item['root'], task, pp, report, receipt, export, bridge.sha(path))
    return item, task, ref


def capture_check(item, task, reference, live, output_dir):
    """Verify all actual PNG file hashes before the frozen live/history guard.

    `live` must have been captured by the real native adapter, which is not
    implemented here. Calling this with fixtures cannot certify native origin.
    """
    root = Path(output_dir)
    frames = live['warm_frames'] + [live['goal_frame']]
    need(len(frames) == 34, '33 warm frames and one plan0 frame required')
    names = set()
    for frame in frames:
        receipt = frame['receipt']
        name = receipt['png']
        need(type(name) is str and Path(name).name == name and name.endswith('.png') and name not in names,
             'unique relative PNG')
        names.add(name)
        p = root / name
        need(p.is_file() and not p.is_symlink() and hashlib.sha256(p.read_bytes()).hexdigest() == receipt['file_sha256'],
             'live PNG bytes')
    return bridge.check_live_only(item, task, reference, live)


def record_preflight(journal, item, task, reference, live, output_dir):
    """Durably record an accepted/rejected guard, without issuing any intent."""
    try:
        result = capture_check(item, task, reference, live, output_dir)
        need(result['physical_dispatch_ready'] is False, 'bridge must remain non-dispatching')
    except Exception as exc:
        journal.emit('guard_rejected', {'slot': item['slot'], 'error_type': type(exc).__name__})
        raise
    journal.emit('guard_checked', {'slot': item['slot'], 'result': result,
                                  'capacity_proved': False, 'dispatch_authorized': False})
    return result


def consume_preflight(path):
    """Audit only the journal seam, not native provenance or physical output."""
    events = transport.read_journal(path)
    for event in events:
        need(event['kind'] in ('guard_checked', 'guard_rejected'), 'unexpected event')
        payload = event['payload']
        need(type(payload['slot']) is int and payload['slot'] in (3, 4, 5, 9, 10, 11), 'slot identity')
        if event['kind'] == 'guard_checked':
            need(payload['capacity_proved'] is False and payload['dispatch_authorized'] is False and
                 payload['result']['physical_dispatch_ready'] is False, 'no dispatch authorization')
    return {'event_count': len(events), 'native_provenance_verified': False,
            'physical_dispatch_authorized': False, 'full_saved_consume': False}
