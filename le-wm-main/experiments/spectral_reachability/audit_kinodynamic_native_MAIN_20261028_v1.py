"""Independent saved-only native arithmetic and journal census, no producer import."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = '01d5485208fe40a71a53ae20bb2920740894d6ef7c114c139a991f4707e5ebe7'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def audit(folder, expected):
    folder = Path(folder)
    require(sha(folder/'report.json') == expected, 'external report seal')
    report = json.loads((folder/'report.json').read_bytes())
    require(report['errors'] == [] and len(report['rows']) == 2 and
        report['status'] == 'matched_domain_beam_physical_recorded_not_admitted', 'complete fixed two rows')
    require(sha(folder/'journal.jsonl') == report['journal_sha256'], 'full journal file')
    events = [json.loads(x) for x in (folder/'journal.jsonl').read_bytes().splitlines()]
    previous = None
    counts = {k: dict(attempted=0, completed=0) for k in report['counts']}
    pending, returned, images = {}, {}, 0
    for index, event in enumerate(events):
        body = {k: v for k, v in event.items() if k != 'event_sha256'}
        # This source journal owns the default-space JSON dialect, not runtime compact JSON.
        digest = hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest()
        require(event['index'] == index and event['previous_sha256'] == previous and
            digest == event['event_sha256'], 'independent full event hash chain')
        previous = digest
        kind = event['kind']
        if kind in ('operation_dispatch_intent', 'operation_returned'):
            receipt = dict(event['receipt'])
            status = receipt.pop('status')
            ident, resource = receipt['operation_index'], receipt['resource']
            if kind == 'operation_dispatch_intent':
                require(ident not in pending and ident not in returned and status == 'pending', 'unique actual intent')
                pending[ident] = receipt
                counts[resource]['attempted'] += 1
            else:
                require(pending.pop(ident, None) == receipt and status == 'returned', 'exact intent to actual return')
                returned[ident] = dict(receipt, status=status)
                counts[resource]['completed'] += 1
        elif kind == 'fresh_observation':
            receipt = event['receipt']
            name = receipt['png']
            require(Path(name).name == name and sha(folder/name) == receipt['file_sha256'], 'PNG actual byte identity')
            with Image.open(folder/name) as im:
                pixels = np.asarray(im.convert('RGB'))
            require(pixels.shape == (224, 224, 3) and
                hashlib.sha256(pixels.tobytes()).hexdigest() == receipt['pixel_sha256'], 'all actual RGB pixels')
            require(receipt['state'] == receipt['observation_state'] == receipt['snapshot']['state'], 'observed full native7')
            images += 1
        elif kind == 'returned_transition' and event['phase'] == 'plan':
            row = report['rows'][event['branch']]
            delta = np.asarray(event['state'])-np.asarray(row['goal'])
            angle = abs(float(delta[4]))
            success = bool(np.linalg.norm(delta[:4]) < 20 and min(angle, 2*np.pi-angle) < np.pi/9)
            require(np.isclose(-np.linalg.norm(delta), event['reward'], rtol=0, atol=1e-10) and
                success == event['terminated'] and event['truncated'] is False, 'all20 plan native rewards and stops')
    require(not pending and counts == report['counts'] and images == 44 and
        events[-1]['kind'] == 'run_finished' and previous == report['journal_chain_tail'], 'complete terminal resource census')
    require([returned[i] for i in sorted(returned)] == report['actual_operation_receipts'], 'complete operation receipt joins')
    endpoints = []
    for row in report['rows']:
        state, goal = np.asarray(row['final_state']), np.asarray(row['goal'])
        delta = state-goal
        metrics = row['final_metrics']
        require(np.isclose(np.linalg.norm(delta), metrics['native_distance'], rtol=0, atol=1e-10) and
            np.isclose(np.linalg.norm(delta[2:4]), metrics['object_position_error'], rtol=0, atol=1e-10), 'independent endpoint distances')
        require(row['full_horizon'] and row['own_returned'] == 10, 'both complete raw10')
        endpoints.append(dict(root_slot=row['root_slot'], native7_distance=float(np.linalg.norm(delta)),
            object_position_error=float(np.linalg.norm(delta[2:4])), native_success=row['native_success']))
    require(sha(folder/'report.json') == expected, 'report unchanged during audit')
    return dict(schema='MAIN.kinodynamic_native_audit.v1', report_sha256=expected,
        journal_events=len(events), verified_RGB=images, counts=counts, endpoints=endpoints,
        neural_calls=0, physics_calls=0, independent_confirmation=False, P1_admission=False)


def main():
    result = audit(ROOT/'results/agent_kinodynamic_matched_physical_20261028_v1_MAIN01', EXPECTED)
    destination = ROOT/'results/audit_kinodynamic_native_MAIN_20261028_v1_MAIN01.json'
    payload = json.dumps(result, sort_keys=True, allow_nan=False).encode()
    with destination.open('xb') as stream:
        require(stream.write(payload) == len(payload), 'complete new receipt')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(output=str(destination), sha256=sha(destination), result=result)))


if __name__ == '__main__':
    main()
