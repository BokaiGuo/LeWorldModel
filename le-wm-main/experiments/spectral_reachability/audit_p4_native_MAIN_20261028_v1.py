"""Independent saved RGB/native audit; no driver, model or simulator imports."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.spectral_reachability.audit_kinodynamic_native_MAIN_20261028_v1 import require, sha

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = '60489b7db3d4eff70d1be46d47d92209e6adcf333ae1d0a0e389b259ec1f6084'


def audit(folder, expected):
    folder = Path(folder)
    require(sha(folder/'report.json') == expected, 'external real report SHA')
    report = json.loads((folder/'report.json').read_bytes())
    require(report['errors'] == [] and report['pending'] == {} and
        report['status'] == 'known_moving_multiboundary_recorded' and len(report['rows']) == 8, 'all eight real rows')
    require(sha(folder/'journal.jsonl') == report['journal_sha256'], 'complete journal byte hash')
    events = [json.loads(x) for x in (folder/'journal.jsonl').read_bytes().splitlines()]
    previous = None
    observations, controls = {}, {}
    images = 0
    for index, event in enumerate(events):
        body = {k: v for k, v in event.items() if k != 'event_sha256'}
        digest = hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest()
        require(event['index'] == index and event['previous_sha256'] == previous and
            digest == event['event_sha256'], 'independent journal full chain')
        previous = digest
        if event['kind'] == 'fresh_observation':
            receipt = event['receipt']
            path = Path(receipt['path'])
            require(path.parent == folder and not path.is_symlink() and sha(path) == receipt['file_sha256'], 'own real PNG bytes')
            with Image.open(path) as im:
                require(im.mode == 'RGB' and im.size == (224, 224), 'native RGB224')
                pixels = np.asarray(im)
            require(hashlib.sha256(pixels.tobytes()).hexdigest() == receipt['pixel_sha256'], 'all native pixels')
            require(receipt['state'] == receipt['snapshot']['state'], 'full native7 snapshot identity')
            key = (event['episode_id'], receipt['phase'], receipt['step'])
            require(key not in observations, 'unique actual observation')
            observations[key] = receipt
            images += 1
        elif event['kind'] == 'control_return':
            eid, step = event['episode_id'], event['step']
            receipt = observations[eid, 'plan', step]
            delta = np.asarray(receipt['state'])-np.asarray(receipt['goal'])
            angle = abs(float(delta[4]))
            success = bool(np.linalg.norm(delta[:4]) < 20 and min(angle, 2*np.pi-angle) < np.pi/9)
            require(np.isclose(-np.linalg.norm(delta), event['reward'], rtol=0, atol=1e-10) and
                event['terminated'] == success and event['truncated'] is False, 'every real native reward and stop')
            returns = controls.setdefault(eid, [])
            require(step == len(returns)+1 and (not returns or not returns[-1]['terminated']), 'contiguous returns, no post-stop dispatch')
            returns.append(event)
    require(images == 247 and previous == report['journal_chain_tail'] and
        events[-1]['kind'] == 'run_finished', 'all actual frames and terminal chain')
    rows = []
    for row in report['rows']:
        eid = row['episode_id']
        returns = controls[eid]
        require(len(returns) == row['control_returned'] and [e['command'] for e in returns] == row['executed_raw'], 'actual executed commands join')
        last = observations[eid, 'plan', len(returns)]
        metrics = row['final_metrics']
        delta = np.asarray(last['state'])-np.asarray(last['goal'])
        require(metrics['native_success'] == row['native_success'] == returns[-1]['terminated'] and
            np.isclose(metrics['native_distance'], np.linalg.norm(delta), rtol=0, atol=1e-10), 'terminal native result from real observed state')
        rows.append(dict(episode_id=eid, control_returned=len(returns), native_success=row['native_success'],
            object_position_error=float(np.linalg.norm(delta[2:4])), native7_distance=float(np.linalg.norm(delta))))
    require(sum(len(v) for v in controls.values()) == 151 and sha(folder/'report.json') == expected, 'all control returns; unchanged report')
    return dict(schema='MAIN.p4_native_audit.v1', report_sha256=expected, journal_events=len(events),
        verified_RGB=images, verified_native_control_returns=151, rows=rows, known_roots=2,
        neural_calls=0, physics_calls=0, P1_admission=False, P4_admission=False, independent_confirmation=False)


def main():
    result = audit(ROOT/'results/agent_p4_moving_multiboundary_20261028_v1_MAIN02', EXPECTED)
    destination = ROOT/'results/audit_p4_native_MAIN_20261028_v1_MAIN01.json'
    payload = json.dumps(result, sort_keys=True, allow_nan=False).encode()
    with destination.open('xb') as stream:
        require(stream.write(payload) == len(payload), 'complete independent receipt')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(output=str(destination), sha256=sha(destination), result=result)))


if __name__ == '__main__':
    main()
