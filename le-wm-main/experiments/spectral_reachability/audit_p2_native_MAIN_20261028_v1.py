"""Independent native7 arithmetic for all eight sealed P2 physical traces."""
import json
import os
from pathlib import Path

import numpy as np

from experiments.spectral_reachability.audit_kinodynamic_native_MAIN_20261028_v1 import require, sha

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = 'ccb790aeb65313b391abaf9fcb671c24b8246f9e27a1e1bb162a3c233b1654b5'


def audit(folder, expected):
    folder = Path(folder)
    require(sha(folder/'report.json') == expected, 'external P2 report seal')
    report = json.loads((folder/'report.json').read_bytes())
    require(report['status'] == 'calibration_recorded' and not report['errors'] and len(report['rows']) == 8, 'complete original eight rows')
    rows, checked = [], 0
    for outer in report['rows']:
        physical = outer['physical']
        goal = np.asarray(physical['context_identity']['goal'], np.float64)
        require(goal.shape == (7,) and len(physical['trace']) == 10, 'full native7 goal and raw10 trace')
        for step, event in enumerate(physical['trace'], 1):
            state = np.asarray(event['state'], np.float64)
            require(state.shape == (7,) and np.isfinite(state).all(), 'finite native7 actual state')
            delta = state-goal
            angle = abs(float(delta[4]))
            success = bool(np.linalg.norm(delta[:4]) < 20 and min(angle, 2*np.pi-angle) < np.pi/9)
            require(event['primitive_step'] == step and event['terminated'] == success and
                event['truncated'] is False and np.isclose(event['reward'], -np.linalg.norm(delta), rtol=0, atol=1e-10), 'every real reward/stop independently derived')
            checked += 1
        official = physical['official']
        require(np.isclose(official['official_position_error'], np.linalg.norm(delta[:4]), rtol=0, atol=1e-10) and
            np.isclose(official['official_state_distance'], np.linalg.norm(delta), rtol=0, atol=1e-10) and
            np.isclose(official['object_position_error'], np.linalg.norm(delta[2:4]), rtol=0, atol=1e-10) and
            official['official_success'] == success, 'all final metric axes separately checked')
        rows.append(dict(root_slot=outer['root_slot'], arm=outer['arm'], native_success=success,
            jointXY=float(np.linalg.norm(delta[:4])), native7=float(np.linalg.norm(delta)),
            objectXY=float(np.linalg.norm(delta[2:4]))))
    require(checked == 80 and sha(folder/'report.json') == expected, 'all80, unchanged input')
    return dict(schema='MAIN.p2_native_audit.v1', report_sha256=expected, rows=rows,
        native_returns_verified=checked, known_roots=4, new_neural_calls=0, new_physics_calls=0,
        P1_admission=False, P2_admission=False, independent_confirmation=False)


def main():
    result = audit(ROOT/'results/agent_p2_composition_execution_20261028_v1_physical_MAIN01', EXPECTED)
    destination = ROOT/'results/audit_p2_native_MAIN_20261028_v1_MAIN01.json'
    payload = json.dumps(result, sort_keys=True, allow_nan=False).encode()
    with destination.open('xb') as stream:
        require(stream.write(payload) == len(payload), 'complete independent receipt')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(output=str(destination), sha256=sha(destination), checked=result['native_returns_verified'])))


if __name__ == '__main__':
    main()
