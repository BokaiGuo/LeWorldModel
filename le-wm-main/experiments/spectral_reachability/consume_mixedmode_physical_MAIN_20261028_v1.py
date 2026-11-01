"""Strict mixed physical acceptance and independent paired native arithmetic."""
import json
import os
import numpy as np
from experiments.spectral_reachability import agent_mixedmode_physical_20261028_v1 as producer


def independent(row):
    goal = np.asarray(row['goal'], np.float64)
    state = np.asarray(row['final_state'], np.float64)
    delta = state-goal
    joint = float(np.linalg.norm(delta[:4]))
    angle = float(abs(delta[4]))
    native_angle = min(angle, float(2*np.pi-angle))
    distance = float(np.linalg.norm(delta))
    success = bool(joint < 20 and native_angle < np.pi/9)
    if success != row['native_success'] or not np.isclose(distance,
            row['final_metrics']['native_distance'], rtol=0, atol=1e-12):
        raise ValueError('independent native task metric mismatch')
    return dict(joint_xy_error=joint, native_angle_term=native_angle,
                native7_distance=distance, native_success=success,
                checkpoints=[dict(step=c['step'], joint_xy_error=float(np.linalg.norm(
                    np.asarray(c['state'], np.float64)[:4]-goal[:4]))) for c in row['checkpoints']])


def main():
    folder = producer.ROOT/'results/agent_mixedmode_physical_20261028_v1_MAIN01'
    expected = '58bd3b4fcb2042e79d796f3eb26e0004f88b098c70ffdb044d1b03a3eedcc965'
    output = producer.ROOT/'results/consume_mixedmode_physical_MAIN_20261028_v1_MAIN01.json'
    if output.exists():
        raise FileExistsError(output)
    receipt = producer.consume(folder, expected)
    old = json.loads((producer.OLD_OUTPUT/'report.json').read_bytes())
    if producer.sha(producer.OLD_OUTPUT/'report.json') != producer.OLD_REPORT_SHA:
        raise ValueError('old actual report seal')
    numerical = []
    for row, previous in zip(receipt['rows'], old['rows'], strict=True):
        evidence = row['inherited_evidence'] if row['route'] == 'inherit' else row
        if (row['root_slot'], row['condition']) != (previous['root_slot'], previous['condition']):
            raise ValueError('paired task identity mismatch')
        current, prior = independent(evidence), independent(previous)
        numerical.append(dict(root_slot=row['root_slot'], condition=row['condition'],
            origin=row['evidence_origin'], current=current, prior=prior,
            joint_xy_change=current['joint_xy_error']-prior['joint_xy_error']))
    result = dict(schema='MAIN.mixedmode_physical_receipt.v1', receipt=receipt,
        independent_native_metrics=numerical, new_neural_calls=0, new_physics_calls=0,
        P1_admission=False, freshness=False, equal_total_compute=False,
        caveat='Two known roots/four conditions, two newly executed and two inherited; not independent confirmation or non-greedy mechanism.')
    payload = producer.raw(result)
    with output.open('xb') as stream:
        if stream.write(payload) != len(payload):
            raise OSError('short receipt write')
        stream.flush(); os.fsync(stream.fileno())
    producer.paired.fsync_dir(output.parent)
    print(producer.source.canonical(dict(output=str(output), sha256=producer.sha(output),
        status=receipt['status'], independent_native_metrics=numerical)))


if __name__ == '__main__':
    main()
