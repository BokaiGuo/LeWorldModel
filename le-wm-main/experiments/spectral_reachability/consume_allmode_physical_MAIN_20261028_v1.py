"""Strict saved physical acceptance plus independent native metrics; no model."""
import json
import os

import numpy as np

from experiments.spectral_reachability import agent_allmode_physical_20261028_v1 as physical


def main():
    root = physical.ROOT
    folder = root / 'results/agent_allmode_physical_20261028_v1_MAIN01'
    expected = '3ca4891a5f0823bede431e85442513a095b1e9bc8d49c87feefbb56663dca3b4'
    output = root / 'results/consume_allmode_physical_MAIN_20261028_v1_MAIN01.json'
    if output.exists():
        raise FileExistsError(output)
    receipt = physical.consume(folder, expected)
    numerical = []
    for row in receipt['rows']:
        goal = np.asarray(row['goal'], dtype=np.float64)
        endpoint = np.asarray(row['final_state'], dtype=np.float64)
        joint_xy = float(np.linalg.norm(goal[:4] - endpoint[:4]))
        angle = float(abs(goal[4] - endpoint[4]))
        native_angle = min(angle, float(2 * np.pi - angle))
        native_distance = float(np.linalg.norm(goal - endpoint))
        success = bool(joint_xy < 20 and native_angle < np.pi / 9)
        if success != row['native_success'] or not np.isclose(
                native_distance, row['final_metrics']['native_distance'], rtol=0, atol=1e-12):
            raise ValueError('independent native endpoint metric mismatch')
        checkpoints = []
        for checkpoint in row['checkpoints']:
            state = np.asarray(checkpoint['state'], dtype=np.float64)
            checkpoints.append(dict(step=checkpoint['step'],
                joint_xy_error=float(np.linalg.norm(goal[:4] - state[:4]))))
        numerical.append(dict(root_slot=row['root_slot'], condition=row['condition'],
            joint_xy_error=joint_xy, native_angle_term=native_angle,
            native_distance_including_velocity=native_distance,
            native_success=success, checkpoints=checkpoints))
    result = dict(schema='MAIN.allmode_physical_receipt.v1', receipt=receipt,
        independent_native_metrics=numerical, new_neural_calls=0, new_physics_calls=0,
        P1_admission=False, freshness=False,
        caveat='Four root-condition branches from two known roots; unchanged selected actions; not a new mechanism or independent confirmation.')
    payload = physical.raw(result)
    with output.open('xb') as stream:
        if stream.write(payload) != len(payload):
            raise OSError('short receipt write')
        stream.flush()
        os.fsync(stream.fileno())
    physical.paired.fsync_dir(output.parent)
    print(physical.source.canonical(dict(output=str(output), sha256=physical.sha(output),
        status=receipt['status'], independent_native_metrics=numerical)))


if __name__ == '__main__':
    main()
