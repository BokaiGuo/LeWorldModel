"""Independent arithmetic over MAIN-accepted probes; never imports the analyzer."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def diff(a, b, space):
    slices = dict(joint_xy=slice(0, 4), agent_xy=slice(0, 2), object_xy=slice(2, 4), object_angle_wrapped=slice(4, 5))
    delta = np.asarray(a, dtype=np.float64)[slices[space]]-np.asarray(b, dtype=np.float64)[slices[space]]
    return np.arctan2(np.sin(delta), np.cos(delta)) if space == 'object_angle_wrapped' else delta


def check_projection(actual, delta, direction):
    expected = float(delta @ direction)
    if actual is None or not np.isclose(actual, expected, rtol=0, atol=1e-12):
        raise ValueError('independent signed projection mismatch')
    return expected


def main():
    result_dir = ROOT/'results'
    analysis_path = result_dir/'agent_mixedmode_capability_analysis_20261028_v1_MAIN01.json'
    receipt_path = result_dir/'consume_mixedmode_capability_MAIN_20261028_v1_MAIN01.json'
    prepared_path = result_dir/'agent_mixedmode_capability_probe_20261028_v2_prepare_MAIN01.json'
    expected = {
        str(analysis_path): '9f1cf0f84f8199ec5336ff9ef7b4a24293d99534c37296c136b9fe4ec4d2389d',
        str(receipt_path): 'd1efceb6aa0eb4e72a586bb0c1ba297ecb433f12e048e7a0c966450b71156537',
        str(prepared_path): 'e8e4554ea2448a8c4bcf4e916934800fa8c7afe3eda335174056a551620da3d5'}
    for path, digest in expected.items():
        if sha(path) != digest:
            raise ValueError('external actual input SHA')
    analysis, receipt, prepared = [json.loads(path.read_bytes()) for path in (analysis_path, receipt_path, prepared_path)]
    for path, digest in analysis['own_file_sha256'].items():
        if sha(path) != digest:
            raise ValueError('analyzer source/tests/doc drift')
    accepted = receipt['receipt']
    if accepted['status'] != 'saved_probe_responses_not_admitted' or accepted['fixed_denominator'] != 40:
        raise ValueError('actual full MAIN acceptance required')
    endpoints = {(r['task'], r['role'], r['probe_id']): r['end_state'] for r in accepted['rows'] if r['status'] == 'completed'}
    if len(endpoints) != 40 or len(analysis['rows']) != 8:
        raise ValueError('full40 endpoint denominator')
    projections, secants, comparisons = 0, 0, []
    for group in analysis['rows']:
        task = prepared['tasks'][group['task']]['native']
        for space, values in group['spaces'].items():
            residual = diff(task['goal'], task['root_state'], space)
            norm = np.linalg.norm(residual)
            if norm == 0:
                raise ValueError('current actual task has unexpected zero residual; do not invent direction')
            direction = residual/norm
            if not np.array_equal(direction, values['original_task_direction']):
                raise ValueError('common original task direction')
            key = (group['task'], group['role'])
            zero = endpoints[(*key, 'zero')]
            for response in values['finite_responses']:
                delta = diff(endpoints[(*key, response['probe_id'])], zero, space)
                if not np.array_equal(delta, response['zero_relative_delta']):
                    raise ValueError('zero-relative full response')
                check_projection(response['signed_original_task_projection'], delta, direction)
                projections += 1
            for index, axis in enumerate(('x', 'y')):
                delta = diff(endpoints[(*key, '+'+axis)], endpoints[(*key, '-'+axis)], space)/(2*group['probe_amplitude_raw'])
                check_projection(values['signed_original_task_secants'][index], delta, direction)
                secants += 1
    for comparison in analysis['comparisons']:
        task = comparison['task']
        before, after = analysis['rows'][2*task:2*task+2]
        for space, values in comparison['spaces'].items():
            for row, first, last in zip(values['signed_response_changes'], before['spaces'][space]['finite_responses'], after['spaces'][space]['finite_responses'], strict=True):
                if row['after_minus_before'] != last['signed_original_task_projection']-first['signed_original_task_projection']:
                    raise ValueError('signed before/after difference')
        comparisons.append(dict(task=task, root_slot=comparison['root_slot'], condition=comparison['condition'],
            joint_xy_secant_changes=comparison['spaces']['joint_xy']['signed_secant_changes'],
            causal_capability_gain='unknown; time/state confounded'))
    for path, digest in expected.items():
        if sha(path) != digest:
            raise ValueError('inputs changed during independent arithmetic')
    result = dict(schema='MAIN.mixedmode_signed_independent.v1', input_sha256=expected,
        signed_projections_checked=projections, signed_secants_checked=secants, comparisons=comparisons,
        P1_admission=False, independent_confirmation=False, new_neural_calls=0, new_physics_calls=0)
    destination = result_dir/'audit_mixedmode_signed_MAIN_20261028_v1_MAIN01.json'
    payload = json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    with destination.open('xb') as stream:
        if stream.write(payload) != len(payload):
            raise OSError('short independent write')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(output=str(destination), sha256=sha(destination), result=result), sort_keys=True))


if __name__ == '__main__':
    main()
