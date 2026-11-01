"""Independent saved arithmetic for the predeclared 222-prefix calibration.

No neural or physical calls. Strict producer consumption is performed first.
The first42 equality check is an engineering drift check, not a success gate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def summarize(report, original, prepared):
    if len(report['rows']) != 4 or len(original['rows']) != 4:
        raise ValueError('complete four-condition denominator required')
    rows = []
    for slot, row, old in zip(prepared['slots'], report['rows'], original['rows'], strict=True):
        result, reference = row['result'], old['result']
        candidates = result['candidates']
        if len(candidates) != 222 or candidates[:42] != reference['candidates']:
            raise ValueError('original42 full candidate reproduction failed')
        if result['root_nominal'] != reference['root_nominal'] or result['root_jacobian'] != reference['root_jacobian']:
            raise ValueError('original root dynamics reproduction failed')
        q = np.asarray(slot['q'], dtype=np.float64)
        goal = np.asarray(slot['packet']['goal'], dtype=np.float64)
        def cost(z):
            residual = q @ (np.asarray(z, dtype=np.float64) - goal)
            return float(np.dot(residual, residual) / residual.size)
        initial = cost(slot['packet']['root_context']['latent_history'][-1])
        for index, candidate in enumerate(candidates):
            if candidate['index'] != index:
                raise ValueError('candidate order mismatch')
            residual = q @ (goal - np.asarray(candidate['nominal_terminal']))
            step = np.asarray(candidate['suffix_params'], dtype=np.float64)
            remaining = residual - (q @ np.asarray(candidate['jacobian'])) @ step
            own = float((np.dot(remaining, remaining) + .002 * np.dot(step, step)) / residual.size)
            actual = (cost(candidate['predicted_prefix']), own, cost(candidate['predicted_terminal']))
            recorded = (candidate['stage_cost'], candidate['own_reach_value'], candidate['nonlinear_terminal_cost'])
            if not np.allclose(actual, recorded, atol=1e-12, rtol=0):
                raise ValueError('independent stage/reach/terminal arithmetic mismatch')
            if not np.isclose(actual[0] + actual[1], candidate['objective'], atol=1e-12, rtol=0):
                raise ValueError('fixed beta1 objective mismatch')
        winner = min(candidates, key=lambda c: (c['objective'], c['index']))
        greedy = min(candidates, key=lambda c: (c['stage_cost'], c['index']))
        frontier = [c['index'] for c in candidates if not any(
            d['stage_cost'] <= c['stage_cost'] and d['own_reach_value'] <= c['own_reach_value'] and
            (d['stage_cost'] < c['stage_cost'] or d['own_reach_value'] < c['own_reach_value'])
            for d in candidates)]
        rows.append(dict(root_slot=row['root_slot'], condition=row['condition'],
            original42_exact=True, selected_index=winner['index'], greedy_index=greedy['index'],
            selected_is_greedy=winner['index'] == greedy['index'],
            selected_raw_unchanged=winner['commands'] == reference['winner']['commands'],
            selected_initial_Q=initial, selected_prefix_Q=cost(winner['predicted_prefix']),
            selected_terminal_Q=cost(winner['predicted_terminal']),
            selected_prefix_worsens_initial=winner['stage_cost'] > initial,
            count_worse_than_initial=sum(c['stage_cost'] > initial for c in candidates),
            count_worse_than_stage_greedy=sum(c['stage_cost'] > greedy['stage_cost'] for c in candidates),
            pareto_indices=frontier, mixed_pareto_indices=[i for i in frontier if i >= 42],
            worse_than_initial_pareto_indices=[i for i in frontier if candidates[i]['stage_cost'] > initial]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-sha', required=True)
    args = parser.parse_args()
    from experiments.spectral_reachability import agent_mixedmode_prefix_20261028_v1 as producer
    folder = producer.ROOT / 'results/agent_mixedmode_prefix_20261028_v1_cuda_MAIN01'
    report_path = folder / 'report.json'
    old_path = producer.ROOT / 'results/agent_allmode_prefix_20261028_v2_cuda_MAIN01/report.json'
    old_sha = 'c3e38899a12cd163d75918da79ef029adb14b2479f74f15bdfac0b6ffc57242d'
    output = producer.ROOT / 'results/analyze_mixedmode_MAIN_20261028_v1_MAIN01.json'
    if output.exists():
        raise FileExistsError(output)
    if sha(report_path) != args.report_sha or sha(old_path) != old_sha:
        raise ValueError('external actual report seals required')
    receipt = producer.consume(folder, args.report_sha)
    report = json.loads(report_path.read_bytes())
    prepared = json.loads(Path(report['prepared']).read_bytes())
    original = json.loads(old_path.read_bytes())
    rows = summarize(report, original, prepared)
    if sha(report_path) != args.report_sha or sha(old_path) != old_sha:
        raise ValueError('input drift during analysis')
    value = dict(schema='MAIN.mixedmode222_analysis.v1', rows=rows, strict_receipt=receipt,
        input_sha256={str(report_path): args.report_sha, str(old_path): old_sha},
        resources=report['resources'], math_resources=report['math_resources'],
        known_roots=2, new_independent_roots=0, new_neural_calls=0, new_physics_calls=0,
        P1_admission=False, equal_total_compute=False,
        caveat='Fixed pairwise library only; model evidence does not establish real capability, success or independent confirmation.')
    payload = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    with output.open('xb') as stream:
        if stream.write(payload) != len(payload):
            raise OSError('short analysis write')
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    print(json.dumps(dict(output=str(output), sha256=sha(output), rows=rows), sort_keys=True))


if __name__ == '__main__':
    main()
