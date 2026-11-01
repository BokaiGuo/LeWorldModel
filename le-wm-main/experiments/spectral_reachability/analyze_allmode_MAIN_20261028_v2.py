"""Independent arithmetic for full42 versus original10; no neural imports."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT/'results/p1_moving_overlay_recompile_cuda_20261027_v1_cuda_MAIN01'
PARENT_SHA = '2b2e5064f74f8aca712ae0d0415fcbfb64a2c8e6325bdffe24ae4fdc3b373381'
SUBSET = (0, 1, 2, 3, 4, 5, 22, 23, 24, 25)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    body = path.read_bytes()
    return json.loads(gzip.decompress(body) if path.suffix == '.gz' else body)


def cost(state, goal):
    d = np.asarray(state, np.float64)-np.asarray(goal, np.float64)
    require(d.shape == (192,) and np.isfinite(d).all(), 'literal finite original latent')
    return float(np.mean(d**2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--report-sha', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    directory = Path(args.input).resolve()
    require(sha(directory/'report.json') == args.report_sha and sha(PARENT/'report.json') == PARENT_SHA,
            'externally fixed actual reports')
    report, parent = read(directory/'report.json'), read(PARENT/'report.json')
    require(report['status'] == 'all4_allmode_plans_sealed_not_admitted' and not report['errors'] and
            report['model_unchanged'] and report['source_verified'], 'full terminal result only')
    prepared = read(report['prepared'])
    require(sha(report['prepared']) == report['prepare_sha256'], 'actual preparation binding')
    bindings = {str(directory/'report.json'): args.report_sha, str(PARENT/'report.json'): PARENT_SHA,
                report['prepared']: report['prepare_sha256']}
    rows = []
    for index, (slot, row) in enumerate(zip(prepared['slots'], report['rows'], strict=True)):
        result = row['result']; candidates = result['candidates']; goal = slot['packet']['goal']
        old_row = parent['rows'][2*index+1]
        old_path = PARENT/old_row['result_file']; task_path = PARENT/old_row['task_file']
        for path in (old_path, task_path):
            expected = parent['artifact_sha256'][path.name]
            require(sha(path) == expected, 'old full result/task source')
            bindings[str(path)] = expected
        old, task = read(old_path), read(task_path)
        require(row['index'] == index and row['root_slot'] == old_row['root_slot'] == slot['root_slot'] and
                row['condition'] == old_row['condition'] == slot['condition'] and
                old_row['arm'] == 'full10_conditional_D2P' and old_row['task_id'] == slot['task_id'], 'same four tasks')
        require(slot['packet']['root_context'] == task['task']['root_context'] and
                goal == task['task']['goal'] and np.array_equal(slot['q'], np.eye(192)), 'unchanged original task/Q')
        require(len(candidates) == 42 and len(old['candidates']) == 10, 'all original and extended candidates')
        original_subset_equal = []
        for old_index, new_index in enumerate(SUBSET):
            oc, nc = old['candidates'][old_index], candidates[new_index]
            equal = all(oc[k] == nc[k] for k in ('prefix_params', 'prefix_commands', 'suffix_params',
                        'suffix_commands', 'predicted_prefix', 'predicted_terminal', 'objective',
                        'stage_cost', 'own_reach_value', 'nonlinear_terminal_cost', 'context'))
            original_subset_equal.append(equal)
        require(all(original_subset_equal), 'full42 retained original10 numerical execution unchanged')
        for i, candidate in enumerate(candidates):
            require(candidate['index'] == i and candidate['completed'] is True and
                    candidate['stage_cost'] == cost(candidate['predicted_prefix'], goal) and
                    candidate['nonlinear_terminal_cost'] == cost(candidate['predicted_terminal'], goal) and
                    candidate['objective'] == candidate['stage_cost']+candidate['own_reach_value'], 'actual candidate objective')
        winner = min(candidates, key=lambda c: (c['objective'], c['index']))
        greedy = min(candidates, key=lambda c: (c['stage_cost'], c['index']))
        initial = cost(slot['packet']['root_context']['latent_history'][-1], goal)
        require(result['winner'] == winner and row['plan']['commands'] == winner['commands'], 'actual selected raw command')
        non_greedy=[c for c in candidates if c['stage_cost'] >= initial]
        frontier=[c for c in candidates if not any(
            d['stage_cost'] <= c['stage_cost'] and d['own_reach_value'] <= c['own_reach_value'] and
            (d['stage_cost'] < c['stage_cost'] or d['own_reach_value'] < c['own_reach_value'])
            for d in candidates)]
        rows.append(dict(index=index, root_slot=slot['root_slot'], condition=slot['condition'],
            old_selected_index=old['winner']['index'], selected_index=winner['index'],
            selected_kind=winner['proposal_kind'], greedy_index=greedy['index'],
            selected_equals_greedy=winner['index'] == greedy['index'],
            initial_Q=initial, selected_prefix_Q=winner['stage_cost'],
            selected_terminal_Q=winner['nonlinear_terminal_cost'],
            selected_predicted_non_greedy=winner['stage_cost'] >= initial,
            selected_raw_unchanged=winner['commands'] == old['winner']['commands'],
            new_mode_selected=winner['index'] not in SUBSET,
            original10_unchanged=original_subset_equal,
            count_predicted_non_greedy=len(non_greedy),
            pareto_indices=[c['index'] for c in frontier],
            non_greedy_pareto_indices=[c['index'] for c in frontier if c['stage_cost'] >= initial],
            min_non_greedy_reach=min((c['own_reach_value'] for c in non_greedy),default=None),
            min_non_greedy_terminal_Q=min((c['nonlinear_terminal_cost'] for c in non_greedy),default=None),
            winner_reach=winner['own_reach_value'],
            winner_objective=winner['objective'], old_winner_objective=old['winner']['objective']))
    require(len(rows) == 4, 'fixed full four conditions')
    for path, expected in bindings.items():
        require(sha(path) == expected, 'final input bytes unchanged')
    output = Path(args.output).resolve()
    require(output.parent == ROOT/'results' and output.name.startswith('analyze_allmode_MAIN_20261028_v2_'),
            'new MAIN analysis outside producer directory')
    result = dict(schema='MAIN.allmode42_independent.v2', input_sha256=bindings, rows=rows,
        resources=report['resources'], math_resources=report['math_resources'],
        known_roots=2, new_independent_roots=0, P1_admission=False, new_physics=0,
        new_neural_calls=0, equal_total_compute=False,
        caveat='allmode basis enumeration is not exhaustive continuous prefix search; no physical result here')
    with output.open('xb') as stream:
        stream.write((json.dumps(result, sort_keys=True, allow_nan=False)+'\n').encode())
        stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(dict(output=str(output), sha256=sha(output), rows=rows)))


if __name__ == '__main__':
    main()

