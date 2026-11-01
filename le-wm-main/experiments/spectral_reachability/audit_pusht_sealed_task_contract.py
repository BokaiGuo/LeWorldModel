"""Re-label saved physical states with installed official evaluation, no neural work.

Does not change the object-only research task or select a new policy.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def labels(state, goal):
    state, goal = np.asarray(state, float), np.asarray(goal, float)
    if state.shape != (7,) or goal.shape != (7,) or not np.isfinite([state, goal]).all():
        raise ValueError('finite seven-state and goal required')
    difference = abs(goal[4]-state[4])
    # Deliberately match installed evaluator, not an improved circular metric.
    angle = min(difference, 2*np.pi-difference)
    return {'official_position_error': float(np.linalg.norm(goal[:4]-state[:4])),
            'agent_position_error': float(np.linalg.norm(goal[:2]-state[:2])),
            'object_position_error': float(np.linalg.norm(goal[2:4]-state[2:4])),
            'official_state_distance': float(np.linalg.norm(goal-state)),
            'official_success': bool(np.linalg.norm(goal[:4]-state[:4]) < 20 and angle < np.pi/9),
            'object_only_success': bool(np.linalg.norm(goal[2:4]-state[2:4]) < 20 and angle < np.pi/9),
            'angle_error': float(angle)}


def run(execution, output):
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal
    execution, output = Path(execution), Path(output)
    report = json.loads(execution.read_text())
    if report['status'] != 'completed' or not all(v is True for v in report['gates'].values()):
        raise ValueError('completed physical execution required')
    compilation_path = Path(report['protocol']['preflight'])
    if base.digest(compilation_path) != report['protocol']['preflight_sha256']:
        raise ValueError('compilation changed')
    compilation = json.loads(compilation_path.read_text())
    bindings = compilation['protocol']['bindings']
    if not all(base.digest(Path(p)) == d for p, d in bindings.items()):
        raise ValueError('upstream binding changed')
    candidates = [json.loads(Path(p).read_text()) for p in bindings if Path(p).name == 'report.json']
    intervention = next(r for r in candidates if 'bank_path' in r.get('protocol', {}))
    bank = json.loads(Path(intervention['protocol']['bank_path']).read_text())
    upstream = base.physical._load_preflight(Path(bank['upstream_path']))
    output.mkdir(exist_ok=False)
    env = base.gym.make('swm/PushT-v1', render_mode='rgb_array')
    evaluator_path = Path(env.unwrapped.eval_state.__code__.co_filename)
    inputs = {str(execution.resolve()): base.digest(execution), str(compilation_path.resolve()): base.digest(compilation_path), str(evaluator_path): base.digest(evaluator_path)}
    seal = create_seal(files=[Path(__file__), evaluator_path, Path(base.collect_nonterminal_boundary_contact_roots.__code__.co_filename)], args_json=json.dumps(inputs), output_dir=output/'source_seal')
    rows = []
    try:
        roots = base.collect_nonterminal_boundary_contact_roots(env, count=4, seed=int(upstream['protocol']['root_seed']), oversample_factor=4)
        physical = {r['root_sha256']: r for r in report['rows']}
        if len(physical) != 4 or set(physical) != {base.root_fingerprint(root) for root in roots}:
            raise ValueError('root identity mismatch')
        for root in roots:
            saved = physical[base.root_fingerprint(root)]
            branches = []
            for branch in saved['branches']:
                if branch['status'] != 'executed':
                    raise ValueError('complete execution required')
                stages = []
                for stage in branch['stages']:
                    value = labels(stage['state'], root['goal'])
                    success, distance = env.unwrapped.eval_state(np.asarray(root['goal']), np.asarray(stage['state']))
                    if bool(success) != value['official_success'] or not np.isclose(distance, value['official_state_distance'], atol=1e-10):
                        raise ValueError('independent evaluator disagrees with installed official code')
                    stages.append(value)
                branches.append({'candidate_id': branch['candidate_id'], 'stages': stages})
            rows.append({'root_sha256': saved['root_sha256'], 'root_index': saved['root_index'],
                         'goal_state': np.asarray(root['goal']).tolist(), 'selected_candidate_id': saved['selected_candidate_id'], 'branches': branches})
    finally:
        env.close()
    verified = verify_seal(output_dir=seal['output_dir'], expected_manifest_sha256=seal['manifest_sha256'])
    if not all(base.digest(Path(p)) == d for p, d in inputs.items()):
        raise ValueError('inputs changed')
    result = {'schema': 'pusht_sealed_task_contract.v1', 'status': 'completed', 'rows': rows,
              'inputs': inputs, 'source_seal': seal, 'source_verification': verified,
              'scope': 'offline saved-state re-labeling; no neural computation or new future execution',
              'P1_admission': False, 'task_changed_for_selection': False,
              'official_goal_includes_agent_position': True}
    base.write_new(output/'report.json', result)
    for row in rows:
        print(json.dumps({'root': row['root_index'], 'official_terminal_successes': sum(b['stages'][-1]['official_success'] for b in row['branches']),
                          'object_only_terminal_successes': sum(b['stages'][-1]['object_only_success'] for b in row['branches'])}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execution', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    run(**vars(p.parse_args()))
