"""Saved P2 objective decomposition; no neural execution or scientific admission."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def metric(vector, compiled):
    goal = np.asarray(compiled['goal_vectors'][0], dtype=np.float64)
    projection = np.asarray(compiled['scoring']['metric']['projection'], dtype=np.float64)
    vector = np.asarray(vector, dtype=np.float64)
    require(vector.shape == goal.shape == (192,) and projection.shape == (192, 192),
            'expected saved 192-dimensional metric')
    require(all(np.isfinite(x).all() for x in (vector, goal, projection)), 'nonfinite metric input')
    residual = projection @ (vector - goal)
    return float(residual @ residual / 192)


def indexed(rows):
    require(len(rows) == 4, 'all four known roots required')
    result = {r['root_sha256']: r for r in rows}
    require(len(result) == 4, 'duplicate root')
    return result


def direction(value):
    return 'composition_lower' if value < 0 else 'flat_lower' if value > 0 else 'equal'


def decompose(compilation, planning, physical):
    compiled, planned, observed = (indexed(d['rows']) for d in (compilation, planning, physical))
    require(compiled.keys() == planned.keys() == observed.keys(), 'root identity census mismatch')
    result = []
    for root_id, root in compiled.items():
        plan_rows = planned[root_id]['rows']
        branches = observed[root_id]['branches']
        require(len(plan_rows) == len(branches) == 2, 'two arms required')
        plans = {r['arm']: r for r in plan_rows}
        physics = {r['arm']: r for r in branches}
        require(plans.keys() == physics.keys() == {'composition', 'flat_search'}, 'arm census mismatch')
        arms = []
        for name in ('composition', 'flat_search'):
            plan, branch = plans[name], physics[name]
            witness = plan['witness']
            require(witness['independent_full_replay_exact'] is True, 'independent saved replay required')
            actions = [a for e in witness['edges'] for a in e['actions']]
            require(len(actions) == branch['planned_commands'] == 10, 'fixed H2 plan required')
            trace = branch['trace']
            count = branch['completed_commands']
            require(type(count) is int and 0 <= count <= 10 and count == len(trace), 'trace count mismatch')
            require(branch['remaining_commands_not_executed'] == 10-count, 'withheld count mismatch')
            require([t['primitive_step'] for t in trace] == list(range(1, count+1)), 'trace step order mismatch')
            require([t['command'] for t in trace] == actions[:count], 'planned/executed raw action mismatch')
            if trace:
                require(trace[-1]['state'] == branch['final_state'], 'terminal state mismatch')
            predicted = metric(witness['edges'][-1]['output_context']['latent_history'][-1], root['compiled'])
            require(np.isclose(predicted, plan['selected_model_cost'], rtol=0, atol=1e-12),
                    'saved selected score mismatch')
            realized = metric(branch['final_vector'], root['compiled'])
            pose = float(branch['pose']['pose_cost'])
            official = float(branch['official']['official_state_distance'])
            require(np.isfinite([pose, official]).all(), 'nonfinite physical scores')
            arms.append(dict(arm=name, status=branch['status'], completed_commands=count,
                fixed_horizon_complete=branch['status'] == 'completed' and count == 10,
                predicted_q_mse=predicted, real_encoder_q_mse=realized,
                q_score_prediction_error=realized-predicted, object_pose_cost=pose,
                official_state_distance=official, official_success=branch['official']['official_success']))
        first, second = arms
        gaps = {key: first[key]-second[key] for key in
                ('predicted_q_mse', 'real_encoder_q_mse', 'object_pose_cost', 'official_state_distance')}
        signs = {key: direction(value) for key, value in gaps.items()}
        result.append(dict(root_sha256=root_id, arms=arms, composition_minus_flat=gaps,
            ordering=signs, fixed_horizon_pair=all(a['fixed_horizon_complete'] for a in arms),
            predicted_vs_real_q_order_same=signs['predicted_q_mse'] == signs['real_encoder_q_mse'],
            real_q_vs_object_pose_order_same=signs['real_encoder_q_mse'] == signs['object_pose_cost'],
            real_q_vs_official_order_same=signs['real_encoder_q_mse'] == signs['official_state_distance']))
    return dict(schema='p2_objective_decomposition_20261027.v1',
        status='saved_objective_decomposition_completed', rows=result,
        scope='known roots, selected branches only; unequal abort endpoints retained, no population inference',
        new_neural_calls=0, new_physical_steps=0, P1_admission=False, P2_admission=False,
        theory_negative=False, root_cause_proven=False)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(project, output):
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal
    project, output = Path(project).resolve(), Path(output).resolve()
    require(not output.exists(), 'exclusive new output required')
    files = [project/'results'/p for p in (
        'pusht_d2p_metric_program_preflight4_v1_20261026_cuda/report.json',
        'pusht_p2_matched_composition4_v1_20261027_cuda/p2_matched_report.json',
        'pusht_p2_matched_physical4_v1_20261027_cuda/report.json')]
    hashes = {str(p): digest(p) for p in files}
    compilation, planning, physical = [json.loads(p.read_text()) for p in files]
    require(hashes[str(files[0])] == 'd5f544689a9d2b5c6807b2454c1a98b9db6d88bb374badc457bdcc6defa87c8a',
            'frozen compilation hash mismatch')
    for key, expected in physical['protocol']['bindings'].items():
        require(digest(key) == expected, 'physical source input binding mismatch: '+key)
    for row in physical['rows']:
        for branch in row['branches']:
            require(digest(branch['terminal_rgb_path']) == branch['terminal_rgb_sha256'], 'terminal RGB hash mismatch')
    result = decompose(compilation, planning, physical)
    output.mkdir(exist_ok=False)
    seal = create_seal(files=[Path(__file__), project/'tests/test_p2_objective_decomposition_20261027.py'],
                       args_json=json.dumps(hashes, sort_keys=True), output_dir=output/'source_seal')
    require(hashes == {str(p): digest(p) for p in files}, 'inputs changed during audit')
    result.update(inputs=hashes, source_seal=seal, source_verification=verify_seal(
        output_dir=seal['output_dir'], expected_manifest_sha256=seal['manifest_sha256']))
    with (output/'report.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'rows': [dict(root=r['root_sha256'][:8],
        fixed_horizon_pair=r['fixed_horizon_pair'], ordering=r['ordering']) for r in result['rows']]}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.project, args.output)
