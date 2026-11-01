"""Observed-prefix one-shot DLS intervention, fixed Q/lambda/prefixes.

Known calibration control only: not an independent P1 or fair-budget victory.
No GN iterations, candidate filtering, selector change or outcome tuning.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def one_shot(goal, nominal, jacobian, projection, damping=.01):
    arrays = [np.asarray(v) for v in (goal, nominal, jacobian, projection)]
    if any(v.dtype.kind not in 'fi' or not np.isfinite(v).all() for v in arrays):
        raise ValueError('finite real goal, nominal, Jacobian and projection required')
    goal, nominal, jacobian, projection = [v.astype(float) for v in arrays]
    if (goal.ndim != 1 or nominal.shape != goal.shape or jacobian.shape != (goal.size, 2)
            or projection.shape != (goal.size, goal.size) or type(damping) not in (float, int)
            or not np.isfinite(damping) or damping <= 0):
        raise ValueError('one-token two-coordinate DLS contract required')
    response = projection@jacobian
    residual = projection@(goal-nominal)
    system = np.vstack((response, np.sqrt(damping)*np.eye(2)))
    delta = np.linalg.lstsq(system, np.r_[residual, np.zeros(2)], rcond=None)[0]
    return delta.reshape(1, 2)


def run(execution, policy, output):
    import torch
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    from experiments.spectral_reachability import preflight_pusht_d2p_p1_reconfiguration as p1
    from experiments.spectral_reachability.run_pusht_observed_prefix_prediction import observed_context
    from experiments.spectral_reachability.run_pusht_d2p_sealed_program_execution import checked_commands
    from experiments.spectral_reachability.audit_pusht_sealed_task_contract import labels
    from experiments.spectral_reachability.audit_pusht_d2p_metric_program_arrays import audit
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal
    from experiments.spectral_reachability.d2p_model_call_instrumentation import PredictInstrumentation
    execution, policy, output = map(Path, (execution, policy, output))
    physical = json.loads(execution.read_text())
    if physical['status'] != 'completed' or not physical['gates'] or not all(v is True for v in physical['gates'].values()):
        raise ValueError('complete physical calibration required')
    compilation_path = Path(physical['protocol']['preflight'])
    if base.digest(compilation_path) != physical['protocol']['preflight_sha256']:
        raise ValueError('compilation digest changed')
    compilation = json.loads(compilation_path.read_text())
    if compilation['status'] != 'compiled_calibration' or not all(v is True for v in compilation['gates'].values()):
        raise ValueError('complete sealed compilation required')
    seal = compilation['source_seal']
    verify_seal(output_dir=seal['output_dir'], expected_manifest_sha256=seal['manifest_sha256'])
    bindings = dict(compilation['protocol']['bindings'])
    bindings[str(execution.resolve())] = base.digest(execution)
    bindings[str(compilation_path.resolve())] = base.digest(compilation_path)
    for name in ('weights.pt', 'config.json'):
        if str((policy/name).resolve()) not in bindings:
            raise ValueError('policy not bound')
    if not all(base.digest(Path(p)) == d for p, d in bindings.items()):
        raise ValueError('inputs changed')
    for row in compilation['rows']:
        audit(row['compiled'])
    candidates = [json.loads(Path(p).read_text()) for p in bindings if Path(p).name == 'report.json']
    intervention = next(r for r in candidates if 'bank_path' in r.get('protocol', {}))
    bank = json.loads(Path(intervention['protocol']['bank_path']).read_text())
    upstream = base.physical._load_preflight(Path(bank['upstream_path']))
    output.mkdir(exist_ok=False)
    stages = output/'stages'
    stages.mkdir()
    protocol = {'bindings': bindings, 'damping': .01, 'suffix_horizon': 1, 'expected_predict_samples': 192,
                'prefixes_unchanged': True, 'selector_unchanged': True, 'gn_iterations': 0,
                'scope': 'known-root feedback-only one-shot suffix intervention; not matched-budget efficacy', 'P1_admission': False}
    paths = [Path(__file__), Path(base.__file__), Path(base.p0.__file__), Path(p1.__file__), Path(base.physical.__file__), Path(base.p0.packets.__file__),
             Path(observed_context.__code__.co_filename), Path(checked_commands.__code__.co_filename), Path(labels.__code__.co_filename), Path(audit.__code__.co_filename),
             Path(base.load_compatible_pretrained.__code__.co_filename), Path(base.collect_nonterminal_boundary_contact_roots.__code__.co_filename),
             Path(PredictInstrumentation.__module__.replace('.', '/')).with_suffix('.py')]
    seal = create_seal(files=list(dict.fromkeys(p.resolve() for p in paths)), args_json=json.dumps(protocol), output_dir=output/'source_seal')
    base.write_new(output/'protocol.json', protocol)
    host = base.require_cuda_allocation('cuda')
    device = torch.device('cuda')
    model = base.load_compatible_pretrained(policy, device=device).eval().requires_grad_(False)
    before = base.p0._parameter_hash(model)
    buffers = {k: v.clone() for k, v in model.named_buffers()}
    env = base.gym.make('swm/PushT-v1', render_mode='rgb_array')
    transform = base.p0.image_transform()
    cm = {r['root_sha256']: r for r in compilation['rows']}
    pm = {r['root_sha256']: r for r in physical['rows']}
    rows = []
    try:
        roots = base.collect_nonterminal_boundary_contact_roots(env, count=4, seed=int(upstream['protocol']['root_seed']), oversample_factor=4)
        if len(cm) != 4 or set(cm) != set(pm) or set(cm) != {base.root_fingerprint(r) for r in roots}:
            raise ValueError('root identities differ')
        with PredictInstrumentation(model) as counter:
            with counter.phase('planning', method='observed_one_shot_suffix', horizon_tokens=1):
                for index, root in enumerate(roots):
                    key = base.root_fingerprint(root)
                    compiled = cm[key]['compiled']
                    prior = {b['candidate_id']: b for b in pm[key]['branches']}
                    projection = np.asarray(compiled['scoring']['metric']['projection'])
                    goal = np.asarray(compiled['goal_vectors'][0])
                    values = []
                    for receipt in compiled['receipts']:
                        name = receipt['candidate_id']
                        saved = prior[name]
                        frames, history, rgbgoal = base.physical._reset_and_warmup(env, root, history_len=3)
                        if base.physical._sha256(rgbgoal) != upstream['rows'][index]['goal_rgb_sha256']:
                            raise ValueError('goal RGB changed')
                        command = checked_commands(receipt)[0]
                        base.p0.packets.execute_packet(env, base.p0.packets.repeat_command(command))
                        rgb = np.asarray(env.render(), np.uint8).copy()
                        if base.physical._sha256(rgb) != saved['stages'][0]['rgb_sha256'] or not np.array_equal(base.p0.packets.state_of(env), saved['stages'][0]['state']):
                            raise ValueError('prefix replay changed')
                        frames, history = observed_context(frames, history, rgb, command)
                        emb, actions = p1._encode_context(model, frames, history, transform, device)
                        def terminal(flat):
                            return p1._suffix_terminal(model, emb, actions, flat.reshape(1, 1, 2))[0]
                        zero = torch.zeros(2, device=device, dtype=torch.float32)
                        with torch.no_grad():
                            nominal = terminal(zero)
                        jac = torch.autograd.functional.jacobian(terminal, zero.requires_grad_(True), vectorize=True)
                        params = one_shot(goal, nominal.detach().cpu().numpy(), jac.detach().cpu().numpy(), projection)
                        actual_params = torch.tensor(params, dtype=torch.float32, device=device)
                        live = torch.tanh(actual_params).detach().cpu().numpy()
                        with torch.no_grad():
                            forecast = terminal(actual_params.reshape(-1)).cpu().numpy().astype(float)
                        plan = {'candidate_id': name, 'root_sha256': key, 'projection': projection.tolist(), 'goal': goal.tolist(),
                                'nominal': nominal.detach().cpu().tolist(), 'jacobian': jac.detach().cpu().tolist(),
                                'params_float64': params.tolist(), 'executed_params': actual_params.cpu().tolist(),
                                'live_command': live.tolist(), 'predicted_terminal_vector': forecast.tolist(), 'future_observations_used': False}
                        plan_path = stages/f'root_{index}.{name}.json'
                        base.write_new(plan_path, plan)
                        base.p0.packets.execute_packet(env, base.p0.packets.repeat_command(live[0]))
                        final_rgb = np.asarray(env.render(), np.uint8).copy()
                        state = base.p0.packets.state_of(env)
                        with torch.no_grad():
                            vector = model.encode({'pixels': base.p0._image_tensor(final_rgb, transform=transform, device=device)})['emb'][0, 0].cpu().numpy().astype(float)
                        values.append({'candidate_id': name, 'plan_path': str(plan_path.resolve()), 'plan_sha256': base.digest(plan_path), 'prefix_replay_exact': True,
                                       'final_state': state.tolist(), 'final_vector': vector.tolist(), 'final_rgb_sha256': base.physical._sha256(final_rgb),
                                       'pose': base.physical._pose_label(state, root['goal']), 'official': labels(state, root['goal']),
                                       'actual_metric_cost': float(np.mean((projection@(vector-goal))**2))})
                    row = {'root_index': index, 'root_sha256': key, 'selected_candidate_id': compiled['selected_candidate_id'], 'branches': values}
                    base.write_new(output/f'root_{index}.json', row)
                    rows.append(row)
                    print(f'root {index}: 16 observed one-shot suffixes physically executed', flush=True)
            records = counter.records
    finally:
        env.close()
    gates = {'64_physical_branches': len(rows) == 4 and sum(len(r['branches']) for r in rows) == 64,
             '192_predict_samples': sum(r['predict_samples'] for r in records) == 192 and all(r['completed'] for r in records),
             'parameters_unchanged': before == base.p0._parameter_hash(model), 'buffers_unchanged': all(torch.equal(buffers[k], v) for k, v in model.named_buffers()),
             'eval': all(not m.training for m in model.modules()), 'inputs_unchanged': all(base.digest(Path(p)) == d for p, d in bindings.items())}
    verified = verify_seal(output_dir=seal['output_dir'], expected_manifest_sha256=seal['manifest_sha256'])
    base.write_new(output/'report.json', {'schema': 'pusht_observed_oneshot_suffix_control.v1', 'status': 'completed' if all(gates.values()) else 'integrity_fail',
        'protocol': protocol, 'host': host, 'gates': gates, 'rows': rows, 'predict': records, 'source_seal': seal, 'source_verification': verified, 'P1_admission': False})
    if not all(gates.values()):
        raise RuntimeError('intervention integrity failed')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('execution', 'policy', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    run(**vars(p.parse_args()))
