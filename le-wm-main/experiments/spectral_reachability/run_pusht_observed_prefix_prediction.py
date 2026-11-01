"""Fixed-suffix prediction after exact physical prefix replay; not a policy."""
import argparse
import json
from pathlib import Path
import numpy as np


def observed_context(frames, actions, rgb, command):
    """F1,F2,F3 + observed F4; incoming A0,A1,A2 + executed A3."""
    actions, command = np.asarray(actions), np.asarray(command)
    if (len(frames) != 3 or actions.shape != (3, 2) or command.shape != (2,)
            or actions.dtype.kind not in 'fi' or command.dtype.kind not in 'fi'
            or not np.isfinite(actions).all() or not np.isfinite(command).all()
            or np.any(np.abs(actions) > 1) or np.any(np.abs(command) > 1)):
        raise ValueError('three aligned incoming commands and one legal executed command required')
    images = [np.asarray(image) for image in [*frames, rgb]]
    if any(image.shape != (224, 224, 3) or image.dtype != np.uint8 for image in images):
        raise ValueError('official RGB boundaries required')
    return [image.copy() for image in images[1:]], np.concatenate((actions[1:], command[None]), axis=0)


def run(execution, policy, output):
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    from experiments.spectral_reachability import preflight_pusht_d2p_p1_reconfiguration as p1
    from experiments.spectral_reachability.run_pusht_d2p_sealed_program_execution import checked_commands
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal
    from experiments.spectral_reachability.d2p_model_call_instrumentation import PredictInstrumentation
    import torch
    execution, policy, output = map(Path, (execution, policy, output))
    physical = json.loads(execution.read_text())
    if physical['status'] != 'completed' or not all(v is True for v in physical['gates'].values()):
        raise ValueError('complete physical evidence required')
    path = Path(physical['protocol']['preflight'])
    if base.digest(path) != physical['protocol']['preflight_sha256']:
        raise ValueError('compilation changed')
    compilation = json.loads(path.read_text())
    seal = compilation['source_seal']
    verify_seal(output_dir=seal['output_dir'], expected_manifest_sha256=seal['manifest_sha256'])
    bindings = compilation['protocol']['bindings'].copy()
    bindings[str(execution.resolve())] = base.digest(execution)
    bindings[str(path.resolve())] = base.digest(path)
    for name in ('weights.pt', 'config.json'):
        if str((policy/name).resolve()) not in bindings:
            raise ValueError('policy not bound')
    if not all(base.digest(Path(p)) == d for p, d in bindings.items()):
        raise ValueError('upstream changed')
    candidates = [json.loads(Path(p).read_text()) for p in bindings if Path(p).name == 'report.json']
    intervention = next(r for r in candidates if 'bank_path' in r.get('protocol', {}))
    bank = json.loads(Path(intervention['protocol']['bank_path']).read_text())
    upstream = base.physical._load_preflight(Path(bank['upstream_path']))
    output.mkdir(exist_ok=False)
    protocol = {'bindings': bindings, 'expected_predict_samples': 64,
                'physical_prefix_replays': 64, 'suffixes_changed': False,
                'selection_changed': False, 'P1_admission': False,
                'scope': 'known-root observed-prefix vs imagined-prefix fixed-suffix prediction diagnostic'}
    sources = [Path(__file__), Path(base.__file__), Path(base.p0.__file__), Path(p1.__file__),
               Path(base.physical.__file__), Path(base.p0.packets.__file__),
               Path(checked_commands.__code__.co_filename), Path(PredictInstrumentation.__module__.replace('.', '/')).with_suffix('.py'),
               Path(base.load_compatible_pretrained.__code__.co_filename), Path(base.collect_nonterminal_boundary_contact_roots.__code__.co_filename)]
    seal = create_seal(files=list(dict.fromkeys(p.resolve() for p in sources)), args_json=json.dumps(protocol), output_dir=output/'source_seal')
    base.write_new(output/'protocol.json', protocol)
    host = base.require_cuda_allocation('cuda')
    model = base.load_compatible_pretrained(policy, device='cuda').eval().requires_grad_(False)
    before = base.p0._parameter_hash(model)
    buffers = {k: v.clone() for k, v in model.named_buffers()}
    env = base.gym.make('swm/PushT-v1', render_mode='rgb_array')
    rows = []
    compiled = {r['root_sha256']: r for r in compilation['rows']}
    measured = {r['root_sha256']: r for r in physical['rows']}
    transform = base.p0.image_transform()
    try:
        roots = base.collect_nonterminal_boundary_contact_roots(env, count=4, seed=int(upstream['protocol']['root_seed']), oversample_factor=4)
        if len(compiled) != 4 or set(compiled) != set(measured) or set(compiled) != {base.root_fingerprint(root) for root in roots}:
            raise ValueError('complete identical root sets required')
        with PredictInstrumentation(model) as counter:
            with counter.phase('planning', method='observed_prefix_fixed_suffix', horizon_tokens=1):
                for index, root in enumerate(roots):
                    key = base.root_fingerprint(root)
                    branches = {r['candidate_id']: r for r in measured[key]['branches']}
                    values = []
                    for receipt in compiled[key]['compiled']['receipts']:
                        saved = branches[receipt['candidate_id']]
                        frames, actions, _ = base.physical._reset_and_warmup(env, root, history_len=3)
                        command = checked_commands(receipt)[0]
                        base.p0.packets.execute_packet(env, base.p0.packets.repeat_command(command))
                        rgb = np.asarray(env.render(), np.uint8).copy()
                        if base.physical._sha256(rgb) != saved['stages'][0]['rgb_sha256'] or not np.array_equal(base.p0.packets.state_of(env), saved['stages'][0]['state']):
                            raise ValueError('physical prefix replay changed')
                        frames, actions = observed_context(frames, actions, rgb, command)
                        emb, act = p1._encode_context(model, frames, actions, transform, torch.device('cuda'))
                        suffix = torch.tensor(receipt['executed_suffix_params'], dtype=torch.float32, device='cuda')[None]
                        with torch.no_grad():
                            predicted = p1._suffix_terminal(model, emb, act, suffix)[0].cpu().numpy().astype(float)
                        if not np.isfinite(predicted).all():
                            raise ValueError('nonfinite forecast')
                        values.append({'candidate_id': receipt['candidate_id'], 'prefix_replay_exact': True,
                                       'observed_prefix_terminal_prediction': predicted.tolist()})
                    row = {'root_index': index, 'root_sha256': key, 'branches': values}
                    base.write_new(output/f'root_{index}.json', row)
                    rows.append(row)
                    print(f'root {index}: 16 exact prefix replays and CUDA predictions', flush=True)
            records = counter.records
    finally:
        env.close()
    gates = {'four_roots': len(rows) == 4, 'predict_samples_64': sum(r['predict_samples'] for r in records) == 64 and all(r['completed'] for r in records),
             'parameters_unchanged': before == base.p0._parameter_hash(model),
             'buffers_unchanged': all(torch.equal(buffers[k], v) for k, v in model.named_buffers()),
             'eval': all(not m.training for m in model.modules()),
             'inputs_unchanged': all(base.digest(Path(p)) == d for p, d in bindings.items())}
    verified = verify_seal(output_dir=seal['output_dir'], expected_manifest_sha256=seal['manifest_sha256'])
    base.write_new(output/'report.json', {'schema': 'pusht_observed_prefix_prediction.v1', 'status': 'completed' if all(gates.values()) else 'integrity_fail',
        'protocol': protocol, 'host': host, 'gates': gates, 'rows': rows, 'predict': records, 'source_seal': seal, 'source_verification': verified, 'P1_admission': False})
    if not all(gates.values()):
        raise RuntimeError('diagnostic integrity failed')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('execution', 'policy', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    run(**vars(p.parse_args()))
