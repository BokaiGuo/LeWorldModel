"""Execute every sealed fixed-bank program; calibration only, never P1."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def checked_commands(receipt):
    params = np.asarray(receipt['executed_program_params'])
    commands = np.asarray(receipt['program_live_commands'])
    if (params.dtype.kind not in 'fi' or commands.dtype.kind not in 'fi'
            or params.shape != (2, 2) or commands.shape != (2, 2)
            or not np.isfinite(params).all() or not np.isfinite(commands).all()
            or np.any(np.abs(commands) > 1)
            or not np.allclose(commands, np.tanh(params), atol=2e-7, rtol=2e-7)):
        raise ValueError('sealed finite two-token tanh commands required')
    return commands.astype(np.float32)


def run(*, preflight, policy, output):
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    from experiments.spectral_reachability.audit_pusht_d2p_metric_program_arrays import audit
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal
    import torch
    preflight, policy, output = map(Path, (preflight, policy, output))
    source = json.loads(preflight.read_text())
    if source['status'] != 'compiled_calibration' or not source['gates'] or not all(v is True for v in source['gates'].values()):
        raise ValueError('successful sealed compilation required')
    oldseal = source['source_seal']
    verify_seal(output_dir=oldseal['output_dir'], expected_manifest_sha256=oldseal['manifest_sha256'])
    bindings = source['protocol']['bindings']
    if not all(base.digest(Path(p)) == digest for p, digest in bindings.items()):
        raise ValueError('compiler inputs changed')
    if str((policy/'weights.pt').resolve()) not in bindings or str((policy/'config.json').resolve()) not in bindings:
        raise ValueError('policy not bound to compilation')
    if len(source['rows']) != 4:
        raise ValueError('all four calibration roots required')
    for row in source['rows']:
        audit(row['compiled'])
        if len(row['compiled']['receipts']) != 16:
            raise ValueError('all sixteen programs required')
        for receipt in row['compiled']['receipts']:
            checked_commands(receipt)
    # Read the exact upstream root seed, never choose roots by physical outcomes.
    candidates = [json.loads(Path(p).read_text()) for p in bindings if Path(p).name == 'report.json']
    interventions = [r for r in candidates if 'bank_path' in r.get('protocol', {})]
    if len(interventions) != 1:
        raise ValueError('unique intervention binding required')
    bank = json.loads(Path(interventions[0]['protocol']['bank_path']).read_text())
    upstream = base.physical._load_preflight(Path(bank['upstream_path']))
    protocol = {'preflight': str(preflight.resolve()), 'preflight_sha256': base.digest(preflight),
                'programs': 64, 'selection_changed': False, 'reoptimization': False,
                'scope': 'known calibration roots; open-loop fixed programs; not P1', 'P1_admission': False}
    output.mkdir(exist_ok=False)
    sources = [Path(__file__), Path(base.__file__), Path(base.physical.__file__), Path(base.p0.__file__),
               Path(base.p0.packets.__file__), Path(base.load_compatible_pretrained.__code__.co_filename),
               Path(base.collect_nonterminal_boundary_contact_roots.__code__.co_filename)]
    seal = create_seal(files=list(dict.fromkeys(p.resolve() for p in sources)), args_json=json.dumps(protocol), output_dir=output/'prerun_source_seal')
    base.write_new(output/'protocol.json', protocol)
    host = base.require_cuda_allocation('cuda')
    model = base.load_compatible_pretrained(policy, device='cuda').eval().requires_grad_(False)
    before = base.p0._parameter_hash(model)
    buffers = {k: v.clone() for k, v in model.named_buffers()}
    env = base.gym.make('swm/PushT-v1', render_mode='rgb_array')
    rows = []
    transform = base.p0.image_transform()
    try:
        roots = base.collect_nonterminal_boundary_contact_roots(env, count=4, seed=int(upstream['protocol']['root_seed']), oversample_factor=4)
        for index, root in enumerate(roots):
            saved = source['rows'][index]
            if saved['root_index'] != index or base.root_fingerprint(root) != saved['root_sha256']:
                raise ValueError('root identity mismatch')
            compiled = saved['compiled']
            projection = np.asarray(compiled['scoring']['metric']['projection'])
            goal = np.asarray(compiled['goal_vectors'][0])
            branches = []
            for receipt in compiled['receipts']:
                branch = {'candidate_id': receipt['candidate_id'], 'commands': checked_commands(receipt).tolist()}
                try:
                    _, _, goal_rgb = base.physical._reset_and_warmup(env, root, history_len=3)
                    if base.physical._sha256(goal_rgb) != upstream['rows'][index]['goal_rgb_sha256']:
                        raise ValueError('goal render mismatch')
                    stages = []
                    for command in checked_commands(receipt):
                        base.p0.packets.execute_packet(env, base.p0.packets.repeat_command(command))
                        rgb = np.asarray(env.render(), dtype=np.uint8).copy()
                        state = base.p0.packets.state_of(env)
                        with torch.no_grad():
                            vector = model.encode({'pixels': base.p0._image_tensor(rgb, transform=transform, device=torch.device('cuda'))})['emb'][0, 0]
                        value = vector.detach().cpu().numpy().astype(float)
                        if value.shape != goal.shape or not np.isfinite(value).all():
                            raise ValueError('invalid CUDA encoding')
                        stages.append({'rgb_sha256': base.physical._sha256(rgb), 'state': state.tolist(),
                                       'pose': base.physical._pose_label(state, root['goal']),
                                       'metric_cost': float(np.mean((projection@(value-goal))**2)), 'vector': value.tolist()})
                    branch.update(status='executed', stages=stages,
                                  predicted_terminal_cost=receipt['nonlinear_terminal_cost_diagnostic_only'])
                except Exception as exc:
                    branch.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                branches.append(branch)
            row = {'root_index': index, 'root_sha256': saved['root_sha256'],
                   'selected_candidate_id': compiled['selected_candidate_id'], 'branches': branches}
            base.write_new(output/f'root_{index}.json', row)
            rows.append(row)
            print(f'root {index}: {sum(b["status"] == "executed" for b in branches)}/16 executed', flush=True)
    finally:
        env.close()
    gates = {'all_64_executed': len(rows) == 4 and sum(b['status'] == 'executed' for r in rows for b in r['branches']) == 64,
             'parameters_unchanged': before == base.p0._parameter_hash(model),
             'buffers_unchanged': all(torch.equal(buffers[k], v) for k, v in model.named_buffers()),
             'eval': all(not m.training for m in model.modules()),
             'inputs_unchanged': base.digest(preflight) == protocol['preflight_sha256'] and all(base.digest(Path(p)) == d for p, d in bindings.items())}
    verified = verify_seal(output_dir=seal['output_dir'], expected_manifest_sha256=seal['manifest_sha256'])
    base.write_new(output/'report.json', {'schema': 'pusht_sealed_execution.v1', 'status': 'completed' if all(gates.values()) else 'integrity_fail',
        'protocol': protocol, 'host': host, 'rows': rows, 'gates': gates, 'source_seal': seal, 'source_verification': verified, 'P1_admission': False})
    if not all(gates.values()):
        raise RuntimeError('execution integrity failed; all failed branches preserved')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('preflight', 'policy', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    run(**vars(parser.parse_args()))
