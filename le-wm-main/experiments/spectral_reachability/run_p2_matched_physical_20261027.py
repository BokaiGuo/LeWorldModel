"""Execute every sealed H2 paired selection; calibration, NOT P2 admission."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def selected_actions(report, protocol, compilation):
    """Non-neural witness/bank/goal validation before any physical outcome."""
    from experiments.spectral_reachability.d2p_composition_witness_v2 import digest
    from experiments.spectral_reachability.run_pusht_program_search_cuda import metric_for
    if (report.get('status') != 'calibration_completed' or not report.get('gates')
            or not all(v is True for v in report['gates'].values())
            or report.get('protocol_sha256') != digest(protocol)):
        raise ValueError('complete, protocol-bound P2 calibration required')
    source = {r['root_sha256']: r for r in compilation['rows']}
    rows = report['rows']
    if (len(rows) != 4 or len(source) != 4 or len({r['root_sha256'] for r in rows}) != 4
            or {r['root_sha256'] for r in rows} != set(source)):
        raise ValueError('all four unique source roots required')
    actions = {}
    for row in rows:
        key = row['root_sha256']
        declaration = row['declaration']
        if digest(declaration) != row['declaration_sha256']:
            raise ValueError('declaration changed')
        if row.get('fixture_only') is not False or row.get('prediction_budget_matched') is not True:
            raise ValueError('actual CUDA prediction-matched pair required')
        arms = row['rows']
        if len(arms) != 2 or {a['arm'] for a in arms} != {'composition', 'flat_search'}:
            raise ValueError('exactly two distinct arms required')
        compiled = source[key]['compiled']
        initial = compiled['input_context']
        expected_context = dict(layout='states_with_between_state_actions_v1', context_length=3,
            latent_dim=192, action_dim=10, latent_history=initial['emb'][0],
            action_history=initial['actions'][0][1:])
        if declaration['root']['context'] != expected_context:
            raise ValueError('selected plan predictor context differs from source root')
        bank = [np.repeat(np.asarray(r['program_live_commands'][0], dtype=float)[None], 5, axis=0)
                for r in compiled['receipts']]
        if not np.array_equal(declaration['bank'], bank):
            raise ValueError('candidate bank differs from frozen compilation')
        if declaration['root']['root_id'] != key:
            raise ValueError('declaration root mismatch')
        for arm in arms:
            path = arm['selected_path']
            if (len(path) != 2 or any(type(i) is not int or not 0 <= i < 16 for i in path)
                    or arm['replay_error'] is not None or arm['witness'] is None):
                raise ValueError('complete two-token replayed selection required')
            expected = np.concatenate([bank[i] for i in path])
            if arm['witness'].get('independent_full_replay_exact') is not True:
                raise ValueError('independent full replay required')
            receipt = arm['witness']['receipt']
            body = {k: v for k, v in receipt.items() if k != 'receipt_sha256'}
            if digest(body) != receipt['receipt_sha256']:
                raise ValueError('witness receipt digest changed')
            if receipt['root'] != declaration['root'] or not np.array_equal(receipt['actions'], expected):
                raise ValueError('root or raw action witness mismatch')
            if (expected.shape != (10, 2) or not np.isfinite(expected).all()
                    or np.any(np.abs(expected) > 1)):
                raise ValueError('bounded ten-step raw plan required')
            cost = arm['selected_model_cost']
            if (type(cost) not in (int, float) or not np.isfinite(cost)
                    or abs(metric_for(compiled)(receipt['terminal_context']) - cost) > 1e-12):
                raise ValueError('saved selected metric differs')
            actions[key, arm['arm']] = expected
    return actions


def run(paired_report, compilation, policy, output):
    import inspect
    import torch
    from PIL import Image
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    from experiments.spectral_reachability import d2p_composition_witness_v2 as witness
    from experiments.spectral_reachability.run_pusht_program_search_cuda import metric_for
    from experiments.spectral_reachability.pusht_visible_command_executor import execute
    from experiments.spectral_reachability.pusht_rgb_observability_guard import RgbObservabilityGuard
    from experiments.spectral_reachability.audit_pusht_sealed_task_contract import labels
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal
    paired_report, compilation, policy, output = map(Path, (paired_report, compilation, policy, output))
    source, bindings, config = checked.validate_inputs(compilation, policy)
    report = json.loads(paired_report.read_text())
    protocol_path = paired_report.parent / 'p2_matched_protocol.json'
    snapshots_path = paired_report.parent / 'p2_matched_sources.json'
    protocol = json.loads(protocol_path.read_text())
    snapshots = json.loads(snapshots_path.read_text())
    if protocol['bindings'] != bindings or witness.digest(snapshots) != protocol['source_manifest_sha256']:
        raise ValueError('P2 source/input manifest differs')
    for s in snapshots:
        if (base.digest(Path(s['path'])) != s['sha256']
                or hashlib.sha256(s['source'].encode()).hexdigest() != s['sha256']):
            raise ValueError('P2 current or archived source differs')
    actions = selected_actions(report, protocol, source)
    for p in (paired_report, protocol_path, snapshots_path):
        bindings[str(p.resolve())] = base.digest(p)
    candidates = [json.loads(Path(p).read_text()) for p in bindings if Path(p).name == 'report.json']
    intervention = next(r for r in candidates if 'bank_path' in r.get('protocol', {}))
    bank = json.loads(Path(intervention['protocol']['bank_path']).read_text())
    upstream = base.physical._load_preflight(Path(bank['upstream_path']))
    references = {r['root_sha256']: r for r in upstream['rows']}
    for row in report['rows']:
        root_binding = row['declaration']['root']
        reference = references[row['root_sha256']]
        if (root_binding['root_rgb_sha256'] != reference['history_rgb_sha256']
                or root_binding['goal_rgb_sha256'] != reference['goal_rgb_sha256']):
            raise ValueError('selected plan source RGB or goal binding differs')
    out_protocol = dict(bindings=bindings, selections={f'{k}:{a}': v.tolist() for (k,a),v in actions.items()},
        roots=[r['root_sha256'] for r in source['rows']], arms=['composition', 'flat_search'],
        planned_horizon_tokens=2, planned_primitive_steps=10, root_selection_changed=False,
        live_RGB_guard=True, retain_all_aborts=True, physical_outcomes_used_for_selection=False,
        scope='known calibration cohort; selected-plan execution only; not efficacy or full D2P',
        P1_admission=False, P2_admission=False, fair_total_budget_comparison=False)
    output.mkdir(exist_ok=False)
    files = checked.sources_before_allocate(source, config)
    files += [Path(__file__), Path(inspect.getfile(execute)), Path(inspect.getfile(RgbObservabilityGuard)),
              Path(inspect.getfile(labels)), Path(inspect.getfile(metric_for)), Path(witness.__file__),
              Path(base.physical.__file__), Path(base.__file__)]
    files = sorted(set(p.resolve() for p in files))
    seals = [create_seal(files=files[i:i+50], args_json=json.dumps(out_protocol),
                        output_dir=output/f'source_seal_{i//50}') for i in range(0,len(files),50)]
    base.write_new(output/'protocol.json', out_protocol)
    host = base.require_cuda_allocation('cuda')
    device = torch.device('cuda:0')
    model = base.load_compatible_pretrained(policy, device=device).eval().requires_grad_(False)
    before = checked.tensor_state_hash(model)
    transform = base.p0.image_transform()
    env = base.gym.make('swm/PushT-v1', render_mode='rgb_array')
    rows = []
    try:
        roots = base.collect_nonterminal_boundary_contact_roots(env, count=4,
            seed=int(upstream['protocol']['root_seed']), oversample_factor=4)
        if len(roots) != 4 or {base.root_fingerprint(r) for r in roots} != set(out_protocol['roots']):
            raise ValueError('physical roots differ from sealed cohort')
        if tuple(env.unwrapped.variation_space['agent']['color'].value.tolist()) != (65,105,225):
            raise ValueError('unsupported agent renderer')
        for index, root in enumerate(roots):
            key = base.root_fingerprint(root)
            # Fixed alternate order, never selected using control outcomes.
            order = out_protocol['arms'] if index % 2 == 0 else list(reversed(out_protocol['arms']))
            branches = []
            for arm in order:
                frames, _, goalrgb = base.physical._reset_and_warmup(env, root, history_len=3)
                if (base.physical._sha256(goalrgb) != references[key]['goal_rgb_sha256']
                        or base.physical._sha256(np.stack(frames)) != references[key]['history_rgb_sha256']):
                    raise ValueError('physical source RGB/goal differs')
                trace = []
                def record(step, command, transition, rgb):
                    trace.append(dict(primitive_step=step, command=command.tolist(),
                        rgb_sha256=base.physical._sha256(rgb),
                        state=base.p0.packets.state_of(env).tolist()))
                result = execute(env, actions[key, arm], root_id=key, agent_rgb=(65,105,225), on_observation=record)
                rgb = result.pop('terminal_rgb')
                state = base.p0.packets.state_of(env)
                image_path = output / f'root_{index}_{arm}_terminal.png'
                Image.fromarray(rgb).save(image_path)
                with torch.no_grad():
                    vector = model.encode({'pixels':base.p0._image_tensor(rgb, transform=transform, device=device)})['emb'][0,0]
                if vector.device.type != 'cuda' or not torch.isfinite(vector).all():
                    raise ValueError('finite actual CUDA encoding required')
                branch = dict(arm=arm, **result, trace=trace, final_vector=vector.cpu().tolist(),
                    final_state=state.tolist(), terminal_rgb_path=str(image_path.resolve()),
                    terminal_rgb_sha256=base.digest(image_path), official=labels(state,root['goal']),
                    pose=base.physical._pose_label(state,root['goal']))
                branches.append(branch)
                base.write_new(output/f'root_{index}_{arm}.json', branch)
                print(f'P2 root {index} {arm}: {branch["status"]}, {branch["completed_commands"]}/10',flush=True)
            rows.append(dict(root_index=index, root_sha256=key, branches=branches))
    finally:
        env.close()
    branches = [b for r in rows for b in r['branches']]
    gates = dict(all_eight_retained=len(rows)==4 and len(branches)==8,
        terminal_encodings_cuda=len(branches)==8,
        no_executor_errors=all(b['status'] in ('completed','observation_unavailable','environment_terminated') for b in branches),
        all_actions_match_sealed_selection=all(np.array_equal(b['trace'][i]['command'], actions[r['root_sha256'],b['arm']][i])
            for r in rows for b in r['branches'] for i in range(len(b['trace']))),
        model_unchanged=before==checked.tensor_state_hash(model),
        frozen_eval=all(not m.training for m in model.modules()) and all(not p.requires_grad for p in model.parameters()),
        inputs_unchanged=all(base.digest(Path(p))==d for p,d in bindings.items()))
    verified = [verify_seal(output_dir=s['output_dir'], expected_manifest_sha256=s['manifest_sha256']) for s in seals]
    base.write_new(output/'report.json', dict(schema='p2_matched_physical_20261027.v1',
        status='calibration_execution_completed' if all(gates.values()) else 'integrity_fail',
        protocol=out_protocol, host=host, rows=rows, gates=gates, source_seals=seals,
        source_verification=verified, actual_terminal_encode_samples=len(branches),
        P1_admission=False, P2_admission=False, efficacy_verified=False))
    if not all(gates.values()):
        raise RuntimeError('paired physical integrity failed; retain all artifacts')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('paired_report','compilation','policy','output'):
        p.add_argument('--'+name.replace('_','-'), dest=name, type=Path, required=True)
    run(**vars(p.parse_args()))
