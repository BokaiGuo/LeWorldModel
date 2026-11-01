"""Execute already sealed search witnesses with the shared live RGB guard.

Known roots, variable horizon; not efficacy, equal-budget or P1 evidence.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def run(search_report, compilation, policy, output):
    import inspect
    import torch
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    from experiments.spectral_reachability.run_pusht_program_search_cuda import metric_for
    from experiments.spectral_reachability.pusht_visible_command_executor import execute
    from experiments.spectral_reachability.pusht_rgb_observability_guard import RgbObservabilityGuard
    from experiments.spectral_reachability.audit_pusht_sealed_task_contract import labels
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal
    compilation, policy, search_report, output = map(Path, (compilation, policy, search_report, output))
    compiled, bindings, config = checked.validate_inputs(compilation, policy)
    search = json.loads(search_report.read_text())
    if search['status'] != 'integration_completed' or not search['gates'] or not all(v is True for v in search['gates'].values()):
        raise ValueError('completed search integrity required')
    for seal in search['source_seals']:
        verify_seal(output_dir=seal['output_dir'], expected_manifest_sha256=seal['manifest_sha256'])
    if search['protocol']['bindings'] != bindings:
        raise ValueError('search input bindings differ')
    bindings[str(search_report.resolve())] = base.digest(search_report)
    byroot = {r['root_sha256']: r for r in search['rows']}
    if len(search['rows']) != 4 or len(byroot) != 4 or set(byroot) != {r['root_sha256'] for r in compiled['rows']}:
        raise ValueError('four unique matching roots required')
    # Validate complete witnesses before any physical outcomes are opened.
    actions = {}
    for row in compiled['rows']:
        plan = byroot[row['root_sha256']]['search']['best_plan']
        indices = plan['primitive_indices']
        if (len(indices) != plan['depth'] or plan['depth'] not in (1, 2)
                or any(type(i) is not int or not 0 <= i < 16 for i in indices)):
            raise ValueError('sealed bank-index witness required')
        bank = row['compiled']['receipts']
        expected = np.concatenate([np.repeat(np.asarray(bank[i]['program_live_commands'][0])[None], 5, axis=0) for i in indices])
        if not np.array_equal(expected, plan['actions']):
            raise ValueError('action witness differs from frozen bank')
        if abs(metric_for(row['compiled'])(plan['context']) - plan['cost']) > 1e-12:
            raise ValueError('search saved metric mismatch')
        actions[row['root_sha256']] = expected
    candidates = [json.loads(Path(p).read_text()) for p in bindings if Path(p).name == 'report.json']
    intervention = next(r for r in candidates if 'bank_path' in r.get('protocol', {}))
    bank = json.loads(Path(intervention['protocol']['bank_path']).read_text())
    upstream = base.physical._load_preflight(Path(bank['upstream_path']))
    output.mkdir(exist_ok=False)
    protocol = dict(bindings=bindings, feedback_every_primitive=True, agent_rgb=[65,105,225],
        planned_commands=sum(len(a) for a in actions.values()), candidate_selection_changed=False,
        variable_horizon=True, fair_comparison=False, P1_admission=False,
        scope='known-root sealed search execution feasibility, all outcomes retained')
    sources = checked.sources_before_allocate(compiled, config)
    sources += [Path(__file__).resolve(), Path(inspect.getfile(execute)), Path(inspect.getfile(RgbObservabilityGuard)),
                Path(inspect.getfile(labels)), Path(base.physical.__file__), Path(base.__file__)]
    sources = sorted(set(p.resolve() for p in sources))
    seals = [create_seal(files=sources[i:i+50], args_json=json.dumps(protocol), output_dir=output/f'source_seal_{i//50}') for i in range(0,len(sources),50)]
    base.write_new(output/'protocol.json', protocol)
    host = base.require_cuda_allocation('cuda')
    device = torch.device('cuda')
    model = base.load_compatible_pretrained(policy, device=device).eval().requires_grad_(False)
    before = base.p0._parameter_hash(model)
    buffers = {k:v.clone() for k,v in model.named_buffers()}
    env = base.gym.make('swm/PushT-v1', render_mode='rgb_array')
    rows = []
    transform = base.p0.image_transform()
    try:
        roots = base.collect_nonterminal_boundary_contact_roots(env, count=4, seed=int(upstream['protocol']['root_seed']), oversample_factor=4)
        if len(roots) != 4 or {base.root_fingerprint(r) for r in roots} != set(byroot):
            raise ValueError('root reconstruction mismatch')
        if tuple(env.unwrapped.variation_space['agent']['color'].value.tolist()) != (65,105,225):
            raise ValueError('unsupported renderer')
        reference = {r['root_sha256']:r for r in upstream['rows']}
        for index, root in enumerate(roots):
            key = base.root_fingerprint(root)
            frames, _, goalrgb = base.physical._reset_and_warmup(env, root, history_len=3)
            if (base.physical._sha256(goalrgb) != reference[key]['goal_rgb_sha256']
                    or base.physical._sha256(np.stack(frames)) != reference[key]['history_rgb_sha256']):
                raise ValueError('root history or goal RGB mismatch')
            trace = []
            def record(step, command, transition, rgb):
                trace.append(dict(primitive_step=step, command=command.tolist(),
                    rgb_sha256=base.physical._sha256(rgb), state=base.p0.packets.state_of(env).tolist()))
            result = execute(env, actions[key], root_id=key, agent_rgb=(65,105,225), on_observation=record)
            rgb = result.pop('terminal_rgb')
            state = base.p0.packets.state_of(env)
            with torch.no_grad():
                vector = model.encode({'pixels':base.p0._image_tensor(rgb,transform=transform,device=device)})['emb'][0,0]
            if vector.device.type != 'cuda' or not torch.isfinite(vector).all():
                raise ValueError('finite CUDA encoding required')
            row = dict(root_index=index, root_sha256=key, **result, trace=trace,
                search_depth=byroot[key]['search']['best_plan']['depth'], final_vector=vector.cpu().tolist(),
                final_state=state.tolist(), official=labels(state, root['goal']),
                pose=base.physical._pose_label(state,root['goal']))
            rows.append(row)
            base.write_new(output/f'root_{index}.json',row)
            print(f'root {index}: {row["status"]}, {row["completed_commands"]}/{len(actions[key])} commands',flush=True)
    finally:
        env.close()
    gates = dict(all_four_retained=len(rows)==4, no_execution_errors=all(r['status'] in ('completed','observation_unavailable','environment_terminated') for r in rows),
        commands_match_sealed_plan=all(np.array_equal(np.asarray(r['trace'][i]['command']), actions[r['root_sha256']][i]) for r in rows for i in range(len(r['trace']))),
        parameters_unchanged=before==base.p0._parameter_hash(model),
        buffers_unchanged=all(torch.equal(buffers[k],v) for k,v in model.named_buffers()),
        frozen_eval=all(not m.training for m in model.modules()) and all(not p.requires_grad for p in model.parameters()),
        inputs_unchanged=all(base.digest(Path(p))==d for p,d in bindings.items()))
    verified = [verify_seal(output_dir=s['output_dir'],expected_manifest_sha256=s['manifest_sha256']) for s in seals]
    base.write_new(output/'report.json',dict(schema='pusht_search_physical_cuda.v1',status='integration_completed' if all(gates.values()) else 'integrity_fail',
        protocol=protocol,host=host,rows=rows,gates=gates,source_seals=seals,source_verification=verified,P1_admission=False))
    if not all(gates.values()):
        raise RuntimeError('search physical integration failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('search_report','compilation','policy','output'):
        parser.add_argument('--'+name.replace('_','-'),dest=name,type=Path,required=True)
    run(**vars(parser.parse_args()))
