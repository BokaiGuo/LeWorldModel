"""Live guard replay of all sealed programs; not P1 efficacy or recovery."""
import argparse
import json
from pathlib import Path
import numpy as np


def run(execution, policy, output):
    import inspect
    import torch
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    from experiments.spectral_reachability.run_pusht_d2p_budget_guard_cuda import validate_inputs
    from experiments.spectral_reachability.run_pusht_d2p_sealed_program_execution import checked_commands
    from experiments.spectral_reachability.pusht_visible_command_executor import execute
    from experiments.spectral_reachability.pusht_rgb_observability_guard import RgbObservabilityGuard
    from experiments.spectral_reachability.audit_pusht_sealed_task_contract import labels
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal
    execution, policy, output = map(Path, (execution, policy, output))
    original = json.loads(execution.read_text())
    if original['status'] != 'completed' or not original['gates'] or not all(v is True for v in original['gates'].values()):
        raise ValueError('complete sealed physical evidence required')
    oldseal = original['source_seal']
    verify_seal(output_dir=oldseal['output_dir'], expected_manifest_sha256=oldseal['manifest_sha256'])
    cp = Path(original['protocol']['preflight'])
    if base.digest(cp) != original['protocol']['preflight_sha256']:
        raise ValueError('original compilation changed')
    compilation, bindings, _ = validate_inputs(cp, policy)
    bindings[str(execution.resolve())] = base.digest(execution)
    candidates = [json.loads(Path(p).read_text()) for p in bindings if Path(p).name == 'report.json']
    intervention = next(r for r in candidates if 'bank_path' in r.get('protocol', {}))
    bank = json.loads(Path(intervention['protocol']['bank_path']).read_text())
    upstream = base.physical._load_preflight(Path(bank['upstream_path']))
    output.mkdir(exist_ok=False)
    protocol = {'bindings': bindings, 'programs': 64, 'agent_rgb': [65,105,225], 'feedback_every_primitive': True,
                'actions_changed': False, 'selector_changed': False, 'expected_final_encoder_calls': 64,
                'scope': 'live RGB-presence abstention integration; no recovery, prevention, efficacy or P1 claim', 'P1_admission': False}
    helpers = (validate_inputs, checked_commands, execute, RgbObservabilityGuard, labels,
               base.load_compatible_pretrained, base.collect_nonterminal_boundary_contact_roots)
    sources = [Path(__file__), Path(base.__file__), Path(base.p0.__file__), Path(base.p0.packets.__file__), Path(base.physical.__file__),
               *[Path(inspect.getfile(h)) for h in helpers]]
    seal = create_seal(files=list(dict.fromkeys(p.resolve() for p in sources)), args_json=json.dumps(protocol), output_dir=output/'source_seal')
    base.write_new(output/'protocol.json', protocol)
    host = base.require_cuda_allocation('cuda')
    device = torch.device('cuda')
    model = base.load_compatible_pretrained(policy, device=device).eval().requires_grad_(False)
    before = base.p0._parameter_hash(model)
    buffers = {k:v.clone() for k,v in model.named_buffers()}
    counts = {'encode_calls':0, 'predict_calls':0}
    encode_original, predict_original = model.encode, model.predict
    def encode(*args, **kwargs):
        counts['encode_calls'] += 1
        return encode_original(*args, **kwargs)
    def predict(*args, **kwargs):
        counts['predict_calls'] += 1
        raise RuntimeError('no planning prediction in fixed executor replay')
    model.encode, model.predict = encode, predict
    env = base.gym.make('swm/PushT-v1', render_mode='rgb_array')
    transform = base.p0.image_transform()
    om = {r['root_sha256']:r for r in original['rows']}
    cm = {r['root_sha256']:r for r in compilation['rows']}
    rows = []
    try:
        roots = base.collect_nonterminal_boundary_contact_roots(env, count=4, seed=int(upstream['protocol']['root_seed']), oversample_factor=4)
        if len(om) != 4 or set(om) != set(cm) or set(om) != {base.root_fingerprint(r) for r in roots}:
            raise ValueError('four identical roots required')
        if tuple(env.unwrapped.variation_space['agent']['color'].value.tolist()) != (65,105,225):
            raise ValueError('noncanonical renderer unsupported')
        for index, root in enumerate(roots):
            key = base.root_fingerprint(root)
            old = {b['candidate_id']:b for b in om[key]['branches']}
            branches = []
            for receipt in cm[key]['compiled']['receipts']:
                name = receipt['candidate_id']
                _, _, goalrgb = base.physical._reset_and_warmup(env, root, history_len=3)
                if base.physical._sha256(goalrgb) != upstream['rows'][index]['goal_rgb_sha256']:
                    raise ValueError('goal render changed')
                commands = np.repeat(checked_commands(receipt), 5, axis=0)
                trace = []
                def record(step, command, transition, rgb):
                    state = base.p0.packets.state_of(env)
                    if not np.array_equal(state, transition[0]['state']):
                        raise ValueError('observed state does not match actual transition')
                    if step == 5 and (base.physical._sha256(rgb) != old[name]['stages'][0]['rgb_sha256'] or not np.array_equal(state, old[name]['stages'][0]['state'])):
                        raise ValueError('prefix replay changed')
                    trace.append({'primitive_step':step, 'command':command.tolist(), 'state':state.tolist(), 'rgb_sha256':base.physical._sha256(rgb)})
                result = execute(env, commands, root_id=key, agent_rgb=(65,105,225), on_observation=record)
                rgb = result.pop('terminal_rgb')
                state = base.p0.packets.state_of(env)
                matches = None
                if result['status'] == 'completed':
                    matches = base.physical._sha256(rgb) == old[name]['stages'][-1]['rgb_sha256'] and np.array_equal(state, old[name]['stages'][-1]['state'])
                with torch.no_grad():
                    vector = model.encode({'pixels':base.p0._image_tensor(rgb, transform=transform, device=device)})['emb'][0,0]
                if vector.device.type != 'cuda' or not torch.isfinite(vector).all():
                    raise ValueError('finite CUDA final encoding required')
                branch = {'candidate_id':name, **result, 'trace':trace, 'original_final_exact_if_completed':matches,
                          'final_state':state.tolist(), 'final_vector':vector.cpu().tolist(), 'final_rgb_sha256':base.physical._sha256(rgb),
                          'official':labels(state,root['goal']), 'pose':base.physical._pose_label(state,root['goal'])}
                base.write_new(output/f'root_{index}.{name}.json',branch)
                branches.append(branch)
            row={'root_index':index,'root_sha256':key,'selected_candidate_id':cm[key]['compiled']['selected_candidate_id'],'branches':branches}
            base.write_new(output/f'root_{index}.json',row)
            rows.append(row)
            print(f'root {index}: all16 programs retained, {sum(b["status"]=="observation_unavailable" for b in branches)} observability abstentions',flush=True)
    finally:
        env.close()
        model.encode, model.predict = encode_original, predict_original
    allbranches=[b for r in rows for b in r['branches']]
    gates={'all64_retained':len(allbranches)==64, 'no_execution_errors':all(b['status'] in ('completed','observation_unavailable','environment_terminated') for b in allbranches),
           'completed_programs_exact':all(b['original_final_exact_if_completed'] is True for b in allbranches if b['status']=='completed'),
           'no_commands_after_observation_loss':all(b['completed_commands']==b['guard']['observations'][-1]['primitive_step'] for b in allbranches if b['status']=='observation_unavailable'),
           'actual_encoder64_predict0':counts=={'encode_calls':64,'predict_calls':0},
           'parameters_unchanged':before==base.p0._parameter_hash(model), 'buffers_unchanged':all(torch.equal(buffers[k],v) for k,v in model.named_buffers()),
           'frozen_eval':all(not m.training for m in model.modules()) and all(not p.requires_grad for p in model.parameters()),
           'inputs_unchanged':all(base.digest(Path(p))==d for p,d in bindings.items())}
    verified=verify_seal(output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256'])
    base.write_new(output/'report.json',{'schema':'pusht_visible_executor_cuda.v1','status':'integration_completed' if all(gates.values()) else 'integrity_fail',
        'protocol':protocol,'host':host,'gates':gates,'rows':rows,'counters':counts,'source_seal':seal,'source_verification':verified,'P1_admission':False})
    if not all(gates.values()): raise RuntimeError('visible executor integration integrity failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('execution','policy','output'): parser.add_argument('--'+name,type=Path,required=True)
    run(**vars(parser.parse_args()))
