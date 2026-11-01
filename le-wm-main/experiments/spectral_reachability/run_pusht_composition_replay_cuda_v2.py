"""Replay fixed sealed programs through CUDA composition adapter; NOT P2."""
import argparse
import json
from pathlib import Path
import numpy as np


def root_image_binding(row):
    # Upstream hashes the entire stacked RGB history, not a list of frame hashes.
    value = row['history_rgb_sha256']
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('whole stacked-context RGB digest required')
    return {'root_rgb_sha256': value, 'root_rgb_scope': 'stacked_three_frame_context_not_single_frame'}


def run(preflight, policy, output):
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    from experiments.spectral_reachability import d2p_composition_witness_v2 as witness
    from experiments.spectral_reachability.pusht_composition_cuda_replay_v2 import CudaReplay
    from experiments.spectral_reachability.run_pusht_d2p_sealed_program_execution import checked_commands
    from experiments.spectral_reachability.audit_pusht_d2p_metric_program_arrays import audit
    from experiments.spectral_reachability.d2p_model_call_instrumentation import PredictInstrumentation
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal
    import torch
    preflight, policy, output = map(Path, (preflight, policy, output))
    source = json.loads(preflight.read_text())
    if source['status'] != 'compiled_calibration' or not source['gates'] or not all(v is True for v in source['gates'].values()):
        raise ValueError('successful compilation required')
    old = source['source_seal']
    verify_seal(output_dir=old['output_dir'], expected_manifest_sha256=old['manifest_sha256'])
    bindings = dict(source['protocol']['bindings'])
    bindings[str(preflight.resolve())] = base.digest(preflight)
    if not all(base.digest(Path(p)) == d for p, d in bindings.items()):
        raise ValueError('changed compilation bindings')
    for name in ('weights.pt', 'config.json'):
        if str((policy/name).resolve()) not in bindings:
            raise ValueError('policy must match sealed compilation')
    if len(source['rows']) != 4 or len({r['root_sha256'] for r in source['rows']}) != 4:
        raise ValueError('all four unique calibration roots required')
    for row in source['rows']:
        audit(row['compiled'])
        if len(row['compiled']['receipts']) != 16:
            raise ValueError('all sixteen candidate witnesses required')
    upstream_candidates = [json.loads(Path(p).read_text()) for p in bindings if Path(p).name == 'report.json']
    upstream = next(r for r in upstream_candidates if all('history_rgb_sha256' in row for row in r.get('rows', [])) and len(r.get('rows', [])) == 4)
    roots = {r['root_sha256']: r for r in upstream['rows']}
    if set(roots) != {r['root_sha256'] for r in source['rows']}:
        raise ValueError('root image binding mismatch')
    output.mkdir(exist_ok=False)
    protocol = {'bindings': bindings, 'programs': 64, 'predict_samples_expected': 384,
                'reference_atol': 1e-5, 'reference_rtol': 1e-5,
                'selection_changed': False, 'P2_admission': False,
                'scope': 'known fixed programs; split/full CUDA recurrence equivalence, not program search or efficacy'}
    paths = [Path(__file__), Path(witness.__file__), Path(CudaReplay.__module__.replace('.', '/')).with_suffix('.py'),
             Path(checked_commands.__code__.co_filename), Path(audit.__code__.co_filename),
             Path(base.p0.__file__), Path(base.p0.packets.__file__), Path(base.__file__),
             Path(base.load_compatible_pretrained.__code__.co_filename),
             Path(PredictInstrumentation.__module__.replace('.', '/')).with_suffix('.py')]
    seal = create_seal(files=paths, args_json=json.dumps(protocol), output_dir=output/'source_seal')
    base.write_new(output/'protocol.json', protocol)
    host = base.require_cuda_allocation('cuda')
    device = torch.device('cuda:0')
    model = base.load_compatible_pretrained(policy, device=device).eval().requires_grad_(False)
    before = base.p0._parameter_hash(model)
    buffers = {k: v.clone() for k, v in model.named_buffers()}
    replay = CudaReplay(model, device)
    rows = []
    with PredictInstrumentation(model) as counter:
        with counter.phase('planning', method='composition_replay_contract', horizon_tokens=2):
            for row in source['rows']:
                compiled = row['compiled']
                initial = compiled['input_context']
                context = {'layout': 'states_with_between_state_actions_v1', 'context_length': 3,
                           'latent_dim': 192, 'action_dim': 10, 'latent_history': initial['emb'][0],
                           'action_history': initial['actions'][0][1:]}
                root = {'root_id': row['root_sha256'], **root_image_binding(roots[row['root_sha256']]),
                        'goal_rgb_sha256': roots[row['root_sha256']]['goal_rgb_sha256'], 'context': context}
                provenance = {k: seal['manifest_sha256'] for k in witness.PROVENANCE_KEYS}
                provenance.update(checkpoint_sha256=bindings[str((policy/'weights.pt').resolve())],
                                  encoder_sha256=bindings[str((policy/'weights.pt').resolve())],
                                  protocol_sha256=witness.digest(protocol), metric_sha256=compiled['scoring']['metric']['sha256'],
                                  candidate_bank_sha256=base.digest(preflight))
                domain = {'name': 'raw', 'units': 'official PushT live command', 'low': [-1., -1.], 'high': [1., 1.],
                          'action_block': 5, 'normalization_sha256': witness.digest({'mean': base.p0.PUBLIC_ACTION_MEAN.tolist(), 'std': base.p0.PUBLIC_ACTION_STD.tolist()})}
                values = []
                for receipt in compiled['receipts']:
                    commands = checked_commands(receipt)
                    first, second = (np.repeat(command[None], 5, axis=0).tolist() for command in commands)
                    after = replay(context, first)
                    final = replay(after, second)
                    edges = [dict(program_id='prefix', kind='progress', root_id=root['root_id'], provenance=provenance, action_domain=domain,
                                  input_context=context, output_context=after, actions=first),
                             dict(program_id='suffix', kind='progress', root_id=root['root_id'], provenance=provenance, action_domain=domain,
                                  input_context=after, output_context=final, actions=second)]
                    # Kinds are structural declarations only, not proof of progress.
                    composed = witness.compose(root=root, provenance=provenance, action_domain=domain, edges=edges, replay=replay, max_edges=2, max_actions=10)
                    full = replay(context, first+second)
                    if witness.digest(full) != witness.digest(composed['terminal_context']):
                        raise ValueError('split/full recurrence differs')
                    np.testing.assert_allclose(full['latent_history'][-1], receipt['predicted_terminal_vector'], atol=1e-5, rtol=1e-5)
                    values.append({'candidate_id': receipt['candidate_id'], 'receipt': composed, 'split_full_exact': True,
                                   'saved_forecast_matches': True, 'program_kind_mechanism_verified': False})
                result = {'root_sha256': row['root_sha256'], 'root_index': row['root_index'], 'programs': values}
                base.write_new(output/f'root_{row["root_index"]}.json', result)
                rows.append(result)
                print(f'root {row["root_index"]}: 16 split/full CUDA compositions verified', flush=True)
        records = counter.records
    gates = {'64_programs': sum(len(r['programs']) for r in rows) == 64,
             '384_predict_samples': sum(r['predict_samples'] for r in records) == 384 and all(r['completed'] for r in records),
             'parameters_unchanged': before == base.p0._parameter_hash(model),
             'buffers_unchanged': all(torch.equal(buffers[k], v) for k, v in model.named_buffers()),
             'eval': all(not m.training for m in model.modules()),
             'inputs_unchanged': all(base.digest(Path(p)) == d for p, d in bindings.items())}
    verified = verify_seal(output_dir=seal['output_dir'], expected_manifest_sha256=seal['manifest_sha256'])
    base.write_new(output/'report.json', {'schema': 'pusht_composition_replay_cuda.v2', 'status': 'completed' if all(gates.values()) else 'integrity_fail',
        'protocol': protocol, 'host': host, 'gates': gates, 'rows': rows, 'predict': records, 'source_seal': seal, 'source_verification': verified, 'P2_admission': False})
    if not all(gates.values()):
        raise RuntimeError('composition integration integrity failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('preflight', 'policy', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    run(**vars(parser.parse_args()))
