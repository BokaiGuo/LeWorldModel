"""Fixed observed-context central NEURAL FD; explicit HOST CUDA launch only.

Eight exact physical prefix replays, no physical suffixes, 72 predict samples.
No AD recomputation, selector, thresholds, fitting, training, or P1 admission.
"""
import argparse
import inspect
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability.run_pusht_observed_suffix_physical_fd import (
    _array, _completed, prefix_ids, probes,
)


def keyed(rows, field):
    values = {r[field]: r for r in rows}
    if not rows or len(values) != len(rows):
        raise ValueError(f'empty/duplicate {field}')
    return values


def validate_probes(values, dimension):
    """Require the exact frozen physical probe definition, order and params."""
    if len(values) != 9:
        raise ValueError('exactly nine physical probes required')
    for actual, expected in zip(values, probes(), strict=True):
        if any(actual.get(k) != v for k, v in expected.items()):
            raise ValueError('physical probe definition changed')
        if actual.get('status') != 'executed' or _array(actual['vector']).shape != (dimension,):
            raise ValueError('complete finite physical probe vectors required')
        if not np.array_equal(_array(actual['executed_params']), np.asarray(expected['params'], np.float32).astype(float)):
            raise ValueError('physical executed parameter differs')


def matrix_comparison(left, right):
    left, right = _array(left), _array(right)
    if left.shape != right.shape:
        raise ValueError('Jacobian shapes differ')
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    return dict(mse=float(np.mean((left-right)**2)),
        relative_error_to_right=float(np.linalg.norm(left-right)/np.linalg.norm(right)) if np.linalg.norm(right) else None,
        flattened_cosine=float(np.sum(left*right)/denom) if denom else None,
        left_norm=float(np.linalg.norm(left)), right_norm=float(np.linalg.norm(right)),
        left_column_norms=np.linalg.norm(left, axis=0).tolist(),
        right_column_norms=np.linalg.norm(right, axis=0).tolist())


def central_jacobian(values, epsilon):
    columns = []
    for axis in range(2):
        pair = []
        for sign in (-1, 1):
            matched = [v for v in values if v['epsilon'] == epsilon and v['axis'] == axis and v['sign'] == sign]
            if len(matched) != 1 or matched[0]['status'] != 'executed':
                raise ValueError('missing/duplicate/failed FD probe')
            pair.append(_array(matched[0]['vector']))
        columns.append((pair[1]-pair[0])/(2*float(np.float32(epsilon))))
    return np.stack(columns, axis=1)


def comparisons(neural, physical, plan):
    ad, q = _array(plan['jacobian']), _array(plan['projection'])
    if ad.ndim != 2 or ad.shape[1] != 2 or q.shape != (len(ad), len(ad)):
        raise ValueError('raw AD J[D,2] and frozen left metric[D,D] required')
    validate_probes(neural, len(ad))
    validate_probes(physical, len(ad))
    result = []
    for epsilon in (.01, .05):
        nf, pf = central_jacobian(neural, epsilon), central_jacobian(physical, epsilon)
        result.append(dict(epsilon=epsilon, neural_jacobian=nf.tolist(), physical_jacobian=pf.tolist(),
            neural_vs_saved_ad=matrix_comparison(nf, ad),
            neural_vs_physical=matrix_comparison(nf, pf),
            saved_ad_vs_physical=matrix_comparison(ad, pf),
            projected_neural_vs_saved_ad=matrix_comparison(q@nf, q@ad),
            projected_neural_vs_physical=matrix_comparison(q@nf, q@pf)))
    return result


def evaluate_probes(terminal, make_params, to_array, physical_probes, save):
    """Production call loop shared with tests; exactly one terminal call/probe.

    The runtime adapters below enforce CUDA, float32 and no_grad. No injected
    callable is exposed through CLI and no CPU neural runtime is provided.
    """
    values = []
    for probe in physical_probes:
        definition = {k: probe[k] for k in ('id', 'epsilon', 'axis', 'sign', 'params')}
        params = make_params(probe['executed_params'])
        vector = _array(to_array(terminal(params)))
        if vector.ndim != 1 or vector.shape != _array(probe['vector']).shape:
            raise ValueError('neural terminal dimension mismatch')
        value = dict(definition, status='executed', executed_params=probe['executed_params'], vector=vector.tolist())
        save(value)
        values.append(value)
    return values


def run(*, physical_fd, observed, execution, policy, output):
    # Imports load helpers only; no model/CUDA allocation until sealing below.
    import torch
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    from experiments.spectral_reachability import preflight_pusht_d2p_p1_reconfiguration as p1
    from experiments.spectral_reachability.run_pusht_observed_prefix_prediction import observed_context
    from experiments.spectral_reachability.run_pusht_d2p_sealed_program_execution import checked_commands
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal
    from experiments.spectral_reachability.d2p_model_call_instrumentation import PredictInstrumentation
    physical_fd, observed, execution, policy, output = [Path(p).resolve() for p in (physical_fd, observed, execution, policy, output)]
    fd, feedback, original = [json.loads(p.read_text()) for p in (physical_fd, observed, execution)]
    reports = (fd, feedback, original)
    for report, schema in zip(reports, ('pusht_observed_suffix_physical_fd.v1', 'pusht_observed_oneshot_suffix_control.v1', 'pusht_sealed_execution.v1'), strict=True):
        _completed(report)
        if report['schema'] != schema or report.get('P1_admission') is not False:
            raise ValueError('exact known-calibration schema/no-admission required')
        seal = report['source_seal']
        verify_seal(output_dir=seal['output_dir'], expected_manifest_sha256=seal['manifest_sha256'])
    if fd['protocol']['probes'] != probes() or feedback['protocol']['suffix_horizon'] != 1 or feedback['protocol']['damping'] != .01 or feedback['protocol']['gn_iterations'] != 0:
        raise ValueError('fixed probe/observed one-shot contract differs')
    bindings = dict(fd['protocol']['bindings'])
    for path in (observed, execution):
        if bindings.get(str(path)) != base.digest(path):
            raise ValueError('physical FD must bind supplied reports')
    for path, sha in feedback['protocol']['bindings'].items():
        if bindings.get(path) != sha:
            raise ValueError('conflicting/missing observed provenance')
    cp = Path(original['protocol']['preflight']).resolve()
    if bindings.get(str(cp)) != original['protocol']['preflight_sha256'] or base.digest(cp) != original['protocol']['preflight_sha256']:
        raise ValueError('original compilation binding differs')
    compilation = json.loads(cp.read_text())
    if compilation['status'] != 'compiled_calibration' or not compilation['gates'] or not all(v is True for v in compilation['gates'].values()):
        raise ValueError('complete sealed compilation required')
    s = compilation['source_seal']
    verify_seal(output_dir=s['output_dir'], expected_manifest_sha256=s['manifest_sha256'])
    for path, sha in compilation['protocol']['bindings'].items():
        if bindings.get(path) != sha:
            raise ValueError('compilation input binding differs')
    for name in ('weights.pt', 'config.json'):
        if str(policy/name) not in bindings:
            raise ValueError('checkpoint must be bound')
    bindings[str(physical_fd)] = base.digest(physical_fd)
    cm, pm, fm, dm = [keyed(r['rows'], 'root_sha256') for r in (compilation, original, feedback, fd)]
    if len(cm) != 4 or not set(cm) == set(pm) == set(fm) == set(dm):
        raise ValueError('four identical unique roots required')
    plans = {}
    for key, row in cm.items():
        ids = prefix_ids(row['compiled'])
        if pm[key]['selected_candidate_id'] != ids[0] or fm[key]['selected_candidate_id'] != ids[0]:
            raise ValueError('original selection differs')
        db = keyed(dm[key]['branches'], 'candidate_id')
        if set(db) != set(ids):
            raise ValueError('FD contexts must be original selected and prefix0 only')
        fb, pb = keyed(fm[key]['branches'], 'candidate_id'), keyed(pm[key]['branches'], 'candidate_id')
        for name in ids:
            path = Path(fb[name]['plan_path']).resolve()
            sha = fb[name]['plan_sha256']
            if bindings.get(str(path)) != sha or base.digest(path) != sha:
                raise ValueError('observed stageplan binding changed')
            plan = json.loads(path.read_text())
            if plan['candidate_id'] != name or plan['root_sha256'] != key:
                raise ValueError('stageplan identity mismatch')
            j, nominal, q = [_array(plan[k]) for k in ('jacobian', 'nominal', 'projection')]
            if nominal.ndim != 1 or j.shape != (nominal.size, 2) or q.shape != (nominal.size, nominal.size):
                raise ValueError('saved observed nominal/J/Q dimensions differ')
            if not np.array_equal(q, _array(row['compiled']['scoring']['metric']['projection'])):
                raise ValueError('frozen metric differs')
            if not np.array_equal(j, _array(db[name]['saved_model_jacobian'])) or not np.array_equal(nominal, _array(db[name]['saved_model_zero_nominal'])):
                raise ValueError('physical FD and observed plan model arrays differ')
            validate_probes(db[name]['probes'], nominal.size)
            for probe in db[name]['probes']:
                if probe.get('prefix_replay_exact') is not True or probe['prefix_rgb_sha256'] != pb[name]['stages'][0]['rgb_sha256'] or not np.array_equal(_array(probe['prefix_state']), _array(pb[name]['stages'][0]['state'])):
                    raise ValueError('physical FD prefix context binding differs')
            plans[key, name] = plan
    candidates = [json.loads(Path(p).read_text()) for p in bindings if Path(p).name == 'report.json']
    interventions = [r for r in candidates if 'bank_path' in r.get('protocol', {})]
    if len(interventions) != 1:
        raise ValueError('unique root provenance required')
    provenance = interventions[0]['protocol']
    bankpath = Path(provenance['bank_path']).resolve()
    if base.digest(bankpath) != provenance['bank_sha256']:
        raise ValueError('root bank changed')
    bank = json.loads(bankpath.read_text())
    upstream_path = Path(bank['upstream_path']).resolve()
    if base.digest(upstream_path) != bank['upstream_sha256'] or bank['upstream_sha256'] != provenance['upstream_sha256']:
        raise ValueError('root upstream changed')
    for path in (bankpath, upstream_path):
        digest = base.digest(path)
        if str(path) in bindings and bindings[str(path)] != digest:
            raise ValueError('conflicting root provenance binding')
        bindings[str(path)] = digest
    upstream = base.physical._load_preflight(upstream_path)
    if not all(base.digest(Path(p)) == sha for p, sha in bindings.items()):
        raise ValueError('input hashes changed')
    protocol = dict(bindings=bindings, probes=probes(), expected_contexts=8, expected_predict_calls=72,
        expected_predict_samples=72, expected_prefix_replays=8, physical_suffixes=0,
        selection='original_sealed_selected_and_prefix_0', selector_changed=False,
        thresholds_changed=False, suffix_horizon=1, scope='known-calibration neural FD vs saved AD and physical FD; no efficacy or causal verdict')
    output.mkdir(exist_ok=False)
    helpers = (observed_context, checked_commands, create_seal, verify_seal, PredictInstrumentation,
        probes, base.load_compatible_pretrained, base.collect_nonterminal_boundary_contact_roots)
    sources = [Path(__file__), Path(p1.__file__), Path(base.__file__), Path(base.p0.__file__),
        Path(base.physical.__file__), Path(base.p0.packets.__file__), *[Path(inspect.getfile(h)) for h in helpers]]
    for report in (*reports, compilation):
        manifest = json.loads((Path(report['source_seal']['output_dir'])/'manifest.json').read_text())
        sources.extend(Path(f['source']) for f in manifest['files'])
    seal = create_seal(files=list(dict.fromkeys(p.resolve() for p in sources)), args_json=json.dumps(protocol), output_dir=output/'source_seal')
    base.write_new(output/'protocol.json', protocol)
    verify_seal(output_dir=seal['output_dir'], expected_manifest_sha256=seal['manifest_sha256'])
    # FIRST CUDA allocation/model loading occurs only after bindings and seal.
    host = base.require_cuda_allocation('cuda')
    device = torch.device('cuda')
    model = base.load_compatible_pretrained(policy, device=device).eval().requires_grad_(False)
    if any(p.device.type != 'cuda' or p.requires_grad for p in model.parameters()) or any(b.device.type != 'cuda' for b in model.buffers()):
        raise ValueError('frozen CUDA-only model required')
    before = base.p0._parameter_hash(model)
    buffers = {k: v.clone() for k, v in model.named_buffers()}
    env = base.gym.make('swm/PushT-v1', render_mode='rgb_array')
    rows, records, replay_count, error = [], [], 0, None
    transform = base.p0.image_transform()
    try:
        roots = base.collect_nonterminal_boundary_contact_roots(env, count=4, seed=int(upstream['protocol']['root_seed']), oversample_factor=4)
        if len(roots) != 4 or {base.root_fingerprint(r) for r in roots} != set(cm):
            raise ValueError('recreated roots differ')
        with PredictInstrumentation(model) as counter:
            try:
                for index, root in enumerate(roots):
                    key = base.root_fingerprint(root)
                    if any(mapping[key]['root_index'] != index for mapping in (cm, pm, fm, dm)):
                        raise ValueError('root order differs')
                    values = []
                    for name in prefix_ids(cm[key]['compiled']):
                        saved = keyed(pm[key]['branches'], 'candidate_id')[name]
                        physical = keyed(dm[key]['branches'], 'candidate_id')[name]
                        receipt = keyed(cm[key]['compiled']['receipts'], 'candidate_id')[name]
                        frames, history, rgbgoal = base.physical._reset_and_warmup(env, root, history_len=3)
                        if base.physical._sha256(rgbgoal) != upstream['rows'][index]['goal_rgb_sha256']:
                            raise ValueError('goal RGB differs')
                        command = checked_commands(receipt)[0]
                        base.p0.packets.execute_packet(env, base.p0.packets.repeat_command(command))
                        rgb = np.asarray(env.render(), np.uint8).copy()
                        state = base.p0.packets.state_of(env)
                        if base.physical._sha256(rgb) != saved['stages'][0]['rgb_sha256'] or not np.array_equal(state, saved['stages'][0]['state']):
                            raise ValueError('prefix RGB/state exact replay failed')
                        replay_count += 1
                        frames, history = observed_context(frames, history, rgb, command)
                        with torch.no_grad():
                            emb, actions = p1._encode_context(model, frames, history, transform, device)
                            if emb.device.type != 'cuda' or actions.device.type != 'cuda' or emb.dtype != torch.float32 or actions.dtype != torch.float32 or not torch.isfinite(emb).all() or not torch.isfinite(actions).all():
                                raise ValueError('finite CUDA float32 observed context required')
                            with counter.phase('evaluation', method=f'neural_fd_root{index}_{name}', horizon_tokens=1):
                                terminal = lambda params: p1._suffix_terminal(model, emb, actions, params.reshape(1, 1, 2))[0]
                                def to_array(vector):
                                    if vector.device.type != 'cuda' or vector.dtype != torch.float32:
                                        raise ValueError('CUDA float32 neural prediction required')
                                    return vector.cpu().numpy().astype(float)
                                def save_probe(value):
                                    value.update(root_sha256=key, root_index=index, candidate_id=name,
                                        prefix_rgb_sha256=base.physical._sha256(rgb), prefix_replay_exact=True)
                                    base.write_new(output/f'root_{index}.{name}.{value["id"]}.json', value)
                                neural = evaluate_probes(terminal,
                                    lambda params: torch.tensor(params, dtype=torch.float32, device=device), to_array,
                                    physical['probes'], save_probe)
                        plan = plans[key, name]
                        zero = _array(neural[0]['vector'])
                        branch = dict(candidate_id=name, prefix_replay_exact=True, prefix_rgb_sha256=base.physical._sha256(rgb),
                            prefix_state=state.tolist(), saved_ad_jacobian=plan['jacobian'], saved_nominal=plan['nominal'],
                            zero_nominal_exact=np.array_equal(zero, _array(plan['nominal'])),
                            zero_nominal_max_abs=float(np.max(np.abs(zero-_array(plan['nominal'])))),
                            probes=neural, comparisons=comparisons(neural, physical['probes'], plan))
                        base.write_new(output/f'root_{index}.{name}.json', branch)
                        values.append(branch)
                    rows.append(dict(root_index=index, root_sha256=key, branches=values))
                    base.write_new(output/f'root_{index}.json', rows[-1])
                    print(f'root {index}: 2 exact prefix replays, 18 CUDA neural predictions; no physical suffixes', flush=True)
            finally:
                records = counter.records
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        env.close()
    verified = verify_seal(output_dir=seal['output_dir'], expected_manifest_sha256=seal['manifest_sha256'])
    branches = [b for r in rows for b in r['branches']]
    gates = dict(eight_exact_contexts=len(branches)==8 and replay_count==8,
        all72_predictions=sum(len(b['probes']) for b in branches)==72,
        predict72=len(records)==8 and sum(r['predict_calls'] for r in records)==72 and sum(r['predict_samples'] for r in records)==72 and all(r['completed'] and r['failed_predict_calls']==0 for r in records),
        saved_nominal_exact=len(branches)==8 and all(b['zero_nominal_exact'] for b in branches),
        parameters_unchanged=before==base.p0._parameter_hash(model),
        buffers_unchanged=set(buffers)==set(dict(model.named_buffers())) and all(torch.equal(buffers[k],v) for k,v in model.named_buffers()),
        frozen_eval_cuda=all(not m.training for m in model.modules()) and all(not p.requires_grad and p.device.type=='cuda' for p in model.parameters()) and all(b.device.type=='cuda' for b in model.buffers()),
        inputs_unchanged=all(base.digest(Path(p))==sha for p,sha in bindings.items()), no_runtime_error=error is None)
    base.write_new(output/'report.json', dict(schema='pusht_observed_suffix_neural_fd.v1',
        status='completed' if all(gates.values()) else 'integrity_fail', protocol=protocol, host=host,
        rows=rows, predict=records, physical_prefix_replays=replay_count, error=error, gates=gates,
        source_seal=seal, source_verification=verified, P1_admission=False, selection_changed=False))
    if not all(gates.values()):
        raise RuntimeError('neural FD integrity failed; partial receipts retained: '+str(error))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('physical-fd', 'observed', 'execution', 'policy', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    run(**vars(parser.parse_args()))
