"""Frozen local physical parameter derivative diagnostic; no selection or P1.

Only run explicitly on parent-coordinated HOST CUDA. Import/tests do no neural
work. Nine fixed suffix probes per originally selected/zero prefix, 72 total.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def probes():
    result = [dict(id='zero', epsilon=0., axis=None, sign=0, params=[0., 0.])]
    for epsilon in (.01, .05):
        for axis in range(2):
            for sign in (-1, 1):
                params = [0., 0.]
                params[axis] = sign * epsilon
                result.append(dict(id=f'e{epsilon}.a{axis}.s{sign}', epsilon=epsilon,
                                   axis=axis, sign=sign, params=params))
    return result


def prefix_ids(row):
    selected = row['selected_candidate_id']
    if selected not in {f'prefix_{i}' for i in range(1, 16)}:
        raise ValueError('original selected must be a distinct nonzero sealed prefix')
    return [selected, 'prefix_0']


def _array(value):
    # Check scalars recursively before numpy can erase mixed bool types.
    if isinstance(value, list):
        for v in value:
            _array(v)
    elif isinstance(value, (bool, str, dict)):
        raise ValueError('finite real numeric arrays required')
    a = np.asarray(value)
    if a.dtype.kind not in 'fi' or not np.isfinite(a).all():
        raise ValueError('finite real numeric arrays required')
    return a.astype(float)


def compare(rows, jacobian, projection):
    j, q = _array(jacobian), _array(projection)
    if j.ndim != 2 or j.shape[1] != 2 or q.shape != (j.shape[0], j.shape[0]):
        raise ValueError('saved raw J[D,2] and frozen projection[D,D] required')
    results = []
    for epsilon in (.01, .05):
        try:
            columns = []
            for axis in range(2):
                values = []
                for sign in (-1, 1):
                    matched = [r for r in rows if r['epsilon'] == epsilon and r['axis'] == axis and r['sign'] == sign]
                    if len(matched) != 1 or matched[0]['status'] != 'executed':
                        raise ValueError('missing/failed/duplicate physical probe')
                    vector = _array(matched[0]['vector'])
                    if vector.shape != (j.shape[0],):
                        raise ValueError('physical latent dimension mismatch')
                    values.append(vector)
                # Denominator is the actual float32 PARAMETER displacement,
                # not a live-command displacement (tanh chain already in J).
                columns.append((values[1]-values[0])/(2*float(np.float32(epsilon))))
            physical = np.stack(columns, axis=1)
            delta = physical-j
            results.append(dict(epsilon=epsilon, status='compared', physical_jacobian=physical.tolist(),
                raw_mse=float(np.mean(delta**2)), projected_mse=float(np.mean((q@delta)**2)),
                model_column_norms=np.linalg.norm(j, axis=0).tolist(),
                physical_column_norms=np.linalg.norm(physical, axis=0).tolist()))
        except ValueError as exc:
            results.append(dict(epsilon=epsilon, status='unavailable', error=str(exc)))
    return results


def _completed(report):
    if report['status'] != 'completed' or not report['gates'] or not all(v is True for v in report['gates'].values()):
        raise ValueError('complete integrity-checked upstream report required')


def run(*, execution, observed, policy, output):
    import torch
    import inspect
    from PIL import Image
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    from experiments.spectral_reachability.run_pusht_d2p_sealed_program_execution import checked_commands
    from experiments.spectral_reachability.audit_pusht_sealed_task_contract import labels
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal
    execution, observed, policy, output = map(Path, (execution, observed, policy, output))
    physical, feedback = [json.loads(p.read_text()) for p in (execution, observed)]
    for report in (physical, feedback):
        _completed(report)
        s = report['source_seal']
        verify_seal(output_dir=s['output_dir'], expected_manifest_sha256=s['manifest_sha256'])
    cp = Path(physical['protocol']['preflight'])
    if base.digest(cp) != physical['protocol']['preflight_sha256']:
        raise ValueError('compilation changed')
    compilation = json.loads(cp.read_text())
    if compilation['status'] != 'compiled_calibration' or not compilation['gates'] or not all(v is True for v in compilation['gates'].values()):
        raise ValueError('sealed compilation required')
    s = compilation['source_seal']
    verify_seal(output_dir=s['output_dir'], expected_manifest_sha256=s['manifest_sha256'])
    bindings = dict(compilation['protocol']['bindings'])
    for p in (execution, cp):
        if feedback['protocol']['bindings'].get(str(p.resolve())) != base.digest(p):
            raise ValueError('observed report does not bind original execution/compilation')
    for p, sha in feedback['protocol']['bindings'].items():
        if p in bindings and bindings[p] != sha:
            raise ValueError('conflicting provenance')
        bindings[p] = sha
    for name in ('weights.pt', 'config.json'):
        if str((policy/name).resolve()) not in bindings:
            raise ValueError('policy not bound')
    for p in (execution, observed, cp):
        bindings[str(p.resolve())] = base.digest(p)
    cm = {r['root_sha256']: r for r in compilation['rows']}
    pm = {r['root_sha256']: r for r in physical['rows']}
    fm = {r['root_sha256']: r for r in feedback['rows']}
    if any(len(r['rows']) != 4 for r in (physical, feedback, compilation)) or len(cm) != 4 or set(cm) != set(pm) or set(cm) != set(fm):
        raise ValueError('four unique matching roots required')
    plans = {}
    for key, row in cm.items():
        ids = prefix_ids(row['compiled'])
        if pm[key]['selected_candidate_id'] != ids[0]:
            raise ValueError('original sealed selection differs')
        for name in ids:
            matches = [b for b in fm[key]['branches'] if b['candidate_id'] == name]
            if len(matches) != 1:
                raise ValueError('unique observed stageplan required')
            b = matches[0]
            path = Path(b['plan_path'])
            if base.digest(path) != b['plan_sha256']:
                raise ValueError('saved observed stageplan changed')
            plan = json.loads(path.read_text())
            if plan['root_sha256'] != key or plan['candidate_id'] != name:
                raise ValueError('stageplan identity mismatch')
            compiled = row['compiled']
            if not np.array_equal(_array(plan['projection']), _array(compiled['scoring']['metric']['projection'])):
                raise ValueError('frozen Q differs')
            nominal, j = _array(plan['nominal']), _array(plan['jacobian'])
            if nominal.ndim != 1 or j.shape != (nominal.size, 2):
                raise ValueError('observed nominal/J dimensions differ')
            compare([], j, plan['projection'])
            plans[key, name] = plan
            bindings[str(path.resolve())] = b['plan_sha256']
    candidates = [json.loads(Path(p).read_text()) for p in bindings if Path(p).name == 'report.json']
    interventions = [r for r in candidates if 'bank_path' in r.get('protocol', {})]
    if len(interventions) != 1:
        raise ValueError('unique root provenance required')
    bankpath = Path(interventions[0]['protocol']['bank_path'])
    if base.digest(bankpath) != interventions[0]['protocol']['bank_sha256']:
        raise ValueError('bank provenance digest changed')
    if str(bankpath.resolve()) not in bindings:
        # Freeze the actual bank and upstream used for root recreation too.
        bindings[str(bankpath.resolve())] = base.digest(bankpath)
    bank = json.loads(bankpath.read_text())
    upstream_path = Path(bank['upstream_path'])
    if base.digest(upstream_path) != bank['upstream_sha256'] or base.digest(upstream_path) != interventions[0]['protocol']['upstream_sha256']:
        raise ValueError('upstream root provenance digest changed')
    upstream = base.physical._load_preflight(upstream_path)
    bindings[str(upstream_path.resolve())] = base.digest(upstream_path)
    if not all(base.digest(Path(p)) == sha for p, sha in bindings.items()):
        raise ValueError('input hashes changed')
    protocol = dict(bindings=bindings, probes=probes(), expected_branches=72, expected_actual_encodes=72,
        expected_predict_calls=0, raw_steps_per_suffix=5, selection='original_sealed_selected_and_prefix_0_only',
        selector_changed=False, P1_admission=False, scope='known-root local physical parameter derivative diagnostic; no efficacy or causal proof')
    output.mkdir(exist_ok=False)
    sources = [Path(__file__), Path(base.__file__), Path(base.physical.__file__), Path(base.p0.__file__),
        Path(base.p0.packets.__file__), Path(inspect.getfile(checked_commands)), Path(inspect.getfile(labels)),
        Path(inspect.getfile(create_seal)), Path(base.load_compatible_pretrained.__code__.co_filename),
        Path(base.collect_nonterminal_boundary_contact_roots.__code__.co_filename)]
    # Extend upstream sealed implementation surface without copying data/weights.
    for report in (compilation, feedback):
        manifest = json.loads((Path(report['source_seal']['output_dir'])/'manifest.json').read_text())
        sources.extend(Path(f['source']) for f in manifest['files'])
    seal = create_seal(files=list(dict.fromkeys(p.resolve() for p in sources)),
                       args_json=json.dumps(protocol), output_dir=output/'source_seal')
    base.write_new(output/'protocol.json', protocol)
    verify_seal(output_dir=seal['output_dir'], expected_manifest_sha256=seal['manifest_sha256'])
    # First CUDA allocation is strictly after input validation and run seal.
    host = base.require_cuda_allocation('cuda')
    device = torch.device('cuda')
    model = base.load_compatible_pretrained(policy, device=device).eval().requires_grad_(False)
    if any(p.device.type != 'cuda' or p.requires_grad for p in model.parameters()) or any(b.device.type != 'cuda' for b in model.buffers()):
        raise ValueError('frozen CUDA-only model required')
    before = base.p0._parameter_hash(model)
    buffers = {k: v.clone() for k, v in model.named_buffers()}
    counts = dict(actual_encoder_calls=0, predict_calls=0)
    def count_encode(module, args):
        counts['actual_encoder_calls'] += 1
    def forbid_predict(module, args):
        counts['predict_calls'] += 1
        raise RuntimeError('model.predict forbidden in physical FD diagnostic')
    # encode/predict are methods, so wrap them, restoring in finally.
    original_encode, original_predict = model.encode, model.predict
    def encode(*args, **kwargs):
        count_encode(None, None)
        return original_encode(*args, **kwargs)
    def predict(*args, **kwargs):
        forbid_predict(None, None)
    model.encode, model.predict = encode, predict
    env = base.gym.make('swm/PushT-v1', render_mode='rgb_array')
    transform = base.p0.image_transform()
    rows = []
    try:
        roots = base.collect_nonterminal_boundary_contact_roots(env, count=4, seed=int(upstream['protocol']['root_seed']), oversample_factor=4)
        if len(roots) != 4 or {base.root_fingerprint(r) for r in roots} != set(cm):
            raise ValueError('recreated root set differs')
        for index, root in enumerate(roots):
            key = base.root_fingerprint(root)
            if cm[key]['root_index'] != index or pm[key]['root_index'] != index:
                raise ValueError('root order differs from original provenance')
            compiled = cm[key]['compiled']
            branches = []
            for name in prefix_ids(compiled):
                receipt = next(r for r in compiled['receipts'] if r['candidate_id'] == name)
                saved = next(b for b in pm[key]['branches'] if b['candidate_id'] == name)
                plan = plans[key, name]
                values = []
                for probe in probes():
                    value = dict(probe, status='failed')
                    try:
                        _, _, goalrgb = base.physical._reset_and_warmup(env, root, history_len=3)
                        if base.physical._sha256(goalrgb) != upstream['rows'][index]['goal_rgb_sha256']:
                            raise ValueError('goal RGB changed')
                        prefix = checked_commands(receipt)[0]
                        base.p0.packets.execute_packet(env, base.p0.packets.repeat_command(prefix))
                        rgb = np.asarray(env.render(), np.uint8).copy()
                        state = base.p0.packets.state_of(env)
                        if base.physical._sha256(rgb) != saved['stages'][0]['rgb_sha256'] or not np.array_equal(state, saved['stages'][0]['state']):
                            raise ValueError('prefix RGB/state replay differs')
                        params = torch.tensor(probe['params'], dtype=torch.float32, device=device)
                        live = torch.tanh(params).cpu().numpy()
                        packet = base.p0.packets.repeat_command(live)
                        if np.asarray(packet).shape != (5, 2):
                            raise ValueError('exactly five raw commands required')
                        value.update(prefix_replay_exact=True, prefix_rgb_sha256=base.physical._sha256(rgb),
                            prefix_state=state.tolist(), executed_params=params.cpu().tolist(), raw_commands=np.asarray(packet).tolist())
                        base.p0.packets.execute_packet(env, packet)
                        rgb = np.asarray(env.render(), np.uint8).copy()
                        state = base.p0.packets.state_of(env)
                        imagepath = output/f'root_{index}.{name}.{probe["id"]}.png'
                        Image.fromarray(rgb).save(imagepath)
                        official = labels(state, root['goal'])
                        success, distance = env.unwrapped.eval_state(np.asarray(root['goal']), np.asarray(state))
                        if bool(success) != official['official_success'] or not np.isclose(distance, official['official_state_distance'], atol=1e-10, rtol=0):
                            raise ValueError('official evaluator disagrees')
                        value.update(rgb_path=str(imagepath.resolve()), rgb_file_sha256=base.digest(imagepath),
                            rgb_sha256=base.physical._sha256(rgb), state=state.tolist(),
                            object_pose=base.physical._pose_label(state, root['goal']), official=official)
                        with torch.no_grad():
                            vector = model.encode({'pixels': base.p0._image_tensor(rgb, transform=transform, device=device)})['emb'][0, 0]
                        if vector.device.type != 'cuda':
                            raise ValueError('terminal encoding not on CUDA')
                        vector = vector.cpu().numpy().astype(float)
                        if vector.shape != np.asarray(plan['nominal']).shape or not np.isfinite(vector).all():
                            raise ValueError('invalid terminal encoding')
                        value.update(status='executed', vector=vector.tolist())
                    except Exception as exc:
                        value.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                    values.append(value)
                    base.write_new(output/f'root_{index}.{name}.{probe["id"]}.json', value)
                zero = values[0]
                zero_error = (float(np.mean((_array(zero['vector'])-_array(plan['nominal']))**2))
                              if zero['status']=='executed' else None)
                branches.append(dict(candidate_id=name, saved_model_jacobian=plan['jacobian'],
                    actual_zero_vs_saved_nominal_mse=zero_error,
                    saved_model_zero_nominal=plan['nominal'], probes=values,
                    comparisons=compare(values, plan['jacobian'], plan['projection'])))
            row = dict(root_index=index, root_sha256=key, branches=branches)
            base.write_new(output/f'root_{index}.json', row)
            rows.append(row)
            print(f'root {index}: fixed 18 physical FD probe receipts saved', flush=True)
    finally:
        env.close()
        model.encode, model.predict = original_encode, original_predict
    verified = verify_seal(output_dir=seal['output_dir'], expected_manifest_sha256=seal['manifest_sha256'])
    gates = dict(all_72_executed=len(rows)==4 and sum(p['status']=='executed' for r in rows for b in r['branches'] for p in b['probes'])==72,
        actual_encoder72=counts['actual_encoder_calls']==72, no_predict=counts['predict_calls']==0,
        parameters_unchanged=before==base.p0._parameter_hash(model),
        buffers_unchanged=set(buffers)==set(dict(model.named_buffers())) and all(torch.equal(buffers[k],v) for k,v in model.named_buffers()),
        frozen_eval=all(not m.training for m in model.modules()) and all(not p.requires_grad for p in model.parameters()),
        inputs_unchanged=all(base.digest(Path(p))==sha for p,sha in bindings.items()))
    base.write_new(output/'report.json', dict(schema='pusht_observed_suffix_physical_fd.v1',
        status='completed' if all(gates.values()) else 'integrity_fail', protocol=protocol,
        host=host, rows=rows, counters=counts, gates=gates, source_seal=seal,
        source_verification=verified, P1_admission=False, selection_changed=False))
    if not all(gates.values()):
        raise RuntimeError('physical FD integrity failure; failed probe receipts retained')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('execution', 'observed', 'policy', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    run(**vars(parser.parse_args()))
