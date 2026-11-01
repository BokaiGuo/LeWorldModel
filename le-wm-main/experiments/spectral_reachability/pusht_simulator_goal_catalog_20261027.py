"""Official reset/render goal catalog; privileged task-spec diagnostic, not P1.

No edited RGB, body assignments, learned readout, action selection or CPU neural
fallback. Eligibility uses requested task geometry and actual GOAL-render state,
not the restored reset observation, future probe states or control outcomes.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def pixels_sha(rgb):
    return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()


def goal_state(value):
    a = np.asarray(value)
    if a.shape != (7,) or a.dtype.kind not in 'fi' or not np.isfinite(a).all():
        raise ValueError('finite seven-coordinate task goal required')
    if np.any(a[-2:] != 0):
        raise ValueError('nonzero goal agent velocity unsupported; cannot freeze scene by ignoring it')
    return a.astype(float).copy()


def penetration_records(unwrapped):
    """Actual simulator geometry, including walls; strictly negative depth only."""
    records = []
    for shape in unwrapped.agent.shapes:
        for hit in unwrapped.space.shape_query(shape):
            if hit.shape.body is unwrapped.agent:
                continue
            depths = [float(p.distance) for p in hit.contact_point_set.points]
            if any(d < 0 for d in depths):
                records.append(dict(shape_type=type(hit.shape).__name__, depths=depths,
                                    block=hit.shape.body is unwrapped.block))
    return records


def render_goal_view(env, requested):
    """Capture the render INSIDE reset, before its second _set_state.

    Wrap only render, never physics/body state. Keep requested, actual state,
    RGB hash, overlap and drift independently. Numerical equality tolerance
    1e-9 is for floating state representation, not physical task success.
    """
    requested = goal_state(requested)
    u = env.unwrapped
    original = u.render
    existed, previous = 'render' in vars(u), vars(u).get('render')
    captures = []

    def capture():
        before = np.asarray(u._get_obs(), float).copy()
        overlaps = penetration_records(u)
        visible = all(s.bb.left >= 0 and s.bb.bottom >= 0 and
                      s.bb.right <= u.window_size and s.bb.top <= u.window_size
                      for s in u.agent.shapes)
        rgb = np.asarray(original()).copy()
        after = np.asarray(u._get_obs(), float).copy()
        captures.append((before, after, overlaps, rgb, visible))
        return rgb

    u.render = capture
    try:
        _, info = env.reset(seed=0, options={'state': requested.copy(), 'goal_state': requested.copy()})
    finally:
        if existed:
            u.render = previous
        else:
            delattr(u, 'render')
    if len(captures) != 1:
        raise ValueError('official reset must render exactly one goal before restoring state')
    actual, after, overlaps, rgb, visible = captures[0]
    if (rgb.dtype != np.uint8 or rgb.shape != (224, 224, 3) or
            actual.shape != (7,) or not np.isfinite(actual).all() or
            not np.array_equal(actual, after) or not np.array_equal(rgb, info['goal'])):
        raise ValueError('goal capture shape/state/cache/render invariance failed')
    delta = actual - requested
    delta[4] = np.arctan2(np.sin(delta[4]), np.cos(delta[4]))
    reasons = []
    if not np.allclose(delta[2:5], 0, rtol=0, atol=1e-9):
        reasons.append('goal_object_drift_during_reset')
    if not np.allclose(delta[[0, 1, 5, 6]], 0, rtol=0, atol=1e-9):
        reasons.append('goal_agent_drift_during_reset')
    if overlaps:
        reasons.append('agent_penetrates_object_or_wall')
    if not visible:
        reasons.append('agent_geometry_not_fully_in_canvas')
    return dict(requested_state=requested.tolist(), actual_goal_render_state=actual.tolist(),
                state_delta=delta.tolist(), penetrations=overlaps, valid=not reasons,
                invalid_reasons=reasons, agent_fully_in_canvas=visible,
                rgb_sha256=pixels_sha(rgb), rgb=rgb)


def build_catalog(env, task_goal):
    goal = goal_state(task_goal)
    u = env.unwrapped
    if (type(u).__module__ != 'stable_worldmodel.envs.pusht.env' or
            u.render_mode != 'rgb_array' or u.window_size != 512):
        raise ValueError('installed official 512-coordinate PushT renderer required')
    positions = np.rint(np.linspace(16, 208, 16)).astype(int)
    entries = []
    for y in positions:
        for x in positions:
            target = goal.copy()
            target[:2] = np.asarray([x, y]) * (512 / 224)
            view = render_goal_view(env, target)
            view.update(id=f'agent_{x}_{y}', grid_image_xy=[int(x), int(y)])
            entries.append(view)
    if len(entries) != 256 or not any(e['valid'] for e in entries):
        raise ValueError('all256 grid slots and at least one physically eligible goal required')
    return entries


def common_id_catalogs(simulator, clean):
    """Match generated grid identities only; never mix original five goals."""
    aligned_sim, aligned_clean, coverage = {}, {}, []
    if set(simulator) != set(clean) or len(simulator) != 4:
        raise ValueError('all four simulator/clean roots required')
    for key, sim in simulator.items():
        old = clean[key]
        maps = []
        for catalog in (sim, old):
            if len(catalog['ids']) != len(catalog['vectors']) or len(set(catalog['ids'])) != len(catalog['ids']):
                raise ValueError('unique complete catalog identities required')
            maps.append(dict(zip(catalog['ids'], catalog['vectors'], strict=True)))
        sm, cm = maps
        sids = {i for i in sm if i.startswith('agent_')}
        cids = {i for i in cm if i.startswith('agent_')}
        shared = sorted(sids & cids)
        if not shared:
            raise ValueError('empty common grid catalog; cannot compare renderer paths')
        aligned_sim[key] = dict(root_sha256=key, ids=shared, vectors=[sm[i] for i in shared])
        aligned_clean[key] = dict(root_sha256=key, ids=shared, vectors=[cm[i] for i in shared])
        coverage.append(dict(root_sha256=key, common_ids=shared, simulator_grid_ids=sorted(sids),
                             clean_grid_ids=sorted(cids), simulator_only_ids=sorted(sids-cids),
                             clean_only_ids=sorted(cids-sids),
                             clean_original_goals_excluded=sorted(set(cm)-cids)))
    return aligned_sim, aligned_clean, coverage


def diagnose(catalogs, physical, control):
    """All saved outcomes retained, including guards; no fitting or selection."""
    from experiments.spectral_reachability.pusht_goal_set_metric import distances
    for report, status in ((physical, 'diagnostic_completed'), (control, 'calibration_completed')):
        if (report['status'] != status or not report['gates'] or
                not all(v is True for v in report['gates'].values()) or
                len(report['rows']) != 4 or
                len({r['root_sha256'] for r in report['rows']}) != 4 or
                {r['root_sha256'] for r in report['rows']} != set(catalogs)):
            raise ValueError('all four complete saved outcome roots required')
    probes, controls = [], []
    for row in physical['rows']:
        vectors = catalogs[row['root_sha256']]['vectors']
        contexts = []
        for branch in row['contexts']:
            if len(branch['probes']) != 5:
                raise ValueError('all five probes must be retained')
            result = distances([p['vector'] for p in branch['probes']], vectors)
            contexts.append(dict(candidate_id=branch['candidate_id'],
                statuses=[p['status'] for p in branch['probes']], **result,
                cost_range=float(np.ptp(result['costs'])),
                object_pose_invariant=len({tuple(p['state'][2:5]) for p in branch['probes']}) == 1))
        probes.append(dict(root_sha256=row['root_sha256'], contexts=contexts))
    for row in control['rows']:
        result = distances([b['final_vector'] for b in row['branches']], catalogs[row['root_sha256']]['vectors'])
        controls.append(dict(root_sha256=row['root_sha256'], branches=[dict(
            candidate_id=b['candidate_id'], status=b['status'], goal_set_cost=result['costs'][i],
            goal_index=result['goal_indices'][i], object_pose_cost=b['pose']['pose_cost'])
            for i, b in enumerate(row['branches'])]))
    return dict(probe_diagnostics=probes, control_diagnostics=controls)


def run(compilation, probes, calibration, clean_report, policy, output, prepare_only=False):
    from PIL import Image
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal
    from experiments.spectral_reachability.pusht_goal_set_metric import distances
    compilation, probes, calibration, clean_report, policy, output = map(Path, (compilation, probes, calibration, clean_report, policy, output))
    if 'pusht_simulator_goal_catalog_20261027' not in output.name.lower():
        raise ValueError('dedicated new output directory name required')
    source, bindings, config = checked.validate_inputs(compilation, policy)
    for path in (probes, calibration):
        bindings[str(path.resolve())] = sha(path)  # bytes only, no outcome deserialization yet
    clean_meta = json.loads(clean_report.read_text())
    if (clean_meta['status'] != 'construction_completed' or not clean_meta['gates'] or
            not all(v is True for v in clean_meta['gates'].values())):
        raise ValueError('complete prior clean catalog required')
    # Do not consult embedded outcome diagnostics for eligibility.
    bindings[str(clean_report.resolve())] = sha(clean_report)
    bindings.update(clean_meta['catalog_hashes'])
    for path, expected_sha in clean_meta['catalog_hashes'].items():
        if sha(path) != expected_sha:
            raise ValueError('clean catalog hash changed')
    upstream = next(json.loads(Path(p).read_text()) for p in bindings
        if Path(p).name == 'report.json' and p not in (str(compilation.resolve()), str(clean_report.resolve())) and
        p not in (str(probes.resolve()), str(calibration.resolve())) and
        'root_seed' in json.loads(Path(p).read_text()).get('protocol', {}))
    output.mkdir(exist_ok=False)
    env = base.gym.make('swm/PushT-v1', render_mode='rgb_array')
    renderer_path = Path(inspect.getfile(type(env.unwrapped)))
    prepared = []
    try:
        roots = base.collect_nonterminal_boundary_contact_roots(env, count=4,
            seed=int(upstream['protocol']['root_seed']), oversample_factor=4)
        expected = {r['root_sha256']: r for r in source['rows']}
        if len(expected) != 4 or {base.root_fingerprint(r) for r in roots} != set(expected):
            raise ValueError('must reconstruct all four sealed roots; no reselection')
        for i, root in enumerate(roots):
            key = base.root_fingerprint(root)
            # Original supplied goal image binding, captured at goal-render time.
            reference = render_goal_view(env, root['goal'])
            refrow = next(r for r in upstream['rows'] if r['root_sha256'] == key)
            if reference['rgb_sha256'] != refrow['goal_rgb_sha256'] or not reference['valid']:
                raise ValueError('protocol insufficient: original goal RGB/state/geometry incompatible; '
                                 'stop all roots without relaxing eligibility: '+str(reference['invalid_reasons']))
            entries = build_catalog(env, root['goal'])
            for entry in entries:
                rgb = entry.pop('rgb')
                name = f'pusht_simulator_goal_catalog_20261027_root_{i}_{entry["id"]}.png'
                Image.fromarray(rgb).save(output/name)
                entry.update(rgb_path=str((output/name).resolve()), rgb_file_sha256=sha(output/name))
            reference.pop('rgb')
            prepared.append(dict(root_index=i, root_sha256=key, task_goal=np.asarray(root['goal']).tolist(),
                                 original_goal_reference=reference, entries=entries))
            print(f'root {i}: simulator valid {sum(e["valid"] for e in entries)}/256; all slots retained', flush=True)
    finally:
        env.close()
    image_bindings = {e['rgb_path']: e['rgb_file_sha256'] for r in prepared for e in r['entries']}
    bindings.update(image_bindings)
    protocol = dict(schema='pusht_simulator_goal_catalog_20261027.v1', bindings=bindings,
        prepared=prepared, grid_image_positions=np.rint(np.linspace(16,208,16)).astype(int).tolist(),
        grid_image_to_world_scale=512/224, grid_coordinates='224-image grid mapped to512 simulator coordinates',
        goal_state_provenance='privileged original task specification; NOT future outcomes',
        planner_permission='additional simulator-generated goal RGB catalog; NOT single-RGB deployment',
        eligibility='goal-render actual state within1e-9 of task request and no geometry penetration',
        all_invalid_slots_retained=True, goal_object_pose_fixed=True, original_goal_vectors_added=False,
        uses_edited_RGB=False, direct_body_state_assignment=False, batch_size=16,
        expected_encoder_samples=sum(e['valid'] for r in prepared for e in r['entries']),
        physical_outcomes_used_for_eligibility=False, P1_admission=False)
    prefix = 'pusht_simulator_goal_catalog_20261027'
    write_new(output/f'{prefix}_protocol.json', protocol)
    files = checked.sources_before_allocate(source, config) + [Path(__file__).resolve(),
        Path(inspect.getfile(distances)), renderer_path]
    # Official renderer and utility code, collector, compatibility loader and
    # neural constructors are covered alongside tests/docs; no weight copies.
    files += [Path(base.__file__), Path(inspect.getfile(base.collect_nonterminal_boundary_contact_roots)),
              Path(inspect.getfile(base.load_compatible_pretrained)),
              Path(__file__).resolve().parents[2]/'tests/test_pusht_simulator_goal_catalog_20261027.py',
              Path(__file__).resolve().parents[3]/'docs/PUSHT_SIMULATOR_GOAL_CATALOG_20261027.md']
    files = sorted(set(p.resolve() for p in files))
    # args only bind protocol digest: large image manifest is saved separately.
    from experiments.spectral_reachability.d2p_composition_witness_v2 import digest
    seals = [create_seal(files=files[j:j+50], args_json=json.dumps(dict(protocol_sha256=digest(protocol))),
             output_dir=output/f'{prefix}_source_seal_{j//50}') for j in range(0,len(files),50)]
    if prepare_only:
        verified = [verify_seal(output_dir=s['output_dir'], expected_manifest_sha256=s['manifest_sha256']) for s in seals]
        write_new(output/f'{prefix}_prepared.json', dict(status='render_prepared_no_neural_work',
            protocol_sha256=digest(protocol), source_seals=seals, source_verification=verified,
            P1_admission=False, actual_encoder_samples=0))
        return
    import torch
    host = base.require_cuda_allocation('cuda')
    device = torch.device('cuda:0')
    model = base.load_compatible_pretrained(policy, device=device).eval().requires_grad_(False)
    before = base.p0._parameter_hash(model)
    buffers = {k: v.clone() for k,v in model.named_buffers()}
    original_predict = model.predict
    def forbid(*args, **kwargs):
        raise RuntimeError('simulator goal catalog permits encode only')
    model.predict = forbid
    catalogs, calls, samples = {}, 0, 0
    transform = base.p0.image_transform()
    try:
        for row in prepared:
            eligible = [e for e in row['entries'] if e['valid']]
            vectors = []
            for start in range(0,len(eligible),16):
                group = eligible[start:start+16]
                images = [np.asarray(Image.open(e['rgb_path']).convert('RGB')) for e in group]
                tensor = torch.cat([base.p0._image_tensor(im, transform=transform, device=device) for im in images])
                calls += 1
                samples += len(group)  # attempts counted before encode, no refunds
                with torch.no_grad():
                    encoded = model.encode({'pixels': tensor})['emb'][:,0]
                if encoded.device.type != 'cuda' or encoded.shape != (len(group),192) or not torch.isfinite(encoded).all():
                    raise ValueError('finite host CUDA frozen goal encodings required')
                vectors.extend(encoded.cpu().tolist())
            catalog = dict(root_sha256=row['root_sha256'], root_index=row['root_index'],
                           ids=[e['id'] for e in eligible], vectors=vectors)
            catalogs[row['root_sha256']] = catalog
            write_new(output/f'{prefix}_catalog_{row["root_index"]}.json', catalog)
    finally:
        model.predict = original_predict
    # Deserialize future evaluator evidence ONLY after catalogs are saved.
    physical, control = (json.loads(p.read_text()) for p in (probes, calibration))
    for r in (physical, control):
        for s in r['source_seals']:
            verify_seal(output_dir=s['output_dir'], expected_manifest_sha256=s['manifest_sha256'])
    diagnostics = diagnose(catalogs, physical, control)
    clean_catalogs = {}
    for path in clean_meta['catalog_hashes']:
        catalog = json.loads(Path(path).read_text())
        if catalog['root_sha256'] in clean_catalogs:
            raise ValueError('duplicate clean root')
        clean_catalogs[catalog['root_sha256']] = catalog
    common_sim, common_clean, coverage = common_id_catalogs(catalogs, clean_catalogs)
    common = dict(coverage=coverage,
                  simulator_common_ids=diagnose(common_sim, physical, control),
                  clean_common_ids=diagnose(common_clean, physical, control),
                  clean_full_catalog=diagnose(clean_catalogs, physical, control),
                  interpretation='same grid IDs isolates catalog coverage, not proof of physical task fidelity')
    gates = dict(all_four_roots=len(catalogs)==4, all1024_slots_retained=sum(len(r['entries']) for r in prepared)==1024,
        all_eligible_encoded=samples==protocol['expected_encoder_samples'],
        parameters_unchanged=before==base.p0._parameter_hash(model),
        buffers_unchanged=all(torch.equal(buffers[k],v) for k,v in model.named_buffers()),
        frozen_eval=all(not m.training for m in model.modules()) and all(not p.requires_grad for p in model.parameters()),
        inputs_unchanged=all(sha(p)==d for p,d in bindings.items()))
    verified = [verify_seal(output_dir=s['output_dir'], expected_manifest_sha256=s['manifest_sha256']) for s in seals]
    write_new(output/f'{prefix}_report.json', dict(status='diagnostic_completed' if all(gates.values()) else 'integrity_fail',
        protocol_sha256=digest(protocol), source_seals=seals, source_verification=verified, gates=gates,
        host=host, actual_encoder_calls=calls, actual_encoder_samples=samples, **diagnostics,
        common_id_comparison=common,
        P1_admission=False, physical_benefit_verified=False, control_selection_changed=False,
        single_RGB_deployment=False, augmentation_artifact_hypothesis_resolved=False))
    if not all(gates.values()):
        raise RuntimeError('simulator catalog integrity failed; preserve all artifacts')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('compilation','probes','calibration','clean-report','policy','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    run(**vars(parser.parse_args()))
