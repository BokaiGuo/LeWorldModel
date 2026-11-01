"""Fixed real suffix-response diagnostic, NOT a deployed policy or P1 pass.

Only model-qualified sealed v3 roots; no physical outcome based selection.
Simulator and CUDA dependencies are imported at runtime, never by algebra tests.
"""
import argparse
from copy import deepcopy
import inspect
import json
import os
from pathlib import Path
import shutil

import numpy as np

from experiments.spectral_reachability import pusht_geometry_prefix_20261027_v3 as planning
from experiments.spectral_reachability import p3_matched_physical_20261027_v1 as durable
from experiments.spectral_reachability.d2p_composition_witness_v2 import digest as content_digest

TAG = 'pusht_geometry_capability_20261027_v1'
ROLES = ('selected', 'greedy', 'zero')
EPSILONS = (.05, .1)
MAX_BYTES = 1024**3
RESERVE_BYTES = 512*1024**2
BRANCH_BYTES = 3*1024**2
OVERHEAD_BYTES = 64*1024**2
write, sha = durable.write, durable.digest
finite = planning.packet.finite


def suffix_bank():
    bank = [dict(probe_id='zero', epsilon=None, axis=None, sign=0, logits=[0.]*10)]
    for grid, epsilon in enumerate(EPSILONS):
        for axis in range(10):
            for sign in (-1, 1):
                logits = np.zeros(10); logits[axis] = sign*epsilon
                bank.append(dict(probe_id=f'e{grid}_a{axis}_{"minus" if sign < 0 else "plus"}',
                    epsilon=epsilon, axis=axis, sign=sign, logits=logits.tolist()))
    return bank


def plans_from_saved(report, source):
    """Use the stable v3 mathematical verifier BEFORE model-only qualification."""
    if (report.get('schema') != planning.TAG or report.get('status') != 'plans_sealed'
            or not report.get('gates') or not all(v is True for v in report['gates'].values())):
        raise ValueError('fully plans_sealed v3 with nonempty passing gates required')
    planning.verify_plans(report, source)
    roots = source['protocol']['roots']
    if len(roots) != 4 or len(set(roots)) != 4 or report['protocol']['roots'] != roots:
        raise ValueError('original four-root identities/order required')
    index = {r['root_sha256']: r for r in report['plans']}
    plans = []
    for i, root in enumerate(roots):
        row = index[root]
        if type(row['root_index']) is not int or row['root_index'] != i:
            raise ValueError('root index differs')
        arm = next(a for a in row['arms'] if a['arm'] == 'geometry')
        if arm['selected_candidate_id'] == arm['greedy_candidate_id']:
            continue
        candidates = {c['candidate_id']: c for c in arm['candidates']}
        role_ids = dict(selected=arm['selected_candidate_id'], greedy=arm['greedy_candidate_id'], zero='prefix_0')
        if np.any(finite(candidates['prefix_0']['prefix_params'], (10,))):
            raise ValueError('prefix_0 must be the zero prefix')
        for role in ROLES:  # aliases are independent labelled branches, never deduplicated
            c = candidates[role_ids[role]]
            params = finite(c['prefix_params'], (10,))
            saved_commands = finite(c['commands'], (10, 2))
            prefix = saved_commands[:5]
            if (np.linalg.norm(params) > planning.packet.CAP+1e-6
                    or np.any(np.abs(prefix) > 1)
                    or not np.allclose(prefix, np.tanh(params).reshape(5, 2), rtol=0, atol=1e-7)):
                raise ValueError('bounded saved tanh prefix required')
            for probe in suffix_bank():
                commands = np.concatenate((prefix, np.tanh(probe['logits']).reshape(5, 2)))
                plan = dict(root_index=i, root_sha256=root, role=role, candidate_id=role_ids[role],
                    role_candidate_ids=role_ids, aliases=[r for r in ROLES if role_ids[r] == role_ids[role]],
                    probe=probe, method=f'{role}_{probe["probe_id"]}', commands=commands.tolist(),
                    prefix_logits=params.tolist(), planned_horizon_tokens=2, planned_commands=10,
                    corrected_search_horizon=2,
                    saved_candidate=dict(predicted_prefix=finite(c['predicted_prefix'], (192,)).tolist(),
                        predicted_terminal=finite(c['nominal_terminal'], (192,)).tolist()))
                plan['selection_sha256'] = content_digest(plan)
                plans.append(plan)
    census(plans)
    return plans


def census(plans, rows=None):
    roots = list(dict.fromkeys(p['root_sha256'] for p in plans))
    expected = {(r, f'{role}_{b["probe_id"]}') for r in roots for role in ROLES for b in suffix_bank()}
    key = lambda p: (p['root_sha256'], p['method'])
    if len(roots) > 4 or len(plans) != len(expected) or {key(p) for p in plans} != expected:
        raise ValueError('complete 3-role x 41-probe census required')
    for p in plans:
        if (p['selection_sha256'] != content_digest({k:v for k,v in p.items() if k != 'selection_sha256'})
                or p['planned_commands'] != 10 or p['planned_horizon_tokens'] != 2):
            raise ValueError('sealed diagnostic plan changed')
    if rows is not None:
        if len(rows) != len(plans) or {key(r) for r in rows} != expected:
            raise ValueError('all diagnostic branches must precede CUDA')
        planmap = {key(p): p for p in plans}
        for r in rows:
            p = planmap[key(r)]
            if (any(r[k] != p[k] for k in ('root_index','selection_sha256','planned_commands','planned_horizon_tokens'))
                    or r['repetition'] != 0):
                raise ValueError('physical branch binding differs')
    return dict(qualified_roots=roots, branches=len(plans), planned_steps=10*len(plans),
        role_aliases_preserved=True, complete=True)


def disk_guard(output, branches, *, free_bytes=None, consumed_bytes=0):
    """Fixed conservative estimate, no override, no deletion, before env creation."""
    estimate = OVERHEAD_BYTES+BRANCH_BYTES*branches+consumed_bytes
    probe = Path(output).resolve()
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free if free_bytes is None else free_bytes
    if estimate > MAX_BYTES or free-(estimate-consumed_bytes) < RESERVE_BYTES:
        raise ValueError(f'disk gate: planned={estimate} free={free}; limit={MAX_BYTES}, reserve={RESERVE_BYTES}; no deletion')
    return dict(planned_bytes=estimate, free_bytes=free, limit_bytes=MAX_BYTES, reserve_bytes=RESERVE_BYTES,
        bytes_per_branch=BRANCH_BYTES, overhead_bytes=OVERHEAD_BYTES)


def sync_directory(output):
    fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def prepare(planning_report):
    """Runtime import of original-root metadata API; never read outcome reports."""
    path = Path(planning_report).resolve()
    report = json.loads(path.read_text())
    if report.get('schema') != planning.TAG or report.get('status') != 'plans_sealed':
        raise ValueError('wait for fully sealed v3 report; failed v2 forbidden')
    from experiments.spectral_reachability import pusht_geometry_physical_20261027_v2 as consumer
    context = consumer.prepare(path)
    base, checked, source, config, bindings, _, upstream, refs, protocol = context
    plans = plans_from_saved(report, source)
    return base, checked, source, config, bindings, plans, upstream, refs, protocol


def phi(state):
    state = finite(state)
    if state.ndim != 1 or len(state) < 5:
        raise ValueError('state indices 2:5 required')
    x, y, angle = state[2:5]
    return np.array([x, y, 20*np.cos(angle), 20*np.sin(angle)])/512


def endpoint_slots(plans, rows, output, helper):
    """Persist every exact-step slot BEFORE CUDA, including absent fixed-H slots."""
    census(plans, rows)
    physical = json.loads((output/'physical_report.json').read_text())
    if physical['rows'] != rows:
        raise ValueError('physical report differs')
    slots = []
    for row in rows:
        path = output/f'root_{row["root_index"]}_{row["method"]}_replay0.physical.json'
        if json.loads(path.read_text()) != row:
            raise ValueError('durable branch differs')
        for kind, step in (('prefix', 5), ('terminal', 10)):
            slot = dict(root_sha256=row['root_sha256'], method=row['method'], kind=kind,
                primitive_step=step, status='missing', state=None, phi=None, rgb_path=None)
            try:
                image = helper.image_receipt(row, kind)
                events = [json.loads(s) for s in Path(row['journal_path']).read_text().splitlines()]
                event = next(e for e in events if e['kind'] == 'observation' and e['primitive_step'] == step)
                slot.update(status='observed', state=event['state'], phi=phi(event['state']).tolist(),
                    rgb_path=str(image), rgb_sha256=sha(image), journal_sha256=row['journal_sha256'])
            except (Exception, KeyboardInterrupt) as exc:
                slot['failure'] = durable.failure(exc, 'exact_endpoint')
            slots.append(slot)
    write(output/'endpoint_receipts.json', slots)
    sync_directory(output)
    return slots


def response_analysis(plans, slots, encodings, metrics, goals):
    """Central FD in original latent/Q and fixed evaluator-only physical phi.

    Common residual is Q(goal0 - zero-role zero-suffix terminal), held EXACTLY
    fixed across all roles and epsilons. Fits are diagnostic, never executed.
    """
    slotmap = {(s['root_sha256'], s['method'], s['kind']): s for s in slots}
    encmap = {(e['root_sha256'], e['method'], e['kind']): e for e in encodings}
    def vector(root, method, space):
        slot = slotmap[root, method, 'terminal']
        if slot['status'] != 'observed':
            raise ValueError('missing exact step10 terminal; no early endpoint substitution')
        if space == 'physical_phi':
            return finite(slot['phi'], (4,))
        e = encmap.get((root, method, 'terminal'))
        if e is None or e['status'] != 'completed' or not e['comparable']:
            raise ValueError('exact step10 HOST CUDA encoding unavailable')
        return finite(e['vector'], (192,))
    results = []
    for root in dict.fromkeys(p['root_sha256'] for p in plans):
        goal, q = metrics[root]
        goal, q = finite(goal, (192,)), finite(q, (192, 192))
        for space in ('real_Q', 'physical_phi'):
            projection = q if space == 'real_Q' else np.eye(4)
            for role in ROLES:
                result = dict(root_sha256=root, role=role, space=space, status='blocked',
                    surrogate_not_official_success=space == 'physical_phi', grids=[])
                try:
                    target = goal if space == 'real_Q' else phi(goals[root])
                    result['target'] = target.tolist()
                    base = vector(root, f'{role}_zero', space)
                    common_base = vector(root, 'zero_zero', space)
                    own_raw, common_raw = target-base, target-common_base
                    own, common = projection@own_raw, projection@common_raw
                    result.update(base_terminal=base.tolist(), zero_role_terminal=common_base.tolist(),
                        own_raw_residual=own_raw.tolist(), common_raw_residual=common_raw.tolist(),
                        own_residual=own.tolist(), common_residual=common.tolist())
                    for grid, epsilon in enumerate(EPSILONS):
                        g = dict(epsilon=epsilon, status='blocked', paired_logs=[])
                        result['grids'].append(g)
                        try:
                            columns = []
                            for axis in range(10):
                                methods = [f'{role}_e{grid}_a{axis}_{sign}' for sign in ('minus','plus')]
                                minus, plus = [vector(root, m, space) for m in methods]
                                columns.append((plus-minus)/(2*epsilon))
                                g['paired_logs'].append(dict(axis=axis, minus_method=methods[0], plus_method=methods[1],
                                    minus=minus.tolist(), plus=plus.tolist(), denominator=2*epsilon))
                            raw = np.stack(columns, axis=1); k = projection@raw
                            g.update(status='completed', raw_jacobian=raw.tolist(), response=k.tolist(),
                                singular_values=np.linalg.svd(k, compute_uv=False).tolist(),
                                own_fit=planning.packet.bounded_ridge(k, own),
                                common_zero_role_fit=planning.packet.bounded_ridge(k, common))
                        except (Exception, KeyboardInterrupt) as exc:
                            g['failure'] = durable.failure(exc, 'paired_real_fd')
                    if all(g['status'] == 'completed' for g in result['grids']):
                        first, second = result['grids']
                        a, b = map(lambda g: np.asarray(g['response']), (first, second))
                        result.update(status='completed', epsilon_sensitivity=dict(
                            response_frobenius_delta=float(np.linalg.norm(b-a)),
                            relative_to_eps005=float(np.linalg.norm(b-a)/max(np.linalg.norm(a), 1e-12)),
                            own_fit_cost_delta=second['own_fit']['own_reach_value']-first['own_fit']['own_reach_value'],
                            common_fit_cost_delta=second['common_zero_role_fit']['own_reach_value']-first['common_zero_role_fit']['own_reach_value']))
                except (Exception, KeyboardInterrupt) as exc:
                    result['failure'] = durable.failure(exc, 'real_response')
                results.append(result)
    comparisons = []
    idx = {(r['root_sha256'], r['space'], r['role']): r for r in results}
    for root in dict.fromkeys(p['root_sha256'] for p in plans):
        for space in ('real_Q','physical_phi'):
            for baseline in ('greedy','zero'):
                selected, other = [idx[root, space, role] for role in ('selected', baseline)]
                c = dict(root_sha256=root, space=space, contrast='selected_minus_'+baseline, status='blocked', grids=[])
                if selected['status'] == other['status'] == 'completed':
                    c['status'] = 'completed'
                    for a, b in zip(selected['grids'], other['grids'], strict=True):
                        c['grids'].append(dict(epsilon=a['epsilon'], **{name+'_cost_delta':
                            a[name]['own_reach_value']-b[name]['own_reach_value']
                            for name in ('own_fit','common_zero_role_fit')}))
                comparisons.append(c)
    return dict(responses=results, selected_vs_baselines=comparisons,
        interpretation='C2 mechanism gap only; rank is not P1 pass; no oracle reselection/admission/significance')


def run(*, planning_report, output, encode_terminals=False, preflight_only=False):
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal
    base, checked, source, config, bindings, plans, upstream, refs, protocol = prepare(planning_report)
    output = Path(output).resolve()
    if not output.name.startswith(TAG):
        raise ValueError('exclusive diagnostic output basename required')
    if output.exists():
        raise FileExistsError(output)
    disk = disk_guard(output, len(plans))
    counts = census(plans)
    if preflight_only:
        print(json.dumps(dict(status='preflight_only', census=counts, disk=disk, new_predict_calls=0, P1_admission=False)))
        return
    from experiments.spectral_reachability import pusht_full_packet_physical_20261027 as helper
    from experiments.spectral_reachability.run_pusht_sampling_comparison_20261027_v2 import verify_rgb_context, official_checked
    output.mkdir(exist_ok=False)
    frozen = dict(schema=TAG, planning_report=str(Path(planning_report).resolve()), bindings=bindings,
        plans=plans, census=counts, disk=disk, epsilons=list(EPSILONS), damping=.002,
        cap=planning.packet.CAP, suffix_base='zero logits, NOT optimized model tail',
        physical_before_CUDA=True, encode_terminals=encode_terminals, P1_admission=False,
        scope='known-root C2 mechanism diagnostic; not deployed policy, rank admission, efficacy or significance')
    write(output/'protocol.json', frozen); protocol_hash = sha(output/'protocol.json')
    rows, failures, seals, encodings = [], [], [], []
    env = model = before = host = None
    def unchanged():
        if any(sha(p) != h for p, h in bindings.items()) or sha(output/'protocol.json') != protocol_hash:
            raise ValueError('bound inputs/protocol changed')
        if not seals:
            raise ValueError('diagnostic source seals required')
        return [verify_seal(output_dir=s['output_dir'], expected_manifest_sha256=s['manifest_sha256']) for s in seals]
    try:
        if plans:
            env = base.gym.make('swm/PushT-v1', render_mode='rgb_array')
        files = checked.sources_before_allocate(source, config)+[
            Path(__file__), Path(planning.__file__), Path(planning.packet.__file__), Path(helper.__file__),
            Path(durable.__file__), Path(inspect.getfile(durable.RgbObservabilityGuard)),
            Path(verify_rgb_context.__code__.co_filename), Path(verify_seal.__code__.co_filename),
            Path(__file__).resolve().parents[2]/f'tests/test_{TAG}.py',
            Path(__file__).resolve().parents[2]/f'docs/{TAG}.md']
        from experiments.spectral_reachability import pusht_geometry_physical_20261027_v2 as consumer
        files.append(Path(consumer.__file__))
        if env is not None:
            files.append(Path(env.unwrapped.eval_state.__code__.co_filename))
        planning_snapshot = json.loads(Path(planning_report).read_text())
        for s in planning_snapshot['source_seals']:
            files += [Path(e['source']) for e in json.loads((Path(s['output_dir'])/'manifest.json').read_text())['files']]
        files = sorted(set(p.resolve() for p in files))
        seals = [create_seal(files=files[i:i+40], args_json=json.dumps(dict(protocol_sha256=protocol_hash)),
            output_dir=output/f'source_seal_{i//40}') for i in range(0, len(files), 40)]
        write(output/'source_seals.json', seals)
        sync_directory(output)
        unchanged()
        current = prepare(planning_report)
        if current[4] != bindings or current[5] != plans:
            raise ValueError('model-only snapshot changed before physics')
        if plans:
            if tuple(env.unwrapped.variation_space['agent']['color'].value.tolist()) != (65,105,225):
                raise ValueError('unsupported renderer')
            roots = base.collect_nonterminal_boundary_contact_roots(env, count=4,
                seed=int(upstream['protocol']['root_seed']), oversample_factor=4)
            roots = {base.root_fingerprint(r): r for r in roots}
            if len(roots) != 4 or set(roots) != set(refs):
                raise ValueError('original roots differ; no replacement')
        for n, plan in enumerate(plans):
            used = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
            disk_guard(output, len(plans)-n, consumed_bytes=used)
            root = roots[plan['root_sha256']]
            def reset():
                if base.root_fingerprint(root) != plan['root_sha256']:
                    raise ValueError('root changed')
                frames, history, goal = base.physical._reset_and_warmup(env, root, history_len=3)
                identity = verify_rgb_context(frames, goal, refs[plan['root_sha256']])
                identity.update(history=np.asarray(history).tolist(), state=base.p0.packets.state_of(env).tolist(),
                    goal=np.asarray(root['goal']).tolist())
                return identity
            def evaluate(state):
                return dict(official=official_checked(env, state, root['goal']), pose=base.physical._pose_label(state, root['goal']))
            rows.append(durable.physical_branch(plan=plan, repetition=0, output=output, reset=reset,
                env=env, state_of=base.p0.packets.state_of, evaluate=evaluate))
            sync_directory(output)
            print(f'capability {n+1}/{len(plans)} {plan["method"]}: {rows[-1]["status"]}', flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        failures.append(durable.failure(exc, 'physical_setup_or_execution'))
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                failures.append(durable.failure(exc, 'env_close'))
    helper.materialize_missing(plans, rows, output)
    census(plans, rows)
    write(output/'physical_report.json', dict(rows=rows, failures=deepcopy(failures), P1_admission=False, new_predict_calls=0))
    slots = endpoint_slots(plans, rows, output, helper)
    metrics = {r['root_sha256']: (r['compiled']['goal_vectors'][0], r['compiled']['scoring']['metric']['projection']) for r in source['rows']}
    goals = {}
    for r in rows:
        goal = r.get('context_identity', {}).get('goal')
        if goal is not None:
            key = r['root_sha256']
            if key in goals and goals[key] != goal:
                failures.append(dict(phase='root_goal_binding', message='root goal changed between branches'))
            goals[key] = goal
    if encode_terminals and plans:
        def setup():
            nonlocal model, before, host
            unchanged()
            # Verify all durable branches and both endpoint slots before allocation.
            census(plans, rows)
            if json.loads((output/'endpoint_receipts.json').read_text()) != slots:
                raise ValueError('endpoint receipts changed')
            if json.loads((output/'physical_report.json').read_text())['rows'] != rows:
                raise ValueError('physical census changed')
            for r in rows:
                p = output/f'root_{r["root_index"]}_{r["method"]}_replay0.physical.json'
                if json.loads(p.read_text()) != r:
                    raise ValueError('physical branch changed')
            import torch
            from PIL import Image
            host = base.require_cuda_allocation('cuda'); device = torch.device('cuda:0')
            model = base.load_compatible_pretrained(Path(protocol['policy']), device=device).eval().requires_grad_(False)
            before = checked.tensor_state_hash(model)
            def forbid(*args, **kwargs):
                raise RuntimeError('diagnostic forbids model.predict')
            model.predict = forbid
            transform = base.p0.image_transform()
            def encode(path):
                with Image.open(path) as im:
                    rgb = np.asarray(im.convert('RGB')).copy()
                with torch.no_grad():
                    v = model.encode({'pixels': base.p0._image_tensor(rgb, transform=transform, device=device)})['emb'][0,0]
                if v.device.type != 'cuda' or not torch.isfinite(v).all():
                    raise ValueError('finite HOST CUDA vector required; no CPU fallback')
                torch.cuda.synchronize(device)
                return v.cpu().tolist()
            return encode
        encodings = helper.encode_observations(rows, plans, output, setup,
            {r:(np.asarray(g), np.asarray(q)) for r,(g,q) in metrics.items()})
    # Goal is bound to reset metadata, not selected from outcomes; absent roots stay blocked.
    analyses = response_analysis(plans, slots, encodings, metrics, goals)
    checks = {}
    try:
        checks['sources_and_inputs_unchanged'] = bool(unchanged())
    except Exception as exc:
        checks['sources_and_inputs_unchanged'] = False
        failures.append(durable.failure(exc, 'final_source_check'))
    actionmap = {(p['root_sha256'], p['method']):p['commands'] for p in plans}
    checks.update(census_complete=census(plans, rows)['complete'], trace_matches_commands=all(
        len(r['trace']) == r['completed_commands'] and all(e['primitive_step'] == i+1 and
        e['command'] == actionmap[r['root_sha256'], r['method']][i] for i,e in enumerate(r['trace'])) for r in rows),
        no_execution_errors=all(r['status'] in ('completed','observation_unavailable','environment_terminated') for r in rows))
    if model is not None:
        checks.update(model_unchanged=before == checked.tensor_state_hash(model),
            frozen_eval=all(not m.training for m in model.modules()) and all(not p.requires_grad for p in model.parameters()))
    endpoints = [dict(root_sha256=p['root_sha256'], role=p['role'], candidate_id=p['candidate_id'], aliases=p['aliases'],
        probe=p['probe'], method=p['method'], status=r['status'], full_horizon_completed=r['full_horizon_completed'],
        actual_endpoint_steps=r['completed_commands'], official=r.get('official'), pose=r.get('pose'), final_state=r.get('final_state'))
        for p in plans for r in rows if (p['root_sha256'], p['method']) == (r['root_sha256'], r['method'])]
    completeness = dict(expected_response_groups=6*len(counts['qualified_roots']),
        completed_response_groups=sum(r['status'] == 'completed' for r in analyses['responses']),
        observed_prefix_slots=sum(s['status'] == 'observed' for s in slots if s['kind'] == 'prefix'),
        observed_terminal_slots=sum(s['status'] == 'observed' for s in slots if s['kind'] == 'terminal'),
        completed_encoding_samples=sum(e['status'] == 'completed' for e in encodings))
    status = 'no_model_qualified_roots' if not plans else ('diagnostic_recorded' if
        completeness['completed_response_groups'] == completeness['expected_response_groups'] else 'partial_diagnostic_recorded')
    if failures or not all(checks.values()):
        status = 'integrity_fail'
    write(output/'report.json', dict(schema=TAG, status=status, measurement_completeness=completeness,
        protocol_sha256=protocol_hash, census=counts, gates=checks, failures=failures, source_seals=seals,
        rows=rows, endpoint_receipts=slots, actual_official_pose_endpoints=endpoints, encodings=encodings, analysis=analyses, host=host,
        attempted_encode_samples=sum(e['attempted_encode_samples'] for e in encodings),
        physical_env_step_attempts=sum(r['physical_env_step_attempts'] for r in rows),
        physical_env_step_returns=sum(r['physical_env_step_returns'] for r in rows),
        new_predict_calls=0, P1_admission=False, efficacy_verified=False, significance_claim=False,
        physical_phi_is_surrogate_not_official_success=True))
    if failures or not all(checks.values()):
        raise RuntimeError('diagnostic integrity failed; all planned slots retained')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--planning-report', type=Path, default=Path('results/pusht_geometry_prefix_20261027_v3_cuda_v1/report.json'))
    parser.add_argument('--output', type=Path, default=Path('results/'+TAG+'_cuda_v1'))
    parser.add_argument('--encode-terminals', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    run(**vars(parser.parse_args()))
