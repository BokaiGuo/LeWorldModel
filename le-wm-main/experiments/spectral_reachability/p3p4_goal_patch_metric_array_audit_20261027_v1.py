"""Independent saved-array/RGB audit; no planner, model, or simulator imports."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal

NAME = 'p3p4_goal_patch_metric_array_audit_20261027_v1'
COMPILER = 'p3p4_goal_patch_metric_20261027_v1'
ROOT = Path(__file__).resolve().parents[2]
ATOL, RTOL = 1e-10, 1e-8


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    # Matches the upstream packet identity serialization, not numerical helpers.
    return json.dumps(value, sort_keys=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    p = Path(path)
    require(p.is_absolute() and '..' not in p.parts and p.is_file() and
            not any(v.is_symlink() for v in (p, *p.parents)), 'absolute regular non-symlink file required')
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def unique(pairs):
    result = {}
    for k, v in pairs:
        require(k not in result, 'duplicate JSON key')
        result[k] = v
    return result


def read_json(path):
    require(Path(path).stat().st_size <= 64 * 1024**2, 'bounded JSON required')
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda x: require(False, 'nonfinite JSON'))


def write(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def array(value, shape):
    a = np.asarray(value)
    require(a.shape == shape and a.dtype.kind in 'fi' and np.isfinite(a).all(), 'finite aligned real array required')
    return a.astype(np.float64)


def near(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return bool(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all() and
                np.allclose(a, b, rtol=RTOL, atol=ATOL))


def reconstruct(goal_vectors, legacy_vectors, operation=None):
    """Three independent SVDs: legacy span, extended nuisance, object residual."""
    v = np.asarray(goal_vectors)
    require(v.ndim == 2 and v.shape[0] > 6, 'ordered goal catalog required')
    v = array(v, (len(v), 192))
    g = array(legacy_vectors, (5, 192))
    require(np.allclose(v[0], g[0], atol=1e-5, rtol=1e-5), 'original native goal mismatch')
    old_directions = (g[1:] - g[0]).T
    def svd(matrix, role):
        if operation is not None:
            operation(role, False)
        result = np.linalg.svd(matrix, full_matrices=False)
        if operation is not None:
            operation(role, True)
        return result
    old_u, old_s, _ = svd(old_directions, 'legacy')
    old_cutoff = old_s[0] * max(old_directions.shape) * np.finfo(np.float64).eps
    old_rank = int(np.count_nonzero(old_s > old_cutoff))
    old_basis = old_u[:, :old_rank]
    old_q = np.eye(192) - old_basis @ old_basis.T
    nuisance = np.column_stack([*(g[k] - g[0] for k in range(1, 5)),
                                v[0] - v[1], *(v[k] - v[1] for k in range(2, 6))])
    u, ns, _ = svd(nuisance, 'nuisance')
    ncut = max(ns[0] * 1e-6, max(nuisance.shape) * np.finfo(float).eps * np.linalg.norm(nuisance))
    n = u[:, ns > ncut]
    objects = (v[6:] - v[1]).T
    residual = objects - n @ (n.T @ objects)
    t, ts, _ = svd(residual, 'object')
    tcut = max(ts[0] * 1e-6, max(residual.shape) * np.finfo(float).eps * np.linalg.norm(objects))
    task = t[:, ts > tcut]
    require(task.shape[1] > 0, 'zero task rank: no fallback')
    q = task @ task.T
    require(np.linalg.norm(q @ n) < 1e-7 and near(q @ q, q), 'invalid reconstructed projector')
    return dict(old_basis=old_basis, nuisance_basis=n, task_basis=task, old_Q=old_q, new_Q=q,
                old_singular=old_s, nuisance_singular=ns, object_singular=ts,
                old_cutoff=float(old_cutoff), nuisance_cutoff=float(ncut), object_cutoff=float(tcut))


def goal_catalog(image):
    """Only the immutable original goal RGB is accepted; no endpoint argument."""
    a = np.asarray(image)
    require(a.shape == (224, 224, 3) and a.dtype == np.uint8, 'uint8 goal RGB required')
    mask = (a == np.array([119, 136, 153], np.uint8)).all(axis=2)
    agent = (a == np.array([65, 105, 225], np.uint8)).all(axis=2)
    require(mask.any() and agent.any(), 'missing canonical colors')
    ys, xs = np.where(agent)
    x0, x1, y0, y1 = int(xs.min())-2, int(xs.max())+3, int(ys.min())-2, int(ys.max())+3
    require(0 <= x0 < x1 <= 224 and 0 <= y0 < y1 <= 224, 'clipped sprite')
    sprite = a[y0:y1, x0:x1]
    require(not mask[y0:y1, x0:x1].any() and
            all(np.all(edge == 255) for edge in (sprite[0], sprite[-1], sprite[:, 0], sprite[:, -1])),
            'nonisolated sprite')
    clean = a.copy()
    clean[y0:y1, x0:x1] = 255
    h, w = sprite.shape[:2]
    ids = ['original', 'clean']
    for cx, cy in ((28, 28), (196, 28), (28, 196), (196, 196)):
        left, top = cx-w//2, cy-h//2
        require(0 <= left and 0 <= top and left+w <= 224 and top+h <= 224 and
                np.all(clean[top:top+h, left:left+w] == 255), 'fixed sprite corner ineligible')
        ids.append(f'nuisance_{cx}_{cy}')
    cells = []
    for row, col in np.ndindex(16, 16):
        left, top = 14*col, 14*row
        if mask[top:top+14, left:left+14].any():
            ident = f'object_cell_{row}_{col}'
            cells.append(dict(id=ident, row=row, col=col, bounds=[left, top, left+14, top+14]))
            ids.append(ident)
    return dict(agent_bbox=[x0, y0, x1, y1], agent_pixels=int(agent.sum()), object_pixels=int(mask.sum()),
                cells=cells, image_ids=ids, expected_encoder_samples=len(ids),
                goal_rgb_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
                mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest())


def candidate_key(candidate):
    h, i = candidate['horizon'], candidate['index']
    require(type(h) is int and h in (2, 3) and type(i) is int and i >= 0, 'H2/H3 integer candidate identity')
    return h, i


def common_bank(source_rows, packet):
    root = packet['root_sha256']
    rows = [r for r in source_rows if r['root_sha256'] == root]
    require(len(rows) == 2 and {r['mode'] for r in rows} == {'uniform', 'dynamic'}, 'two source scheduler arms required')
    require(packet['input_sha256'] == digest({k: v for k, v in packet.items() if k != 'input_sha256'}), 'packet input identity')
    bank = {}
    for row in rows:
        require(row['status'] == row['search']['status'] == 'completed' and
                row['input_sha256'] == packet['input_sha256'], 'source root/input/completion binding')
        seen = set()
        for c in row['search']['candidates']:
            key = candidate_key(c)
            require(key not in seen and c['completed'] is True, 'duplicate or partial source candidate')
            seen.add(key)
            actions = array(c['actions'], (5*key[0], 2))
            require(np.all(np.abs(actions) <= 1), 'source action bounds')
            context = c['context']
            require(context['layout'] == 'states_with_between_state_actions_v1' and
                    context['context_length'] == 3 and context['latent_dim'] == 192 and context['action_dim'] == 10,
                    'source context layout')
            array(context['latent_history'], (3, 192))
            array(context['action_history'], (2, 10))
            if key in bank:
                require(canonical(bank[key]) == canonical(c), 'cross-arm shared witness mismatch')
            bank[key] = c
    require(bank, 'empty common bank')
    return bank


def audit_selection(selection, bank, packet, reconstruction):
    rows = selection['candidates']
    keys = [candidate_key(c) for c in rows]
    require(len(set(keys)) == len(keys) and set(keys) == set(bank), 'complete common candidate census')
    goal = array(packet['goal'], (192,))
    records, checks = [], {}
    q_old, q_new = reconstruction['old_Q'], reconstruction['new_Q']
    for c, key in zip(rows, keys, strict=True):
        source = bank[key]
        require(canonical(c['actions']) == canonical(source['actions']) and
                canonical(c['context']) == canonical(source['context']), 'root/horizon action/context binding')
        residual = array(source['context']['latent_history'], (3, 192))[-1] - goal
        scores = {arm: float(np.dot(q @ residual, q @ residual) / 192)
                  for arm, q in (('old_Q', q_old), ('goal_patch_Q', q_new))}
        ident = f'H{key[0]}_i{key[1]}'
        for arm, score in scores.items():
            checks[ident+'_'+arm] = near(c[arm+'_cost'], score)
        checks[ident+'_inherited_cost'] = near(source['cost'], scores['old_Q'])
        records.append(dict(horizon=key[0], index=key[1], actions=source['actions'],
                            context_sha256=digest(source['context']), actions_sha256=digest(source['actions']),
                            old_Q_cost=scores['old_Q'], goal_patch_Q_cost=scores['goal_patch_Q']))
    winners = {}
    require(set(selection['winners']) == {'old_Q', 'goal_patch_Q'}, 'two winner slots required')
    indexed = {candidate_key(c): c for c in rows}
    for arm in ('old_Q', 'goal_patch_Q'):
        winner = min(records, key=lambda c: (c[arm+'_cost'], c['horizon'], c['index']))
        key = candidate_key(winner)
        checks[arm+'_argmin_and_full_witness'] = canonical(selection['winners'][arm]) == canonical(indexed[key])
        winners[arm] = winner
    checks['selection_counts'] = (selection['common_candidates'] == len(rows) and
                                 selection['cost_evaluations'] == 2*len(rows) and
                                 selection['key_comparisons'] == 2*(len(rows)-1))
    checks['not_deployable'] = (selection['physical_ready'] is False and selection['new_predict_calls'] == 0 and
                                selection['independent_selected_double_replay_required'] is True)
    return dict(checks=checks, candidates=records, winners=winners, common_candidates=len(rows))


class Evidence:
    def __init__(self):
        self.bindings = {}

    def bind(self, path, expected=None):
        p = Path(path).absolute()
        h = file_sha(p)
        require(expected is None or h == expected, 'input SHA mismatch: '+str(p))
        require(str(p) not in self.bindings or self.bindings[str(p)] == h, 'input changed during load')
        self.bindings[str(p)] = h
        return p

    def json(self, path, expected=None):
        return read_json(self.bind(path, expected))

    def seals(self, seals):
        for seal in seals:
            verify_seal(output_dir=seal['output_dir'], expected_manifest_sha256=seal['manifest_sha256'])
            folder = Path(seal['output_dir'])
            manifest = self.json(folder/'manifest.json', seal['manifest_sha256'])
            self.bind(folder/'args.json', manifest['args_sha256'])
            for entry in manifest['files']:
                self.bind(entry['source'], entry['sha256'])
                self.bind(folder/entry['archive'], entry['sha256'])

    def verify(self):
        require(all(file_sha(p) == h for p, h in self.bindings.items()), 'evidence changed during audit')


def load_inputs(compile_dir, report_sha):
    e = Evidence()
    folder = Path(compile_dir).absolute()
    report = e.json(folder/'report.json', report_sha)
    execution = e.json(folder/'execution_protocol.json')
    prepared = Path(execution['prepared'])
    receipt = e.json(prepared/'prepare.json', execution['prepare_sha256'])
    p = e.json(prepared/'protocol.json', receipt['protocol_sha256'])
    require(p['schema'] == COMPILER+'.protocol' and report['schema'] == COMPILER+'.report' and
            report['preparation_sha256'] == execution['prepare_sha256'], 'compiler preparation identity')
    require(report['status'] == 'model_only_metric_compiled' and report['failure'] is None and
            report['model_unchanged'] is True, 'completed frozen-model compilation required')
    require(all(report[k] is False for k in ('physical_ready', 'root_cause_established', 'efficacy_claim',
                                            'P1_admission', 'P3_admission', 'P4_admission')), 'calibration scope')
    for path, h in {**p['bindings'], **p['sources_sha256']}.items():
        e.bind(path, h)
    e.seals(receipt['source_seals'])
    planner_dir = Path(p['planner_prepared'])
    planner_receipt = e.json(planner_dir/'prepare.json', p['planner_prepare_sha256'])
    envelope = e.json(planner_dir/'protocol.json', planner_receipt['protocol_sha256'])
    body = envelope['protocol']
    require(digest(body) == envelope['protocol_id'] == planner_receipt['protocol_id'] == p['planner_protocol_id'],
            'planner protocol identity')
    require(body['packets'] == p['packets'] and body['model_bindings'] == p['model_bindings'], 'original root contexts/goal/Q')
    e.seals(planner_receipt['source_seals'])
    source_path = str(Path(p['planner_output'])/'report.json')
    require(source_path in p['bindings'], 'source candidate report not bound')
    source = e.json(source_path, p['bindings'][source_path])
    require(source['status'] == 'cuda_completed' and source['failure'] is None and source['model_unchanged'] is True and
            source['protocol_id'] == p['planner_protocol_id'], 'source planner identity/completion')
    legacy_paths = [path for path in p['bindings'] if Path(path).parent.name == 'pusht_d2p_metric_program_preflight4_v1_20261026_cuda']
    require(len(legacy_paths) == 1, 'unique original legacy vector report')
    legacy = e.json(legacy_paths[0], p['bindings'][legacy_paths[0]])
    shards = [e.json(folder/f'root_{i}.json') for i in range(4)]
    require(shards == report['rows'], 'per-root immutable shard/report binding')
    journal_path = e.bind(folder/'compile.events.jsonl', report['compile_journal_sha256'])
    events = [json.loads(line, object_pairs_hook=unique) for line in journal_path.read_text().splitlines()]
    images = {}
    for goal in p['goals']:
        path = goal['goal_path']
        require(path in p['bindings'], 'goal PNG not input-bound')
        with Image.open(e.bind(path, p['bindings'][path])) as im:
            require(im.size == (224, 224) and im.mode == 'RGB', 'native goal PNG layout')
            images[goal['root_sha256']] = np.array(im)
    e.verify()
    return e, report, p, source, legacy, images, events


def root_audit(row, packet, goal, legacy_row, source_rows, image, operation=None):
    require(row['root_index'] == goal['root_index'] == legacy_row['root_index'] and
            row['root_sha256'] == goal['root_sha256'] == packet['root_sha256'] == legacy_row['root_sha256'], 'root/index binding')
    catalog = goal_catalog(image)
    require(catalog == goal['catalog'] == row['catalog'], 'goal-only mask/catalog provenance mismatch')
    legacy = legacy_row['compiled']['goal_vectors']
    require(canonical(legacy) == canonical(goal['legacy_goal_vectors']) and
            np.array_equal(legacy[0], packet['goal']), 'legacy goal provenance/original anchor')
    require(len(row['goal_vectors']) == catalog['expected_encoder_samples'] == row['actual_encoder_samples'], 'goal vector/catalog count')
    reconstruction = reconstruct(row['goal_vectors'], legacy, operation)
    metric = row['metric']
    q, n, t = (reconstruction[k] for k in ('new_Q', 'nuisance_basis', 'task_basis'))
    anchor = float(np.max(np.abs(q @ (np.asarray(row['goal_vectors'][0])-np.asarray(row['goal_vectors'][1])))))
    checks = dict(new_Q=near(metric['projection'], q), old_Q=near(packet['projection'], reconstruction['old_Q']),
                  nuisance_rank=metric['nuisance_rank'] == n.shape[1], task_rank=metric['task_rank'] == t.shape[1],
                  nuisance_singular=near(metric['nuisance_singular_values'], reconstruction['nuisance_singular']),
                  object_singular=near(metric['object_singular_values'], reconstruction['object_singular']),
                  anchor=near(metric['goal_anchor_max_difference'], anchor) and anchor < 1e-7,
                  no_labels=metric['physical_labels_used'] is False and metric['learned_parameters'] == 0,
                  not_physical=row['physical_ready'] is False)
    bank = common_bank(source_rows, packet)
    result = audit_selection(row['selection'], bank, packet, reconstruction)
    checks.update(result.pop('checks'))
    return dict(root_sha256=row['root_sha256'], root_index=row['root_index'], checks=checks,
                all_checks=all(checks.values()), nuisance_rank=n.shape[1], task_rank=t.shape[1],
                old_rank=reconstruction['old_basis'].shape[1], goal_only_catalog=catalog,
                numeric_diagnostics=dict(Q_max_abs_error=float(np.max(np.abs(q-np.asarray(metric['projection'])))),
                                         nuisance_leak_norm=float(np.linalg.norm(q@n)),
                                         idempotence_norm=float(np.linalg.norm(q@q-q))), **result), reconstruction


def source_files():
    return [Path(__file__).resolve(), ROOT/'tests'/f'test_{NAME}.py', ROOT/'innerdocs'/f'{NAME}.md',
            ROOT/'experiments/spectral_reachability/d2p_prerun_source_seal.py']


def prepare(compile_dir, report_sha, output):
    e, *_ = load_inputs(compile_dir, report_sha)
    out = Path(output).absolute()
    out.mkdir(exist_ok=False)
    protocol = dict(schema=NAME+'.protocol', compile_dir=str(Path(compile_dir).absolute()), report_sha256=report_sha,
                    bindings=e.bindings, sources={str(p): file_sha(p) for p in source_files()},
                    atol=ATOL, rtol=RTOL, comparison='projectors not signed SVD columns; stable cost/H/index argmin',
                    scope='saved arrays and original goal RGB only; no fresh replay or physical efficacy')
    write(out/'protocol.json', protocol)
    seal = create_seal(files=source_files(), args_json=canonical(dict(protocol_sha256=file_sha(out/'protocol.json'))),
                       output_dir=out/'source_seal')
    e.verify()
    result = dict(schema=NAME+'.prepare', protocol_sha256=file_sha(out/'protocol.json'), source_seal=seal)
    write(out/'prepare.json', result)
    return result


def run(prepared, prepared_sha, output):
    prepared = Path(prepared).absolute()
    require(file_sha(prepared/'prepare.json') == prepared_sha, 'retained auditor preparation SHA')
    receipt = read_json(prepared/'prepare.json')
    require(file_sha(prepared/'protocol.json') == receipt['protocol_sha256'], 'auditor protocol drift')
    protocol = read_json(prepared/'protocol.json')
    require(protocol['schema'] == NAME+'.protocol' and protocol['atol'] == ATOL and protocol['rtol'] == RTOL, 'auditor contract')
    verify_seal(output_dir=receipt['source_seal']['output_dir'], expected_manifest_sha256=receipt['source_seal']['manifest_sha256'])
    require(protocol['sources'] == {str(p): file_sha(p) for p in source_files()}, 'auditor source drift')
    e, report, p, source, legacy, images, events = load_inputs(protocol['compile_dir'], protocol['report_sha256'])
    require(e.bindings == protocol['bindings'], 'frozen audit input binding drift')
    out = Path(output).absolute()
    out.mkdir(exist_ok=False)
    result = dict(schema=NAME+'.report', status='integrity_fail', failure=None, rows=[], artifacts={},
                  prepared_sha256=prepared_sha, compiler_report_sha256=protocol['report_sha256'],
                  neural_calls=0, physical_steps=0, P1_admission=False, P3_admission=False, P4_admission=False,
                  inherited_neural_cost='80 visual encoder + 80 existing projector forwards; candidate acquisition inherited, not free',
                  attempted_audit_array_svds=0, completed_audit_array_svds=0,
                  limitation='array consistency and goal-only RGB mask provenance, not encoder regeneration, JVP/pose semantics, replay or task benefit')
    try:
        roots = [g['root_sha256'] for g in p['goals']]
        require(len(roots) == len(set(roots)) == 4 and roots == [v['root_sha256'] for v in p['packets']] ==
                [v['root_sha256'] for v in report['rows']], 'strict four-root census/order')
        require([v['root_index'] for v in p['goals']] == list(range(4)), 'fixed root indices')
        require(len(source['rows']) == 8 and {v['root_sha256'] for v in source['rows']} == set(roots), 'source eight-arm census')
        legacy_map = {r['root_sha256']: r for r in legacy['rows']}
        require(len(legacy_map) == len(legacy['rows']) == 4 and set(legacy_map) == set(roots), 'legacy root census')
        with (out/'audit.events.jsonl').open('x') as journal:
            for row, packet, goal in zip(report['rows'], p['packets'], p['goals'], strict=True):
                def operation(role, done):
                    counter = 'completed_audit_array_svds' if done else 'attempted_audit_array_svds'
                    result[counter] += 1
                    journal.write(canonical(dict(kind='audit_array_svd', role=role, completed=done,
                                                 root_index=row['root_index']))+'\n')
                    journal.flush()
                    os.fsync(journal.fileno())
                audited, arrays = root_audit(row, packet, goal, legacy_map[row['root_sha256']], source['rows'], images[row['root_sha256']], operation)
                for name in ('old_basis', 'nuisance_basis', 'task_basis', 'old_Q', 'new_Q'):
                    path = out/f'root_{row["root_index"]}.{name}.npy'
                    with path.open('xb') as stream:
                        np.save(stream, arrays[name], allow_pickle=False)
                        stream.flush()
                        os.fsync(stream.fileno())
                    result['artifacts'][str(path)] = file_sha(path)
                result['rows'].append(audited)
                journal.write(canonical(dict(root_index=row['root_index'], all_checks=audited['all_checks']))+'\n')
                journal.flush()
                os.fsync(journal.fileno())
        # Reconcile saved entry journal, not just scalar declarations.
        expected = p['expected_encoder_samples']
        checks = {}
        for kind, attempted, completed in (('encoder', 'actual_encoder_calls', 'completed_encoder_calls'),
                                            ('existing_projector', 'actual_existing_projector_calls', 'completed_existing_projector_calls')):
            for done in (False, True):
                entries = [v for v in events if v['kind'] == kind and v['completed'] is done]
                checks[kind+str(done)] = [v['index'] for v in entries] == list(range(expected))
            checks[kind+'_count'] = report[attempted] == report[completed] == expected
        checks['metric_svds'] = report['attempted_metric_svds'] == report['completed_metric_svds'] == p['expected_metric_svds'] == 8
        for root in roots:
            for role in ('nuisance', 'object'):
                for done in (False, True):
                    checks[root+'_'+role+'_'+str(done)] = sum(v['kind'] == 'svd' and v.get('root_sha256') == root and
                        v.get('role') == role and v['completed'] is done for v in events) == 1
        checks['neural_entry_totals'] = report['actual_encoder_samples'] == expected and report['actual_neural_module_entries'] == 2*expected
        checks['no_new_prediction'] = all(report[k] == 0 for k in ('new_predict_calls', 'new_jacobians', 'new_vjp_directions', 'forbidden_predict_attempts'))
        e.verify()
        verify_seal(output_dir=receipt['source_seal']['output_dir'], expected_manifest_sha256=receipt['source_seal']['manifest_sha256'])
        result['checks'] = checks
        result['all_checks'] = all(checks.values()) and all(r['all_checks'] for r in result['rows'])
        result['status'] = 'saved_goal_patch_metric_consistent' if result['all_checks'] else 'saved_goal_patch_metric_inconsistent'
    except Exception as exc:
        result['failure'] = dict(type=type(exc).__name__, message=str(exc))
        result['all_checks'] = False
    result['bindings'] = e.bindings
    if (out/'audit.events.jsonl').exists():
        result['artifacts'][str(out/'audit.events.jsonl')] = file_sha(out/'audit.events.jsonl')
    write(out/'report.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'audit'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compile-dir', type=Path)
    parser.add_argument('--report-sha256')
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--prepared-sha256')
    args = parser.parse_args()
    if args.command == 'prepare':
        require(args.compile_dir is not None and args.report_sha256 is not None, 'explicit compile directory/report SHA required')
        result = prepare(args.compile_dir, args.report_sha256, args.output)
    else:
        require(args.prepared is not None and args.prepared_sha256 is not None, 'explicit audit preparation/SHA required')
        result = run(args.prepared, args.prepared_sha256, args.output)
    print(canonical(dict(status=result.get('status', 'prepared'), output=str(args.output), failure=result.get('failure'))))
    if result.get('all_checks') is False:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
