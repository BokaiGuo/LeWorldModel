"""Pure saved, preregistered blind queue. No producer/model/physics imports."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np

STEM = 'moving_budget_scheduler_prospective_queue_20261027_v1'
DESIGN_SHA = '8c6eee820a49dd982649df10ccffbd15412ce0117eff7c7a11553986166704c6'
MODES = ('population', 'signal_separated')
AXES = ('predict', 'action_encoder', 'jacobian', 'vjp_directions', 'svd', 'solve')
CAP = dict(zip(AXES, (8, 8, 2, 384, 2, 0)))
OUTPUT_CAP = 2 * 1024**2
FLOOR = 200 * 1024**2


def require(ok, message):
    if not ok:
        raise ValueError(message)


def keys(value, expected):
    require(isinstance(value, dict) and set(value) == set(expected.split()),
            'strict schema keys: ' + expected)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(value):
    require(isinstance(value, str) and len(value) == 64 and
            all(c in '0123456789abcdef' for c in value), 'SHA256 required')
    return value


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode()


def parse(data):
    def pairs(items):
        result = {}
        for k, v in items:
            require(k not in result, 'duplicate JSON key')
            result[k] = v
        return result
    return json.loads(data, object_pairs_hook=pairs,
                      parse_constant=lambda x: require(False, 'nonfinite JSON'))


def file_loader(root):
    root = Path(root).resolve()

    def load(name):
        require(isinstance(name, str) and name and not Path(name).is_absolute(),
                'relative artifact path required')
        path = (root / name).resolve()
        require(path.is_relative_to(root) and '..' not in Path(name).parts,
                'artifact path escapes input root')
        require(path.stat().st_size <= 2 * 1024**2, 'input file exceeds 2MiB')
        return path.read_bytes()
    return load


def bound(load, ref):
    keys(ref, 'file sha256')
    data = load(ref['file'])
    require(sha(data) == digest(ref['sha256']), 'artifact SHA mismatch')
    return data


def policy(mode):
    require(mode in MODES, 'explicit population or signal_separated mode required')
    return dict(design_sha256=DESIGN_SHA, mode=mode, denominator=8 if mode == 'population' else 16,
                select_cap=8, threshold_low=.44, threshold_high=.56,
                order='manifest_order_no_replacement_no_repairing', blocks=20,
                candidates_per_block=3, root_block_floor=5, root_block_ceiling=15,
                freshness='external_MAIN_attestation_required', outcome_access=False)


def validate_manifest(m):
    keys(m, 'schema mode panels excluded_root_sha256 approval')
    require(m['schema'] == STEM + '.manifest', 'manifest schema')
    p = policy(m['mode'])
    keys(m['approval'], 'authority no_signal_or_outcome_seen')
    require(isinstance(m['approval']['authority'], str) and m['approval']['authority'].strip()
            and m['approval']['no_signal_or_outcome_seen'] is True, 'pre-signal approval missing')
    excluded = m['excluded_root_sha256']
    require(isinstance(excluded, list) and len(excluded) >= 2 and
            len(set(excluded)) == len(excluded), 'explicit prior-root exclusion set required')
    for h in excluded:
        digest(h)
    require(isinstance(m['panels'], list) and len(m['panels']) == p['denominator'],
            'fixed complete panel denominator required')
    seen = {k: set() for k in ('panel_id', 'root_sha256', 'task_id', 'blind_sha256')}
    for panel in m['panels']:
        keys(panel, 'panel_id roots')
        pid = panel['panel_id']
        require(isinstance(pid, str) and pid.strip() and pid not in seen['panel_id'], 'unique panel_id')
        seen['panel_id'].add(pid)
        require(isinstance(panel['roots'], list) and len(panel['roots']) == 2, 'two ordered roots')
        for root in panel['roots']:
            keys(root, 'root_sha256 task_id blind_sha256 assets')
            for k in ('root_sha256', 'task_id', 'blind_sha256'):
                v = root[k]
                require(isinstance(v, str) and v.strip() and v not in seen[k], 'global unique ' + k)
                seen[k].add(v)
            digest(root['root_sha256']); digest(root['blind_sha256'])
            require(root['root_sha256'] not in excluded, 'previous root reuse forbidden')
            keys(root['assets'], 'source encoder anchor model normalizer q candidate_bank_rule')
            for h in root['assets'].values():
                digest(h)
    return p


def freeze(manifest, now_ns=None):
    p = validate_manifest(manifest)
    return dict(schema=STEM + '.freeze', policy=p, manifest=manifest,
                manifest_sha256=sha(encode(manifest)), source_sha256=sha(Path(__file__).read_bytes()),
                frozen_at_ns=time.time_ns() if now_ns is None else now_ns)


def verify_freeze(f):
    keys(f, 'schema policy manifest manifest_sha256 source_sha256 frozen_at_ns')
    require(f['schema'] == STEM + '.freeze', 'freeze schema')
    require(f['policy'] == validate_manifest(f['manifest']), 'frozen policy changed')
    require(f['manifest_sha256'] == sha(encode(f['manifest'])), 'frozen manifest changed')
    require(f['source_sha256'] == sha(Path(__file__).read_bytes()), 'queue source changed since freeze')
    require(type(f['frozen_at_ns']) is int and f['frozen_at_ns'] > 0, 'freeze timestamp')


def allocation(weights):
    require(len(weights) == 2 and all(math.isfinite(x) and x >= 0 for x in weights), 'invalid weights')
    total = sum(weights)
    require(math.isfinite(total), 'weight sum overflow')
    p = weights[0] / total if total else None
    d = math.floor(5 + 10 * p + .5) if p is not None else 10
    s = math.floor(5 + 10 * (1 - p) + .5) if p is not None else 10
    vectors = dict(U=[10, 10], D=[d, 20-d], S=[s, 20-s], N=[10, 10])
    separated = p is not None and (p <= .44 or p >= .56)
    if separated:
        require(len({tuple(vectors[k]) for k in ('U', 'D', 'S')}) == 3, 'integer separation invariant')
    return dict(weights=weights, p=p, eligible=separated, blocks=vectors,
                candidates={k: [3*x for x in v] for k, v in vectors.items()})


def array(load, ref, shape):
    data = bound(load, ref)
    a = np.load(io.BytesIO(data), allow_pickle=False)
    require(isinstance(a, np.ndarray) and a.dtype == np.dtype('float64') and
            a.shape == shape and np.isfinite(a).all(), 'float64 finite exact array shape required')
    return a


def root_signal(load, row, identity):
    keys(row, 'identity nominal goal q jacobian u sigma vt')
    require(row['identity'] == identity, 'ordered stable root/asset identity mismatch')
    nominal = array(load, row['nominal'], (192,))
    goal = array(load, row['goal'], (192,))
    q = array(load, row['q'], (192, 192))
    require(np.array_equal(q, np.eye(192)), 'original identity192 Q required')
    j = array(load, row['jacobian'], (192, 20))
    u = array(load, row['u'], (192, 20))
    s = array(load, row['sigma'], (20,))
    vt = array(load, row['vt'], (20, 20))
    require(np.all(s >= 0) and np.all(s[:-1] >= s[1:]), 'sorted nonnegative singular values')
    # Pure saved algebra, no fresh SVD/solve/AD; fixed verification tolerances.
    require(np.allclose(u.T @ u, np.eye(20), rtol=1e-7, atol=1e-9)
            and np.allclose(vt @ vt.T, np.eye(20), rtol=1e-7, atol=1e-9)
            and np.allclose((u*s) @ vt, q@j, rtol=1e-7, atol=1e-9), 'saved SVD decomposition mismatch')
    residual = q @ (goal - nominal)
    energy = float(residual @ residual)
    active = s > (float(s[0]) * 1e-6)
    basis = u[:, active]
    outside = residual - basis @ (basis.T @ residual)
    chi = min(1., max(0., float(outside @ outside) / energy)) if energy else 0.
    w = (1-chi) * energy
    require(math.isfinite(w), 'nonfinite recomputed signal')
    return dict(w=w, chi=chi, energy=energy, rank=int(active.sum()))


def counters(c, complete):
    keys(c, 'attempted completed')
    for kind in ('attempted', 'completed'):
        keys(c[kind], ' '.join(AXES))
        require(all(type(v) is int and 0 <= v <= CAP[k] for k, v in c[kind].items()), 'counter cap/type')
    require(all(c['completed'][k] <= c['attempted'][k] for k in AXES), 'completed exceeds attempted')
    if complete:
        require(c['completed'] == CAP and c['attempted'] == CAP, 'complete screen resource vector')
    return c


def screen(load, entry, panel, f, freeze_sha):
    keys(entry, 'panel_id freshness screen')
    require(entry['panel_id'] == panel['panel_id'], 'panel identity mismatch')
    cert = parse(bound(load, entry['freshness']))
    keys(cert, 'schema freeze_sha256 panel authority issued_at_ns independent_roots not_used_for_design no_outcome_or_candidate_cost_used')
    require(cert['schema'] == STEM + '.freshness' and cert['freeze_sha256'] == freeze_sha
            and cert['panel'] == panel and isinstance(cert['authority'], str) and cert['authority'].strip(),
            'freshness authority/binding')
    require(all(cert[k] is True for k in ('independent_roots', 'not_used_for_design',
                                        'no_outcome_or_candidate_cost_used')), 'freshness not certified')
    r = parse(bound(load, entry['screen']))
    keys(r, 'schema freeze_sha256 panel status started_at_ns finished_at_ns counters errors roots')
    require(r['schema'] == STEM + '.screen' and r['freeze_sha256'] == freeze_sha and r['panel'] == panel,
            'screen/freeze identity binding')
    ts = [f['frozen_at_ns'], cert['issued_at_ns'], r['started_at_ns'], r['finished_at_ns']]
    require(all(type(t) is int for t in ts) and ts[0] < ts[1] <= ts[2] <= ts[3], 'prospective chronology required')
    require(r['status'] in ('screen_completed', 'screen_failed') and isinstance(r['errors'], list), 'terminal screen required')
    complete = r['status'] == 'screen_completed'
    cost = counters(r['counters'], complete)
    if not complete:
        require(r['errors'] and r['roots'] == [], 'failed screen requires errors; no partial selection')
        return dict(status='screen_failed', counters=cost, errors=r['errors'], eligible=False)
    require(not r['errors'] and len(r['roots']) == 2, 'complete roots/no errors')
    sig = [root_signal(load, row, root) for row, root in zip(r['roots'], panel['roots'])]
    return dict(status='screen_completed', counters=cost, errors=[], signals=sig,
                **allocation([x['w'] for x in sig]))


def assemble(f, freeze_sha, index, load):
    verify_freeze(f)
    digest(freeze_sha)
    require(freeze_sha == sha(encode(f)), 'external freeze SHA (canonical JSON) mismatch')
    keys(index, 'schema freeze_sha256 panels')
    require(index['schema'] == STEM + '.index' and index['freeze_sha256'] == freeze_sha, 'index binding')
    require(isinstance(index['panels'], list), 'index panels list')
    entries = {}
    for e in index['panels']:
        keys(e, 'panel_id freshness screen')
        require(e['panel_id'] not in entries, 'duplicate index panel')
        entries[e['panel_id']] = e
    ids = [p['panel_id'] for p in f['manifest']['panels']]
    require(set(entries) <= set(ids), 'unregistered panel forbidden')
    rows, blocked = [], []
    for panel in f['manifest']['panels']:
        pid = panel['panel_id']
        row = dict(panel_id=pid, panel=panel, input_refs=entries.get(pid))
        try:
            require(pid in entries, 'missing actual signal/freshness receipts')
            row.update(screen(load, entries[pid], panel, f, freeze_sha))
        except (ValueError, KeyError, OSError, TypeError, EOFError) as exc:
            row.update(status='blocked', eligible=False, error=dict(type=type(exc).__name__, message=str(exc)))
            blocked.append(pid)
        rows.append(row)
    population = f['policy']['mode'] == 'population'
    selected = ids if population else [r['panel_id'] for r in rows if r.get('eligible')][:8]
    failed = [r['panel_id'] for r in rows if r['status'] == 'screen_failed']
    ready = not blocked and (not failed if population else True)
    released = selected if ready else []
    total = {kind: {k: sum(r.get('counters', {}).get(kind, {}).get(k, 0) for r in rows)
                    for k in AXES} for kind in ('attempted', 'completed')}
    status = ('queue_blocked' if not ready else 'queue_empty_no_search' if not selected
              else 'saved_queue_verified_engineering_only')
    return dict(schema=STEM + '.report', status=status, freeze_sha256=freeze_sha, policy=f['policy'],
                index_canonical_sha256=sha(encode(index)),
                denominator=len(ids), rows=rows, selected_ids=selected, released_ids=released,
                blocked_ids=blocked, failed_ids=failed, screening_cost_lower_bound=total,
                screening_cost_complete=not blocked,
                planned_search_predict_upper_bound=512*len(released),
                planned_full_queue_cap=dict(predict=4160 if population else 4224,
                                            action_encoder=4160 if population else 4224,
                                            jacobian=16 if population else 32,
                                            vjp_directions=3072 if population else 6144,
                                            svd=16 if population else 32, solve=0),
                missing_prerequisites=blocked,
                external_freshness_attestation_not_independently_proven=True,
                actual_neural_calls=0, actual_physics_calls=0,
                P1_admission=False, P3_admission=False, independent_confirmation=False,
                compute_advantage=False)


def publish(output, value):
    data = encode(value)
    require(len(data) <= OUTPUT_CAP, '2MiB output cap')
    output = Path(output)
    require(shutil.disk_usage(output.parent).free >= FLOOR + OUTPUT_CAP, '200MiB disk floor + output reserve')
    # Exclusive single-file publication: no existing artifact can be overwritten.
    with output.open('xb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return sha(data)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('freeze', help='before any new signal; canonical exclusive prereg receipt')
    p.add_argument('--manifest', required=True); p.add_argument('--manifest-sha', required=True)
    p.add_argument('--output', required=True)
    p = sub.add_parser('consume', help='saved arrays only; blocked is exit 2')
    for arg in ('freeze', 'freeze-sha', 'index', 'index-sha', 'input-root', 'output'):
        p.add_argument('--' + arg, required=True)
    args = parser.parse_args(argv)
    if args.command == 'freeze':
        data = Path(args.manifest).read_bytes()
        require(len(data) <= OUTPUT_CAP and sha(data) == digest(args.manifest_sha), 'manifest bytes/SHA')
        value = freeze(parse(data))
    else:
        data = Path(args.freeze).read_bytes()
        require(len(data) <= OUTPUT_CAP and sha(data) == digest(args.freeze_sha), 'freeze file SHA')
        f = parse(data)
        index_data = Path(args.index).read_bytes()
        require(len(index_data) <= OUTPUT_CAP and sha(index_data) == digest(args.index_sha), 'index file SHA')
        value = assemble(f, args.freeze_sha, parse(index_data), file_loader(args.input_root))
    h = publish(args.output, value)
    print(json.dumps(dict(output=args.output, sha256=h, status=value.get('status', 'queue_frozen_not_run'))))
    return 2 if value.get('status') == 'queue_blocked' else 0


if __name__ == '__main__':
    raise SystemExit(main())
