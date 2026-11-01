"""Main-only reset/render diagnostic. Import/prepare never constructs an env."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil

import numpy as np

from experiments.spectral_reachability import p1_conditional_procedural_source_20261027_v1 as source

NAME = 'p1_source_reset_goal_audit_20261027_v1'
FLOOR = 200 * 1024**2
CAP = 8 * 1024**2
AUTH = 'MAIN_RESET_RENDER_DIAGNOSTIC_ONLY'
require = source.require


def sha(path):
    return source.file_sha(path)


def image_sha(image):
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def material(request, report, request_sha, report_sha):
    """Validate the complete historical census, without refreshing exclusions."""
    request, report = Path(request).resolve(), Path(report).resolve()
    for path, expected in ((request, request_sha), (report, report_sha)):
        source.hash_value(expected)
        require(path.is_file() and not path.is_symlink() and sha(path) == expected, 'external input SHA drift')
    req = json.loads(request.read_text()); saved = json.loads(report.read_text())
    p, e = req['plan'], req['exclusions']
    require(source.digest(p) == req['plan_sha256'] and source.digest(e) == req['exclusion_sha256'], 'request digests')
    require(len(p['slots']) == 2 and p['proposal_limit_per_seed'] == p['reset_attempts_per_proposal'] == 2,
            'fixed two-slot/eight-attempt diagnostic only')
    require(p['rgb_shape'] == [224, 224, 3], 'official RGB required')
    for path, h in p['source_closure'].items():
        require(sha(path) == h, 'sealed source drift: ' + path)
    ids = {r['metadata']['root_sha256']: source.digest(r['metadata']) for r in saved['slots'] if r['status'] == 'completed'}
    source.to_admission_scan(p, e, saved, report.parent, ids)
    require(len(ids) == 2, 'this batch must contain both selected roots')
    events = source.read_journal(report.parent / 'journal.jsonl')
    operations = {}
    for event in events:
        if event['kind'] == 'operation_completed':
            key = (event['candidate_id'], event['proposal_index'], event['attempt_index'], event['operation'])
            require(key not in operations, 'duplicate historical operation')
            operations[key] = event['receipt']
    require(len(operations) == 24, 'all 16 resets and 8 explicit renders required')
    packets = {r['slot']['candidate_id']: source.load_raw_source(report.parent / r['raw_source_file'], p) for r in saved['slots']}
    return req, saved, operations, packets


def prepare(request, report, request_sha, report_sha, output):
    material(request, report, request_sha, report_sha)
    output = Path(output)
    require(shutil.disk_usage(output.parent).free >= FLOOR + CAP, '200MiB plus 8MiB diagnostic reserve')
    result = dict(schema=NAME+'.prepare', request=str(Path(request).resolve()), report=str(Path(report).resolve()),
                  request_sha256=request_sha, report_sha256=report_sha,
                  driver_sha256=sha(__file__), source_sha256=sha(source.__file__),
                  denominator=8, budget=dict(resets=16, explicit_renders=8, set_state_calls=32,
                  internal_goal_renders=16, public_steps=0, neural_calls=0), artifact_cap_bytes=CAP,
                  diagnostic_only=True, P1_admission=False)
    write_json(output, result)
    return result


def write_json(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    require(shutil.disk_usage(Path(path).parent).free >= FLOOR + len(raw), '200MiB write floor')
    with Path(path).open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def snapshot(body):
    state = source.vector(body._get_obs())
    velocity = np.asarray(tuple(body.block.velocity), dtype=np.float64)
    angular = float(body.block.angular_velocity)
    require(velocity.shape == (2,) and np.isfinite(velocity).all() and np.isfinite(angular), 'finite hidden velocities')
    return dict(state7=state.tolist(), block_velocity=velocity.tolist(), block_angular_velocity=angular)


def difference(actual, requested):
    delta = source.vector(actual) - source.vector(requested)
    wrapped = delta.copy(); wrapped[4] = (delta[4]+np.pi) % (2*np.pi)-np.pi
    return dict(exact=np.array_equal(actual, requested), raw_delta=delta.tolist(),
                angle_wrapped_delta=wrapped.tolist())


def execute(prepared, output, factory, *, authorization):
    """factory is called only after all read-only bindings and storage gates.

    Hooks observe calls, not change physics. A mismatch is diagnostic data;
    an exception aborts with the full eight-row denominator retained.
    """
    require(authorization == AUTH and prepared['schema'] == NAME+'.prepare', 'explicit main authorization')
    require(prepared['driver_sha256'] == sha(__file__) and prepared['source_sha256'] == sha(source.__file__), 'driver/source seal drift')
    req, saved, operations, packets = material(prepared['request'], prepared['report'], prepared['request_sha256'], prepared['report_sha256'])
    p = req['plan']; output = Path(output)
    require(shutil.disk_usage(output.parent).free >= FLOOR + CAP, '200MiB plus full diagnostic reserve')
    output.mkdir(exist_ok=False)
    journal = source.DurableJournal(output/'journal.jsonl', CAP)
    rows = [dict(candidate_id=s['candidate_id'], proposal_index=i, attempt_index=j, status='not_started')
            for s in p['slots'] for i in range(2) for j in range(2)]
    result = dict(schema=NAME+'.report', prepare=deepcopy(prepared), status='running', denominator=8, rows=rows,
                  counts={k:dict(attempted=0, completed=0) for k in ('reset','explicit_render','set_state','goal_render')},
                  first_error=None, secondary_errors=[], public_steps=0, neural_calls=0, P1_admission=False)
    env = body = None; originals = {}; active = {}; current = None; in_reset = False

    def retain(exc, phase):
        err = dict(phase=phase, type=type(exc).__name__, message=str(exc))
        if result['first_error'] is None: result['first_error'] = err
        else: result['secondary_errors'].append(err)
        result['status'] = 'failed'

    def emit(kind, **value):
        journal.append(kind, coordinates=deepcopy(active), **value)

    def counted(kind, fn):
        result['counts'][kind]['attempted'] += 1
        emit(kind+'_started')
        value = fn()
        result['counts'][kind]['completed'] += 1
        return value

    def hooked_set(requested):
        entry = dict(requested=source.vector(requested).tolist(), before=snapshot(body), completed=False)
        current.setdefault('set_state', []).append(entry)
        def call():
            emit('set_state_before', receipt=deepcopy(entry))
            try:
                value = originals['_set_state'][1](requested)
            except Exception as exc:
                entry['error'] = dict(type=type(exc).__name__, message=str(exc))
                try:
                    entry['after_failure'] = snapshot(body)
                    emit('set_state_failed', receipt=deepcopy(entry))
                except Exception as secondary:
                    result['secondary_errors'].append(dict(phase='setter_failure_capture', type=type(secondary).__name__, message=str(secondary)))
                raise
            entry.update(after=snapshot(body), completed=True)
            entry['requested_after_difference'] = difference(entry['after']['state7'], entry['requested'])
            emit('set_state_after', receipt=deepcopy(entry))
            return value
        return counted('set_state', call)

    def hooked_render():
        kind = 'goal_render' if in_reset else 'explicit_render'
        entry = dict(kind=kind, before=snapshot(body), completed=False)
        current.setdefault('renders', []).append(entry)
        def call():
            emit('render_before', receipt=deepcopy(entry))
            image = source.rgb(originals['render'][1](), p)
            entry.update(after=snapshot(body), rgb_sha256=image_sha(image), completed=True)
            if in_reset:
                entry['requested_goal'] = source.vector(body.goal_state).tolist()
                entry['goal_actual_difference'] = difference(entry['before']['state7'], entry['requested_goal'])
            emit('render_after', receipt=deepcopy(entry))
            return image
        return counted(kind, call)

    def forbidden(*args, **kwargs):
        raise RuntimeError('public step forbidden in reset-only diagnostic')

    def reset(seed, options=None):
        nonlocal in_reset
        active['reset_phase'] = 'random_reset' if options is None else 'placement_reset'
        in_reset = True
        try:
            value = counted('reset', lambda: env.reset(seed=seed, **({} if options is None else {'options': options})))
            summary = source.reset_summary(value, p)
            emit('reset_returned', receipt=summary, actual=snapshot(body))
            current['reset_receipt'] = summary
            return value
        finally:
            in_reset = False

    try:
        emit('run_started', prepare=prepared)
        env = factory(); body = env.unwrapped
        path = inspect.getsourcefile(type(body))
        require(path is not None and p['source_closure'].get(str(Path(path).resolve())) == p['environment_sha256'], 'actual env source binding')
        for name, hook in (('_set_state', hooked_set), ('render', hooked_render), ('step', forbidden)):
            originals[name] = (name in body.__dict__, getattr(body, name))
            setattr(body, name, hook)
        for row in rows:
            row['status'] = 'running'
            active = {k:row[k] for k in ('candidate_id','proposal_index','attempt_index')}
            cid, i, j = active.values(); slot = next(s for s in p['slots'] if s['candidate_id'] == cid)
            seed = slot['reset_seed_span'][0]+i*2+j
            row['random_reset_seed'] = seed; row['placement_reset_seed'] = p['placement_reset_seed']
            current = row['random_reset'] = {}
            initial = reset(seed)
            goal = source.vector(initial[1]['goal_state'])
            requested = source.sample_state(goal, list(body.block.shapes), slot, i, j, p)
            row['placement_requested_state'] = requested.tolist()
            emit('placement_request', state=requested.tolist(), goal=goal.tolist())
            current = row['placement_reset'] = {}
            placed = reset(p['placement_reset_seed'], dict(state=requested.copy(), goal_state=goal.copy()))
            frame = env.render()
            checks = row['checks'] = {}
            for op in ('random_reset', 'placement_reset'):
                checks[op+'_historical_exact'] = row[op]['reset_receipt'] == operations[cid,i,j,op]
                checks[op+'_two_set_one_goal_render'] = len(row[op].get('set_state', [])) == 2 and sum(r['kind']=='goal_render' for r in row[op].get('renders', [])) == 1
            checks['explicit_rgb_historical_exact'] = image_sha(frame) == operations[cid,i,j,'render']['rgb_sha256']
            actual = source.vector(placed[0]['state'])
            checks['returned_state_render_exact'] = np.array_equal(actual, current['renders'][-1]['before']['state7'])
            checks['render_is_state_preserving'] = all(r['before'] == r['after'] for op in ('random_reset','placement_reset') for r in row[op]['renders'])
            packet = packets[cid]; selected = packet['metadata']['selected_attempt']
            row['selected'] = packet['metadata']['source']['proposal_index'] == i and selected['attempt_index'] == j
            if row['selected']:
                checks.update(selected_state_exact=np.array_equal(actual, packet['state']),
                              selected_goal_exact=np.array_equal(placed[1]['goal_state'], packet['goal']),
                              selected_rgb_exact=np.array_equal(frame, packet['rgb_history'][0]),
                              selected_goal_rgb_exact=np.array_equal(placed[1]['goal'], packet['goal_rgb']),
                              selected_root_exact=source.root_hash(actual, placed[1]['goal_state']) == packet['metadata']['root_sha256'])
            row['checks'] = {k:bool(v) for k,v in checks.items()}
            row['status'] = 'completed'; emit('attempt_finished', receipt=deepcopy(row))
        expected = dict(reset=16, explicit_render=8, set_state=32, goal_render=16)
        require(all(result['counts'][k] == dict(attempted=v, completed=v) for k,v in expected.items()), 'actual hooked call census mismatch')
        result['all_reproduction_checks'] = all(all(r['checks'].values()) for r in rows)
        result['status'] = 'diagnostic_completed'
    except Exception as exc:
        retain(exc, 'execution')
        for row in rows:
            if row['status'] == 'running': row['status'] = 'failed'
        try: emit('execution_failed', error=result['first_error'])
        except Exception as secondary: retain(secondary, 'failure_journal')
    finally:
        if body is not None:
            for name, (owned, original) in originals.items():
                try:
                    if owned: setattr(body, name, original)
                    else: delattr(body, name)
                except Exception as exc: retain(exc, 'restore_'+name)
        if env is not None:
            try: env.close()
            except Exception as exc: retain(exc, 'close')
        final_bindings = [(prepared['request'],prepared['request_sha256']), (prepared['report'],prepared['report_sha256']),
                          (__file__,prepared['driver_sha256']), *p['source_closure'].items()]
        parent = Path(prepared['report']).parent
        if 'journal_sha256' in saved:
            final_bindings.append((parent/'journal.jsonl', saved['journal_sha256']))
        final_bindings.extend((parent/r['raw_source_file'],r['raw_source_sha256']) for r in saved.get('slots',[]) if r['status']=='completed')
        for path, expected in final_bindings:
            try: require(sha(path) == expected, 'final source/input SHA drift: '+str(path))
            except Exception as exc: retain(exc, 'final_binding')
        try: emit('run_finished', status=result['status'], counts=result['counts'], first_error=result['first_error'])
        except Exception as exc: retain(exc, 'final_journal')
        try: journal.close()
        except Exception as exc: retain(exc, 'journal_close')
        try: result['journal_sha256'] = sha(output/'journal.jsonl')
        except Exception as exc: retain(exc, 'journal_sha')
        try:
            require(sum(x.stat().st_size for x in output.iterdir())+len(source.canonical(result).encode())+65536 <= CAP, 'artifact cap')
            write_json(output/'report.json', result)
        except Exception as exc:
            retain(exc, 'report_write')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path); parser.add_argument('--report', type=Path)
    parser.add_argument('--request-sha256'); parser.add_argument('--report-sha256')
    parser.add_argument('--prepare', type=Path); parser.add_argument('--run-main', type=Path)
    parser.add_argument('--prepare-sha256'); parser.add_argument('--output', type=Path)
    parser.add_argument('--authorize-reset-render', action='store_true')
    args = parser.parse_args()
    require(bool(args.prepare) != bool(args.run_main), 'choose prepare or explicit main run')
    if args.prepare:
        value = prepare(args.request, args.report, args.request_sha256, args.report_sha256, args.prepare)
    else:
        require(args.authorize_reset_render and args.output is not None, 'explicit main authorization and new output')
        require(sha(args.run_main) == args.prepare_sha256, 'external prepare SHA')
        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401
            return gym.make('swm/PushT-v1', render_mode='rgb_array', resolution=224)
        value = execute(json.loads(args.run_main.read_text()), args.output, factory, authorization=AUTH)
    print(source.canonical(value))
    if value.get('status') == 'failed': raise SystemExit(2)


if __name__ == '__main__':
    main()
