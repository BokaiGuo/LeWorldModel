"""Saved-only finite probe arithmetic. No physics, neural runtime or fitted policy."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

NAME = 'moving_overlay_capability_probe_analysis_20261027_v1'
PRODUCER = 'moving_overlay_capability_probe_physical_20261027_v1'
ARMS = ('line_search_GN', 'full10_conditional_D2P')
CONDITIONS = ('off', 'on')
PROBES = ('zero', '+x', '-x', '+y', '-y')
RESOURCES = ('constructor', 'reset', 'public_step', 'physics_step', 'set_state',
             'goal_setter', 'goal_render', 'goal_bind', 'explicit_render', 'close')
UNITS = dict(position='px', angle='rad', action='raw_command', time='public_step')
MAX_JSON = 64 * 1024**2


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            value.update(chunk)
    return value.hexdigest()


def hash_value(value):
    require(type(value) is str and len(value) == 64 and
            all(c in '0123456789abcdef' for c in value), 'literal SHA256 required')


def integer(value, minimum=0):
    require(type(value) is int and value >= minimum, 'literal bounded count required')
    return value


def vector(value, shape):
    # No booleans, string-to-float coercion or missing-value imputation.
    def numbers(item):
        if type(item) is list:
            return all(numbers(x) for x in item)
        return type(item) in (int, float) and math.isfinite(item)
    require(numbers(value), 'literal finite numeric array required')
    a = np.asarray(value, dtype=np.float64)
    require(a.shape == shape and np.isfinite(a).all(), 'finite exact array shape required')
    return a


def commands(value, length):
    a = vector(value, (length, 2)) if length else np.empty((0, 2), dtype=np.float64)
    require(type(value) is list and len(value) == length, 'literal commands list')
    b = a.astype('<f4')
    require(np.isfinite(b).all() and np.array_equal(a, b.astype(np.float64)) and
            np.all(np.abs(a) <= 1), 'lossless bounded raw float32 commands required')
    return b


def exact(a, b):
    return canonical(a) == canonical(b)


def snapshot(state, value):
    s = vector(state, (7,))
    require(type(value) is dict and set(value) == {'state', 'state_float64_bits', 'bodies'},
            'full exported state/body snapshot required')
    require(np.asarray(value['state'], dtype='<f8').tobytes() == s.astype('<f8').tobytes() and
            value['state_float64_bits'] == s.astype('<f8').tobytes().hex(), 'snapshot state7 bits mismatch')
    vector(value['state'], (7,))
    require(set(value['bodies']) == {'agent', 'block'}, 'both exported bodies required')
    bodies = {}
    for name in ('agent', 'block'):
        b = value['bodies'][name]
        require(set(b) == {'position', 'velocity', 'unwrapped_angle', 'angular_velocity',
                           'force', 'torque', 'is_sleeping', 'float64_bits'} and
                type(b['is_sleeping']) is bool, 'complete native body snapshot')
        x = np.concatenate((vector(b['position'], (2,)), vector(b['velocity'], (2,)),
            vector([b['unwrapped_angle'], b['angular_velocity']], (2,)),
            vector(b['force'], (2,)), vector([b['torque']], (1,))))
        require(x.astype('<f8').tobytes().hex() == b['float64_bits'], 'body bits mismatch')
        bodies[name] = x
    require(np.array_equal(s[:2], bodies['agent'][:2]) and
            np.array_equal(s[2:4], bodies['block'][:2]) and
            np.array_equal(s[5:7], bodies['agent'][2:4]) and
            s[4] == bodies['block'][4] % (2 * np.pi), 'state/body identity mismatch')
    return s


def start(value):
    require(type(value) is dict and set(value) == {'state', 'snapshot', 'history_sha256'}, 'complete start identity')
    snapshot(value['state'], value['snapshot'])
    hash_value(value['history_sha256'])


def key(value):
    return value['root_slot'], value['condition'], value['arm']


def case_keys():
    return [(s, c, a) for s in (0, 1) for c in CONDITIONS for a in ARMS]


def protocol_check(p):
    require(type(p) is dict and set(p) == {'roots', 'cases', 'probes', 'prefix_steps',
            'probe_steps', 'warmup_steps', 'meter_per_full_branch'}, 'complete frozen analysis protocol')
    w, prefix, h = (integer(p[n], 1) for n in ('warmup_steps', 'prefix_steps', 'probe_steps'))
    require(w == 10, 'original warmup10 required')
    require(len(p['roots']) == 2, 'two fixed calibration roots')
    for i, root in enumerate(p['roots']):
        require(type(root['root_slot']) is int and root['root_slot'] == i and
                set(root) == {'root_slot', 'task_id', 'source_root_state', 'goal_state'}, 'root identity/order')
        hash_value(root['task_id'])
        vector(root['source_root_state'], (7,)); vector(root['goal_state'], (7,))
    require(p['roots'][0]['task_id'] != p['roots'][1]['task_id'], 'distinct source task identities')
    require(len(p['cases']) == 8 and [key(c) for c in p['cases']] == case_keys(), 'eight fixed case denominator/order')
    for c in p['cases']:
        require(type(c['root_slot']) is int and set(c) == {'root_slot', 'condition', 'arm',
                'task_id', 'selection_digest', 'prefix_commands', 'expected_start'}, 'complete case identity')
        hash_value(c['task_id']); hash_value(c['selection_digest']); commands(c['prefix_commands'], prefix); start(c['expected_start'])
    for s in (0,1):
        require(p['cases'][4*s]['task_id']==p['roots'][s]['task_id'] and
                all(p['cases'][4*s+j]['task_id']==p['cases'][4*s+j+1]['task_id'] for j in (0,2)),
                'same-condition GN/D2P task identity, off reference root task')
    require(len(p['probes']) == 5 and [b['probe_id'] for b in p['probes']] == list(PROBES), 'same fixed five-probe dictionary')
    bank = {}
    for b in p['probes']:
        require(set(b) == {'probe_id', 'commands'}, 'literal probe dictionary')
        bank[b['probe_id']] = commands(b['commands'], h)
    require(not np.any(bank['zero']), 'real zero probe commands required')
    amplitudes = []
    for axis, label in enumerate(('x', 'y')):
        plus, minus = bank['+' + label], bank['-' + label]
        a = float(plus[0, axis])
        require(a > 0 and np.all(plus[:, axis] == a) and not np.any(plus[:, 1-axis]) and
                np.array_equal(minus, -plus), 'fixed axis-only equal-and-opposite raw probes')
        amplitudes.append(a)
    caps = p['meter_per_full_branch']
    require(set(caps) == set(RESOURCES), 'complete meter resources')
    for count in caps.values(): integer(count)
    require(caps['constructor'] == caps['close'] == 1 and caps['reset'] == 2 and
            caps['public_step'] == w + prefix + h and caps['physics_step'] == 4 + 10*(w+prefix+h) and
            caps['set_state'] == 4 and caps['goal_setter'] == 3 and caps['goal_render'] == 2 and
            caps['goal_bind'] == 1, 'original native full-branch meter identities')
    return bank, amplitudes


def ledger(row, caps):
    counts = row['counts']
    require(set(counts) == set(caps), 'all meter fields required, never impute zero')
    actual = {name: dict(attempted=0, completed=0) for name in caps}
    active = {name: False for name in caps}
    for event in row['journal']:
        require(type(event) is dict and set(event) == {'resource', 'kind'} and
                event['resource'] in caps, 'resource event identity')
        resource, kind = event['resource'], event['kind']
        require(kind in ('attempt', 'return', 'failed'), 'resource event kind')
        if kind == 'attempt':
            require(not active[resource], 'overlapping same-resource attempts')
            active[resource] = True; actual[resource]['attempted'] += 1
        else:
            require(active[resource], 'return/failure without actual attempt')
            active[resource] = False
            if kind == 'return': actual[resource]['completed'] += 1
    for name, count in counts.items():
        require(set(count) == {'attempted', 'completed'}, 'attempt/return meter fields')
        a, c = integer(count['attempted']), integer(count['completed'])
        require(c <= a <= caps[name] and count == actual[name], 'journal/count/cap mismatch')
    return active


def validate(manifest):
    require(manifest['schema'] == PRODUCER + '.saved_analysis' and manifest['units'] == UNITS,
            'saved producer schema and native units')
    for field in ('prepare_sha256', 'source_report_sha256', 'planner_report_sha256'): hash_value(manifest[field])
    p = manifest['protocol']; bank, amplitudes = protocol_check(p)
    expected = [(*k, b) for k in case_keys() for b in PROBES]
    rows = manifest['rows']
    require(len(rows) == 40 and [(*key(r), r['probe_id']) for r in rows] == expected,
            'all forty rows in fixed order, including every failure/missing')
    cases = {key(c): c for c in p['cases']}
    roots = p['roots']; probe_hashes = {b['probe_id']: digest(b) for b in p['probes']}
    for r in rows:
        require(type(r['root_slot']) is int, 'literal row slot')
        c = cases[key(r)]
        require(r['selection_digest'] == c['selection_digest'] and
                r['task_id'] == c['task_id'] and
                r['probe_sha256'] == probe_hashes[r['probe_id']], 'case/task/probe identity mismatch')
        require(all(r[n] == manifest[n] for n in ('prepare_sha256', 'source_report_sha256', 'planner_report_sha256')),
                'row upstream source mismatch')
        require(r['status'] in ('completed', 'native_stopped', 'engineering_failed', 'not_started'), 'explicit row status')
        active = ledger(r, p['meter_per_full_branch'])
        warm = integer(r['warmup_returned']); prefix_len = len(r['prefix_commands'])
        require(warm <= p['warmup_steps'] and prefix_len <= p['prefix_steps'], 'fixed warmup/prefix denominator')
        returned_prefix = commands(r['prefix_commands'], prefix_len)
        planned_prefix = commands(c['prefix_commands'], p['prefix_steps'])
        require(returned_prefix.tobytes() == planned_prefix[:prefix_len].tobytes(), 'actual sealed prefix bits')
        require(not prefix_len or warm == p['warmup_steps'], 'prefix before complete warmup')
        n = len(r['returned_commands'])
        require(n <= p['probe_steps'], 'fixed probe horizon')
        returned = commands(r['returned_commands'], n)
        require(returned.tobytes() == bank[r['probe_id']][:n].tobytes(), 'actual probe command bits')
        require(r['counts']['public_step']['completed'] == warm + prefix_len + n, 'actual public-step accounting')
        trace = r['trajectory']
        if not any(active.values()) and all(v['attempted']==v['completed'] for v in r['counts'].values()):
            counts = {k:v['completed'] for k,v in r['counts'].items()}
            require(counts['physics_step'] == 2*counts['reset']+10*counts['public_step'] and
                    counts['set_state'] == 2*counts['reset'] and counts['goal_render'] == counts['reset'] and
                    counts['goal_setter'] == counts['reset']+counts['goal_bind'], 'actual native meter identities')
        if r['start'] is None:
            require(not trace and n == 0, 'no fabricated start/endpoint')
        else:
            start(r['start'])
            require(exact(r['start'], c['expected_start']) and warm == p['warmup_steps'] and
                    prefix_len == p['prefix_steps'], 'same exact rebuilt start/history/snapshot')
            require(len(trace) == n+1, 'every returned probe state, including step0')
            for step, t in enumerate(trace):
                require(type(t['step']) is int and t['step'] == step, 'continuous real trajectory')
                snapshot(t['state'], t['snapshot'])
                require(type(t['terminated']) is bool and type(t['truncated']) is bool and
                        (t['native_success'] is None or type(t['native_success']) is bool), 'native flags retained, absent success stays null')
                require(t['reward'] is None if step == 0 else type(t['reward']) in (int, float) and math.isfinite(t['reward']),
                        'native returned reward, no invented step0 reward')
                if step < n: require(not t['terminated'] and not t['truncated'], 'no step after native stop')
            require(exact(trace[0]['state'], r['start']['state']) and
                    exact(trace[0]['snapshot'], r['start']['snapshot']), 'step0/start exact identity')
        if r['status'] == 'completed':
            require(r['start'] is not None and n == p['probe_steps'] and not any(active.values()) and
                    all(v == dict(attempted=p['meter_per_full_branch'][k], completed=p['meter_per_full_branch'][k])
                        for k, v in r['counts'].items()), 'complete full-horizon branch and actual census')
        if r['status'] == 'native_stopped':
            if trace:
                require(trace[-1]['terminated'] or trace[-1]['truncated'] or trace[-1]['native_success'],
                        'native stop evidence required')
            else:
                stop = r.get('preprobe_stop')
                require(type(stop) is dict and stop['phase'] in ('warmup','prefix'), 'preprobe native stop evidence required')
                require(type(stop['step']) is int and stop['step'] == (warm if stop['phase']=='warmup' else prefix_len),
                        'native preprobe stop step')
                require(stop['phase'] != 'warmup' or prefix_len == 0, 'no prefix after warmup stop')
                snapshot(stop['state'], stop['snapshot'])
                require(all(type(stop[k]) is bool for k in ('native_success','terminated','truncated')) and
                        any(stop[k] for k in ('native_success','terminated','truncated')), 'real preprobe stop flags')
        if r['status'] == 'not_started':
            require(r['start'] is None and not trace and not warm and not prefix_len and
                    all(v == dict(attempted=0, completed=0) for v in r['counts'].values()), 'not-started cannot hide real execution')
    return p, rows, amplitudes


def wrapped(value):
    return float(math.atan2(math.sin(value), math.cos(value)))


def feature(t, space):
    state = np.asarray(t['state'], dtype=np.float64)
    if space == 'object_xy': return state[2:4]
    if space == 'agent_xy': return state[:2]
    if space == 'joint_xy': return state[:4]
    return np.array([t['snapshot']['bodies']['block']['unwrapped_angle']], dtype=np.float64)


def direction(root, space):
    goal, s = (np.asarray(root[n], dtype=np.float64) for n in ('goal_state', 'source_root_state'))
    d = goal[2:4]-s[2:4] if space == 'object_xy' else goal[:2]-s[:2] if space == 'agent_xy' else goal[:4]-s[:4]
    length = float(np.linalg.norm(d))
    return None if length == 0 else d/length


def svd_summary(matrix):
    values = np.linalg.svd(matrix, compute_uv=False)
    singular = bool(values[-1] == 0)
    ratio = None if singular else float(values[0]/values[-1])
    return dict(singular_values=values.tolist(), condition_number=ratio if ratio is None or math.isfinite(ratio) else None,
                condition_status='singular' if singular else 'overflow' if not math.isfinite(ratio) else 'finite',
                condition_definition='2-norm ratio of min(m,n) singular values; no numerical-rank threshold',
                input_dimensions=matrix.shape[1], output_dimensions=matrix.shape[0],
                input_nullspace_dimension_lower_bound=max(0,matrix.shape[1]-matrix.shape[0]))


def analyze(manifest):
    """Pure array arithmetic on validated saved states; rejects contradictions."""
    p, rows, amplitudes = validate(manifest)
    results = []
    for i, c in enumerate(p['cases']):
        group = rows[5*i:5*i+5]; root = p['roots'][c['root_slot']]
        endpoints = {r['probe_id']: r['trajectory'][-1] if r['start'] is not None and
                     len(r['returned_commands']) == p['probe_steps'] and r['status'] in ('completed', 'native_stopped') else None
                     for r in group}
        missing = [b for b in PROBES if endpoints[b] is None]
        result = {k: deepcopy(c[k]) for k in ('root_slot', 'condition', 'arm', 'task_id', 'selection_digest')}
        result.update(status='complete' if not missing else 'missing', missing_probes=missing,
            native=[dict(probe_id=r['probe_id'], status=r['status'],
                native_success=r['trajectory'][-1]['native_success'] if r['trajectory'] else None,
                terminated=r['trajectory'][-1]['terminated'] if r['trajectory'] else None,
                truncated=r['trajectory'][-1]['truncated'] if r['trajectory'] else None,
                reward=r['trajectory'][-1]['reward'] if r['trajectory'] else None,
                preprobe_stop=deepcopy(r.get('preprobe_stop')),
                returned_steps=len(r['returned_commands']), fixed_endpoint_available=endpoints[r['probe_id']] is not None)
                for r in group], spaces={})
        for space in ('object_xy', 'agent_xy', 'joint_xy', 'object_angle_unwrapped', 'object_angle_wrapped_diagnostic'):
            angular = space.startswith('object_angle'); wrap = space.endswith('diagnostic')
            u = None if angular else direction(root, space)
            reason = None if angular or u is not None else 'zero original task direction'
            def difference(a, b):
                d = feature(a, space)-feature(b, space)
                return np.array([wrapped(d[0])]) if wrap else d
            columns = []; asymmetry = []; failures = []
            for axis, a in zip(('x', 'y'), amplitudes, strict=True):
                plus, minus, zero = endpoints['+'+axis], endpoints['-'+axis], endpoints['zero']
                if plus is None or minus is None:
                    columns.append(None); asymmetry.append(None); failures.append('missing signed pair '+axis)
                else:
                    columns.append((difference(plus, minus)/(2*a)).tolist())
                    asymmetry.append(None if zero is None else (difference(plus, zero)+difference(minus, zero)).tolist())
                    failures.append(None)
            matrix = None if any(v is None for v in columns) else np.asarray(columns).T
            finite = []
            for b in PROBES:
                end, zero = endpoints[b], endpoints['zero']
                delta = None if end is None or zero is None else difference(end, zero)
                finite.append(dict(probe_id=b, delta=None if delta is None else delta.tolist(),
                    task_projection=None if delta is None or u is None else float(u@delta),
                    reason='missing fixed endpoint or zero endpoint' if delta is None else reason))
            zero = endpoints['zero']
            drift = None if zero is None else difference(zero, dict(state=c['expected_start']['state'], snapshot=c['expected_start']['snapshot']))
            projected = [None if col is None or u is None else float(u@np.asarray(col)) for col in columns]
            result['spaces'][space] = dict(matrix=None if matrix is None else matrix.tolist(), columns=columns,
                column_order=['raw_x', 'raw_y'], column_missing_reasons=failures,
                units='rad/raw_command' if angular else 'px/raw_command',
                finite_delta_units='rad' if angular else 'px',
                spectrum=None if matrix is None else svd_summary(matrix),
                asymmetry_columns=asymmetry, zero_drift=None if drift is None else drift.tolist(),
                task_direction=None if u is None else u.tolist(), task_direction_reason=reason,
                projected_columns=projected, finite_responses=finite,
                response_norm=None if any(x is None for x in projected) else float(np.linalg.norm(projected)),
                angle_branch_cut_ambiguous=None if not angular else [None if endpoints['+'+a] is None or endpoints['-'+a] is None
                    else bool(abs(float(feature(endpoints['+'+a], space)[0]-feature(endpoints['-'+a], space)[0])) >= math.pi)
                    for a in ('x', 'y')], diagnostic_only=angular)
        results.append(result)
    index = {key(r): r for r in results}; comparisons = []
    pairs = [(dict(root_slot=s, condition=c, contrast='D2P_minus_GN'), (s,c,ARMS[1]), (s,c,ARMS[0]))
             for s in (0,1) for c in CONDITIONS]
    pairs += [(dict(root_slot=s, arm=a, contrast='on_minus_off'), (s,'on',a), (s,'off',a))
              for s in (0,1) for a in ARMS]
    for label, plus, minus in pairs:
        left, right = index[plus], index[minus]; label['spaces'] = {}
        for space in left['spaces']:
            a, b = left['spaces'][space], right['spaces'][space]
            label['spaces'][space] = dict(matrix_difference=None if a['matrix'] is None or b['matrix'] is None else
                (np.asarray(a['matrix'])-np.asarray(b['matrix'])).tolist(),
                finite_task_projection_difference=[dict(probe_id=x['probe_id'], delta=None if
                    x['task_projection'] is None or y['task_projection'] is None else x['task_projection']-y['task_projection'])
                    for x,y in zip(a['finite_responses'], b['finite_responses'], strict=True)])
        label['status'] = 'complete' if left['status'] == right['status'] == 'complete' else 'missing'
        comparisons.append(label)
    output = dict(schema=NAME+'.report', status='analyzed' if all(r['status']=='complete' for r in results) else 'analyzed_with_missing',
        source_report_sha256=manifest['source_report_sha256'], planner_report_sha256=manifest['planner_report_sha256'],
        prepare_sha256=manifest['prepare_sha256'], manifest_digest=digest(manifest),
        denominators=dict(roots=2,cases=8,probes_per_case=5,branches=40,comparisons=8),
        probe_amplitudes_raw=amplitudes, rows=results, comparisons=comparisons,
        P1_admission=False, causal_generalization=False, policy_fit=False,
        limitations=['two known calibration roots, not independent confirmation',
                     'finite-horizon single-scale secants, not infinitesimal derivatives or global controllability',
                     'wrapped angles diagnostic only; native flags unchanged',
                     'exported snapshots do not serialize hidden contact/solver caches'])
    canonical(output)
    return output


def read(path):
    path = Path(path)
    require(path.is_file() and path.stat().st_size <= MAX_JSON, 'bounded saved JSON file')
    def pairs(items):
        result = {}
        for k,v in items:
            require(k not in result, 'duplicate JSON key'); result[k] = v
        return result
    def constant(value): raise ValueError('nonfinite JSON constant: '+value)
    value = json.loads(path.read_bytes(), object_pairs_hook=pairs, parse_constant=constant)
    canonical(value)
    return value


def load_saved(path, expected):
    """Independently bind actual saved report+prepare, without producer imports."""
    hash_value(expected); path = Path(path).resolve()
    require(sha(path) == expected, 'external saved-analysis manifest SHA')
    m = read(path); bindings = {str(path): expected}; loaded = {}
    for name in ('source', 'prepare'):
        ref = m[name]; hash_value(ref['sha256']); target = Path(ref['path'])
        require(target.is_absolute() and target.resolve() != path and str(target.resolve()) not in bindings,
                'distinct absolute report/prepare paths')
        require(sha(target) == ref['sha256'], 'saved '+name+' SHA mismatch')
        loaded[name] = read(target); bindings[str(target.resolve())] = ref['sha256']
    require(m['prepare']['sha256'] == m['prepare_sha256'], 'externally sealed prepare identity')
    for name, artifact in loaded.items():
        require(artifact['schema'] == PRODUCER + ('.report' if name == 'source' else '.prepare'), 'producer artifact schema')
        require(exact(artifact['analysis_protocol'], m['protocol']), 'saved prepare/report protocol mismatch')
        require(all(artifact[k] == m[k] for k in ('source_report_sha256', 'planner_report_sha256')), 'saved upstream mismatch')
    require(loaded['source']['prepare_sha256'] == m['prepare_sha256'] and
            exact(loaded['source']['analysis_rows'], m['rows']), 'actual source rows/prepare mismatch')
    result = analyze(m)
    for filename, h in bindings.items(): require(sha(filename) == h, 'saved input drift')
    result['input_bindings'] = bindings
    return result


def native_digest(value):
    """The physical producer's existing spaced JSON hash convention, not ours."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_journal(path, tail):
    path = Path(path)
    require(path.stat().st_size <= MAX_JSON, 'bounded native journal')
    data = path.read_bytes()
    require(data.endswith(b'\n'), 'complete journal lines')
    events = [json.loads(line) for line in data.splitlines()]
    previous = None
    for i,e in enumerate(events):
        require(type(e['index']) is int and e['index'] == i and e['previous_sha256'] == previous and
                e['event_sha256'] == native_digest({k:v for k,v in e.items() if k!='event_sha256'}), 'native journal chain')
        previous = e['event_sha256']
    require(events and previous == tail and events[-1]['kind'] == 'run_finished', 'native terminal journal tail')
    canonical(events)
    return events


def normalize_native(p, report, events, upstream, prepare_sha):
    """Adapt Nash saved structs only; does not call/import any producer helper."""
    require(p['schema'] == PRODUCER+'.prepare' and report['schema'] == PRODUCER+'.report', 'native producer schemas')
    require(report['prepare_sha256'] == prepare_sha and exact(p['contract'], report['contract']) and
            p['physical_report_sha256'] == report['physical_report_sha256'] and
            p['planner_report_sha256'] == report['planner_report_sha256'], 'native report/prepare join')
    require(report['fixed_denominator'] == 40 and report['independent_roots'] == 2 and
            report['inputs_final_verified'] is True and report['new_neural_calls'] == 0 and
            report['P1_admission'] is False and report['status'] in ('capability_probes_recorded','partial_or_failed'),
            'native fixed census/final input seal')
    c = p['contract']
    expected_fields = dict(anchors=8,probes_per_anchor=5,fixed_denominator=40,independent_roots=2,
        warmup=10,prefix=5,suffix=5,raw_dtype='float32',action_scale=100,relative=True,
        native_position_clip=False,raw_action_bounds=[-1,1],state_dimension=7,no_neural=True,no_goal_pose_edit=True,P1_admission=False)
    require(all(type(c[k]) is type(v) and c[k] == v for k,v in expected_fields.items()), 'native action/time/state contract')
    probes = []
    directions = ((0,0),(.1,0),(-.1,0),(0,.1),(0,-.1))
    require(len(c['probes']) == 5, 'native fixed probe dictionary')
    for b,name,d in zip(c['probes'], PROBES, directions, strict=True):
        require(b['name'] == name and commands(b['commands'],5).tobytes() ==
                np.tile(np.asarray(d,dtype='<f4'),(5,1)).tobytes(), 'native sealed raw0.1 five-step probes')
        probes.append(dict(probe_id=name,commands=deepcopy(b['commands'])))
    full = dict(constructor=1,close=1,reset=2,public_step=20,physics_step=204,set_state=4,
                goal_setter=3,goal_render=2,goal_bind=1,explicit_render=66)
    require(c['caps'] == {k:40*v for k,v in full.items()}, 'native aggregate cap identities')
    require(len(p['plans']) == len(p['anchors']) == len(upstream['rows']) == 8 and
            [key(x) for x in p['plans']] == case_keys() and [key(x) for x in upstream['rows']] == case_keys(), 'native eight anchors/order')
    require(upstream['schema'] == 'p1_moving_overlay_recompile_physical_20261027_v1.report' and
            upstream['prepare_sha256'] == p['physical_prepare_sha256'] and
            upstream['planner_report_sha256'] == p['planner_report_sha256'], 'upstream original physical identity')
    roots=[]; cases=[]
    hashes=dict(prepare_sha256=prepare_sha,source_report_sha256=upstream['source_report_sha256'],
                planner_report_sha256=p['planner_report_sha256'])
    for s in (0,1):
        r=upstream['rows'][4*s]
        roots.append(dict(root_slot=s,task_id=r['task_id'],source_root_state=deepcopy(r['root_state']),goal_state=deepcopy(r['goal'])))
    for i,(plan,anchor,old) in enumerate(zip(p['plans'],p['anchors'],upstream['rows'],strict=True)):
        require(anchor['index'] == i and all(plan[k] == old[k] for k in ('root_slot','condition','arm','selection_digest','task_id')) and
                plan['source_report_sha256'] == hashes['source_report_sha256'] and
                plan['planner_report_sha256'] == hashes['planner_report_sha256'], 'native plan/source/task join')
        root=roots[plan['root_slot']]
        require(exact(old['root_state'],root['source_root_state']) and exact(old['goal'],root['goal_state']), 'common original task direction per root')
        cp=anchor['checkpoint']; snapshot(cp['state'],cp['snapshot'])
        require(cp['step'] == 5 and len(anchor['trace']) == 51, 'native full prefix anchor trace')
        saved_start=dict(state=deepcopy(cp['state']),snapshot=deepcopy(cp['snapshot']),history_sha256=digest(anchor['trace']))
        cases.append(dict(root_slot=plan['root_slot'],condition=plan['condition'],arm=plan['arm'],
            task_id=plan['task_id'],selection_digest=plan['selection_digest'],prefix_commands=commands(plan['commands'],10)[:5].tolist(),expected_start=saved_start))
    require(len(report['rows']) == 40 and [(*key(r),r['probe']) for r in report['rows']] ==
            [(*k,b) for k in case_keys() for b in PROBES], 'native forty branch denominator')
    branch_keys={(*k,b) for k in case_keys() for b in PROBES}
    for e in events:
        if e['kind'] in ('operation_attempted','operation_completed','operation_failed','returned_transition','fresh_observation'):
            require((e.get('root_slot'),e.get('condition'),e.get('arm'),e.get('probe')) in branch_keys,
                    'no unassigned native work outside forty branches')
    rows=[]; totals={k:dict(attempted=0,completed=0) for k in RESOURCES}
    for i,r in enumerate(report['rows']):
        ci=i//5; plan=p['plans'][ci]; anchor=p['anchors'][ci]; case=cases[ci]; probe=probes[i%5]
        require(type(r['index']) is int and type(r['anchor_index']) is int and type(r['root_slot']) is int and
                r['index']==i and r['anchor_index']==ci and all(r[k]==plan[k] for k in
                ('root_slot','condition','arm','selection_digest','task_id')), 'native branch identity')
        local=[e for e in events if all(e.get(k)==r[k] for k in ('root_slot','condition','arm','probe'))]
        counts={k:dict(attempted=0,completed=0) for k in RESOURCES}; journal=[]
        transitions=[]
        for e in local:
            if e['kind'] in ('operation_attempted','operation_completed','operation_failed'):
                name=e['resource']; require(name in counts,'native known meter resource')
                kind={'operation_attempted':'attempt','operation_completed':'return','operation_failed':'failed'}[e['kind']]
                journal.append(dict(resource=name,kind=kind))
                if kind in ('attempt','return'): counts[name]['attempted' if kind=='attempt' else 'completed']+=1
            if e['kind']=='returned_transition': transitions.append(e)
        warm=[e for e in transitions if e['phase']=='warmup']; prefix=[e for e in transitions if e['phase']=='plan']
        suffix=[e for e in transitions if e['phase']=='probe']
        require(len(warm)+len(prefix)+len(suffix)==len(transitions) and
                len(warm)==r['warmup_returned'] and len(prefix)+len(suffix)==r['own_returned'], 'actual native phase/count identities')
        warm_commands=np.concatenate((np.tile(np.asarray([.1,0],dtype='<f4'),(5,1)),np.tile(np.asarray([0,.1],dtype='<f4'),(5,1))))
        for phase,tt,aa in (('warmup',warm,warm_commands),('plan',prefix,commands(case['prefix_commands'],5)),('probe',suffix,commands(probe['commands'],5))):
            require(len(tt)<=len(aa) and all(type(e['step']) is int for e in tt) and
                    [e['step'] for e in tt]==list(range(1,len(tt)+1)), 'real continuous native phases')
            for j,e in enumerate(tt):
                require(e['requested_bits']==e['actual_bits']==aa[j].tobytes().hex() and e['actual_dtype']=='<f4', 'actual native raw command witness')
                snapshot(e['state'],e['snapshot'])
                require(type(e['terminated']) is bool and type(e['truncated']) is bool,'literal native flags')
                if j<len(tt)-1: require(not e['terminated'] and not e['truncated'],'no continuation after native stop')
        verified=r['start_verified']; require(type(verified) is bool,'literal start verification')
        saved_start=None; trajectory=[]
        if verified:
            require(len(warm)==10 and len(prefix)==5 and exact(r['start_state'],case['expected_start']['state']), 'native exact prefix start state')
            observations=[e for e in local if e['kind']=='fresh_observation' and e['receipt']['phase'] in ('warmup','plan')]
            require(len(observations)==51,'all native original history/prefix triplets')
            fields=('phase','step','condition','overlay_index','file_sha256','pixel_sha256','state','snapshot')
            for e,old in zip(observations,anchor['trace'],strict=True):
                require(exact(e['pre'],e['post']) and exact(e['pre'],old['pre']) and exact(old['pre'],old['post']) and
                        all(exact(e['receipt'][k],old['receipt'][k]) for k in fields), 'native exact reconstructed state/history identity')
            require(any(e['kind']=='suffix_start_verified' for e in local), 'actual suffix start journal witness')
            require(exact(prefix[-1]['state'],case['expected_start']['state']) and
                    exact(prefix[-1]['snapshot'],case['expected_start']['snapshot']), 'last real prefix/anchor identity')
            saved_start=deepcopy(case['expected_start'])
            trajectory=[dict(step=0,state=deepcopy(saved_start['state']),snapshot=deepcopy(saved_start['snapshot']),
                native_success=None,reward=None,terminated=False,truncated=False)]
            trajectory.extend(dict(step=e['step'],state=deepcopy(e['state']),snapshot=deepcopy(e['snapshot']),
                native_success=e.get('native_success'),reward=e['reward'],terminated=e['terminated'],truncated=e['truncated']) for e in suffix)
        else:
            require(not suffix and r['start_state'] is None,'no response before exact source reconstruction')
        require(len(r['responses'])<=len(suffix),'no invented native response')
        for response,e in zip(r['responses'],suffix):
            delta=np.asarray(e['state'])-np.asarray(saved_start['state'])
            require(response['step']==e['step'] and exact(response['state'],e['state']) and
                    response['terminated'] is e['terminated'] and response['truncated'] is e['truncated'] and
                    np.array_equal(np.asarray(response['delta_native7']),delta) and
                    response['delta_angle_wrapped']==float(np.arctan2(np.sin(delta[4]),np.cos(delta[4]))), 'native response/transition consistency')
        statuses={'probe_completed':'completed','native_stopped':'native_stopped','engineering_failed':'engineering_failed','not_started':'not_started'}
        require(r['status'] in statuses,'native stable branch status')
        if r['status'] in ('probe_completed','native_stopped'): require(len(r['responses'])==len(suffix),'complete returned native responses')
        row=dict(root_slot=r['root_slot'],condition=r['condition'],arm=r['arm'],probe_id=r['probe'],selection_digest=r['selection_digest'],
            task_id=r['task_id'],**hashes,probe_sha256=digest(probe),status=statuses[r['status']],start=saved_start,
            prefix_commands=commands(case['prefix_commands'],5)[:len(prefix)].tolist(),warmup_returned=len(warm),trajectory=trajectory,
            returned_commands=commands(probe['commands'],5)[:len(suffix)].tolist(),counts=counts,journal=journal)
        rows.append(row)
        for k in totals:
            for n in ('attempted','completed'): totals[k][n]+=counts[k][n]
    require(totals==report['counts'],'global actual native meter denominator')
    if report['status']=='capability_probes_recorded':
        require(report['errors']==[] and all(r['status'] in ('completed','native_stopped') for r in rows), 'native successful report census')
    protocol=dict(roots=roots,cases=cases,probes=probes,prefix_steps=5,probe_steps=5,warmup_steps=10,meter_per_full_branch=full)
    return dict(schema=PRODUCER+'.saved_analysis',units=deepcopy(UNITS),**hashes,protocol=protocol,rows=rows)


def load_native(output, report_sha, prepared, prepare_sha):
    """Direct Nash report/prepare/journal adapter; actual hashes supplied by MAIN."""
    bindings={}
    def bind(path, expected):
        hash_value(expected); target=Path(path).resolve()
        require(sha(target)==expected,'native saved file SHA mismatch: '+str(target))
        require(str(target) not in bindings or bindings[str(target)]==expected,'native seal conflict')
        bindings[str(target)]=expected; return target
    directory=Path(output).resolve(); report=read(bind(directory/'report.json',report_sha))
    p=read(bind(prepared,prepare_sha))
    # Read every frozen dependency, including native source/PNG/NPZ, without execution.
    for path,h in p['bindings'].items():
        require(Path(path).is_absolute(),'native absolute frozen dependency'); bind(path,h)
    upstream_path=Path(p['physical_output'])/'report.json'
    upstream=read(bind(upstream_path,p['physical_report_sha256']))
    binding=read(directory/'execution_binding.json')
    require(binding==dict(prepare_sha256=prepare_sha,physical_report_sha256=p['physical_report_sha256'],
                         planner_report_sha256=p['planner_report_sha256']), 'native execution binding')
    bindings[str((directory/'execution_binding.json').resolve())]=sha(directory/'execution_binding.json')
    events=read_journal(bind(directory/'journal.jsonl',report['journal_sha256']),report['journal_chain_tail'])
    manifest=normalize_native(p,report,events,upstream,prepare_sha)
    result=analyze(manifest)
    for path,h in bindings.items(): require(sha(path)==h,'native input drift')
    result.update(input_bindings=bindings,physical_probe_report_sha256=report_sha,
                  input_adapter='Nash native report/prepare/journal, no producer import')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest')
    parser.add_argument('--manifest-sha')
    parser.add_argument('--physical-output')
    parser.add_argument('--physical-report-sha')
    parser.add_argument('--prepared')
    parser.add_argument('--prepare-sha')
    parser.add_argument('--output', required=True)
    a = parser.parse_args(argv)
    if a.manifest is not None:
        require(a.manifest_sha is not None and all(x is None for x in
                (a.physical_output,a.physical_report_sha,a.prepared,a.prepare_sha)), 'choose one saved interface')
        result=load_saved(a.manifest,a.manifest_sha)
    else:
        require(a.manifest_sha is None and all(x is not None for x in
                (a.physical_output,a.physical_report_sha,a.prepared,a.prepare_sha)), 'all native saved paths and external SHA required')
        result=load_native(a.physical_output,a.physical_report_sha,a.prepared,a.prepare_sha)
    path = Path(a.output).absolute(); require(not path.exists(), 'new exclusive analysis output only')
    value = canonical(result)
    require(len(value) <= MAX_JSON, 'bounded saved analysis output')
    with path.open('xb') as stream:
        require(stream.write(value) == len(value), 'short analysis write'); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(dict(output=str(path),sha256=sha(path),status=result['status']), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
