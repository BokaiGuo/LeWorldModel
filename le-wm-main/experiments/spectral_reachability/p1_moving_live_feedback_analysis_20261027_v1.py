"""Saved-only O/I/R descriptive analysis. No producer import or neural/physics runtime."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import shutil
import numpy as np

NAME = 'p1_moving_live_feedback_analysis_20261027_v1'
DRIVER = 'p1_moving_live_feedback_physical_20261027_v1'
DRIVER_SHA = '6742a1b35102e433d7772e4a208b4f97b37b76f815b991cfc0e472763754ace9'
ENCODER = 'p1_constructive_moving_history_encode_cuda_20261027_v1'
ARMS = ('line_search_GN', 'full10_conditional_D2P')
MODES = ('O', 'I', 'R')
LAYERS = tuple((s, a) for s in (0, 1) for a in ARMS)
KEYS = tuple((s, a, m) for s, a in LAYERS for m in MODES)
TASK_SHAS = ('a42bfbb1ea692782355e2572a5df2cd048b72cf3e6c41fee40eb12503674959b',
             '101add21a7a40591167a6ab8e8b22c3c62cb1fd700c741fd36cd2073e2d5df74')
NEURAL = dict(encoder=24, projector=24, predict=40, predictor=40, action_encoder=40,
              pred_proj=40, ad_forward=8, jacobian=8, svd=8, jacobian_api=8,
              grad_api=8, vjp_directions=1536)
SUFFIX = dict(predict=5, predictor=5, action_encoder=5, pred_proj=5, ad_forward=1,
              jacobian=1, svd=1, jacobian_api=1, grad_api=1, vjp_directions=192)
CAP, FLOOR = 1024**2, 200*1024**2


def require(value, message):
    if not value: raise ValueError(message)


def canonical(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode()


def digest(value):
    # Driver source.digest uses DEFAULT spaces and no newline.
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha_bytes(value): return hashlib.sha256(value).hexdigest()


def hash_value(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None, 'explicit lowercase SHA256')
    return value


def finite(value):
    require(type(value) in (int, float) and math.isfinite(value), 'finite literal scalar, not bool')
    return float(value)


def array(value, shape, dtype=None):
    a = np.asarray(value)
    require(a.shape == shape and a.dtype.kind in 'fi' and np.isfinite(a).all(), 'finite exact array shape')
    if dtype is not None: require(a.dtype == dtype, 'literal original array dtype')
    return a


def read_bytes(path, expected, cap=16*1024**2):
    hash_value(expected); p = Path(path)
    require(p.is_file() and not p.is_symlink() and p.stat().st_size <= cap, 'bounded regular sealed input: '+str(p))
    value = p.read_bytes(); require(len(value) <= cap and sha_bytes(value) == expected, 'external input SHA: '+str(p))
    return value


def parse(value):
    def pairs(items):
        out = {}
        for k, v in items:
            require(k not in out, 'duplicate JSON key'); out[k] = v
        return out
    def invalid(x): raise ValueError('nonfinite JSON '+x)
    return json.loads(value, object_pairs_hook=pairs, parse_constant=invalid)


def local(root, name):
    root = Path(root).resolve(); p = Path(name)
    if not p.is_absolute(): p = root/p
    require(p.is_relative_to(root) and all(x not in ('.', '..') for x in p.relative_to(root).parts), 'own saved output path')
    require(not any(q.is_symlink() for q in [p, *p.parents] if q != root.parent), 'no symlink input')
    return p


def key(row):
    s, a, m = row['source_slot'], row['prefix_arm'], row['mode']
    require(type(s) is int and (s, a, m) in KEYS, 'fixed source_slot/prefix_arm/mode; no unknown interface')
    return s, a, m


def native_metrics(state, goal):
    """Saved-state arithmetic matching old native eval, including its non-mod angle bug."""
    state = array(state, (5,)).astype(float); d = state-array(goal, (5,)).astype(float)
    angle = abs(float(d[4])); native_angle = min(angle, 2*np.pi-angle)
    wrapped = abs(float(np.arctan2(np.sin(d[4]), np.cos(d[4]))))
    return dict(native_success=bool(np.linalg.norm(d[:4]) < 20 and native_angle < np.pi/9),
                native_distance=float(np.linalg.norm(d)), native_angle_term=float(native_angle),
                negative_native_angle=bool(native_angle < 0),
                wrapped_allxy_success=bool(np.linalg.norm(d[:4]) < 20 and wrapped < np.pi/9),
                wrapped_object_angle_error=wrapped, object_position_error=float(np.linalg.norm(d[2:4])),
                object_pose=state[2:5].tolist(), object_success=None)


def observed_q(z, goal):
    z = array(z, (192,), np.dtype('float32')); goal = array(goal, (192,)).astype(float)
    return float(np.mean((z.astype(float)-goal)**2))


def validate_contract(p, report):
    require(p['schema'] == DRIVER+'.prepare' and report['schema'] == DRIVER+'.report', 'explicit driver schema only')
    require(report['status'] in ('live_feedback_calibration_recorded', 'partial_or_failed'), 'terminal saved report required')
    c = p['contract']
    require(report['contract'] == c and c['layers'] == [list(x) for x in LAYERS] and c['modes'] == list(MODES) and
            c['episodes'] == report['fixed_denominator'] == 12 and c['feedback_step'] == 5 and c['horizon'] == 10 and
            c['warmup'] == 10 and c['with_target'] is False and c['neural_caps'] == NEURAL,
            'fixed12 / original continuous moving / exact neural budget')
    require(report['P1_admission'] is False and report['P4_admission'] is False and report['training'] is False and
            report['independent_confirmation'] is False and report['independent_tasks'] == 2, 'calibration only, no admission')
    require([(x['source_slot'], x['arm']) for x in p['layers']] == list(LAYERS) and
            all(type(x['source_slot']) is int for x in p['layers']), 'four predeclared layers')
    for layer in p['layers']:
        require(layer['task_sha256'] == TASK_SHAS[layer['source_slot']], 'original moving task SHA')
        a = array(layer['commands'], (10, 2))
        require(np.array_equal(a, a.astype(np.float32)) and (abs(a) <= 1).all(), 'literal selected full10 raw commands')


def census(events, report, p):
    """Count saved attempted/returned operations; predict API and predictor NOT added together."""
    if events is None: return None
    caps = dict(p['contract']['physical_caps'], **NEURAL)
    counts = lambda: {k: dict(attempted=0, completed=0) for k in caps}
    total = counts(); by_episode = {f'e{i:02}_{m}': dict(all=counts(), suffix=counts())
                                  for i in range(4) for m in MODES}
    pending = {}; seen = set()
    require(events and events[0]['kind'] == 'run_started' and events[0]['prepare'] == p, 'journal exact original prepare')
    for e in events:
        if e['kind'] not in ('operation_attempted', 'operation_completed'): continue
        k, ident, ep, units = e['resource'], e['operation_id'], e['episode_id'], e['units']
        require(k in caps and ep in by_episode and type(ident) is int and type(units) is int and units > 0,
                'known resource and episode census')
        payload = {x: e[x] for x in ('resource', 'episode_id', 'units', 'region')}
        if e['kind'] == 'operation_attempted':
            require(ident not in seen, 'unique operation attempt'); seen.add(ident); pending[ident] = payload; field = 'attempted'
        else:
            require(ident in pending and pending.pop(ident) == payload, 'matching saved operation completion'); field = 'completed'
        total[k][field] += units; by_episode[ep]['all'][k][field] += units
        if e['region'] == 'compiling': by_episode[ep]['suffix'][k][field] += units
    require(total == report['counts'] and all(v['completed'] <= v['attempted'] <= caps[k] for k, v in total.items()),
            'saved global census and cap; no attempt refunds')
    if report['status'] == 'live_feedback_calibration_recorded': require(not pending, 'recorded run cannot hide pending operations')
    return dict(total=total, episodes=by_episode, pending_operation_ids=sorted(pending))


def checkpoint(row, step):
    found = [x for x in row.get('checkpoints', []) if x['phase'] == 'plan' and x['step'] == step]
    require(len(found) <= 1, 'unique literal checkpoint')
    return found[0] if found else None


def flatten(endpoint):
    out = {'Q': endpoint['observed_Q']}
    for name in ('native_distance', 'object_position_error', 'wrapped_object_angle_error',
                 'native_angle_term', 'native_success', 'wrapped_allxy_success'):
        value = endpoint['native_recomputed']
        out[name] = None if value is None else value[name]
    return out


def analyze(report, p, events, goals, load_latent):
    """Pure saved-data transform. Missing observations stay null; no neural callback."""
    validate_contract(p, report); ledger = census(events, report, p)
    rows_by_key = {}
    for row in report['rows']:
        k = key(row); require(k not in rows_by_key, 'duplicate episode row')
        require(row['episode_id'] == f'e{LAYERS.index(k[:2]):02}_{k[2]}', 'episode id must match root/arm/mode')
        require(row['status'] in ('not_started', 'running', 'failed', 'completed10', 'native_stopped_warmup',
                'native_stopped_prefix', 'native_stopped_suffix'), 'known driver episode status')
        rows_by_key[k] = row
    result_rows = []; endpoints = {}
    for s, a, mode in KEYS:
        row = rows_by_key.get((s, a, mode)); ep = f'e{LAYERS.index((s,a)):02}_{mode}'
        issues = [] if row is not None else ['missing_episode_row']
        if row is not None and row.get('prefix_pairing') is not None:
            cp5 = checkpoint(row, 5); layer = next(x for x in p['layers'] if (x['source_slot'], x['arm']) == (s,a))
            require(cp5 is not None and row['prefix_pairing'] == dict(pixel_sha256=cp5['pixel_sha256'], state=cp5['state'],
                snapshot=cp5['snapshot'], actual_raw_sha256=sha_bytes(np.asarray(layer['commands'][:5], np.float32).tobytes()),
                goal_sha256=digest(cp5['goal'])), 'prefix pairing must describe own checkpoint and fixed commands')
        cps = [] if row is None else [checkpoint(row, step) for step in (0, 5, 10)]
        goals_present = [c['goal'] for c in cps if c is not None]
        require(not goals_present or all(g == goals_present[0] for g in goals_present), 'native goal cannot change between checkpoints')
        item = dict(source_slot=s, prefix_arm=a, mode=mode, episode_id=ep,
                    status='missing' if row is None else row['status'], endpoints={},
                    original_final_metrics=None if row is None else deepcopy(row.get('final_metrics')),
                    native_success=None if row is None else row.get('native_success'),
                    terminated=None if row is None else row.get('terminated'),
                    truncated=None if row is None else row.get('truncated'),
                    full_horizon=None if row is None else row.get('full_horizon'),
                    first_error=None if row is None else row.get('first_error'),
                    secondary_errors=[] if row is None else deepcopy(row.get('secondary_errors', [])), issues=issues)
        for step in (5, 10):
            cp = None if row is None else checkpoint(row, step)
            end = dict(observed_Q=None, native_recomputed=None, native_origin='recomputed_saved_state_not_native_receipt',
                       original_report_Q=None if row is None else row.get(f'actual_Q{step}'), checkpoint_present=cp is not None)
            if cp is None: issues.append(f'missing_checkpoint{step}')
            else:
                end['native_recomputed'] = native_metrics(cp['state'], cp['goal'])
                response = row.get(f'encoding{step}')
                if response is None: issues.append(f'missing_encoding{step}')
                else:
                    require(response['schema'] == DRIVER+'.encode_response' and response['episode_id'] == ep and
                            response['observed_step'] == step and response['device_type'] == 'cuda' and
                            response['source_verified'] is True and response['model_unchanged'] is True and
                            response['model_hash_before'] == response['model_hash_after'] and
                            response['resources'] == {x: dict(attempted=1, completed=1) for x in ('encoder', 'projector')},
                            'own original CUDA encoding receipt; never CPU fallback')
                    z = load_latent(response['latent'])
                    if z is None: issues.append(f'missing_latent{step}')
                    else:
                        end['observed_Q'] = observed_q(z, goals[s])
                        if end['original_report_Q'] is not None:
                            require(finite(end['original_report_Q']) == end['observed_Q'], 'reported Q differs from saved native192/goal')
                if step == 10 and row.get('final_metrics') is not None:
                    require(cp['state'] == row['final_state'] and row['final_metrics'] == end['native_recomputed'],
                            'original native metrics must match actual terminal checkpoint')
            item['endpoints'][str(step)] = end; endpoints[s, a, mode, step] = end
        item['suffix_budget_expected'] = {k: dict(attempted=(0 if mode == 'O' else v),
            completed=(0 if mode == 'O' else v)) for k, v in SUFFIX.items()}
        item['suffix_budget_actual'] = None if ledger is None else {k: ledger['episodes'][ep]['suffix'][k] for k in SUFFIX}
        item['episode_actual_resources'] = None if ledger is None else ledger['episodes'][ep]['all']
        if ledger is None: issues.append('missing_journal_census')
        item['suffix_budget_matches_full_protocol'] = None if ledger is None else item['suffix_budget_actual'] == item['suffix_budget_expected']
        item['Q10_minus_Q5'] = difference(item['endpoints']['10']['observed_Q'], item['endpoints']['5']['observed_Q'])
        result_rows.append(item)
    contrasts = []
    analyzed = {(x['source_slot'],x['prefix_arm'],x['mode']):x for x in result_rows}
    for s, a in LAYERS:
        for lhs, rhs in (('I', 'O'), ('R', 'I'), ('R', 'O')):
            lr, rr = rows_by_key.get((s,a,lhs)), rows_by_key.get((s,a,rhs))
            pairing = None
            if lr is not None and rr is not None and lr.get('prefix_pairing') is not None and rr.get('prefix_pairing') is not None:
                pairing = lr.get('initial_exact') is True and rr.get('initial_exact') is True and lr['prefix_pairing'] == rr['prefix_pairing']
            row = dict(source_slot=s, prefix_arm=a, contrast=lhs+'-'+rhs, prefix_pairing_exact=pairing,
                       reason=None if pairing is True else 'missing_or_mismatched_prefix_pairing', endpoints={})
            for step in (5, 10):
                l, r = flatten(endpoints[s,a,lhs,step]), flatten(endpoints[s,a,rhs,step])
                row['endpoints'][str(step)] = {k: difference(l[k], r[k]) if pairing is True else None for k in l}
            l, r = analyzed[s,a,lhs]['suffix_budget_actual'], analyzed[s,a,rhs]['suffix_budget_actual']
            row['suffix_budget_contrast'] = None if ledger is None or pairing is not True else {
                k: {field:difference(l[k][field],r[k][field]) for field in ('attempted','completed')} for k in SUFFIX}
            contrasts.append(row)
    return dict(schema=NAME+'.report', status='saved_descriptive_analysis_not_admitted', rows=result_rows, contrasts=contrasts,
                fixed_episode_denominator=12, fixed_layer_denominator=4, independent_tasks=2,
                observed_endpoint_count=sum(e['observed_Q'] is not None for e in endpoints.values()),
                census=ledger, source_status=report['status'], source_errors=deepcopy(report['errors']),
                interpretation='paired descriptions only; R-I isolates context conditional on pairing, not universal causality',
                independent_samples=False, significance=None, P1_admission=False, P4_admission=False,
                new_neural=0, new_physics=0, CPU_neural_fallback=False, training=False,
                unknown_predictions='not filled or recomputed; suffix model work reported separately from observed Q')


def difference(a, b):
    if a is None or b is None: return None
    return finite(int(a) if type(a) is bool else a)-finite(int(b) if type(b) is bool else b)


def journal(value, tail):
    events = []; previous = None
    for i, line in enumerate(value.splitlines()):
        e = parse(line)
        require(e['index'] == i and e['previous_sha256'] == previous and
                e['event_sha256'] == digest({k:v for k,v in e.items() if k != 'event_sha256'}), 'saved journal chain')
        events.append(e); previous = e['event_sha256']
    require(previous == tail, 'saved terminal chain tail'); return events


def build(report_path, report_sha, prepared, prepare_sha):
    """Explicit four-argument saved entry, no source/model execution or directory search."""
    report_path, prepared = Path(report_path).resolve(), Path(prepared).resolve(); root = report_path.parent
    report = parse(read_bytes(report_path, report_sha)); p = parse(read_bytes(prepared, prepare_sha, 256*1024))
    bound = {str(report_path): report_sha, str(prepared): prepare_sha}
    binding_path = root/'execution_binding.json'; b = binding_path.read_bytes()
    binding = parse(b); require(Path(binding['prepared']).resolve() == prepared and binding['prepare_sha256'] == prepare_sha,
                              'external execution binding to exact supplied prepare')
    bound[str(binding_path)] = sha_bytes(b); validate_contract(p, report)
    driver_path = Path(__file__).with_name(DRIVER+'.py').resolve()
    require(p['bindings'].get(str(driver_path)) == DRIVER_SHA, 'prepare must bind MAIN frozen driver authority')
    read_bytes(driver_path, DRIVER_SHA); bound[str(driver_path)] = DRIVER_SHA
    goals = {}
    for layer in p['layers']:
        s = layer['source_slot']; path, h = Path(layer['task_export']), layer['task_sha256']
        require(path.is_absolute() and p['bindings'].get(str(path)) == h, 'explicit prepared task binding')
        compressed = read_bytes(path, h, 1024**2)
        with gzip.GzipFile(fileobj=BytesIO(compressed)) as f: task_bytes = f.read(4*1024**2+1)
        require(len(task_bytes) <= 4*1024**2, 'bounded task expansion'); export = parse(task_bytes)
        require(export['schema'] == ENCODER+'.task_export' and type(export['source_slot']) is int and export['source_slot'] == s and
                export['task_id'] == layer['task_id'] and export['with_target'] is False and export['P1_admission'] is False and
                export['task_contract']['metric'] == 'identity_native192_no_fit' and
                np.array_equal(export['task']['projection'], np.eye(192)), 'original moving native192 task/goal authority')
        g = array(export['task']['goal'], (192,)).astype(float)
        require(np.array_equal(g, g.astype(np.float32)), 'original float32 goal')
        require(s not in goals or np.array_equal(goals[s], g), 'same root same goal across prefix arms')
        goals[s] = g; bound[str(path)] = h
    jp = root/'journal.jsonl'; events = None
    if report.get('journal_sha256') is not None and jp.exists():
        h = report['journal_sha256']; events = journal(read_bytes(jp, h, 64*1024**2), report['journal_chain_tail']); bound[str(jp)] = h
    for row in report['rows']:
        key(row)
        for step in (5, 10):
            if row.get(f'encoding{step}') is not None:
                expected_path = root/(row['episode_id']+f'_encode{step}')/'native192.npy'
                require(local(root, row[f'encoding{step}']['latent']['path']) == expected_path,
                        'own episode/step NPY namespace; never borrowed encoding')
    def load_latent(receipt):
        path = local(root, receipt['path'])
        if not path.exists(): return None
        value = read_bytes(path, receipt['file_sha256'], 1024**2); stream = BytesIO(value); a = np.load(stream, allow_pickle=False)
        require(stream.tell() == len(value) and receipt['dtype'] == 'float32' and receipt['shape'] == [192] and
                sha_bytes(np.ascontiguousarray(a).tobytes()) == receipt['array_sha256'], 'single original NPY receipt')
        array(a, (192,), np.dtype('float32')); bound[str(path)] = receipt['file_sha256']; return a
    result = analyze(report, p, events, goals, load_latent)
    for path, h in bound.items(): read_bytes(path, h, 64*1024**2)
    result['inputs'] = dict(report=str(report_path), report_sha256=report_sha, prepared=str(prepared), prepare_sha256=prepare_sha)
    result['input_sha256'] = bound
    result['validation_scope'] = 'saved report/prepare/execution binding/task goal/NPY/journal; not a runtime correctness audit or rehash of model closure'
    files = [Path(__file__), Path(__file__).parents[2]/'tests'/('test_'+NAME+'.py'), Path(__file__).parents[2]/'innerdocs'/(NAME+'.md')]
    result['analysis_source_sha256'] = {str(f.resolve()): sha_bytes(f.read_bytes()) for f in files}
    return result


def save(output, result):
    path = Path(output); payload = canonical(result)
    require(not path.exists() and len(payload) <= CAP and shutil.disk_usage(path.parent).free >= FLOOR+len(payload),
            'exclusive analysis <=1MiB and 200MiB floor')
    with path.open('xb') as f:
        require(f.write(payload) == len(payload), 'short analysis write'); f.flush(); os.fsync(f.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return dict(output=str(path.resolve()), sha256=sha_bytes(payload), bytes=len(payload))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('report', 'report-sha', 'prepared', 'prepare-sha', 'output'): parser.add_argument('--'+arg, required=True)
    a = parser.parse_args(); print(canonical(save(a.output, build(a.report, a.report_sha, a.prepared, a.prepare_sha))).decode(), end='')


if __name__ == '__main__': main()
