"""Existing sealed P2 programs -> existing calibration runner -> strict saved IO.

No new planner, manifest, model or environment implementation. Only MAIN may
call run-calibration. All other entries consume saved metadata/RGB arithmetic.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import agent_p2_execution_assets_20261028_v1 as assets
from experiments.spectral_reachability.audit_pusht_sealed_task_contract import labels

NAME = 'agent_p2_composition_execution_20261028_v1'
ROOT = assets.ROOT
ASSETS_SHA = 'e8c1dff3f62de4caf89e83bed0ebaed6dc2d0c4a1da1ade02966f7c1d7e2452d'
AUTH = 'MAIN_APPROVED_P2_KNOWN_ROOT_CALIBRATION_ONLY'
require, sha, digest = assets.require, assets.sha, assets.digest
physical = assets.physical
PHASES = ('root_collection', 'warmup', 'selected')
FLAGS = dict(P1_admission=False, P2_admission=False, scientific_admission=False, freshness=False,
    physical_execution_authorized=False, independent_confirmation=False, equal_full_resource_cost=False)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def fields(value, names):
    require(type(value) is dict and set(value) == set(names.split()), 'exact saved fields: '+names)


def hash_value(value):
    require(type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value), 'literal actual SHA256')


def pairs(items):
    result = {}
    for k, v in items:
        require(k not in result, 'duplicate JSON key'); result[k] = v
    return result


def read(path, cap=assets.OUTPUT_CAP):
    require(Path(path).stat().st_size <= cap, 'bounded saved JSON')
    value = json.loads(Path(path).read_bytes(), object_pairs_hook=pairs)
    canonical(value); return value


def lines(path):
    require(Path(path).stat().st_size <= assets.OUTPUT_CAP, 'bounded saved journal')
    result = []
    with Path(path).open('rb') as stream:
        for line in stream:
            require(len(line) <= 1024**2 and line.endswith(b'\n'), 'complete bounded journal line')
            value = json.loads(line, object_pairs_hook=pairs); canonical(value); result.append(value)
    return result


def finite(value, shape):
    def check(x):
        if type(x) is list:
            for v in x: check(v)
        else: require(type(x) in (int, float) and np.isfinite(x), 'literal finite numbers, not bool')
    check(value); a = np.asarray(value)
    require(a.shape == shape and a.dtype.kind in 'fi', 'exact finite shape')
    return a


def integer(value):
    require(type(value) is int and value >= 0, 'literal nonnegative counter'); return value


def own_files():
    return [Path(__file__).resolve(), ROOT/f'tests/{NAME}_test.py', ROOT/f'innerdocs/{NAME}.md']


def source_check():
    require(sha(assets.__file__) == ASSETS_SHA, 'existing execution-assets source changed; MAIN must resolve new version')
    files = own_files()+[Path(assets.__file__), Path(labels.__code__.co_filename)]
    return {str(p.resolve()): sha(p) for p in files}


def verify(prepared, expected):
    hash_value(expected); source_check()
    require(sha(prepared) == expected, 'external physical prepare SHA')
    p = assets.verify(prepared, expected)
    require(p['schema'] == assets.NAME+'.prepare' and p['contract'] == assets.contract() and
        p['formal_execution_authorized'] is p['P1_admission'] is p['P2_admission'] is False and len(p['plans']) == 8,
        'existing four-root calibration contract only, not scientific release')
    require(sha(prepared) == expected, 'unchanged prepare after source/library rebuild')
    return p


def artifact(output, path):
    p = Path(path)
    require(p.is_absolute() and p.parent == output and p.name not in ('', '.', '..') and
        p.is_file() and not p.is_symlink(), 'flat actual artifact inside this exclusive run')
    return p


def png(output, event):
    from PIL import Image
    path = artifact(output, event['rgb_path'])
    require(sha(path) == event['rgb_file_sha256'], 'exact PNG bytes')
    with Image.open(path) as image:
        require(image.format == 'PNG' and image.mode == 'RGB' and image.size == (224, 224), 'native RGB224 PNG, no conversion')
        rgb = np.asarray(image).copy()
    require(rgb.dtype == np.uint8 and rgb.shape == (224, 224, 3), 'actual uint8 RGB')
    return rgb


def meter(p, receipt, records, rows):
    """Reconcile actual public-call attempts/returns, not internal physics."""
    if receipt is None:
        require(records == [] and all(r['physical'] is None for r in rows), 'unknown meter cannot hide branch work')
        return None, {}
    fields(receipt, 'steps resets journal_errors total_public_env_step_attempts total_public_env_step_returns total_env_reset_attempts total_env_reset_returns scope')
    require(receipt['journal_errors'] == [], 'broken meter journal is not verified evidence')
    counts = {name: {phase: dict(attempts=0, returns=0) for phase in PHASES} for name in ('steps', 'resets')}
    local = {}; order = {(x['root_sha256'], x['execution_plan']['method']): i for i, x in enumerate(p['plans'])}
    rank = -1; selection_started = set()
    require(len(records) % 2 == 0, 'no dangling public-call attempt')
    for before, after in zip(records[::2], records[1::2], strict=True):
        require(before['kind'] in ('env_step', 'env_reset') and before['phase'] in PHASES and before['completed'] is False,
            'actual public step/reset intent')
        phase = before['phase']; name = 'steps' if before['kind'] == 'env_step' else 'resets'
        binding = [] if phase == 'root_collection' else ['root_sha256', 'method', 'selection_sha256']
        keys = ['kind', 'phase', 'index', 'completed']+binding+(['command'] if name == 'steps' else [])
        fields(before, ' '.join(keys)); fields(after, ' '.join(keys+(['error'] if 'error' in after else [])))
        require(type(after['completed']) is bool and all(after[k] == before[k] for k in keys if k != 'completed'), 'exact call attempt/disposition pairing')
        require(after['completed'] is True or (type(after.get('error')) is str and bool(after['error'])), 'failed call not silently completed')
        require(after['completed'] is False or 'error' not in after, 'no error suppressed as return')
        c = counts[name][phase]; require(type(before['index']) is int and before['index'] == c['attempts'], 'phase-specific public index order')
        c['attempts'] += 1; c['returns'] += int(after['completed'])
        if phase == 'root_collection':
            require(rank == -1 and name == 'resets', 'original root collection has no public steps')
        else:
            key = (before['root_sha256'], before['method']); require(key in order, 'fixed original root/arm, not Jason moving beam')
            i = order[key]; require(i >= rank, 'fixed root-major counterbalanced branch order'); rank = i
            plan = p['plans'][i]['execution_plan']
            require(before['selection_sha256'] == plan['selection_sha256'], 'meter bound to exact sealed raw10 selection')
            group = local.setdefault((i, phase, name), []); group.append((before, after))
            if phase == 'warmup': require(i not in selection_started, 'no warmup after selected dispatch')
            else:
                require(name == 'steps', 'no selected-phase reset/retry')
                selection_started.add(i)
                require(len(local.get((i, 'warmup', 'steps'), [])) == 15, 'full original warmup before selected calls')
            if name == 'steps':
                command = finite(before['command'], (2,)); require(np.all(np.abs(command) <= 1), 'unmodified bounded raw command')
                require(len(group) <= (15 if phase == 'warmup' else 10), 'no padding or repeated branch')
                if phase == 'selected': require(before['command'] == plan['commands'][len(group)-1], 'native selected call uses exact sealed command')
    require(receipt['steps'] == counts['steps'] and receipt['resets'] == counts['resets'], 'complete actual meter receipt/journal equality')
    expected = dict(total_public_env_step_attempts=sum(x['attempts'] for x in counts['steps'].values()),
        total_public_env_step_returns=sum(x['returns'] for x in counts['steps'].values()),
        total_env_reset_attempts=sum(x['attempts'] for x in counts['resets'].values()),
        total_env_reset_returns=sum(x['returns'] for x in counts['resets'].values()))
    require(all(type(receipt[k]) is int and receipt[k] == v for k, v in expected.items()), 'public reset/step totals, no internal-work estimates')
    require(counts['steps']['warmup']['attempts'] <= 120 and counts['steps']['selected']['attempts'] <= 80, 'original full public-step caps')
    return dict(counts=counts, **expected, internal_physics_substeps=None, reset_internal_work=None,
        constructor_calls=None, setter_calls=None, render_calls=None, goal_bind_calls=None,
        wall_seconds=None, scope='actual public reset/step only; uninstrumented work UNKNOWN, not zero'), local


def branch(output, asset, outer, local, index, references):
    fields(outer, 'root_slot arm handoff_sha256 status physical')
    require(all(outer[k] == asset[k] for k in ('root_slot', 'arm', 'handoff_sha256')), 'exact eight selected-program identities/order')
    plan = asset['execution_plan']; result = outer['physical']
    summary = dict(root_slot=asset['root_slot'], root_sha256=asset['root_sha256'], arm=asset['arm'],
        handoff_sha256=asset['handoff_sha256'], status=outer['status'], planned_commands=10,
        attempted_commands=None, returned_commands=None, native_success=None, final_state=None,
        endpoint_fresh=None, full_horizon_completed=None, physical=None, native_result_scope=None)
    calls = local.get((index, 'selected', 'steps'), [])
    if result is None:
        require(outer['status'] in ('not_started', 'failed_partial') and not calls, 'missing branch cannot conceal selected physical calls')
        return summary, set(), None
    require(outer['status'] in ('returned', 'failed_partial'), 'physical result disposition')
    stem = f'root_{plan["root_index"]}_{plan["method"]}'
    journal = artifact(output, result['journal_path'])
    require(journal.name == stem+'.events.jsonl' and sha(journal) == result['journal_sha256'], 'exact branch journal artifact')
    disk = output/(stem+'.physical.json')
    require(disk.is_file() and not disk.is_symlink() and read(disk) == result, 'exact durable branch result')
    physical.verify_branch(plan, result)
    events = lines(journal); images = set(); trace = []; attempted = returned = 0
    last_state = None; next_kind = 'initial'; visible_failure = False
    guard = physical.durable.RgbObservabilityGuard(root_id=plan['root_sha256'], agent_rgb=(65, 105, 225))
    for e in events:
        kind = e['kind']
        require(kind == next_kind, 'ordered initial/attempt/return/observation physical events')
        if kind == 'attempt':
            fields(e, 'kind primitive_step command'); attempted += 1
            require(e['primitive_step'] == attempted and e['command'] == plan['commands'][attempted-1], 'exact next raw action witness')
            require(not visible_failure, 'no blind continuation after RGB guard abort')
            guard.dispatch(e['command']); next_kind = 'step_returned'
        elif kind == 'step_returned':
            fields(e, 'kind primitive_step'); returned += 1
            require(e['primitive_step'] == returned, 'actual public return index'); next_kind = 'observation'
        else:
            fields(e, 'kind state rgb_path rgb_file_sha256' if kind == 'initial' else
                'kind primitive_step command state reward terminated truncated rgb_path rgb_file_sha256')
            finite(e['state'], (7,)); last_state = e['state']
            step = 0 if kind == 'initial' else returned
            require(Path(e['rgb_path']).name == stem+f'_rgb_{step}.png', 'exact step-specific PNG filename')
            rgb = png(output, e); images.add(Path(e['rgb_path']).name)
            if kind == 'observation':
                require(e['primitive_step'] == returned and e['command'] == plan['commands'][returned-1] and
                    type(e['terminated']) is type(e['truncated']) is bool and type(e['reward']) in (int, float) and np.isfinite(e['reward']), 'actual native transition fields')
                trace.append({k: v for k, v in e.items() if k not in ('rgb_path', 'rgb_file_sha256')})
            try: guard.observe(rgb, primitive_step=step)
            except physical.durable.ObservationUnavailable: visible_failure = True
            next_kind = 'attempt'
            if visible_failure or (kind == 'observation' and (e['terminated'] or e['truncated'])):
                next_kind = 'stopped'
    require(result['trace'] == trace and result['final_state'] == last_state and result['guard'] == guard.receipt(), 'saved state/trace and independently recomputed RGB guard')
    require(result['physical_env_step_attempts'] == attempted == len(calls) and
        result['physical_env_step_returns'] == returned == sum(after['completed'] for _, after in calls), 'branch journal vs actual native public calls')
    require(result['remaining_commands_not_executed'] == 10-returned and result['actual_endpoint_primitive_steps'] == returned, 'no missing/padded steps')
    if result['status'] == 'completed': require(returned == len(trace) == 10 and not visible_failure and next_kind == 'attempt', 'complete raw10 observation path')
    elif result['status'] == 'environment_terminated': require(bool(trace) and (trace[-1]['terminated'] or trace[-1]['truncated']) and not visible_failure, 'real native early stop')
    elif result['status'] == 'observation_unavailable': require(visible_failure, 'actual saved guard abort, not invented stop')
    else: require(result['status'] in ('branch_failed', 'execution_failed') and result['failures'], 'explicit engineering failure')
    require(result['status'] != 'completed' or not any(t['terminated'] or t['truncated'] for t in trace), 'cannot continue beyond native stop')
    identity = result['context_identity']; ref = references[plan['root_sha256']]
    require(all(identity[k] == ref[k] for k in ('history_rgb_sha256', 'goal_rgb_sha256')), 'original saved source RGB/history/goal identity')
    require(identity['state'] == events[0]['state'], 'actual initial seven-state equals reset identity')
    finite(identity['state'], (7,)); finite(identity['goal'], (7,))
    history = physical.helpers.validate_reset_history(identity['raw_history'], plan['predictor_context']['action_history'])
    require(all(identity[k] == v for k, v in history.items()), 'original normalized two-token warmup history')
    warm = local.get((index, 'warmup', 'steps'), [])
    require(len(warm) == 15 and all(after['completed'] for _, after in warm) and
        [before['command'] for before, _ in warm] == np.repeat(np.asarray(identity['raw_history']), 5, axis=0).tolist(), 'all fifteen actual warmup commands, no borrowed context')
    official = None
    if last_state is not None:
        official = labels(last_state, identity['goal'])
        require(result.get('official') == official, 'unchanged original agent+object official labels from real seven-state')
        state, goal = np.asarray(last_state), np.asarray(identity['goal'])
        xy = float(np.linalg.norm(state[2:4]-goal[2:4]))
        angle = float(abs(np.arctan2(np.sin(state[4]-goal[4]), np.cos(state[4]-goal[4]))))
        require(result.get('pose') == dict(xy_distance=xy, angle_distance=angle, pose_cost=xy+20.*angle), 'original wrapped diagnostic only, never replace native result')
    summary.update(attempted_commands=attempted, returned_commands=returned,
        native_success=None if official is None else official['official_success'], final_state=last_state,
        endpoint_fresh=result['endpoint_fresh'], full_horizon_completed=result['full_horizon_completed'],
        physical=deepcopy(result), native_result_scope='observed_endpoint' if result['endpoint_fresh'] else 'last_observed_state_not_current_endpoint')
    return summary, {journal.name, disk.name}|images, identity


def validate_saved(p, report, execution, budget, output):
    """Pure saved verification, including valid early stops and negative rows."""
    require(report['schema'] == assets.NAME+'.report' and report['contract'] == assets.contract() and
        type(report['root_denominator']) is type(report['arm_denominator']) is int and
        report['root_denominator'] == 4 and report['arm_denominator'] == 8 and len(report['rows']) == len(p['plans']) == 8 and
        report['model_loaded'] is False and type(report['new_neural_calls']) is int and report['new_neural_calls'] == 0 and
        report['P1_admission'] is report['P2_admission'] is False and report['known_root_calibration'] is True,
        'original complete eight-branch known-root envelope, no admission')
    require(execution == dict(schema=assets.NAME+'.execution', prepared=report['prepared'], prepare_sha256=report['prepare_sha256'],
        main_review_sha256=execution['main_review_sha256'], planner_report_sha256=assets.REPORT_SHA,
        known_root_calibration=True, P1_admission=False, P2_admission=False), 'exact existing execution binding')
    resources, local = meter(p, report['physical_calls'], budget, report['rows'])
    summaries = []; claimed = {'execution.json'}; identities = {}; success = 0
    if report['physical_calls'] is not None: claimed.add('physical_budget.jsonl')
    for i, (asset, row) in enumerate(zip(p['plans'], report['rows'], strict=True)):
        summary, files, identity = branch(output, asset, row, local, i, p['references'])
        summaries.append(summary); claimed |= files
        if identity is not None:
            key = asset['root_sha256']; require(key not in identities or identities[key] == identity, 'paired original root/state/goal/history exactly equal')
            identities[key] = identity
        if summary['native_success'] is True: success += 1
    require(type(report['native_success_count']) is int and report['native_success_count'] == success, 'native results retained including false/zero, not object-only projection')
    terminal = report['errors'] == [] and all(r['status'] == 'returned' for r in report['rows'])
    if terminal:
        require(all(r['physical'] is not None and r['physical']['status'] in ('completed', 'observation_unavailable', 'environment_terminated')
            for r in report['rows']), 'engineering failures cannot become complete by dropping errors')
    require(report['status'] == ('calibration_recorded' if terminal else 'partial_or_failed') and
        report['first_error'] == (report['errors'][0] if report['errors'] else None), 'partial failures never silently promoted')
    require(claimed == set(report['artifact_sha256']), 'no orphan/missing/borrowed execution artifacts')
    return dict(schema=NAME+'.consume', status='saved_calibration_verified_not_admitted' if terminal else 'partial_saved_verified_not_complete',
        root_denominator=4, arm_denominator=8, rows=summaries, public_resources=resources,
        complete_branches=sum(r['full_horizon_completed'] is True for r in summaries),
        native_success_count=success, errors=report['errors'],
        inherited_predict_resources=p['inherited_resources'], new_neural_calls=0,
        backward_engine_calls=None, CUDA_ns=None, FLOPs=None,
        scientific_effect=None, generalization=None, physical_benefit_claim=False,
        labels_scope='saved original native-label arithmetic; no new native evaluator/physics or neural invocation',
        journal_scope='original fsynced JSONL plus externally sealed file hashes; NOT a newly invented hashchain', **FLAGS)


def consume(output, report_sha256, prepared, prepare_sha256, main_review_sha256):
    for value in (report_sha256, prepare_sha256, main_review_sha256): hash_value(value)
    sources = source_check(); output = Path(output).absolute()
    require(output.is_dir() and not output.is_symlink(), 'actual exclusive output directory')
    require(sha(output/'report.json') == report_sha256, 'external actual physical report SHA')
    report = read(output/'report.json'); p = verify(prepared, prepare_sha256)
    fields(report, 'schema status prepared prepare_sha256 rows errors first_error artifact_sha256 physical_calls contract root_denominator arm_denominator model_loaded new_neural_calls native_success_count P1_admission P2_admission known_root_calibration')
    require(report['prepared'] == str(Path(prepared).absolute()) and report['prepare_sha256'] == prepare_sha256, 'actual caller-specified prepare binding')
    require(type(report['artifact_sha256']) is dict and set(report['artifact_sha256'])|{'report.json'} == {f.name for f in output.iterdir()}, 'exact artifact census')
    require(sum(f.stat().st_size for f in output.iterdir()) <= assets.OUTPUT_CAP, 'actual output cap, not claimed hard preemption')
    for name, h in report['artifact_sha256'].items():
        require(Path(name).name == name and name not in ('', '.', '..'), 'flat artifact name')
        hash_value(h); require(sha(artifact(output, str(output/name))) == h, 'actual full artifact SHA')
    require(not (output/'report.json').is_symlink(), 'regular external report')
    execution = read(output/'execution.json'); require(execution['main_review_sha256'] == main_review_sha256, 'externally provided actual MAIN review reference')
    budget = lines(output/'physical_budget.jsonl') if 'physical_budget.jsonl' in report['artifact_sha256'] else []
    result = validate_saved(p, report, execution, budget, output)
    require(sha(output/'report.json') == report_sha256 and sha(prepared) == prepare_sha256 and
        all(sha(output/n) == h for n, h in report['artifact_sha256'].items()) and
        all(sha(f) == h for f, h in p['bindings'].items()) and source_check() == sources, 'unchanged final source/model/plan/artifact closure')
    result.update(report_sha256=report_sha256, prepare_sha256=prepare_sha256,
        main_review_sha256=main_review_sha256, consumer_sources=sources)
    return result


def write_receipt(path, value):
    path = Path(path).absolute(); raw = canonical(value)
    require(path.parent == ROOT/'results' and path.name.startswith(NAME) and not path.exists() and not path.is_symlink(), 'new exclusive own receipt')
    require(len(raw) <= assets.PREPARE_CAP, 'bounded small consumer receipt')
    with path.open('xb') as f:
        require(f.write(raw) == len(raw), 'complete receipt write'); f.flush(); os.fsync(f.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return dict(receipt=str(path), sha256=sha(path))


def run_calibration(prepared, expected, output, review, authorization):
    require(authorization == AUTH, 'MAIN-only known-root calibration physics; no scientific admission')
    verify(prepared, expected); hash_value(review)
    path = Path(output).absolute()
    require(path.parent == ROOT/'results' and path.name.startswith(NAME) and not path.exists() and not path.is_symlink(), 'new exclusive own calibration result')
    return assets.execute(prepared, expected, path, review)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='cmd', required=True)
    sub.add_parser('verify-inputs')
    a = sub.add_parser('prepare'); a.add_argument('--output', required=True)
    for name in ('verify', 'run-calibration', 'consume'):
        a = sub.add_parser(name); a.add_argument('--prepared', required=True); a.add_argument('--prepare-sha256', required=True)
        if name != 'verify': a.add_argument('--output', required=True); a.add_argument('--main-review-sha256', required=True)
        if name == 'run-calibration': a.add_argument('--authorization', required=True)
        if name == 'consume': a.add_argument('--report-sha256', required=True); a.add_argument('--receipt')
    args = parser.parse_args(argv)
    if args.cmd == 'verify-inputs':
        sources = source_check(); p = assets.build()
        result = dict(status='existing_saved_library_verified_no_dispatch', plans=len(p['plans']),
            root_denominator=4, arm_denominator=8, consumer_sources=sources, new_neural_calls=0, new_physics_steps=0, **FLAGS)
    elif args.cmd == 'prepare': source_check(); result = assets.prepare(args.output)
    elif args.cmd == 'verify': result = dict(status='verified_no_dispatch', plans=len(verify(args.prepared, args.prepare_sha256)['plans']), **FLAGS)
    elif args.cmd == 'run-calibration':
        report = run_calibration(args.prepared, args.prepare_sha256, args.output, args.main_review_sha256, args.authorization)
        result = dict(status=report['status'], report_sha256=sha(Path(args.output)/'report.json'), **FLAGS)
    else:
        result = consume(args.output, args.report_sha256, args.prepared, args.prepare_sha256, args.main_review_sha256)
        if args.receipt: result = write_receipt(args.receipt, result)
    print(canonical(result).decode())


if __name__ == '__main__': main()
