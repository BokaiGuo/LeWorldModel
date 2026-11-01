"""Saved-only P1 task prerequisite census, not a reachability solver/admission."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import p1_same_prefix_saved_audit_20261027_v1 as saved

NAME = 'p1_pusht_task_feasibility_20261027_v1'
ROOT = saved.ROOT
FLOOR = 200*1024**2
LIMIT = 200_000
EXTRA = (
    ('p1_witness_physical_20261027_v1_main01', '48df27e54eca33159c44a834ce0be1a13723553dc9d64dde996b7b4ca9eaf747', 'p1_witness_physical_20261027_v1_prepared01/prepare.json', 8),
    ('p1_sampling_physical_20261027_v1_main01', 'c3eef328bee57ce5d42b25fcf5a9935af2d2282c9de3f8caf46f2b35b53a2ef6', 'p1_sampling_physical_20261027_v1_prepared01/prepare.json', 4),
    ('p1_b8_physical_20261027_v1_main01', 'e94a6cbf55dd9b408d350e690d17bfdb9ca693b2ef360a12ab904024f272cd7a', 'p1_b8_physical_20261027_v1_prepare01.json', 4),
)
require = saved.require


def action_support(commands, cap, mean, std):
    a = np.asarray(commands)
    require(a.shape == (10, 2) and a.dtype.kind in 'fi' and np.isfinite(a).all(), 'finite raw10x2 required')
    require(np.array_equal(a, a.astype(np.float32)) and (np.abs(a) < 1).all(), 'exact float32 tanh raw action')
    require(np.isfinite(cap) and cap > 0, 'positive original packet cap')
    # Inversion is diagnostic: float32 tanh/atanh is not an exact inverse.
    norms = np.linalg.norm(np.arctanh(a).reshape(2, 10), axis=1)
    require((norms <= cap+2e-6).all(), 'original per-packet logit ball')
    mean = np.asarray(mean); std = np.asarray(std)
    require(mean.shape == std.shape == (2,) and np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all(), 'original normalizer')
    token = (a.astype(np.float32)-mean.astype(np.float32))/std.astype(np.float32)
    return dict(packet_logit_norms=norms.tolist(), raw_max_abs=float(np.abs(a).max()),
        normalized_min=float(token.min()), normalized_max=float(token.max()),
        nominal_target_offset_sum_px=float(100*np.linalg.norm(a, axis=1).sum()),
        interpretation='target offsets, NOT agent displacement bound or training-support certificate')


def trajectory(row, plan, packet, task):
    require(row['root_sha256'] == plan['root_sha256'] and row['selection_sha256'] == plan['selection_sha256'] and row['arm'] == plan['arm'], 'root/selection/arm identity')
    require(row['status'] == 'completed' and row['full_horizon'] is True and row['initial_exact'] is True and row['endpoint_rgb_fresh'] is True and
        row['first_error'] is None and row['secondary_errors'] == [] and row['attempted_steps'] == row['returned_steps'] == len(row['trace']) == 10, 'complete real H2 branch')
    require(np.array_equal(row['checkpoints'][0]['state'], packet['state']), 'original root state')
    support = action_support(plan['commands'], task['packet_logit_cap'], task['action_mean'], task['action_std'])
    state = saved.vec(packet['state']); goal = saved.vec(packet['goal']); diagnostics = []
    for i, t in enumerate(row['trace'], 1):
        require(t['step'] == i and t['command'] == plan['commands'][i-1] and t['attempted'] is t['returned'] is True and t['truncated'] is False, 'own legal command/chronology')
        require(np.array_equal(t['official_info']['goal_state'], goal), 'unchanged native goal')
        d = saved.metrics(goal, t['actual_state']); diagnostics.append(d)
        require(t['native_official_evaluation'] == saved.official(d) and t['reward'] == -d['native_formula']['state_distance'] and t['terminated'] is d['native_formula']['success'], 'native reward/success/termination unchanged')
    require(row['final_state'] == row['trace'][-1]['actual_state'] and row['official_evaluation'] == saved.official(diagnostics[-1]), 'actual endpoint identity')
    movements = [saved.delta(t['actual_state'], state) for t in row['trace']]
    return dict(root_sha256=row['root_sha256'], arm=row['arm'], support=support,
        native_successes=sum(d['native_formula']['success'] for d in diagnostics),
        wrapped_successes=sum(d['independent']['same_allxy_threshold_wrapped_success'] for d in diagnostics),
        native_angle_bug_steps=sum(d['native_formula']['negative_angle_term'] for d in diagnostics),
        endpoint_angle=diagnostics[-1]['independent']['wrapped_angle_error'],
        endpoint_allxy=diagnostics[-1]['independent']['all_xy_position_error'],
        endpoint_agent=diagnostics[-1]['independent']['agent_position_error'],
        endpoint_object=diagnostics[-1]['independent']['object_position_error'],
        max_observed_object_rotation=max(abs(x[4]) for x in movements),
        command_sha256=saved.digest(plan['commands']))


def chain(reader, path, expected, rows):
    path = reader.bind(path, expected); previous = None; closes = []; last = None; count = 0
    with path.open() as stream:
        for count, line in enumerate(stream, 1):
            require(line.endswith('\n'), 'complete journal line')
            e = json.loads(line); body = {k:v for k,v in e.items() if k != 'event_sha256'}
            require(e['index'] == count-1 and e['previous_sha256'] == previous and saved.digest(body) == e['event_sha256'], 'whole physical journal chain')
            previous = e['event_sha256']; last = e
            if e['kind'] == 'branch_closed': closes.append(e['receipt'])
    require(closes == rows and last is not None and last['kind'] == 'run_finished' and last['errors'] == [], 'durable complete branch receipts')
    return count


def audit():
    # Reuse the independent full 738-PNG/34196-event validator, not the producer.
    independent = saved.audit()
    reader = saved.Reader(); p, actual, packets = saved.load_inputs(reader)
    parent = reader.read(ROOT/'results/p1_witness_physical_20261027_v1_prepared01/prepare.json')
    wp = Path(parent['inputs']['arguments']['witness_prepared'])
    seal = reader.read(wp/'prepare.json', parent['inputs']['arguments']['witness_prepare_sha'])
    protocol = reader.read(wp/'protocol.json', seal['protocol_sha256'])
    tasks = {x['root_sha256']:x['task'] for x in protocol['inputs']['tasks']}
    source = reader.read(ROOT/'results/p1_conditional_calibration2_main_20261027_v1/sourcepack.json', '728f14cb83c4fcd0ea4f5348bad2bff5b5dd372ab0185b6e6b3a26dbfda58057')
    require(source['plan'] == p['inputs']['source_plan'] and source['status'] == 'completed_calibration_source_only', 'original generation contract')
    rows = []; batches = []
    inputs = [(saved.ACTUAL.name, p, actual, saved.REPORT_SHA)]
    for name, hashed, prepare, n in EXTRA:
        r = reader.read(ROOT/'results'/name/'report.json', hashed)
        pr = reader.read(ROOT/'results'/prepare, r['prepare_sha256'])
        require(r['status'] == 'physical_calibration_recorded' and r['errors'] == [] and len(r['rows']) == n and r['neural_calls'] == 0, 'complete supplemental physical report')
        chain(reader, ROOT/'results'/name/'journal.jsonl', r['journal_sha256'], r['rows'])
        inputs.append((name, pr, r, hashed))
    for name, prepared, report, hashed in inputs:
        plans = prepared['inputs']['plans']; require(len(plans) == len(report['rows']), 'all plans retained')
        require({x['root_sha256'] for x in report['rows']} == set(tasks), 'only original two roots')
        for plan, row in zip(plans, report['rows'], strict=True):
            root = row['root_sha256']; rows.append(dict(batch=name, **trajectory(row, plan, packets[root], tasks[root])))
        batches.append(dict(name=name, report_sha256=hashed, branches=len(report['rows']), public_steps=10*len(report['rows'])))
    roots = []
    for root, packet in packets.items():
        own = [r for r in rows if r['root_sha256'] == root]; task = tasks[root]
        slot = next(x for x in source['slots'] if x['metadata']['root_sha256'] == root)
        attempts = [a for prop in slot['proposals'] for a in prop['attempts'] if a.get('selected')]
        require(len(attempts) == 1 and np.array_equal(attempts[0]['state'], packet['state']) and np.array_equal(attempts[0]['goal'], packet['goal']), 'selected original source arrays')
        initial = saved.metrics(packet['goal'], packet['state'])
        groups = []
        for role in saved.ROLES:
            rs = [x for x in own if x['batch'] == saved.ACTUAL.name and x['arm'].startswith(role+'_probe_')]
            require(len(rs) == 41, 'all sameprefix probes')
            groups.append(dict(role=role, branches=41, min_endpoint_angle=min(x['endpoint_angle'] for x in rs),
                min_endpoint_allxy=min(x['endpoint_allxy'] for x in rs), max_observed_object_rotation=max(x['max_observed_object_rotation'] for x in rs)))
        roots.append(dict(root_sha256=root, state=packet['state'].tolist(), goal=packet['goal'].tolist(),
            original_source=slot['metadata']['source'], selected_attempt=slot['metadata']['selected_attempt'], initial=initial,
            packet_logit_cap=task['packet_logit_cap'], action_mean=task['action_mean'], action_std=task['action_std'],
            branches=len(own), distinct_saved_command_sequences=len({x['command_sha256'] for x in own}),
            success_branches=sum(x['native_successes'] > 0 for x in own), wrapped_success_branches=sum(x['wrapped_successes'] > 0 for x in own),
            negative_native_angle_steps=sum(x['native_angle_bug_steps'] for x in own),
            min_saved_endpoint_angle=min(x['endpoint_angle'] for x in own), min_saved_endpoint_allxy=min(x['endpoint_allxy'] for x in own),
            normalized_action_range=[min(x['support']['normalized_min'] for x in own), max(x['support']['normalized_max'] for x in own)], sameprefix_groups=groups))
    own_files = [Path(__file__), ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md')]
    for f in own_files: reader.bind(f)
    reader.recheck()
    return dict(schema=NAME+'.report', status='saved_task_prerequisites_not_established', roots=roots, batches=batches,
        source_recipe=source['plan']['recipe'], source_domain=source['plan']['domain'], branches=len(rows),
        sameprefix_validation=dict(report_sha256=independent['actual_report_sha256'], branches=independent['branches'], PNGs=independent['PNGs'], events=independent['resources']['events']),
        observed_action_support=saved.digest(rows), bindings=reader.bindings,
        physical_reach_witness_found=any(x['success_branches'] for x in roots),
        global_unreachability_proven=False, training_distribution_support_verified=False,
        direct_plateau_verified=False, P1_admission=False, new_neural_calls=0, new_physics_steps=0,
        scope='262 recorded branches on TWO reused roots; finite probes not a global action search; inverse-tanh tolerance only',
        caveats=['No torque/speed upper bound: angle gap alone cannot prove infeasible.',
            'Native goal includes agent AND object; removing agent changes task.',
            'Normalizer constants and action legality do not certify checkpoint training distribution.',
            'Full independent PNG/journal audit for sameprefix; supplemental journal/report/command/state arithmetic only.',
            'No KKT certificate or independently frozen plateau/target threshold established here.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path)
    args = parser.parse_args(); result = audit(); raw = (saved.core.canonical(result)+'\n').encode()
    if args.output is not None:
        require(not args.output.exists() and len(raw) < LIMIT and shutil.disk_usage(args.output.parent).free >= FLOOR+len(raw), 'NEW small output plus200MiB reserve')
        with args.output.open('xb') as stream: stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        print(json.dumps(dict(status=result['status'], path=str(args.output.resolve()), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())))
    else: print(raw.decode(), end='')


if __name__ == '__main__': main()
