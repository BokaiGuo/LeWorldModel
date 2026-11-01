"""Saved paired task/cost arithmetic after a complete strict physical handoff.

No model, environment, renderer, or experiment execution entrypoint.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import agent_p3_budget_physical_handoff_20261028_v1 as bridge

NAME = 'agent_p3_budget_paired_analysis_20261028_v1'
ROOT, require, sha, digest = bridge.ROOT, bridge.require, bridge.sha, bridge.digest
FLAGS = dict(bridge.FLAGS)
CAP = 16*1024**2
HANDOFF_SHA = '5c39919c632fd8b7feb8b22ad310900e424d5bcb226f1afd99a6a5b89b01fc9a'
PINS = dict(zip(map(str, bridge.own_files()), (
    '6a325fc5a24204db553c11dffd211ef298b2d54566ed8ae5a7848b2761e48c5c',
    'f353da1df1fd2bb0099ac60095b6c208fe48ac72d55159db2060fe712bc04025',
    '56ad626df0367275b93acb53aedd51743d37dc4d14b2add3aee9a79e95ccdc0e'), strict=True))
METRICS = ('native7_distance', 'native7_reward', 'joint_xy_error', 'object_position_error',
           'wrapped_angle_error', 'native_angle_term', 'velocity_error', 'native_success')


def own_files():
    return [Path(__file__).resolve(), ROOT/f'tests/{NAME}_test.py', ROOT/f'innerdocs/{NAME}.md']


def known(values):
    """No imputation; known-only means are explicitly conditional summaries."""
    xs = [float(x) for x in values if x is not None]
    require(all(np.isfinite(x) for x in xs), 'finite known observations')
    return dict(denominator=len(values), known=len(xs), missing=len(values)-len(xs),
                known_sum=sum(xs) if xs else None,
                known_only_mean=sum(xs)/len(xs) if xs else None,
                full_denominator_mean=sum(xs)/len(xs) if xs and len(xs)==len(values) else None)


def task_metrics(frame):
    # Frozen native metric is arithmetic only. Recompute from saved state/goal,
    # never from model Q or final native7 reward as a success proxy.
    metrics = bridge.physical.metric(frame['goal'], frame['state'])
    return {k: metrics['native_distance'] if k == 'native7_distance' else metrics[k] for k in METRICS}


def resource_total(receipts):
    require(receipts, 'at least one actually paid receipt')
    axes = set(receipts[0]['attempted'])
    for r in receipts:
        for kind in ('attempted', 'completed'):
            require(set(r[kind]) == axes and all(type(x) is int and x >= 0 for x in r[kind].values()),
                    'complete literal nonnegative integer resource axes')
        require(all(r['completed'][k] <= r['attempted'][k] for k in axes), 'return cannot precede attempt')
    return {kind: {k: sum(r[kind][k] for r in receipts) for k in sorted(axes)}
            for kind in ('attempted', 'completed')}


def summarize(rows):
    return dict(slot_denominator=len(rows), measured_task_slots=sum(r['final'] is not None for r in rows),
        native_stopped_slots=sum(r['native_earlystop'] is True for r in rows),
        before_step5_stopped_slots=sum(r['own_returned'] is not None and r['own_returned'] < 5 for r in rows),
        task={k: known([None if r['final'] is None else r['final'][k] for r in rows]) for k in METRICS},
        improvement={k: known([None if r['improvement'] is None else r['improvement'][k] for r in rows])
                     for k in METRICS if k != 'native_success'},
        resources=None if not any(r['resources'] is not None for r in rows) else
            resource_total([r['resources'] for r in rows if r['resources'] is not None]),
        measured_resource_slots=sum(r['resources'] is not None for r in rows),
        unknown_resource_slots=sum(r['resources'] is None for r in rows),
        route_census={k: sum(r['route'] == k for r in rows) for k in
                      ('reused_saved', 'pending_new_physics', 'paid_refused', 'not_reached')})


def assemble(p, host_report, full_receipt):
    """Pure arithmetic; material() authenticates the complete upstream closure."""
    h = p['handoff']; source_rows = h['rows']; paid = full_receipt['full_rows']
    require(p['schema'] == bridge.NAME+'.prepare' and len(source_rows) == len(paid) == 48 and
            h['fixed_denominator'] == 48 and h['child_denominator'] == 24 and
            h['independent_roots'] == 2 and all(p[k] is h[k] is False for k in FLAGS),
            'full closed-gate24child48slot handoff')
    require(full_receipt['actual_resources'] == p['host_actual_resources'] and
            bridge.canonical(paid) == bridge.canonical(host_report['session']['rows']),
            'entire strict actual resources and same full host rows')
    rows = []
    for i, (r, paid_row) in enumerate(zip(source_rows, paid, strict=True)):
        plan = r['runner_plan']; child = plan['child']; mode = bridge.contract.MODES[i % 2]
        require(r['ordinal'] == i and r['mode'] == plan['mode'] == paid_row['mode'] == mode and
                r['child_id'] == child['child_id'] == paid_row['child_id'] and
                child['root_slot'] in (0, 1) and r['paid_status'] == paid_row['status'] and
                r['decision'] == paid_row['decision'], 'same ordinal/mode/child/paid decision ancestry')
        route = r['route']; resources = None; initial = final = improvement = None
        returned = earlystop = physical_status = None; physics_ref = None
        if route != 'not_reached':
            require(paid_row['signal_resources'] is not None, 'refusal retains independent paid signal')
            receipts = [paid_row['signal_resources']]
            if paid_row['compiler'] is not None: receipts.append(paid_row['compiler']['resource_receipt'])
            resources = resource_total(receipts)
        else:
            require(paid_row['status'] == 'not_reached' and paid_row['signal_resources'] is None and
                    paid_row['compiler'] is None and r['reuse_evidence'] is None, 'unreached remains missing')
        if route == 'reused_saved':
            e = r['reuse_evidence']; old = e['full_old_row']; selected = paid_row['compiler']['selected']
            require(e['report_sha256'] == bridge.saved.ACTUAL_REPORT_SHA and e['fresh_execution'] is False and
                    e['original_ordinal'] == old['ordinal'] == i and old['initial_exact'] is True and
                    old['error'] is None and old['status'] in ('returned5', 'native_stopped') and
                    r['selection_digest'] == selected['selection_digest'] and
                    r['command'] == bridge.raw_identity(selected['raw5'], 5) and
                    e['command_sha256'] == r['command']['sha256'], 'exact same-slot selected command and saved physics')
            bridge.physical.same_frame(old['boundary'], plan['source_frames'][-1])
            initial = task_metrics(old['boundary']); final = task_metrics(old['last_observation'])
            require(old['final_metrics'] == bridge.physical.metric(old['last_observation']['goal'],
                    old['last_observation']['state']) and old['native_success'] is final['native_success'] and
                    old['terminated'] is final['native_success'] and old['final_state'] == old['last_observation']['state'],
                    'independent final native metrics/termination, not native7 success threshold')
            returned = old['own_returned']; earlystop = old['terminated'] or old['truncated']; physical_status = old['status']
            require(type(returned) is int and 1 <= returned <= 5 and
                    (returned == 5 or earlystop) and (old['status'] == 'native_stopped') == earlystop,
                    'real native earlystop without padding')
            improvement = {k: (final[k]-initial[k] if k == 'native7_reward' else initial[k]-final[k])
                           for k in METRICS if k != 'native_success'}
            physics_ref = dict(report_sha256=e['report_sha256'], prepare_sha256=e['prepare_sha256'],
                ordinal=i, plan_sha256=e['original_plan_sha256'], full_row_sha256=digest(old), fresh=False)
        elif route == 'paid_refused':
            require(paid_row['compiler'] is None and r['reuse_evidence'] is None and plan['selected'] is None,
                    'refused paid signal cannot fabricate task result')
        elif route == 'pending_new_physics':
            require(paid_row['compiler'] is not None and r['reuse_evidence'] is None, 'new physics remains missing')
        else: require(route == 'not_reached', 'known route')
        rows.append(dict(ordinal=i, child_id=child['child_id'], root_slot=child['root_slot'], mode=mode,
            boundary=child['boundary'], source_mode=child['source_mode'], prefix_arm=child['prefix_arm'],
            episode_id=child['episode_id'], route=route,
            missing_reason=None if final is not None else route, source_ancestry=deepcopy(child['ancestry']),
            context_sha256=child['context_sha256'], start_sha256=r['start_sha256'], goal_sha256=r['goal_sha256'],
            command_sha256=None if r['command'] is None else r['command']['sha256'],
            selection_digest=r['selection_digest'], paid_status=r['paid_status'], decision=deepcopy(r['decision']),
            resources=resources, paid_ledger_sha256=digest(dict(signal=paid_row['signal_resources'],
                compiler=None if paid_row['compiler'] is None else paid_row['compiler']['resource_receipt'])),
            initial=initial, final=final, improvement=improvement, own_returned=returned,
            native_earlystop=earlystop, physical_status=physical_status, physical_reference=physics_ref,
            new_physics_resources=None, exclusive_method_wall_seconds=None))
    pairs = []
    for j in range(24):
        f, a = rows[2*j:2*j+2]
        require(f['child_id'] == a['child_id'] and f['root_slot'] == a['root_slot'] and
                f['context_sha256'] == a['context_sha256'] and f['source_ancestry'] == a['source_ancestry'] and
                f['start_sha256'] == a['start_sha256'] and f['goal_sha256'] == a['goal_sha256'],
                'paired identical child/task/source/start/goal/context')
        if f['initial'] is not None and a['initial'] is not None:
            require(f['initial'] == a['initial'], 'identical paired native initial task')
        pairs.append(dict(child_id=f['child_id'], root_slot=f['root_slot'],
            routes=[f['route'], a['route']],
            task_delta_adaptive_minus_fixed=None if f['final'] is None or a['final'] is None else
                {k: float(a['final'][k])-float(f['final'][k]) for k in METRICS},
            actual_delta_adaptive_minus_fixed=None if f['resources'] is None or a['resources'] is None else
                {kind: {k: a['resources'][kind][k]-f['resources'][kind][k] for k in f['resources'][kind]}
                 for kind in ('attempted', 'completed')}))
    modes = {mode: summarize([r for r in rows if r['mode'] == mode]) for mode in bridge.contract.MODES}
    require(all(modes[mode]['resources'] == full_receipt['actual_resources'][mode] for mode in modes),
            'all actual paid signal plus compiler resources reconcile, no omitted refusals')
    clusters = []
    for root in (0, 1):
        rp = [x for x in pairs if x['root_slot'] == root]
        clusters.append(dict(root_slot=root, child_denominator=len(rp), independent_root_count=1,
            modes={mode: summarize([x for x in rows if x['root_slot'] == root and x['mode'] == mode]) for mode in modes},
            paired_task_delta={k: known([None if x['task_delta_adaptive_minus_fixed'] is None else
                x['task_delta_adaptive_minus_fixed'][k] for x in rp]) for k in METRICS}))
    delta = {kind: {k: modes['adaptive']['resources'][kind][k]-modes['fixed']['resources'][kind][k]
                   for k in modes['fixed']['resources'][kind]} for kind in ('attempted', 'completed')}
    return dict(schema=NAME+'.analysis', status='saved_paired_descriptive_not_admitted',
        fixed_denominator=48, child_denominator=24, independent_known_roots=2, rows=rows, pairs=pairs,
        modes=modes, root_clusters=clusters,
        paired_task_delta={k: known([None if pair['task_delta_adaptive_minus_fixed'] is None else
            pair['task_delta_adaptive_minus_fixed'][k] for pair in pairs]) for k in METRICS},
        equal_root_weight_paired_delta={k: known([c['paired_task_delta'][k]['known_only_mean'] for c in clusters])
                                       for k in METRICS},
        resource_delta_adaptive_minus_fixed=delta, actual_vectors_equal=all(v == 0 for v in delta['completed'].values()),
        resource_savings=dict(lower_axes=[k for k,v in delta['completed'].items() if v < 0],
            higher_axes=[k for k,v in delta['completed'].items() if v > 0], FLOPs=None,
            uniform_actual_savings=all(v <= 0 for v in delta['completed'].values()) and any(v < 0 for v in delta['completed'].values())),
        wall=dict(host_panel_seconds=host_report['wall_seconds'], scope=bridge.host.profile()['wall_scope'],
            fixed_seconds=None, adaptive_seconds=None, historical_physics_not_new_cost=True, nested_not_added=True),
        new_neural_calls=0, new_physics_calls=0,
        C3_missing=['actual vectors unequal; shared caps are not equal work',
            'five paid refusals and two unreached slots lack task outcomes',
            'two known roots only; dependent child/repeated trajectories are not independent roots',
            'reused historical native evidence is not fresh physical execution',
            'no measured FLOPs or exclusive per-method wall'], **FLAGS)


def material(prepared, expected):
    path = bridge.contract.absolute(prepared); bridge.core.hash_value(expected)
    require(sha(path) == expected, 'external handoff SHA before upstream saved consumer')
    bridge.contract.check_sources(PINS)
    p = bridge.verify(path, expected)  # complete original arrays/native closure; NEVER physics
    require(p['authority']['reuse_policy'] == 'strict_saved', 'strict saved physical evidence required')
    _, report, receipt = bridge.load_host(p['authority'])  # persisted full strict; no host.consume
    result = assemble(p, report, receipt)
    sources = deepcopy(p['sources']); bridge.contract.merge(sources, PINS)
    bridge.contract.merge(sources, {str(f): sha(f) for f in own_files()})
    bridge.contract.merge(sources, {str(path): expected})
    bridge.contract.check_sources(sources)
    require(sha(path) == expected, 'handoff unchanged after saved analysis')
    result.update(prepared=str(path), prepare_sha256=expected, sources=sources,
                  host_report_sha256=p['authority']['report_sha256'], strict_receipt_sha256=p['authority']['receipt_sha256'])
    require(len(bridge.canonical(result)) <= CAP, 'bounded full analysis output')
    return result


def verify(path, expected):
    path = bridge.contract.absolute(path); bridge.core.hash_value(expected)
    require(sha(path) == expected, 'external actual analysis SHA')
    result = bridge.READ(path, CAP); bridge.contract.check_sources(result['sources'])
    rebuilt = material(result['prepared'], result['prepare_sha256'])
    require(bridge.canonical(rebuilt) == bridge.canonical(result) and sha(path) == expected,
            'complete deterministic paired analysis reconstruction')
    return dict(status='saved_paired_analysis_verified_not_admitted', analysis_sha256=expected,
                slots=48, children=24, independent_roots=2, **FLAGS)


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('analyze'); a.add_argument('--prepared', required=True)
    a.add_argument('--prepare-sha256', required=True); a.add_argument('--output', required=True)
    a = sub.add_parser('verify'); a.add_argument('--analysis', required=True); a.add_argument('--analysis-sha256', required=True)
    args = parser.parse_args()
    if args.cmd == 'analyze':
        output = bridge.contract.absolute(args.output)
        require(output.parent == ROOT/'results' and output.name.startswith(NAME+'_') and not output.exists(),
                'NEW own analysis output only, reject reentry before upstream validation')
        result = bridge.physical.write_new(output, material(args.prepared, args.prepare_sha256), cap=CAP)
    else: result = verify(args.analysis, args.analysis_sha256)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__': main()
