"""Saved native7 squared-error accounting, not a native prediction decoder.

Preserves all 24 pairs/48 slots and two known-root clusters. No model/physics run.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import agent_p3_budget_paired_analysis_20261028_v1 as paired

NAME = 'agent_p3_task_resource_decomposition_20261028_v1'
ROOT, bridge = paired.ROOT, paired.bridge
require, sha, digest = paired.require, paired.sha, paired.digest
FLAGS = dict(paired.FLAGS)
CAP = 16*1024**2
PAIRED_SHA = '55d5ab8c49d31b7d0af3ddcb803f114d272c220c6ab9524076e32df035223288'
PINS = dict(zip(map(str, paired.own_files()), (
    '323c4c156256c08816f32c2a6bc93a4f62938ed1f252e2457151ab8bff261a8f',
    'b8e6d2847e2b7c8fbd7d1768333ae2ca193864e3e005afaff1d24a9729922551',
    '42475d3ec5a95bc7f9fdf63f8388a47e696f9eed927b00b19b2cb79f62f1cde9'), strict=True))
COMPONENTS = ('joint_xy_squared', 'raw_angle_squared', 'velocity_squared')
SQUARED = ('native7_squared', *COMPONENTS, 'agent_xy_squared', 'object_xy_squared')


def own_files():
    return [Path(__file__).resolve(), ROOT/f'tests/{NAME}_test.py', ROOT/f'innerdocs/{NAME}.md']


def roundoff(a, b):
    # Fixed float64 arithmetic guard only, NOT a scientific threshold or fit.
    return abs(a-b) <= 16*np.finfo(np.float64).eps*max(1., abs(a), abs(b))


def decompose(frame):
    state = bridge.core.finite(frame['state'], (7,)).astype(np.float64)
    goal = bridge.core.finite(frame['goal'], (7,)).astype(np.float64)
    d = state-goal
    sq = d*d
    result = dict(native7_squared=float(np.dot(d, d)),
        joint_xy_squared=float(np.sum(sq[:4])), raw_angle_squared=float(sq[4]),
        velocity_squared=float(np.sum(sq[5:])), agent_xy_squared=float(np.sum(sq[:2])),
        object_xy_squared=float(np.sum(sq[2:4])), residual7=d.tolist(),
        native7_distance=float(np.linalg.norm(d)), joint_xy_error=float(np.linalg.norm(d[:4])),
        object_position_error=float(np.linalg.norm(d[2:4])),
        goal_sha256=digest(frame['goal']), observation_sha256=digest(frame))
    require(all(np.isfinite(result[k]) for k in SQUARED) and
            roundoff(result['native7_squared'], sum(result[k] for k in COMPONENTS)) and
            roundoff(result['joint_xy_squared'], result['agent_xy_squared']+result['object_xy_squared']),
            'finite float64 exact-axis squared decomposition')
    # Native7 uses the literal saved angle difference; neither wrapped angle nor
    # the native success min-angle term can replace its fifth squared component.
    result['native7_component_sum'] = sum(result[k] for k in COMPONENTS)
    return result


def model_view(task, child, compiler):
    """Original Q arithmetic ONLY. Never turn latent features into native state."""
    bridge.core.validate_task(task)
    require(len(task['goal']) == 192 and len(task['projection']) == 192,
            'frozen full192 projected latent objective, not native7 coordinates')
    result = dict(task_binding_sha256=digest(task['binding']),
        latent_goal_sha256=task['binding']['goal_sha256'], projection_sha256=task['binding']['projection_sha256'],
        objective='mean((projection @ (latent_goal - endpoint.latent_history[-1]))**2)',
        objective_dimension=192, selected_horizon_packets=None, physical_execution_steps=None,
        parent_latent_Q=None, prefix5_latent_Q=None, selected_terminal_latent_Q=None,
        prefix5_latent_Q_gain=None, terminal_latent_Q_gain=None,
        native_goal_prediction=None, native_joint_xy_prediction=None, native_velocity_prediction=None,
        native_mapping_status='unknown_no_bound_latent_to_native7_component_mapping',
        native_mapping_reason='frozen task/context/Q ABI contains projected latent arrays, no native7 decoder or component prediction contract',
        Q_is_native_reward=False, Q_is_native_success=False)
    if child['status'] == 'not_reached':
        result['selected_status'] = 'not_reached'; return result
    result['parent_latent_Q'] = bridge.core.model_cost(task, child['context'])
    if compiler is None:
        result['selected_status'] = 'paid_refused'; return result
    selected = compiler['selected']; candidates = compiler['candidates']
    winner = min(candidates, key=lambda c: (c['cost'], c['index']))
    require(selected['index'] == winner['index'] and selected['prefix'] == winner['whole'][:1] and
        bridge.raw_identity(selected['raw5'], 5) == bridge.raw_identity(winner['actions'][:5], 5),
        'strict selected candidate same actual raw5 and prefix endpoint')
    terminal = bridge.core.model_cost(task, winner['whole'][-1])
    prefix = bridge.core.model_cost(task, selected['prefix'][-1])
    require(terminal == selected['terminal_cost'] == winner['cost'], 'saved original latent Q terminal cost')
    result.update(selected_status='selected', selected_horizon_packets=len(winner['whole']),
        selected_terminal_latent_Q=terminal, prefix5_latent_Q=prefix,
        prefix5_latent_Q_gain=result['parent_latent_Q']-prefix,
        terminal_latent_Q_gain=result['parent_latent_Q']-terminal,
        selected_candidate_index=selected['index'], selection_digest=selected['selection_digest'],
        selected_prefix_sha256=digest(selected['prefix']),
        selected_terminal_context_sha256=digest(winner['whole'][-1]))
    return result


def summarize_pairs(pairs):
    measured = [p for p in pairs if p['squared_delta_adaptive_minus_fixed'] is not None]
    component = {k: paired.known([None if p['squared_delta_adaptive_minus_fixed'] is None else
        p['squared_delta_adaptive_minus_fixed'][k] for p in pairs]) for k in SQUARED}
    if measured:
        require(roundoff(component['native7_squared']['known_sum'],
            sum(component[k]['known_sum'] for k in COMPONENTS)), 'same known-pair aggregate squared accounting')
    return dict(pair_denominator=len(pairs), known_pairs=len(measured), missing_pairs=len(pairs)-len(measured),
        squared_delta_adaptive_minus_fixed=component,
        distance_delta_adaptive_minus_fixed={k: paired.known([None if p['distance_delta_adaptive_minus_fixed'] is None else
            p['distance_delta_adaptive_minus_fixed'][k] for p in pairs])
            for k in ('native7_distance', 'joint_xy_error', 'object_position_error')},
        native7_down_jointXY_up_known_pairs=sum(p['native7_down_jointXY_up'] is True for p in pairs),
        velocity_offsets_jointXY_known_pairs=sum(p['velocity_offsets_jointXY'] is True for p in pairs),
        native_prediction_comparison=dict(native_goal='unknown', jointXY='unknown', velocity='unknown',
            reason='no bound native component prediction mapping; latent-Q signs are not native predictions'))


def assemble(analysis, handoff, hp, report, receipt):
    require(analysis['schema'] == paired.NAME+'.analysis' and analysis['fixed_denominator'] == 48 and
        analysis['child_denominator'] == 24 and analysis['independent_known_roots'] == 2 and
        all(analysis[k] is False for k in FLAGS), 'full closed-gate paired analysis authority')
    # Reconcile to the SAME already authenticated full host/physical objects.
    # This is saved arithmetic, not a new model evaluation or physical trial.
    rebuilt = paired.assemble(handoff, report, receipt)
    require(all(bridge.canonical(analysis[k]) == bridge.canonical(v) for k, v in rebuilt.items()),
            'exact original full paired arithmetic, not altered rows or summary')
    require(len(handoff['handoff']['rows']) == len(analysis['rows']) == 48, 'all48 slots retained')
    rows = []
    for i, (prior, source) in enumerate(zip(analysis['rows'], handoff['handoff']['rows'], strict=True)):
        child = source['runner_plan']['child']; paid = receipt['full_rows'][i]
        task = hp['wiring']['tasks'][child['root_slot']]
        require(prior['ordinal'] == source['ordinal'] == i and prior['child_id'] == child['child_id'] == paid['child_id'] and
            prior['mode'] == source['mode'] == paid['mode'] and prior['route'] == source['route'] and
            prior['context_sha256'] == child['context_sha256'] and prior['source_ancestry'] == child['ancestry'],
            'exact child/method/source/context correspondence')
        initial = final = None
        if prior['final'] is not None:
            old = source['reuse_evidence']['full_old_row']; ref = prior['physical_reference']
            require(ref['full_row_sha256'] == digest(old) and ref['report_sha256'] == bridge.saved.ACTUAL_REPORT_SHA and
                ref['ordinal'] == old['ordinal'] == i and source['goal_sha256'] == digest(old['last_observation']['goal']) and
                old['boundary']['goal'] == old['last_observation']['goal'], 'exact same native goal and full original physical row')
            initial, final = decompose(old['boundary']), decompose(old['last_observation'])
            for label, value in (('initial', initial), ('final', final)):
                require(all(value[k] == prior[label][k] for k in ('native7_distance', 'joint_xy_error', 'object_position_error')),
                        'same original native scalar metrics from independent squared decomposition')
        q = model_view(task, child, paid['compiler'])
        q['physical_execution_steps'] = prior['own_returned']
        q['same_five_step_model_and_physical_horizon'] = None if prior['own_returned'] is None else prior['own_returned'] == 5
        rows.append(dict(ordinal=i, child_id=prior['child_id'], root_slot=prior['root_slot'], mode=prior['mode'],
            route=prior['route'], boundary=prior['boundary'], source_mode=prior['source_mode'], prefix_arm=prior['prefix_arm'],
            source_ancestry=deepcopy(prior['source_ancestry']), context_sha256=prior['context_sha256'],
            start_sha256=prior['start_sha256'], command_sha256=prior['command_sha256'],
            selection_digest=prior['selection_digest'], native_goal_sha256=prior['goal_sha256'],
            physical_reference=deepcopy(prior['physical_reference']), own_returned=prior['own_returned'],
            native_earlystop=prior['native_earlystop'], native_success=None if prior['final'] is None else prior['final']['native_success'],
            initial=initial, final=final, missing_reason=prior['missing_reason'],
            initial_minus_final_squared=None if final is None else {k: initial[k]-final[k] for k in SQUARED},
            actual_resources=deepcopy(prior['resources']), model_Q=q))
    pairs = []
    for j in range(24):
        f, a = rows[2*j:2*j+2]; original_pair = analysis['pairs'][j]
        require(f['child_id'] == a['child_id'] == original_pair['child_id'] and
                f['root_slot'] == a['root_slot'] and f['native_goal_sha256'] == a['native_goal_sha256'] and
                f['start_sha256'] == a['start_sha256'], 'same native paired origin and goal, never nearest outcome join')
        delta = distances = contradiction = velocity_offset = None
        if f['final'] is not None and a['final'] is not None:
            delta = {k: a['final'][k]-f['final'][k] for k in SQUARED}
            require(roundoff(delta['native7_squared'], sum(delta[k] for k in COMPONENTS)),
                    'paired native7 squared delta equals jointXY+rawangle+velocity delta')
            distances = {k: a['final'][k]-f['final'][k] for k in
                ('native7_distance', 'joint_xy_error', 'object_position_error')}
            require(all(distances[k] == original_pair['task_delta_adaptive_minus_fixed'][k] for k in distances),
                    'same all-known original paired distance comparison')
            contradiction = delta['native7_squared'] < 0 and delta['joint_xy_squared'] > 0
            velocity_offset = contradiction and delta['velocity_squared'] < 0
        qdelta = None
        if f['model_Q']['prefix5_latent_Q'] is not None and a['model_Q']['prefix5_latent_Q'] is not None:
            qdelta = a['model_Q']['prefix5_latent_Q']-f['model_Q']['prefix5_latent_Q']
        pairs.append(dict(child_id=f['child_id'], root_slot=f['root_slot'], routes=[f['route'], a['route']],
            squared_delta_adaptive_minus_fixed=delta, distance_delta_adaptive_minus_fixed=distances,
            native7_down_jointXY_up=contradiction, velocity_offsets_jointXY=velocity_offset,
            prefix5_latent_Q_delta_adaptive_minus_fixed=qdelta,
            Q_predicted_native_jointXY_delta=None, Q_predicted_native_velocity_delta=None,
            Q_native_comparison_status='unknown_no_bound_native_component_prediction'))
    clusters = [dict(root_slot=root, independent_root_count=1,
        summary=summarize_pairs([p for p in pairs if p['root_slot'] == root])) for root in (0, 1)]
    return dict(schema=NAME+'.analysis', status='saved_axis_decomposition_not_admitted',
        fixed_denominator=48, child_denominator=24, pair_denominator=24, independent_known_roots=2,
        axis_spec=dict(jointXY=[0,1,2,3], raw_angle=[4], velocity=[5,6],
            residual='literal state7 minus SAME saved goal7, including nonzero goal velocity',
            scaling='original mixed native units; no normalization, refit or weighting'),
        rows=rows, pairs=pairs, all_pairs=summarize_pairs(pairs), root_clusters=clusters,
        root_equal_weight_known_subset_squared_delta={k: paired.known([
            c['summary']['squared_delta_adaptive_minus_fixed'][k]['known_only_mean'] for c in clusters]) for k in SQUARED},
        root_equal_weight_scope='means of known pairs within each root, NOT completed 24pair outcomes or 24 independent roots',
        actual_resources={m: deepcopy(analysis['modes'][m]['resources']) for m in bridge.contract.MODES},
        actual_resource_coverage={m: dict(measured_slots=analysis['modes'][m]['measured_resource_slots'],
            missing_slots=analysis['modes'][m]['unknown_resource_slots'], slot_denominator=24)
            for m in bridge.contract.MODES},
        resource_delta_adaptive_minus_fixed=deepcopy(analysis['resource_delta_adaptive_minus_fixed']),
        actual_vectors_equal=analysis['actual_vectors_equal'], FLOPs=None,
        explanation=dict(identity='native7_squared = jointXY_squared + literal_raw_angle_squared + velocity_squared',
            accounting='paired delta is adaptive minus fixed on the SAME known pairs; negative native7 can coexist with positive jointXY if other squared terms offset it',
            native_success='original jointXY+native-angle criterion; native7 reward and objectXY alone are not task success',
            not_causal=True, no_latent_to_native_substitution=True, no_Q_fitting=True,
            earlystop='use actual last returned native observation, never pad to model five-step endpoint',
            unknown='five paid refusals/two unreached slots keep missing native outcomes; no zero imputation',
            C3='not established: unequal actual resources, missing pairs, known-root reused evidence and no native Q mapping'),
        new_neural_calls=0, new_physics_calls=0, **FLAGS)


def material(path, expected):
    path = bridge.contract.absolute(path); bridge.core.hash_value(expected)
    require(sha(path) == expected == PAIRED_SHA, 'external actual complete saved01 SHA before any consumer')
    bridge.contract.check_sources(PINS)
    paired.verify(path, expected)  # full frozen arrays/native source closure, no experiment
    analysis = bridge.READ(path, paired.CAP)
    prepared = bridge.contract.absolute(analysis['prepared'])
    require(sha(prepared) == analysis['prepare_sha256'], 'same original handoff prepare edge')
    p = bridge.READ(prepared, 64*1024**2)
    hp, report, receipt = bridge.load_host(p['authority'])  # persisted full strict, not host.consume
    result = assemble(analysis, p, hp, report, receipt)
    sources = deepcopy(analysis['sources']); bridge.contract.merge(sources, PINS)
    bridge.contract.merge(sources, {str(f): sha(f) for f in own_files()})
    bridge.contract.merge(sources, {str(path): expected})
    bridge.contract.check_sources(sources)
    require(sha(path) == expected and sha(prepared) == analysis['prepare_sha256'], 'unchanged saved authority after arithmetic')
    result.update(paired_analysis=str(path), paired_analysis_sha256=expected,
        handoff_prepare_sha256=analysis['prepare_sha256'], sources=sources)
    require(len(bridge.canonical(result)) <= CAP, 'bounded full axis decomposition output')
    return result


def verify(path, expected):
    path = bridge.contract.absolute(path); bridge.core.hash_value(expected)
    require(sha(path) == expected, 'external decomposition SHA')
    p = bridge.READ(path, CAP); bridge.contract.check_sources(p['sources'])
    rebuilt = material(p['paired_analysis'], p['paired_analysis_sha256'])
    require(bridge.canonical(p) == bridge.canonical(rebuilt) and sha(path) == expected,
            'full source/48slots/24pairs squared decomposition and Q unknown reconstruction')
    return dict(status='saved_axis_decomposition_verified_not_admitted', sha256=expected,
        slots=48, pairs=24, roots=2, known_pairs=p['all_pairs']['known_pairs'], **FLAGS)


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('analyze'); a.add_argument('--paired-analysis', required=True)
    a.add_argument('--paired-sha256', required=True); a.add_argument('--output', required=True)
    a = sub.add_parser('verify'); a.add_argument('--analysis', required=True); a.add_argument('--analysis-sha256', required=True)
    args = parser.parse_args()
    if args.cmd == 'analyze':
        out = bridge.contract.absolute(args.output)
        require(out.parent == ROOT/'results' and out.name.startswith(NAME+'_') and not out.exists(),
                'NEW exclusive own analysis output before saved consumers')
        result = bridge.physical.write_new(out, material(args.paired_analysis, args.paired_sha256), cap=CAP)
    else: result = verify(args.analysis, args.analysis_sha256)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__': main()
