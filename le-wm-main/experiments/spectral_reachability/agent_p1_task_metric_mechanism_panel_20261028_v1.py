"""Full fixed40 saved mechanism panel; trajectories are not capability interventions."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import itertools
import os
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import agent_p1_teacher_library40_task_metric_20261028_v1 as base

NAME = 'agent_p1_task_metric_mechanism_panel_20261028_v1'
ROOT, rgb, teacher = base.ROOT, base.rgb, base.teacher
require, sha, canonical = base.require, base.sha, base.canonical
INPUT = ROOT/'results/agent_p1_teacher_library40_task_metric_20261028_v1_analysis_NASH01.json'
INPUT_SHA = '52c0739d8e2b6f996dc751bfe3f085074549120cfa0355eb6283e80d653d6373'
BOUNDARIES = ((10, 'warmup', 10), (15, 'prefix', 15), (20, 'suffix', 5))
SEGMENTS = {'prefix': ['10', '15'], 'suffix': ['15', '20'], 'total': ['10', '20']}
CAP = 32*1024**2


def delta(a, b):
    if a is None or b is None:
        return None
    out = float(a)-float(b)
    base.rank.sign(out)
    return out


def row_panel(row, boundaries):
    require(set(boundaries) == {'10', '15', '20'}, 'all three literal boundaries')
    require(boundaries['20']['metrics'] == row['physical_metrics'], 'terminal original metric equality')
    for step, item in boundaries.items():
        require(item['absolute_step'] == int(step) and item['goal'] == row['physical_goal'] and
                item['outline_pose'] == row['outline_pose'], 'unchanged actual task goal and renderer contract')
        require(set(base.METRICS).issubset(item['metrics']), 'complete metric axes, unknown retained')
        for k in base.METRICS:
            if item['metrics'][k] is not None:
                base.rank.sign(float(item['metrics'][k]))
    changes = {}
    for segment, (start, end) in SEGMENTS.items():
        changes[segment] = {}
        for k, orientation in base.METRICS.items():
            raw = delta(boundaries[end]['metrics'][k], boundaries[start]['metrics'][k])
            changes[segment][k] = dict(raw_end_minus_start=raw,
                lower_is_better_delta=None if raw is None else orientation*raw)
    patterns = {}
    for k in base.METRICS:
        p = changes['prefix'][k]['lower_is_better_delta']
        s = changes['suffix'][k]['lower_is_better_delta']
        patterns[k] = dict(prefix_worsens=None if p is None else p > 0,
            suffix_improves=None if s is None else s < 0,
            observed_worse_then_better=None if p is None or s is None else p > 0 and s < 0,
            truly_physical_non_greedy=None, capability_change=None,
            identification='unknown_no_matched_capability_intervention_or_greedy_counterfactual')
    return dict(**deepcopy(row), boundary_trace=boundaries, segment_changes=changes,
        trajectory_patterns=patterns, causal_mechanism_identified=False)


def pair(a, b):
    require(a['index'] < b['index'], 'fixed ascending pair order')
    for k in ('root_slot', 'condition', 'task_id', 'goal192', 'goal_binding', 'physical_goal', 'outline_pose'):
        require(a[k] == b[k], 'same native task/context-goal authority: '+k)
    same_role = a['role'] == b['role']
    same_word = a['library_index'] == b['library_index']
    if same_role:
        base.rank.matching(a, b, 'within_role_library')
    elif same_word:
        base.rank.matching(a, b, 'same_word_ZS')
    dp, da = delta(a['predicted_Q'], b['predicted_Q']), delta(a['actual_Q'], b['actual_Q'])
    physical = {}
    for k, orientation in base.METRICS.items():
        raw = delta(a['physical_metrics'][k], b['physical_metrics'][k])
        oriented = None if raw is None else orientation*raw
        physical[k] = dict(raw_delta=raw, lower_is_better_delta=oriented, orientation=orientation,
            predicted_Q_to_physical=base.classify(dp, oriented),
            actual_Q_to_physical=base.classify(da, oriented),
            segment_comparisons={})
        for segment in SEGMENTS:
            d = delta(a['segment_changes'][segment][k]['lower_is_better_delta'],
                      b['segment_changes'][segment][k]['lower_is_better_delta'])
            physical[k]['segment_comparisons'][segment] = dict(oriented_change_delta=d,
                predicted_Q_to_change=base.classify(dp, d), actual_Q_to_change=base.classify(da, d))
    return dict(i=a['index'], j=b['index'], root_slot=a['root_slot'], condition=a['condition'],
        task_id=a['task_id'], roles=[a['role'], b['role']], words=[a['library_label'], b['library_label']],
        same_role=same_role, same_suffix_word=same_word,
        prefix_contexts_differ=not same_role, matched_suffix_bits=a['suffix_raw_bits'] == b['suffix_raw_bits'],
        predicted_Q_to_actual_Q=base.classify(dp, da), physical_metrics=physical,
        missing_actual_latent=[r['index'] for r in (a, b) if r['actual_Q'] is None],
        causal_interpretation=False)


def tally(items):
    names = ('concordant', 'discordant', 'both_tie', 'left_tie_only', 'right_tie_only', 'unknown')
    counts = {k: sum(p['classification'] == k for p in items) for k in names}
    require(sum(counts.values()) == len(items), 'all pair classifications retained')
    return dict(fixed=len(items), classifications=counts,
        available=len(items)-counts['unknown'], unknown=counts['unknown'],
        strict_nonzero=counts['concordant']+counts['discordant'],
        sign_equal=sum(p.get('sign_a') == p.get('sign_b') for p in items if p['available']),
        reversal_count=counts['discordant'], ties=sum(counts[k] for k in names if 'tie' in k))


def summary(pairs):
    return dict(predicted_Q_to_actual_Q=tally([p['predicted_Q_to_actual_Q'] for p in pairs]),
        physical_metrics={k: dict(
            predicted_Q_to_physical=tally([p['physical_metrics'][k]['predicted_Q_to_physical'] for p in pairs]),
            actual_Q_to_physical=tally([p['physical_metrics'][k]['actual_Q_to_physical'] for p in pairs]),
            segments={s: {axis: tally([p['physical_metrics'][k]['segment_comparisons'][s][axis] for p in pairs])
                for axis in ('predicted_Q_to_change', 'actual_Q_to_change')} for s in SEGMENTS}) for k in base.METRICS})


def assemble(rows):
    base.assemble(deepcopy(rows))  # Frozen fixed40 identities/missing12; no filtered library.
    groups, all_pairs, role_pairs, zs_pairs = [], [], [], []
    for task in range(4):
        members = rows[task*10:task*10+10]
        ps = [pair(a, b) for a, b in itertools.combinations(members, 2)]
        all_pairs.extend(ps)
        role_pairs.extend(p for p in ps if p['same_role'])
        zs_pairs.extend(p for p in ps if not p['same_role'] and p['same_suffix_word'])
        groups.append(dict(task_id=members[0]['task_id'], root_slot=members[0]['root_slot'],
            condition=members[0]['condition'], indices=[r['index'] for r in members], pairs=ps, summary=summary(ps)))
    require((len(all_pairs), len(role_pairs), len(zs_pairs)) == (180, 80, 20), 'fixed pair denominators')
    roles = []
    for anchor in range(8):
        indices = list(range(anchor*5, anchor*5+5))
        ps = [p for p in role_pairs if p['i'] in indices]
        roles.append(dict(indices=indices, root_slot=rows[indices[0]]['root_slot'],
            condition=rows[indices[0]]['condition'], role=rows[indices[0]]['role'], pairs=ps, summary=summary(ps)))
    return dict(rows=rows, task_groups=groups, role_groups=roles,
        task_all_pair_summary=summary(all_pairs), within_role_summary=summary(role_pairs),
        same_word_ZS_pairs=zs_pairs, same_word_ZS_summary=summary(zs_pairs), fixed_slots=40,
        actual_latent_available=39, missing_actual_latent=[12],
        denominators=dict(task_all_pairs=180, within_role_pairs=80, same_word_ZS_pairs=20,
            overlapping_views_not_independent=True))


def read_boundary(event, row, directory, receipts):
    info = event['info']; path = teacher.flat(directory, info['file'])
    require(sha(path) == info['sha256'], 'actual boundary info SHA')
    values = {}
    with np.load(path, allow_pickle=False) as arrays:
        for key, size in (('goal_state', 7), ('goal_pose', 3), ('block_pose', 3), ('pos_agent', 2), ('vel_agent', 2)):
            meta = info['tree'][key]; x = arrays[meta['array']]
            require(x.dtype.str == meta['dtype'] and list(x.shape) == meta['shape'] and x.shape == (size,) and
                    np.isfinite(x).all(), 'complete native boundary array '+key)
            receipts[str(path)+':'+meta['array']] = dict(file_sha256=info['sha256'], dtype=x.dtype.str,
                shape=list(x.shape), array_sha256=hashlib.sha256(x.tobytes()).hexdigest())
            values[key] = x.copy()
    state = base.vector(event['state'], 7)
    require(values['goal_pose'].tolist() == row['outline_pose'] and
            np.array_equal(state[:2], values['pos_agent']) and np.array_equal(state[5:], values['vel_agent']),
            'actual state/renderer at each boundary, never inferred from terminal')
    returned_metrics = base.task_metrics(state, values['goal_state'], values['block_pose'], values['goal_pose'],
        event['native_success'], event['native_distance'], event['reward'])
    prebind = (event['phase'], event['step']) == ('warmup', 10)
    require(prebind or values['goal_state'].tolist() == row['physical_goal'], 'actual post-bind task goal invariant')
    if prebind:
        d = state-base.vector(row['physical_goal'], 7)
        angle = abs(float(d[4])); distance = float(np.linalg.norm(d))
        success = bool(np.linalg.norm(d[:4]) < 20 and min(angle, 2*np.pi-angle) < np.pi/9)
        metrics = base.task_metrics(state, row['physical_goal'], values['block_pose'], values['goal_pose'],
            success, distance, -distance)
    else:
        metrics = returned_metrics
    return dict(state=state.tolist(), goal=deepcopy(row['physical_goal']), outline_pose=values['goal_pose'].tolist(),
        returned_goal=values['goal_state'].tolist(), returned_native_metrics=returned_metrics,
        metric_origin='saved_state_under_task_goal_checked_against_root_checkpoint' if prebind else 'actual_native_task_return',
        warmup_return_precedes_task_goal_bind=prebind,
        metrics=metrics, native_reward=event['reward'], terminated=event['terminated'], truncated=event['truncated'],
        event_sha256=event['event_sha256'], native_info=deepcopy(info))


def build():
    original = rgb.read(INPUT, INPUT_SHA, CAP)
    require(original == base.build(), 'full accepted task-metric native saved rebuild')
    path = ROOT/rgb.INPUTS['physical'][0]
    report = rgb.read(path, rgb.INPUTS['physical'][1])
    events = base.physics.read_events(path.parent/report['journal']['file'], report['journal_tail'])
    rows, receipts = [], {}
    for row in original['rows']:
        returns = [e for e in events if e['kind'] == 'raw_return' and e['job_digest'] == row['job_digest']]
        expected = [('warmup', i) for i in range(1, 11)]+[('prefix', i) for i in range(11, 16)]+[('suffix', i) for i in range(1, 6)]
        require([(e['phase'], e['step']) for e in returns] == expected, 'exact full20 ordered native returns')
        boundaries = {}
        for absolute, phase, step in BOUNDARIES:
            e = returns[absolute-1]
            require((e['phase'], e['step']) == (phase, step) and not e['terminated'] and not e['truncated'], 'literal live boundary')
            item = read_boundary(e, row, path.parent, receipts)
            item.update(absolute_step=absolute, phase=phase, step=step)
            boundaries[str(absolute)] = item
        physical_row = report['rows'][row['index']]
        checkpoint = physical_row['checkpoints'][0]
        require(checkpoint['step'] == 0 and checkpoint['absolute_step'] == 10 and
                checkpoint['native_distance'] == boundaries['10']['metrics']['native_distance'] and
                checkpoint['native_success'] == boundaries['10']['metrics']['task_success'] and
                physical_row['root_state'] == boundaries['10']['state'],
                'same actual warmup10 state at post-bind root checkpoint, original native task score')
        boundaries['10']['post_bind_root_checkpoint'] = deepcopy(checkpoint)
        rows.append(row_panel(row, boundaries))
    bindings = {**original['bindings'], str(INPUT): INPUT_SHA}
    own = [Path(__file__).resolve(), ROOT/f'tests/test_{NAME}.py', ROOT/f'innerdocs/{NAME}.md']
    for f in own:
        bindings[str(f)] = sha(f)
    rgb.verify_bindings(bindings)
    return dict(schema=NAME+'.analysis', status='saved_full40_descriptive_mechanism_panel_not_admitted',
        bindings=bindings, source_files={str(f): bindings[str(f)] for f in own}, boundary_array_receipts=receipts,
        input_analysis_sha256=INPUT_SHA, known_roots=2, independent_fresh_roots=0,
        P1_admission=False, physical_release=False, C2_failure_inferred=False,
        winner_selection=False, Q_or_goal_changed=False, fitted_score=False, new_neural_calls=0, new_physics_calls=0,
        protocol=dict(fixed_before_new_execution=True, pairing='four literal task groups; all 45 ascending pairs each; role80 and samewordZS20 are overlapping views',
            metrics=list(base.METRICS), stages=[10, 15, 20], segments=SEGMENTS, exact_sign_zero_is_tie=True,
            success_filter=False, physical_non_greedy_identifiable=False, capability_change_identifiable=False,
            reason='observational prefix/suffix changes are available, not matched capability interventions or a greedy counterfactual',
            retrospective_saved_protocol_not_prospective_confirmation=True), **assemble(rows))


def write(path, analysis):
    p = Path(path).absolute(); raw = canonical(analysis)
    require(p.resolve() == p and p.parent == ROOT/'results' and p.name.startswith(NAME+'_') and
            p.suffix == '.json' and len(raw) < CAP, 'bounded exclusive new own-stem report')
    with p.open('xb') as stream:
        require(stream.write(raw) == len(raw), 'complete report write')
        stream.flush(); os.fsync(stream.fileno())
    rgb.sync_dir(p.parent)
    return dict(output=str(p), analysis_sha256=sha(p))


def verify(path, expected_sha, rebuilt):
    require(rgb.read(path, expected_sha, CAP) == rebuilt, 'whole mechanism panel deterministic rebuild')
    return dict(verified=True, analysis_sha256=expected_sha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('analyze', 'verify'))
    parser.add_argument('--output'); parser.add_argument('--analysis'); parser.add_argument('--analysis-sha')
    args = parser.parse_args(); analysis = build()
    out = verify(args.analysis, args.analysis_sha, analysis) if args.command == 'verify' else write(args.output, analysis)
    print(canonical(out).decode())


if __name__ == '__main__':
    main()
