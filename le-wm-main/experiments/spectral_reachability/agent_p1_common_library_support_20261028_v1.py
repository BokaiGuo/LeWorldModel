"""Fixed common-library finite support, saved-only; not a deployable physical oracle."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import agent_p1_task_metric_mechanism_panel_20261028_v1 as panel

NAME = 'agent_p1_common_library_support_20261028_v1'
ROOT, rgb, teacher = panel.ROOT, panel.rgb, panel.teacher
require, sha, canonical = panel.require, panel.sha, panel.canonical
math = teacher.acquisition.math
INPUT = ROOT/'results/agent_p1_task_metric_mechanism_panel_20261028_v1_analysis_NASH02.json'
INPUT_SHA = '7eda1333a3bf2c8ef1d46deaf67d74c6d464498c018b52cf35e9802204501e5d'
SPACES = ('joint_xy', 'agent_xy', 'object_xy', 'object_angle_wrapped')
DEPLOYMENT = (('deployment_Z_uZ', 'Z', 1), ('deployment_S_uS', 'S', 3),
              ('cross_Z_uS', 'Z', 3), ('cross_S_uZ', 'S', 1))
CAP = 32*1024**2


def vector(value):
    return panel.base.vector(value, 7)


def fixed_roster(rows):
    require(len(rows) == 40 and [r['index'] for r in rows] == list(range(40)), 'complete original40 ordered slots')
    for i, r in enumerate(rows):
        task, rem = divmod(i, 10); role, word = divmod(rem, 5)
        require((r['root_slot'], r['condition'], r['role'], r['library_index'], r['library_label']) ==
                (task//2, ('off', 'on')[task%2], ('Z', 'S')[role], word, teacher.LIBRARY[word]),
                'literal 4task x 2role x original five suffix words')
        require((r['actual_Q'] is None) == (i == 12), 'original latent missing12, no filling or new missing latent')


def raw_library(rows):
    require(len(rows) == 10, 'both full five-word roles, no subset')
    buffers = []
    for r in rows:
        b = bytes.fromhex(r['suffix_raw_bits'])
        require(len(b) == 40, 'five float32 raw2 suffix commands')
        a = np.frombuffer(b, dtype='<f4').reshape(5, 2)
        require(np.isfinite(a).all() and np.max(np.abs(a)) <= 1, 'original finite raw action domain')
        buffers.append(a)
    for role in range(2):
        bank = buffers[role*5:role*5+5]
        require(bank[0].tobytes() == np.zeros((5, 2), '<f4').tobytes(), 'literal positive-zero suffix5')
        for plus, minus in ((1, 2), (3, 4)):
            require(bank[minus].tobytes() == (-bank[plus]).tobytes(), 'exact float32 sign flip, not time reversal')
    require(all(buffers[w].tobytes() == buffers[5+w].tobytes() for w in range(5)),
            'same five suffix raw buffers after both prefixes, duplicates not removed')
    return [dict(library_index=w, library_label=teacher.LIBRARY[w], raw_bits=buffers[w].tobytes().hex(),
        raw_sha256=hashlib.sha256(buffers[w].tobytes()).hexdigest(), commands=buffers[w].astype(float).tolist()) for w in range(5)]


def complete_envelope(values, reason='missing_required_slot'):
    require(len(values) == 5, 'all five slots required before evaluation upper envelope')
    for x in values:
        if x is not None:
            panel.base.rank.sign(x)
    missing = [i for i, x in enumerate(values) if x is None]
    if missing:
        return dict(label='evaluation upper envelope', fixed_slots=5, available=5-len(missing),
            missing_library_indices=missing, value=None, attaining_library_indices=None, reason=reason)
    value = max(values)
    return dict(label='evaluation upper envelope', fixed_slots=5, available=5, missing_library_indices=[],
        value=value, attaining_library_indices=[i for i, x in enumerate(values) if x == value], reason=None)


def finite_support(rows, root, goal, space):
    require(len(rows) == 5 and [r['library_index'] for r in rows] == list(range(5)), 'fixed ordered full five')
    direction = math.task_direction(dict(root_state=vector(root).tolist(), goal=vector(goal).tolist()), space)
    zero = rows[0]['returned_state']; responses = []
    for r in rows:
        state = r['returned_state']
        d = None if state is None or zero is None else math.difference(vector(state), vector(zero), space)
        projection = math.scalar_projection(d, direction)
        responses.append(dict(index=r['index'], library_index=r['library_index'], library_label=r['library_label'],
            zero_relative_delta=None if d is None else d.tolist(), projection=projection))
    envelope = complete_envelope([r['projection'] for r in responses],
        'zero_original_root_goal_direction' if direction is None else 'missing_required_physical_endpoint_or_zero')
    return dict(original_root_to_goal_unit_direction=None if direction is None else direction.tolist(),
        direction_status='zero_direction_unknown' if direction is None else 'original_shared_root10_to_goal',
        finite_responses=responses, support=envelope, units='rad' if space == 'object_angle_wrapped' else 'px',
        continuous_controllable_set_inferred=False, capability_gain_inferred=False)


def gains(row, zero):
    physical = {}
    for k, orientation in panel.base.METRICS.items():
        d = panel.delta(zero['physical_metrics'][k], row['physical_metrics'][k])
        physical[k] = None if d is None else orientation*d
    return dict(physical_metrics_higher_is_better=physical,
        native_reward_gain=panel.delta(row['returned_reward'], zero['returned_reward']),
        predicted_Q_gain=panel.delta(zero['predicted_Q'], row['predicted_Q']),
        actual_latent_Q_gain=panel.delta(zero['actual_Q'], row['actual_Q']),
        baseline_index=zero['index'], baseline_word='zero', comparison='same task and prefix, actual suffix5 versus own zero suffix5',
        model_Q_is_physical_gain=False)


def assemble(rows):
    fixed_roster(rows)
    tasks, output_rows = [], []
    for task in range(4):
        rr = rows[task*10:task*10+10]; first = rr[0]
        root, goal = first['boundary_trace']['10']['state'], first['physical_goal']
        vector(root); vector(goal)
        for r in rr:
            require(all(r[k] == first[k] for k in ('task_id', 'physical_goal', 'outline_pose', 'goal192', 'goal_binding')) and
                    r['boundary_trace']['10']['state'] == root, 'same original root/task goal across all10, never after residual')
            if r['returned_state'] is not None:
                vector(r['returned_state'])
                require(r['returned_state'] == r['boundary_trace']['20']['state'], 'actual endpoint20 not prefix or last available PNG')
        bank = raw_library(rr); roles = []
        for role_index, role in enumerate(('Z', 'S')):
            own = rr[role_index*5:role_index*5+5]; zero = own[0]
            require(all(r['boundary_trace']['15']['state'] == zero['boundary_trace']['15']['state'] for r in own),
                    'same role all five real suffix starts, no outcome conditioning')
            support = {space: finite_support(own, root, goal, space) for space in SPACES}
            native_envelopes = {k: complete_envelope([gains(r, zero)['physical_metrics_higher_is_better'][k] for r in own])
                for k in panel.base.METRICS}
            roles.append(dict(role=role, indices=[r['index'] for r in own], zero_index=zero['index'],
                actual_suffix_start15=deepcopy(zero['boundary_trace']['15']['state']), spaces=support,
                native_metric_zero_relative_evaluation_upper_envelopes=native_envelopes))
            for r in own:
                output_rows.append(dict(**deepcopy(r), own_zero_comparison=gains(r, zero),
                    common_direction_projections={space: deepcopy(support[space]['finite_responses'][r['library_index']]) for space in SPACES}))
        z, s = roles
        changes = {space: dict(H_Z=z['spaces'][space]['support']['value'], H_S=s['spaces'][space]['support']['value'],
            H_S_minus_H_Z=panel.delta(s['spaces'][space]['support']['value'], z['spaces'][space]['support']['value']),
            capability_change=None, interpretation='finite common-library evaluation upper envelope only') for space in SPACES}
        deployment = []
        for label, role, word in DEPLOYMENT:
            r = output_rows[task*10+(0 if role == 'Z' else 5)+word]
            deployment.append(dict(label=label, role=role, library_index=word, library_label=teacher.LIBRARY[word],
                index=r['index'], job_digest=r['job_digest'], plan_digest=r['plan_digest'],
                chosen_by_physical_outcome=False, origin='prephysical frozen model-generated suffix, no reselection',
                predicted_Q=r['predicted_Q'], actual_Q=r['actual_Q'], physical_metrics=deepcopy(r['physical_metrics']),
                returned_reward=r['returned_reward'], own_zero_comparison=deepcopy(r['own_zero_comparison']),
                common_direction_projections=deepcopy(r['common_direction_projections'])))
        tasks.append(dict(task=task, root_slot=first['root_slot'], condition=first['condition'], task_id=first['task_id'],
            original_root10=deepcopy(root), task_goal=deepcopy(goal), library=bank, roles=roles,
            support_comparison=changes, fixed_deployment_and_cross_terms=deployment,
            causal_non_greedy_prefix_identified=False, controllable_set_expansion=None))
    return dict(rows=output_rows, tasks=tasks, fixed_slots=40, task_conditions=4, known_roots=2,
        fresh_roots=0, roles_per_task=2, suffix_slots_per_role=5, physical_endpoints_available=sum(r['returned_state'] is not None for r in rows),
        actual_latent_available=39, missing_actual_latent=[12])


def build():
    authority = rgb.read(INPUT, INPUT_SHA, CAP)
    require(authority == panel.build(), 'entire NASH02 panel and original native/array/model/source closure rebuilt')
    require(authority['denominators']['task_all_pairs'] == 180 and authority['known_roots'] == 2, 'actual40/180 authority not older partial panel')
    receipt_path, receipt_sha = rgb.INPUTS['compiler_receipt']
    receipt = rgb.read(ROOT/receipt_path, receipt_sha)
    require(len(receipt['plans']) == 40, 'original all40 model-generated plans')
    action_receipts = []
    for r, plan in zip(authority['rows'], receipt['plans'], strict=True):
        require(all(r[k] == plan[k] for k in rgb.IDENTITY if k != 'job_digest'), 'native task/model action witness identity')
        raw = np.asarray(plan['commands'], '<f4')
        require(raw.shape == (10, 2) and raw[5:].tobytes().hex() == r['suffix_raw_bits'] and
                raw[:5].tobytes() == np.asarray(plan['prefix_commands'], '<f4').tobytes(), 'literal original prefix5/suffix5 buffer')
        action_receipts.append(dict(index=r['index'], plan_digest=r['plan_digest'], raw10_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
            suffix5_sha256=hashlib.sha256(raw[5:].tobytes()).hexdigest(), compiler_receipt_sha256=receipt_sha))
    bindings = {**authority['bindings'], str(INPUT): INPUT_SHA, str(ROOT/receipt_path): receipt_sha}
    require(bindings[str(teacher.PREREG)] == teacher.PREREG_SHA and sha(teacher.PREREG) == teacher.PREREG_SHA,
            'original next_mechanism_test exact authority')
    own = [Path(__file__).resolve(), ROOT/f'tests/test_{NAME}.py', ROOT/f'innerdocs/{NAME}.md']
    for f in own:
        bindings[str(f)] = sha(f)
    require(str(Path(math.__file__).resolve()) in bindings, 'original feature/wrap/direction arithmetic source bound')
    rgb.verify_bindings(bindings)
    return dict(schema=NAME+'.analysis', status='common_library_finite_support_described_not_admitted',
        input_panel_sha256=INPUT_SHA, bindings=bindings, source_files={str(f): bindings[str(f)] for f in own},
        action_array_receipts=action_receipts, boundary_array_receipts=deepcopy(authority['boundary_array_receipts']),
        P1_admission=False, physical_release=False, C2_failure_inferred=False, new_neural_calls=0, new_physics_calls=0,
        protocol=dict(authority=str(teacher.PREREG), authority_sha256=teacher.PREREG_SHA, spaces=list(SPACES),
            library=list(teacher.LIBRARY), support='max of all five dot(root-to-goal unit direction, endpoint(b)-endpoint(zero)); angle difference separately wrapped',
            missing_required_slot='support=null, never subset max', zero_direction='support=null',
            physical_oracle_label='evaluation upper envelope', continuous_controllability_inferred=False,
            physical_oracle_is_planner=False, deployment_fixed_before_physics=True, success_filter=False,
            no_new_Q_or_threshold=True, no_independent_sample_inflation=True), **assemble(authority['rows']))


def write(path, value):
    p = Path(path).absolute(); raw = canonical(value)
    require(p.resolve() == p and p.parent == ROOT/'results' and p.name.startswith(NAME+'_') and p.suffix == '.json' and
            len(raw) < CAP, 'exclusive own-stem bounded saved output')
    with p.open('xb') as stream:
        require(stream.write(raw) == len(raw), 'whole report write')
        stream.flush(); os.fsync(stream.fileno())
    rgb.sync_dir(p.parent)
    return dict(output=str(p), analysis_sha256=sha(p))


def verify(path, expected, rebuilt):
    require(rgb.read(path, expected, CAP) == rebuilt, 'full deterministic saved support rebuild')
    return dict(verified=True, analysis_sha256=expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('analyze', 'verify'))
    parser.add_argument('--output'); parser.add_argument('--analysis'); parser.add_argument('--analysis-sha')
    args = parser.parse_args(); value = build()
    out = write(args.output, value) if args.command == 'analyze' else verify(args.analysis, args.analysis_sha, value)
    print(canonical(out).decode())


if __name__ == '__main__':
    main()
