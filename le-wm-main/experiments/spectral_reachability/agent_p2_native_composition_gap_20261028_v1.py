"""Read-only, all-four-root P2 score reconciliation. No neural/physics imports."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

NAME = 'agent_p2_native_composition_gap_20261028_v1'
ROOT = Path(__file__).resolve().parents[2]
ARMS = ('joint_h2_no_composition', 'conditional_suffix_composition')
INPUTS = {
    'physical': ('results/agent_p2_composition_execution_20261028_v1_physical_MAIN01/report.json',
                 'ccb790aeb65313b391abaf9fcb671c24b8246f9e27a1e1bb162a3c233b1654b5'),
    'strict': ('results/agent_p2_composition_execution_20261028_v1_consume_MAIN01.json',
               '2afd4879481fb236514e3d18dd3d219cbc12b3b29295e1f204b9cbe9d48fbb79'),
    'audit': ('results/audit_p2_native_MAIN_20261028_v1_MAIN01.json',
              '2f2060f00512aa8710456202cdea289a6c7379a076a65eda9dff136442323c16'),
    'prepare': ('results/agent_p2_composition_execution_20261028_v1_prepare_MAIN01.json',
                '58d0c982d550b016efaa839cf8e24abe524d830a3da637ff5bd22286c84f0b83'),
    'teacher': ('results/p2_composition_predict_frontier_physical_20261027_v1_main01/report.json',
                '9144c3f25f2880af62d94dd88257918fa508b5ddbea51d98e9ccbc3d3b7532e2'),
    'teacher_protocol': ('results/p2_composition_predict_frontier_physical_20261027_v1_prepared01/protocol.json',
                         '7d1131e9dce9ed152ce53e5ae35f22cb4e7c328f8de50381d4363a52feadf2ef'),
}
COMPONENTS = ('agent_xy_sq', 'object_xy_sq', 'raw_angle_sq', 'agent_velocity_sq')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encode(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array(value, shape):
    # Reject booleans/strings/null before conversion; never cast missing to zero.
    raw = np.asarray(value, dtype=object)
    require(raw.shape == shape and all(type(x) in (int, float) for x in raw.flat),
            'literal numeric shape ' + str(shape))
    out = np.asarray(value, dtype=np.float64)
    require(np.isfinite(out).all(), 'finite saved values')
    return out


def close(actual, expected, label):
    # Only float64 recomputation roundoff, NOT an acceptance/scientific threshold.
    require(type(actual) in (int, float) and np.isfinite(actual), label + ': numeric')
    require(np.isclose(actual, expected, rtol=64 * np.finfo(float).eps,
                       atol=64 * np.finfo(float).eps), label + ': recomputation mismatch')


def decompose(state, goal):
    s, g = array(state, (7,)), array(goal, (7,))
    e = s - g
    a, o, angle, velocity = (float(e[:2] @ e[:2]), float(e[2:4] @ e[2:4]),
                              float(e[4] ** 2), float(e[5:] @ e[5:]))
    raw_angle = float(abs(e[4]))
    # Preserve installed eval_state exactly; wrapped is ONLY another diagnostic.
    native_angle = float(min(raw_angle, 2 * np.pi - raw_angle))
    joint = float(np.linalg.norm(e[:4]))
    return dict(agent_xy_sq=a, object_xy_sq=o, raw_angle_sq=angle,
                agent_velocity_sq=velocity, native7_sq=float(e @ e),
                native7=float(np.linalg.norm(e)), jointXY=joint,
                agentXY=float(np.linalg.norm(e[:2])), objectXY=float(np.linalg.norm(e[2:4])),
                agent_velocity_error=float(np.linalg.norm(e[5:])), raw_angle_error=raw_angle,
                native_angle_error=native_angle,
                wrapped_angle_diagnostic=float(abs(np.arctan2(np.sin(e[4]), np.cos(e[4])))),
                native_success=bool(joint < 20 and native_angle < np.pi / 9),
                object_only_success=bool(np.linalg.norm(e[2:4]) < 20 and native_angle < np.pi / 9))


def latent_cost(context, goal, projection):
    z = array(context['latent_history'], (3, 192))[-1]
    g, q = array(goal, (192,)), array(projection, (192, 192))
    delta = (z - g) @ q.T
    return float(np.mean(delta * delta))


def census(rows):
    require(len(rows) == 8, 'full eight-branch denominator')
    index = {}
    for row in rows:
        i, arm = row['root_slot'], row['arm']
        require(type(i) is int and 0 <= i < 4 and arm in ARMS and (i, arm) not in index,
                'unique literal root/arm')
        index[i, arm] = row
    require(set(index) == {(i, a) for i in range(4) for a in ARMS}, 'all four paired roots')
    return index


def pair_rows(rows):
    index = census(rows)
    pairs = []
    metrics = ('predicted_Q', 'teacher_Q', 'native7', 'jointXY', 'objectXY', 'agentXY',
               'agent_velocity_error', 'raw_angle_error', 'wrapped_angle_diagnostic')
    for i in range(4):
        j, c = index[i, ARMS[0]], index[i, ARMS[1]]
        require(j['root_sha256'] == c['root_sha256'] and j['goal'] == c['goal'] and
                j['initial_state'] == c['initial_state'], 'same paired root/state/goal')
        differences, winners = {}, {}
        for metric in metrics:
            def value(row):
                if metric == 'predicted_Q':
                    return row['prediction']['Q_cost']
                if metric == 'teacher_Q':
                    return row['prediction']['physical_teacher_Q']
                return row['observed'][metric]
            x, y = value(j), value(c)
            if x is None or y is None:
                differences[metric] = winners[metric] = None
                continue
            differences[metric] = y - x
            winners[metric] = 'tie' if y == x else (ARMS[1] if y < x else ARMS[0])
        contributions = {k: c['observed'][k] - j['observed'][k] for k in COMPONENTS}
        delta_sq = c['observed']['native7_sq'] - j['observed']['native7_sq']
        close(sum(contributions.values()), delta_sq, 'pair squared closure')
        pairs.append(dict(root_slot=i, root_sha256=j['root_sha256'],
                          composition_minus_joint=differences,
                          native7_squared_difference=delta_sq,
                          native7_squared_contributions=contributions, lower_is_better=winners,
                          predicted_vs_native_rank_disagreement=winners['predicted_Q'] != winners['native7'],
                          predicted_vs_teacher_rank_disagreement=None if winners['teacher_Q'] is None else
                          winners['predicted_Q'] != winners['teacher_Q'],
                          joint_vs_native_rank_disagreement=winners['jointXY'] != winners['native7'],
                          object_vs_joint_rank_disagreement=winners['objectXY'] != winners['jointXY']))
    return pairs


def analyze_bundle(physical, strict, audit, prepared, predictions, compilations):
    """Pure saved/RAM arithmetic; callers must authenticate file bytes separately."""
    require(physical['status'] == 'calibration_recorded' and
            strict['status'] == 'saved_calibration_verified_not_admitted', 'accepted terminal statuses')
    for item in (physical, strict):
        require(item['root_denominator'] == 4 and item['arm_denominator'] == 8 and
                item['P1_admission'] is False and item['P2_admission'] is False, 'fixed census/no admission')
    require(audit['known_roots'] == 4 and audit['native_returns_verified'] == 80 and
            audit['P1_admission'] is False and audit['P2_admission'] is False, 'MAIN complete native audit')
    require(strict['scientific_admission'] is False, 'no scientific release')
    pidx, sidx, aidx, plans = (census(x) for x in
                              (physical['rows'], strict['rows'], audit['rows'], prepared['plans']))
    require(len(predictions) == 4 and {x['root_slot'] for x in predictions} == set(range(4)),
            'four prediction roots')
    pred = {x['root_slot']: x for x in predictions}
    rows = []
    for i in range(4):
        root = pred[i]
        require([x['arm'] for x in root['results']] == list(ARMS), 'prediction arm order')
        for arm in ARMS:
            outer, sealed, checked, handoff = pidx[i, arm], sidx[i, arm], aidx[i, arm], plans[i, arm]
            b, plan = outer['physical'], handoff['execution_plan']
            r = handoff['root_sha256']
            require(outer['status'] == 'returned' and sealed['physical'] == b and
                    outer['handoff_sha256'] == sealed['handoff_sha256'] == handoff['handoff_sha256'],
                    'physical/strict/handoff identity')
            unsealed = {k: v for k, v in handoff.items() if k != 'handoff_sha256'}
            require(digest(unsealed) == handoff['handoff_sha256'], 'handoff digest')
            require(root['root_sha256'] == b['root_sha256'] == plan['root_sha256'] == r and
                    b['root_index'] == plan['root_index'] == i and b['method'] == plan['method'] == arm,
                    'exact root/program identity')
            result = root['results'][ARMS.index(arm)]
            win = result['winner']
            chosen = [x for x in result['attempts'] if x['candidate_id'] == handoff['candidate_id']]
            require(len(chosen) == 1, 'one sealed selected candidate')
            chosen = chosen[0]
            require(digest(chosen) == handoff['source_selection_digest'] and
                    chosen['candidate_id'] == win['candidate_id'] and chosen['cost'] == win['cost'] and
                    chosen['witness'] == win['witness'] and
                    chosen['actions'] == plan['commands'] == win['witness']['receipt']['actions'] and
                    handoff['native_receipt_sha256'] == win['witness']['receipt']['receipt_sha256'],
                    'sealed winner/actions/receipt; no reselection')
            require(win['independent_full_replay_exact'] is True and
                    plan['selected_predicted_context'] == win['whole_context'] ==
                    win['witness']['receipt']['terminal_context'], 'saved whole/segmented replay closure')
            compiled = compilations[r]
            require(plan['goal'] == compiled['goal_vectors'][0] and
                    plan['active_metric_id'] == b['active_metric_id'] == compiled['scoring']['metric']['sha256'],
                    'original compiled goal/Q identity')
            q = compiled['scoring']['metric']['projection']
            cost = latent_cost(win['whole_context'], plan['goal'], q)
            close(win['cost'], cost, 'original projected latent objective')
            actions = array(plan['commands'], (10, 2))
            require(np.array_equal(actions, actions.astype(np.float32)) and np.max(abs(actions)) <= 1,
                    'sealed literal raw float32 actions')
            require(b['full_horizon_completed'] is True and b['endpoint_fresh'] is True and
                    b['physical_env_step_returns'] == b['completed_commands'] == 10 and
                    len(b['trace']) == 10, 'all ten actual returns, no padded endpoints')
            goal, initial = b['context_identity']['goal'], b['context_identity']['state']
            initial_metrics = decompose(initial, goal)
            trajectory = []
            for step, event in enumerate(b['trace'], 1):
                require(event['primitive_step'] == step and event['kind'] == 'observation' and
                        event['command'] == plan['commands'][step - 1], 'ordered actual action/state trace')
                m = decompose(event['state'], goal)
                close(event['reward'], -m['native7'], 'original reward')
                require(event['terminated'] is m['native_success'] and event['truncated'] is False,
                        'native termination unchanged')
                trajectory.append(dict(step=step, state=event['state'], metrics=m))
            require(b['final_state'] == b['trace'][-1]['state'], 'actual terminal state')
            observed = trajectory[-1]['metrics']
            fields = dict(agent_position_error='agentXY', object_position_error='objectXY',
                          official_position_error='jointXY', official_state_distance='native7',
                          angle_error='native_angle_error')
            for old, new in fields.items():
                close(b['official'][old], observed[new], 'original native label ' + old)
            for flag in ('native_success', 'object_only_success'):
                old = 'official_success' if flag == 'native_success' else flag
                require(b['official'][old] is observed[flag], 'original success label')
            for key in ('jointXY', 'native7', 'objectXY'):
                close(checked[key], observed[key], 'independent MAIN audit ' + key)
            require(checked['native_success'] is observed['native_success'], 'MAIN native success')
            refs = prepared['references'][r]
            require(all(b['context_identity'][k] == refs[k] for k in
                        ('history_rgb_sha256', 'goal_rgb_sha256')), 'original physical RGB identities')
            rows.append(dict(root_slot=i, root_sha256=r, arm=arm, candidate_id=win['candidate_id'],
                             selection_digest=handoff['source_selection_digest'], goal=goal, initial_state=initial,
                             initial_metrics=initial_metrics, observed=observed, trajectory=trajectory,
                             prediction=dict(Q_cost=cost, saved_Q_cost=win['cost'],
                                             Q_sha256=digest(q), metric_sha256=plan['active_metric_id'],
                                             nuisance_span_rank=compiled['scoring']['metric']['rank'],
                                             physical_teacher_Q=None, physical_latent_prediction_error=None),
                             terminal_rgb_path=b['terminal_rgb_path'], terminal_rgb_sha256=b['terminal_rgb_sha256']))
    pairs = pair_rows(rows)
    return dict(schema=NAME + '.analysis', status='all_saved_rows_reconciled_not_admitted',
                root_denominator=4, arm_denominator=8, actual_selected_step_returns=80, rows=rows, pairs=pairs,
                native_successes={a: sum(x['observed']['native_success'] for x in rows if x['arm'] == a)
                                  for a in ARMS},
                composition_distance_wins={k: None if any(p['lower_is_better'][k] is None for p in pairs) else
                                           sum(p['lower_is_better'][k] == ARMS[1] for p in pairs)
                                           for k in pairs[0]['lower_is_better']},
                units=dict(agentXY='native position units (render pixel coordinates)',
                           objectXY='native position units', agent_velocity='native agent velocity units',
                           angle='radians', native7='unscaled mixed-unit native L2; not a position distance',
                           Q_cost='mean squared projected frozen latent residual; NOT physical state distance'),
                interpretation='Squared contributions are algebraic, not causal interventions. Zero native successes do not prove impossibility.',
                next_test=dict(status='proposal_for_MAIN_preregistration_not_executed',
                               population='same four roots, both sealed programs, no root/action reselection',
                               intervention='none; after terminal teacher reuse, encode all eight saved step5 RGB under same original frozen encoder/Q',
                               primary='all4 paired step5/10 signed Q and unchanged agent/object/velocity/angle errors; no winner selection',
                               no_fit=True, missing='step5 teacher latent identity closure and matched saved intermediate predictions; MAIN CUDA authorization',
                               split='localize midpoint vs terminal mismatch without new physics; ranking diagnostic, not causal identification'),
                new_neural_calls=0, new_physics_steps=0, equal_full_resource_cost=False,
                independent_confirmation=False, freshness=False, P1_admission=False, P2_admission=False,
                scientific_admission=False)


def attach_teacher(result, physical, prepared, compilations, teacher, protocol):
    """Reuse old actual CUDA arrays ONLY after exact program/image/state/model joins.

    Old and new selection hashes differ because their export schemas differ. We do
    not equate those hashes: old receipt joins old plan, new receipt joins new plan,
    then the raw10/full predictor context/goal/Q and actual endpoints must match.
    """
    require(teacher['status'] == 'guarded_calibration_recorded' and teacher['model_unchanged'] is True and
            teacher['source_verified'] is True and teacher['host']['device'] == 'cuda' and
            protocol['model_bindings'] == prepared['model_bindings'], 'original teacher CUDA/model authority')
    require(teacher['encoding_stats'] == dict(encoder_attempts=8, encoder_completed=8,
                                             projector_attempts=8, projector_completed=8,
                                             forbidden_predict_attempts=0), 'teacher full eight sample accounting')
    require(len(teacher['encodings']) == len(teacher['rows']) == len(protocol['plans']) == 8,
            'teacher all8 denominator')

    def index(items):
        out = {(x['root_sha256'], x['method']): x for x in items}
        require(len(out) == 8, 'unique teacher root/program')
        return out

    enc, oldrows, oldplans = (index(items) for items in
                              (teacher['encodings'], teacher['rows'], protocol['plans']))
    newplans, newrows = census(prepared['plans']), census(physical['rows'])
    for row in result['rows']:
        key, pair = (row['root_sha256'], row['arm']), (row['root_slot'], row['arm'])
        e, old, op = enc[key], oldrows[key], oldplans[key]
        new, np_ = newrows[pair]['physical'], newplans[pair]['execution_plan']
        require(e['status'] == 'completed' and e['attempted_encode_samples'] == 1 and
                e['selection_sha256'] == old['selection_sha256'] == op['selection_sha256'],
                'old teacher receipt/plan join')
        require(e['terminal_rgb_sha256'] == old['terminal_rgb_sha256'] == new['terminal_rgb_sha256'] and
                old['final_state'] == new['final_state'] and old['context_identity'] == new['context_identity'] and
                old['trace'] == new['trace'], 'exact old/new image/state/action trajectory')
        require(all(op[k] == np_[k] for k in ('commands', 'goal', 'predictor_context',
                                              'selected_predicted_context', 'active_metric_id')) and
                e['active_metric_id'] == np_['active_metric_id'], 'same executed program and compiled objective')
        metrics = protocol['root_metrics'][key[0]]
        q = compilations[key[0]]['scoring']['metric']['projection']
        require(metrics['goal'] == np_['goal'] and metrics['metrics']['source_Q']['projection'] == q,
                'teacher exact original goal/projector')
        z = array(e['terminal_vector'], (192,))
        predicted = array(np_['selected_predicted_context']['latent_history'], (3, 192))[-1]
        delta = (z - array(np_['goal'], (192,))) @ array(q, (192, 192)).T
        cost = float(np.mean(delta * delta))
        close(e['actual_costs']['source_Q'], cost, 'teacher original Q recomputation')
        close(e['predicted_costs']['source_Q'], row['prediction']['Q_cost'], 'teacher predicted Q identity')
        row['prediction'].update(physical_teacher_Q=cost,
                                 physical_latent_prediction_error=float(np.linalg.norm(predicted - z)),
                                 predicted_minus_teacher_Q=row['prediction']['Q_cost'] - cost,
                                 teacher_terminal_vector_sha256=digest(e['terminal_vector']))
    result['pairs'] = pair_rows(result['rows'])
    for pair in result['pairs']:
        d = pair['composition_minus_joint']
        # Exact signed pair identity: Δpredicted = Δteacher + Δ(predicted-teacher).
        pair['paired_rollout_Q_gap'] = d['predicted_Q'] - d['teacher_Q']
        pair['teacher_vs_native_rank_disagreement'] = pair['lower_is_better']['teacher_Q'] != pair['lower_is_better']['native7']
    result['composition_distance_wins']['teacher_Q'] = sum(
        p['lower_is_better']['teacher_Q'] == ARMS[1] for p in result['pairs'])
    result['teacher_reuse'] = dict(samples=8, new_encode_calls=0,
                                  scope='same image bytes/raw actions/full trajectory/goal/Q/model metadata; old encoder not rerun',
                                  scientific_independence=False)


def load_saved():
    """Authenticate consumed leaves, not a new independent run of old consumers."""
    consumed = {}

    def read(path, expected):
        path = Path(path).resolve()
        require(sha(path) == expected, 'input SHA drift: ' + str(path))
        consumed[str(path)] = expected
        return json.loads(path.read_text())

    docs = {key: read(ROOT / path, h) for key, (path, h) in INPUTS.items()}
    physical, strict, audit, prepared = (docs[k] for k in ('physical', 'strict', 'audit', 'prepare'))
    require(strict['report_sha256'] == audit['report_sha256'] == INPUTS['physical'][1] and
            physical['prepare_sha256'] == strict['prepare_sha256'] == INPUTS['prepare'][1], 'receipt report/prepare binding')
    bindings = prepared['bindings']

    def bound(path):
        path = Path(path).resolve()
        require(str(path) in bindings, 'missing inherited leaf binding: ' + str(path))
        return read(path, bindings[str(path)])

    for suffix in ('envs/pusht/env.py', 'run_pusht_program_search_cuda.py',
                   'audit_pusht_sealed_task_contract.py'):
        matches = [p for p in bindings if p.endswith(suffix)]
        require(len(matches) == 1 and sha(matches[0]) == bindings[matches[0]],
                'frozen metric/state semantics: ' + suffix)
        consumed[matches[0]] = bindings[matches[0]]

    # SHA-check every original physical artifact (including actual RGB and journals).
    for name, h in physical['artifact_sha256'].items():
        path = (ROOT / INPUTS['physical'][0]).parent / name
        require(path.resolve().parent == (ROOT / INPUTS['physical'][0]).parent, 'artifact path scope')
        require(sha(path) == h, 'physical artifact drift: ' + name)
        consumed[str(path)] = h
    planner_path = ROOT / 'results/agent_p2_runtime_guard_20261028_v1_known_MAIN01/report.json'
    planner = bound(planner_path)
    require(strict['inherited_predict_resources'] == prepared['inherited_resources'], 'inherited resource identity')
    require(sha(planner_path) == prepared['planner_report_sha256'], 'planner report authority')
    predictions = [bound(planner_path.parent / f'root_{i}.json') for i in range(4)]
    wrapper = bound(ROOT / 'results/agent_p2_runtime_guard_20261028_v1_prepared01.json')
    directory = Path(wrapper['native_directory'])
    native = bound(directory / 'prepared.json')
    protocol = bound(directory / 'protocol.json')
    # These two historical identifiers are canonical-JSON digests, not byte SHAs.
    # Raw file bytes have already been checked against inherited bindings above.
    require(digest(native) == wrapper['native_prepared_sha256'] and
            digest(protocol) == native['protocol_sha256'], 'native protocol closure')
    compilation = bound(protocol['context']['compilation'])
    compiled = {r['root_sha256']: r['compiled'] for r in compilation['rows']}
    result = analyze_bundle(physical, strict, audit, prepared, predictions, compiled)
    attach_teacher(result, physical, prepared, compiled, docs['teacher'], docs['teacher_protocol'])
    result['provenance'] = dict(consumed_sha256=consumed, source_sha256=sha(__file__),
                                inherited_resources=prepared['inherited_resources'],
                                scope='All consumed metadata and original physical artifacts verified; old strict receipt reused. No model weights loaded or full historical consumer rerun.')
    # End-of-read seal check guards concurrent changes; no new evidence file is fabricated.
    for path, h in consumed.items():
        require(sha(path) == h, 'input changed during saved analysis')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='optional NEW analysis JSON; default stdout only')
    args = parser.parse_args()
    result = load_saved()
    raw = encode(result)
    if args.output is not None:
        with args.output.open('xb') as stream:
            stream.write(raw + b'\n')
        print(json.dumps(dict(output=str(args.output.resolve()), sha256=sha(args.output),
                              root_denominator=4, arm_denominator=8, scientific_admission=False)))
    else:
        print(raw.decode())


if __name__ == '__main__':
    main()
