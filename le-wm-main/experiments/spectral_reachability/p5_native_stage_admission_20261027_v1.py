"""Fixed native-C1 snapshot admission audit; stdlib saved JSON only.

Exit 0: the snapshot is consistent, NOT scientific admission. Exit 2: invalid
or missing evidence. --require-admission exits 3 for a valid but blocked snapshot.
No model/environment imports, execution authority, threshold fitting or scan of
new outcomes. Report hashes are FILE hashes, not canonical bridge digests.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

NAME = 'p5_native_stage_admission_20261027_v1'
ROOT = Path(__file__).resolve().parents[2]
ARMS = ('compiled', 'nominal', 'factual_reference')
EXEC = 'results/p5_native_c1_executor_20261027_v2/native_main_v1'
SPECS = {
    'executor': (EXEC + '/report.json', 'e2f8245f8f69e41eeb35ec5507d8e744f7f32e1e8f76948756927ddf42c44ae6'),
    'ready': ('results/p5_native_c1_executor_20261027_v2/manifest_ready_frozen.json', '1099ee3b001ddb9b88fad4d9c31e1fac39fef07b1120501ccee44a907e6c95b3'),
    'outcome': ('results/p5_native_c1_outcome_audit_20261027_v2_main_v1/report.json', '3d822b29738f6240a68ed4a104e62dcf984b8d2e10f802b19c369eb7519ec811'),
    'geometry': ('results/p5_native_c1_outcome_audit_20261027_v2_main_v1/contact_geometry_main_v1.json', 'aad43b5696ecdc71d2061d9106835304c218f8ca4c0b4a3f4be41da237dfa50f'),
    'compiler': ('results/p5_prospective_c1_cuda_compiler_20261027_v1_cuda_main_v2/compiler_receipt.json', '4c3529de98d55d736df5bd7140dbfd9948e05d4cadb0d4699f6d69a352866b0a'),
    'ridge': ('results/p5_prospective_ridge_array_audit_main_20261027_v1/cuda_v2_audit.json', '71ee8e419e6e5cefc4ee51940fea2b28c6505a41b44ffd4434fd9a6368a45e60'),
    'fd': ('results/p5_prospective_c1_native_fd_20261027_v2_cuda_main_v1/fd_receipt.json', '38c87c28bee81366ba1e59f14f6cec89718f3ee6abaa3c64e629c0d214dc295f'),
    'fd_audit': ('results/p5_prospective_c1_native_fd_20261027_v2_cuda_main_v1/independent_array_audit_main_v1.json', 'a5f698017f3ae09eee1fd6c63ce181647f5b9a7e349f6f1949f7e168b8d53467'),
}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key: ' + key)
        result[key] = value
    return result


def finite_tree(value):
    if isinstance(value, float):
        require(math.isfinite(value), 'nonfinite JSON number')
    elif isinstance(value, dict):
        for child in value.values():
            finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            finite_tree(child)


class Reader:
    """Bounded, nonsymlink, project-local reads, with end-of-audit drift check."""
    def __init__(self, root):
        self.root = Path(root).absolute()
        self.bindings = {}
        self.bytes_read = 0

    def read(self, path, expected):
        p = Path(path)
        p = p if p.is_absolute() else self.root / p
        require('..' not in p.parts and p.is_relative_to(self.root), 'path outside project')
        require(not any(q.is_symlink() for q in (p, *p.parents)), 'symlink input')
        require(p.is_file() and p.stat().st_size <= 1024 * 1024, 'missing/oversized input: ' + str(p))
        b = p.read_bytes()
        self.bytes_read += len(b)
        require(self.bytes_read <= 8 * 1024 * 1024 and len(b) <= 1024 * 1024, 'read budget')
        digest = hashlib.sha256(b).hexdigest()
        require(digest == expected, 'SHA mismatch: ' + str(p))
        require(str(p) not in self.bindings or self.bindings[str(p)] == digest, 'input drift')
        self.bindings[str(p)] = digest
        data = json.loads(b, object_pairs_hook=unique_pairs)
        finite_tree(data)
        return data

    def stable(self):
        for p, digest in list(self.bindings.items()):
            self.read(p, digest)


def all_checks(report, count):
    checks = report['checks']
    return (report['all_checks'] is True and len(checks) == count
            and report.get('check_count', count) == count
            and all(v is True for v in checks.values()))


def by_arm(rows):
    require(len(rows) == 3 and {r['arm'] for r in rows} == set(ARMS), 'three unique arms required')
    return {r['arm']: r for r in rows}


def evaluate(data, frames, commands):
    """Arithmetic/schema checks on already SHA-bound inputs, also fixture-testable."""
    e, o, g, c, f, r, a, ready = (data[k] for k in
        ('executor', 'outcome', 'geometry', 'compiler', 'fd', 'ridge', 'fd_audit', 'ready'))
    checks = {}

    def check(key, condition):
        checks[key] = bool(condition)

    check('outcome_542_checks', all_checks(o, 542) and o['status'] == 'c1_saved_physical_outcomes_audited'
          and o['source_data_hashes_unchanged'] is True)
    check('geometry_60_checks', all_checks(g, 60) and g['status'] == 'saved_native_geometry_consistent')
    check('FD_91_checks', all_checks(a, 91) and a['status'] == 'saved_fd_arithmetic_consistent')
    check('ridge_saved_arithmetic', r['status'] == 'saved_prospective_ridge_consistent'
          and r['arithmetic']['all_checks'] is True and bool(r['arithmetic']['checks'])
          and all(v is True for v in r['arithmetic']['checks'].values()))
    check('compiler_file_links', o['authority']['compiler_receipt_file_sha256'] == SPECS['compiler'][1]
          == f['compiler_receipt_sha256'] == r['compiler_receipt_sha256'])
    check('FD_file_link', a['fd_receipt_sha256'] == SPECS['fd'][1])
    check('ready_manifest_file_link', e['ready_manifest_file_sha256'] == SPECS['ready'][1])
    check('same_chart_FD', f['compiler_manifest_sha256'] == c['manifest_file_sha256']
          and f['compiler_inputs_sha256'] == c['compiler_inputs_sha256']
          and f['model_before'] == c['model_before'])
    for key, rec in [('compiler', c), ('FD', f)]:
        check(key + '_completed_no_training', rec['attempted'] == rec['completed']
              and rec['counts_match'] is True and rec['error'] is None
              and rec['finalization_errors'] == [] and rec['partial_work_unknown'] is False
              and rec['training'] is False and rec['physical_calls'] == 0
              and rec['model_before'] == rec['model_after'] and rec['used_observed_frame15'] is False)
        check(key + '_strict_frozen_model', set(rec['model_before']) ==
              {'official_encoder', 'predictor', 'proprio_encoder'}
              and all(v['eval'] is True and v['frozen'] is True and v['all_cuda_finite'] is True
                      and len(v['state_sha256']) == 64 for v in rec['model_before'].values())
              and all(v['exact_match'] is True for v in rec['strict_loads'].values()))
    check('compiler_census_replay', c['status'] == 'selected_packet_replayed'
          and c['completed'] == ready['contract']['compiler']['expected_completed']
          and c['selected_replay_exact'] is True and c['source_check_errors'] == [])
    check('FD_census', f['status'] == 'native_JVP_FD_comparison_recorded'
          and f['completed'] == dict(predictor=40, autograd_grad=0, jvp_columns=0,
                                    official_encoder=0, official_rgb_frames=0, proprio_encoder=0))
    check('FD_no_posthoc_acceptance', set(f['comparisons']) == {'0.005', '0.01'}
          and all(v['accepted'] is None and v['acceptance_threshold'] is None
                  and len(v['columns']) == 10 for v in f['comparisons'].values()))
    check('selection_unchanged', c['selected_env_packet'] == o['authority']['selected_env_packet']
          == ready['selected_packet']['packet'] and c['selected_id'] == o['authority']['selected_id'])
    check('single_root_layout', ready['contract']['decision_frame'] == 10
          and ready['contract']['terminal_frame'] == 15 and ready['contract']['seed'] == o['protocol']['seed']
          and len({b['root_identity_sha256'] for b in ready['branches']}) == 1)
    check('execution_complete', e['status'] == 'native_c1_three_branch_execution_recorded'
          and e['global_error'] is None and e['physics_executed'] is True and e['neural_executed'] is False)
    check('census_totals', all(o['counts'][k] == v for k, v in ready['contract']['execution']['expected'].items()))
    er, ore, gr, br = (by_arm(v) for v in (e['rows'], o['rows'], g['rows'], ready['branches']))
    observations = []
    for arm in ARMS:
        x, y, z = er[arm], ore[arm], gr[arm]
        m = y['metrics']
        fs = frames[arm]
        require(len(fs) == 16 and len(commands[arm]) == 15, 'complete saved frame/command slots')
        check(arm + '_complete_prefix_close', x['status'] == y['status'] == 'completed'
              and x['frames'] == y['frame_count'] == 16 and x['prefix_gate_frames'] == list(range(11))
              and len(y['prefix_matches']) == 11
              and [q['frame'] for q in y['prefix_matches']] == list(range(11))
              and all(q['all5_byte_exact'] is True for q in y['prefix_matches'])
              and y['interrupted_intents'] == [] and y['close_failures'] == []
              and x['close_receipt'] == y['close_receipt']
              and all(y['close_receipt'][k] is True for k in ('call_attempted', 'call_completed', 'return_durable'))
              and y['close_receipt']['status'] == 'completed' and x['error'] is None and y['error'] is None)
        contacts = [s['env']['n_contact_points'] for s in fs]
        require(all(type(v) is int and v >= 0 for v in contacts), 'nonnegative integer contacts')
        check(arm + '_contacts_crosscheck', contacts == z['contacts'] == m['contact_points_frames0_to15'])
        block = [s['bodies']['block'] for s in fs]
        stationary = all((v['position'], v['angle']) == (block[10]['position'], block[10]['angle']) for v in block[10:])
        check(arm + '_object_motion_crosscheck', stationary == z['object_suffix_stationary']
              == m['object_pose_exactly_stationary_since_decision']
              and [v['position'] for v in block[10:]] == m['object_suffix_xy']
              and [v['angle'] for v in block[10:]] == m['object_suffix_angles_native'])
        packet = br[arm]['prefix_env_commands'] + br[arm]['suffix_env_packet']
        check(arm + '_raw_action_binding', [q['raw_env_argument'] for q in commands[arm]] == packet)
        check(arm + '_native_units', all(s['env']['settings']['relative'] is True
              and s['env']['settings']['action_scale'] == 100 and s['env']['sim_hz'] == 100
              and s['env']['control_hz'] == 10 and s['env']['seed'] == ready['contract']['seed'] for s in fs))
        check(arm + '_geometry_summary', len(z['frame_clearances']) == 16
              and z['decision_clearance'] == z['frame_clearances'][10]
              and z['pd_position_max_abs_error'] == z['pd_velocity_max_abs_error'] == 0)
        for label, idx in [('decision', 10), ('terminal', 15)]:
            goal = ready['goal_pose']
            xy = math.dist(block[idx]['position'], goal[:2])
            angle = abs(math.atan2(math.sin(block[idx]['angle'] - goal[2]), math.cos(block[idx]['angle'] - goal[2])))
            check(arm + '_' + label + '_task_metric', math.isclose(xy, m[label]['object_xy_goal_error'], rel_tol=0, abs_tol=1e-10)
                  and math.isclose(angle, m[label]['object_angle_goal_error'], rel_tol=0, abs_tol=1e-10)
                  and m[label]['goal_pose'] == goal)
        observations.append(dict(arm=arm, prefix_contact_points_sum=sum(contacts[1:11]),
            suffix_contact_points_sum=sum(contacts[11:]), object_suffix_stationary=stationary,
            decision_clearance_native=z['decision_clearance'],
            min_reconstructed_suffix_clearance_native=z['min_reconstructed_suffix_substep_clearance'],
            terminal=m['terminal'], agent_suffix_path_native=m['agent_suffix_path_px']))
    differences = {}
    for arm in ARMS[1:]:
        differences[arm] = {k: ore['compiled']['metrics']['terminal'][k] - ore[arm]['metrics']['terminal'][k]
            for k in ('object_xy_goal_error', 'object_angle_goal_error', 'native_final_coverage')}
    check('paired_metrics_arithmetic', differences == o['comparison']['compiled_minus_control'])
    check('no_upstream_scientific_admission', all(v['P5_admission'] is False for v in data.values()))
    return dict(checks=checks, all_checks=all(checks.values()), observations=observations,
                compiled_minus_controls=differences,
                root_identity_sha256=ready['branches'][0]['root_identity_sha256'],
                model_state_identities=c['model_before'],
                FD_matrix_diagnostics={eps: v['matrix'] for eps, v in f['comparisons'].items()})


def blockers():
    # These are missing evidence obligations, never fitted numerical success gates.
    return [dict(id=k, status='not_established_in_fixed_snapshot', required_evidence=v) for k, v in [
        ('native_contact_task_witness', 'Main-owned NEW preregistered task/root protocol: legal native controls, contact then object response and task reach witness; current suffix is noncontact. Prefix contact is not suffix manipulation. Preserve failures; do not select roots by method outcomes.'),
        ('task_metric_alignment', 'Independent pre-outcome validation that the deployable goal/response metric tracks object task progress under contact. Mixed-scene patch MSE and same-model JVP/FD agreement do not establish physical contact derivatives.'),
        ('same_root_C2_conjunction', 'On the same admitted root, deployable non-greedy selection changes future reachable capability AND improves the actual object task against frozen greedy/nominal and strong matched controls; use original preregistered margin/CI gates, not fitted beta or contact thresholds.'),
        ('independent_calibration_outcome', 'Seal separate calibration/outcome root identities, native start/goal distribution, horizon/termination, legal controls and all cost units before new outcomes. This opened single root is diagnostic, not an independent panel.'),
        ('architecture_provenance_and_replication', 'Frozen independent architecture/weights and lawful source provenance, model-specific P0/task prerequisites, same core program/mechanism and real benefit replicated on independent tasks/roots. Local strict-load identity is not official provenance or P5 replication.'),
        ('matched_baseline_resource_contract', 'Bind strong baselines to the same checkpoint/root/goal/feedback/action domain; report predictor, encoder, JVP/VJP/backward, searches, loading and wall time separately. No paper-reported numbers as paired local outcomes.'),
    ]]


def audit(root=ROOT):
    report = dict(schema=NAME, scope='fixed already-opened native C1 snapshot; no execution authority',
        P5_admission=False, C2_established=False, theory_failure=False, execution_authorized=False,
        neural_calls=0, physics_calls=0, training=False, blockers=blockers())
    reader = Reader(root)
    try:
        data = {k: reader.read(p, h) for k, (p, h) in SPECS.items()}
        bindings = data['geometry']['input_bindings']
        expected = {str(reader.root / EXEC / arm / f'frame_{i:02d}' / 'full_public_state.json')
                    for arm in ARMS for i in range(16)}
        expected |= {str(reader.root / EXEC / arm / f'command_{i:02d}.json') for arm in ARMS for i in range(1, 16)}
        expected.add(str(reader.root / SPECS['executor'][0]))
        require(set(bindings) == expected, 'exact geometry input closure (94 files)')
        raw = {p: reader.read(p, h) for p, h in bindings.items()}
        frames = {arm: [raw[str(reader.root / EXEC / arm / f'frame_{i:02d}' / 'full_public_state.json')]
                        for i in range(16)] for arm in ARMS}
        commands = {arm: [raw[str(reader.root / EXEC / arm / f'command_{i:02d}.json')]
                          for i in range(1, 16)] for arm in ARMS}
        report.update(evaluate(data, frames, commands))
        reader.stable()
        report['check_count'] = len(report['checks'])
        report['status'] = 'evidence_consistent_scientific_admission_blocked' if report['all_checks'] else 'invalid_evidence'
        report['failed_checks'] = [k for k, v in report['checks'].items() if not v]
        report['stage_gates'] = dict(
            saved_engineering_snapshot='verified' if report['all_checks'] else 'invalid',
            native_suffix_contact='observed' if any(v['suffix_contact_points_sum'] > 0 for v in report['observations']) else 'not_observed',
            native_suffix_object_response='observed' if any(not v['object_suffix_stationary'] for v in report['observations']) else 'not_observed',
            same_root_C2='not_established', independent_architecture_P5='not_established')
    except (ValueError, OSError, KeyError, TypeError, IndexError, OverflowError, RecursionError) as exc:
        report.update(status='invalid_evidence', all_checks=False, error=f'{type(exc).__name__}: {exc}')
    report['input_files_checked'] = len(reader.bindings)
    report['bytes_read_including_drift_check'] = reader.bytes_read
    report['snapshot_files'] = {k: dict(path=p, sha256=h) for k, (p, h) in SPECS.items()}
    report['limitation'] = ('Consumes SHA-pinned existing audits; rehashes geometry JSON closure and recomputes '
        'contact/motion/paired-metric arithmetic. Does not rerun upstream journal, NPY, weights or full source-seal audits. '
        'Reconstructed substeps are not observations or continuous swept-path proof. No new numeric efficacy thresholds.')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='exclusive NEW same-stem JSON, default stdout')
    parser.add_argument('--require-admission', action='store_true')
    args = parser.parse_args(argv)
    if args.output:
        require(args.output.name == NAME + '.json', 'output must use exact task stem')
        require(not any(p.is_symlink() for p in (args.output.absolute(), *args.output.absolute().parents)), 'symlink output')
        require(not args.output.exists(), 'output already exists')
    result = audit()
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(encoded)
    print(encoded, end='')
    return 2 if not result['all_checks'] else (3 if args.require_admission else 0)


if __name__ == '__main__':
    raise SystemExit(main())
