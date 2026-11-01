"""All8 saved lineage and objective separation; no neural/physics runtime."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import agent_p2_midpoint_teacher_suffix_20261028_v1 as suffix

NAME = 'agent_p2_midpoint_history_lineage_20261028_v1'
mid, gap = suffix.mid, suffix.mid.gap
ROOT, ARMS = suffix.ROOT, suffix.ARMS
require, read, sha, digest = suffix.require, suffix.read, suffix.sha, suffix.digest
INPUTS = {
    'report': ('results/agent_p2_midpoint_teacher_suffix_20261028_v1_cuda_MAIN01/report.json',
               '08bec783a5791c15a03396131df4693e3d4c63c43c9523c38f3e499d65e9fe66'),
    'strict': ('results/agent_p2_midpoint_teacher_suffix_20261028_v1_consume_MAIN01.json',
               '36864d57aa6914f0d09ff4e947ef141ccf0037a2fc23f9d7e3f414fb217d04ac'),
    'audit': ('results/audit_p2_suffix_MAIN_20261028_v1_MAIN01.json',
              '90c79b18ecf23f1cb9bb9298e6147ecb4a3e393d8f92c51c7b7a13396af6366d'),
}
FROZEN = dict(zip((ROOT / f'experiments/spectral_reachability/{suffix.NAME}.py',
                  ROOT / f'tests/test_{suffix.NAME}.py', ROOT / f'innerdocs/{suffix.NAME}.md'),
                 ('ed4954aed9b705ddb4074c150b2b64c2ea9cd20a3abc6c7a2f7157839765eae7',
                  '855e8ccab76bae9dc457ec11c38fcaab725c014b9c765908ce8fcfc22ac8d63f',
                  'b8850bf5c7f3e67bd3dce69717aeeb7f7dc9ff4f6960c1c1119d40a2be6ace5d')))
PROTOCOL = ROOT / 'results/p2_composition_predict_frontier_cuda_20261027_v2_prepared01/protocol.json'


def function_evidence(path, name):
    """Read source only, including sealed historical bytes; never import it."""
    source = mid.regular(path).read_text()
    nodes = [n for n in ast.walk(ast.parse(source)) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    require(len(nodes) == 1, 'one explicit source function: ' + name)
    n = nodes[0]
    return dict(file=str(path), file_sha256=sha(path), function=name, start_line=n.lineno,
                end_line=n.end_lineno, source='\n'.join(source.splitlines()[n.lineno-1:n.end_lineno]))


def trace_job(job, initial, compiled_input, declaration, physical, paid_buffer, native_input):
    """Exact recurrence identities. Observed origin additionally needs source evidence."""
    lat, history = mid.context(initial)
    suffix.f32(lat.tolist(), (3, 192))
    require(initial == declaration['context'] and declaration['root_id'] == job['root_sha256'], 'declared root/context identity')
    require(compiled_input['emb'] == [initial['latent_history']] and
            compiled_input['actions'][0][1:] == initial['action_history'], 'original compiled encoder context identity')
    suffix.f32(compiled_input['actions'], (1, 3, 10))
    require(job['paid_mid_context_sha256'] == digest(job['paid_mid_context']), 'paid context digest')
    paid, _ = mid.context(job['paid_mid_context'])
    require(np.array_equal(paid_buffer, paid.astype(np.float32)) and
            job['paid_mid_context']['latent_history'][:2] == initial['latent_history'][1:], 'older2 exact source slots1/2')
    require(job['teacher_mid_context']['latent_history'][:2] == initial['latent_history'][1:] and
            job['teacher_mid_context']['action_history'] == job['paid_mid_context']['action_history'], 'no extra history intervention')
    expected, buffers = suffix.intervention(job['paid_mid_context'], job['teacher_mid_context']['latent_history'][-1],
                                           job['raw10'], job['action_normalization'])
    require(expected == job['teacher_mid_context'] and native_input.dtype == np.float32 and
            native_input.shape == (1, 3, 192) and np.array_equal(native_input[0], buffers['teacher_latents']), 'actual CUDA suffix input identity')
    identity = physical['context_identity']
    require(physical['root_sha256'] == declaration['root_id'] and physical['method'] == job['arm'] and
            declaration['root_rgb_sha256'] == identity['history_rgb_sha256'] and
            declaration['goal_rgb_sha256'] == identity['goal_rgb_sha256'], 'physical same original history/goal SHA')
    mean, std = (suffix.f32(job['action_normalization'][k], (2,)) for k in ('mean', 'std'))
    raw_history = suffix.f32(identity['raw_history'], (3, 2))
    # These are three genuine zero-action packets, not missing data filled with zero.
    require(np.array_equal(raw_history, np.zeros((3, 2), np.float32)), 'saved actual three zero packets')
    tokens = np.tile(((raw_history - mean) / std)[:, None, :], (1, 5, 1)).reshape(3, 10)
    require(np.array_equal(tokens, suffix.f32(compiled_input['actions'][0], (3, 10))) and
            identity['normalized_between_history'] == initial['action_history'], 'warmup raw/time-major/normalizer identity')
    prefix = buffers['paid_latents']  # paid midpoint[-1] is the only newly predicted slot.
    raw = suffix.f32(job['raw10'], (10, 2))
    require(job['raw_suffix5'] == job['raw10'][5:], 'same sealed raw suffix5')
    prefix_token = ((raw[:5] - mean) / std).reshape(10).tolist()
    require(job['paid_mid_context']['action_history'] == [initial['action_history'][-1], prefix_token], 'midpoint incoming actions are A2,A3')
    older = []
    for slot in range(2):
        older.append(dict(midpoint_slot=slot, initial_context_slot=slot+1,
                          original_warmup_primitive_step=10+5*slot, relative_to_planning_root=-5+5*slot,
                          vector_sha256=digest(initial['latent_history'][slot+1]),
                          exact_in_compilation_paid_teacher_native_input=True,
                          origin='observed_RGB_encode_by_sealed_source_path_not_imagined_rollout',
                          original_per_frame_native_encoder_receipt=None))
    return dict(root_slot=job['root_slot'], arm=job['arm'], root_sha256=job['root_sha256'],
                job_binding_sha256=job['job_binding_sha256'], older2=older,
                history_stack_rgb_sha256=identity['history_rgb_sha256'], goal_rgb_sha256=identity['goal_rgb_sha256'],
                initial_context_sha256=digest(initial), paid_mid_context_sha256=digest(job['paid_mid_context']),
                teacher_mid_context_sha256=digest(expected), paid_mid_last_sha256=digest(prefix[-1].tolist()),
                actual_z5_sha256=digest(buffers['actual_z5'].tolist()), incoming_action_labels=['A2', 'A3'],
                outgoing_action_labels=['A2', 'A3', 'A4'], suffix_raw5_sha256=digest(job['raw_suffix5']),
                goal_vector_sha256=digest(job['goal']), metric_id=job['metric_id'], projection_sha256=digest(job['projection']),
                older2_are_predicted_rollout_states=False,
                older2_representation_correctness='unknown; observed origin does not certify representation fidelity',
                per_frame_RGB_to_encoder_buffer_independently_proven=None)


def winner(value):
    if value is None:
        return None
    require(type(value) in (float, int) and np.isfinite(value), 'finite signed difference')
    return 'tie' if value == 0 else ARMS[1] if value < 0 else ARMS[0]


def objective_identity(job, compiled, physical):
    require(job['goal'] == compiled['goal_vectors'][0] and
            job['projection'] == compiled['scoring']['metric']['projection'] and
            job['metric_id'] == compiled['scoring']['metric']['sha256'] == physical['active_metric_id'],
            'exact original goal/Q content/metric identity')


def diagnose(strict_rows, native_rows):
    """Separate same-Q prediction mismatch from Q/native objective mismatch.

    Reuse MAIN-authenticated suffix costs, do not repeat its NPY numerical audit.
    Never subtract quantities across unlike units.
    """
    si, ni = gap.census(strict_rows), gap.census(native_rows)
    rows = []
    for i in range(4):
        for arm in ARMS:
            s, n = si[i, arm], ni[i, arm]
            require(s['root_sha256'] == n['root_sha256'], 'Q/native root join')
            gap.close(s['actual_terminal_Q'], n['prediction']['physical_teacher_Q'], 'unchanged terminal Q')
            require(s['original_open_loop']['predicted_Q'] == n['prediction']['Q_cost'], 'original Q identity')
            rows.append(dict(root_slot=i, arm=arm, root_sha256=n['root_sha256'],
                             Q=dict(original_predicted=s['original_open_loop']['predicted_Q'],
                                    teacher_suffix_predicted=s['teacher_forced_suffix']['predicted_Q'],
                                    actual_terminal=s['actual_terminal_Q']),
                             native=n['observed'], native_goal=n['goal'], initial_state=n['initial_state'],
                             latent_MSE_change_from_MAIN_strict=s['latent_MSE_change']))
    index, pairs = gap.census(rows), []
    for i in range(4):
        j, c = index[i, ARMS[0]], index[i, ARMS[1]]
        require(j['root_sha256'] == c['root_sha256'] and j['native_goal'] == c['native_goal'] and
                j['initial_state'] == c['initial_state'], 'same paired task/native goal/state')
        diffs = {k+'_Q': None if j['Q'][k] is None or c['Q'][k] is None else c['Q'][k]-j['Q'][k] for k in j['Q']}
        for k in ('native7', 'jointXY', 'objectXY', 'agentXY', 'agent_velocity_error', 'raw_angle_error', 'wrapped_angle_diagnostic'):
            x, y = j['native'][k], c['native'][k]
            diffs[k] = None if x is None or y is None else y-x
        lower = {k: winner(v) for k, v in diffs.items()}
        def match(a, b):
            return None if lower[a] is None or lower[b] is None else lower[a] == lower[b]
        contributions = {k: None if j['native'][k] is None or c['native'][k] is None else c['native'][k]-j['native'][k]
                         for k in gap.COMPONENTS}
        sq = None if any(v is None for v in contributions.values()) else sum(contributions.values())
        if sq is not None:
            gap.close(sq, c['native']['native7_sq']-j['native']['native7_sq'], 'native squared decomposition')
        pairs.append(dict(root_slot=i, root_sha256=j['root_sha256'], composition_minus_joint=diffs,
                          lower_is_better=lower, original_prediction_matches_actual_Q=match('original_predicted_Q', 'actual_terminal_Q'),
                          teacher_prediction_matches_actual_Q=match('teacher_suffix_predicted_Q', 'actual_terminal_Q'),
                          actual_Q_matches_native7=match('actual_terminal_Q', 'native7'),
                          actual_Q_matches_jointXY=match('actual_terminal_Q', 'jointXY'),
                          actual_Q_matches_objectXY=match('actual_terminal_Q', 'objectXY'),
                          objectXY_matches_jointXY=match('objectXY', 'jointXY'),
                          native7_squared_contributions=contributions, native7_squared_difference=sq,
                          official_success={a: index[i, a]['native']['native_success'] for a in ARMS}))
    return rows, pairs


def next_protocol(jobs):
    """A fixed proposal, intentionally independent of scores/rank/MSE/outcome."""
    gap.census(jobs)
    require([(j['root_slot'], j['arm']) for j in jobs] == [(i, a) for i in range(4) for a in ARMS], 'canonical proposal population')
    return dict(status='proposal_for_MAIN_review_not_authorized_or_executed', root_denominator=4, arm_denominator=8,
                population=[dict(root_slot=j['root_slot'], arm=j['arm'], root_sha256=j['root_sha256'],
                                 input_binding_sha256=j['job_binding_sha256']) for j in jobs],
                stage_A=dict(question='independent native API execution equivalence at identical saved teacher inputs',
                             intervention='none; independent minimal adapter, same full3 latents/outgoing tokens/native model/precision',
                             endpoints=8, predict_API=8, action_encoder=8, predictor=8, pred_proj=8,
                             encoder=0, projector=0, AD=0, physics=0,
                             comparisons='all native AE/predictor/pred_proj/API full3 buffers; first unequal buffer retained, no averaging/tolerance tuning',
                             decision='exact mismatch localizes implementation/runtime/precision divergence; exact match only rules out this tested execution-route divergence',
                             limitations='does not independently validate the shared action/history protocol or prove dynamics causality'),
                stage_B=dict(question='public rollout versus manual recurrence with actual contiguous history',
                             protocol='same eight tasks/actions; official rollout history_size3, cached observed z[-5,0,5], exactly3 outgoing tokens predicts one packet',
                             public_ABI=dict(cached_emb=[1, 1, 3, 192], genuine_pixels=[1, 1, 3, 3, 224, 224],
                                             action_sequence=[1, 1, 3, 10], history_size=3, n_steps=0,
                                             loop_iterations=1, bootstrap_or_extra_future_packet=False),
                             blocking_dependencies=['matching actual warmup10 and warmup15/decision-root and actual step5 RGB triplet identities',
                                                    'same original transform/native context encode receipt or MAIN-authorized reconstruction with full metering',
                                                    'separately frozen independent public-API adapter and MAIN authorization'],
                             raw_triplet_proven_in_consumed_closure=None,
                             useful_prediction_budget=dict(public_rollout_predict=8, public_action_encoder=8,
                                                           manual_predict=8, manual_action_encoder=8,
                                                           total_predict=16, total_action_encoder=16,
                                                           predictor=16, pred_proj=16),
                             encoder_if_original_full_context_receipt_unavailable=dict(samples=24, projector_samples=24),
                             reconstruction_physics_if_needed='unknown until MAIN inventories genuine raw history; never zero-fill, repeat RGB, or rerender silently',
                             decision='input/context identity failure is protocol/reconstruction failure, not a negative dynamics finding; matching routes with residual terminal error is conditional model/representation discrepancy'),
                immutable='all8 no root/action/mode selection, original goal/Q/normalizer/weights/precision; no new objective, fit or thresholds',
                native_endpoint='reuse original actual terminal only if exact same branch identities; never equate Q and native task',
                P1_admission=False, P2_admission=False, scientific_admission=False)


def load_saved():
    consumed = {}
    def bound(path, expected):
        path = mid.regular(path)
        require(sha(path) == expected, 'bound input/source SHA: ' + str(path))
        consumed[str(path)] = expected
        return path
    def doc(path, expected):
        return read(bound(path, expected))
    for path, expected in FROZEN.items():
        bound(path, expected)
    docs = {k: doc(ROOT/path, h) for k, (path, h) in INPUTS.items()}
    report, strict, audit = (docs[k] for k in ('report', 'strict', 'audit'))
    folder = (ROOT/INPUTS['report'][0]).parent
    # Authenticate old prepare and inputs; no producer run, no suffix NPY score reaudit.
    launch_path = folder/'execution_binding.json'
    launch = read(launch_path)
    p = suffix.verify(launch['prepared'], launch['prepare_sha256'])
    for path, h in {**p['bindings'], **p['sources_sha256'], **p['model_bindings']}.items():
        bound(path, h)
    for path in (launch_path, Path(launch['prepared'])/'prepare.json', Path(launch['prepared'])/'protocol.json'):
        bound(path, sha(path))
    require(report['status'] == 'all8_suffix_predictions_saved' and strict['status'] == 'all8_teacher_suffix_saved_verified_not_admitted' and
            strict['report_sha256'] == INPUTS['report'][1] and strict['prepare_sha256'] == report['prepare_sha256'] == launch['prepare_sha256'] and
            strict['root_denominator'] == 4 and strict['arm_denominator'] == 8 and strict['scientific_admission'] is False,
            'MAIN completed suffix/strict binding')
    require(audit['status'] == 'all8_saved_suffix_arithmetic_independently_verified' and audit['known_roots'] == 4 and
            len(audit['rows']) == 8 and audit['scientific_admission'] is False, 'MAIN independent arithmetic receipt reused, not rerun')
    gap.census(audit['rows'])
    jobs = p['jobs']
    require([(j['root_slot'], j['arm']) for j in jobs] == [(i, a) for i in range(4) for a in ARMS], 'canonical all8 job order')
    compiled_path = read(PROTOCOL)['context']['compilation']
    require(str(PROTOCOL) in consumed and compiled_path in consumed, 'inherited original protocol/compilation binding')
    compiled = read(compiled_path)
    source_seal = compiled['source_seal']
    archive = Path(source_seal['output_dir'])
    manifest = doc(archive/'manifest.json', source_seal['manifest_sha256'])
    doc(archive/'args.json', manifest['args_sha256'])
    sources = {}
    for entry in manifest['files']:
        rel = Path(entry['archive'])
        require(not rel.is_absolute() and '..' not in rel.parts, 'source archive scope')
        path = bound(archive/rel, entry['sha256'])
        require(path.stat().st_size == entry['size'], 'archived source size')
        sources[Path(entry['source']).name] = path
    # Follow SEALED historical source, not whichever dirty current code is installed.
    functions = []
    for file, names in {
        'run_pusht_d2p_metric_program_preflight.py': ['run'],
        'run_pusht_d2p_p1_physical_calibration.py': ['_reset_and_warmup'],
        'preflight_pusht_d2p_p0.py': ['_collect_causal_rgb_context', '_collect_root_anchored_rgb_context', '_frames_tensor'],
        'preflight_pusht_d2p_p1_reconfiguration.py': ['_encode_context'],
    }.items():
        for name in names:
            functions.append(function_evidence(sources[file], name))
    # Packet boundary source is inherited from the completed physical closure;
    # compiler seal did NOT include it transitively. State that evidence limit.
    packet = ROOT/'experiments/spectral_reachability/pusht_d2p_packet_contract.py'
    original_prepare = doc(ROOT/gap.INPUTS['prepare'][0], gap.INPUTS['prepare'][1])
    require(str(packet) in original_prepare['bindings'], 'physical inherited packet source binding')
    bound(packet, original_prepare['bindings'][str(packet)])
    functions += [function_evidence(packet, 'execute'), function_evidence(packet, 'execute_packet')]
    official = [path for path in consumed if path.endswith('/stable_worldmodel/wm/lewm/lewm.py')]
    require(len(official) == 1, 'bound official encode/predict/rollout source')
    functions += [function_evidence(Path(official[0]), name) for name in ('encode', 'predict', 'rollout')]
    ready = {c['root_index']: c['root'] for c in read(PROTOCOL)['context']['cases']}
    ci = {c['root_index']: c for c in compiled['rows']}
    physical = read(ROOT/gap.INPUTS['physical'][0])
    pi = gap.census(physical['rows'])
    plans = gap.census(original_prepare['plans'])
    gap_report = read(mid.GAP)
    require(sha(mid.GAP) == mid.GAP_SHA, 'frozen native/Q gap analysis')
    rows = []
    require(len(report['rows']) == 8 and len(report['entries']) == 32, 'all8 suffix saved native denominator')
    for index, job in enumerate(jobs):
        i, arm = job['root_slot'], job['arm']
        branch = pi[i, arm]['physical']
        plan = plans[i, arm]['execution_plan']
        require(ci[i]['root_sha256'] == job['root_sha256'] and plan['root_sha256'] == job['root_sha256'] and
                plan['commands'] == job['raw10'], 'same compilation/root/actions')
        objective_identity(job, ci[i]['compiled'], branch)
        r = report['rows'][index]
        require(r['job_id'] == job['job_id'] and r['job_binding_sha256'] == job['job_binding_sha256'] and
                r['status'] == 'predicted_saved', 'actual job identity')
        entry = report['entries'][index*4+1]
        require(entry['job_id'] == job['job_id'] and entry['kind'] == 'predict' and entry['saved'] is True, 'actual native predict entry')
        def array(receipt):
            bound(folder/receipt['file'], receipt['sha256'])
            return mid.load_array(folder, receipt)
        paid = array(r['arrays']['paid_latents'])
        actual_input = array(entry['arrays']['input_0'])
        row = trace_job(job, plan['predictor_context'], ci[i]['compiled']['input_context'], ready[i], branch, paid, actual_input)
        row['actual_native_input_receipt'] = entry['arrays']['input_0']
        rows.append(row)
    diagnostic_rows, pairs = diagnose(strict['rows'], gap_report['rows'])
    result = dict(schema=NAME+'.analysis', status='all8_saved_lineage_and_objectives_separated_not_admitted',
                  root_denominator=4, arm_denominator=8, lineage=rows, diagnostic_rows=diagnostic_rows, pairs=pairs,
                  source_path_evidence=functions,
                  evidence_scope=dict(older2='exact original encoded-context JSON -> initial context -> paid mid -> teacher mid -> actual CUDA input NPY',
                                      observed_origin='sealed compiler/helper code path, plus matching original physical history stack SHA',
                                      original_per_frame_encoder_native_receipts=None,
                                      compiler_seal_transitively_covers_packet_helper=False,
                                      packet_helper='separately authenticated physical inherited source, not claimed original compiler transitive seal',
                                      arithmetic='MAIN strict and independent suffix audit authenticated and reused; no duplicate suffix score audit'),
                  units=dict(Q='original frozen projected-latent MSE; not native distance',
                             native7='unscaled mixed position/radian/velocity norm', jointXY='agent+object native position norm',
                             objectXY='object native position norm only, not complete native success'),
                  interpretation=['older2 are not imagined rollout states; do not propose fixing nonexistent earlier rollout error',
                                  'observed origin and exact transfer do not prove encoder representation or history protocol fidelity',
                                  'prediction-to-teacher-Q mismatch and teacher-Q-to-native mismatch are distinct; no cross-unit subtraction',
                                  '8 lower latent MSE values do not imply all4 pair ranks or native success improve',
                                  'four known roots do not certify freshness, causal generalization or theoretical impossibility'],
                  next_fixed_protocol=next_protocol(jobs),
                  consumed_sha256=consumed, historical_suffix_counts=report['counts'],
                  source_files_sha256={str(path): sha(path) for path in (
                      Path(__file__).resolve(), ROOT/f'tests/test_{NAME}.py', ROOT/f'innerdocs/{NAME}.md')},
                  new_neural_calls=0, new_physics_steps=0, P1_admission=False, P2_admission=False, scientific_admission=False)
    for path, expected in consumed.items():
        require(sha(path) == expected, 'end-of-read input drift')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='optional NEW JSON; otherwise stdout, no writes')
    args = parser.parse_args()
    result = load_saved()
    if args.output is None:
        print(gap.encode(result).decode())
    else:
        mid.write(args.output, result)
        print(gap.encode(dict(output=str(args.output.absolute()), sha256=sha(args.output))).decode())


if __name__ == '__main__':
    main()
