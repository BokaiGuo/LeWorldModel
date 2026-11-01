"""Read-only saved-P4 interface/accounting audit; NEVER scientific admission.

Only stdlib/NumPy arithmetic and source-seal verification. No torch, model,
simulator, training, new outcome selection, downloads, or historical rewrites.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np

from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal

TAG = 'p4_feedback_admission_20261027_v1'
SOURCE = 'p4_feedback_matched_20261027_v2'
SOURCE_RUN = SOURCE+'_cuda_main_v1'
ARMS = ('imagined', 'actual')
ROOTS = (
    '9173218f71898716eac8fe0f3946b8e2cf20527ac71d84555c20c4e1b7fbf333',
    '23b8de7837c39fb4e0fcffe7678732a9d840c57de7a0c8376e22076770ef6050',
    '4afefa3be02a327aa69c22f38c17376d82340ba352aaf4aa29f95b070f5503ca',
    'cf1b9bb913bf7815c9184d5de649b77bf09cc268dd80905a0feca9bd2a2edbf2',
)
MEAN = np.asarray([-.0087, .0068], dtype=np.float32)
STD = np.asarray([.2019, .2002], dtype=np.float32)
TEST_COMMAND = ('.venv/bin/python -B -m pytest -q -p no:cacheprovider '
                'tests/test_p4_feedback_admission_20261027_v1.py')


def require(value, message):
    if not value:
        raise ValueError(message)


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key: '+key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('nonfinite JSON constant: '+value)
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    # Also reject overflow such as 1e999 accepted by JSON's float parser.
    json.dumps(value, allow_nan=False)
    return value


def content_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def rgb_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def array(value, shape):
    raw = np.asarray(value)
    require(raw.shape == shape and raw.dtype.kind in 'fiu' and np.isfinite(raw).all(),
            f'finite numeric array {shape} required')
    return raw


def integer(value, expected=None):
    require(type(value) is int and value >= 0 and (expected is None or value == expected),
            f'exact nonnegative integer required: {expected}')
    return value


def normalize(raw):
    raw = np.asarray(raw)
    require(raw.dtype.kind in 'fi' and raw.shape[-2:] == (5, 2)
            and np.isfinite(raw).all() and np.all(np.abs(raw) <= 1), 'legal raw time-major packets required')
    # Same float32 operation order as the sealed CUDA adapter; no extra /100.
    raw = raw.astype(np.float32).reshape(*raw.shape[:-2], 10)
    return (raw-np.tile(MEAN, 5))/np.tile(STD, 5)


def context(value):
    require(value['layout'] == 'states_with_between_state_actions_v1', 'between-state layout required')
    for field, expected in (('context_length', 3), ('latent_dim', 192), ('action_dim', 10)):
        integer(value[field], expected)
    return array(value['latent_history'], (3, 192)), array(value['action_history'], (2, 10))


def schedule(index):
    integer(index)
    require(index < 4, 'four-root index required')
    ordered = [(i, j) for i in range(16) for j in range(16)]
    return [list(ordered[int(i)]) for i in np.random.default_rng(202610274+index).permutation(256)[:80]]


class Inputs:
    """Bound explicit reads and rehash every read at exit; no directory crawling."""
    def __init__(self):
        self.hashes = {}

    def hash(self, path):
        path = Path(path).resolve()
        require(path.is_file(), 'regular input file required: '+str(path))
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024), b''):
                h.update(chunk)
        value = h.hexdigest()
        require(str(path) not in self.hashes or self.hashes[str(path)] == value,
                'input changed during audit: '+str(path))
        self.hashes[str(path)] = value
        return value

    def raw(self, path, limit=16*1024*1024):
        path = Path(path).resolve()
        require(path.is_file() and path.stat().st_size <= limit, 'bounded regular input required: '+str(path))
        raw = path.read_bytes()
        require(len(raw) <= limit, 'input exceeds size limit')
        value = hashlib.sha256(raw).hexdigest()
        require(str(path) not in self.hashes or self.hashes[str(path)] == value, 'input changed during audit')
        self.hashes[str(path)] = value
        return raw

    def json(self, path):
        return strict_json(self.raw(path))

    def npy(self, path, shape, dtype=None):
        value = np.load(io.BytesIO(self.raw(path, 2*1024*1024)), allow_pickle=False)
        array(value, shape)
        if dtype is not None:
            require(value.dtype == dtype, 'saved array dtype differs')
        return value

    def seal(self, receipt):
        output = Path(receipt['output_dir'])
        manifest = self.json(output/'manifest.json')
        require(self.hashes[str((output/'manifest.json').resolve())] == receipt['manifest_sha256'], 'seal manifest binding differs')
        self.raw(output/'args.json')
        for entry in manifest['files']:
            self.hash(entry['source'])
            self.hash(output/entry['archive'])
        return verify_seal(output_dir=output, expected_manifest_sha256=receipt['manifest_sha256'])

    def stable(self):
        before = dict(self.hashes)
        for path in before:
            self.hash(path)
        return True


def accounting(row):
    """Rebuild phase/horizon/context units from scopes AND dispatch events."""
    rounds = row['rounds']
    require(row['status'] in ('two_rounds_completed', 'guarded_abort'), 'unsupported/failed branch status')
    require(len(rounds) in (1, 2) and [r['round'] for r in rounds] == list(range(len(rounds))), 'contiguous actual rounds required')
    records, events = row['predict'], row['resource_events']
    require(len(records) == 43*len(rounds) and len(events) == 44*len(rounds), 'complete per-round scope/event census required')
    summaries = []
    for k in range(len(rounds)):
        group = records[k*43:(k+1)*43]
        dispatch = events[k*44:(k+1)*44]
        encode = dispatch[0]
        require(encode['kind'] == 'rgb_encode' and encode['completed'] is True, 'actual RGB encode event required')
        integer(encode['frames'], 3)
        for j, (record, event) in enumerate(zip(group, dispatch[1:], strict=True)):
            phase, h = ('planning', 2) if j < 40 else ('evaluation', (2, 1, 1)[j-40])
            require(record['phase'] == phase and record['method'] == row['arm'] and record['completed'] is True,
                    'scope phase/method/completion mismatch')
            for field, n in (('round', k), ('horizon_tokens', h), ('predict_calls', h),
                             ('predict_samples', h), ('failed_predict_calls', 0)):
                integer(record[field], n)
            require(record['context_lengths'] == {'3':h} or record['context_lengths'] == {3:h}, 'actual context3 invocation histogram differs')
            require(len(record['context_lengths']) == 1, 'unique context histogram bucket required')
            integer(next(iter(record['context_lengths'].values())), h)
            require(event['kind'] == 'predict_dispatch' and event['phase'] == phase
                    and event['completed'] is True, 'dispatch event differs')
            integer(event['round'], k); integer(event['tokens'], h)
        for event in dispatch:
            require(type(event['wall_seconds']) in (int, float) and np.isfinite(event['wall_seconds'])
                    and event['wall_seconds'] >= 0, 'finite nonnegative event time required')
        summaries.append(dict(round=k, candidate_pairs=40, planning_predict_calls=sum(x['predict_calls'] for x in group[:40]),
            planning_predict_samples=sum(x['predict_samples'] for x in group[:40]),
            evaluation_predict_calls=sum(x['predict_calls'] for x in group[40:]),
            evaluation_predict_samples=sum(x['predict_samples'] for x in group[40:]),
            processed_context_tokens=sum(3*x['predict_samples'] for x in group),
            rgb_encode_calls=1, rgb_encode_frames=3, action_encoder_calls=84, action_encoder_samples=84,
            physical_step_attempts=integer(rounds[k]['execution']['attempted_commands']),
            physical_step_returns=integer(rounds[k]['execution']['completed_commands']),
            recorded_dispatch_wall_seconds=sum(x['wall_seconds'] for x in dispatch[1:]),
            recorded_encode_wall_seconds=encode['wall_seconds']))
    n = len(rounds)
    for field, expected in (('predict_planning_calls', 80*n), ('predict_planning_samples', 80*n),
                            ('predict_evaluation_calls', 4*n), ('predict_evaluation_samples', 4*n),
                            ('image_encode_calls', n), ('image_encode_frames', 3*n)):
        integer(row[field], expected)
    require(set(row['action_encoder_work']) == {'calls','samples'}, 'action encoder hook fields differ')
    for value in row['action_encoder_work'].values(): integer(value, 84*n)
    integer(row['physics_steps'], sum(r['execution']['completed_commands'] for r in rounds))
    require(type(row['wall_seconds']) in (int,float) and np.isfinite(row['wall_seconds'])
            and row['wall_seconds'] >= 0, 'actual branch wall time required')
    return summaries


def round_interface(*, decision, rgb, between, terminal, execution, bank, compiled,
                    expected_schedule, previous=None, previous_rgb=None, previous_between=None,
                    previous_terminal=None):
    """Saved numeric/identity/shift checks, not model forward or RGB re-encoding."""
    k = integer(decision['round'])
    require(k in (0, 1) and decision['arm'] in ARMS, 'explicit arm/round required')
    require(decision['schedule'] == expected_schedule, 'same frozen per-root per-round candidate schedule required')
    integer(decision['search_horizon_tokens'], 2)
    obs = decision['observed']
    require(rgb.shape == (3,224,224,3) and rgb.dtype == np.uint8
            and terminal.shape == (224,224,3) and terminal.dtype == np.uint8, 'actual uint8 RGB endpoints required')
    require(obs['frame_sha256'] == [rgb_sha(f) for f in rgb]
            and obs['stack_sha256'] == rgb_sha(rgb), 'saved RGB frame/whole-stack hash differs')
    require(np.array_equal(array(obs['between_raw_packets'], (2,5,2)), between), 'saved raw between packets differ')
    observed_z, observed_a = context(obs['encoded_context'])
    planning_z, planning_a = context(decision['planning_context'])
    require(obs['context_sha256'] == content_sha(obs['encoded_context'])
            and decision['planning_context_sha256'] == content_sha(decision['planning_context']), 'context content SHA differs')
    require(np.array_equal(observed_a, normalize(between)), 'action normalization/history shift mismatch')
    uses_observed = not (decision['arm'] == 'imagined' and k == 1)
    require(decision['observed_context_used_for_selection'] is uses_observed, 'feedback-source selection flag differs')
    if k == 0:
        require(previous is None and decision['previous_decision_sha256'] is None, 'round0 must not have previous decision')
        expected_actions = array(compiled['input_context']['actions'], (1,3,10))[0,1:]
        require(np.array_equal(observed_a, expected_actions), 'initial normalized action history differs from compilation')
    else:
        require(previous is not None, 'round1 requires complete previous physical packet')
        require(np.array_equal(rgb, np.concatenate((previous_rgb[1:], previous_terminal[None]))), 'RGB history is not old last2 plus REAL endpoint')
        raw = np.stack((previous_between[1], previous['executed_packet']))
        require(np.array_equal(between, raw), 'between shift must be old last packet plus executed raw packet')
    wanted = obs['encoded_context'] if uses_observed else previous['model_next_context']
    require(decision['planning_context'] == wanted, 'wrong actual/imagined planning boundary')
    selected = decision['selected']
    path = selected['path']
    require(path in expected_schedule and len(path) == 2 and all(type(i) is int and 0 <= i < 16 for i in path), 'full H2 selected path required')
    actions = array(selected['actions'], (10,2))
    require(np.array_equal(actions, np.concatenate([bank[i] for i in path])), 'selected full actions differ from frozen bank')
    integer(decision['executed_bank_index'], path[0])
    packet = array(decision['executed_packet'], (5,2))
    require(np.array_equal(packet, actions[:5]), 'execution must consume selected first packet only')
    mid_z, mid_a = context(decision['model_next_context'])
    end_z, end_a = context(selected['terminal'])
    tokens = normalize(actions.reshape(2,5,2))
    require(np.array_equal(mid_z[:2], planning_z[1:]) and np.array_equal(end_z[0], planning_z[-1])
            and np.array_equal(end_z[1], mid_z[-1]), 'saved latent rollout length/history shift differs')
    require(np.array_equal(mid_a, np.stack((planning_a[1], tokens[0]))) and np.array_equal(end_a, tokens), 'rollout action order/normalization differs')
    require(decision['replay_exact'] == [True, True] and all(type(x) is bool for x in decision['replay_exact']), 'saved independent replay declarations required')
    goal = array(compiled['goal_vectors'][0], (192,))
    q = array(compiled['scoring']['metric']['projection'], (192,192))
    cost = float(np.mean(((end_z[-1]-goal)@q.T)**2))
    require(type(selected['cost']) in (int,float) and np.isclose(selected['cost'], cost, rtol=1e-12, atol=1e-12), 'selected model cost recomputation differs')
    require(execution['status'] in ('completed','observation_unavailable','environment_terminated'), 'failed physical branch retained, not admitted')
    done = integer(execution['completed_commands'])
    attempts = integer(execution['attempted_commands'])
    require(0 <= done <= attempts <= 5, 'actual attempt/return counts differ')
    integer(execution['planned_commands'], 5)
    integer(execution['remaining_commands_not_executed'], 5-done)
    require(len(execution['trace']) == done and len(execution['attempt_before_step']) == attempts, 'physical trace/attempt census differs')
    if execution['status'] == 'completed': integer(done, 5)
    for j, event in enumerate(execution['trace']):
        integer(event['primitive_step'], 5*k+j+1)
        require(np.array_equal(array(event['command'], (2,)), packet[j]), 'executed trace changed sealed raw command')
        array(event['state'], (7,))
    for j, event in enumerate(execution['attempt_before_step']):
        integer(event['primitive_step'], 5*k+j+1)
        require(event['status'] == 'dispatch_authorized_before_env_step'
                and np.array_equal(array(event['command'], (2,)), packet[j]), 'pre-step authorization differs')
    require(execution['terminal_rgb_sha256'] == rgb_sha(terminal), 'persisted real terminal RGB changed')
    if done:
        require(execution['trace'][-1]['rgb_sha256'] == rgb_sha(terminal), 'trace endpoint RGB differs')
    guard = execution['guard']
    require(guard['root_id'] == decision['root_sha256'] and guard['P1_admission'] is False, 'guard root/admission differs')
    integer(guard['primitive_commands_authorized'], attempts)
    require([g['primitive_step'] for g in guard['observations']] == list(range(done+1)), 'guard observation sequence differs')
    if execution['status'] == 'completed':
        require(guard['abstained'] is False and all(g['canonical_agent_pixels'] > 0 for g in guard['observations']), 'completed branch must have visible guard observations')
    return dict(round=k, arm=decision['arm'], selected_path=path, executed_bank_index=path[0],
        observed_context_used=uses_observed, model_cost_recomputed=cost,
        physical_completed_commands=done, physical_attempted_commands=attempts,
        full_packet_completed=execution['status'] == 'completed' and done == 5,
        rgb_history_verified=True, normalization_verified=True, saved_rollout_shift_verified=True,
        new_model_replay=False, rgb_reencoding_verified=False,
        observed_vs_planning_latent_max_abs=float(np.max(np.abs(observed_z-planning_z))))


def audit_candidates(events, decision):
    require(len(events) == 40 and [e['path'] for e in events] == decision['schedule'], 'candidate journal count/order differs')
    for event in events:
        require(event['completed'] is True and type(event['cost']) in (int,float)
                and np.isfinite(event['cost']), 'failed/nonfinite candidate must not be filtered')
        require(np.isfinite(event['wall_seconds']) and event['wall_seconds'] >= 0, 'candidate time invalid')
    ranked = sorted(events, key=lambda e:(e['cost'], tuple(e['path'])))
    require(decision['selected']['path'] == ranked[0]['path'] and decision['selected']['cost'] == ranked[0]['cost'], 'selected path/cost is not stable model-only argmin')
    return dict(winner_path=ranked[0]['path'], margin_to_second=ranked[1]['cost']-ranked[0]['cost'],
        top5_paths=[e['path'] for e in ranked[:5]], recorded_candidate_wall_seconds=sum(e['wall_seconds'] for e in events))


def boundary(interface_passed):
    return dict(P1_admission=False, P4_admission=False, scientific_status='blocked_upstream',
        interface_audit_passed=interface_passed, feedback_ineffective=False, feedback_harm_generalized=False,
        policy_equality_is_null_effect_proof=False, equal_total_compute=False, efficacy_verified=False,
        missing_requirements=[
            'PushT-specific P1 and adopted P2/P3 scientific admission evidence',
            'independent frozen start/goal distribution and root-level paired uncertainty',
            'multi-round RGB feedback to task termination, not only two packets',
            'matched strong controls and complete optimization/latency resource contract',
            'validated task metric plus approach/contact/push/closure mechanism decomposition'],
        next_executable_gap=dict(action='offline frozen contact-stage annotation specification from saved 80 state/action/RGB receipts',
            scope='prepare independent labels/threshold provenance and blinded annotation contract; do not infer contact forces from absent telemetry',
            requires_new_gpu=False, requires_new_physics=False, unlocks_P4=False,
            subsequent_main_only='after upstream admission, freeze a root-disjoint long-horizon same-round-budget feedback panel before outcomes'))


def audit(report_path):
    t = Inputs()
    report_path = Path(report_path).resolve()
    folder = report_path.parent
    report = t.json(report_path)
    errors, rows, pairs, verifications = [], [], [], []
    def check(name, fn):
        try:
            return fn()
        except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
            errors.append(dict(check=name, type=type(exc).__name__, message=str(exc)))
            return None
    protocol = report['protocol']
    def protocol_check():
        require(report['schema'] == SOURCE and protocol['schema'] == SOURCE, 'source schema differs')
        require(report['status'] == 'engineering_calibration_recorded' and not report['failures']
                and report['resources']['setup_failure'] is None, 'completed source run with no runtime failures required')
        require(folder.name == SOURCE_RUN and report_path.name == 'report.json', 'explicit source run/report required')
        require(protocol['roots'] == list(ROOTS) and protocol['arms'] == list(ARMS), 'known-root/arm census differs')
        require(t.json(folder/'protocol.json') == protocol, 'report protocol differs from persisted protocol')
        for key, value in (('rounds',2), ('executed_steps_per_round',5), ('search_horizon_tokens',2), ('seed',202610274)):
            integer(protocol[key], value)
        require(protocol['paths_per_root'] == [schedule(i) for i in range(4)], 'frozen candidate paths/seed differs')
        expected_round = dict(candidate_pairs=40, planning_predict_calls=80, planning_predict_samples=80,
            verification_predict_calls=4, verification_predict_samples=4, rgb_encode_calls=1, rgb_encode_frames=3)
        require(protocol['per_round'] == expected_round, 'declared per-round cap differs')
        require(protocol['per_unaborted_arm'] == {key:2*value for key,value in expected_round.items()
            if key != 'candidate_pairs'}, 'declared complete-branch budget differs')
        for field in ('backward_calls','optimizer_steps','goal_encode_calls'):
            integer(report['resources'][field],0)
        for flag in ('training','readout_fit','P1_admission','P4_admission'):
            require(protocol[flag] is False, 'engineering contract changed: '+flag)
        require(protocol['cuda_matmul_allow_tf32'] is False and report['resources']['cuda_matmul_allow_tf32'] is False, 'TF32 policy differs')
        for flag in ('P1_admission','P4_admission','efficacy_verified','equal_total_compute'):
            require(report[flag] is False, 'old report scientific boundary changed')
        for path, expected in protocol['bindings'].items():
            require(t.hash(path) == expected, 'official input binding changed: '+path)
        expected_roster = [dict(root_index=i, root_sha256=r, arm=a, status='pending_setup') for i,r in enumerate(ROOTS) for a in ARMS]
        require(t.json(folder/'branch_roster.json') == expected_roster, 'pre-setup roster differs')
        require(len(report['rows']) == 8 and {(r['root_sha256'],r['arm']) for r in report['rows']} == {(r,a) for r in ROOTS for a in ARMS}, 'all eight unique branches required')
    check('protocol_and_inputs', protocol_check)
    repo = Path(__file__).resolve().parents[3]
    for name in ('D2P_PARALLEL_CUDA_PROGRESS_20261027.md', 'P45_PREREQUISITE_SNAPSHOT_20261027.md',
                 'SPECTRAL_REACHABILITY_ATLAS_RESEARCH_PLAN_20261023.md'):
        t.raw(repo/'docs'/name)
    def normalizer_source():
        path = Path(__file__).resolve().parents[2]/'experiments/run_pusht_counterfactual.py'
        tree = ast.parse(t.raw(path).decode())
        for name, expected in (('PUBLIC_ACTION_MEAN',MEAN),('PUBLIC_ACTION_STD',STD)):
            node = next(n for n in tree.body if isinstance(n,ast.Assign)
                and any(isinstance(target,ast.Name) and target.id == name for target in n.targets))
            values = np.asarray(ast.literal_eval(node.value.args[0]),dtype=np.float32)
            require(np.array_equal(values,expected), 'normalization constants differ from sealed source')
    check('normalizer_source_constants', normalizer_source)
    seals = report['resources'].get('source_seals', [])
    require(len(seals) == 3, 'three runtime/evaluator source seals required')
    for i, seal in enumerate(seals):
        result = check(f'old_source_seal_{i}', lambda s=seal:t.seal(s))
        if result is not None: verifications.append(result)
    comp_path = next(Path(p) for p in protocol['bindings'] if Path(p).parent.name == 'pusht_d2p_metric_program_preflight4_v1_20261026_cuda')
    compilation = t.json(comp_path)
    require(compilation['protocol']['roots'] == list(ROOTS), 'compilation root order differs')
    compiled = {r['root_sha256']:r['compiled'] for r in compilation['rows']}
    refs_path = next(Path(p) for p in protocol['bindings'] if Path(p).parent.name == 'pusht_d2p_p1_root_anchored_v5_calibration_20261026_cuda')
    refs = {r['root_sha256']:r for r in t.json(refs_path)['rows']}
    banks = array(protocol['banks'], (4,16,5,2))
    require(np.all(np.abs(banks) <= 1), 'bank action bound differs')
    for i, root in enumerate(ROOTS):
        original = np.stack([np.repeat(array(r['program_live_commands'][0], (2,))[None],5,axis=0) for r in compiled[root]['receipts']])
        require(np.array_equal(banks[i], original), 'P4 bank differs from saved compilation')
    decision_map, summaries, rowmap = {}, {}, {}
    for row in report['rows']:
        root, arm, i = row['root_sha256'], row['arm'], row['root_index']
        rowmap[root,arm] = row
        def branch():
            integer(i, ROOTS.index(root)); require(arm in ARMS, 'unknown arm')
            require(t.json(folder/f'root_{i}.{arm}.json') == row, 'branch/report content differs')
            require(row['initial_rgb_identity'] == {k:refs[root][k] for k in ('history_rgb_sha256','goal_rgb_sha256')}, 'root history/goal RGB reference differs')
            return accounting(row)
        budget = check(f'branch_{i}_{arm}_accounting', branch)
        summaries[root,arm] = budget
        candidates = [strict_json(line) for line in t.raw(folder/f'root_{i}.{arm}.candidates.jsonl').splitlines()]
        check(f'root_{i}_{arm}_candidate_census',lambda:require(
            len(candidates) == 40*len(row['rounds'])
            and all(type(e['round']) is int for e in candidates)
            and [e['round'] for e in candidates] == [k for k in range(len(row['rounds'])) for _ in range(40)],
            'all candidate journal rounds must be retained in exact schedule order'))
        previous = previous_rgb = previous_between = previous_terminal = previous_execution = None
        round_results = []
        for k, entry in enumerate(row['rounds']):
            prefix = folder/f'root_{i}.{arm}.round_{k}'
            def inspect_round():
                integer(entry['round'], k)
                if k:
                    require(previous_execution is not None and previous_execution['status'] == 'completed'
                            and previous_execution['completed_commands'] == 5, 'partial first packet cannot authorize round1')
                dpath = Path(str(prefix)+'.decision.json')
                decision = t.json(dpath)
                require(t.hash(dpath) == entry['decision_sha256'], 'sealed decision file changed')
                require(decision['root_sha256'] == root and decision['arm'] == arm, 'decision root/arm mismatch')
                integer(decision['root_index'], i); integer(decision['round'], k)
                require(decision['goal_rgb_sha256'] == refs[root]['goal_rgb_sha256'], 'decision goal RGB differs')
                require(decision['protocol_sha256'] == t.hash(folder/'protocol.json')
                        and decision['source_manifest_sha256'] == [s['manifest_sha256'] for s in seals], 'decision protocol/seal binding differs')
                if k: require(decision['previous_decision_sha256'] == row['rounds'][k-1]['decision_sha256'], 'previous decision hash chain differs')
                rgb = t.npy(str(prefix)+'.rgb.npy', (3,224,224,3), np.dtype('uint8'))
                between = t.npy(str(prefix)+'.between.npy', (2,5,2))
                terminal = t.npy(str(prefix)+'.terminal.npy', (224,224,3), np.dtype('uint8'))
                execution = t.json(str(prefix)+'.execution.json')
                require(execution == entry['execution'], 'embedded execution/file differs')
                if k == 0: require(rgb_sha(rgb) == refs[root]['history_rgb_sha256'], 'initial whole-stack identity differs')
                result = round_interface(decision=decision,rgb=rgb,between=between,terminal=terminal,
                    execution=execution,bank=banks[i],compiled=compiled[root],expected_schedule=schedule(i)[k*40:(k+1)*40],
                    previous=previous,previous_rgb=previous_rgb,previous_between=previous_between,previous_terminal=previous_terminal)
                result['candidate_ranking'] = audit_candidates([c for c in candidates if c['round'] == k], decision)
                for j, event in enumerate(execution['trace']):
                    require(t.json(str(prefix)+f'.step_{j+1}.json') == event, 'individual step file changed')
                for j, event in enumerate(execution['attempt_before_step']):
                    require(t.json(str(prefix)+f'.attempt_{j+1}.json') == event, 'individual attempt file changed')
                decision_map[root,arm,k] = decision
                return decision,rgb,between,terminal,execution,result
            result = check(f'root_{i}_{arm}_round_{k}', inspect_round)
            if result is not None:
                previous,previous_rgb,previous_between,previous_terminal,previous_execution,value = result
                round_results.append(value)
            else:
                previous = previous_execution = None
                round_results.append(dict(round=k,status='interface_failed'))
        if row['status'] == 'two_rounds_completed':
            check(f'root_{i}_{arm}_full_completion', lambda:require(len(round_results) == 2
                and all(r.get('full_packet_completed') is True for r in round_results), 'full completion requires two complete audited packets'))
        rows.append(dict(root_index=i,root_sha256=root,arm=arm,reported_status=row['status'],
            rounds=round_results,recomputed_costs=budget,saved_final_official=row.get('final_official'),
            saved_final_pose=row.get('final_pose'),recorded_branch_wall_seconds=row.get('wall_seconds')))
    for i, root in enumerate(ROOTS):
        for k in range(2):
            def compare():
                left,right = [decision_map[root,a,k] for a in ARMS]
                require(left['schedule'] == right['schedule'], 'same root/round schedule mismatch')
                a,b = [summaries[root,arm][k] for arm in ARMS]
                budget_fields = [key for key in a if 'wall_seconds' not in key]
                require(all(a[key] == b[key] for key in budget_fields), 'per-round resource vector mismatch')
                dz = np.asarray(right['planning_context']['latent_history'])-np.asarray(left['planning_context']['latent_history'])
                if k == 0:
                    require(left['planning_context'] == right['planning_context'] and left['observed'] == right['observed'], 'round0 must share the identical real boundary')
                return dict(root_index=i,root_sha256=root,round=k,interface_and_recorded_budget_matched=True,
                    search_horizon_tokens=2,planning_context_max_abs=float(np.max(np.abs(dz))),
                    planning_context_mse=float(np.mean(dz**2)),
                    normalized_action_histories_equal=left['planning_context']['action_history'] == right['planning_context']['action_history'],
                    selected_paths_equal=left['selected']['path'] == right['selected']['path'],
                    executed_packets_equal=left['executed_packet'] == right['executed_packet'],
                    same_policy_implies_feedback_ineffective=False)
            result = check(f'paired_root_{i}_round_{k}', compare)
            pairs.append(result or dict(root_index=i,root_sha256=root,round=k,interface_and_recorded_budget_matched=False))
    stable = check('all_read_inputs_unchanged', t.stable) is True
    passed = not errors and stable
    outcome_pairs = []
    for i, root in enumerate(ROOTS):
        imagined, actual = [rowmap[root,a] for a in ARMS]
        outcome_pairs.append(dict(root_index=i,root_sha256=root,
            saved_pose_cost={a:rowmap[root,a]['final_pose']['pose_cost'] for a in ARMS},
            saved_official_distance={a:rowmap[root,a]['final_official']['official_state_distance'] for a in ARMS},
            saved_official_success={a:rowmap[root,a]['final_official']['official_success'] for a in ARMS},
            pose_cost_actual_minus_imagined=actual['final_pose']['pose_cost']-imagined['final_pose']['pose_cost'],
            official_distance_actual_minus_imagined=actual['final_official']['official_state_distance']-imagined['final_official']['official_state_distance'],
            interpretation='descriptive same-root saved labels only; not an independent held-out effect or equivalence test'))
    totals = dict(planning_predict_calls=sum(r['predict_planning_calls'] for r in report['rows']),
        planning_predict_samples=sum(r['predict_planning_samples'] for r in report['rows']),
        evaluation_predict_calls=sum(r['predict_evaluation_calls'] for r in report['rows']),
        evaluation_predict_samples=sum(r['predict_evaluation_samples'] for r in report['rows']),
        rgb_encode_calls=sum(r['image_encode_calls'] for r in report['rows']),
        rgb_encode_frames=sum(r['image_encode_frames'] for r in report['rows']),
        physical_step_attempts=sum(r['execution']['attempted_commands'] for row in report['rows'] for r in row['rounds']),
        physical_step_returns=sum(r['execution']['completed_commands'] for row in report['rows'] for r in row['rounds']),
        processed_context_tokens=sum(s['processed_context_tokens'] for group in summaries.values() if group is not None for s in group),
        action_encoder_calls=sum(r['action_encoder_work']['calls'] for r in report['rows']),
        action_encoder_samples=sum(r['action_encoder_work']['samples'] for r in report['rows']),
        physical_step_cost_scope='selected paths only; root collection/reset/warmup excluded',
        actual_cost_scope='recorded executed invocations re-counted; not profiler traces or FLOPs',
        recorded_runtime_wall_seconds=report['resources']['total_wall_seconds'],
        recorded_peak_cuda_allocated_bytes=report['resources']['peak_cuda_allocated_bytes'],
        recorded_peak_cuda_reserved_bytes=report['resources']['peak_cuda_reserved_bytes'],
        unmeasured=report['resources']['not_measured'], equal_total_compute=False)
    return dict(schema=TAG,status='interface_audit_passed_scientific_blocked' if passed else 'integrity_fail',
        reported_source_status=report['status'],reported_source_gates=report['gates'],
        audit_passed=passed,errors=errors,rows=rows,paired_root_rounds=pairs,totals=totals,
        input_sha256=t.hashes,all_read_inputs_unchanged=stable,source_verification=verifications,
        saved_outcome_pairs=outcome_pairs,
        **boundary(passed),
        limitations=['RGB-to-latent values and nonlinear model transitions are saved witnesses, not recomputed neural forwards',
            'candidate costs recomputed only for selected saved terminal; unselected terminal latents were not persisted',
            'environment task labels are copied from mutually consistent saved branch/report, not freshly evaluated',
            'no held-out roots, force/contact telemetry, long-horizon efficacy, FLOPs or uncertainty inference'],
        new_predict_calls=0,new_image_encode_calls=0,new_physics_steps=0)


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def run(*, report, output):
    output = Path(output).resolve()
    require(output.name.startswith(TAG), 'exclusive output basename must start with '+TAG)
    output.mkdir(exist_ok=False)
    own = Path(__file__).resolve()
    files = [own,own.parents[2]/f'tests/test_{TAG}.py',own.parents[2]/f'docs/{TAG}.md',
             Path(verify_seal.__code__.co_filename).resolve()]
    seal = create_seal(files=files,args_json=json.dumps(dict(report=str(Path(report).resolve()),
        task='offline interface/resource/admission audit',new_neural_or_physics_work=False)),output_dir=output/'source_seal')
    try:
        result = audit(report)
    except (OSError,ValueError,TypeError,KeyError,IndexError,StopIteration) as exc:
        result = dict(schema=TAG,status='integrity_fail',audit_passed=False,
            errors=[dict(check='input_or_census',type=type(exc).__name__,message=str(exc))],**boundary(False),
            new_predict_calls=0,new_image_encode_calls=0,new_physics_steps=0)
    result['audit_source_seal'] = seal
    result['audit_source_verification'] = verify_seal(output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256'])
    write(output/'report.json',result)
    text = ('# P4 feedback admission audit\n\n'
        f'Status: {result["status"]}. P1/P4 admission: false.\n\n'
        'Same selected action is not proof of feedback ineffectiveness; a different context is not proof of task benefit.\n\n'
        'Costs are historical recorded invocations/frames/steps, not new inference, equal FLOPs or wall-cap matching.\n\n'
        'Next executable gap: specify offline blinded approach/contact/push/closure labels from saved traces; absent contact-force telemetry remains unknown.\n')
    with (output/'report.md').open('x') as stream:
        stream.write(text); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(dict(status=result['status'],audit_passed=result['audit_passed'],P4_admission=False,
        errors=len(result['errors']),output=str(output)),sort_keys=True))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,default='results/'+SOURCE_RUN+'/report.json')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    outcome = run(**vars(args))
    raise SystemExit(0 if outcome['audit_passed'] else 2)
