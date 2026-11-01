"""Independent saved-only live audit; no live producer, torch or gym imports.

Reconstruct algebra/causality/journal, not neural encodings or physics. Small
report + source/tests/doc <=300KiB. No pytest tmp or large protocol copies.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
from experiments.spectral_reachability import p4_feedback_root_cuda_response_saved_audit_20261027_v1 as algebra

NAME = 'p4_feedback_live_saved_audit_20261027_v1'
REPO = algebra.REPO
DEFAULT = REPO / 'results/p4_feedback_live_cuda_20261027_v1_cuda_main01'
RECEIPT_SHA = 'e3c91195540b7465e2fcbf8431d7f182c06bfdff9da6ab160489ae6fe23bbb47'
PREPARE_SHA = '2762a6d9ee3107da0d27519d5a37aba8c8378b11e615cb6ec4526d81ef548481'
LIVE_SHA = 'c878a8cd6e1c7a5d645221e299993da5cd8e3609c30038ca1a50e4faaecdcaeb'
ARMS = ('imagined', 'actual')
BOUNDARIES = (5, 10, 15)
COUNTS = dict(action_encoder=80, predict=80, encoder=8, projector=8, jacobian_api=6, grad_api=1152, engine_api=1152, backward_api=0)
ROUND_COUNTS = dict(action_encoder=13, predict=13, jacobian_api=1, grad_api=192, engine_api=192)
PHYSICS = dict(root_resets=32, branch_resets=2, warmup_steps=30, selected_steps=40)
require, sha, read, context = algebra.require, algebra.sha, algebra.read, algebra.context
close, history = algebra.close, algebra.history


def same(a, b, label): require(algebra.canonical(a) == algebra.canonical(b), label)


def rgb_sha(rgb): return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()


def cost_metrics(native, predicted, goal, q):
    native = algebra.array(native, (192,)); predicted = algebra.array(predicted, (192,))
    goal = algebra.array(goal, (192,)); q = algebra.array(q, (192, 192)); delta = native-predicted
    return dict(latent_MSE=float(np.mean(delta**2)), Q_prediction_MSE=float(np.mean((q@delta)**2)),
                Q_goal_MSE=float(np.mean((q@(native-goal))**2)))


def physical_metrics(state, goal):
    state = algebra.array(state, (7,)); goal = algebra.array(goal, (7,))
    raw = abs(float(state[4]-goal[4])); native_angle = min(raw, 2*math.pi-raw)
    wrapped = abs(math.atan2(math.sin(state[4]-goal[4]), math.cos(state[4]-goal[4])))
    agent = float(np.linalg.norm(state[:2]-goal[:2])); obj = float(np.linalg.norm(state[2:4]-goal[2:4]))
    pos = float(np.linalg.norm(state[:4]-goal[:4]))
    official = dict(agent_position_error=agent, object_position_error=obj, official_position_error=pos,
        official_state_distance=float(np.linalg.norm(state-goal)), angle_error=native_angle,
        official_success=bool(pos < 20 and native_angle < math.pi/9), object_only_success=bool(obj < 20 and native_angle < math.pi/9))
    diagnostic = dict(raw_angle_difference=raw, wrapped_angle_error=wrapped, native_angle_domain_valid=raw <= 2*math.pi,
        wrapped_official_success=bool(pos < 20 and wrapped < math.pi/9),
        wrapped_object_only_success=bool(obj < 20 and wrapped < math.pi/9),
        native_wrapped_success_agree=official['official_success'] == bool(pos < 20 and wrapped < math.pi/9))
    pose = dict(xy_distance=obj, angle_distance=wrapped, pose_cost=obj+20*wrapped)
    return dict(official=official, pose=pose, wrapped_diagnostic=diagnostic)


def compare_fields(saved, expected, label):
    require(set(saved) == set(expected), label+' fields')
    for k, v in expected.items():
        if isinstance(v, bool): require(type(saved[k]) is bool and saved[k] == v, label+' '+k)
        else: close(saved[k], v, label+' '+k, 1e-9)


def inventory():
    names = {'execution_binding.json', 'events.jsonl'}
    for arm in ARMS:
        names.update(arm+s for s in ('.goal.png', '.reset_identity.json', '.prefix.selection.json', '.prefix.prediction.json', '.step20.native192.npy', '.result.json'))
        for packet in ('warmup0', 'warmup1', 'warmup2', 'from0', 'from5', 'from10', 'from15'):
            stem = arm+'.'+packet; names.add(stem+'.physics.json')
            for i in range(1, 6): names.update((stem+f'.step{i}.json', stem+f'.step{i}.png'))
        for b in BOUNDARIES:
            names.update(arm+f'.step{b}'+s for s in ('.observed.npy', '.observed.json', '.J.npy', '.request.json', '.selection.json'))
    return names


class Files:
    def __init__(self, folder, hashes): self.folder, self.hashes = Path(folder), hashes
    def path(self, name):
        require(Path(name).name == name and name in self.hashes, 'own inventoried filename')
        p = self.folder/name; require(sha(p) == self.hashes[name], 'saved SHA '+name); return p
    def json(self, name): return read(self.path(name))
    def array(self, name, shape):
        a = np.load(self.path(name), allow_pickle=False)
        require(a.dtype == np.float32 and a.shape == shape and np.isfinite(a).all(), 'native float32 array entry '+name)
        return a
    def rgb(self, name):
        with Image.open(self.path(name)) as image:
            require(image.mode == 'RGB' and image.size == (224, 224), 'native PNG')
            return np.array(image, dtype=np.uint8)


def blue_count(rgb): return int(np.count_nonzero(np.all(rgb == [65, 105, 225], axis=-1)))


def packet_check(files, stem, commands, root, previous=None):
    commands = algebra.array(commands, (5, 2)); require(np.all(np.abs(commands) <= 1), 'legal bounded raw actions')
    r = files.json(stem+'.physics.json'); g = r['guard']
    require(r['status'] == 'completed' and r['error'] is None and
        r['attempted_commands'] == r['completed_commands'] == r['planned_commands'] == 5 and r['remaining_commands_not_executed'] == 0, 'five real public steps')
    require(g['root_id'] == root and g['abstained'] is g['P1_admission'] is False and g['primitive_commands_authorized'] == 5 and
        [o['primitive_step'] for o in g['observations']] == list(range(6)) and all(o['canonical_agent_pixels'] > 0 for o in g['observations']), 'fresh six-observation guard grammar')
    if previous is not None: require(blue_count(previous) == g['observations'][0]['canonical_agent_pixels'], 'fresh preceding terminal guard count')
    for i in range(1, 6):
        row = files.json(stem+f'.step{i}.json'); require(row['terminated'] is row['truncated'] is False, 'no filtered termination')
        close(row['command'], commands[i-1], 'executed command identity', 0.)
        name = stem+f'.step{i}.png'; rgb = files.rgb(name); meta = row['rgb']
        require(meta == dict(file=name, file_sha256=files.hashes[name], pixel_sha256=rgb_sha(rgb)), 'own fresh PNG file/pixel binding')
        require(blue_count(rgb) == g['observations'][i]['canonical_agent_pixels'], 'independent RGB guard recount')
    require(rgb_sha(rgb) == r['terminal_pixel_sha256'], 'guard terminal image identity')
    return rgb


def selection_check(p, request_context, s, jac):
    context(request_context); same(s['request']['context'], request_context, 'own planning context')
    require(s['request']['context_sha256'] == algebra.content_sha(request_context), 'request context SHA')
    require(s['fresh_context_specific_AD'] is True, 'fresh AD receipt marker')
    same(s['jacobian_forward'], s['nominal'], 'AD forward/nominal exact')
    history(request_context, s['nominal'], np.zeros((5, 2)), p['normalization'])
    require(np.linalg.norm(jac) > 0, 'nonzero saved derivative')
    params = algebra.bank_check(jac.astype(float), np.asarray(s['nominal']['latent_history'])[-1], np.asarray(p['goal']), np.asarray(p['q']), s['bank'])
    require(len(s['candidates']) == 10, 'ten scored candidates')
    for i, c in enumerate(s['candidates']):
        require(c['index'] == i and c['kind'] == s['bank']['kinds'][i], 'bank roles/order')
        close(c['logits'], params[i], 'bank logits', 0.)
        commands = np.tanh(params[i].astype(np.float32)).reshape(5, 2)
        close(c['commands'], commands, 'float32 tanh action entries', 0.)
        history(request_context, c['context'], commands, p['normalization'])
        score = cost_metrics(np.asarray(c['context']['latent_history'])[-1], np.zeros(192), p['goal'], p['q'])['Q_goal_MSE']
        close(c['cost'], score, 'independent nonlinear original-Q candidate cost', 1e-12)
    winner = min(s['candidates'], key=lambda c: (c['cost'], c['index']))
    same(s['winner'], winner, 'stable minimum winner'); same(s['replay'], winner['context'], 'independent replay full successor equality')
    return winner


def planning_context(arm, predicted, observed):
    require(arm in ARMS, 'arm identity'); result = deepcopy(predicted)
    if arm == 'actual': result['latent_history'] = observed.tolist()
    return result


def expected_journal(hashes):
    """Independent exact nested event grammar; no producer Ledger import."""
    events = []; entries = []; rounds = []; physics = dict.fromkeys(PHYSICS, 0)
    arm = boundary = phase = None
    def slot(b, ph, frames=0):
        nonlocal boundary, phase
        boundary, phase = b, ph
        events.append(dict(event='slot', arm=arm, boundary=b, phase=ph, frames=frames))
    def physical(key, n):
        for _ in range(n):
            physics[key] += 1
            for action in ('physical_attempt', 'physical_return'):
                events.append(dict(event=action, arm=arm, boundary=boundary, kind=key, attempt=physics[key]))
    def call(kind, inside=None):
        entry = dict(index=len(entries), arm=arm, boundary=boundary, phase=phase, kind=kind, completed=False)
        entries.append(entry); events.append(dict(event='attempt', **entry))
        if inside: inside()
        entry['completed'] = True; events.append(dict(event='return', **entry))
    def forward(): call('action_encoder'); call('predict')
    def ad():
        forward()
        for _ in range(192): call('grad_api', lambda: call('engine_api'))
    def dispatch(b, name):
        events.append(dict(event='dispatch_authorized', arm=arm, boundary=b, selection=dict(file=name, sha256=hashes[name])))
        physical('selected_steps', 5)
    physical('root_resets', 32)
    for arm in ARMS:
        slot(0, 'reset'); physical('branch_resets', 1); physical('warmup_steps', 15)
        dispatch(0, arm+'.prefix.selection.json'); slot(0, 'prefix'); forward()
        for b in BOUNDARIES:
            slot(b, 'observe', 3); call('encoder'); call('projector')
            for ph in ('nominal', 'jacobian', *('candidate_'+str(i) for i in range(10)), 'replay'):
                slot(b, ph)
                if ph == 'jacobian': call('jacobian_api', ad)
                else: forward()
            row = dict(arm=arm, boundary=b, counts=ROUND_COUNTS); rounds.append(row)
            events.append(dict(event='round_complete', **row))
            name = arm+f'.step{b}.selection.json'
            events.append(dict(event='selection_sealed', arm=arm, boundary=b, file=name, sha256=hashes[name]))
            dispatch(b, name)
        slot(20, 'terminal', 1); call('encoder'); call('projector')
    events.append(dict(event='run_finished', complete=True, errors=[]))
    require(physics == PHYSICS, 'independent public budget algebra')
    return events, entries, rounds


def provenance(folder, receipt_sha):
    folder = Path(folder).resolve(); require(sha(folder/'receipt.json') == receipt_sha, 'external actual HOST receipt SHA')
    r = read(folder/'receipt.json'); require(r['schema'] == 'p4_feedback_live_cuda_20261027_v1.receipt' and
        r['status'] == 'two_live_20step_3round_arms_completed' and r['errors'] == [] and
        r['journal_failed'] is False and r['source_verified'] is True, 'completed live receipt')
    binding = read(folder/'execution_binding.json'); prepared = Path(binding['prepared'])
    require(binding['prepare_sha256'] == r['prepare_sha256'] == PREPARE_SHA and sha(prepared/'prepare.json') == PREPARE_SHA, 'actual prepared identity')
    pr = read(prepared/'prepare.json'); p = read(prepared/'protocol.json')
    require(pr['schema'] == 'p4_feedback_live_cuda_20261027_v1.prepare' and pr['status'] == 'prepared_not_run' and sha(prepared/'protocol.json') == pr['protocol_sha256'], 'live protocol SHA')
    for seal in pr['source_seals']:
        algebra.verify_seal(output_dir=seal['output_dir'], expected_manifest_sha256=seal['manifest_sha256'])
        same(read(Path(seal['output_dir'])/'args.json'), dict(protocol_sha256=pr['protocol_sha256']), 'source seal args')
    require(p['sources_sha256'][str(REPO/'experiments/spectral_reachability/p4_feedback_live_cuda_20261027_v1.py')] == LIVE_SHA, 'frozen live implementation')
    for path, h in {**p['sources_sha256'], **p['physical_sources'], **p['bindings'], **p['model_bindings']}.items(): require(sha(path) == h, 'source/model/input SHA '+path)
    response_proof = algebra.audit(algebra.DEFAULT, algebra.RECEIPT_SHA)
    require(response_proof['provenance_verified'] and p['response_receipt_sha256'] == r['response_receipt_sha256'] == algebra.RECEIPT_SHA, 'actual CUDA response provenance only')
    from experiments.spectral_reachability import p4_feedback_root_sourcepack_20261027_v1 as sourcepack
    origin = sourcepack.verify(p['sourcepack'], p['sourcepack_prepare_sha256'])
    for key in ('root_sha256', 'initial_context', 'prefix_commands', 'goal', 'q', 'normalization', 'model_bindings', 'active_metric_id'):
        same(p[key], origin[key], 'original sourcepack '+key)
    require(r['root_sha256'] == p['root_sha256'] and r['model_bindings'] == p['model_bindings'] and
        r['model_hash_before'] == r['model_hash_after'] == read(algebra.DEFAULT/'receipt.json')['model_hash_before'], 'reported frozen native model identity')
    for value in (r, p):
        require(value['known_roots_only'] is True and all(value[k] is False for k in ('P1_admission', 'P4_admission', 'independent_confirmation')), 'known-root/no admission')
    require(r['training'] is False and r['host']['device'] == 'cuda:0', 'recorded CUDA no training')
    require(set(r['output_files_sha256']) == inventory() and {x.name for x in folder.iterdir()} == inventory()|{'receipt.json'}, 'exact own198-file inventory')
    physical_path = next(Path(x) for x in p['bindings'] if x.endswith('/p3_joint_horizon_goal_metric_physical_20261027_v1_main01/report.json'))
    old_rows = [x for x in read(physical_path)['rows'] if x['root_sha256'] == p['root_sha256'] and x['active_metric_id'] == p['active_metric_id']]
    require(bool(old_rows), 'original physical root/metric rows')
    goal_state = old_rows[0]['context_identity']['goal']
    require(all(x['context_identity']['goal'] == goal_state for x in old_rows), 'original physical goal identity')
    return r, p, origin, goal_state


def audit(folder=DEFAULT, receipt_sha=RECEIPT_SHA):
    folder = Path(folder).resolve(); r, p, origin, goal_state = provenance(folder, receipt_sha)
    files = Files(folder, r['output_files_sha256'])
    for name in inventory(): files.path(name)
    events, entries, rounds = expected_journal(files.hashes)
    journal = [json.loads(line) for line in files.path('events.jsonl').read_text().splitlines()]
    same(journal, events, 'all5291 durable events exact causal/nested grammar')
    same(r['entries'], entries, 'all2486 API entries exact'); same(r['round_ledgers'], rounds, 'six independent round ledgers')
    require(sha(folder/'events.jsonl') == r['journal_sha256'], 'journal receipt SHA')
    same(r['counts'], {k:dict(attempted=v, completed=v) for k, v in COUNTS.items()}, 'total neural budgets')
    same(r['physics_attempts'], PHYSICS, 'physics attempts'); same(r['physics_returns'], PHYSICS, 'physics returns')
    limits = r['limits']; same(limits, p['limits'], 'sealed resources')
    require(limits['neural'] == COUNTS and limits['per_round'] == ROUND_COUNTS and limits['public_steps'] == 70 and limits['resets'] == 34 and
        limits['warmup_steps'] == 30 and limits['selected_steps'] == 40 and limits['encoded_frames'] == 20 and limits['processed_context_tokens'] == 240 and
        limits['mathematical_VJP_directions'] == 1152 and limits['FD'] == 0 and limits['retries'] is limits['training'] is False and
        limits['wall_seconds'] == 600 and limits['allocated_bytes'] == 3221225472 and limits['reserved_bytes'] == 4294967296 and
        limits['output_bytes'] == 16*1024**2 and limits['disk_floor'] == 200*1024**2, 'full prospective caps unchanged')
    require(0 <= r['wall_seconds'] <= 600 and 0 <= r['peaks'][0] <= 3221225472 and r['peaks'][0] <= r['peaks'][1] <= 4294967296, 'recorded resource completion')
    require([row['arm'] for row in r['rows']] == list(ARMS), 'one known root, ordered pair not two batches')
    arms = []
    for row in r['rows']:
        arm = row['arm']; require(row['status'] == 'completed20', 'complete20 each arm')
        same(files.json(arm+'.result.json'), row, 'durable arm result')
        previous_rgb = None; frames = []
        for i in range(3):
            previous_rgb = packet_check(files, arm+f'.warmup{i}', np.zeros((5, 2)), p['root_sha256'], previous_rgb); frames.append(previous_rgb)
        reset = files.json(arm+'.reset_identity.json'); reference = p['references'][p['root_sha256']]
        require(reset['actual_warmup_steps'] == 15 and reset['root'] == p['root_sha256'], 'own15 warmup')
        goal_name = arm+'.goal.png'; goal_rgb = files.rgb(goal_name)
        same(reset['goal_rgb'], dict(file=goal_name, file_sha256=files.hashes[goal_name], pixel_sha256=rgb_sha(goal_rgb)), 'goal PNG binding')
        same(reset['identity'], dict(history_rgb_sha256=rgb_sha(np.stack(frames)), goal_rgb_sha256=rgb_sha(goal_rgb)), 'reset stack identity')
        for key in reset['identity']: require(reset['identity'][key] == reference[key], 'original root RGB reference')
        initial = p['initial_context']; _, initial_history = context(initial)
        mean, std = (np.asarray(p['normalization'][k], np.float32) for k in ('mean', 'std'))
        zero = ((np.zeros((5, 2), np.float32)-mean)/std).reshape(10)
        require(np.array_equal(initial_history, np.stack([zero, zero])), 'causal warmup action history')
        same(files.json(arm+'.prefix.selection.json'), dict(root=p['root_sha256'], commands=p['prefix_commands'], initial_context=initial), 'fixed sealed original prefix')
        previous_rgb = packet_check(files, arm+'.from0', p['prefix_commands'], p['root_sha256'], previous_rgb)
        require(rgb_sha(previous_rgb) == origin['prefix_evidence']['pixel_sha256'], 'fresh replayed prefix matches original actual source')
        frames = frames[1:]+[previous_rgb]; predicted = files.json(arm+'.prefix.prediction.json')
        history(initial, predicted, p['prefix_commands'], p['normalization'])
        observed_previous = np.asarray(initial['latent_history']); audited_rounds = []
        require([x['boundary'] for x in row['rounds']] == list(BOUNDARIES), 'three causal planning boundaries')
        for b, rr in zip(BOUNDARIES, row['rounds'], strict=True):
            stem = arm+f'.step{b}'; obs = files.json(stem+'.observed.json'); observed = files.array(stem+'.observed.npy', (3, 192))
            require(obs['arm'] == arm and obs['boundary'] == b and obs['latent_file'] == stem+'.observed.npy' and
                obs['latent_sha256'] == files.hashes[obs['latent_file']] and obs['actual_observation_consumed'] is (arm == 'actual'), 'own observed array identity/consumption marker')
            same(obs['pixel_sha256'], [rgb_sha(f) for f in frames], 'fresh own three boundary PNGs')
            error = float(np.max(np.abs(observed[:2].astype(float)-observed_previous[1:])))
            require(error <= 2e-5, 'overlap numerical consistency only, not exact equality'); observed_previous = observed.copy()
            expected = planning_context(arm, predicted, observed); same(obs['planning_context'], expected, 'actual observed vs persistent imagined context')
            same(obs['action_history'], expected['action_history'], 'original executed causal action history')
            s = files.json(stem+'.selection.json'); request = s['request']; same(files.json(stem+'.request.json'), request, 'durable request identity')
            require(request['arm'] == arm and request['boundary'] == b and request['root_sha256'] == p['root_sha256'] and
                request['goal'] == p['goal'] and request['q_sha256'] == algebra.content_sha(p['q']), 'same root goalQ/own round')
            jac_name = stem+'.J.npy'; require(s['jacobian_file'] == jac_name and s['jacobian_sha256'] == files.hashes[jac_name], 'own J array file identity')
            winner = selection_check(p, expected, s, files.array(jac_name, (192, 10)))
            same(rr['selection'], dict(file=stem+'.selection.json', sha256=files.hashes[stem+'.selection.json']), 'sealed selection receipt')
            require(rr['winner_index'] == winner['index'], 'recorded winner index'); close(rr['model_cost'], winner['cost'], 'recorded model cost', 0.)
            boundary_error = cost_metrics(observed[-1], np.asarray(predicted['latent_history'])[-1], p['goal'], p['q'])
            audited_rounds.append(dict(boundary=b, selection_sha256=files.hashes[stem+'.selection.json'], winner_index=winner['index'],
                winner_model_cost=winner['cost'], overlap_max_abs=error, planning_uses_own_observed=arm == 'actual',
                observed_boundary_metrics=boundary_error))
            previous_rgb = packet_check(files, arm+f'.from{b}', winner['commands'], p['root_sha256'], previous_rgb)
            frames = frames[1:]+[previous_rgb]; predicted = deepcopy(winner['context'])
        ev = row['evaluation']; name = arm+'.step20.native192.npy'; terminal = files.array(name, (192,))
        require(ev['latent_file'] == name and ev['latent_sha256'] == files.hashes[name] and ev['pixel_sha256'] == rgb_sha(previous_rgb) and
                ev['consumed_for_planning'] is False, 'terminal own RGB/latent, never feedback')
        same(ev['predicted_context'], predicted, 'sealed15to20 prediction identity')
        metrics = cost_metrics(terminal, np.asarray(predicted['latent_history'])[-1], p['goal'], p['q'])
        compare_fields({k:ev[k] for k in metrics}, metrics, 'terminal latent/Q recomputation')
        require(ev['physical']['goal'] == goal_state, 'original physical goal state')
        physical = physical_metrics(ev['physical']['state'], goal_state)
        compare_fields(ev['physical']['official'], physical['official'], 'installed official labels')
        compare_fields(ev['physical']['pose'], physical['pose'], 'independent wrapped pose')
        arms.append(dict(arm=arm, rounds=audited_rounds, terminal_metrics=metrics, physical=physical,
            selected_steps=20, warmup_steps=15, actual_context_rgb_frames_consumed=9 if arm == 'actual' else 0))
    for name in inventory(): files.path(name)
    require(sha(folder/'receipt.json') == receipt_sha, 'actual receipt unchanged through audit')
    return dict(schema=NAME+'.report', status='independent_saved_live_audit_passed', receipt=str(folder/'receipt.json'), receipt_sha256=receipt_sha,
        prepare_sha256=PREPARE_SHA, live_source_sha256=LIVE_SHA, root_sha256=p['root_sha256'],
        gates=dict(source_model_goalQ_identity=True, exact_198_file_hash_inventory=True, own_live_RGB_array_context_linkage=True,
                   six_fresh_AD_recorded_rounds=True, ten_candidates_each_recomputed=True, selection_before_dispatch_journal=True,
                   all70_public_steps_guarded=True, terminal20_evaluation_only=True, official_wrapped_Q_recomputed=True),
        journal_events=len(events), API_entries=len(entries), API_counts=r['counts'], physics=r['physics_attempts'],
        recorded_peak_cuda_bytes=r['peaks'], recorded_wall_seconds=r['wall_seconds'], arms=arms,
        interpretation='actual lower terminal Q-goal MSE but worse object pose; both wrapped/official failures; no physical improvement claim',
        limitations=['saved PNG/array/context links and frozen source/journal are checked; no independent CUDA encoding or derivative replay',
            'predict means model.predict API entry, not model.predictor internal entry or FLOPs',
            'overlap atol2e-5 is numerical consistency only, not bitwise equality',
            '70 step PNGs and82 of84 guard pixel counts independently recounted; two initial warmup render counts only recorded',
            'source seal/recorded event order is not hardware attestation or proof against malicious fabricated execution',
            'single known-root paired arms, not two batches; no independent P1 admission'],
        new_neural_calls=0, new_physical_steps=0, P1_admission=False, P4_admission=False, independent_confirmation=False)


def write_report(output, report):
    output = Path(output).absolute()
    sources = [Path(__file__), REPO/'tests'/('test_'+NAME+'.py'), REPO/'innerdocs'/(NAME+'.md')]
    report = dict(report, audit_sources_sha256={str(f):sha(f) for f in sources})
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    total = sum(f.stat().st_size for f in sources)+len(raw)
    require(total <= 300*1024 and not output.exists() and shutil.disk_usage(output.parent).free >= 200*1024**2+len(raw)+65536, 'NEW <=300KiB total task footprint /200MiB floor')
    output.mkdir()
    with (output/'report.json').open('xb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
    fd = os.open(output, os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return dict(report_sha256=sha(output/'report.json'), task_bytes=total, report_bytes=len(raw))


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run-dir', type=Path, default=DEFAULT)
    parser.add_argument('--receipt-sha', default=RECEIPT_SHA); parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args(); report = audit(a.run_dir, a.receipt_sha)
    print(json.dumps(dict(status=report['status'], **write_report(a.output, report)), sort_keys=True))


if __name__ == '__main__': main()
