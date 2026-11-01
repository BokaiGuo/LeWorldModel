"""Independent, bounded saved-data attribution diagnostics; never P4 admission.

Imports only stdlib and NumPy. No runtime/compiler/environment imports or calls.
Missing observations are unknown, not zero, not an efficacy failure or success.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np

TAG = 'p4_feedback_attribution_audit_20261027_v1'
REPO = Path(__file__).resolve().parents[2]
ARMS = ('imagined', 'actual')
BOUNDARIES = (5, 10, 15)
MAX_FILE = 32 * 1024 * 1024
MAX_TOTAL = 128 * 1024 * 1024


def require(ok, message):
    if not ok:
        raise ValueError(message)


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=pairs)
    json.dumps(value, allow_nan=False)
    return value


def content_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class Inputs:
    def __init__(self):
        self.hashes = {}
        self.bytes_read = 0

    def raw(self, path):
        path = Path(path).resolve()
        require(path.is_file() and path.stat().st_size <= MAX_FILE, 'missing/oversize input: ' + str(path))
        require(self.bytes_read + path.stat().st_size <= MAX_TOTAL, 'total input budget exceeded')
        with path.open('rb') as stream:
            raw = stream.read(MAX_FILE + 1)
        require(len(raw) <= MAX_FILE, 'input grew beyond cap')
        self.bytes_read += len(raw)
        digest = hashlib.sha256(raw).hexdigest()
        require(str(path) not in self.hashes or self.hashes[str(path)] == digest, 'input changed')
        self.hashes[str(path)] = digest
        return raw

    def json(self, path):
        return strict_json(self.raw(path))

    def stable(self):
        for path, expected in list(self.hashes.items()):
            require(hashlib.sha256(self.raw(path)).hexdigest() == expected, 'input changed: ' + path)


def array(value, shape):
    a = np.asarray(value)
    require(a.shape == shape and a.dtype.kind in 'fiu' and np.isfinite(a).all(), 'finite array required: ' + str(shape))
    return a.astype(float)


def context(c, dim):
    require(c['layout'] == 'states_with_between_state_actions_v1' and
            c['context_length'] == 3 and c['latent_dim'] == dim and c['action_dim'] == 10, 'context layout')
    return array(c['latent_history'], (3, dim)), array(c['action_history'], (2, 10))


def normalized(raw):
    a = array(raw, (2, 5, 2)).astype(np.float32)
    require(np.all(np.abs(a) <= 1), 'raw history out of support')
    return (a.reshape(2, 10) - np.tile(np.array([-.0087, .0068], np.float32), 5)) / np.tile(
        np.array([.2019, .2002], np.float32), 5)


def cost(v, goal, q):
    d = q @ (goal - array(v, goal.shape))
    return float(np.mean(d*d))


def discrepancy(predicted, observed, goal, q):
    p, o = array(predicted, goal.shape), array(observed, goal.shape)
    d, r = q @ (o-p), q @ (goal-p)
    square, cross = float(np.mean(d*d)), float(-2*np.mean(r*d))
    delta = cost(o, goal, q) - cost(p, goal, q)
    require(np.isclose(square+cross, delta, rtol=1e-10, atol=1e-10), 'residual identity')
    return dict(raw_l2=float(np.linalg.norm(o-p)), projected_mse=square,
                goal_cross_term=cross, observed_minus_predicted_cost=delta,
                identity_error=square+cross-delta)


def pose(state, goal):
    s, g = array(state, (7,)), array(goal, (7,))
    d = s-g
    angle = float(abs(np.arctan2(np.sin(d[4]), np.cos(d[4]))))
    xy = float(np.linalg.norm(d[2:4]))
    return dict(object_xy=xy, wrapped_angle=angle, pose_cost=xy+20*angle,
                official_state_distance=float(np.linalg.norm(d)),
                official_position_error=float(np.linalg.norm(d[:4])))


def indexed(events, kind, field):
    values = [e for e in events if e['kind'] == kind]
    result = {e[field]: e for e in values}
    require(len(result) == len(values), 'duplicate ' + kind)
    return result


def one(events, kind):
    values = [e for e in events if e['kind'] == kind]
    require(len(values) == 1, 'missing/duplicate ' + kind)
    return values[0]


def verify_rgb_history(p, events, inputs):
    """Decode saved files only; no rendering, model or simulator."""
    from PIL import Image
    frames = np.load(io.BytesIO(inputs.raw(p['root_rgb'])), allow_pickle=False)
    require(frames.shape == (3,224,224,3) and frames.dtype == np.uint8, 'root RGB stack')
    pixel_sha = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
    initial = [pixel_sha(a) for a in frames]
    with Image.open(io.BytesIO(inputs.raw(p['prefix_rgb']))) as im:
        prefix_rgb = np.asarray(im).copy()
    for arm in ARMS:
        ev = [e for e in events if e.get('arm') == arm]
        observations = indexed(ev, 'observation', 'primitive_step')
        pixels = {}
        for e in [one(ev,'initial'), *observations.values()]:
            raw = inputs.raw(e['rgb_path'])
            require(hashlib.sha256(raw).hexdigest() == e['rgb_sha256'], 'RGB file SHA')
            with Image.open(io.BytesIO(raw)) as im:
                rgb = np.asarray(im).copy()
            require(rgb.shape == (224,224,3) and rgb.dtype == np.uint8 and
                    pixel_sha(rgb) == e['pixel_sha256'], 'RGB pixel SHA')
            pixels[e.get('primitive_step',0)] = rgb
        require(pixel_sha(pixels[0]) == initial[-1], 'root frame identity')
        require(np.array_equal(pixels[5],prefix_rgb), 'original prefix RGB identity')
        history = list(initial)
        begins = indexed(ev,'boundary_begin','boundary')
        for b in BOUNDARIES:
            history = history[1:] + [pixel_sha(pixels[b])]
            require(begins[b]['frames_pixel_sha256'] == history, 'observed RGB rolling history')
        guard = next(r for r in one(events,'final_census')['rows'] if r['arm']==arm)['guard']
        require(guard['root_id'] == p['root_sha256'] and guard['abstained'] is False and
                guard['primitive_commands_authorized'] == 20 and guard['observations'] == [
                    dict(primitive_step=k,canonical_agent_pixels=int(np.count_nonzero(np.all(pixels[k]==[65,105,225],axis=-1))))
                    for k in range(21)], 'saved RGB guard')


def accounting(report, events):
    expected = {}
    for arm in ARMS:
        expected[arm+'_prefix'] = (1, 'evaluation')
        for b in BOUNDARIES:
            for suffix, count, phase in (('compile', 2, 'planning'), ('bank', 10, 'planning'),
                                          ('independent_replay', 1, 'evaluation')):
                expected[f'{arm}_{b}_{suffix}'] = (count, phase)
    receipts = report['predict']
    require(len(receipts) == len(expected) and len({r['method'] for r in receipts}) == len(receipts), 'predict receipt census')
    for r in receipts:
        require(r['method'] in expected, 'unexpected predict method')
        n, phase = expected[r['method']]
        require(r['completed'] is True and r['failed_predict_calls'] == 0 and
                r['phase'] == phase and r['predict_samples'] == r['predict_calls'] == n and
                r['context_lengths'] == {'3': n} and r['horizon_tokens'] == 1, 'predict stage budget')
    c = report['cost']
    for key, n in dict(predict_samples=80, selected_attempts=40, selected_returns=40,
                       encode_attempts=6, encode_completed=6, encode_frames_attempted=18,
                       visual_encoder_attempts=6, visual_encoder_completed=6,
                       visual_encoder_frames_attempted=18, visual_encoder_frames_completed=18,
                       jacobian_attempts=6, jacobian_completed=6, bank_attempts=6, bank_completed=6,
                       action_encoder_attempts=80, action_encoder_completed=80, action_encoder_input_tokens=240).items():
        require(type(c[key]) is int and c[key] == n, 'cost mismatch: ' + key)
    for kind, key in (('step_attempt', 'selected_attempts'), ('step_returned', 'selected_returns'),
                      ('setup_step_attempt', 'setup_attempts'), ('setup_step_returned', 'setup_returns'),
                      ('encode_begin', 'encode_attempts'), ('jacobian_begin', 'jacobian_attempts'),
                      ('bank_begin', 'bank_attempts')):
        require(sum(e['kind'] == kind for e in events) == c[key], 'journal census: ' + kind)
    final = one(events, 'final_census')
    require(final['cost'] == c and final['predict'] == receipts and final['rows'] == report['rows'] and
            final['slots'] == report['slots'], 'final journal/report mismatch')
    require(report['actual_predict_samples'] == 80 and report['scalar_vjp_directions_completed'] == 1152 and
            report['svd_calls_completed'] == 12, 'derived budget mismatch')
    return dict(predict_samples=80, compiler_predict_samples=78, selected_steps=40,
                setup_steps=c['setup_returns'], encode_calls=6, encode_frames=18,
                jacobians=6, scalar_vjp_directions=1152, svd_calls=12,
                processed_action_context_tokens=240, total_compute_matched=False,
                derivative_and_svd_counts='saved receipts and derived dimensions; not independent API hooks')


def analyze(p, report, events):
    """Pure arithmetic core; caller handles provenance, tests use explicit mocks."""
    root = p['root_sha256']
    goal = array(p['goal'], (len(p['goal']),)); dim = len(goal)
    q = array(p['q'], (dim, dim))
    require([e['event_index'] for e in events] == list(range(len(events))), 'event order/gap')
    require(report['status'] == 'calibration_recorded' and not report['failures'] and
            report['model_unchanged'] is True, 'runtime incomplete')
    require(len(report['rows']) == 2 and {r['arm'] for r in report['rows']} == set(ARMS), 'arm roster')
    require(len(report['slots']) == 6 and {(s['arm'], s['boundary']) for s in report['slots']} ==
            {(a,b) for a in ARMS for b in BOUNDARIES} and all(s['root_sha256'] == root and
            s['status'] == 'completed' for s in report['slots']), 'slot census/root')
    budget = accounting(report, events)
    rows, endpoints = [], {}
    for arm in ARMS:
        row = next(r for r in report['rows'] if r['arm'] == arm)
        require(row['root_sha256'] == root and row['status'] == 'completed' and
                row['completed_packets'] == 4 and row['selected_attempts'] == row['selected_returns'] == 20,
                'branch identity/completeness')
        ev = [e for e in events if e.get('arm') == arm]
        obs = indexed(ev, 'observation', 'primitive_step')
        attempts = indexed(ev, 'step_attempt', 'primitive_step')
        returns = indexed(ev, 'step_returned', 'primitive_step')
        require(set(obs) == set(attempts) == set(returns) == set(range(1,21)), 'missing physical endpoint')
        for step in obs:
            array(obs[step]['state'],(7,))
            require(attempts[step]['event_index'] < returns[step]['event_index'] < obs[step]['event_index'], 'dispatch order')
        contexts = indexed(ev, 'planning_context', 'boundary')
        selections = indexed(ev, 'selection', 'boundary')
        begins = indexed(ev, 'boundary_begin', 'boundary')
        packets = indexed(ev, 'packet_complete', 'boundary')
        require(all(set(v) == set(BOUNDARIES) for v in (contexts, selections, begins, packets)), 'round census')
        require(one(ev, 'prefix_identity')['pixels_exact'] is True, 'prefix identity')
        raw = np.stack([p['between_raw'][1], p['prefix']]).tolist()
        previous = None
        previous_observed = None
        for j in range(5):
            require(attempts[j+1]['command'] == p['prefix'][j], 'prefix dispatch action')
        for b in BOUNDARIES:
            c, s, begin = contexts[b], selections[b], begins[b]
            require(s['root_sha256'] == root, 'selection root')
            slot = next(v for v in report['slots'] if v['arm']==arm and v['boundary']==b)
            require(slot['selection_sha256'] == s['selection_sha256'] and
                    slot['selected_candidate_id'] == s['compiler_plan']['selected_candidate_id'], 'slot selection binding')
            require(s['selection_sha256'] == content_sha({k:v for k,v in s.items() if k not in
                ('kind','event_index','monotonic_seconds','selection_sha256')}), 'selection hash')
            require(begin['between_raw'] == s['incoming_between_raw'] == raw, 'raw rolling history')
            ol, oa = context(c['observed_context'], dim)
            pl, pa = context(c['predicted_context'], dim)
            require(np.array_equal(oa, normalized(raw)) and np.array_equal(pa, oa), 'normalization/history')
            chosen = c['observed_context'] if arm == 'actual' else c['predicted_context']
            require(c['chosen_context'] == chosen and s['context_sha256'] == content_sha(chosen), 'feedback arm substitution')
            if previous is not None:
                require(c['predicted_context'] == previous['successor'], 'predicted rolling history')
                require(np.array_equal(ol[:2],previous_observed[1:]), 'observed latent rolling history')
            else:
                il, _ = context(p['initial_context'],dim)
                require(np.array_equal(pl[:2],il[1:]) and pl[-1].tolist() == p['saved_predicted_prefix'], 'prefix predicted context')
            plan = s['compiler_plan']
            require(plan['root_sha256'] == root and plan['arm'] == arm and plan['context'] == chosen and
                    plan['goal'] == p['goal'] and plan['q'] == p['q'] and plan['prefix'] == p['prefix'] and
                    s['metric_sha256'] == content_sha([p['goal'],p['q']]), 'plan root/context/metric')
            candidates = plan['candidates']
            require(len(candidates) == 10 and len({v['candidate_id'] for v in candidates}) == 10, 'candidate roster')
            for v in candidates:
                require(np.isclose(cost(v['predicted_terminal'], goal, q), v['nonlinear_terminal_cost'],
                                   rtol=0, atol=1e-12), 'candidate score')
            winner = min(candidates, key=lambda v:v['nonlinear_terminal_cost'])
            require(np.all(np.abs(array(s['commands'],(5,2))) <= 1), 'command support')
            require(winner['candidate_id'] == plan['selected_candidate_id'] and
                    plan['independent_replay_exact'] is True and winner['commands'][:5] == p['prefix'] and
                    winner['commands'][5:] == s['commands'], 'winner/replay/actions')
            sl, sa = context(s['successor'], dim)
            cl, ca = context(chosen, dim)
            require(np.array_equal(sl[:2], cl[1:]) and np.array_equal(sa[0], ca[1]) and
                    np.array_equal(sa, normalized([raw[1], s['commands']])) and
                    sl[-1].tolist() == winner['predicted_terminal'], 'successor history')
            require(begin['event_index'] < c['event_index'] < s['event_index'] < attempts[b+1]['event_index'], 'selection before dispatch')
            for j in range(5):
                require(attempts[b+j+1]['command'] == s['commands'][j], 'executed action mismatch')
            before, after = pose(obs[b]['state'], p['physical_goal']), pose(obs[b+5]['state'], p['physical_goal'])
            nxt = contexts.get(b+5)
            next_latent = None if nxt is None else nxt['observed_context']['latent_history'][-1]
            rows.append(dict(root_sha256=root, arm=arm, round=b//5, boundary=b, end=b+5,
                selected_candidate_id=winner['candidate_id'], boundary_feedback=discrepancy(pl[-1],ol[-1],goal,q),
                predicted_Q_cost=cost(sl[-1],goal,q),
                next_latent_status='unconfirmed_not_saved' if nxt is None else 'saved_at_next_boundary',
                prediction_error=None if nxt is None else discrepancy(sl[-1],next_latent,goal,q),
                observed_Q_cost_change=None if nxt is None else cost(next_latent,goal,q)-cost(ol[-1],goal,q),
                physical_before=before, physical_after=after,
                physical_change={k:after[k]-before[k] for k in before},
                prediction_conditioning='observed start' if arm=='actual' else 'imagined rolling start; accumulated drift',
                causal_attribution='unconfirmed; no counterfactual intervention'))
            raw = [raw[1],s['commands']]
            require(packets[b]['next_between_raw'] == raw and packets[b]['predicted_context'] == s['successor'], 'packet receipt')
            previous = s
            previous_observed = ol
        endpoints[arm] = pose(obs[20]['state'], p['physical_goal'])
    paired = []
    for b in BOUNDARIES:
        actual = next(v for v in rows if v['arm']=='actual' and v['boundary']==b)
        imagined = next(v for v in rows if v['arm']=='imagined' and v['boundary']==b)
        paired.append(dict(root_sha256=root,boundary=b,end=b+5,
            actual_minus_imagined_endpoint={k:actual['physical_after'][k]-imagined['physical_after'][k]
                                           for k in actual['physical_after']},
            matched_initial_state_only=b==5,
            interpretation='divergent own histories after first dispatch; not isolated per-round causal effect'))
    return dict(root_sha256=root, rounds=rows, paired_rounds=paired, budget=budget,
                actual_minus_imagined={k:endpoints['actual'][k]-endpoints['imagined'][k] for k in endpoints['actual']},
                arithmetic_verified=True, P1_admission=False, P4_admission=False, efficacy_verified=False)


def discover(results):
    """Inventory only P4 feedback protocol metadata; never use outcome values to select."""
    found, inventory = [], []
    for path in sorted(Path(results).glob('p4*/protocol.json')):
        if path.stat().st_size > MAX_FILE:
            inventory.append(dict(path=str(path), status='unconfirmed_oversize_protocol')); continue
        p = strict_json(path.read_bytes())
        c = p.get('contract', {})
        eligible = c.get('planned_steps_per_arm') == 20 and c.get('boundaries') == list(BOUNDARIES)
        report = path.parent/'report.json'
        inventory.append(dict(path=str(path), eligible_20step3round=eligible, report_exists=report.is_file()))
        if eligible and report.is_file():
            found.append(report.resolve())
    return sorted(found, key=lambda p:(p.stat().st_mtime_ns,str(p)), reverse=True)[:2], inventory


def audit_report(path, inputs):
    path = Path(path).resolve(); folder = path.parent
    report, p = inputs.json(path), inputs.json(folder/'protocol.json')
    prepare = inputs.json(folder/'prepare_receipt.json')
    c = p['contract']
    for key,value in dict(planned_steps_per_arm=20, boundaries=[5,10,15], arms=list(ARMS),
                          scheduled_predict_samples=80, scheduled_compiler_predict_samples=78,
                          scheduled_jacobians=6, scheduled_encode_calls=6, scheduled_encode_frames=18,
                          scheduled_scalar_vjp_directions=1152, scheduled_svd_calls=12,
                          scheduled_selected_steps=40).items():
        require(c[key] == value, 'unsupported protocol budget: ' + key)
    require(c['root_sha256'] == p['root_sha256'] and prepare['status']=='prepared_not_run', 'protocol root/status')
    require(len(p['goal']) == 192, 'unsupported saved runtime latent dimension')
    require(report['protocol_sha256'] == prepare['protocol_sha256'] == inputs.hashes[str(folder/'protocol.json')], 'protocol hash')
    raw = inputs.raw(folder/'runtime.events.jsonl')
    require(hashlib.sha256(raw).hexdigest() == report['journal_sha256'], 'journal hash')
    events = [strict_json(line) for line in raw.splitlines()]
    require(len(events) <= 2000, 'event cap')
    for arm in ARMS:
        for b in BOUNDARIES:
            method=f'{arm}_{b}'
            derivative=[e for e in events if e['kind']=='derivative_receipt' and e.get('method')==method]
            require(len(derivative)==1, 'derivative receipt census')
            array(derivative[0]['jacobian'],(192,10))
            array(derivative[0]['nominal_terminal'],(192,))
            candidates=[e for e in events if e['kind']=='candidate' and e.get('method')==method]
            require(len(candidates)==10, 'candidate event census')
            selection=next(e for e in events if e['kind']=='selection' and e['arm']==arm and e['boundary']==b)
            require([e['candidate'] for e in candidates] == selection['compiler_plan']['candidates'], 'candidate journal/plan binding')
    verify_rgb_history(p,events,inputs)
    result = analyze(p, report, events)
    result.update(report_path=str(path), rgb_history_verified=True,
                  provenance_scope='protocol/prepare/journal/RGB bytes; upstream source seals and hidden simulator state not independently reconstructed')
    return result


def run(output, results=REPO/'results', reports=None):
    output = Path(output).resolve()
    require(not output.exists() and output.name.startswith(TAG), 'new exclusive output required')
    paths, inventory = discover(results) if reports is None else ([Path(p).resolve() for p in reports], [])
    require(len(paths) <= 2 and len(set(paths)) == len(paths), 'at most two distinct runs')
    inputs, runs = Inputs(), []
    for source in (Path(__file__),REPO/f'tests/test_{TAG}.py',REPO.parent/f'docs/{TAG}.md'):
        if source.is_file():
            inputs.raw(source)
    for path in paths:
        try:
            runs.append(audit_report(path, inputs))
        except (ValueError, KeyError, TypeError, OSError, IndexError) as exc:
            runs.append(dict(report_path=str(path), arithmetic_verified=False, status='unconfirmed_integrity_failure', error=str(exc)))
    inputs.stable()
    result = dict(schema=TAG, status='saved_diagnostics_recorded_not_admission', runs=runs,
        requested_runs=2, located_runs=len(paths), missing_runs=2-len(paths), inventory=inventory,
        two_run_evidence_verified=len(runs)==2 and all(r['arithmetic_verified'] for r in runs),
        missing_run_status='unconfirmed' if len(paths)<2 else 'located', inputs=inputs.hashes,
        inputs_unchanged=True, P1_admission=False, P4_admission=False, efficacy_verified=False,
        new_neural_calls=0, new_physics_steps=0, training=False,
        limitations=['P1 not passed; P4 cannot pass', 'step20 real latent not saved; never infer it from pose or predictions',
                     'root/arm/round diagnostics are descriptive, not unique causal attribution',
                     'only one known root per run; repeated roots/runs are not independent samples'])
    encoded = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    require(len(encoded) < 2*1024*1024, 'output cap')
    output.mkdir(parents=True, exist_ok=False)
    with (output/'report.json').open('xb') as stream:
        stream.write(encoded)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--results', type=Path, default=REPO/'results')
    parser.add_argument('--report', type=Path, action='append', dest='reports')
    args = parser.parse_args()
    result = run(**vars(args))
    print(json.dumps({k:result[k] for k in ('status','located_runs','missing_runs','two_run_evidence_verified','P4_admission')}))
