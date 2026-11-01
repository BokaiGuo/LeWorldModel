"""Independent saved-array arithmetic only. No executor imports or model execution."""
from __future__ import annotations

import argparse
import base64
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import zlib

import numpy as np

NAME = 'p1_teacher_fidelity_error_decomposition_20261027_v1'
TEACHER = 'p1_moving_live_feedback_teacher_fidelity_cuda_20261027_v2'
ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'results' / (TEACHER + '_cuda_main01') / 'report.json'
REPORT_SHA = '7323347a9e74c8f1751550642503426d07ddce55549ddaef4331f3c362567128'
ARMS = ('line_search_GN', 'full10_conditional_D2P')
LAYERS = tuple((s, a) for s in (0, 1) for a in ARMS)
KEYS = tuple((s, a, m) for s, a in LAYERS for m in ('O', 'I', 'R'))
RESOURCES = ('action_encoder', 'predict_API', 'predictor', 'pred_proj')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()


def digest(value):
    return sha(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def parse(raw):
    def pairs(items):
        out = {}
        for k, v in items:
            require(k not in out, 'duplicate JSON key: ' + k)
            out[k] = v
        return out
    def invalid(value):
        raise ValueError('nonfinite JSON: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def hash_value(value):
    require(type(value) is str and len(value) == 64 and
            all(c in '0123456789abcdef' for c in value), 'literal SHA256')


class Saved:
    """Injected byte reader also permits RAM-only tests; no writes in analysis."""

    def __init__(self, reader=None, listing=None):
        self.reader = reader or self.disk_read
        self.listing = listing or (lambda p: {f.name for f in Path(p).iterdir()})
        self.files = {}
        self.arrays = {}

    @staticmethod
    def disk_read(path):
        p = Path(path)
        require(p.is_file() and not p.is_symlink(), 'regular non-symlink input: ' + str(p))
        require(p.stat().st_size <= 128 * 1024**2, 'bounded saved input')
        return p.read_bytes()

    def read(self, path, expected):
        hash_value(expected)
        path = str(Path(path).absolute())
        raw = self.reader(path)
        require(sha(raw) == expected, 'file SHA mismatch: ' + path)
        require(path not in self.files or self.files[path]['sha256'] == expected,
                'conflicting input binding: ' + path)
        self.files[path] = dict(sha256=expected, bytes=len(raw))
        return raw

    def json(self, path, expected):
        return parse(self.read(path, expected))

    def array(self, path, expected, shape, array_sha=None):
        raw = self.read(path, expected)
        stream = BytesIO(raw)
        a = np.load(stream, allow_pickle=False)
        require(isinstance(a, np.ndarray) and stream.tell() == len(raw), 'single NPY, no trailing data')
        require(a.dtype == np.dtype('float32') and list(a.shape) == list(shape) and
                np.isfinite(a).all(), 'finite original float32 array/shape')
        h = sha(np.ascontiguousarray(a).tobytes())
        require(array_sha is None or h == array_sha, 'array SHA mismatch: ' + str(path))
        self.arrays[str(Path(path).absolute())] = dict(
            file_sha256=expected, array_sha256=h, dtype=str(a.dtype), shape=list(a.shape))
        return a

    def receipt(self, entry):
        require(entry['dtype'] == 'float32', 'native receipt dtype')
        return self.array(entry['path'], entry['file_sha256'], entry['shape'], entry['array_sha256'])


def local(root, name):
    require(type(name) is str and Path(name).name == name and name not in ('', '.', '..'),
            'flat local artifact name')
    return Path(root) / name


def decode_prepare(raw):
    require(len(raw) <= 262144, 'prepare envelope cap')
    e = parse(raw)
    require(set(e) == {'schema', 'codec', 'semantic_bytes', 'semantic_sha256', 'payload'} and
            e['schema'] == TEACHER + '.prepare_envelope' and e['codec'] == 'single-gzip+base64',
            'original prepare envelope')
    require(type(e['semantic_bytes']) is int and 0 < e['semantic_bytes'] <= 1024**2, 'semantic cap')
    packed = base64.b64decode(e['payload'], validate=True)
    require(base64.b64encode(packed).decode() == e['payload'], 'canonical base64')
    decoder = zlib.decompressobj(31)
    raw = decoder.decompress(packed, 1024**2 + 1)
    require(len(raw) == e['semantic_bytes'] and decoder.eof and not decoder.unused_data and
            not decoder.unconsumed_tail and sha(raw) == e['semantic_sha256'], 'single bounded gzip/SHA')
    p = parse(raw)
    require(canonical(p) == raw and p['schema'] == TEACHER + '.prepare', 'canonical semantic prepare')
    return p


def vector(value):
    a = np.asarray(value)
    require(a.shape == (192,) and a.dtype.kind == 'f' and np.isfinite(a).all() and
            np.array_equal(a, a.astype(np.float32)), 'exact finite native192 values')
    return a.astype(np.float64)


def decomposition(pred, actual, goal):
    """Q=||z-g||²/192. Sign is cost, not reward; no fitted normalizer."""
    p, a, g = map(vector, (pred, actual, goal))
    e, residual = p - a, a - g
    cross = float(2 * np.sum(residual * e, dtype=np.float64))
    square = float(np.sum(e * e, dtype=np.float64))
    qp = float(np.mean((p - g)**2, dtype=np.float64))
    qa = float(np.mean(residual**2, dtype=np.float64))
    delta = qp - qa
    reconstructed = (cross + square) / 192
    tolerance = 64 * np.finfo(np.float64).eps * max(1., abs(qp), abs(qa), abs(cross / 192), square / 192)
    require(abs(delta - reconstructed) <= tolerance, 'Q decomposition numerical identity')
    return dict(latent_MSE=square / 192, predicted_Q=qp, actual_Q=qa,
                predicted_Q_minus_actual_Q=delta, absolute_Q_error=abs(delta),
                squared_Q_error=delta**2, normalization_denominator=192,
                raw_cross_2_actual_minus_goal_dot_pred_minus_actual=cross,
                raw_squared_pred_minus_actual_norm=square,
                normalized_cross=cross / 192, normalized_squared_error=square / 192,
                reconstructed_Q_error=reconstructed, identity_residual=delta - reconstructed,
                identity_tolerance=tolerance)


def sign(value):
    return int(value > 0) - int(value < 0)


def contrasts(rows):
    by_key = {}
    for row in rows:
        i = row['identity']
        key = (i['source_slot'], i['prefix_arm'], i['mode'])
        require(key not in by_key, 'duplicate episode key')
        by_key[key] = row
    require(set(by_key) == set(KEYS), 'complete fixed12 cohort, no subset')
    out = []
    for layer_index, (slot, arm) in enumerate(LAYERS):
        o, r = (by_key[(slot, arm, mode)] for mode in ('O', 'R'))
        require(o['identity']['parent_task_id'] == r['identity']['parent_task_id'] and
                o['identity']['parent_selection_sha256'] == r['identity']['parent_selection_sha256'] and
                o['prefix_array_sha256'] == r['prefix_array_sha256'] and
                o['goal_array_sha256'] == r['goal_array_sha256'], 'exact paired R/O prefix and goal')
        om, rm = o['endpoint'], r['endpoint']
        predicted = rm['predicted_Q'] - om['predicted_Q']
        actual = rm['actual_Q'] - om['actual_Q']
        error = rm['predicted_Q_minus_actual_Q'] - om['predicted_Q_minus_actual_Q']
        cross = rm['normalized_cross'] - om['normalized_cross']
        square = rm['normalized_squared_error'] - om['normalized_squared_error']
        require(np.isclose(predicted - actual, cross + square, rtol=1e-12, atol=1e-14),
                'paired margin error decomposition')
        out.append(dict(layer_index=layer_index, source_slot=slot, prefix_arm=arm,
                        O_episode=o['identity']['episode_id'], R_episode=r['identity']['episode_id'],
                        margin_definition='Q_R - Q_O; negative favors R (lower latent cost)',
                        predicted_R_minus_O=predicted, actual_R_minus_O=actual,
                        predicted_sign=sign(predicted), actual_sign=sign(actual),
                        strict_reversal=sign(predicted) * sign(actual) == -1,
                        exact_tie=predicted == 0 or actual == 0,
                        margin_error=error, normalized_cross_difference=cross,
                        normalized_squared_error_difference=square,
                        identity_residual=(predicted - actual) - (cross + square)))
    return out


def analyze(report_path=REPORT, report_sha=REPORT_SHA, saved=None):
    s = saved or Saved()
    report_path = Path(report_path).absolute()
    root = report_path.parent
    r = s.json(report_path, report_sha)
    require(r['schema'] == TEACHER + '.report' and r['status'] == 'teacher_factual_recorded_not_admitted'
            and r['errors'] == [] and r['first_error'] is None and r['blocked'] == []
            and r['source_verified'] is True and r['model_unchanged'] is True
            and r['fixed_denominator'] == 12 and r['independent_tasks'] == 2
            and len(r['rows']) == 12 and r['host']['device'].startswith('cuda:'), 'complete CUDA teacher input')
    p = decode_prepare(s.read(r['prepared'], r['prepare_sha256']))
    require(len(p['inputs']) == len(p['evaluators']) == 12 and r['contract'] == p['contract'] and
            r['model_hash_before'] == r['model_hash_after'] == p['expected_model_hash'] and
            r['resources'] == {k: dict(attempted=12, completed=12) for k in RESOURCES}, 'original fixed12 contract')
    require(p['contract']['order'] == [list(k) for k in KEYS] and p['contract']['horizon'] == 1 and
            p['contract']['packet_steps'] == 5 and p['contract']['latent_dim'] == 192, 'original H1/5/192')
    require(p['bindings'], 'nonempty source/durable closure')
    for path, h in p['bindings'].items():
        s.read(path, h)  # Hash only: never load models, execute sources, or encode images.
    hashes = r['artifact_sha256']
    require(s.listing(root) == set(hashes) | {'report.json'}, 'exact teacher artifact inventory')
    for name, h in hashes.items():
        raw = s.read(local(root, name), h)
        if name.endswith('.npy'):
            stream = BytesIO(raw)
            a = np.load(stream, allow_pickle=False)
            require(isinstance(a, np.ndarray), 'NPY only')
            s.array(local(root, name), h, a.shape)
    def artifact(name):
        return s.json(local(root, name), hashes[name])
    def load(entry):
        require(entry['dtype'] == 'float32' and hashes[entry['file']] == entry['sha256'], 'artifact receipt SHA')
        return s.array(local(root, entry['file']), entry['sha256'], entry['shape'])
    require(artifact('binding.json') == dict(prepared=r['prepared'], prepare_sha256=r['prepare_sha256']), 'execution binding')
    seal = artifact('predictions_seal.json')
    require(hashes['predictions_seal.json'] == r['predictions_seal_sha256'] and
            seal['schema'] == TEACHER + '.predictions_seal' and seal['ledger'] == r['ledger'] and
            seal['resources'] == r['resources'] and len(seal['rows']) == 12 and len(r['ledger']) == 48,
            'original full prediction seal')
    previous = None
    events = []
    for index, line in enumerate(s.read(root / 'events.jsonl', hashes['events.jsonl']).splitlines()):
        event = parse(line)
        require(event['index'] == index and event['previous_sha256'] == previous and
                event['event_sha256'] == digest({k: v for k, v in event.items() if k != 'event_sha256'}), 'event hash chain')
        previous = event['event_sha256']
        events.append(event)
    sealed = [e for e in events if e['kind'] == 'all_predictions_sealed']
    scored = [e for e in events if e['kind'] == 'episode_scored']
    require(len(sealed) == 1 and sealed[0]['sha256'] == r['predictions_seal_sha256'] and
            len(scored) == 12 and all(e['index'] > sealed[0]['index'] for e in scored), 'seal before all scoring')
    out = []
    for index, (row, req, ev) in enumerate(zip(r['rows'], p['inputs'], p['evaluators'], strict=True)):
        ident = row['identity']
        slot, arm, mode = KEYS[index]
        require(ident == req['identity'] == ev['identity'] == seal['rows'][index]['identity'] and
                type(ident['index']) is int and ident['index'] == index and type(ident['source_slot']) is int and
                (ident['source_slot'], ident['prefix_arm'], ident['mode']) == (slot, arm, mode) and
                ident['episode_id'] == f'e{index // 3:02}_{mode}', 'fixed own episode identity')
        require(row['status'] == 'scored' and scored[index]['receipt'] == row and
                seal['rows'][index]['status'] == 'predicted' and
                seal['rows'][index]['prediction'] == row['prediction'], 'sealed/scored row join')
        require(artifact(f'input_{index:02}.json') == req and req['input_digest'] == digest(
            {k: v for k, v in req.items() if k != 'input_digest'}), 'original saved forward input')
        entries = r['ledger'][4*index:4*index+4]
        require(all(e['index'] == 4*index+j and e['episode_index'] == index and e['resource'] == RESOURCES[j]
                    and e['completed'] is True for j, e in enumerate(entries)), 'full module chronology')
        ae, api, pred, proj = entries
        require(len(ae['inputs']) == len(proj['inputs']) == 1 and len(pred['inputs']) == 2, 'module arity')
        ain, aout = load(ae['inputs'][0]), load(ae['output'])
        pin, pact, pout = load(pred['inputs'][0]), load(pred['inputs'][1]), load(pred['output'])
        qin, projected = load(proj['inputs'][0]), load(proj['output'])
        history = np.asarray(req['history'], np.float32)
        raw5, mean, std = (np.asarray(req[k], np.float32) for k in ('raw5', 'mean', 'std'))
        require(history.shape == (2, 10) and raw5.shape == (5, 2) and mean.shape == std.shape == (2,)
                and (std > 0).all(), 'normalization dimensions')
        actions = np.concatenate((history, ((raw5-mean)/std).reshape(1, 10)))
        z = np.asarray(req['latents'], np.float32)
        require(ain.shape == (1, 3, 10) and np.array_equal(ain[0], actions) and
                pin.shape == (1, 3, 192) and np.array_equal(pin[0], z) and
                np.array_equal(pact, aout) and pout.shape == (1, 3, 192) and
                qin.shape == projected.shape == (3, 192) and np.array_equal(qin, pout.reshape(3, 192)), 'saved full3 array bridges')
        prediction, label = load(row['prediction']), load(row['label'])
        for kind in ('prefix', 'label'):
            receipt = ev[kind]
            require(p['bindings'][receipt['path']] == receipt['file_sha256'], 'original receipt in prepare closure')
            expected_path = f'{ident["episode_id"]}_encode{5 if kind == "prefix" else 10}/native192.npy'
            require(str(receipt['path']).endswith('/' + expected_path), 'own episode receipt namespace')
        actual, prefix = s.receipt(ev['label']), s.receipt(ev['prefix'])
        require(np.array_equal(label, actual) and np.array_equal(z[2], prefix) and
                np.array_equal(prediction, projected[-1]), 'own factual endpoint/prefix/final-token identity')
        endpoint = decomposition(prediction, actual, ev['goal'])
        old = dict(factual_MSE=endpoint['latent_MSE'], predicted_Q=endpoint['predicted_Q'], actual_Q=endpoint['actual_Q'],
                   predicted_Q_minus_actual_Q=endpoint['predicted_Q_minus_actual_Q'],
                   actual_prefix_terminal_mse=float(np.mean((vector(prefix)-vector(actual))**2)))
        require(old == row['metrics'], 'independent exact teacher metrics recomputation')
        goal_native = vector(ev['goal']).astype(np.float32)
        steps = []
        for token, (target, scope) in enumerate(((z[1], 'saved_source_warmup_anchor_not_own_factual'),
                                               (prefix, 'own_actual_prefix5_context_teacher_forced'),
                                               (actual, 'own_factual_suffix_H1_5_to_10'))):
            steps.append(dict(projected_token_index=token, prediction_horizon_packets=1, packet_physics_steps=5,
                              label_scope=scope, latent_MSE=decomposition(projected[token], target, ev['goal'])['latent_MSE'],
                              prediction_array_sha256=sha(projected[token].tobytes()),
                              actual_array_sha256=sha(np.ascontiguousarray(target).tobytes())))
        out.append(dict(identity=ident, prediction_steps=steps, endpoint=endpoint,
                        prefix_array_sha256=sha(prefix.tobytes()), goal_array_sha256=sha(goal_native.tobytes()),
                        prediction_receipt=s.arrays[str((root / row['prediction']['file']).absolute())],
                        actual_receipt=s.arrays[str(Path(ev['label']['path']).absolute())],
                        goal_receipt=dict(container=r['prepared'], container_sha256=r['prepare_sha256'],
                                          json_pointer=f'/evaluators/{index}/goal', dtype='float32', shape=[192],
                                          array_sha256=sha(goal_native.tobytes()))))
    paired = contrasts(out)
    # Recheck every consumed file at completion; no stale snapshot/TOCTOU success.
    for path, receipt in list(s.files.items()):
        s.read(path, receipt['sha256'])
    return dict(schema=NAME + '.report', status='saved_array_diagnostic_not_admitted',
                input_report=str(report_path), input_report_sha256=report_sha,
                prepare_semantic_sha256=parse(s.read(r['prepared'], r['prepare_sha256']))['semantic_sha256'],
                fixed_denominator=12, independent_tasks=2, rows=out, R_minus_O=paired,
                strict_reversal_count=sum(c['strict_reversal'] for c in paired), layer_denominator=4,
                Q_definition='mean192((latent-goal)^2), float64 arithmetic; lower is better',
                normalization=dict(kind='exact mean over native coordinates', denominator=192,
                                   relative_Q_error='not computed; no division by potentially zero actual_Q'),
                inputs=s.files, arrays=s.arrays, training=False, Qfit=False, selection=False,
                new_model_calls=0, new_physics=0, P1_admission=False, P4_admission=False,
                limitations=['Known-root post-hoc saved diagnostic, not independent confirmation.',
                             'Only token2 is own suffix H1; token0 source anchor, token1 teacher-forced context.',
                             'No intermediate physics-step latent labels; missing labels are not zero-filled.',
                             'Latent errors and Q margins do not establish physical benefit or generalization.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=REPORT)
    parser.add_argument('--report-sha', default=REPORT_SHA)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'exclusive NEW output; preserve old reports')
    result = analyze(args.report, args.report_sha)
    result['implementation_sha256'] = sha(Path(__file__).read_bytes())
    raw = canonical(result)
    require(len(raw) <= 4 * 1024**2, 'bounded report')
    with args.output.open('xb') as stream:
        require(stream.write(raw) == len(raw), 'complete report write')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(output=str(args.output.absolute()), sha256=sha(raw),
                          episodes=12, reversals=result['strict_reversal_count'], layers=4)))


if __name__ == '__main__':
    main()
