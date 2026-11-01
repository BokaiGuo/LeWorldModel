"""Independent B8 saved RGB audit. NumPy/Pillow only; stdout, no model imports."""
import hashlib
import io
import json
import math
from pathlib import Path
import zlib

import numpy as np
from PIL import Image

NAME = 'p1_b8_rgb_saved_audit_20261027_v1'
ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'results/p1_b8_rgb_cuda_20261027_v1_cuda_main01'
REPORT_SHA = '3f1b346979cd8e53e357aba0576a91d11e4268cc922dc84d63f5785622742d8e'
PREPARE_SHA = '5c14c90a5498df36ef476cc037f9d685ccb909b93b1a25fcc524373e7f54a543'
PHYSICAL_SHA = 'e94a6cbf55dd9b408d350e690d17bfdb9ca693b2ef360a12ab904024f272cd7a'
PHYSICAL_PREPARE_SHA = 'e872fdbfef99ef47a00d48ea9263a590cc3f792dd9b9e3184cebdbb56aaad706'
ARMS = ('full_H2_GN', 'full_H2_gradient')
STEPS = (0, 5, 10)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value):
    # Original task/frame/journal identities use default stdlib separators.
    return sha(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def bad(value):
        raise ValueError('nonfinite JSON constant: '+value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)


def decompress(raw, limit=48*1024**2):
    require(len(raw) <= 24*1024**2, 'bounded gzip input')
    d = zlib.decompressobj(31)
    value = d.decompress(raw, limit+1)
    require(len(value) <= limit and d.eof and not d.unused_data and not d.unconsumed_tail,
        'complete single gzip stream required')
    return decode(value)


class Reader:
    """Streaming file hash checks; large weights never deserialized or loaded."""
    def __init__(self):
        self.bindings = {}

    def bind(self, path, expected):
        p = Path(path)
        require(p.is_absolute() and p.is_file() and not p.is_symlink(), 'absolute regular source file')
        require(str(p) not in self.bindings or self.bindings[str(p)] == expected, 'conflicting file bindings')
        if str(p) not in self.bindings:
            h = hashlib.sha256()
            with p.open('rb') as stream:
                for block in iter(lambda: stream.read(1024**2), b''):
                    h.update(block)
            require(h.hexdigest() == expected, 'file SHA drift: '+str(p))
            self.bindings[str(p)] = expected
        return p

    def raw(self, path, expected, limit=16*1024**2):
        p = self.bind(path, expected)
        require(p.stat().st_size <= limit, 'bounded saved input')
        raw = p.read_bytes()
        require(sha(raw) == expected, 'source changed during audit')
        return raw

    def read(self, path, expected):
        return decode(self.raw(path, expected))


def array(value, shape):
    a = np.asarray(value)
    require(a.shape == shape and a.dtype.kind in 'fi' and np.isfinite(a).all(), 'finite real aligned array')
    return a.astype(np.float64)


def native_npy(raw):
    require(len(raw) <= 4096, 'bounded raw NPY')
    stream = io.BytesIO(raw)
    a = np.load(stream, allow_pickle=False)
    require(a.shape == (192,) and a.dtype == np.float32 and np.isfinite(a).all(), 'raw native float32 finite192')
    require(stream.tell() == len(raw), 'no trailing NPY payload')
    return a


def png(raw, pixel_sha):
    with Image.open(io.BytesIO(raw)) as image:
        require(image.format == 'PNG' and image.mode == 'RGB' and image.size == (224, 224), 'original RGB224 PNG')
        rgb = np.asarray(image).copy()
    require(rgb.dtype == np.uint8 and sha(rgb.tobytes()) == pixel_sha, 'PNG decoded pixel SHA')
    return int(np.count_nonzero(np.all(rgb == [65, 105, 225], axis=-1)))


def census(frames, roots):
    require(len(roots) == len(set(roots)) == 2, 'two unique original roots')
    expected = [(r, a, s) for r in roots for a in ARMS for s in STEPS]
    require(len(frames) == 12 and [f['order'] for f in frames] == list(range(12)) and
        [(f['root_sha256'], f['arm'], f['step']) for f in frames] == expected, 'all12 original slots, no filtering or dedup')


def metrics(task, actual, predicted):
    q = array(task['projection'], (192, 192))
    goal = array(task['goal'], (192,))
    actual, predicted = array(actual, (192,)), array(predicted, (192,))
    ra, rp = q@(actual-goal), q@(predicted-goal)
    return dict(actual_Q_mse=float(np.mean(ra*ra)), predicted_Q_mse=float(np.mean(rp*rp)),
        native_prediction_mse=float(np.mean((actual-predicted)**2)))


def close(actual, expected):
    require(type(actual) in (int, float) and math.isfinite(actual) and
        math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), 'saved scalar differs from raw recomputation')


def check_analysis(analysis, rebuilt):
    require(analysis['schema'] == 'p1_b8_rgb_cuda_20261027_v1.analysis' and analysis['frames'] == 12 and
        len(analysis['rows']) == len(rebuilt) == 12 and
        all(analysis[k] is False for k in ('P1_admission', 'training', 'new_Q', 'independent_confirmation')), 'all12 saved analysis scope')
    for a, expected in zip(analysis['rows'], rebuilt, strict=True):
        require(set(a) == set(expected), 'exact diagnostic metric fields')
        for k, v in expected.items():
            if k in ('actual_Q_mse', 'predicted_Q_mse', 'native_prediction_mse'):
                close(a[k], v)
            else:
                require(a[k] == v, 'analysis row identity or physical diagnostic drift')


def check_image_events(events, frames, rows, entries, resources):
    expected, terminal = [], []
    require(len(rows) == 12, 'all12 saved rows')
    for i, (f, row) in enumerate(zip(frames, rows, strict=True)):
        common = dict(frame_order=i, root_sha256=f['root_sha256'], arm=f['arm'], step=f['step'], pixel_sha256=f['pixel_sha256'])
        expected.append(dict(common, kind='frame_started', task_id=f['task_id'], frame_sha256=digest(f)))
        for j, resource in enumerate(('encoder', 'projector')):
            entry = dict(index=2*i+j, frame_order=i, resource=resource, completed=False)
            expected.append(dict(common, kind='module_attempted', entry=entry))
            done = dict(entry, completed=True)
            expected.append(dict(common, kind='module_completed', entry=done))
            terminal.append(done)
        expected.append(dict(common, kind='native_saved', latent_file=row['latent_file'], latent_sha256=row['latent_sha256']))
    require(events == expected and entries == terminal, 'exact72 image events /24 module terminals; no omissions/reorder')
    require(resources == {k: dict(attempted=12, completed=12) for k in ('encoder', 'projector')}, 'all12 resource denominator')


def check_chain(raw):
    require(raw.endswith(b'\n'), 'no torn journal suffix')
    events, previous = [], None
    for i, line in enumerate(raw.splitlines()):
        event = decode(line)
        body = {k: v for k, v in event.items() if k != 'event_sha256'}
        require(event['index'] == i and event['previous_sha256'] == previous and
            digest(body) == event['event_sha256'], 'physical journal hash chain')
        previous = event['event_sha256']
        events.append(event)
    return events


def audit():
    reader = Reader()
    r = reader.read(RUN/'report.json', REPORT_SHA)
    require(r['schema'] == 'p1_b8_rgb_cuda_20261027_v1.report' and r['status'] == 'all12_B8_checkpoints_encoded' and
        r['errors'] == [] and r['first_error'] is None and r['forbidden'] == [], 'actual successful image run')
    require(r['source_verified'] is r['model_unchanged'] is True and
        r['model_hash_before'] == r['model_hash_after'] and r['model_hash_before'] is not None, 'recorded unchanged model')
    for key in ('training', 'new_Q', 'P1_admission', 'independent_confirmation'):
        require(r[key] is False, 'no scope upgrade')
    for key in ('new_predict_calls', 'new_AD_calls', 'new_physical_steps', 'new_independent_samples'):
        require(type(r[key]) is int and r[key] == 0, 'zero new work / no independent sample')
    files = r['output_files_sha256']
    expected_files = {'analysis.json', 'execution_binding.json', 'image.events.jsonl'} | {
        f'frame_{i:02}.{suffix}' for i in range(12) for suffix in ('intent.json', 'result.json', 'native192.npy')}
    require(set(files) == expected_files and {p.name for p in RUN.iterdir()} == expected_files | {'report.json'}, 'exact durable output census')
    output_bytes = sum(p.stat().st_size for p in RUN.iterdir())
    require(r['output_cap_bytes'] == 2*1024**2 and output_bytes <= r['output_cap_bytes'] and
        r['disk_floor_bytes'] == 200*1024**2, 'actual output byte cap / unchanged declared floor')
    for name, h in files.items():
        reader.bind(RUN/name, h)
    binding = reader.read(RUN/'execution_binding.json', files['execution_binding.json'])
    require(binding['prepare_sha256'] == r['prepare_sha256'] == PREPARE_SHA, 'external RGB prepare SHA')
    p = reader.read(binding['prepared'], PREPARE_SHA)
    require(p['schema'] == 'p1_b8_rgb_cuda_20261027_v1.protocol' and p['expected_model_hash'] == r['model_hash_before'] and
        p['model_bindings'] == r['model_bindings'], 'prepared frozen model identity')
    require(p['physical_report_sha256'] == r['physical_report_sha256'] == PHYSICAL_SHA and
        p['physical_prepare_sha256'] == r['physical_prepare_sha256'] == PHYSICAL_PREPARE_SHA, 'original physical provenance')
    require(p['encoder_cap'] == p['projector_cap'] == 12 and
        all(p[k] == 0 for k in ('predict_cap', 'action_encoder_cap', 'AD_cap', 'physical_steps_cap')), 'image-only contract')
    for path, h in p['bindings'].items():
        reader.bind(path, h)
    for path, h in p['model_bindings'].items():
        require(p['bindings'][path] == h, 'model file in source closure')
    physical_dir = ROOT/'results/p1_b8_physical_20261027_v1_main01'
    pr = reader.read(physical_dir/'report.json', PHYSICAL_SHA)
    pp = reader.read(ROOT/'results/p1_b8_physical_20261027_v1_prepare01.json', PHYSICAL_PREPARE_SHA)
    require(pr['status'] == 'physical_calibration_recorded' and pr['errors'] == [] and pr['prepare_sha256'] == PHYSICAL_PREPARE_SHA,
        'original completed physical report')
    pe = check_chain(reader.raw(physical_dir/'journal.jsonl', pr['journal_sha256']))
    require(pe[0]['kind'] == 'run_started' and pe[0]['prepare_sha256'] == PHYSICAL_PREPARE_SHA and
        pe[-1]['kind'] == 'run_finished' and pe[-1]['errors'] == [] and pe[-1]['counts'] == pr['counts'], 'physical run boundaries')
    require([e['receipt'] for e in pe if e['kind'] == 'branch_closed'] == pr['rows'], 'all durable physical branch rows')
    for kind, n in (('reset', 8), ('render', 44), ('step', 40)):
        require(pr['counts'][kind] == dict(attempted=n, completed=n) and
            all(sum(e['kind'] == kind+'_'+suffix for e in pe) == n for suffix in ('attempted', 'returned')), 'physical operation census')
    plans = pp['inputs']['plans']
    require(len(plans) == len(pr['rows']) == 4, 'four original branches')
    roots = [plans[0]['root_sha256'], plans[2]['root_sha256']]
    require([(x['root_sha256'], x['arm']) for x in plans] == [(root, arm) for root in roots for arm in ARMS], 'original GN/GD branch order')
    census(p['frames'], roots)
    require(len(p['tasks']) == 2 and [x['root_sha256'] for x in p['tasks']] == roots, 'two original task references')
    tasks = {}
    for ref in p['tasks']:
        encoded = decompress(reader.raw(ref['task_file'], ref['task_file_sha256']))
        task = encoded['task']
        require(encoded['root_sha256'] == ref['root_sha256'] and encoded['task_id'] == ref['task_id'] == digest(task) and
            encoded['history_semantics'] == p['history'] == 'observed_not_padding_even_when_numerically_identical', 'original task identity/history')
        require(digest({k: task[k] for k in ('goal', 'projection', 'root_context', 'action_mean', 'action_std', 'model_sha256')}) ==
            ref['goal_Q_context_sha256'], 'original goal/Q/context binding')
        q = array(task['projection'], (192, 192))
        require(np.allclose(q, q.T, rtol=0, atol=1e-8) and np.allclose(q@q, q, rtol=0, atol=1e-8) and np.trace(q) > .5,
            'original nonzero orthogonal task Q')
        require(task['model_sha256'] in p['model_bindings'].values() and task['model_config_sha256'] in p['model_bindings'].values(), 'same original task model/config')
        tasks[ref['root_sha256']] = task
    rows = r['rows']
    census(rows, roots)
    rebuilt, all_pixels, latents = [], [], []
    for b, (plan, branch) in enumerate(zip(plans, pr['rows'], strict=True)):
        root, arm = plan['root_sha256'], plan['arm']
        task = tasks[root]
        require(branch['root_sha256'] == root and branch['arm'] == arm and branch['selection_sha256'] == plan['selection_sha256'] and
            branch['status'] == 'completed' and branch['first_error'] is None and branch['secondary_errors'] == [] and
            branch['attempted_steps'] == branch['returned_steps'] == 10 and
            [x['step'] for x in branch['checkpoints']] == list(STEPS), 'complete original branch identity')
        ref = p['tasks'][b//2]
        require(all(plan[k] == ref[k] for k in ('task_file', 'task_file_sha256', 'task_id', 'goal_Q_context_sha256')) and
            plan['history_semantics'] == p['history'], 'plan original task reference, no task/Q substitution')
        own = [e for e in pe if e.get('selection_sha256') == plan['selection_sha256']]
        require([e['plan'] for e in own if e['kind'] == 'branch_started'] == [plan] and
            [e['receipt'] for e in own if e['kind'] == 'checkpoint_saved'] == branch['checkpoints'], 'physical plan/checkpoint journal')
        require([x['command'] for x in branch['trace']] == plan['commands'] and len(branch['trace']) == 10, 'actual commands not reselected')
        require([x['step'] for x in branch['trace']] == list(range(1, 11)) and
            all(x['attempted'] is x['returned'] is True and x['terminated'] is x['truncated'] is False for x in branch['trace']),
            'full ten-step actual chronology, no early-stop replacement')
        original = decompress(reader.raw(plan['origin'], plan['origin_sha256'], 24*1024**2))
        run = original['direct_runs'][0]
        require(original['status'] == 'direct_component_completed_not_admitted' and len(original['direct_runs']) == 1 and
            len(run) == 9 and [x['iteration'] for x in run] == list(range(9)), 'full original B0..B8 path')
        prediction = run[-1]['incumbent']
        require(digest(prediction) == plan['prediction_sha256'] and prediction['commands'] == plan['commands'] and
            prediction['task_id'] == plan['task_id'] == digest(task), 'original B8 strict incumbent prediction')
        for previous, current in zip(run, run[1:]):
            accepted = current['proposal']['cost'] < previous['incumbent']['cost']
            require(current['accepted'] is accepted and current['incumbent'] ==
                (current['proposal'] if accepted else previous['incumbent']), 'strict incumbent selection, no physical feedback')
        vectors = [task['root_context']['latent_history'][-1], *[c['latent_history'][-1] for c in prediction['contexts']]]
        require(len(vectors) == 3, 'root plus two predicted packet boundaries')
        for j, step in enumerate(STEPS):
            i = 3*b+j
            f, row, check = p['frames'][i], rows[i], branch['checkpoints'][j]
            path = Path(check['path'])
            path = path if path.is_absolute() else ROOT/path
            diagnostic = branch['initial_pose_diagnostics'] if j == 0 else branch['trace'][step-1]['pose_diagnostics']
            require(f == dict(order=i, root_sha256=root, arm=arm, step=step, task_id=digest(task), path=str(path),
                file_sha256=check['file_sha256'], pixel_sha256=check['rgb_sha256'], selection_sha256=plan['selection_sha256'],
                prediction_sha256=plan['prediction_sha256'], predicted_native192=vectors[j], physical_diagnostic=diagnostic), 'independent original frame/prediction/diagnostic reconstruction')
            require(path.parent == physical_dir and check['state'] == (branch['checkpoints'][0]['state'] if j == 0 else branch['trace'][step-1]['actual_state']), 'original checkpoint state chronology')
            pixels = png(reader.raw(path, f['file_sha256'], 1024**2), f['pixel_sha256'])
            require(pixels == branch['guard_receipt']['observations'][step]['canonical_agent_pixels'] and pixels > 0, 'PNG canonical agent guard recount')
            all_pixels.append(f['pixel_sha256'])
            stem = f'frame_{i:02}'
            intent = dict(order=i, root_sha256=root, arm=arm, step=step, status='attempted', frame_sha256=digest(f))
            require(reader.read(RUN/(stem+'.intent.json'), files[stem+'.intent.json']) == intent, 'durable original frame intent')
            expected = dict(intent, status='encoded_and_saved', latent_file=stem+'.native192.npy',
                latent_sha256=files[stem+'.native192.npy'], latent_dtype='float32', latent_shape=[192])
            require(row == expected and reader.read(RUN/(stem+'.result.json'), files[stem+'.result.json']) == expected, 'durable raw latent row result')
            z = native_npy(reader.raw(RUN/row['latent_file'], row['latent_sha256'], 4096))
            latents.append(row['latent_sha256'])
            rebuilt.append(dict(order=i, root_sha256=root, arm=arm, step=step, task_id=digest(task),
                **metrics(task, z, vectors[j]), physical_diagnostic=diagnostic))
    require(len(set(f['path'] for f in p['frames'])) == 12, 'no repeated PNG paths')
    events_raw = reader.raw(RUN/'image.events.jsonl', files['image.events.jsonl'])
    require(events_raw.endswith(b'\n'), 'no torn image journal')
    events = [decode(line) for line in events_raw.splitlines()]
    check_image_events(events, p['frames'], rows, r['entries'], r['resources'])
    analysis = reader.read(RUN/'analysis.json', files['analysis.json'])
    check_analysis(analysis, rebuilt)
    summaries = []
    for i in range(0, 12, 3):
        a, _, b = rebuilt[i:i+3]
        summaries.append(dict(root_sha256=a['root_sha256'], arm=a['arm'], denominator=3,
            start_Q_mse=a['actual_Q_mse'], actual_endpoint_Q_mse=b['actual_Q_mse'], predicted_endpoint_Q_mse=b['predicted_Q_mse'],
            actual_endpoint_minus_start=b['actual_Q_mse']-a['actual_Q_mse'],
            predicted_endpoint_minus_start=b['predicted_Q_mse']-a['predicted_Q_mse'],
            endpoint_native_prediction_mse=b['native_prediction_mse']))
    return dict(schema=NAME+'.report', status='all12_saved_evidence_recomputed', input_report_sha256=REPORT_SHA,
        denominator=dict(original_roots=2, branches=4, frames=12, unique_png_paths=12,
            unique_pixel_hashes=len(set(all_pixels)), unique_raw_latent_hashes=len(set(latents)), counted_without_dedup=True),
        resources_reconciled=r['resources'], image_event_count=len(events),
        rgb_run_file_bytes=output_bytes, rgb_run_cap_bytes=r['output_cap_bytes'],
        inherited_physical_public_counts=pr['counts'], bindings_checked=len(reader.bindings), bindings=reader.bindings,
        model_file_hashes_checked=True, model_tensor_unchanged='bound execution receipt before==after; not re-executed',
        rows=rebuilt, branch_summaries=summaries,
        new_neural_calls=0, new_physical_steps=0, new_AD_calls=0, new_Q=False,
        P1_admission=False, independent_confirmation=False, new_independent_samples=0,
        scope='actual saved array/source/resource audit, not a new model run, independent root, stationary-point proof, or theoretical failure')


if __name__ == '__main__':
    print(json.dumps(audit(), sort_keys=True, ensure_ascii=False, allow_nan=False))
