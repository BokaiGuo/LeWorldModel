"""MAIN-only twelve B8 checkpoint encodes; saved prepare/verify, no prediction.

Hash-only small manifest; original gzip task/incumbent/physical journal/PNG
bindings. No copying large tasks, deduplication, fitting, new Q or physics.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import io as memory_io
import json
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import p1_b8_physical_20261027_v1 as physical
from experiments.spectral_reachability import p1_sampling_rgb_cuda_20261027_v1 as small

NAME = 'p1_b8_rgb_cuda_20261027_v1'
ROOT = physical.ROOT
RUN = ROOT/'results/p1_b8_physical_20261027_v1_main01'
PREPARED = ROOT/'results/p1_b8_physical_20261027_v1_prepare01.json'
RUN_SHA = 'e94a6cbf55dd9b408d350e690d17bfdb9ca693b2ef360a12ab904024f272cd7a'
PREPARE_SHA = 'e872fdbfef99ef47a00d48ea9263a590cc3f792dd9b9e3184cebdbb56aaad706'
ARMS, STEPS = physical.ARMS, (0, 5, 10)
CAP, FLOOR, FINAL_RESERVE, PREPARE_CAP = 2*1024**2, 200*1024**2, 128*1024, 200_000
io, source, core, require = small.io, physical.source, physical.core, physical.require
canonical, merge_bindings, task_reference = small.canonical, small.merge_bindings, small.task_reference


def validate_physical(report, prepared, events):
    require(report['schema'] == physical.NAME+'.report' and report['status'] == 'physical_calibration_recorded' and
        report['errors'] == [] and report['prepare_sha256'] == PREPARE_SHA and report['neural_calls'] == 0 and
        report['P1_admission'] is report['model_task_eligible'] is report['stall_certified'] is False and
        report['new_independent_samples'] == 0, 'complete actual B8 physical scope')
    expected = {k:dict(attempted=n, completed=n) for k, n in dict(reset=8, render=44, step=40).items()}
    require(report['counts'] == expected and report['contract'] == prepared['contract'] == physical.contract(), 'original B8 physical contract/budget')
    require(events[0]['kind'] == 'run_started' and events[0]['prepare_sha256'] == PREPARE_SHA and
        events[-1]['kind'] == 'run_finished' and events[-1]['status'] == report['status'] and
        events[-1]['counts'] == expected and events[-1]['errors'] == [], 'physical journal complete start/end')
    same_rows = [e['receipt'] for e in events if e['kind'] == 'branch_closed']
    require(same_rows == report['rows'] and len(same_rows) == len(prepared['inputs']['plans']) == 4, 'all four durable B8 branches')
    for kind, n in dict(reset=8, render=44, step=40).items():
        for suffix in ('attempted', 'returned'):
            require(sum(e['kind'] == kind+'_'+suffix for e in events) == n, 'full physical operation journal '+kind)
    plans = prepared['inputs']['plans']; roots = [plans[0]['root_sha256'], plans[2]['root_sha256']]
    require(len(set(roots)) == 2 and [(x['root_sha256'], x['arm']) for x in plans] == [(r, a) for r in roots for a in ARMS], 'original source slots GN/GD order')
    for row, plan in zip(report['rows'], plans, strict=True):
        require(all(row[k] == plan[k] for k in ('root_sha256', 'arm', 'selection_sha256')) and row['status'] == 'completed' and
            row['initial_exact'] is row['full_horizon'] is row['endpoint_rgb_fresh'] is True and row['first_error'] is None and
            row['secondary_errors'] == [] and row['attempted_steps'] == row['returned_steps'] == 10 and
            [c['step'] for c in row['checkpoints']] == list(STEPS), 'complete original branch and checkpoints')
        trace = row['trace']; guard = row['guard_receipt']
        require(len(trace) == 10 and [s['step'] for s in trace] == list(range(1, 11)) and [s['command'] for s in trace] == plan['commands'] and
            all(s['attempted'] is s['returned'] is True and s['terminated'] is s['truncated'] is False for s in trace), 'ten original raw commands/real chronology')
        require(guard['root_id'] == row['root_sha256'] and guard['abstained'] is guard['P1_admission'] is False and
            guard['primitive_commands_authorized'] == 10 and [s['primitive_step'] for s in guard['observations']] == list(range(11)) and
            all(s['canonical_agent_pixels'] > 0 for s in guard['observations']), 'unchanged full RGB guard')
        own = [e for e in events if e.get('selection_sha256') == plan['selection_sha256']]
        require([e['plan'] for e in own if e['kind'] == 'branch_started'] == [plan] and
            [e['receipt'] for e in own if e['kind'] == 'checkpoint_saved'] == row['checkpoints'], 'actual plan/checkpoint journal binding')


def task_table(prepared):
    tasks = {}
    for plan in prepared['inputs']['plans']:
        path = Path(plan['task_file']); require(io.sha(path) == plan['task_file_sha256'], 'original gzip task SHA')
        encoded = physical.gz(path); task = encoded['task']; core.validate_task(task)
        require(encoded['root_sha256'] == plan['root_sha256'] and encoded['task_id'] == plan['task_id'] == core.task_id(task) and
            encoded['history_semantics'] == plan['history_semantics'] == 'observed_not_padding_even_when_numerically_identical', 'original observed task identity/semantics')
        root = plan['root_sha256']; require(root not in tasks or tasks[root] == task, 'same root same complete task')
        tasks[root] = task
    require(len(tasks) == 2, 'two source tasks'); return tasks


def collect():
    require(io.sha(RUN/'report.json') == RUN_SHA, 'fixed actual B8 physical report SHA')
    prepared = physical.verify(PREPARED, PREPARE_SHA); report = physical.read(RUN/'report.json')
    require(io.sha(RUN/'journal.jsonl') == report['journal_sha256'], 'actual whole physical journal SHA')
    events = source.read_journal(RUN/'journal.jsonl'); validate_physical(report, prepared, events)
    tasks = task_table(prepared)
    bindings = {str(RUN/'report.json'):RUN_SHA, str(PREPARED):PREPARE_SHA, str(RUN/'journal.jsonl'):report['journal_sha256']}
    merge_bindings(bindings, prepared['sources']); merge_bindings(bindings, prepared['inputs']['bindings'])
    refs, frames = [], []
    for row, plan in zip(report['rows'], prepared['inputs']['plans'], strict=True):
        task = tasks[plan['root_sha256']]; reference = task_reference(plan['root_sha256'], task)
        require(reference['task_id'] == plan['task_id'] and reference['goal_Q_context_sha256'] == plan['goal_Q_context_sha256'], 'original goalQ/context unchanged')
        if row['arm'] == ARMS[0]:
            refs.append(dict(reference, task_file=plan['task_file'], task_file_sha256=plan['task_file_sha256']))
        require(io.sha(plan['origin']) == plan['origin_sha256'], 'original complete B8 incumbent gzip report')
        original = physical.gz(plan['origin']); prediction = original['direct_runs'][0][-1]['incumbent']
        require(original['status'] == 'direct_component_completed_not_admitted' and len(original['direct_runs'][0]) == 9 and
            source.digest(prediction) == plan['prediction_sha256'] and prediction['commands'] == plan['commands'], 'original B8 final strict incumbent, no reselection')
        core.check_prediction(task, prediction)
        vectors = [task['root_context']['latent_history'][-1], *[c['latent_history'][-1] for c in prediction['contexts']]]
        require(len(vectors) == 3, 'root plus two original predicted boundaries')
        for i, check in enumerate(row['checkpoints']):
            path = Path(check['path']); path = path if path.is_absolute() else ROOT/path; path = path.absolute()
            require(path.parent == RUN and not path.is_symlink(), 'own B8 run PNG only')
            rgb, _ = io.pixels(path, check['file_sha256'], check['rgb_sha256'])
            require(int(np.count_nonzero(np.all(rgb == [65, 105, 225], axis=-1))) == row['guard_receipt']['observations'][STEPS[i]]['canonical_agent_pixels'], 'saved checkpoint guard recount')
            require(check['state'] == (row['checkpoints'][0]['state'] if i == 0 else row['trace'][STEPS[i]-1]['actual_state']), 'PNG actual state chronology')
            merge_bindings(bindings, {str(path):check['file_sha256']})
            frames.append(dict(order=len(frames), root_sha256=row['root_sha256'], arm=row['arm'], step=STEPS[i],
                task_id=reference['task_id'], path=str(path), file_sha256=check['file_sha256'], pixel_sha256=check['rgb_sha256'],
                selection_sha256=plan['selection_sha256'], prediction_sha256=plan['prediction_sha256'],
                predicted_native192=vectors[i], physical_diagnostic=row['initial_pose_diagnostics'] if i == 0 else row['trace'][STEPS[i]-1]['pose_diagnostics']))
    metadata, _ = small.rgb24.storage.native.native_metadata()
    require(all(t['model_sha256'] in metadata['model_bindings'].values() and t['model_config_sha256'] in metadata['model_bindings'].values() for t in tasks.values()), 'same frozen original model/config')
    merge_bindings(bindings, metadata['model_bindings'])
    own = set(io.source_files()) | {Path(__file__).resolve(), Path(physical.__file__).resolve(), Path(small.__file__).resolve(),
        Path(small.rgb24.__file__).resolve(), Path(small.rgb24.storage.__file__).resolve(), Path(small.rgb24.storage.native.__file__).resolve(),
        ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md')}
    merge_bindings(bindings, {str(f.resolve()):io.sha(f) for f in own})
    physical.physical.bindings_check(bindings); validate_frames(frames, tasks)
    require(len({f['path'] for f in frames}) == 12, '12 distinct original paths; no pixel dedup')
    direct = physical.read(prepared['inputs']['arguments']['direct_report'])
    return dict(schema=NAME+'.protocol', frames=frames, tasks=refs, bindings=bindings, **metadata,
        expected_model_hash=direct['model_hash_before'], physical_report_sha256=RUN_SHA, physical_prepare_sha256=PREPARE_SHA,
        encoder_cap=12, projector_cap=12, predict_cap=0, action_encoder_cap=0, AD_cap=0, physical_steps_cap=0,
        output_cap_bytes=CAP, disk_floor_bytes=FLOOR, prepare_cap_bytes=PREPARE_CAP, final_reserve_bytes=FINAL_RESERVE,
        normalization='exact original p0.image_transform -> frozen model.encode -> encoder CLS + existing projector',
        history='observed_not_padding_even_when_numerically_identical', training=False, new_Q=False,
        P1_admission=False, independent_confirmation=False, new_independent_samples=0,
        scope='all12 B8 GN/GD saved physical checkpoints, no selection/feedback or scientific admission')


def validate_frames(frames, tasks):
    require(len(frames) == 12 and len(tasks) == 2, 'complete twelve frames/two original tasks')
    roots = [frames[0]['root_sha256'], frames[6]['root_sha256']]
    require(len(set(roots)) == 2 and set(roots) == set(tasks), 'same two original roots')
    require([(f['root_sha256'], f['arm'], f['step']) for f in frames] == [(r, a, s) for r in roots for a in ARMS for s in STEPS], 'fixed source-slot GN/GD 0/5/10 order')
    for i, f in enumerate(frames):
        require(f['order'] == i and f['task_id'] == core.task_id(tasks[f['root_sha256']]), 'frame task/order identity')
        core.array(f['predicted_native192'], (192,))


def analyze(frames, values, tasks):
    validate_frames(frames, tasks); values = core.array(values, (12, 192)); rows = []
    for f, native in zip(frames, values, strict=True):
        task = tasks[f['root_sha256']]; predicted = core.array(f['predicted_native192'], (192,))
        rows.append(dict(order=f['order'], root_sha256=f['root_sha256'], arm=f['arm'], step=f['step'], task_id=f['task_id'],
            native_prediction_mse=float(np.mean((native-predicted)**2)), predicted_Q_mse=core.model_cost(task, predicted),
            actual_Q_mse=core.model_cost(task, native), physical_diagnostic=f['physical_diagnostic']))
    return dict(schema=NAME+'.analysis', rows=rows, frames=12, training=False, new_Q=False, P1_admission=False,
                independent_confirmation=False, scope='original same goalQ diagnostics, not physical improvement or admission')


def prepare(output):
    output = Path(output).absolute(); require(not output.exists(), 'exclusive NEW prepare file')
    p = collect(); raw = canonical(p)
    require(len(raw) < PREPARE_CAP and shutil.disk_usage(output.parent).free >= FLOOR+len(raw)+4096, 'prepare <200KB /200MiB floor')
    with output.open('xb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
    io.fsync_directory(output.parent); return io.sha(output)


def verify(prepared, expected):
    source.hash_value(expected); io.regular(prepared, PREPARE_CAP)
    require(io.sha(prepared) == expected, 'external small prepare SHA')
    p = physical.read(prepared); require(p == collect(), 'complete original gzip/task/incumbent/PNG/model/source reconstruction'); return p


def check_budget(used, size, free, final=False):
    require(used+size <= CAP-(0 if final else FINAL_RESERVE), 'shared 2MiB output cap')
    require(free >= FLOOR+size, '200MiB write floor')


class Store:
    def __init__(self, output):
        self.output = Path(output).absolute()
        require(not self.output.exists() and shutil.disk_usage(self.output.parent).free >= FLOOR+CAP, 'NEW 2MiB run +200MiB reserve')
        self.output.mkdir(); io.fsync_directory(self.output.parent)
    def write(self, name, raw, *, append=False, final=False):
        require(Path(name).name == name and name not in ('', '.', '..'), 'flat output filename')
        require(not (self.output/name).is_symlink(), 'no output symlink')
        used = sum(p.stat().st_size for p in self.output.iterdir() if p.is_file())
        check_budget(used, len(raw), shutil.disk_usage(self.output).free, final)
        with (self.output/name).open('ab' if append else 'xb') as f:
            require(f.write(raw) == len(raw), 'short write'); f.flush(); os.fsync(f.fileno())
        io.fsync_directory(self.output); return io.sha(self.output/name)
    def put(self, name, value, *, final=False): return self.write(name, canonical(value), final=final)
    def native(self, name, value):
        require(value.dtype == np.float32 and value.shape == (192,) and np.isfinite(value).all(), 'raw native float32 finite192')
        buf = memory_io.BytesIO(); np.save(buf, value, allow_pickle=False); return self.write(name, buf.getvalue())


class Ledger:
    """Separate synchronized module-return counts, nonrefundable attempts."""
    def __init__(self, emit): self.emit = emit; self.frame = None; self.next_order = 0; self.entries = []; self.forbidden = []; self.failed = False
    def begin(self, frame):
        require(not self.failed and self.frame is None and frame['order'] == self.next_order < 12, 'once-only all12 frame order')
        self.frame = frame; self.next_order += 1
    def event(self, kind, **fields):
        require(self.frame is not None, 'active original frame')
        f = self.frame; self.emit(dict(kind=kind, frame_order=f['order'], root_sha256=f['root_sha256'],
            arm=f['arm'], step=f['step'], pixel_sha256=f['pixel_sha256'], **fields))
    def reject(self, kind):
        self.failed = True; self.forbidden.append(kind)
        self.event('forbidden', resource=kind); raise ValueError('image-only forbidden '+kind)
    def call(self, kind, callback, device='cuda', shape=None, synchronize=lambda:None):
        try:
            require(not self.failed and self.frame is not None and device == 'cuda', 'CUDA-only active native encode')
            require(kind in ('encoder', 'projector') and shape == ((1, 3, 224, 224) if kind == 'encoder' else (1, 192)), 'one RGB1 module shape')
            own = [e for e in self.entries if e['frame_order'] == self.frame['order']]
            require(not any(e['resource'] == kind for e in own) and sum(e['resource'] == kind for e in self.entries) < 12, 'one call per frame /12 cap')
            if kind == 'projector': require(any(e['resource'] == 'encoder' and e['completed'] for e in own), 'encoder before existing projector')
            e = dict(index=len(self.entries), frame_order=self.frame['order'], resource=kind, completed=False)
            self.entries.append(e); self.event('module_attempted', entry=deepcopy(e))
            value = callback(); synchronize(); e['completed'] = True
            self.event('module_completed', entry=deepcopy(e)); return value
        except BaseException: self.failed = True; raise
    def end(self):
        require(not self.failed and [e['resource'] for e in self.entries if e['frame_order'] == self.frame['order'] and e['completed']] == ['encoder', 'projector'], 'two synchronized native module returns')
        self.frame = None
    def counts(self):
        return {k:dict(attempted=sum(e['resource'] == k for e in self.entries),
            completed=sum(e['resource'] == k and e['completed'] for e in self.entries)) for k in ('encoder', 'projector')}


def install_hooks(t, model, ledger, hooks, device):
    for module, kind in ((model.encoder, 'encoder'), (model.projector, 'projector')):
        original = module.forward
        def wrapped(*args, _original=original, _kind=kind, **kwargs):
            tensor = args[0]
            return ledger.call(_kind, lambda:_original(*args, **kwargs), device=getattr(getattr(tensor, 'device', None), 'type', None),
                shape=tuple(tensor.shape), synchronize=lambda:t.cuda.synchronize(device))
        hooks.install(module, 'forward', wrapped)
    targets = [(model, 'predict'), (model.predictor, 'forward'), (model.action_encoder, 'forward'),
        (t.autograd, 'grad'), (t.autograd, 'backward'), (t.autograd, '_engine_run_backward'), (t.autograd.graph, '_engine_run_backward'),
        (t.autograd.functional, 'jacobian'), (t.autograd.functional, 'hessian'), (t.autograd.functional, 'jvp'), (t.autograd.functional, 'vjp')]
    if hasattr(model, 'rollout'): targets.append((model, 'rollout'))
    if hasattr(t, 'Tensor'): targets.append((t.Tensor, 'backward'))
    if hasattr(t, 'func'): targets.extend((t.func, k) for k in ('jacrev', 'jacfwd', 'hessian', 'jvp', 'vjp') if hasattr(t.func, k))
    for obj, key in targets:
        if not hasattr(obj, key): continue
        def denied(*args, _key=key, **kwargs): return ledger.reject(_key)
        hooks.install(obj, key, denied)


class Native:
    def __init__(self, torch, model, device): self.t, self.model, self.device = torch, model, device
    def __call__(self, image):
        from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
        with self.t.no_grad():
            pixels = p0._image_tensor(image, transform=p0.image_transform(), device=self.device)
            z = self.model.encode(dict(pixels=pixels))['emb']
        require(tuple(z.shape) == (1, 1, 192) and z.dtype == self.t.float32 and z.device == self.device, 'native CUDA RGB1 projected CLS192')
        return z[0, 0].detach().cpu().numpy()


def encode_frames(frames, encode, ledger, store, rows, errors, reader=io.pixels):
    rows.extend(dict(order=f['order'], root_sha256=f['root_sha256'], arm=f['arm'], step=f['step'], status='not_attempted') for f in frames)
    for frame, row in zip(frames, rows, strict=True):
        if errors: break
        try:
            ledger.begin(frame); row['status'] = 'attempted'; stem = 'frame_'+f"{frame['order']:02}"
            row['frame_sha256'] = source.digest(frame); store.put(stem+'.intent.json', row)
            ledger.event('frame_started', task_id=frame['task_id'], frame_sha256=row['frame_sha256'])
            image, _ = reader(frame['path'], frame['file_sha256'], frame['pixel_sha256'])
            value = np.asarray(encode(image))
            require(value.dtype == np.float32 and value.shape == (192,) and np.isfinite(value).all(), 'unmodified native float32 latent192')
            name = stem+'.native192.npy'; hashed = store.native(name, value)
            row.update(status='encoded_and_saved', latent_file=name, latent_sha256=hashed, latent_dtype='float32', latent_shape=[192])
            store.put(stem+'.result.json', row); ledger.event('native_saved', latent_file=name, latent_sha256=hashed)
            ledger.end()
        except BaseException as exc:
            ledger.failed = True; errors.append(io.error('frame_encoding', exc, frame_order=frame['order']))
            row.update(status='failed', error=deepcopy(errors[-1]))
            try: store.put('frame_'+f"{frame['order']:02}"+'.failure.json', row)
            except BaseException as secondary: errors.append(io.error('frame_failure_preservation', secondary))


def run(prepared, expected, output):
    p = verify(prepared, expected); tasks = task_table(physical.read(PREPARED)); validate_frames(p['frames'], tasks)
    require([dict(task_reference(r['root_sha256'], tasks[r['root_sha256']]), task_file=r['task_file'], task_file_sha256=r['task_file_sha256']) for r in p['tasks']] == p['tasks'], 'runtime original task references')
    store = Store(output); store.put('execution_binding.json', dict(prepared=str(Path(prepared).resolve()), prepare_sha256=expected))
    def emit(event): store.write('image.events.jsonl', canonical(event), append=True)
    ledger = Ledger(emit); rows = []; errors = []; hooks = io.Hooks(); model = None
    before = after = host = None; source_ok = False; peaks = None
    try:
        os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host = require_cuda_allocation('cuda:0'); device = torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(device)
        model = load_compatible_pretrained(p['policy'], device=device).eval().requires_grad_(False)
        io.frozen(model); require(all(b.device == device for b in model.buffers()), 'frozen CUDA buffers')
        before = io.tensor_hash(model); require(before == p['expected_model_hash'], 'same actual B8 frozen model tensor state')
        install_hooks(torch, model, ledger, hooks, device)
        encode_frames(p['frames'], Native(torch, model, device), ledger, store, rows, errors)
        require(not errors and len(rows) == 12 and all(r['status'] == 'encoded_and_saved' for r in rows), 'all12 durable raw latents before analysis')
        values = [np.load(store.output/r['latent_file'], allow_pickle=False) for r in rows]
        store.put('analysis.json', analyze(p['frames'], values, tasks))
        torch.cuda.synchronize(device); peaks = dict(allocated_bytes=torch.cuda.max_memory_allocated(device), reserved_bytes=torch.cuda.max_memory_reserved(device))
    except BaseException as exc: errors.append(io.error('execution', exc))
    finally:
        hooks.restore(errors)
        if model is not None:
            try: io.frozen(model); after = io.tensor_hash(model)
            except BaseException as exc: errors.append(io.error('final_model', exc))
        try: verify(prepared, expected); source_ok = True
        except BaseException as exc: errors.append(io.error('final_source', exc))
        complete = (not errors and not ledger.failed and not ledger.forbidden and source_ok and before is not None and before == after and
            len(rows) == 12 and all(r['status'] == 'encoded_and_saved' for r in rows) and ledger.counts() == {k:dict(attempted=12, completed=12) for k in ('encoder', 'projector')})
        result = dict(schema=NAME+'.report', status='all12_B8_checkpoints_encoded' if complete else 'partial_or_failed',
            prepare_sha256=expected, physical_report_sha256=RUN_SHA, physical_prepare_sha256=PREPARE_SHA,
            rows=rows, errors=errors, first_error=deepcopy(errors[0]) if errors else None, resources=ledger.counts(), entries=ledger.entries,
            forbidden=ledger.forbidden, host=host, model_bindings=p['model_bindings'], model_hash_before=before, model_hash_after=after,
            model_unchanged=before is not None and before == after, source_verified=source_ok, recorded_peak_cuda_bytes=peaks,
            new_predict_calls=0, new_AD_calls=0, new_physical_steps=0, training=False, new_Q=False, P1_admission=False,
            independent_confirmation=False, new_independent_samples=0, output_cap_bytes=CAP, disk_floor_bytes=FLOOR,
            completion_semantics='per-module completed after forward return AND CUDA synchronize; raw native NPY fsynced before analysis')
        result['output_files_sha256'] = {f.name:io.sha(f) for f in store.output.iterdir() if f.is_file()}
        store.put('report.json', result, final=True)
    require(complete, 'partial B8 encoding; preserve all durable artifacts; no automatic retry'); return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('command', choices=('prepare', 'verify', 'run'))
    parser.add_argument('--prepared', type=Path); parser.add_argument('--prepare-sha'); parser.add_argument('--output', type=Path)
    a = parser.parse_args()
    result = prepare(a.output) if a.command == 'prepare' else verify(a.prepared, a.prepare_sha)['schema'] if a.command == 'verify' else run(a.prepared, a.prepare_sha, a.output)['status']
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__': main()
