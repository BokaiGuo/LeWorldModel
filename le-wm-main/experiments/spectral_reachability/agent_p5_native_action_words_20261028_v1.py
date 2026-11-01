"""MAIN HOST CUDA-only native action-word prediction; no witness claim.

Prepare/verify/consume are saved-only NumPy/stdlib, no torch or pickle imports.
Four fixed two-packet programs, no JVP/fidelity rerun, goal fitting or physics.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np

TAG = 'agent_p5_native_action_words_20261028_v1'
ROOT = Path(__file__).resolve().parents[2]
PARENT = 'p5_jepa_fresh_fidelity_main_20261027_v12'
MANIFEST = ROOT / f'results/{PARENT}/manifest_frozen.json'
REPORT = ROOT / f'results/{PARENT}/cuda_main_v1/report.json'
PINS = {str(MANIFEST): '41ef9bf8f5c5ae5c4ff8890a25e6a83137b7e637058a8033281a5eabc9cea07f',
        str(REPORT): 'd8667a4e8537a5e005d13507d4d05b8d7907996549889decc8a00e1db873eda6'}
LOADER = 'p5_jepa_native_action_response_20261027_v8'
LOADER_SHA = 'fc915ddee2e53ed58366994684fe2190da5eaaeb8323ad608ffe3292534c936a'
SEEDS = (2026102702, 2026102703, 2026102704)
PATCHES, FEATURES, PROP = 256, 384, 16
WORDS = ('xy', 'yx', 'x_minus_x', 'zero_zero')
EXPECTED = dict(official_encoder=0, proprio_encoder=1, predictor=8)
CAP, FLOOR, WALL = 96 * 1024**2, 200 * 1024**2, 600
NATIVE = {
 'external/jepa-wms/app/vjepa_wm/video_wm.py': '657bd3e9eb2ec35442cc06e208b4632a53503a35a83a96abbbf82608d2fe9154',
 'external/jepa-wms/app/vjepa_wm/modelcustom/simu_env_planning/vit_enc_preds.py': '105a6768050e4b34c95de8d7d0cc0c7ba07b69d364acf0649ba2625bacce907b',
 'external/jepa-wms/configs/vjepa_wm/pt_sweep/pt_4f_fsk5_ask1_r224_vjtranoaug_predAdaLN_ftprop_depth6_repro_2roll_save.yaml': '7a16e8edbf40b260aab303949e90c1879dd5f778d51f3b6d6cb4bea1d99969e2',
}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha(path):
    path = Path(path)
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
            'regular nonsymlink file required: ' + str(path))
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path, expected=None):
    actual = sha(path)
    require(expected is None or actual == expected, 'JSON SHA mismatch: ' + str(path))
    def pairs(items):
        d = {}
        for k, v in items:
            require(k not in d, 'duplicate JSON key')
            d[k] = v
        return d
    require(Path(path).stat().st_size <= 8 * 1024**2, 'JSON size cap')
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda s: require(False, 'nonfinite JSON: ' + s))


def write_new(path, value):
    with Path(path).open('xb') as f:
        f.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b'\n')
        f.flush()
        os.fsync(f.fileno())


def policy():
    return dict(schema=TAG, seeds=list(SEEDS), words=list(WORDS), suffix_raw_steps=10,
        packets=2, raw_coordinate_dim=2, packet_width=10,
        action_units='native relative u64, pixel_displacement=100*u; support [-1,1] is a policy, not env Box',
        programs='five +.1 x then five +.1 y; reverse order; +.1 x then -.1 x; ten exact zero',
        context='v12 saved patches0/5/10, actual normalized proprio0/5/10 and original between-state actions',
        feedback='append raw predictor visual AND patchwise proprio features; shift both contexts and actions; no teacher forcing',
        semantics='VideoWM.forward_pred AdaLN reshape; EncPredWM.unroll predict_proprio append with actual historical actions, no normalized-zero padding',
        shared='one proprio forward/root shared by all four independent branches; no branch-prefix reuse',
        counts_per_root=EXPECTED, counts_total={k: 3*v for k, v in EXPECTED.items()},
        forward_samples_per_root=EXPECTED, context_frames_per_root=dict(official_encoder=0, proprio_encoder=3, predictor=24),
        AD=0, JVP=0, VJP=0, SVD=0, backward=0, physics=0, FLOPs=None,
        target_used=False, goal_scoring=False, candidate_selection=False, fitting=False,
        training=False, adapter=False, known_roots_not_fresh=True,
        metrics='terminal feature word-order/cancellation contrasts only, no task success or predicted physical units',
        output_cap_bytes=CAP, free_floor_bytes=FLOOR, wall_seconds=WALL,
        scratch_free_cuda_bytes=4*1024**3, cpu_neural_fallback=False,
        physical_release=False, action_witness_established=False,
        scientific_admission=False, P5_admission=False, C2_established=False)


def programs():
    x = np.tile(np.array([.1, 0.], np.float64), (5, 1))
    y = x[:, ::-1].copy()
    z = np.zeros_like(x)
    return dict(zip(WORDS, (np.stack((x, y)), np.stack((y, x)),
                           np.stack((x, -x)), np.stack((z, z)))))


def normalize(raw, stats):
    raw = np.asarray(raw)
    require(raw.shape == (2, 5, 2) and raw.dtype == np.float64
            and np.isfinite(raw).all() and np.all(np.abs(raw) <= 1), 'predeclared raw two-packet support')
    mean, std = (np.asarray(stats[k], np.float32) for k in ('ACTION_MEAN', 'ACTION_STD'))
    require(mean.shape == std.shape == (2,) and np.isfinite(mean).all()
            and np.isfinite(std).all() and np.all(std > 0), 'frozen native action statistics')
    projected = (100. * raw).astype(np.float32) / np.float32(100.)
    return ((projected - mean) / std).reshape(2, 10)


def check_array(value, shape):
    a = np.asarray(value)
    require(a.shape == shape and a.dtype == np.float32 and np.isfinite(a).all(), 'finite original float32 array')
    return a


def array(record, shape, dtype=np.float32):
    p = Path(record['path'])
    require(sha(p) == record['sha256'] and p.stat().st_size == record['bytes'], 'array file binding')
    require(record['shape'] == list(shape) and record['dtype'] == str(np.dtype(dtype)), 'array metadata')
    require(p.stat().st_size <= int(np.prod(shape)) * np.dtype(dtype).itemsize + 1024, 'array size cap')
    a = np.load(p, allow_pickle=False, max_header_size=1024)
    require(a.shape == shape and a.dtype == np.dtype(dtype) and np.isfinite(a).all(), 'array shape/dtype/finite')
    return a


def bound_record(path, shape, bindings):
    path = Path(path)
    h = sha(path)
    require(bindings.get(str(path)) == h, 'input absent from saved closure: ' + str(path))
    r = dict(path=str(path), sha256=h, bytes=path.stat().st_size, shape=list(shape), dtype='float32')
    array(r, shape)
    return r


def verify_bindings(bindings):
    require(isinstance(bindings, dict) and bindings, 'nonempty closure')
    for path, expected in bindings.items():
        require(sha(path) == expected, 'frozen binding drift: ' + path)


def prepare():
    """Pinned v12 saved evidence only; no import/execute of archived v8.prepare."""
    m = read_json(MANIFEST, PINS[str(MANIFEST)])
    r = read_json(REPORT, PINS[str(REPORT)])
    require(m['schema'] == r['schema'] == PARENT and m['status'] == 'prepared_non_neural', 'v12 schema')
    require(r['status'] == 'fresh_native_panel_fidelity_recorded' and r['cuda_executed'] is True
            and r['error'] is None and r['finalization_errors'] == [] and r['source_check_errors'] == []
            and r['sources_unchanged'] is True and r['model_unchanged'] is True
            and r['model_before'] == r['model_after'] and r['protocol'] == m['protocol']
            and r['manifest_sha256'] == digest(m), 'complete actual v12 receipt required')
    require([x['seed'] for x in m['roots']] == [x['seed'] for x in r['rows']] == list(SEEDS), 'fixed root order')
    bindings = dict(m['bindings'])
    bindings.update(PINS)
    source = ROOT / 'experiments/spectral_reachability' / (LOADER + '.py')
    require(sha(source) == LOADER_SHA and bindings.get(str(source)) == LOADER_SHA, 'sealed CUDA loader')
    for relative, expected in NATIVE.items():
        require(bindings.get(str(ROOT / relative)) == expected, 'native append/config source not certified')
    roots = []
    for spec, saved in zip(m['roots'], r['rows']):
        require(spec['root_sha256'] == saved['root_sha256'] and saved['status'] == 'completed'
                and saved['counts_match'] is True and saved['frames_match'] is True
                and saved['model_after'] == r['model_before'], 'complete root binding')
        patches = saved['artifacts']['official_patch_features']
        require(bindings.get(patches['path'], patches['sha256']) == patches['sha256'], 'artifact conflict')
        bindings[patches['path']] = patches['sha256']
        array(patches, (4, PATCHES, FEATURES))  # frame15 in parent file, excluded from predictor/scoring
        base = Path(spec['factual']) / 'a/consumer'
        props = bound_record(base / 'context_proprio.npy', (3, 4), bindings)
        packets = bound_record(base / 'normalized_packets.npy', (3, 10), bindings)
        require(np.array_equal(array(packets, (3, 10)), spec['controls']['actual']), 'packet identity')
        goal_path = Path(spec['factual']) / 'a/frame_10/full_public_state.json'
        require(bindings.get(str(goal_path)) == sha(goal_path), 'source root/goal public binding')
        plan = {name: dict(raw=raw.tolist(), normalized=normalize(raw, spec['stats']).tolist())
                for name, raw in programs().items()}
        roots.append(dict(seed=spec['seed'], root_sha256=spec['root_sha256'], factual=spec['factual'],
            goal_source=dict(path=str(goal_path), sha256=bindings[str(goal_path)]),
            patches=patches, proprio=props, packets=packets, stats=spec['stats'], plans=plan))
    for path in (Path(__file__).resolve(), ROOT / 'tests' / ('test_' + TAG + '.py'),
                 ROOT / 'innerdocs' / (TAG + '.md')):
        bindings[str(path)] = sha(path)
    verify_bindings(bindings)
    return dict(schema=TAG, status='prepared_for_MAIN_model_only', policy=policy(), roots=roots,
        bindings=bindings, loader_manifest=m['parent_manifest'], expected_model=r['model_before'],
        source_report_sha256=PINS[str(REPORT)], scientific_admission=False, physical_release=False)


def verify(m):
    require(m == prepare(), 'prepared contract/source/evidence drift')


def rollout(predict, visual, proprio, history, words, emit):
    """Exact causal array seam shared with CUDA; fixtures are not model evidence.

    Shapes parameterized by supplied cached context for tiny analytic RAM tests.
    Only original float32 concatenate/reshape is done here, no neural arithmetic.
    """
    require(tuple(words) == WORDS, 'all four fixed words, in order')
    require(visual.ndim == 3 and visual.shape[0] == 3 and proprio.shape[:2] == visual.shape[:2]
            and proprio.shape[0] == 3 and history.shape == (2, 10), 'native context axes')
    require(all(a.dtype == np.float32 and np.isfinite(a).all() for a in (visual, proprio, history)), 'context dtype/finite')
    terminals = {}
    for word in WORDS:
        controls = check_array(words[word], (2, 10))
        v, p, h = visual.copy(), proprio.copy(), history.copy()
        for step in range(2):
            a = np.concatenate((h, controls[step:step+1]), axis=0)
            for name, val in (('visual_in', v), ('proprio_in', p), ('actions_in', a)):
                emit(word, step, name, val)
            out_v, out_p = predict(v, a, p)
            out_v = check_array(out_v, v.shape)
            out_p = check_array(out_p, p.shape)
            emit(word, step, 'visual_out', out_v)
            emit(word, step, 'proprio_out', out_p)
            v = np.concatenate((v[1:], out_v[-1:]), axis=0)
            p = np.concatenate((p[1:], out_p[-1:]), axis=0)
            h = a[1:].copy()
        terminals[word] = v[-1].copy()
    return terminals


def contrasts(terminals):
    require(tuple(terminals) == WORDS, 'complete word denominator')
    x = [np.asarray(terminals[k], np.float64) for k in WORDS]
    require(all(a.shape == x[0].shape and np.isfinite(a).all() for a in x), 'terminal shape/finite')
    return dict(xy_vs_yx_patch_mse=float(np.mean((x[0]-x[1])**2)),
                x_minus_x_vs_zero_patch_mse=float(np.mean((x[2]-x[3])**2)),
                interpretation='model-only feature contrasts, not a physical witness or task benefit')


class Meter:
    """Durable attempted/returned hooks; failed attempts never refunded."""
    def __init__(self, event, guard=lambda: None):
        self.attempted, self.completed = dict.fromkeys(EXPECTED, 0), dict.fromkeys(EXPECTED, 0)
        self.event, self.guard = event, guard

    def before(self, key, samples):
        self.guard()
        require(type(samples) is int and samples == 1, 'batch1 only, API and sample units remain separate')
        require(key in EXPECTED and self.attempted[key] < EXPECTED[key], 'predispatch cap: ' + key)
        self.attempted[key] += 1
        self.event('attempted', self.snapshot())

    def after(self, key, samples):
        require(samples == 1 and self.completed[key] < self.attempted[key], 'unpaired return/batch')
        self.completed[key] += 1
        self.event('returned', self.snapshot())

    def snapshot(self):
        return dict(attempted=dict(self.attempted), completed=dict(self.completed),
            samples_attempted=dict(self.attempted), samples_completed=dict(self.completed),
            frames_attempted={k: v*(1 if k == 'official_encoder' else 3) for k, v in self.attempted.items()},
            frames_completed={k: v*(1 if k == 'official_encoder' else 3) for k, v in self.completed.items()})


class Store:
    def __init__(self, output):
        self.output = Path(output).absolute()
        require(self.output.parent == ROOT / 'results' and self.output.name.startswith(TAG + '_'), 'exclusive results child')
        require(not self.output.exists() and not self.output.is_symlink()
                and not any(p.is_symlink() for p in self.output.parents), 'fresh nonsymlink output')
        self.output.mkdir(exist_ok=False)
        self.started, self.bytes, self.serial, self.previous = time.monotonic(), 0, 0, None

    def guard(self, reserve=0):
        require(time.monotonic() - self.started < WALL, 'wall cap')
        require(self.bytes + reserve <= CAP, 'output cap')
        require(shutil.disk_usage(self.output).free >= FLOOR + reserve, 'free floor')

    def event(self, kind, value):
        payload = dict(sequence=self.serial, previous_sha256=self.previous, kind=kind, value=value)
        self.previous = digest(payload)
        self.serial += 1
        path = self.output / f'event_{self.serial:05d}.json'
        write_new(path, {**payload, 'event_sha256': self.previous})
        self.bytes += path.stat().st_size

    def emit(self, seed, word, step, name, value):
        a = np.asarray(value)
        require(a.dtype == np.float32 and a.size <= max(30, 3*PATCHES*FEATURES, 3*PATCHES*PROP)
                and np.isfinite(a).all(), 'bounded original float32 array')
        self.guard(a.nbytes + 1024)
        path = self.output / f'{seed}_{word}_{step}_{name}.npy'
        with path.open('xb') as f:
            np.save(f, a, allow_pickle=False)
            f.flush()
            os.fsync(f.fileno())
        self.bytes += path.stat().st_size
        return dict(path=str(path), sha256=sha(path), shape=list(a.shape), dtype='float32', bytes=path.stat().st_size)


def cuda_case(models, spec, emit):
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    visual = array(spec['patches'], (4, PATCHES, FEATURES))[:3].copy()
    props = array(spec['proprio'], (3, 4))
    packets = array(spec['packets'], (3, 10))
    words = {k: normalize(raw, spec['stats']) for k, raw in programs().items()}
    require({k: a.tolist() for k, a in words.items()} == {k: v['normalized'] for k, v in spec['plans'].items()}, 'word seal')
    def original(t):
        require(t.is_cuda and t.dtype == torch.float32, 'CUDA float32 return required')
        a = t.detach().cpu().numpy().copy()  # synchronize actual output before recording
        require(np.isfinite(a).all(), 'finite original output')
        return a
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        features = models['proprio_encoder'](torch.from_numpy(props.copy()).to('cuda').unsqueeze(0))
        require(features.shape == (1, 3, 1, 16), 'native proprio feature ABI')
        proprio = original(features.expand(1, 3, 256, 16))[0]
        emit('shared', -1, 'context_proprio_features', proprio)
        def predict(v, a, p):
            result = models['predictor'](
                torch.from_numpy(v.copy()).to('cuda').reshape(1, 3, 1, 16, 16, 384),
                torch.from_numpy(a.copy()).to('cuda').unsqueeze(0),
                torch.from_numpy(p.copy()).to('cuda').unsqueeze(0))
            require(len(result) == 3 and result[1] is None and result[0].shape == (1, 3, 256, 384)
                    and result[2].shape == (1, 3, 256, 16), 'native AdaLN returned visual/proprio ABI')
            return original(result[0])[0], original(result[2])[0]
        return contrasts(rollout(predict, visual, proprio, packets[:2], words, emit))


def run_cuda(m, output):
    verify(m)
    require(not os.environ.get('PYTHONPATH'), 'fresh process without physics overlay')
    store = Store(output)
    result = dict(schema=TAG, status='failed', manifest_sha256=digest(m), policy=policy(),
        rows=[dict(seed=s, root_sha256=spec['root_sha256'], status='not_run', artifacts={}, error=None,
                   program_status=dict.fromkeys(WORDS, 'not_run'))
              for s, spec in zip(SEEDS, m['roots'])], errors=[], finalization_errors=[],
        cuda_executed=False, scientific_admission=False, physical_release=False, action_witness_established=False)
    models, handles, current, v4, before = None, [], None, None, None
    old_alarm = signal.getsignal(signal.SIGALRM)
    def timeout(signum, frame):
        raise TimeoutError('600s wall cap; no retry')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(WALL)
    try:
        store.guard()
        import torch
        require(torch.cuda.is_available(), 'CUDA required; no CPU neural fallback')
        free, _ = torch.cuda.mem_get_info()
        require(free >= policy()['scratch_free_cuda_bytes'], '4GiB CUDA free scratch policy')
        loader = importlib.import_module('experiments.spectral_reachability.' + LOADER)
        require(sha(loader.__file__) == LOADER_SHA, 'loader SHA')
        models, strict, sources, before = loader.load_cuda(ROOT, m['loader_manifest'])
        require(before == m['expected_model'], 'same frozen native model')
        _, v5, _ = loader.helper(ROOT).dependencies(ROOT)
        v4 = v5.import_v4(ROOT)
        result.update(cuda_executed=True, model_before=before, strict_loads=strict, imported_sources=sources)
        require(set(models) == set(EXPECTED), 'three frozen modules')
        def pre(key):
            def hook(module, args):
                require(current is not None and args[0].is_cuda, 'unattributed/non-CUDA forward')
                shape = (1, 3, 1, 16, 16, 384) if key == 'predictor' else (1, 3, 4)
                require(tuple(args[0].shape) == shape, 'fixed native context/batch')
                current.before(key, int(args[0].shape[0]))
            return hook
        def post(key):
            def hook(module, args, out):
                current.after(key, int(args[0].shape[0]))
            return hook
        for key, model in models.items():
            handles.extend((model.register_forward_pre_hook(pre(key)), model.register_forward_hook(post(key))))
        for row, spec in zip(result['rows'], m['roots']):
            row['status'] = 'attempted'
            current = Meter(lambda k, v: store.event(k, dict(seed=row['seed'], meter=v)), store.guard)
            try:
                def emit(word, step, name, value):
                    if word in WORDS and step == 0 and name == 'visual_in':
                        row['program_status'][word] = 'attempted'
                    key = f'{word}_{step}_{name}'
                    row['artifacts'][key] = store.emit(row['seed'], word, step, name, value)
                    store.event('artifact', dict(seed=row['seed'], name=key, record=row['artifacts'][key]))
                    if word in WORDS and step == 1 and name == 'proprio_out':
                        row['program_status'][word] = 'completed'
                row['contrasts'] = cuda_case(models, spec, emit)
                require(current.attempted == current.completed == EXPECTED, 'per-root census')
                row['status'] = 'completed'
            except Exception as exc:
                row['status'], row['error'] = 'failed', type(exc).__name__ + ': ' + str(exc)
                for word in WORDS:
                    if row['program_status'][word] == 'attempted':
                        row['program_status'][word] = 'failed'
                raise
            finally:
                row['meter'] = current.snapshot()
                store.event('root_terminal', row)
                current = None
        result['status'] = 'native_action_word_predictions_recorded'
    except Exception as exc:
        result['errors'].append(type(exc).__name__ + ': ' + str(exc))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)
        for handle in handles:
            try:
                handle.remove()
            except Exception as exc:
                result['finalization_errors'].append('hook cleanup: ' + str(exc))
        if models is not None and v4 is not None:
            try:
                result['model_after'] = {k: v4.model_receipt(v) for k, v in models.items()}
                require(result['model_after'] == before == m['expected_model'], 'model state drift')
            except Exception as exc:
                result['finalization_errors'].append('model receipt: ' + str(exc))
        try:
            verify_bindings(m['bindings'])
            store.guard(128*1024)
        except Exception as exc:
            result['finalization_errors'].append('sources/resources: ' + str(exc))
        if result['errors'] or result['finalization_errors']:
            result['status'] = 'failed'
        result.update(wall_seconds=time.monotonic()-store.started, journal_terminal_sha256=store.previous)
        write_new(store.output / 'report.json', result)
    return result


def verify_events(directory, rows, terminal):
    paths = sorted(Path(directory).glob('event_*.json'))
    require(0 < len(paths) <= 1000, 'bounded complete event stream')
    by_seed = {r['seed']: r for r in rows}
    meters = {s: Meter(lambda k, v: None).snapshot() for s in SEEDS}
    roots, artifacts, previous = [], {s: {} for s in SEEDS}, None
    for i, path in enumerate(paths):
        require(path.name == f'event_{i+1:05d}.json', 'event filename gap')
        e = read_json(path)
        require(e['sequence'] == i and e['previous_sha256'] == previous
                and e['event_sha256'] == digest({k: v for k, v in e.items() if k != 'event_sha256'}), 'event hash chain')
        previous = e['event_sha256']
        v, kind = e['value'], e['kind']
        seed = v['seed']
        require(len(roots) < len(SEEDS) and seed in by_seed and seed == SEEDS[len(roots)], 'fixed root order')
        if kind in ('attempted', 'returned'):
            field = 'attempted' if kind == 'attempted' else 'completed'
            old, new = meters[seed], v['meter']
            keys = [k for k in EXPECTED if new[field][k] != old[field][k]]
            require(len(keys) == 1, 'one operation/event')
            key = keys[0]
            replay = Meter(lambda k, v: None)
            replay.attempted, replay.completed = dict(old['attempted']), dict(old['completed'])
            (replay.before if field == 'attempted' else replay.after)(key, 1)
            require(new == replay.snapshot(), 'actual API/sample/frame transition')
            meters[seed] = new
        elif kind == 'artifact':
            require(v['name'] not in artifacts[seed], 'duplicate artifact')
            artifacts[seed][v['name']] = v['record']
        elif kind == 'root_terminal':
            require(v == by_seed[seed] and v['meter'] == meters[seed]
                    and v['artifacts'] == artifacts[seed], 'durable terminal mismatch')
            roots.append(seed)
        else:
            raise ValueError('unknown event kind')
    require(roots == list(SEEDS) and previous == terminal, 'complete durable three-root terminal')


def consume(m, report_path, expected_report_sha):
    verify(m)
    report_path = Path(report_path).absolute()
    r = read_json(report_path, expected_report_sha)
    require(r['schema'] == TAG and r['policy'] == policy() and r['manifest_sha256'] == digest(m), 'contract binding')
    require(r['status'] == 'native_action_word_predictions_recorded' and r['errors'] == []
            and r['finalization_errors'] == [] and r['cuda_executed'] is True
            and r['scientific_admission'] is False and r['physical_release'] is False
            and r['action_witness_established'] is False
            and r['model_before'] == r['model_after'] == m['expected_model'], 'complete frozen CUDA-only run')
    require([row['seed'] for row in r['rows']] == list(SEEDS), 'complete three-root denominator')
    require(math.isfinite(r['wall_seconds']) and 0 <= r['wall_seconds'] < WALL, 'wall receipt')
    require(sum(p.stat().st_size for p in report_path.parent.iterdir() if p.is_file()) <= CAP, 'output cap')
    verify_events(report_path.parent, r['rows'], r['journal_terminal_sha256'])
    for row, spec in zip(r['rows'], m['roots']):
        require(row['status'] == 'completed' and row['error'] is None
                and row['root_sha256'] == spec['root_sha256'], 'root status/identity')
        require(row['program_status'] == dict.fromkeys(WORDS, 'completed'), 'all twelve program statuses')
        meter = Meter(lambda k, v: None)
        meter.attempted, meter.completed = dict(EXPECTED), dict(EXPECTED)
        require(row['meter'] == meter.snapshot(), 'actual operations/sample/frame census')
        records, used = row['artifacts'], set()
        def take(word, step, name, shape):
            key = f'{word}_{step}_{name}'
            rec = records[key]
            require(Path(rec['path']) == report_path.parent / f'{row["seed"]}_{key}.npy', 'exact local array path')
            used.add(key)
            return array(rec, shape)
        visual = array(spec['patches'], (4, PATCHES, FEATURES))[:3].copy()
        proprio = take('shared', -1, 'context_proprio_features', (3, PATCHES, PROP))
        history = array(spec['packets'], (3, 10))[:2]
        words = {k: normalize(raw, spec['stats']) for k, raw in programs().items()}
        cursor = [None, None]
        def capture(word, step, name, expected):
            cursor[:] = word, step
            actual = take(word, step, name, expected.shape)
            require(np.array_equal(actual, expected), 'exact causal context/feedback/action mismatch')
        def predict(v, a, p):
            word, step = cursor
            return (take(word, step, 'visual_out', v.shape),
                    take(word, step, 'proprio_out', p.shape))
        terminal = rollout(predict, visual, proprio, history, words, capture)
        require(set(records) == used and contrasts(terminal) == row['contrasts'], 'complete arrays/contrast recomputation')
    return dict(schema=TAG + '.consume', status='all12_saved_program_predictions_verified',
        report_sha256=sha(report_path), manifest_sha256=digest(m),
        physical_release=False, scientific_admission=False, action_witness_established=False,
        interpretation='saved causal prediction consistency, NOT a physical/action-task witness')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'verify', 'cuda', 'consume'))
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--report', type=Path)
    p.add_argument('--report-sha256')
    args = p.parse_args(argv)
    if args.command == 'prepare':
        result = prepare()
        write_new(args.manifest, result)
    else:
        m = read_json(args.manifest)
        verify(m)
        if args.command == 'verify':
            result = dict(status='verified_non_neural_for_MAIN', scientific_admission=False)
        elif args.command == 'cuda':
            require(args.output is not None, 'explicit new CUDA output')
            result = run_cuda(m, args.output)
        else:
            require(args.report is not None and args.report_sha256 is not None, 'actual report + external SHA')
            result = consume(m, args.report, args.report_sha256)
    print(json.dumps(dict(status=result['status'], scientific_admission=False, physical_release=False)))
    return int(result.get('status') == 'failed')


if __name__ == '__main__':
    raise SystemExit(main())
