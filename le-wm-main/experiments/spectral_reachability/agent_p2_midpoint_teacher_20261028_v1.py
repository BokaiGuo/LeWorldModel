"""Eight actual saved step5 frames -> original frozen encoder/Q, MAIN CUDA only.

prepare/verify/consume: saved NumPy/PIL only. No prediction, physics or fitting.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import sysconfig
import time

import numpy as np
from PIL import Image

from experiments.spectral_reachability import agent_p2_native_composition_gap_20261028_v1 as gap
from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal

NAME = 'agent_p2_midpoint_teacher_20261028_v1'
ROOT, ARMS = gap.ROOT, gap.ARMS
GAP = ROOT / 'results/agent_p2_native_composition_gap_20261028_v1_MAIN01.json'
GAP_SHA = 'fc9d1ae5ff7751821380ea92b1e07c85c374534abb1e7d8cc249d85c9dedd612'
GAP_SOURCE_SHA = '0052fca0c26a84455fa66dc611137ef294af186369f7a3a2a737875ac227f909'
CAP, FLOOR, WALL = 64 * 1024**2, 200 * 1024**2, 900
META = dict(layout='states_with_between_state_actions_v1', context_length=3, latent_dim=192, action_dim=10)
require, sha, digest, encode = gap.require, gap.sha, gap.digest, gap.encode


def regular(path):
    p = Path(path).absolute()
    require('..' not in p.parts and not any(x.is_symlink() for x in (p, *p.parents)) and p.is_file(),
            'regular nonsymlink file required')
    return p


def read(path):
    p = regular(path)
    require(p.stat().st_size < 64 * 1024**2, 'bounded saved JSON')
    def unique(items):
        result = {}
        for k, v in items:
            require(k not in result, 'duplicate JSON key')
            result[k] = v
        return result
    value = json.loads(p.read_bytes(), object_pairs_hook=unique)
    encode(value)  # reject NaN/Infinity
    return value


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write(path, value):
    raw = encode(value) + b'\n'
    require(len(raw) < 16 * 1024**2, 'bounded own metadata')
    with Path(path).open('xb') as stream:
        require(stream.write(raw) == len(raw), 'metadata short write')
        stream.flush()
        os.fsync(stream.fileno())
    sync_dir(Path(path).parent)


def pixels(path, expected):
    require(regular(path).stat().st_size < 1024**2 and sha(path) == expected, 'saved PNG SHA/size')
    with Image.open(path) as image:
        require(image.format == 'PNG' and image.mode == 'RGB' and image.size == (224, 224), 'native RGB PNG')
        rgb = np.asarray(image).copy()
    require(rgb.shape == (224, 224, 3) and rgb.dtype == np.uint8, 'native RGB uint8')
    return rgb, hashlib.sha256(rgb.tobytes()).hexdigest()


def context(c):
    require(all(type(c[k]) is type(v) and c[k] == v for k, v in META.items()), 'full3/2 ABI')
    return gap.array(c['latent_history'], (3, 192)), gap.array(c['action_history'], (2, 10))


def midpoint(plan, result, normalization):
    """An already-executed verification replay, never a fabricated graph edge."""
    win = result['winner']
    require(win is not None and win['independent_full_replay_exact'] is True and
            len(win['segmented_contexts']) == 3, 'missing paid midpoint replay')
    initial, mid, end = win['segmented_contexts']
    require(initial == plan['predictor_context'] and end == win['whole_context'] ==
            plan['selected_predicted_context'] == win['witness']['receipt']['terminal_context'], 'same selected full replay')
    for c in (initial, mid, end):
        context(c)
    raw = gap.array(plan['commands'], (10, 2))
    require(np.array_equal(raw, raw.astype(np.float32)), 'exact raw float32 commands')
    mean, std = (gap.array(normalization[k], (2,)).astype(np.float32) for k in ('mean', 'std'))
    require(np.all(std > 0), 'original positive action std')
    tokens = ((raw.astype(np.float32) - mean) / std).reshape(2, 10).tolist()
    require(mid['latent_history'][:2] == initial['latent_history'][1:] and
            end['latent_history'][:2] == mid['latent_history'][1:] and
            mid['action_history'] == initial['action_history'][1:] + [tokens[0]] and
            end['action_history'] == tokens and mid['latent_history'][-1] == end['latent_history'][1],
            'step5 exact shifted states/tokens, not terminal substitution')
    if plan['method'] == ARMS[1]:
        require(mid == win['witness']['edges'][0]['output_context'] and
                win['witness']['edges'][0]['actions'] == plan['commands'][:5], 'conditional actual prefix edge')
    n = 80 if plan['method'] == ARMS[0] else 40
    replay = result['ledger'][-4:]
    expected = [([n], 0), ([n+1], 0), ([n+2], 0), ([n+2], 1)]
    require(len(replay) == 4 and all(e['phase'] == 'replay' and e['path'] == path and e['packet'] == packet and
            e['completed'] is True and e['predict_calls'] == e['predict_samples'] == e['action_encoder_calls'] == 1
            for e, (path, packet) in zip(replay, expected)), 'paid midpoint/full replay entry receipts')
    return dict(predicted_vector=deepcopy(mid['latent_history'][-1]), predicted_context=deepcopy(mid),
                predicted_context_sha256=digest(mid), predicted_history_index=1,
                origin='already_paid_selected_segmented_verification_replay',
                root_context_sha256=digest(initial), full_context_sha256=digest(end))


def contract():
    return dict(roots=4, arms=list(ARMS), frames=8, primitive_step=5, execution_length=10,
                selection='all root-major canonical arm order, no outcome filtering or replanning',
                encoder_cap=8, existing_projector_cap=8, per_frame_each_cap=1, batch=1, time=1,
                predict_cap=0, action_encoder_cap=0, pred_proj_cap=0, AD_cap=0, physics_cap=0,
                device='cuda:0', CPU_neural_fallback=False, retries=False, training=False,
                transform='original p0._image_tensor + public image_transform; encode pixels only',
                representation='native encoder last_hidden_state CLS -> original projector -> native192 float32',
                precision=precision_contract(),
                Q='same saved source_Q/goal0; no refit, no alternate objective',
                output_cap_bytes=CAP, floor_bytes=FLOOR, wall_seconds=WALL,
                wall_scope='checked between public operations and persistence; not a preemptive watchdog',
                P1_admission=False, P2_admission=False, scientific_admission=False)


def precision_contract():
    return dict(sdpa_math=True, sdpa_flash=False, sdpa_mem_efficient=False, sdpa_cudnn=False,
                matmul_tf32=False, cudnn_tf32=False, autocast_cuda_enabled=False)


def checked_precision(torch):
    cuda = torch.backends.cuda
    state = dict(sdpa_math=cuda.math_sdp_enabled(), sdpa_flash=cuda.flash_sdp_enabled(),
                 sdpa_mem_efficient=cuda.mem_efficient_sdp_enabled(), sdpa_cudnn=cuda.cudnn_sdp_enabled(),
                 matmul_tf32=cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
                 autocast_cuda_enabled=torch.is_autocast_enabled('cuda'))
    require(state == precision_contract(), 'MATH-only/TF32-off/autocast-off precision guard')
    return state


def encoder_abi(config):
    encoder, projector = config['encoder'], config['projector']
    def pair(value):
        if type(value) is int:
            value = [value, value]
        require(type(value) is list and len(value) == 2 and all(type(n) is int and n > 0 for n in value),
                'explicit positive config image/patch dimensions')
        return value
    image, patch = pair(encoder['image_size']), pair(encoder['patch_size'])
    require(image == [224, 224] and all(a % b == 0 for a, b in zip(image, patch)) and
            projector['input_dim'] == projector['output_dim'] == 192, 'original image/patch/projector ABI')
    return dict(image_shape=image, patch_shape=patch, tokens=1 + (image[0] // patch[0]) * (image[1] // patch[1]),
                encoder_dim=projector['input_dim'], native_dim=projector['output_dim'], cls_index=0)


def image_statistics():
    """Read the ORIGINAL public ImageNet constants without importing torch/spt."""
    path = Path(sysconfig.get_paths()['purelib']) / 'stable_pretraining/data/dataset_stats.py'
    nodes = [n.value for n in ast.parse(regular(path).read_text()).body if isinstance(n, ast.Assign) and
             any(isinstance(t, ast.Name) and t.id == 'ImageNet' for t in n.targets)]
    require(len(nodes) == 1 and isinstance(nodes[0], ast.Call) and isinstance(nodes[0].func, ast.Name) and
            nodes[0].func.id == 'dict' and not nodes[0].args, 'original explicit public ImageNet constants')
    stats = {kw.arg: ast.literal_eval(kw.value) for kw in nodes[0].keywords}
    require(set(stats) == {'mean', 'std'} and np.all(gap.array(stats['std'], (3,)) > 0), 'original image mean/std')
    gap.array(stats['mean'], (3,))
    return stats


def normalized_rgb(rgb, stats):
    """Saved arithmetic only: native224 Resize is identity, no torch/no neural."""
    require(rgb.dtype == np.uint8 and rgb.shape == (224, 224, 3), 'native224 no resize needed')
    mean, std = (gap.array(stats[k], (3,)).astype(np.float32)[:, None, None] for k in ('mean', 'std'))
    require(np.all(std > 0), 'original positive image std')
    # torchvision ToDtype(scale=True) does .to(float32).mul_(1/255), then
    # Normalize does sub(mean), div_(std), separately; no fused arithmetic.
    value = rgb.transpose(2, 0, 1).astype(np.float32) * np.float32(1.0 / 255)
    return ((value - mean) / std)[None, None]


def encoding_inventory(frames):
    """Known P2 endpoint receipt convention, not a claim to scan arbitrary arrays."""
    targets = {x['rgb_file_sha256'] for x in frames}
    inventory, matches = {}, []
    for path in sorted((ROOT / 'results').rglob('*.encoding.json')):
        if 'p2' not in str(path) or NAME in str(path):
            continue
        obj = read(path)
        inventory[str(path.resolve())] = sha(path)
        if obj.get('terminal_rgb_sha256', obj.get('rgb_file_sha256')) in targets:
            matches.append(str(path.resolve()))
    require(not matches, 'step5 already encoded; do not duplicate: ' + str(matches))
    return dict(scope='P2 *.encoding.json frame-SHA search; no matching new step5 receipts',
                files_sha256=inventory, matched_existing_step5=matches)


def collect():
    require(sha(gap.__file__) == GAP_SOURCE_SHA and sha(GAP) == GAP_SHA, 'frozen MAIN gap receipt/source')
    g = read(GAP)
    require(g == gap.load_saved(), 'reconstructed all4/8 gap evidence')
    prepared = read(ROOT / gap.INPUTS['prepare'][0])
    physical = read(ROOT / gap.INPUTS['physical'][0])
    protocol_path = ROOT / 'results/p2_composition_predict_frontier_cuda_20261027_v2_prepared01/protocol.json'
    old = read(protocol_path)
    bindings = {**g['provenance']['consumed_sha256'], str(GAP): GAP_SHA, str(Path(gap.__file__).resolve()): GAP_SOURCE_SHA}
    frames = []
    pindex, bindex = gap.census(prepared['plans']), gap.census(physical['rows'])
    for i in range(4):
        root_path = ROOT / f'results/agent_p2_runtime_guard_20261028_v1_known_MAIN01/root_{i}.json'
        root = read(root_path)
        for arm in ARMS:
            handoff, branch = pindex[i, arm], bindex[i, arm]['physical']
            plan = handoff['execution_plan']
            witness = midpoint(plan, root['results'][ARMS.index(arm)], old['context']['normalization'])
            path = Path(branch['journal_path'])
            require(sha(path) == branch['journal_sha256'], 'original per-arm journal')
            events = [json.loads(line) for line in path.read_bytes().splitlines()]
            obs = [e for e in events if e['kind'] == 'observation']
            require(len(obs) == 10 and [x['primitive_step'] for x in obs] == list(range(1, 11)) and
                    [{k: v for k, v in x.items() if k not in ('rgb_path', 'rgb_file_sha256')} for x in obs] ==
                    branch['trace'], 'actual frame/state/action journal closure')
            e = obs[4]
            rgbpath = (ROOT / gap.INPUTS['physical'][0]).parent / f'root_{i}_{arm}_rgb_5.png'
            require(e['rgb_path'] == str(rgbpath) and e['command'] == plan['commands'][4], 'exact step5 actual PNG')
            _, pixel = pixels(rgbpath, e['rgb_file_sha256'])
            require(bindings[str(rgbpath)] == e['rgb_file_sha256'], 'report artifact SHA agrees with journal')
            frame = dict(frame_id=f'root_{i}_{arm}_step5', root_slot=i, root_sha256=plan['root_sha256'], arm=arm,
                         primitive_step=5, rgb_path=str(rgbpath), rgb_file_sha256=e['rgb_file_sha256'], pixel_sha256=pixel,
                         actual_state=e['state'], actual_goal_state=branch['context_identity']['goal'],
                         actual_initial_state=branch['context_identity']['state'],
                         actual_history_rgb_sha256=branch['context_identity']['history_rgb_sha256'],
                         actual_goal_rgb_sha256=branch['context_identity']['goal_rgb_sha256'],
                         raw_actions=plan['commands'], executed_prefix_actions=plan['commands'][:5],
                         selection_digest=handoff['source_selection_digest'], handoff_sha256=handoff['handoff_sha256'],
                         goal_vector=plan['goal'], active_metric_id=plan['active_metric_id'], **witness)
            frame['frame_binding_sha256'] = digest(frame)
            frames.append(frame)
    inventory = encoding_inventory(frames)
    models = prepared['model_bindings']
    for path, h in models.items():
        require(sha(regular(path)) == h, 'original frozen weights/config')
    policy = Path(next(path for path in models if Path(path).name == 'weights.pt')).parent
    config = read(policy / 'config.json')
    require(config['_target_'] == 'stable_worldmodel.wm.lewm.LeWM' and config['encoder']['pretrained'] is False and
            config['projector']['_target_'] == 'stable_worldmodel.wm.lewm.module.MLP', 'local-only original architecture')
    historical = {}
    for rel in ('experiments/checkpoint_compat.py', 'experiments/run_pusht_counterfactual.py',
                'experiments/spectral_reachability/preflight_pusht_d2p_p0.py',
                'experiments/spectral_reachability/p2_composition_predict_frontier_cuda_20261027_v2.py',
                'experiments/spectral_reachability/p2_matched_composition_20261027.py',
                'experiments/spectral_reachability/pusht_composition_cuda_replay_v2.py'):
        path = str(ROOT / rel)
        h = prepared['bindings'][path]
        require(sha(path) == h, 'historical loader/transform drift')
        historical[path] = h
    return dict(schema=NAME + '.protocol', contract=contract(), frames=frames,
                policy=str(policy), model_bindings=models, historical_sources=historical,
                bindings=bindings, existing_encoding_inventory=inventory, gap_report_sha256=GAP_SHA,
                image_statistics=image_statistics(), encoder_abi=encoder_abi(config),
                unique_step5_image_count=len({f['rgb_file_sha256'] for f in frames}),
                normalization=old['context']['normalization'],
                compilation_path=old['context']['compilation'],
                compilation_sha256=bindings[old['context']['compilation']],
                inherited_resources=g['provenance']['inherited_resources'],
                inherited_terminal_encoder_samples=8, new_independent_roots=0)


def source_files():
    site = Path(sysconfig.get_paths()['purelib'])
    rel = ['stable_worldmodel/wm/lewm/lewm.py', 'stable_worldmodel/wm/lewm/module.py',
           'stable_pretraining/backbone/utils.py', 'stable_pretraining/data/dataset_stats.py',
           'transformers/models/vit/modeling_vit.py', 'transformers/models/vit/configuration_vit.py',
           'torchvision/transforms/v2/_container.py', 'torchvision/transforms/v2/_misc.py',
           'torchvision/transforms/v2/_geometry.py', 'torchvision/transforms/v2/functional/_misc.py',
           'torchvision/transforms/v2/functional/_geometry.py', 'torchvision/transforms/v2/functional/_type_conversion.py',
           'torch/nn/attention/__init__.py', 'torch/backends/cuda/__init__.py']
    local = [Path(__file__), Path(gap.__file__), ROOT / f'tests/test_{NAME}.py', ROOT / f'innerdocs/{NAME}.md']
    local += [ROOT / x for x in ('experiments/checkpoint_compat.py', 'experiments/gpu_preflight.py',
              'experiments/run_pusht_counterfactual.py', 'experiments/spectral_reachability/preflight_pusht_d2p_p0.py',
              'experiments/spectral_reachability/d2p_prerun_source_seal.py')]
    return sorted({p.resolve() for p in local + [site / r for r in rel]})


def prepare(output):
    output = Path(output).absolute()
    require(not output.exists() and not output.is_symlink() and shutil.disk_usage(output.parent).free > FLOOR + CAP,
            'NEW prepare with headroom')
    p = collect()
    p['sources_sha256'] = {str(f): sha(regular(f)) for f in source_files()}
    output.mkdir(exist_ok=False)
    write(output / 'protocol.json', p)
    seal = create_seal(files=source_files(), args_json=encode(dict(protocol_sha256=sha(output / 'protocol.json'))).decode(),
                       output_dir=output / 'source_seal')
    receipt = dict(schema=NAME + '.prepare', status='prepared_not_executed', protocol_sha256=sha(output / 'protocol.json'),
                   source_seal=seal, new_neural_calls=0, new_physics_steps=0, scientific_admission=False)
    write(output / 'prepare.json', receipt)
    return dict(prepared=str(output), prepare_sha256=sha(output / 'prepare.json'), **receipt)


def verify(prepared, expected):
    prepared = Path(prepared).absolute()
    require(sha(regular(prepared / 'prepare.json')) == expected, 'external prepare SHA')
    receipt, p = read(prepared / 'prepare.json'), read(prepared / 'protocol.json')
    require(receipt['schema'] == NAME + '.prepare' and receipt['status'] == 'prepared_not_executed' and
            sha(prepared / 'protocol.json') == receipt['protocol_sha256'] and p['contract'] == contract(), 'own prepare contract')
    s = receipt['source_seal']
    verify_seal(output_dir=s['output_dir'], expected_manifest_sha256=s['manifest_sha256'])
    for path, h in {**p['bindings'], **p['sources_sha256'], **p['model_bindings'], **p['historical_sources'],
                    **p['existing_encoding_inventory']['files_sha256']}.items():
        require(sha(regular(path)) == h, 'input/source/model drift: ' + path)
    rebuilt = collect()
    rebuilt['sources_sha256'] = p['sources_sha256']
    require(rebuilt == p, 'all8 midpoint/frame/context reconstruction')
    return p


def used(folder):
    return sum(p.stat().st_size for p in Path(folder).rglob('*') if p.is_file())


def boundary(folder, start):
    require(time.monotonic() - start < WALL, '900s operation-boundary wall')
    require(used(folder) < CAP - 4 * 1024**2 and shutil.disk_usage(folder).free > FLOOR + 4 * 1024**2,
            '64MiB output /200MiB floor')


def save_array(folder, name, value):
    value = np.asarray(value)
    require(value.dtype in (np.dtype('float32'), np.dtype('uint8')) and np.isfinite(value).all(), 'native array dtype/finite')
    path = Path(folder) / (name + '.npy')
    with path.open('xb') as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    sync_dir(path.parent)
    return dict(file=path.name, sha256=sha(path), dtype=str(value.dtype), shape=list(value.shape))


class Journal:
    def __init__(self, stream):
        self.stream, self.index, self.previous = stream, 0, None

    def __call__(self, value):
        event = dict(index=self.index, previous_sha256=self.previous, payload=deepcopy(value))
        event['event_sha256'] = digest(event)
        raw = encode(event) + b'\n'
        require(self.stream.tell() + len(raw) < 1024**2, 'bounded journal')
        self.stream.write(raw.decode())
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.index += 1
        self.previous = event['event_sha256']


def failure(stage, exc):
    return dict(stage=stage, type=type(exc).__name__, message=str(exc)[:1000])


class Ledger:
    """Fixed8 nonrefundable dispatch; RAM adapters need no real torch tensor."""
    def __init__(self, frames, journal, persist, synchronize=lambda: None, check=lambda: None, precision=lambda: None):
        self.order = [(x['frame_id'], k) for x in frames for k in ('encoder', 'existing_projector')]
        require(len(frames) == 8 and len(set(self.order)) == 16, 'fixed8 ledger')
        self.journal, self.persist, self.synchronize, self.check = journal, persist, synchronize, check
        self.precision = precision
        self.active, self.entries, self.blocked, self.journal_errors = None, [], [], []

    def emit(self, event):
        try:
            self.journal(event)
        except BaseException as exc:
            self.journal_errors.append(failure('journal', exc))
            raise

    def dispatch(self, kind, tensor, callback, args, kwargs):
        try:
            self.check()
            require(getattr(getattr(tensor, 'device', None), 'type', None) == 'cuda', 'no CPU neural entry')
            require(kind in ('encoder', 'existing_projector') and tuple(tensor.shape) ==
                    ((1, 3, 224, 224) if kind == 'encoder' else (1, 192)), 'single native module input')
            require(len(self.entries) < 16 and (self.active, kind) == self.order[len(self.entries)] and
                    all(x['saved'] for x in self.entries) and not self.journal_errors and not self.blocked,
                    'fixed order/cap/no retry')
        except BaseException as exc:
            self.blocked.append(failure('predispatch', exc))
            raise
        entry = dict(index=len(self.entries), frame_id=self.active, kind=kind, native_entered=False,
                     native_returned=False, saved=False, arrays={}, wall_seconds=None, precision=None)
        self.entries.append(entry)
        primary, started = None, None
        try:
            self.emit(dict(event='module_attempt', index=entry['index'], frame_id=self.active, kind=kind))
            entry['arrays']['input'] = self.persist(self.active + '.' + kind + '.input', tensor)
            self.check()
            entry['precision'] = self.precision()
            self.emit(dict(event='module_enter', index=entry['index'], frame_id=self.active, kind=kind))
            entry['native_entered'] = True
            started = time.monotonic()
            result = callback(*args, **kwargs)
            entry['native_returned'] = True
            self.synchronize()
            entry['wall_seconds'] = time.monotonic() - started
            output = result.last_hidden_state if kind == 'encoder' else result
            entry['arrays']['output'] = self.persist(self.active + '.' + kind + '.output', output)
            entry['saved'] = True
            return result
        except BaseException as exc:
            primary = exc
            entry['error'] = failure('module', exc)
            if started is not None and entry['wall_seconds'] is None:
                entry['wall_seconds'] = time.monotonic() - started
            raise
        finally:
            try:
                self.emit(dict(event='module_return_or_error', entry=deepcopy(entry)))
            except BaseException:
                if primary is None:
                    raise

    def forbid(self, kind):
        def denied(*args, **kwargs):
            error = failure('forbidden', RuntimeError(kind))
            self.blocked.append(error)
            self.emit(dict(event='forbidden', frame_id=self.active, kind=kind))
            raise RuntimeError('forbidden ' + kind)
        return denied

    def counts(self):
        return {k: dict(attempted=sum(e['kind'] == k for e in self.entries),
                        entered=sum(e['kind'] == k and e['native_entered'] for e in self.entries),
                        returned=sum(e['kind'] == k and e['native_returned'] for e in self.entries),
                        saved=sum(e['kind'] == k and e['saved'] for e in self.entries),
                        wall_seconds=None if any(e['native_entered'] and e['wall_seconds'] is None for e in self.entries if e['kind'] == k)
                        else sum(e['wall_seconds'] for e in self.entries if e['kind'] == k and e['native_entered']))
                for k in ('encoder', 'existing_projector')}


class Hooks:
    def __init__(self):
        self.old = []

    def install(self, obj, name, value):
        self.old.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def restore(self, errors):
        for obj, name, value in reversed(self.old):
            try:
                setattr(obj, name, value)
            except BaseException as exc:
                errors.append(failure('hook_restore', exc))


def model_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        a = value.detach().cpu().contiguous()
        h.update(encode([name, str(a.dtype), list(a.shape)]))
        h.update(a.numpy().tobytes())
    return h.hexdigest()


def frozen(model):
    require(all(not m.training for m in model.modules()) and
            all(p.device.type == 'cuda' and not p.requires_grad for p in model.parameters()) and
            all(b.device.type == 'cuda' for b in model.buffers()), 'whole model frozen CUDA eval')


def encode_frames(p, folder, adapter, ledger, journal, rows, errors, check=lambda: None):
    """RAM-testable orchestration. No fallback or retries; always retain all8 rows."""
    for f in p['frames']:
        row = dict(frame_id=f['frame_id'], frame_binding_sha256=f['frame_binding_sha256'],
                   status='blocked_not_attempted', arrays={})
        if not errors:
            ledger.active = f['frame_id']
            row['status'] = 'attempted'
            try:
                check()
                journal(dict(event='frame_attempt', frame_id=f['frame_id'], frame_binding_sha256=f['frame_binding_sha256']))
                arrays = adapter(f)
                require(set(arrays) == {'rgb', 'normalized_pixels', 'native192'}, 'complete returned input/latent arrays')
                for name, value in arrays.items():
                    row['arrays'][name] = save_array(folder, f['frame_id'] + '.' + name, value)
                require(row['arrays']['native192']['shape'] == [192] and
                        row['arrays']['native192']['dtype'] == 'float32', 'native192 output')
                row['status'] = 'encoded_saved'
                journal(dict(event='frame_saved', frame_id=f['frame_id'], arrays=deepcopy(row['arrays'])))
            except BaseException as exc:
                errors.append(failure('frame.' + f['frame_id'], exc))
                row['status'], row['error'] = 'failed', errors[-1]
            finally:
                ledger.active = None
        rows.append(row)
        try:
            write(Path(folder) / (f['frame_id'] + '.row.json'), row)
        except BaseException as exc:
            errors.append(failure('row_persistence', exc))


def execute_cuda(prepared, expected, output):
    """MAIN-authorized explicit HOST CUDA; never called by saved tests."""
    p = verify(prepared, expected)
    output = Path(output).absolute()
    require(not output.exists() and not output.is_symlink() and shutil.disk_usage(output.parent).free > FLOOR + CAP,
            'exclusive NEW run with reserve')
    # NEW sibling claim, never mutate a frozen preparation.
    write(Path(prepared).absolute().with_name(Path(prepared).name + '.launch.json'),
          dict(output=str(output), prepare_sha256=expected))
    output.mkdir(exist_ok=False)
    write(output / 'execution_binding.json', dict(prepared=str(Path(prepared).absolute()), prepare_sha256=expected,
                                                gap_report_sha256=GAP_SHA))
    rows, errors, host, model, before, after = [], [], None, None, None, None
    source_ok, hooks, start = False, Hooks(), time.monotonic()
    contexts, precision_before, precision_after, original_tf32 = ExitStack(), None, None, None
    with (output / 'events.jsonl').open('x') as stream:
        journal = Journal(stream)
        def persist(name, tensor):
            require(tensor.device.type == 'cuda' and str(tensor.dtype) == 'torch.float32', 'native CUDA float32 buffer')
            expected_shape = ((1, p['encoder_abi']['tokens'], p['encoder_abi']['encoder_dim'])
                              if name.endswith('.encoder.output') else
                              (1, 3, 224, 224) if name.endswith('.encoder.input') else (1, 192))
            require(tuple(tensor.shape) == expected_shape, 'bound config module input/output shape')
            return save_array(output, name, tensor.detach().cpu().numpy().copy())
        ledger = Ledger(p['frames'], journal, persist, check=lambda: boundary(output, start))
        try:
            import torch
            from experiments.checkpoint_compat import load_compatible_pretrained
            from experiments.gpu_preflight import require_cuda_allocation
            from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
            from torch.nn.attention import sdpa_kernel, SDPBackend
            host = require_cuda_allocation('cuda:0')
            device = torch.device('cuda:0')
            original_tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            contexts.enter_context(sdpa_kernel(SDPBackend.MATH))
            contexts.enter_context(torch.autocast(device_type='cuda', enabled=False))
            precision_before = checked_precision(torch)
            ledger.precision = lambda: checked_precision(torch)
            model = load_compatible_pretrained(p['policy'], device=device).eval().requires_grad_(False)
            frozen(model)
            before = model_hash(model)
            ledger.synchronize = lambda: torch.cuda.synchronize(device)
            transform = p0.image_transform()
            for kind, module in (('encoder', model.encoder), ('existing_projector', model.projector)):
                original = module.forward
                def counted(*args, _kind=kind, _original=original, **kwargs):
                    return ledger.dispatch(_kind, args[0], _original, args, kwargs)
                hooks.install(module, 'forward', counted)
            for obj, key, kind in ((model, 'predict', 'predict'), (model.predictor, 'forward', 'predictor'),
                                   (model.action_encoder, 'forward', 'action_encoder'), (model.pred_proj, 'forward', 'pred_proj'),
                                   (torch.autograd, 'grad', 'AD.grad'), (torch.autograd, 'backward', 'AD.backward'),
                                   (torch.Tensor, 'backward', 'AD.Tensor.backward')):
                hooks.install(obj, key, ledger.forbid(kind))
            for key in ('jacobian', 'hessian', 'jvp', 'vjp', 'hvp', 'vhp'):
                hooks.install(torch.autograd.functional, key, ledger.forbid('AD.' + key))
            for key in ('jvp', 'vjp', 'jacrev', 'jacfwd', 'hessian', 'grad', 'grad_and_value'):
                hooks.install(torch.func, key, ledger.forbid('AD.func.' + key))
            def adapter(f):
                rgb, pixel = pixels(f['rgb_path'], f['rgb_file_sha256'])
                require(pixel == f['pixel_sha256'], 'original frame pixel SHA')
                with torch.inference_mode():
                    x = p0._image_tensor(rgb, transform=transform, device=device)
                    require(x.device.type == 'cuda' and x.dtype == torch.float32 and tuple(x.shape) == (1, 1, 3, 224, 224),
                            'original single-frame normalized CUDA input')
                    z = model.encode({'pixels': x})['emb']
                    require(z.device.type == 'cuda' and z.dtype == torch.float32 and tuple(z.shape) == (1, 1, 192) and
                            bool(torch.isfinite(z).all()), 'native single192 CUDA encoding')
                    require([(e['kind'], e['saved']) for e in ledger.entries if e['frame_id'] == f['frame_id']] ==
                            [('encoder', True), ('existing_projector', True)], 'one encoder/projector per frame')
                    return dict(rgb=rgb, normalized_pixels=x.detach().cpu().numpy().copy(),
                                native192=z[0, 0].detach().cpu().numpy().copy())
            encode_frames(p, output, adapter, ledger, journal, rows, errors, lambda: boundary(output, start))
            torch.cuda.synchronize(device)
        except BaseException as exc:
            errors.append(failure('setup_or_execution', exc))
        finally:
            hooks.restore(errors)
            if model is not None:
                try:
                    frozen(model)
                    after = model_hash(model)
                except BaseException as exc:
                    errors.append(failure('final_model_hash', exc))
            if precision_before is not None:
                try:
                    precision_after = checked_precision(torch)
                except BaseException as exc:
                    errors.append(failure('final_precision', exc))
            try:
                contexts.close()
                if original_tf32 is not None:
                    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = original_tf32
            except BaseException as exc:
                errors.append(failure('restore_precision', exc))
            try:
                verify(prepared, expected)
                boundary(output, start)
                source_ok = True
            except BaseException as exc:
                errors.append(failure('final_source_wall_budget', exc))
            if len(rows) < 8:
                encode_frames(dict(frames=p['frames'][len(rows):]), output, None, ledger, journal, rows,
                              errors or [dict(stage='setup', message='no encoder')])
            good_model = before is not None and before == after
            if not good_model:
                errors.append(dict(stage='model_integrity', message='missing or changed model state hash'))
            complete = (not errors and not ledger.blocked and not ledger.journal_errors and good_model and source_ok and
                        precision_before == precision_after == precision_contract() and
                        len(rows) == 8 and all(r['status'] == 'encoded_saved' for r in rows) and len(ledger.entries) == 16)
            if not complete and not errors:
                errors.append(dict(stage='census', message='incomplete frame/module denominator'))
            report = dict(schema=NAME + '.report', status='all8_step5_encoded' if complete else 'partial_or_failed',
                          prepare_sha256=expected, gap_report_sha256=GAP_SHA, contract=contract(), rows=rows,
                          counts=ledger.counts(), entries=ledger.entries, blocked=ledger.blocked,
                          journal_errors=ledger.journal_errors, errors=errors, first_error=errors[0] if errors else None,
                          journal_sha256=sha(output / 'events.jsonl'), journal_tail_sha256=journal.previous,
                          model_hash_before=before, model_hash_after=after, model_unchanged=good_model,
                          model_bindings=p['model_bindings'], source_verified=source_ok, host=host,
                          precision_before=precision_before, precision_after=precision_after,
                          total_wall_seconds=time.monotonic() - start, CUDA_ns=None, FLOPs=None,
                          new_predict_calls=0, new_AD_calls=0, new_physics_steps=0, P1_admission=False,
                          P2_admission=False, scientific_admission=False, equal_full_resource_cost=False)
            write(output / 'report.json', report)
    if not complete:
        raise RuntimeError('step5 encoding failed/partial; full8 report preserved, no retry')
    return dict(status=report['status'], report=str(output / 'report.json'), report_sha256=sha(output / 'report.json'))


def load_array(folder, receipt):
    path = Path(folder) / receipt['file']
    require(path.parent == Path(folder) and path.name == receipt['file'] and sha(regular(path)) == receipt['sha256'],
            'own array path/SHA')
    require(path.stat().st_size < 2 * 1024**2, 'single native buffer cap')
    value = np.load(path, allow_pickle=False)
    require(value.dtype in (np.dtype('float32'), np.dtype('uint8')) and np.isfinite(value).all() and
            str(value.dtype) == receipt['dtype'] and list(value.shape) == receipt['shape'], 'native saved array receipt')
    return value


def numeric(actual, predicted, goal, q):
    a, z, g = (gap.array(x.tolist() if isinstance(x, np.ndarray) else x, (192,)) for x in (actual, predicted, goal))
    q = gap.array(q, (192, 192))
    return dict(teacher_Q=float(np.mean(((a-g) @ q.T)**2)), predicted_Q=float(np.mean(((z-g) @ q.T)**2)),
                latent_prediction_mse=float(np.mean((a-z)**2)), projected_prediction_mse=float(np.mean(((a-z) @ q.T)**2)))


def validate_run(folder, r, p):
    """Pure RAM/saved strict closure, testable without model setup."""
    folder = Path(folder).absolute()
    require(r['schema'] == NAME + '.report' and r['status'] == 'all8_step5_encoded' and r['contract'] == contract() and
            r['errors'] == r['blocked'] == r['journal_errors'] == [] and r['first_error'] is None and
            r['model_hash_before'] is not None and r['model_hash_before'] == r['model_hash_after'] and
            r['model_unchanged'] is r['source_verified'] is True and r['host']['device'] == 'cuda:0' and
            r['model_bindings'] == p['model_bindings'], 'complete actual CUDA source/model receipt')
    require(r['precision_before'] == r['precision_after'] == precision_contract(), 'sealed precision before/after')
    require(r['new_predict_calls'] == r['new_AD_calls'] == r['new_physics_steps'] == 0 and
            r['P1_admission'] is r['P2_admission'] is r['scientific_admission'] is False and
            len(r['rows']) == len(p['frames']) == 8 and len(r['entries']) == 16, 'fixed8 scope/counts')
    require(type(r['total_wall_seconds']) in (int, float) and 0 <= r['total_wall_seconds'] < WALL, 'actual wall cap')
    journal_path = folder / 'events.jsonl'
    require(sha(journal_path) == r['journal_sha256'], 'journal file SHA')
    previous, payloads = None, []
    for i, line in enumerate(journal_path.read_bytes().splitlines()):
        event = json.loads(line)
        h = event.pop('event_sha256')
        require(event['index'] == i and event['previous_sha256'] == previous and digest(event) == h, 'journal chain')
        previous = h
        payloads.append(event['payload'])
    require(previous == r['journal_tail_sha256'], 'journal tail')
    expected, arrays_by_frame = [], []
    for index, (f, row) in enumerate(zip(p['frames'], r['rows'], strict=True)):
        require(digest({k: v for k, v in f.items() if k != 'frame_binding_sha256'}) == f['frame_binding_sha256'],
                'full original frame binding digest')
        require(row['frame_id'] == f['frame_id'] and row['frame_binding_sha256'] == f['frame_binding_sha256'] and
                row['status'] == 'encoded_saved' and read(folder / (f['frame_id'] + '.row.json')) == row and
                set(row['arrays']) == {'rgb', 'normalized_pixels', 'native192'}, 'ordered complete frame/row')
        expected.append(dict(event='frame_attempt', frame_id=f['frame_id'], frame_binding_sha256=f['frame_binding_sha256']))
        native = {}
        for n, kind in enumerate(('encoder', 'existing_projector')):
            entry = r['entries'][2*index+n]
            require(entry['index'] == 2*index+n and entry['frame_id'] == f['frame_id'] and entry['kind'] == kind and
                    entry['native_entered'] is entry['native_returned'] is entry['saved'] is True and
                    type(entry['wall_seconds']) in (int, float) and 0 <= entry['wall_seconds'] < WALL and
                    entry['precision'] == precision_contract() and
                    set(entry['arrays']) == {'input', 'output'}, 'ordered paid module entries')
            for event in ('module_attempt', 'module_enter'):
                expected.append(dict(event=event, index=entry['index'], frame_id=f['frame_id'], kind=kind))
            expected.append(dict(event='module_return_or_error', entry=entry))
            for k, value in entry['arrays'].items():
                require(value['file'] == f['frame_id'] + '.' + kind + '.' + k + '.npy', 'exact module array file')
                native[kind + '_' + k] = load_array(folder, value)
        for key, value in row['arrays'].items():
            require(value['file'] == f['frame_id'] + '.' + key + '.npy', 'exact frame array file')
            native[key] = load_array(folder, value)
        rgb, pixel = pixels(f['rgb_path'], f['rgb_file_sha256'])
        require(pixel == f['pixel_sha256'] and np.array_equal(native['rgb'], rgb), 'actual original frame buffer')
        x, ei, eo, pi, po, z = (native[k] for k in ('normalized_pixels', 'encoder_input', 'encoder_output',
                                                   'existing_projector_input', 'existing_projector_output', 'native192'))
        require(x.dtype == ei.dtype == eo.dtype == pi.dtype == po.dtype == z.dtype == np.float32 and
                x.shape == (1, 1, 3, 224, 224) and ei.shape == (1, 3, 224, 224) and
                eo.shape == (1, p['encoder_abi']['tokens'], p['encoder_abi']['encoder_dim']) and
                pi.shape == po.shape == (1, 192) and z.shape == (192,) and
                np.array_equal(x[0], ei) and np.array_equal(eo[:, 0], pi) and np.array_equal(po[0], z),
                'native pixels->encoder CLS->original projector->latent buffer closure')
        require(np.array_equal(x, normalized_rgb(rgb, p['image_statistics'])), 'original RGB->public normalization arithmetic')
        expected.append(dict(event='frame_saved', frame_id=f['frame_id'], arrays=row['arrays']))
        arrays_by_frame.append(native)
    require(payloads == expected, 'exact attempt/entry/return/full frame journal')
    for kind in ('encoder', 'existing_projector'):
        entries = [x for x in r['entries'] if x['kind'] == kind]
        require(r['counts'][kind] == dict(attempted=8, entered=8, returned=8, saved=8,
                                         wall_seconds=sum(x['wall_seconds'] for x in entries)), 'actual per-axis counts/wall')
    return arrays_by_frame


def consume(folder, expected, output):
    folder = Path(folder).absolute()
    require(sha(regular(folder / 'report.json')) == expected, 'external run report SHA')
    r, launch = read(folder / 'report.json'), read(folder / 'execution_binding.json')
    p = verify(launch['prepared'], launch['prepare_sha256'])
    require(r['prepare_sha256'] == launch['prepare_sha256'] and r['gap_report_sha256'] == launch['gap_report_sha256'] == GAP_SHA,
            'run/prepare/gap binding')
    arrays = validate_run(folder, r, p)
    require(sha(p['compilation_path']) == p['compilation_sha256'], 'original compilation bytes')
    comp = {x['root_sha256']: x['compiled'] for x in read(p['compilation_path'])['rows']}
    g = read(GAP)
    terminal = gap.census(g['rows'])
    rows = []
    for f, buffers in zip(p['frames'], arrays, strict=True):
        c = comp[f['root_sha256']]
        require(c['goal_vectors'][0] == f['goal_vector'] and c['scoring']['metric']['sha256'] == f['active_metric_id'],
                'same compiled goal/Q')
        scores = numeric(buffers['native192'], f['predicted_vector'], f['goal_vector'], c['scoring']['metric']['projection'])
        end = terminal[f['root_slot'], f['arm']]
        rows.append(dict(root_slot=f['root_slot'], root_sha256=f['root_sha256'], arm=f['arm'], primitive_step=5,
                         frame_binding_sha256=f['frame_binding_sha256'], midpoint=scores,
                         observed_midpoint=gap.decompose(f['actual_state'], f['actual_goal_state']),
                         inherited_terminal_prediction=end['prediction'], observed_terminal=end['observed']))
    pairs = []
    for i in range(4):
        j, c = rows[2*i:2*i+2]
        differences = {k: c['midpoint'][k] - j['midpoint'][k] for k in ('teacher_Q', 'predicted_Q')}
        differences['paired_rollout_Q_gap'] = differences['predicted_Q'] - differences['teacher_Q']
        pairs.append(dict(root_slot=i, midpoint_composition_minus_joint=differences,
                          terminal_composition_minus_joint=g['pairs'][i]['composition_minus_joint']))
    result = dict(schema=NAME + '.consume', status='all8_saved_midpoint_verified_not_admitted', report_sha256=expected,
                  prepare_sha256=launch['prepare_sha256'], gap_report_sha256=GAP_SHA, root_denominator=4,
                  arm_denominator=8, rows=rows, pairs=pairs, new_neural_calls=0, new_physics_steps=0,
                  executed_run_counts=r['counts'], inherited_resources=p['inherited_resources'],
                  inherited_terminal_encoder_samples=8, new_independent_roots=0, CUDA_ns=None, FLOPs=None,
                  P1_admission=False, P2_admission=False, scientific_admission=False, equal_full_resource_cost=False,
                  limitations=['known roots only, no independent cohort or causal identification',
                               'step5 and step10 cannot identify error onset within a five-step packet',
                               'original Q rankings are not native success, no fitting or threshold changes'])
    write(output, result)
    return dict(status=result['status'], output=str(Path(output).absolute()), sha256=sha(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify', 'run', 'consume'))
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--prepare-sha')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--report-sha')
    parser.add_argument('--authorize-host-cuda', action='store_true')
    a = parser.parse_args()
    if a.command == 'prepare':
        result = prepare(a.output)
    elif a.command == 'verify':
        result = dict(status='verified_not_executed', frames=len(verify(a.prepared, a.prepare_sha)['frames']))
    elif a.command == 'run':
        require(a.authorize_host_cuda, 'explicit MAIN HOST CUDA authorization required')
        result = execute_cuda(a.prepared, a.prepare_sha, a.output)
    else:
        result = consume(a.run_dir, a.report_sha, a.output)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
