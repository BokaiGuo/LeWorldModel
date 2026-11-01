"""MAIN-only cross-implementation RGB/history3 parity diagnostic, known2/off.

No physics, raw deployment, normalizer estimation, target selection or training.
Whole/segmented Hi remains distinct from the original P2 rolling recurrence.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import agent_p2_hilewm_pairing_catalog_20261029_v1 as cat
from experiments.spectral_reachability import agent_hilewm_rgb_high_low_20261028_v1 as rgb
from experiments.spectral_reachability import agent_hilewm_normalized_transition_20261029_v1 as seam
from experiments.spectral_reachability import agent_hilewm_differentiable_low_20261029_v2 as diff

TAG = 'agent_p2_hilewm_computation_parity_20261029_v1'
ROOT = Path(__file__).resolve().parents[2]
need, encode = cat.need, cat.encode
CAP, RESERVE, FLOOR, WALL = 256*1024**2, 8*1024**2, 200*1024**2, 900
CAT_REPORT = ROOT/'results/agent_p2_hilewm_pairing_catalog_20261029_v1_cuda_MAIN01/report.json'
CAT_RH = '5605d9d7966a38ac146a84f987a09099dae33da5a2864110a69da06fd1970650'
CAT_RECEIPT = ROOT/'results/agent_p2_hilewm_pairing_catalog_20261029_v1_consume_MAIN01.json'
CAT_CH = '9a5406bd076ccfd8a931d41943a64a1f905fa07a6a42711d419db923bcd7abd7'
JOIN = ROOT/'results/agent_p2_hilewm_pairing_catalog_20261029_v1_join_MAIN01.json'
JOIN_SHA = '1523183330336bd3e66884a15fa96e7a15161abd8a343d1f50dba7a0427494b8'
PINS = {
    'experiments/spectral_reachability/agent_p2_hilewm_pairing_catalog_20261029_v1.py':
        '720ea1422a00badf8ce5298cc87b7179220047453cf2856e2777ff3d7031fa71',
    'experiments/spectral_reachability/agent_hilewm_rgb_high_low_20261028_v1.py':
        'ad7e19e8c1024c51227307b219b32bdc448e85ca27b6a3a4cc8e5708258bda2d',
    'experiments/spectral_reachability/agent_hilewm_normalized_transition_20261029_v1.py':
        '56c88a29048833546654a47257e2a34b681d103f66bfd94ccdedb0bd4763c1b2',
    'experiments/spectral_reachability/agent_hilewm_differentiable_low_20261029_v2.py':
        '7b10f644320a9c55d4805a3414901cbbc7472e3505baa03c5b2bb360820f713a',
}
ROUTES = ('p2', 'hi_whole', 'hi_segmented')
PIXEL_KINDS = ('pixel.patch',)+tuple('pixel.layer'+str(i) for i in range(12))+('pixel.finalnorm', 'encoder', 'projector')


def body(path, cap=CAP):
    p = Path(path)
    need(p.is_file() and not p.is_symlink() and p.stat().st_size <= cap, 'regular bounded input')
    b = p.read_bytes(); need(len(b) <= cap, 'input cap'); return b


def sha(path): return hashlib.sha256(body(path, 128*1024**2)).hexdigest()


def read(path, h):
    b = body(path); need(hashlib.sha256(b).hexdigest() == h, 'exact external JSON SHA')
    def pairs(xs):
        d = {}
        for k, v in xs: need(k not in d, 'duplicate JSON key'); d[k] = v
        return d
    return json.loads(b, object_pairs_hook=pairs, parse_constant=lambda _: need(False, 'nonfinite JSON'))


def protocol():
    return dict(schema=TAG+'.protocol', roots=[0, 1], condition='off', known_roots=2, fresh=False,
        images=['observed0', 'observed5', 'observed10', 'goal'], image_batch=1,
        pixel_reference='P2 original model.encode, original p0 image transform',
        pixel_owned='frozen Hi owned_pixel, original NumPy preprocess; no layout coercion',
        low_context='SAME saved real P2 latent_history3 + between-state normalized action_history2',
        low_input_not_replaced_by_reencoded_pixels=True, history_coordinates='P2 engineering convention, not certified Hi scaler',
        routes=list(ROUTES), H=3, B=1, S=1, segments=[1, 1, 1],
        actions='float32((arange30%11)-5)/32, shape1x1x3x10',
        weights='float32((arange192%13)-6)/128', scalar='float64 sum(terminal*weights)',
        acceptance='bitwise dtype/shape/C-index bytes, complete30 gradient, no tolerance; layout reported separately',
        scalar_saved_arithmetic='float64 summation roundoff bound 192*eps*sumabs, not parity tolerance',
        alignment='AE last3; predictor/projection all3 slots; context z last3/a last2; full histories also saved',
        no_early_exit_on_numeric_difference=True, raw_inverse=False, physics=0, training=False,
        new_resources=dict(encoder=16, projector=16, action_encoder=18, low_predictor=18,
            low_projection=18, P2_predict_API=6, Hi_predict_API=0, high=0,
            action_encoder_tokens=dict(p2=18, hi=48), predictor_context_tokens=54,
            projection_rows=54, explicit_scalar_VJP=6, AD_primal_included_low_forwards=18),
        backward_engine_calls=None, FLOPs=None, scientific_admission=False, P2_pairing=False,
        expected_success_NPY_files=786,
        original_class_parity=False, wall_seconds=WALL, output_cap_bytes=CAP,
        disk_floor_bytes=FLOOR, final_reserve_bytes=RESERVE)


def fixed_arrays():
    return ((np.arange(30, dtype=np.float32)%11-5)/32).reshape(1, 1, 3, 10), (np.arange(192, dtype=np.float32)%13-6)/128


def closure():
    out = cat.sources(); out.update(diff.source_closure())
    seam.verify_sources()
    for name, h in PINS.items(): need(sha(ROOT/name) == h, 'frozen computation source'); out[str(ROOT/name)] = h
    # Include constructor own metadata as well as code, BEFORE execution.
    for tag in (rgb.loader.TAG, seam.TAG, TAG):
        for folder, filename in (('experiments/spectral_reachability', tag+'.py'), ('tests', 'test_'+tag+'.py'), ('innerdocs', tag+'.md')):
            p = ROOT/folder/filename; out[str(p)] = sha(p)
    for p in (rgb.STATS, rgb.VIT, ROOT/'module.py', ROOT/'experiments/run_pusht_counterfactual.py',
              ROOT/'.venv/lib/python3.12/site-packages/stable_worldmodel/wm/lewm/lewm.py',
              ROOT/'.venv/lib/python3.12/site-packages/stable_worldmodel/wm/lewm/module.py',
              ROOT/'.venv/lib/python3.12/site-packages/stable_pretraining/backbone/utils.py'):
        out[str(p)] = sha(p)
    out[str(rgb.loader.IR_PATH)] = rgb.loader.IR_SHA
    for name, h in rgb.loader.TEXT_HASHES.items(): out[str(rgb.loader.TEXTS/name)] = h
    for rel in ('nn/functional.py', 'autograd/grad_mode.py', 'autograd/__init__.py',
                'nn/modules/module.py', '_tensor.py', 'nn/attention/__init__.py'):
        p = ROOT/'.venv/lib/python3.12/site-packages/torch'/rel; out[str(p)] = sha(p)
    for rel in ('torchvision/transforms/v2/functional/_misc.py', 'torchvision/transforms/v2/_misc.py',
                'torchvision/transforms/v2/_container.py', 'torchvision/transforms/v2/functional/_geometry.py'):
        p = ROOT/'.venv/lib/python3.12/site-packages'/rel; out[str(p)] = sha(p)
    return out


def task_input(slot, decoder):
    raw = body(slot['task_file'], 2*1024**2)
    need(hashlib.sha256(raw).hexdigest() == slot['task_sha256'], 'original task file')
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
        plain = stream.read(2*1024**2+1)
    need(len(plain) <= 2*1024**2, 'bounded original task expansion')
    task = json.loads(plain); t, c = task['task'], task['task_contract']
    need(task['task_id'] == slot['task_id'] and c['condition'] == slot['condition'] == 'off'
         and c['root_slot'] == slot['root_slot'] and c['blind_path'] == slot['blind_path']
         and c['blind_sha256'] == slot['blind_sha256'] and t['root_context'] == slot['packet']['root_context'], 'source/context/task exact join')
    b = body(slot['blind_path'], 2*1024**2)
    need(hashlib.sha256(b).hexdigest() == slot['blind_sha256'], 'actual RGB/raw packet SHA')
    packet = decoder(b)
    images = [*packet['rgb_history'], packet['goal_rgb']]
    need([hashlib.sha256(x.tobytes()).hexdigest() for x in images] == c['pixel_sha256'], 'all4 original RGB identities')
    need(np.array_equal(packet['raw_action_history'], np.asarray(t['root_raw_action_history'], np.float32))
         and np.array_equal(packet['raw_action_history'], np.asarray(slot['warm_raw10'], np.float32)), 'original actual warm10')
    z = np.asarray(t['root_context']['latent_history'], np.float32)
    a = np.asarray(t['root_context']['action_history'], np.float32)
    need(z.shape == (3, 192) and a.shape == (2, 10) and np.isfinite(z).all() and np.isfinite(a).all(), 'authentic history3 ABI')
    return dict(root=slot['root_slot'], condition='off', task_id=slot['task_id'],
        task_file=slot['task_file'], task_sha256=slot['task_sha256'], blind=slot['blind_path'],
        blind_sha256=slot['blind_sha256'], pixel_sha256=c['pixel_sha256'],
        z=z.tolist(), a=a.tolist(), normalization_claim='P2 recorded normalized tokens only; Hi original stats unknown')


def build():
    accepted, p2rows = cat.consume_catalog(CAT_REPORT, CAT_RH)
    need(accepted == read(CAT_RECEIPT, CAT_CH), 'entire actual P2 catalog receipt')
    joined = read(JOIN, JOIN_SHA); auth = joined['authorities']
    er = read(auth['hi_report'], auth['hi_report_sha256']); eh = read(auth['hi_receipt'], auth['hi_receipt_sha256'])
    need(auth['p2_report_sha256'] == CAT_RH and auth['p2_receipt'] == accepted and auth['hi_report_sha256'] == cat.HI_SHA,
         'actual pairing authority join')
    need(eh == dict(status='saved_data_verified_not_model_equivalence', report_sha256=cat.HI_SHA,
        arrays_verified=850, ir_sha256=cat.IR_SHA, trusted_pickle_loading_allowed=False, scientific_admission=False), 'actual complete exporter receipt')
    expected = cat.join_catalogs(cat.hi_rows(er), p2rows); expected['authorities'] = auth
    need(joined == expected and joined['all_five_tensor_catalog_equal'] and all(not g['layout_differences'] for g in joined['groups']),
         'entire saved five-group result, no outcome configuration')
    saved, spec, facts = rgb.loader.audit()
    model = rgb.config_contract(saved, spec, facts)
    original = cat.read(cat.PREP, cat.PREP_SHA)
    decoder = importlib.import_module('experiments.spectral_reachability.p1_constructive_moving_history_encode_cuda_20261027_v1').decode_blind
    need([(s['root_slot'], s['condition']) for s in original['slots']] == [(0, 'off'), (1, 'off')], 'fixed known2, no replacement')
    slots = [task_input(s, decoder) for s in original['slots']]
    src = closure()
    src.update(read(CAT_REPORT, CAT_RH)['runtime_sources_before'])
    for p, h in ((CAT_REPORT, CAT_RH), (CAT_RECEIPT, CAT_CH), (JOIN, JOIN_SHA),
                 (auth['hi_report'], auth['hi_report_sha256']), (auth['hi_receipt'], auth['hi_receipt_sha256']),
                 (cat.PREP, cat.PREP_SHA)):
        src[str(Path(p).resolve())] = h
    for s in slots:
        src[s['task_file']] = s['task_sha256']; src[s['blind']] = s['blind_sha256']
    for row in er['rows']:
        if row['array'] is not None:
            p = Path(auth['hi_report']).parent/row['array']['file']; src[str(p)] = row['array']['file_sha256']
    need(all(sha(p) == h for p, h in src.items()), 'complete source/data closure')
    expected_hi = {e['key']: r['array']['value_sha256'] for e, r in zip(spec['entries'], er['rows'], strict=True) if r['array'] is not None}
    need(len(expected_hi) == 425, 'all Hi tensor denominator')
    return dict(schema=TAG+'.prepare', protocol=protocol(), sources=src, slots=slots,
        hi_model=model, statistics=rgb.image_stats(), expected_hi_hashes=expected_hi,
        construction_expected=dict(module_count=spec['module_count'], entry_count=len(spec['entries']),
                                   state_key_count=len(expected_hi), ir_sha256=rgb.loader.IR_SHA),
        expected_p2_catalog=p2rows, p2_loader=cat.build()['loader_input'],
        export=dict(report=auth['hi_report'], report_sha256=auth['hi_report_sha256'], receipt=auth['hi_receipt'], receipt_sha256=auth['hi_receipt_sha256']))


def write(path, b):
    with Path(path).open('xb') as f:
        need(f.write(b) == len(b), 'exclusive full write'); f.flush(); os.fsync(f.fileno())


def prepare(output):
    m = build(); b = encode(m); need(len(b) <= RESERVE, 'prepare metadata cap')
    p = Path(output); need(p.parent.resolve() == ROOT/'results' and p.name.startswith(TAG+'_'), 'own prepare path')
    write(p, b); return dict(path=str(p.resolve()), sha256=sha(p))


def verify(path, h):
    m = read(path, h); need(m == build(), 'entire source/authority prepare reconstructed'); return m


def layout(x):
    return dict(shape=list(x.shape), stride=list(x.stride()), storage_offset=x.storage_offset(),
                contiguous=x.is_contiguous(), dtype=str(x.dtype), device=str(x.device), requires_grad=x.requires_grad)


def precision(t):
    value = dict(deterministic=t.are_deterministic_algorithms_enabled(),
        tf32_matmul=t.backends.cuda.matmul.allow_tf32, tf32_cudnn=t.backends.cudnn.allow_tf32,
        autocast=t.is_autocast_enabled('cuda'), grad=t.is_grad_enabled(), inference=t.is_inference_mode_enabled(),
        math=t.backends.cuda.math_sdp_enabled(), flash=t.backends.cuda.flash_sdp_enabled(),
        efficient=t.backends.cuda.mem_efficient_sdp_enabled(), cudnn=t.backends.cuda.cudnn_sdp_enabled())
    need(value == dict(deterministic=True, tf32_matmul=False, tf32_cudnn=False, autocast=False,
        grad=True, inference=False, math=True, flash=False, efficient=False, cudnn=False), 'fixed actual CUDA precision')
    return value


class Store:
    def __init__(self, path):
        self.path = Path(path); self.start = time.monotonic(); self.used = 0; self.arrays = {}
        need(self.path.parent.resolve() == ROOT/'results' and self.path.name.startswith(TAG+'_')
             and not self.path.exists() and shutil.disk_usage(self.path.parent).free >= CAP+FLOOR, 'exclusive bounded own output')
        self.path.mkdir()
    def guard(self, n=0, final=False):
        need(final or time.monotonic()-self.start <= WALL, 'cooperative total wall')
        need(self.used+n <= CAP-(0 if final else RESERVE) and shutil.disk_usage(self.path).free >= FLOOR+n, 'output cap/reserve/floor')
    def put(self, name, b, final=False):
        need(Path(name).name == name, 'flat output'); self.guard(len(b), final)
        write(self.path/name, b); self.used += len(b)
    def array(self, x):
        a = x if isinstance(x, np.ndarray) else x.detach().cpu().contiguous().numpy()
        need(a.dtype.str in ('|u1', '<f4', '<f8') and np.isfinite(a).all(), 'original finite numeric ABI')
        stream = io.BytesIO(); np.save(stream, a, allow_pickle=False); b = stream.getvalue()
        name = 'array_%05d.npy'%len(self.arrays); self.put(name, b)
        self.arrays[name] = dict(file_sha256=hashlib.sha256(b).hexdigest(), file_bytes=len(b), dtype=a.dtype.str,
            shape=list(a.shape), data_sha256=hashlib.sha256(a.tobytes()).hexdigest())
        return name


class Trace:
    def __init__(self, t, store, events): self.t, self.store, self.events = t, store, events; self.stack = []; self.starts = {}
    def begin(self, kind, args):
        self.store.guard()
        e = dict(index=len(self.events), parent=self.stack[-1] if self.stack else None, kind=kind,
            inputs=[self.store.array(x) for x in args], layouts=[layout(x) for x in args],
            precision=precision(self.t), returned=False)
        self.events.append(e); self.stack.append(e['index']); self.starts[e['index']] = time.monotonic(); return e
    def end(self, e, value):
        self.t.cuda.synchronize(); e.update(output=self.store.array(value), output_layout=layout(value), returned=True)
        e['callback_sync_capture_wall_seconds'] = time.monotonic()-self.starts.pop(e['index'])
        need(self.stack.pop() == e['index'], 'nested actual call order'); return value
    def call(self, kind, args, fn):
        e = self.begin(kind, args); start = time.monotonic()
        try: return self.end(e, fn())
        except BaseException as exc: e['error'] = type(exc).__name__+': '+str(exc); raise
        finally: e['callback_sync_capture_wall_seconds'] = time.monotonic()-start


def p2_pixel(t, model, x, trace):
    """Actual original encode with observational hooks, no substituted forward."""
    targets = {'pixel.patch': model.encoder.embeddings.patch_embeddings.projection,
        'pixel.finalnorm': model.encoder.layernorm, 'encoder': model.encoder, 'projector': model.projector}
    targets.update({'pixel.layer'+str(i): model.encoder.layers[i] for i in range(12)})
    hooks = []; pending = {}
    for kind, node in targets.items():
        def before(node, args, kind=kind): pending[kind] = trace.begin(kind, [args[0]])
        def after(node, args, result, kind=kind):
            value = result.last_hidden_state if kind == 'encoder' else result
            trace.end(pending.pop(kind), value)
        hooks.extend((node.register_forward_pre_hook(before), node.register_forward_hook(after)))
    try: return model.encode({'pixels': x[:, None]})['emb'][:, 0]
    finally:
        for hook in hooks: hook.remove()


def p2_low(t, model, z, a, future, trace):
    """Original P2 rolling recurrence in normalized coordinates, no raw inverse.

    Original model.predict still performs predictor -> flatten pred_proj.
    This only removes make_transition's raw normalization, NOT model math.
    """
    zs, acts, pred, contexts = z, a, [], []
    for step in range(3):
        outgoing = t.cat((acts, future[:, :, step].reshape(1, 1, 10)), dim=1)
        emb = trace.call('action_encoder', [outgoing], lambda: model.action_encoder(outgoing))
        def predict():
            pending = {}; hooks = []
            for name, node in (('low_predictor', model.predictor), ('low_pred_proj', model.pred_proj)):
                def before(node, args, name=name): pending[name] = trace.begin(name, list(args))
                def after(node, args, value, name=name): trace.end(pending.pop(name), value)
                hooks.extend((node.register_forward_pre_hook(before), node.register_forward_hook(after)))
            try: return model.predict(zs, emb)
            finally:
                for hook in hooks: hook.remove()
        out = trace.call('P2_predict_API', [zs, emb], predict)
        nxt = out[:, -1:]; zs = t.cat((zs[:, 1:], nxt), dim=1); acts = outgoing[:, 1:]
        pred.append(nxt[:, 0]); contexts.append(seam.FlatContext(zs, acts))
    return seam.Rollout(t.stack(pred, dim=1).reshape(1, 1, 3, 192), contexts[-1], tuple(contexts))


def hi_hashes(root):
    return {k: hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for k, v in root.state_dict().items()}


def dispatch(t, p2, hi, m, store, rows):
    from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
    facade = diff.wrap(hi, t, main_authorized=True)
    actions, weights = fixed_arrays(); transform = p0.image_transform()
    decoder = importlib.import_module('experiments.spectral_reachability.p1_constructive_moving_history_encode_cuda_20261027_v1').decode_blind
    for slot, row in zip(m['slots'], rows, strict=True):
        row['status'] = 'attempted'
        packet = decoder(body(slot['blind'], 2*1024**2)); images = [*packet['rgb_history'], packet['goal_rgb']]
        for image_index, image in enumerate(images):
            panel = dict(image=image_index, rgb=store.array(image), paths={}); row['pixels'].append(panel)
            for backend in ('p2', 'hi'):
                item = dict(events=[], status='attempted'); panel['paths'][backend] = item
                trace = Trace(t, store, item['events'])
                if backend == 'p2': x = p0._image_tensor(image, transform=transform, device=t.device('cuda:0'))[:, 0]
                else: x = t.from_numpy(rgb.preprocess(image, m['statistics'])).to(device='cuda:0')
                item['preprocessed'] = store.array(x); item['preprocessed_layout'] = layout(x)
                # Pixel path has no input gradient. Hi projector remains original inference-only;
                # its output is NOT fed into the differentiable low state.
                out = p2_pixel(t, p2, x, trace) if backend == 'p2' else rgb.owned_pixel(t, hi, x, m['hi_model']['config'], trace)
                item.update(output=store.array(out), status='returned')
        for route in ROUTES:
            item = row['routes'][route]; item['status'] = 'attempted'; trace = Trace(t, store, item['events'])
            z = t.tensor(slot['z'], dtype=t.float32, device='cuda:0').reshape(1, 3, 192)
            a = t.tensor(slot['a'], dtype=t.float32, device='cuda:0').reshape(1, 2, 10)
            u = t.tensor(actions, dtype=t.float32, device='cuda:0', requires_grad=True)
            item.update(z=store.array(z), a=store.array(a), future=store.array(u))
            if route == 'p2': result = p2_low(t, p2, z, a, u, trace)
            else:
                kernels = seam.cuda_kernels(t, facade, main_authorized=True, trace=trace)
                zh, ah = z[:, None], a[:, None]
                result = seam.rollout(zh, ah, u, kernels=kernels) if route == 'hi_whole' else seam.segmented_rollout(zh, ah, u, segments=(1, 1, 1), kernels=kernels)
            item['prediction'] = store.array(result.prediction)
            item['checkpoints'] = [dict(z=store.array(c.z), a=store.array(c.a)) for c in result.contexts]
            w = t.tensor(weights, dtype=t.float32, device='cuda:0')
            scalar = (result.prediction[:, :, -1].to(t.float64)*w.to(t.float64)).sum()
            item['scalar'] = store.array(scalar)
            grad = trace.call('scalar_VJP', [scalar, u], lambda: t.autograd.grad(scalar, u, create_graph=False, retain_graph=False, allow_unused=False)[0])
            item.update(gradient=store.array(grad), status='returned')
        facade.assert_frozen(); row['status'] = 'returned'


def load_array(folder, name, desc):
    need(re_name(name), 'flat original array name')
    b = body(Path(folder)/name, 4*1024**2)
    need(len(b) == desc['file_bytes'] and hashlib.sha256(b).hexdigest() == desc['file_sha256'], 'NPY file SHA/bytes')
    f = io.BytesIO(b); need(np.lib.format.read_magic(f) == (1, 0), 'NPY v1')
    shape, order, dtype = np.lib.format.read_array_header_1_0(f)
    need(dtype.str in ('|u1', '<f4', '<f8') and list(shape) == desc['shape'] and dtype.str == desc['dtype']
         and order is False and math.prod(shape)*dtype.itemsize == len(b)-f.tell(), 'bounded exact NPY header')
    f.seek(0); a = np.load(f, allow_pickle=False)
    need(f.tell() == len(b) and np.isfinite(a).all() and hashlib.sha256(a.tobytes()).hexdigest() == desc['data_sha256'], 'original finite array data')
    return a


def re_name(name):
    return isinstance(name, str) and len(name) == len('array_00000.npy') and name.startswith('array_') and name[6:11].isdigit() and name.endswith('.npy')


def compare(a, b):
    same = a.shape == b.shape and a.dtype == b.dtype
    return dict(same_ABI=same, exact=bool(same and a.tobytes() == b.tobytes()),
        maxabs=float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))) if same else None)


def exact(a, b, msg):
    need(compare(a, b)['exact'], msg)


def validate_events(events, get):
    need(type(events) is list and events, 'actual nonempty trace')
    for i, e in enumerate(events):
        need(e['index'] == i and (e['parent'] is None or type(e['parent']) is int and 0 <= e['parent'] < i)
             and e['returned'] is True and len(e['inputs']) == len(e['layouts']), 'complete event index/parent/return')
        need(e['precision'] == dict(deterministic=True, tf32_matmul=False, tf32_cudnn=False, autocast=False,
             grad=True, inference=False, math=True, flash=False, efficient=False, cudnn=False), 'saved fixed precision')
        for name, lay in zip(e['inputs']+[e['output']], e['layouts']+[e['output_layout']], strict=True):
            a = get(name)
            need(lay['shape'] == list(a.shape) and lay['dtype'] == {'|u1': 'torch.uint8', '<f4': 'torch.float32', '<f8': 'torch.float64'}[a.dtype.str]
                 and lay['device'] == 'cuda:0' and type(lay['requires_grad']) is bool
                 and len(lay['stride']) == a.ndim and all(type(x) is int and x >= 0 for x in lay['stride'])
                 and type(lay['storage_offset']) is int and lay['storage_offset'] >= 0, 'original saved tensor layout ABI')
            stride = 1; contiguous = True
            for size, step in zip(a.shape[::-1], lay['stride'][::-1], strict=True):
                if size > 1: contiguous &= stride == step; stride *= size
            need(lay['contiguous'] is bool(contiguous), 'saved layout contiguity')


def validate_low(slot, route, item, get):
    validate_events(item['events'], get)
    future, weights = fixed_arrays()
    z = np.asarray(slot['z'], np.float32).reshape(1, 3, 192)
    a = np.asarray(slot['a'], np.float32).reshape(1, 2, 10)
    pred = []
    groups = {k: [e for e in item['events'] if e['kind'] == k] for k in ('action_encoder', 'low_predictor', 'low_pred_proj', 'P2_predict_API', 'scalar_VJP')}
    sequence = ['action_encoder', 'P2_predict_API', 'low_predictor', 'low_pred_proj'] if route == 'p2' else ['action_encoder', 'low_predictor', 'low_pred_proj']
    need([e['kind'] for e in item['events']] == sequence*3+['scalar_VJP'], 'literal actual low call order')
    need(len(groups['P2_predict_API']) == (3 if route == 'p2' else 0), 'actual P2 predict API scope')
    for step in range(3):
        a = np.concatenate((a, future[:, :, step].reshape(1, 1, 10)), axis=1)
        ae, predict, project = (groups[k][step] for k in ('action_encoder', 'low_predictor', 'low_pred_proj'))
        need(len(ae['inputs']) == len(project['inputs']) == 1 and len(predict['inputs']) == 2, 'module input arities')
        exact(get(ae['inputs'][0]), a, 'literal normalized outgoing append order')
        emb = get(ae['output']); need(emb.shape == (1, a.shape[1], 192), 'AE all original time tokens')
        exact(get(predict['inputs'][0]), z[:, -3:], 'real full history predictor last3')
        exact(get(predict['inputs'][1]), emb[:, -3:], 'AE delivery to predictor all3 slots')
        h = get(predict['output']); need(h.shape == (1, 3, 192), 'predict all3 slots')
        exact(get(project['inputs'][0]), h.reshape(3, 192), 'literal flatten projection input')
        projected = get(project['output']); need(projected.shape == (3, 192), 'project all3 slots')
        if route == 'p2':
            call = groups['P2_predict_API'][step]
            exact(get(call['inputs'][0]), z, 'original predict API context')
            exact(get(call['inputs'][1]), emb, 'original predict API action delivery')
            exact(get(call['output']), projected.reshape(1, 3, 192), 'original predict reshape output')
        nxt = projected.reshape(1, 3, 192)[:, -1:]
        z = np.concatenate((z[:, 1:] if route == 'p2' else z, nxt), axis=1)
        if route == 'p2': a = a[:, 1:]
        exact(get(item['checkpoints'][step]['z']), z, 'complete exact recurrence latent history')
        exact(get(item['checkpoints'][step]['a']), a, 'complete exact recurrence action history')
        pred.append(nxt[:, 0])
    prediction = np.stack(pred, axis=1).reshape(1, 1, 3, 192)
    exact(get(item['prediction']), prediction, 'all3 prediction checkpoint delivery')
    scalar = get(item['scalar']); gradient = get(item['gradient'])
    need(scalar.shape == () and scalar.dtype == np.float64 and gradient.shape == future.shape
         and gradient.dtype == np.float32, 'scalar64 and complete30 gradient32')
    terms = prediction[:, :, -1].astype(np.float64)*weights.astype(np.float64)
    reference = math.fsum(float(x) for x in terms.flat)
    bound = 192*np.finfo(np.float64).eps*math.fsum(abs(float(x)) for x in terms.flat)
    need(abs(float(scalar)-reference) <= bound, 'saved scalar independent float64 roundoff bound')
    vjp = groups['scalar_VJP'][0]
    exact(get(vjp['inputs'][0]), scalar, 'actual VJP scalar')
    exact(get(vjp['inputs'][1]), future, 'actual VJP action leaf')
    need(vjp['layouts'][0]['requires_grad'] is True and vjp['layouts'][1]['requires_grad'] is True, 'ordinary gradient-connected scalar and action leaf')
    exact(get(vjp['output']), gradient, 'actual all30 VJP output')


def analyze(m, r, get):
    """Saved arithmetic only; canonical slots, not forward recomputation."""
    out = []; resources = dict(encoder=0, projector=0, action_encoder=0, low_predictor=0,
        low_pred_proj=0, P2_predict_API=0, scalar_VJP=0)
    need(len(r['rows']) == 2, 'known2 denominator')
    for slot, row in zip(m['slots'], r['rows'], strict=True):
        need(row['root'] == slot['root'] and row['status'] == 'returned' and len(row['pixels']) == 4, 'all root/pixel paths returned')
        comparisons = []; paid_history_checks = []
        for i, panel in enumerate(row['pixels']):
            need(panel['image'] == i and hashlib.sha256(get(panel['rgb']).tobytes()).hexdigest() == slot['pixel_sha256'][i], 'original RGB byte identity')
            paths = panel['paths']; need(set(paths) == {'p2', 'hi'}, 'both pixel paths')
            comparisons.append(dict(name='pixel%d.preprocessed'%i, **compare(get(paths['p2']['preprocessed']), get(paths['hi']['preprocessed']))))
            for backend in paths:
                need(paths[backend]['status'] == 'returned' and all(e['returned'] for e in paths[backend]['events']), 'complete pixel calls')
                validate_events(paths[backend]['events'], get)
                kinds = [e['kind'] for e in paths[backend]['events']]
                need(len(kinds) == len(PIXEL_KINDS) and set(kinds) == set(PIXEL_KINDS), 'full12-layer encoder panel')
                need(kinds == ['encoder']+list(PIXEL_KINDS[:-2])+['projector'], 'original pixel module call order')
                for key in ('encoder', 'projector'): resources[key] += kinds.count(key)
                by = {e['kind']: e for e in paths[backend]['events']}
                need(paths[backend]['preprocessed_layout'] == by['encoder']['layouts'][0], 'encoder preprocessor layout identity')
                exact(get(by['encoder']['inputs'][0]), get(paths[backend]['preprocessed']), 'actual encoder preprocessed input')
                exact(get(by['projector']['inputs'][0]), get(by['encoder']['output'])[:, 0], 'original CLS to projector')
                exact(get(by['pixel.finalnorm']['output']), get(by['encoder']['output']), 'finalnorm alltoken output')
                exact(get(paths[backend]['output']), get(by['projector']['output']), 'original projected pixel output')
                for step in range(1, 12):
                    exact(get(by['pixel.layer'+str(step)]['inputs'][0]), get(by['pixel.layer'+str(step-1)]['output']), 'all12 original encoder layers linked')
            for kind in PIXEL_KINDS:
                a, b = [next(e for e in paths[k]['events'] if e['kind'] == kind) for k in ('p2', 'hi')]
                comparisons.append(dict(name='pixel%d.%s'%(i, kind), **compare(get(a['output']), get(b['output'])),
                    same_layout=a['output_layout'] == b['output_layout']))
            for backend in ('p2', 'hi'):
                encoded = get(paths[backend]['output'])
                need(encoded.shape == (1, 192) and encoded.dtype == np.float32, 'native projected pixel shape')
                if i < 3:
                    recorded = np.asarray(slot['z'], np.float32)[i:i+1]
                    paid_history_checks.append(dict(name=f'pixel{i}.{backend}.vs_paid_P2_history', **compare(encoded, recorded)))
        routes = row['routes']; need(set(routes) == set(ROUTES), 'all3 gradient routes')
        for route, item in routes.items():
            need(item['status'] == 'returned' and len(item['checkpoints']) == 3 and all(e['returned'] for e in item['events']), 'all3 low slots and VJP returned')
            need(get(item['z']).tobytes() == np.asarray(slot['z'], np.float32).reshape(1, 3, 192).tobytes()
                 and get(item['a']).tobytes() == np.asarray(slot['a'], np.float32).reshape(1, 2, 10).tobytes()
                 and get(item['future']).tobytes() == fixed_arrays()[0].tobytes(), 'same fixed actual context/actions')
            for key in ('action_encoder', 'low_predictor', 'low_pred_proj', 'scalar_VJP'):
                need(sum(e['kind'] == key for e in item['events']) == (1 if key == 'scalar_VJP' else 3), 'all paid low module calls')
                resources[key] += sum(e['kind'] == key for e in item['events'])
            resources['P2_predict_API'] += sum(e['kind'] == 'P2_predict_API' for e in item['events'])
            validate_low(slot, route, item, get)
        ref = routes['p2']
        for route in ('hi_whole', 'hi_segmented'):
            own = routes[route]
            for field in ('prediction', 'scalar', 'gradient'):
                comparisons.append(dict(name=route+'.'+field, **compare(get(ref[field]), get(own[field]))))
            for step in range(3):
                for field, size in (('z', 3), ('a', 2)):
                    comparisons.append(dict(name=f'{route}.cp{step}.{field}',
                        **compare(get(ref['checkpoints'][step][field])[:, -size:], get(own['checkpoints'][step][field])[:, -size:])))
            for kind in ('action_encoder', 'low_predictor', 'low_pred_proj'):
                aa = [e for e in ref['events'] if e['kind'] == kind]; bb = [e for e in own['events'] if e['kind'] == kind]
                for step, (a, b) in enumerate(zip(aa, bb, strict=True)):
                    av, bv = get(a['output']), get(b['output'])
                    if kind == 'action_encoder': av, bv = av[:, -3:], bv[:, -3:]
                    comparisons.append(dict(name=f'{route}.{kind}{step}', **compare(av, bv), same_layout=a['output_layout'] == b['output_layout']))
        out.append(dict(root=row['root'], comparisons=comparisons, paid_history_checks=paid_history_checks,
                        all_values_exact=all(c['exact'] for c in comparisons)))
    need(resources == dict(encoder=16, projector=16, action_encoder=18, low_predictor=18,
         low_pred_proj=18, P2_predict_API=6, scalar_VJP=6), 'complete actual module census')
    return dict(roots=out, all_values_exact=all(x['all_values_exact'] for x in out), resources=resources,
        normalizer_pairing=False, raw_deployment=False, independent_full_network_verification=False, scientific_admission=False)


def census(rows):
    events = [e for row in rows for panel in row['pixels'] for item in panel['paths'].values() for e in item['events']]
    events += [e for row in rows for item in row['routes'].values() for e in item['events']]
    kinds = sorted({e['kind'] for e in events})
    return {k: dict(attempts=sum(e['kind'] == k for e in events),
        returns=sum(e['kind'] == k and e['returned'] for e in events),
        input_rows=sum(e['layouts'][0]['shape'][0] for e in events if e['kind'] == k and e['layouts'][0]['shape']),
        input_tokens=sum(e['layouts'][0]['shape'][0]*e['layouts'][0]['shape'][1] for e in events
            if e['kind'] == k and len(e['layouts'][0]['shape']) == 3)) for k in kinds}


def run_cuda(manifest, mh, output, *, main_authorized=False):
    need(main_authorized is True, 'MAIN authorization before any production I/O/import')
    need(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'fixed CUBLAS workspace')
    store = Store(output); p2 = hi = None; m = None
    r = dict(schema=TAG+'.report', manifest=str(Path(manifest).resolve()), manifest_sha256=mh, arrays=store.arrays,
        rows=[dict(root=i, status='not_started', pixels=[], routes={k: dict(status='not_started', events=[]) for k in ROUTES}) for i in (0, 1)],
        completed=False, errors=[], model_before=None, model_after=None, construction=None, loader=None,
        load_events=[], runtime_before=None, runtime_after=None, sources_after=None, diagnostics=None)
    try:
        m = verify(manifest, mh)
        import torch as t
        from torch.nn.attention import sdpa_kernel, SDPBackend
        need(t.cuda.is_available() and Path(t.__file__).resolve() == ROOT/'.venv/lib/python3.12/site-packages/torch/__init__.py', 'HOST originalvenv CUDA only')
        t.empty(0, device='cuda:0'); t.use_deterministic_algorithms(True)
        t.backends.cuda.matmul.allow_tf32 = False; t.backends.cudnn.allow_tf32 = False
        owner = importlib.import_module(cat.OWNER); sink = cat.LoaderStore()
        r['load_events'] = sink.events
        p2, aggregate = owner.loaded_model(m['p2_loader'], t, t.device('cuda:0'), sink, mh)
        r['loader'] = sink.receipts['loader.json']; owner.validate_loader(r['loader'], m['p2_loader'], mh)
        e = m['export']; hi, r['construction'] = rgb.loader.construct_cuda(e['report'], e['report_sha256'], e['receipt'], e['receipt_sha256'], main_authorized=True)
        need(all(m['sources'].get(p) == h for p, h in r['construction']['own_sha256'].items()), 'all constructor code/test/doc bound BEFORE run')
        r['model_before'] = dict(p2=aggregate, p2_catalog=cat.snapshot(p2, t), hi=hi_hashes(hi))
        need(aggregate == cat.MODEL_HASH and r['model_before']['p2_catalog'] == m['expected_p2_catalog']
             and r['model_before']['hi'] == m['expected_hi_hashes'], 'all frozen paired original tensors')
        r['runtime_before'] = cat.runtime_sources(p2, t); r['runtime_before'].update(diff.source_closure(t))
        need(all(m['sources'].get(p) == h for p, h in r['runtime_before'].items()), 'all used runtime sources prebound')
        with t.enable_grad(), t.autocast('cuda', enabled=False), sdpa_kernel(SDPBackend.MATH):
            dispatch(t, p2, hi, m, store, r['rows'])
        r['model_after'] = dict(p2=owner.enc.tensor_hash(p2), p2_catalog=cat.snapshot(p2, t), hi=hi_hashes(hi))
        need(r['model_before'] == r['model_after'], 'all weights/buffers/layout unchanged')
        owner.enc.frozen(p2)
        r['runtime_after'] = cat.runtime_sources(p2, t); r['runtime_after'].update(diff.source_closure(t))
        need(r['runtime_before'] == r['runtime_after'], 'runtime source unchanged')
        r['sources_after'] = verify(manifest, mh)['sources']
        r['diagnostics'] = analyze(m, r, lambda name: load_array(store.path, name, r['arrays'][name]))
        store.guard(); r['completed'] = True
    except BaseException as exc: r['errors'].append(dict(type=type(exc).__name__, message=str(exc)[:3000]))
    finally:
        if p2 is not None and hi is not None and r['model_after'] is None:
            try: r['model_after'] = dict(p2=importlib.import_module(cat.OWNER).enc.tensor_hash(p2), p2_catalog=cat.snapshot(p2, t), hi=hi_hashes(hi))
            except BaseException as exc: r['errors'].append(dict(stage='final_weights', message=str(exc)[:1000]))
        if m is not None and r['sources_after'] is None:
            try: r['sources_after'] = {p: sha(p) for p in m['sources']}
            except BaseException as exc: r['errors'].append(dict(stage='final_sources', message=str(exc)[:1000]))
        r['wall_seconds'] = time.monotonic()-store.start
        r['actual_call_census'] = census(r['rows'])
        store.put('report.json', encode(r), final=True)
    return dict(report=str(store.path/'report.json'), sha256=sha(store.path/'report.json'))


def consume(report, rh, output):
    r = read(report, rh); m = verify(r['manifest'], r['manifest_sha256'])
    need(r['schema'] == TAG+'.report' and r['completed'] is True and r['errors'] == [], 'complete actual capture, not necessarily numerical parity')
    need(r['model_before'] == r['model_after'] and r['model_before']['p2'] == cat.MODEL_HASH
         and r['model_before']['p2_catalog'] == m['expected_p2_catalog'] and r['model_before']['hi'] == m['expected_hi_hashes'], 'all actual before/after frozen tensor evidence')
    need(r['sources_after'] == m['sources'] and r['runtime_before'] == r['runtime_after']
         and all(sha(p) == h for p, h in r['runtime_before'].items()), 'actual source/runtime join')
    c = r['construction']; need(c['tensor_value_hashes'] == m['expected_hi_hashes'] and all(m['sources'].get(p) == h for p, h in c['own_sha256'].items()), 'entire constructor source closure')
    need(len(m['expected_hi_hashes']) == 425 and all(c[k] == v for k, v in m['construction_expected'].items()), 'all Hi graph/module/key denominator')
    expected_own = {str(ROOT/folder/name) for folder, name in (
        ('experiments/spectral_reachability', rgb.loader.TAG+'.py'), ('tests', 'test_'+rgb.loader.TAG+'.py'), ('innerdocs', rgb.loader.TAG+'.md'))}
    need(set(c['own_sha256']) == expected_own, 'constructor code/test/doc complete own set')
    need(c['schema'] == rgb.loader.TAG+'.construction' and c['export_report_sha256'] == m['export']['report_sha256']
         and c['export_receipt_sha256'] == m['export']['receipt_sha256'] and c['tensor_copy_attempts'] == c['tensor_copy_returns'] == 425
         and c['all_parameters_frozen'] is True and c['forward_calls'] == c['physics'] == 0, 'actual constructor export/copy denominator')
    need(r['actual_call_census'] == census(r['rows']), 'all actual attempted/returned resources incl partial')
    owner = importlib.import_module(cat.OWNER); owner.validate_loader(r['loader'], m['p2_loader'], r['manifest_sha256'])
    need(r['load_events'] == [event for f in r['loader']['files'] for event in (
        dict(kind='load_intent', path=f['path'], expected_sha256=f['sha256']),
        dict(kind='consumed_buffer_frozen', receipt=f))], 'actual loaded buffer receipts and events')
    cache = {}
    need(len(r['arrays']) == 786, 'fixed full NPY denominator')
    need(sum(x['file_bytes'] for x in r['arrays'].values())+Path(report).stat().st_size <= CAP, 'saved payload cap')
    for name, desc in r['arrays'].items(): cache[name] = load_array(Path(report).parent, name, desc)
    need({p.name for p in Path(report).parent.iterdir()} == {'report.json'} | set(cache), 'full file census')
    referenced = set()
    def get(name): referenced.add(name); return cache[name]
    d = analyze(m, r, get); need(d == r['diagnostics'] and referenced == set(cache), 'all comparison arithmetic and exact array reference census')
    receipt = dict(schema=TAG+'.receipt', report_sha256=rh, status='all_known2_arrays_verified_not_scientific_admission',
        all_values_exact=d['all_values_exact'], diagnostics=d, independent_gradient_recompute=False, raw_deployment=False, scientific_admission=False)
    write(output, encode(receipt)); return dict(path=str(output), sha256=sha(output))


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('command', choices=('prepare', 'verify', 'run', 'consume', 'protocol'))
    p.add_argument('--manifest', type=Path); p.add_argument('--manifest-sha256'); p.add_argument('--output', type=Path)
    p.add_argument('--report', type=Path); p.add_argument('--report-sha256'); p.add_argument('--main-authorized', action='store_true')
    a = p.parse_args()
    if a.command == 'prepare': value = prepare(a.output)
    elif a.command == 'verify': value = dict(verified=bool(verify(a.manifest, a.manifest_sha256)))
    elif a.command == 'run': value = run_cuda(a.manifest, a.manifest_sha256, a.output, main_authorized=a.main_authorized)
    elif a.command == 'consume': value = consume(a.report, a.report_sha256, a.output)
    else: value = protocol()
    print(json.dumps(value, sort_keys=True))


if __name__ == '__main__': main()
